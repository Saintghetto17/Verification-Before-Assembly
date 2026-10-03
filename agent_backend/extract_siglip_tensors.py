"""Extract SigLIP image/text embeddings for paired ml_orig ↔ ml_l2_fail figures.

Saves under outputs/tensors/:
  ml_l2_image_embeds.pt  — Dict[str, list[Tensor]]: name -> [fail_img, orig_img]
  ml_l2_text_embeds.pt   — Dict[str, list[Tensor]]: name -> [fail_txt, orig_txt]
  ml_l2_meta.pt          — Dict with paths, texts, labels, same_bytes, etc.

name key = \"{document}/{figure}\" e.g. \"document_8/figure_3\".

Uses fine-tuned SigLIP backbone from checkpoints/sem.pt (same as Judge A).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_backend.text_utils import row_texts  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("extract_siglip_tensors")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_sidecar(img: Path) -> Dict[str, Any]:
    side = img.with_suffix(".json")
    if not side.is_file():
        return {"caption": "", "text_block": "", "gold": {}}
    try:
        return json.loads(side.read_text())
    except Exception as e:
        logger.warning("bad sidecar %s: %s", side, e)
        return {"caption": "", "text_block": "", "gold": {}}


def _texts_for_image(img: Path) -> Tuple[str, str]:
    """Return (premise_for_sem, short_caption) like the agent."""
    side = _read_sidecar(img)
    row = {
        "caption": side.get("caption") or "",
        "text_block": side.get("text_block") or "",
        "premise": side.get("text_block") or side.get("caption") or "",
    }
    return row_texts(row)


def collect_pairs(
    figures_dir: Path,
    *,
    orig_split: str = "ml_orig",
    fail_split: str = "ml_l2_fail",
) -> List[Dict[str, Any]]:
    orig_dir = figures_dir / orig_split
    fail_dir = figures_dir / fail_split
    orig_map = {
        (p.parent.name, p.stem): p
        for p in orig_dir.glob("document_*/figure_*.png")
    }
    fail_map = {
        (p.parent.name, p.stem): p
        for p in fail_dir.glob("document_*/figure_*.png")
    }
    common = sorted(set(orig_map) & set(fail_map))
    rows: List[Dict[str, Any]] = []
    for doc, fig in common:
        op, fp = orig_map[(doc, fig)], fail_map[(doc, fig)]
        o_premise, o_short = _texts_for_image(op)
        f_premise, f_short = _texts_for_image(fp)
        same_size = op.stat().st_size == fp.stat().st_size
        same_bytes = bool(same_size and _md5(op) == _md5(fp))
        name = f"{doc}/{fig}"
        rows.append(
            {
                "name": name,
                "document": doc,
                "figure": fig,
                "orig_path": str(op),
                "fail_path": str(fp),
                "orig_premise": o_premise,
                "fail_premise": f_premise,
                "orig_caption": o_short,
                "fail_caption": f_short,
                "same_bytes": same_bytes,
                "orig_size": op.stat().st_size,
                "fail_size": fp.stat().st_size,
            }
        )
    return rows


def load_siglip(
    pretrained: str,
    sem_ckpt: Optional[Path],
    device: torch.device,
) -> Tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(pretrained)
    model = AutoModel.from_pretrained(pretrained)
    if sem_ckpt is not None and sem_ckpt.is_file():
        blob = torch.load(sem_ckpt, map_location="cpu", weights_only=False)
        if blob.get("kind") == "sem_match_v1":
            model.load_state_dict(blob["backbone_state_dict"], strict=True)
            logger.info("Loaded fine-tuned backbone from %s", sem_ckpt)
        else:
            logger.warning("Unknown sem ckpt kind — using base SigLIP weights")
    else:
        logger.warning("No sem_ckpt — using base SigLIP weights")
    model.to(device)
    model.eval()
    return model, processor


@torch.inference_mode()
def embed_batch(
    model,
    processor,
    image_paths: Sequence[str],
    texts: Sequence[str],
    *,
    device: torch.device,
    max_text_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    texts_clean = [(t or "").strip() or "scientific figure" for t in texts]
    enc = processor(
        text=texts_clean,
        images=images,
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model(**enc)
    if getattr(outputs, "image_embeds", None) is not None:
        img = outputs.image_embeds
        txt = outputs.text_embeds
    else:
        img = model.get_image_features(pixel_values=enc["pixel_values"])
        txt = model.get_text_features(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
        )
        if not torch.is_tensor(img):
            img = img.pooler_output if hasattr(img, "pooler_output") else img
        if not torch.is_tensor(txt):
            txt = txt.pooler_output if hasattr(txt, "pooler_output") else txt
    return img.detach().cpu().float(), txt.detach().cpu().float()


def run(
    *,
    figures_dir: Path,
    out_dir: Path,
    pretrained: str,
    sem_ckpt: Path,
    batch_size: int,
    max_text_length: int,
    device: str,
    text_field: str = "premise",
    limit: Optional[int] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = collect_pairs(figures_dir)
    if limit is not None:
        pairs = pairs[:limit]
    logger.info("Paired figures: %d (same_bytes=%d)", len(pairs), sum(1 for p in pairs if p["same_bytes"]))

    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("device=%s batch_size=%d", dev, batch_size)
    model, processor = load_siglip(pretrained, sem_ckpt, dev)

    # Flat list: for each pair encode fail then orig (keeps order for rebuild)
    flat_paths: List[str] = []
    flat_texts: List[str] = []
    for p in pairs:
        t_fail = p["fail_premise"] if text_field == "premise" else p["fail_caption"]
        t_orig = p["orig_premise"] if text_field == "premise" else p["orig_caption"]
        flat_paths.extend([p["fail_path"], p["orig_path"]])
        flat_texts.extend([t_fail, t_orig])

    all_img: List[torch.Tensor] = []
    all_txt: List[torch.Tensor] = []
    n = len(flat_paths)
    for start in range(0, n, batch_size):
        chunk_p = flat_paths[start : start + batch_size]
        chunk_t = flat_texts[start : start + batch_size]
        img, txt = embed_batch(
            model,
            processor,
            chunk_p,
            chunk_t,
            device=dev,
            max_text_length=max_text_length,
        )
        # store per-row cpu tensors (1, D) then squeeze later
        for i in range(img.shape[0]):
            all_img.append(img[i].contiguous())
            all_txt.append(txt[i].contiguous())
        done = min(start + batch_size, n)
        if done % (batch_size * 4) == 0 or done == n:
            logger.info("embedded %d/%d (%.1f%%)", done, n, 100.0 * done / n)

    image_dict: Dict[str, List[torch.Tensor]] = {}
    text_dict: Dict[str, List[torch.Tensor]] = {}
    meta: Dict[str, Any] = {
        "pretrained": pretrained,
        "sem_ckpt": str(sem_ckpt),
        "text_field": text_field,
        "max_text_length": max_text_length,
        "embed_dim": int(all_img[0].numel()),
        "n_pairs": len(pairs),
        "device": str(dev),
        "pairs": {},
    }

    for i, p in enumerate(pairs):
        # flat order: fail, orig
        fi, oi = all_img[2 * i], all_img[2 * i + 1]
        ft, ot = all_txt[2 * i], all_txt[2 * i + 1]
        name = p["name"]
        image_dict[name] = [fi, oi]  # [fail, orig]
        text_dict[name] = [ft, ot]
        meta["pairs"][name] = {
            "fail_path": p["fail_path"],
            "orig_path": p["orig_path"],
            "fail_text": flat_texts[2 * i],
            "orig_text": flat_texts[2 * i + 1],
            "same_bytes": p["same_bytes"],
            "orig_size": p["orig_size"],
            "fail_size": p["fail_size"],
            "image_l2_dist": float(torch.norm(fi - oi).item()),
            "text_l2_dist": float(torch.norm(ft - ot).item()),
            "image_cosine": float(
                torch.nn.functional.cosine_similarity(fi.unsqueeze(0), oi.unsqueeze(0)).item()
            ),
            "text_cosine": float(
                torch.nn.functional.cosine_similarity(ft.unsqueeze(0), ot.unsqueeze(0)).item()
            ),
        }

    img_path = out_dir / "ml_l2_image_embeds.pt"
    txt_path = out_dir / "ml_l2_text_embeds.pt"
    meta_path = out_dir / "ml_l2_meta.pt"
    meta_json = out_dir / "ml_l2_meta.json"

    torch.save(image_dict, img_path)
    torch.save(text_dict, txt_path)
    torch.save(meta, meta_path)
    # JSON without tensors for quick browsing
    with open(meta_json, "w") as f:
        json.dump(
            {
                k: v
                for k, v in meta.items()
                if k != "pairs"
            }
            | {
                "pairs": meta["pairs"],
                "n_same_bytes": sum(1 for p in pairs if p["same_bytes"]),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Wrote %s (%d keys)", img_path, len(image_dict))
    logger.info("Wrote %s (%d keys)", txt_path, len(text_dict))
    logger.info("Wrote %s and %s", meta_path, meta_json)

    # sanity: identical images should have ~0 image distance
    same = [meta["pairs"][n]["image_l2_dist"] for n in image_dict if meta["pairs"][n]["same_bytes"]]
    diff = [meta["pairs"][n]["image_l2_dist"] for n in image_dict if not meta["pairs"][n]["same_bytes"]]
    if same:
        logger.info(
            "same_bytes image L2: mean=%.6g max=%.6g n=%d",
            sum(same) / len(same),
            max(same),
            len(same),
        )
    if diff:
        logger.info(
            "diff_bytes image L2: mean=%.6g max=%.6g n=%d",
            sum(diff) / len(diff),
            max(diff),
            len(diff),
        )
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-dir", type=Path, default=ROOT / "data/figures")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/tensors")
    ap.add_argument(
        "--pretrained",
        default="/home/jovyan/.cache/huggingface/hub/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    )
    ap.add_argument("--sem-ckpt", type=Path, default=ROOT / "checkpoints/sem.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-text-length", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--text-field",
        choices=("premise", "caption"),
        default="premise",
        help="Text encoded with image (premise=Judge A text, caption=short)",
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(
        figures_dir=args.figures_dir,
        out_dir=args.out_dir,
        pretrained=args.pretrained,
        sem_ckpt=args.sem_ckpt,
        batch_size=args.batch_size,
        max_text_length=args.max_text_length,
        device=args.device,
        text_field=args.text_field,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
