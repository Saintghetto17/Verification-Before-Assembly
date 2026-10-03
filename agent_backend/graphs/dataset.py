"""Build claim→graph JSONL from data/figures (excluding golden)."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..dataset import discover_train_splits
from .extract import extract_graph

logger = logging.getLogger(__name__)

IMAGE_JSON = "figure_*.json"


def _rel(figures_dir: Path, path: Path) -> str:
    """Path relative to repo root (parent of data/figures) when possible."""
    root = figures_dir.resolve().parent.parent
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def family_of_split(split: str) -> str:
    return split.replace("_l2_fail", "").replace("_orig", "")


def figure_key(split: str, document: str, stem: str) -> str:
    return f"{family_of_split(split)}/{document}/{stem}"


def collect_unique_figures(
    figures_dir: Path,
    *,
    splits: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """One row per figure identity (orig preferred over paired fail). Golden skipped."""
    figures_dir = Path(figures_dir)
    split_names = list(splits) if splits else discover_train_splits(figures_dir)
    by_key: Dict[str, Dict[str, Any]] = {}
    for split in split_names:
        split_dir = figures_dir / split
        if not split_dir.is_dir():
            continue
        prefer = 0 if "_orig" in split else 1
        for meta_path in sorted(split_dir.rglob(IMAGE_JSON)):
            try:
                obj = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            claims = obj.get("visual_claims") if isinstance(obj.get("visual_claims"), list) else []
            claims = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
            if not claims:
                continue
            document = meta_path.parent.name
            stem = meta_path.stem
            key = figure_key(split, document, stem)
            rec = {
                "figure_key": key,
                "sample_id": obj.get("sample_id") or f"{split}/{document}/{stem}",
                "split_folder": split,
                "prefer": prefer,
                "sidecar_path": _rel(figures_dir, meta_path),
                "image_path": _rel(figures_dir, meta_path.with_suffix(".png")),
                "claims": claims,
                "role": (obj.get("visual_annotation") or {}).get("role"),
            }
            prev = by_key.get(key)
            if prev is None or prefer < prev["prefer"]:
                by_key[key] = rec
    rows = [by_key[k] for k in sorted(by_key)]
    logger.info(
        "graph figures: unique=%d claims=%d (excl. golden) splits=%s",
        len(rows),
        sum(len(r["claims"]) for r in rows),
        split_names,
    )
    return rows


def split_figures(
    figures: Sequence[Dict[str, Any]],
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Figure-level split, stratified by family (ml/juri/bt)."""
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fig in figures:
        fam = fig["figure_key"].split("/", 1)[0]
        buckets[fam].append(fig)
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for fam, items in sorted(buckets.items()):
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio))) if len(items) >= 5 else max(
            0, int(round(len(items) * val_ratio))
        )
        n_val = min(n_val, len(items) - 1) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def figures_to_examples(figures: Sequence[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fig in figures:
        for i, claim in enumerate(fig["claims"], start=1):
            graph = extract_graph(claim, claim_id=i)
            rows.append(
                {
                    "id": f"{fig['figure_key']}#{i}",
                    "figure_key": fig["figure_key"],
                    "sample_id": fig["sample_id"],
                    "split_folder": fig["split_folder"],
                    "split": split_name,
                    "claim_id": i,
                    "claim": claim,
                    "graph": graph,
                    "image_path": fig["image_path"],
                    "sidecar_path": fig["sidecar_path"],
                }
            )
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_graph_sft_dataset(
    figures_dir: Path,
    out_dir: Path,
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    figures = collect_unique_figures(figures_dir)
    if limit is not None:
        figures = figures[: max(int(limit), 0)]
    train_figs, val_figs = split_figures(figures, val_ratio=val_ratio, seed=seed)
    train_rows = figures_to_examples(train_figs, "train")
    val_rows = figures_to_examples(val_figs, "val")
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    info = {
        "n_figures": len(figures),
        "n_train_figures": len(train_figs),
        "n_val_figures": len(val_figs),
        "n_train_claims": len(train_rows),
        "n_val_claims": len(val_rows),
        "val_ratio": val_ratio,
        "seed": seed,
        "train_jsonl": str(out_dir / "train.jsonl"),
        "val_jsonl": str(out_dir / "val.jsonl"),
        "gold_source": "rule_extractor",
        "held_out": "data/figures/golden",
        "train_figure_keys": [f["figure_key"] for f in train_figs],
        "val_figure_keys": [f["figure_key"] for f in val_figs],
    }
    (out_dir / "split_meta.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "graph SFT data: train_figs=%d val_figs=%d train_claims=%d val_claims=%d → %s",
        len(train_figs),
        len(val_figs),
        len(train_rows),
        len(val_rows),
        out_dir,
    )
    return info


def load_graph_sft_dataset(data_dir: Path) -> Dict[str, Any]:
    """Load a dataset that was built *before* training. Does not extract graphs."""
    data_dir = Path(data_dir)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(
            f"graph dataset not found under {data_dir} "
            "(need train.jsonl and val.jsonl). "
            "Build it first: python scripts/build_graph_sft_dataset.py"
        )
    meta_path = data_dir / "split_meta.json"
    info: Dict[str, Any] = {}
    if meta_path.is_file():
        info = json.loads(meta_path.read_text(encoding="utf-8"))
    info["train_jsonl"] = str(train_path)
    info["val_jsonl"] = str(val_path)
    return info
