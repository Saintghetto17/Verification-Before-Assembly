"""Load figure rows for agent train (non-golden) and eval (golden)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_golden(
    figures_dir: Path, *, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load held-out golden set from data/figures/golden/golden.jsonl."""
    path = Path(figures_dir) / "golden" / "golden.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"golden set not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _is_corrupted_split(split: str) -> bool:
    s = split.lower()
    return s.endswith("_l2_fail") or "l2_fail" in s


def discover_train_splits(figures_dir: Path) -> List[str]:
    figures_dir = Path(figures_dir)
    if not figures_dir.is_dir():
        raise FileNotFoundError(f"figures_dir not found: {figures_dir}")
    return sorted(
        p.name
        for p in figures_dir.iterdir()
        if p.is_dir() and p.name != "golden" and not p.name.startswith("_")
    )


def _read_sidecar(image_path: Path) -> Dict[str, Any]:
    meta_path = image_path.with_suffix(".json")
    if meta_path.is_file():
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            caption = (obj.get("caption") or "").strip()
            text_block = (obj.get("text_block") or obj.get("premise") or "").strip()
            gold = obj.get("gold") if isinstance(obj.get("gold"), dict) else {}
            visual = (obj.get("visual_description") or "").strip()
            claims = obj.get("visual_claims") if isinstance(obj.get("visual_claims"), list) else []
            return {
                "caption": caption,
                "text_block": text_block,
                "visual_description": visual,
                "visual_claims": claims,
                "visual_annotation": obj.get("visual_annotation") if isinstance(obj.get("visual_annotation"), dict) else {},
                "gold": gold,
                "sample_id": obj.get("sample_id"),
                "sidecar_path": str(meta_path),
            }
    for ext in (".txt", ".caption"):
        p = image_path.with_suffix(ext)
        if p.is_file():
            return {
                "caption": "",
                "text_block": p.read_text(encoding="utf-8").strip(),
                "visual_description": "",
                "visual_claims": [],
                "visual_annotation": {},
                "gold": {},
                "sample_id": None,
                "sidecar_path": str(p),
            }
    return {
        "caption": "",
        "text_block": "",
        "visual_description": "",
        "visual_claims": [],
        "visual_annotation": {},
        "gold": {},
        "sample_id": None,
        "sidecar_path": None,
    }


def load_train_figures(
    figures_dir: Path,
    *,
    splits: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    max_clean_per_corrupted: Optional[float] = 3.0,
    seed: int = 42,
    require_sidecar_text: bool = True,
) -> List[Dict[str, Any]]:
    """
    Walk data/figures/{split}/document_*/figure_*.png — **excluding golden**.

    Labels: sidecar gold.is_corrupted if present, else folder name
      *_l2_fail → mismatch, otherwise match.
    Premise = text_block, fallback caption.
    """
    figures_dir = Path(figures_dir)
    split_names = list(splits) if splits else discover_train_splits(figures_dir)
    raw: List[Dict[str, Any]] = []

    for split in split_names:
        split_dir = figures_dir / split
        if not split_dir.is_dir():
            continue
        default_mismatch = _is_corrupted_split(split)
        for img in sorted(split_dir.rglob("figure_*.png")):
            if img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            side = _read_sidecar(img)
            premise = side["text_block"] or side["caption"]
            if require_sidecar_text and not premise:
                continue
            gold = side.get("gold") or {}
            if "is_corrupted" in gold:
                is_corrupted = bool(gold["is_corrupted"])
            else:
                is_corrupted = default_mismatch
            label = "mismatch" if is_corrupted else "match"
            sid = side.get("sample_id") or f"{split}/{img.parent.name}/{img.stem}"
            raw.append(
                {
                    "sample_id": sid,
                    "split": split,
                    "label": label,
                    "label_id": 0 if label == "match" else 1,  # legacy L2 id
                    "gold_is_corrupted": is_corrupted,
                    "image_path": str(img.resolve()),
                    "sidecar_path": side.get("sidecar_path"),
                    "caption": side.get("caption") or "",
                    "text_block": side.get("text_block") or "",
                    "visual_description": side.get("visual_description") or "",
                    "visual_claims": side.get("visual_claims") or [],
                    "visual_annotation": side.get("visual_annotation") or {},
                    "premise": premise,
                    "gold": gold,
                }
            )

    # Optional class balance: cap clean relative to corrupted
    if max_clean_per_corrupted is not None and max_clean_per_corrupted > 0:
        clean = [r for r in raw if r["label"] == "match"]
        bad = [r for r in raw if r["label"] == "mismatch"]
        rng = random.Random(seed)
        rng.shuffle(clean)
        cap = int(len(bad) * max_clean_per_corrupted)
        if cap < len(clean):
            clean = clean[:cap]
        raw = clean + bad
        rng.shuffle(raw)

    if limit is not None:
        rng = random.Random(seed)
        if limit < len(raw):
            raw = rng.sample(raw, limit)

    return raw
