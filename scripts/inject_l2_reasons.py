#!/usr/bin/env python3
"""Inject lie reasons from *_reason.json into figure sidecar JSONs under data/figures."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cyrillic lookalikes → latin
_CYR_TO_LAT = str.maketrans({"а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y"})

_FIG_RE = re.compile(
    r"figure\s*(?:graphical\s*abstract)?\s*([0-9]+)?\s*([a-zA-Zа-яА-Я])?",
    re.I,
)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", (s or "").strip().lower())
    return s.translate(_CYR_TO_LAT)


def _parse_figure_ref(obj: str) -> Tuple[Optional[str], Optional[str], bool]:
    """Return (number, panel_letter, is_graphical_abstract)."""
    o = _norm(obj)
    if "graphical" in o and "abstract" in o:
        return None, None, True
    m = re.search(r"figure\s*([0-9]+)\s*([a-z])?", o)
    if not m:
        return None, None, False
    return m.group(1), m.group(2), False


def resolve_sidecar(doc_dir: Path, obj: str) -> Optional[Path]:
    """Map reason object string → figure_*.json path."""
    if not doc_dir.is_dir():
        return None
    num, panel, is_ga = _parse_figure_ref(obj)
    if is_ga:
        # no dedicated GA file in this dataset — skip
        return None
    if not num:
        return None

    candidates: List[str] = []
    if panel:
        candidates.append(f"figure_{num}{panel}.json")
    candidates.append(f"figure_{num}.json")
    # panel variants on disk: figure_2b when reason says figure 2
    if not panel:
        for p in sorted(doc_dir.glob(f"figure_{num}*.json")):
            candidates.append(p.name)

    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        path = doc_dir / name
        if path.is_file():
            return path
    return None


DEFAULT_MAP = {
    "bt_l2_fail": "bt_l2_fail_reason.json",
    "juri_l2_fail": "juri_l2_fail.json",
    "ml_l2_fail": "ml_l2_reason.json",
}


def inject_split(
    project_root: Path,
    split: str,
    reason_path: Path,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if not reason_path.is_file() or reason_path.stat().st_size == 0:
        return {"split": split, "skipped": True, "reason": f"empty/missing {reason_path}"}

    reasons = json.loads(reason_path.read_text(encoding="utf-8"))
    if not isinstance(reasons, list):
        raise ValueError(f"{reason_path}: expected list")

    fig_root = project_root / "data" / "figures" / split
    updated = 0
    matched = 0
    missing: List[str] = []
    # accumulate lies per sidecar (one file may get several)
    by_path: Dict[Path, List[Tuple[str, str]]] = {}

    for doc in reasons:
        idx = doc.get("index")
        doc_dir = fig_root / f"document_{idx}"
        for item in doc.get("fixed") or []:
            obj = (item.get("object") or "").strip()
            lie = (item.get("lie") or "").strip()
            if not obj or not lie:
                continue
            if "figure" not in _norm(obj):
                continue
            path = resolve_sidecar(doc_dir, obj)
            if path is None:
                missing.append(f"document_{idx}/{obj}")
                continue
            matched += 1
            by_path.setdefault(path, []).append((obj, lie))

    for path, pairs in by_path.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        gold = data.get("gold") if isinstance(data.get("gold"), dict) else {}
        objects = list(gold.get("objects") or [])
        lies = list(gold.get("lies") or [])
        for obj, lie in pairs:
            if obj not in objects:
                objects.append(obj)
            if lie not in lies:
                lies.append(lie)
        gold["objects"] = objects
        gold["lies"] = lies
        gold["is_corrupted"] = True
        gold["from_summary_fixed"] = True
        # convenient single-string explanation
        gold["lie"] = "; ".join(lies)
        data["gold"] = gold
        if not dry_run:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        updated += 1

    return {
        "split": split,
        "reason_file": str(reason_path),
        "matched_fixed": matched,
        "sidecars_updated": updated,
        "missing": len(missing),
        "missing_sample": missing[:20],
        "dry_run": dry_run,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--only",
        choices=list(DEFAULT_MAP),
        action="append",
        default=None,
        help="Limit to split(s); default: all mapped files",
    )
    args = ap.parse_args()
    root = args.project_root
    splits = args.only or list(DEFAULT_MAP)
    reports = []
    for split in splits:
        reason_name = DEFAULT_MAP[split]
        rep = inject_split(root, split, root / reason_name, dry_run=args.dry_run)
        reports.append(rep)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
