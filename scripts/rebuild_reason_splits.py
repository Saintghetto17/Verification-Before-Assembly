#!/usr/bin/env python3
"""Rebuild train/golden: 70/30 inside each photo subgroup.

Subgroups:
  1-3. reason-file fail figures (ml / bt / juri)
  4-6. orig counterparts of those fails (paired correct)
  7.   orig without a fail pair

Paired fail+orig share the same split to avoid leakage.
Source files: data/figures/_rebuild_backup (full corpus).
"""

from __future__ import annotations

import json
import random
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "data" / "figures"
BACKUP = FIG / "_rebuild_backup"
SEED = 42
TRAIN_FRAC = 0.70
# Unpaired orig per split = this fraction of paired-match (≈ mismatch) in that split.
ORIG_ONLY_PER_PAIRED = 0.5

_CYR = str.maketrans({"а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y"})


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "").strip().lower()).translate(_CYR)


def _parse_ref(obj: str) -> Tuple[Optional[str], Optional[str], bool]:
    o = _norm(obj)
    if "graphical" in o and "abstract" in o:
        return None, None, True
    m = re.search(r"figure\s*([0-9]+)\s*([a-z])?", o)
    if not m:
        return None, None, False
    return m.group(1), m.group(2), False


def _doc_id(domain: str, doc: dict) -> Optional[str]:
    if domain == "bt":
        idx = doc.get("index")
        if idx is not None:
            return f"document_{idx}"
        f = str(doc.get("local_pdf_path") or "").replace(".pdf", "")
        return f if f.startswith("document_") else None
    f = str(doc.get("file") or "").replace(".pdf", "")
    return f if f.startswith("document_") else None


def _search_roots() -> List[Path]:
    roots = [BACKUP]
    if FIG.exists():
        roots.append(FIG)
    return roots


def _resolve_png(domain_split: str, did: Optional[str], obj: str) -> Optional[Path]:
    num, panel, is_ga = _parse_ref(obj)
    if is_ga or not num or not did:
        return None
    names: List[str] = []
    if panel:
        names.append(f"figure_{num}{panel}.png")
    names.append(f"figure_{num}.png")
    seen = set()
    for root in _search_roots():
        for loc in (root / domain_split / did, root / "golden" / domain_split / did):
            if not loc.is_dir():
                continue
            cand = list(names)
            if not panel:
                cand.extend(sorted(p.name for p in loc.glob(f"figure_{num}*.png")))
            for name in cand:
                if name in seen:
                    continue
                seen.add(name)
                path = loc / name
                if path.is_file():
                    return path
    return None


def _find_orig(domain: str, doc: str, stem: str) -> Optional[Path]:
    orig_split = f"{domain}_orig"
    for root in _search_roots():
        for loc in (root / orig_split / doc, root / "golden" / orig_split / doc):
            p = loc / f"{stem}.png"
            if p.is_file():
                return p
    return None


def _read_json(path: Path) -> Dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_png_json(src_png: Path, dest_png: Path) -> None:
    dest_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_png, dest_png)
    src_json = src_png.with_suffix(".json")
    if src_json.is_file():
        shutil.copy2(src_json, dest_png.with_suffix(".json"))


def _split_70_30(items: List[Any], rng: random.Random) -> Tuple[List[Any], List[Any]]:
    items = list(items)
    rng.shuffle(items)
    n_train = int(round(len(items) * TRAIN_FRAC))
    n_train = min(max(n_train, 0), len(items))
    train, gold = items[:n_train], items[n_train:]
    key = lambda x: (x["doc"], x["stem"]) if isinstance(x, dict) else (x.parent.name, x.stem)
    return sorted(train, key=key), sorted(gold, key=key)


def collect_reason_fails() -> Dict[str, List[dict]]:
    specs = [
        ("ml", "ml_l2_fail", ROOT / "ml_l2_reason.json", 276),
        ("bt", "bt_l2_fail", ROOT / "bt_l2_fail_reason.json", 214),
        ("juri", "juri_l2_fail", ROOT / "juri_l2_fail.json", 128),
    ]
    out: Dict[str, List[dict]] = {}
    for domain, split, rpath, listed in specs:
        data = json.loads(rpath.read_text(encoding="utf-8"))
        uniq: Dict[Tuple[str, str], dict] = {}
        n_rows = 0
        unresolved = 0
        for doc in data:
            did = _doc_id(domain, doc)
            for fx in doc.get("fixed") or []:
                obj = (fx.get("object") or "").strip()
                lie = (fx.get("lie") or "").strip()
                if "figure" not in _norm(obj):
                    continue
                n_rows += 1
                png = _resolve_png(split, did, obj)
                if png is None:
                    unresolved += 1
                    continue
                key = (png.parent.name, png.stem)
                rec = uniq.setdefault(
                    key,
                    {
                        "domain": domain,
                        "split": split,
                        "doc": png.parent.name,
                        "stem": png.stem,
                        "png": png,
                        "objects": [],
                        "lies": [],
                    },
                )
                if obj and obj not in rec["objects"]:
                    rec["objects"].append(obj)
                if lie and lie not in rec["lies"]:
                    rec["lies"].append(lie)
        items = [uniq[k] for k in sorted(uniq)]
        n_orig = sum(1 for r in items if _find_orig(domain, r["doc"], r["stem"]))
        print(
            f"{domain}: json_figure_rows={n_rows} (listed {listed}) "
            f"unique_png={len(items)} orig_pairs={n_orig} unresolved={unresolved}"
        )
        out[domain] = items
    return out


def collect_orig_only(domain: str, paired_keys: set) -> List[Path]:
    orig_split = f"{domain}_orig"
    found: Dict[Tuple[str, str], Path] = {}
    for root in _search_roots():
        for loc in (root / orig_split, root / "golden" / orig_split):
            if not loc.is_dir():
                continue
            for p in sorted(loc.rglob("figure_*.png")):
                key = (p.parent.name, p.stem)
                if key in paired_keys:
                    continue
                found.setdefault(key, p)
    return [found[k] for k in sorted(found)]


def _rewrite_fail_sidecar(dest_png: Path, rec: dict, *, golden: bool) -> None:
    sidecar = dest_png.with_suffix(".json")
    data = _read_json(sidecar) or _read_json(rec["png"].with_suffix(".json"))
    split = rec["split"]
    prefix = f"golden/{split}" if golden else split
    data["sample_id"] = f"{prefix}/{rec['doc']}/{rec['stem']}"
    data["split"] = split
    gold = data.get("gold") if isinstance(data.get("gold"), dict) else {}
    gold["is_corrupted"] = True
    gold["from_summary_fixed"] = bool(rec["lies"])
    gold["lies"] = list(rec["lies"])
    gold["objects"] = list(rec["objects"])
    gold["lie"] = "; ".join(rec["lies"])
    gold["subgroup"] = f"{rec['domain']}_l2_reason_fail"
    data["gold"] = gold
    data["image_file"] = dest_png.name
    _write_json(sidecar, data)


def _rewrite_orig_sidecar(dest_png: Path, src_png: Path, *, domain: str, golden: bool, subgroup: str) -> None:
    sidecar = dest_png.with_suffix(".json")
    data = _read_json(sidecar) or _read_json(src_png.with_suffix(".json"))
    split = f"{domain}_orig"
    prefix = f"golden/{split}" if golden else split
    doc, stem = dest_png.parent.name, dest_png.stem
    data["sample_id"] = f"{prefix}/{doc}/{stem}"
    data["split"] = split
    gold = data.get("gold") if isinstance(data.get("gold"), dict) else {}
    gold["is_corrupted"] = False
    gold["from_summary_fixed"] = False
    gold["lies"] = []
    gold["objects"] = []
    gold["lie"] = ""
    gold["subgroup"] = subgroup
    gold.pop("note", None)
    gold.pop("edit_status", None)
    gold.pop("error_family", None)
    data["gold"] = gold
    data["image_file"] = dest_png.name
    _write_json(sidecar, data)


def _place_fail(rec: dict, dest_root: Path, *, golden: bool) -> Path:
    dest = dest_root / rec["doc"] / f"{rec['stem']}.png"
    _copy_png_json(rec["png"], dest)
    _rewrite_fail_sidecar(dest, rec, golden=golden)
    return dest


def _place_orig(src: Path, dest: Path, *, domain: str, golden: bool, subgroup: str) -> Path:
    _copy_png_json(src, dest)
    _rewrite_orig_sidecar(dest, src, domain=domain, golden=golden, subgroup=subgroup)
    return dest


def _golden_row(png: Path, *, domain: str, split: str, label: str) -> dict:
    sidecar = png.with_suffix(".json")
    data = _read_json(sidecar)
    gold = data.get("gold") if isinstance(data.get("gold"), dict) else {}
    caption = (data.get("caption") or "").strip()
    text_block = (data.get("text_block") or data.get("premise") or "").strip()
    return {
        "sample_id": f"golden/{split}/{png.parent.name}/{png.stem}",
        "source_sample_id": f"{split}/{png.parent.name}/{png.stem}",
        "split": split,
        "domain": domain,
        "label": label,
        "label_id": 1 if label == "mismatch" else 0,
        "gold_is_corrupted": bool(gold.get("is_corrupted")) if "is_corrupted" in gold else label == "mismatch",
        "image_path": str(png.resolve()),
        "sidecar_path": str(sidecar.resolve()),
        "caption": caption,
        "text_block": text_block,
        "premise": text_block or caption,
        "gold": gold,
    }


def main() -> int:
    rng = random.Random(SEED)
    if not BACKUP.is_dir():
        raise SystemExit(f"backup not found: {BACKUP}")

    fails = collect_reason_fails()
    golden_fails: Dict[str, List[dict]] = {}
    train_fails: Dict[str, List[dict]] = {}
    for domain, items in fails.items():
        tr, gd = _split_70_30(items, rng)
        train_fails[domain], golden_fails[domain] = tr, gd
        print(f"fail split {domain}: train={len(tr)} golden={len(gd)} ({100*len(tr)/max(len(items),1):.0f}/{100*len(gd)/max(len(items),1):.0f})")

    orig_only: Dict[str, List[Path]] = {}
    paired_keys: Dict[str, set] = {}
    for domain in ("ml", "juri", "bt"):
        paired_keys[domain] = {(r["doc"], r["stem"]) for r in fails[domain]}
        orig_only[domain] = collect_orig_only(domain, paired_keys[domain])
        print(f"orig-only {domain}: {len(orig_only[domain])}")

    train_orig_only: Dict[str, List[Path]] = {}
    golden_orig_only: Dict[str, List[Path]] = {}
    for domain, pool in orig_only.items():
        n_paired_train = sum(
            1 for r in train_fails[domain] if _find_orig(domain, r["doc"], r["stem"])
        )
        n_paired_gold = sum(
            1 for r in golden_fails[domain] if _find_orig(domain, r["doc"], r["stem"])
        )
        n_tr = int(round(n_paired_train * ORIG_ONLY_PER_PAIRED))
        n_gd = int(round(n_paired_gold * ORIG_ONLY_PER_PAIRED))
        items = list(pool)
        rng.shuffle(items)
        n_tr = min(max(n_tr, 0), len(items))
        n_gd = min(max(n_gd, 0), len(items) - n_tr)
        key = lambda p: (p.parent.name, p.stem)
        tr = sorted(items[:n_tr], key=key)
        gd = sorted(items[n_tr : n_tr + n_gd], key=key)
        train_orig_only[domain], golden_orig_only[domain] = tr, gd
        print(
            f"orig-only {domain}: paired train/golden={n_paired_train}/{n_paired_gold} "
            f"→ keep train={len(tr)} golden={len(gd)} (×{ORIG_ONLY_PER_PAIRED}) "
            f"from pool {len(pool)}"
        )

    staging = FIG / "_rebuild_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    golden_fail_pngs: Dict[str, List[Path]] = defaultdict(list)
    golden_orig_pngs: Dict[str, List[Path]] = defaultdict(list)

    for domain, recs in golden_fails.items():
        dest_root = staging / "golden" / f"{domain}_l2_fail"
        for rec in recs:
            golden_fail_pngs[domain].append(_place_fail(rec, dest_root, golden=True))
    for domain, recs in train_fails.items():
        dest_root = staging / f"{domain}_l2_fail"
        for rec in recs:
            _place_fail(rec, dest_root, golden=False)

    n_pairs = defaultdict(int)
    for domain in ("ml", "juri", "bt"):
        for rec in golden_fails[domain]:
            src = _find_orig(domain, rec["doc"], rec["stem"])
            if src is None:
                continue
            n_pairs[domain] += 1
            dest = staging / "golden" / f"{domain}_orig" / rec["doc"] / f"{rec['stem']}.png"
            golden_orig_pngs[domain].append(
                _place_orig(
                    src, dest, domain=domain, golden=True,
                    subgroup=f"{domain}_paired_to_failed_but_correct",
                )
            )
        for rec in train_fails[domain]:
            src = _find_orig(domain, rec["doc"], rec["stem"])
            if src is None:
                continue
            n_pairs[domain] += 1
            dest = staging / f"{domain}_orig" / rec["doc"] / f"{rec['stem']}.png"
            _place_orig(
                src, dest, domain=domain, golden=False,
                subgroup=f"{domain}_paired_to_failed_but_correct",
            )

    for domain, paths in golden_orig_only.items():
        for src in paths:
            dest = staging / "golden" / f"{domain}_orig" / src.parent.name / src.name
            golden_orig_pngs[domain].append(
                _place_orig(
                    src, dest, domain=domain, golden=True,
                    subgroup="orig_that_doesnt_have_fail_pair",
                )
            )
    for domain, paths in train_orig_only.items():
        for src in paths:
            dest = staging / f"{domain}_orig" / src.parent.name / src.name
            _place_orig(
                src, dest, domain=domain, golden=False,
                subgroup="orig_that_doesnt_have_fail_pair",
            )
            # count later from disk

    golden_dir = staging / "golden"
    golden_dir.mkdir(exist_ok=True)

    meta = {
        "seed": SEED,
        "train_frac": TRAIN_FRAC,
        "orig_only_per_paired": ORIG_ONLY_PER_PAIRED,
        "json_listed": {"ml": 276, "bt": 214, "juri": 128},
        "unique_fail_png": {d: len(fails[d]) for d in fails},
        "fail_train": {d: len(train_fails[d]) for d in fails},
        "fail_golden": {d: len(golden_fails[d]) for d in fails},
        "paired_orig_total": dict(n_pairs),
        "orig_only_train": {d: len(train_orig_only[d]) for d in orig_only},
        "orig_only_golden": {d: len(golden_orig_only[d]) for d in orig_only},
        "note": (
            "Fail/paired-orig: 70% train / 30% golden, pairs share the split. "
            "Unpaired orig per split = 0.5 × paired-match in that split. "
            "Composition per split: X mismatch, X paired-match, 0.5X unpaired-match "
            "(bt has no orig, so paired/unpaired bt stay 0). "
            "Counts below json_listed are unique resolved PNGs."
        ),
    }
    _write_json(golden_dir / "meta.json", meta)

    live_dirs = [
        "ml_l2_fail", "bt_l2_fail", "juri_l2_fail",
        "ml_orig", "juri_orig", "bt_orig", "golden",
    ]
    trash = FIG / "_rebuild_prev_live"
    if trash.exists():
        shutil.rmtree(trash)
    trash.mkdir()
    for name in live_dirs:
        src = FIG / name
        if src.exists():
            shutil.move(str(src), str(trash / name))
    for child in staging.iterdir():
        shutil.move(str(child), str(FIG / child.name))
    shutil.rmtree(staging)
    shutil.rmtree(trash)

    # Write golden manifests after the move so image_path points at live files.
    rows: List[dict] = []
    for domain in ("ml", "bt", "juri"):
        gdir = FIG / "golden" / f"{domain}_l2_fail"
        if gdir.is_dir():
            for p in sorted(gdir.rglob("figure_*.png")):
                rows.append(_golden_row(p, domain=domain, split=f"{domain}_l2_fail", label="mismatch"))
        gdir = FIG / "golden" / f"{domain}_orig"
        if gdir.is_dir():
            for p in sorted(gdir.rglob("figure_*.png")):
                rows.append(_golden_row(p, domain=domain, split=f"{domain}_orig", label="match"))
    rows.sort(key=lambda r: r["sample_id"])
    match_rows = [r for r in rows if r["label"] == "match"]
    mismatch_rows = [r for r in rows if r["label"] == "mismatch"]
    meta["golden_n"] = len(rows)
    meta["golden_by_label"] = {"match": len(match_rows), "mismatch": len(mismatch_rows)}
    golden_dir = FIG / "golden"
    (golden_dir / "golden.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    _write_json(golden_dir / "match.json", match_rows)
    _write_json(golden_dir / "mismatch.json", mismatch_rows)
    _write_json(golden_dir / "meta.json", meta)

    print("DONE", json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
