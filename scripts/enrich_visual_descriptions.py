#!/usr/bin/env python3
"""Write annotator visual descriptions into data/figures sidecars + golden.jsonl.

Sources (approved .txt next to PNG):
  kirill_mironov/
  Kolya_and_zhenya/rutkovskiy_evgenii/
  Kolya_and_zhenya/nikolay_ai_360_student/

Orig sidecars get a description of that image.
Paired fail sidecars get the *same* text as expected appearance (not of the fail PNG).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "data" / "figures"
SOURCES = [
    (ROOT / "kirill_mironov", "kirill_mironov"),
    (ROOT / "Kolya_and_zhenya" / "rutkovskiy_evgenii", "rutkovskiy_evgenii"),
    (ROOT / "Kolya_and_zhenya" / "nikolay_ai_360_student", "nikolay_ai_360_student"),
]

_CLAIM_RE = re.compile(r"^\s*(\d+)\.\s+", re.M)


def parse_claims(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    starts = [m.start() for m in _CLAIM_RE.finditer(text)]
    if not starts:
        return [text]
    claims: List[str] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[s:e].strip()
        chunk = re.sub(r"^\d+\.\s+", "", chunk).strip()
        if chunk:
            claims.append(chunk)
    return claims or [text]


def orig_key_from_txt(txt: Path, src_root: Path) -> Optional[str]:
    rel = txt.relative_to(src_root)
    parts = list(rel.parts)
    if parts[0] == "golden" and len(parts) >= 4:
        return f"golden/{parts[1]}/{parts[2]}/{txt.stem}"
    if parts[0].endswith("_orig") and len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}/{txt.stem}"
    return None


def collect_annotations() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for src_root, annotator in SOURCES:
        if not src_root.is_dir():
            continue
        for txt in src_root.rglob("figure_*.txt"):
            key = orig_key_from_txt(txt, src_root)
            if not key:
                continue
            body = txt.read_text(encoding="utf-8").strip()
            if not body:
                continue
            if key in out:
                raise SystemExit(f"duplicate annotation for {key}")
            out[key] = {
                "annotator": annotator,
                "source_txt": str(txt.relative_to(ROOT)),
                "visual_description": body,
                "visual_claims": parse_claims(body),
            }
    return out


def sidecar_path(key: str) -> Path:
    return FIG / f"{key}.json"


def fail_keys_for_orig(orig_key: str) -> List[str]:
    """ml_orig/doc/fig -> ml_l2_fail/... and golden variant if present."""
    if orig_key.startswith("golden/"):
        rest = orig_key[len("golden/") :]
        domain_split, doc, stem = rest.split("/")
        domain = domain_split.replace("_orig", "")
        return [f"golden/{domain}_l2_fail/{doc}/{stem}"]
    domain_split, doc, stem = orig_key.split("/")
    domain = domain_split.replace("_orig", "")
    return [f"{domain}_l2_fail/{doc}/{stem}"]


def patch_sidecar(path: Path, payload: Dict[str, Any], *, role: str) -> bool:
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    data["visual_description"] = payload["visual_description"]
    data["visual_claims"] = list(payload["visual_claims"])
    data["visual_annotation"] = {
        "annotator": payload["annotator"],
        "source_txt": payload["source_txt"],
        "n_claims": len(payload["visual_claims"]),
        "role": role,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    vis = path.with_name(path.stem + ".visual.txt")
    vis.write_text(payload["visual_description"] + "\n", encoding="utf-8")
    return True


def rewrite_golden_jsonl() -> Tuple[int, int]:
    jl = FIG / "golden" / "golden.jsonl"
    if not jl.is_file():
        return 0, 0
    rows = [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_hit = 0
    for r in rows:
        sid = r.get("sidecar_path") or ""
        p = Path(sid) if sid else None
        if p is None or not p.is_file():
            # reconstruct from sample_id
            sample = r.get("sample_id") or ""
            if sample.startswith("golden/"):
                p = FIG / f"{sample}.json"
            else:
                continue
        if not p.is_file():
            continue
        side = json.loads(p.read_text(encoding="utf-8"))
        desc = (side.get("visual_description") or "").strip()
        if not desc:
            r.pop("visual_description", None)
            r.pop("visual_claims", None)
            r.pop("visual_annotation", None)
            continue
        r["visual_description"] = desc
        r["visual_claims"] = side.get("visual_claims") or parse_claims(desc)
        r["visual_annotation"] = side.get("visual_annotation")
        n_hit += 1
    jl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return len(rows), n_hit


def main() -> int:
    anns = collect_annotations()
    stats = Counter()
    for orig_key, payload in sorted(anns.items()):
        ok = patch_sidecar(sidecar_path(orig_key), payload, role="image_self")
        stats["orig_written" if ok else "orig_missing_sidecar"] += 1
        inherited = False
        for fk in fail_keys_for_orig(orig_key):
            if patch_sidecar(sidecar_path(fk), payload, role="paired_orig_expected"):
                stats["fail_inherited"] += 1
                inherited = True
        if not inherited:
            stats["orig_no_fail_pair"] += 1
    n_rows, n_gold_hit = rewrite_golden_jsonl()
    stats["golden_jsonl_rows"] = n_rows
    stats["golden_jsonl_with_visual"] = n_gold_hit
    print(json.dumps(dict(stats), indent=2, ensure_ascii=False))
    print(f"annotations={len(anns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
