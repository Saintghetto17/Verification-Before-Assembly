"""Shared text helpers for judges / agent."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


_FIG_CAPTION_RE = re.compile(
    r"((?:fig(?:ure)?\.?\s*\d+[a-z]?)\s*[:.\-–—]?\s*[^\n.!?]{8,200})",
    re.IGNORECASE,
)


def short_caption(
    *,
    caption: Optional[str] = None,
    text_block: Optional[str] = None,
    premise: Optional[str] = None,
    max_chars: int = 280,
) -> str:
    """
    Prefer sidecar caption; else first Fig.N sentence from page text; else head of premise.
    Used by Judges B/C (not semantic).
    """
    cap = (caption or "").strip()
    if cap:
        return cap[:max_chars]

    blob = (text_block or premise or "").strip()
    if not blob:
        return ""

    m = _FIG_CAPTION_RE.search(blob)
    if m:
        frag = re.sub(r"\s+", " ", m.group(0)).strip()
        return frag[:max_chars]

    # First sentence-ish chunk
    cut = re.split(r"(?<=[.!?])\s+", blob, maxsplit=1)[0]
    cut = re.sub(r"\s+", " ", cut).strip()
    return cut[:max_chars]


def row_texts(row: Dict[str, Any]) -> tuple[str, str]:
    """Return (premise_for_sem, caption_for_bc) from a dataset row.

    Judge A still gets paper text_block (64-token SigLIP). visual_description
    is a separate sidecar field for claim/graph judges — do not dump it into
    score_sem until the encoder can consume long structured text.
    """
    caption = (row.get("caption") or "").strip()
    text_block = (row.get("text_block") or "").strip()
    premise = (row.get("premise") or text_block or caption or "").strip()
    short = short_caption(caption=caption, text_block=text_block, premise=premise)
    return premise, short
