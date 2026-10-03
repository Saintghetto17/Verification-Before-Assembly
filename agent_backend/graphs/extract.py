"""Deterministic claim → scene graph for SFT targets.

Closed geo/sem vocabulary. Mentions come from quotes/bold plus
preposition and identification patterns. Not a general parser.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .schema import NODE_KINDS, canonicalize, _guess_kind, _norm_query

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_QUOTE = re.compile(r"[\"“](.+?)[\"”]")

_STOP = {
    "the",
    "a",
    "an",
    "its",
    "their",
    "this",
    "that",
    "these",
    "those",
    "and",
    "or",
    "of",
    "to",
    "for",
}

ATTR_WORDS = {
    "trained",
    "frozen",
    "dashed",
    "dotted",
    "solid",
    "red",
    "blue",
    "orange",
    "purple",
    "yellow",
    "green",
    "gray",
    "grey",
    "black",
    "white",
    "trainable",
    "fixed",
}


def extract_graph(claim: str, claim_id: Optional[int] = None) -> Dict:
    text = (claim or "").strip()
    nodes: List[Dict] = []
    edges: List[Dict] = []
    q2id: Dict[str, str] = {}

    def upsert(raw: str, kind: Optional[str] = None) -> Optional[str]:
        q = _norm_query(raw)
        if not q or q.lower() in _STOP or len(q) < 2:
            return None
        if q.lower() in q2id:
            return q2id[q.lower()]
        nid = f"n{len(nodes) + 1}"
        k = kind if kind in NODE_KINDS else _guess_kind(q)
        nodes.append({"id": nid, "query": q, "kind": k})
        q2id[q.lower()] = nid
        return nid

    for m in _BOLD.finditer(text):
        inner = m.group(1).strip().strip("\"“”")
        kind = "attr" if _norm_query(inner).lower() in ATTR_WORDS else "text"
        if " - " in inner or inner.startswith("🔥") or inner.startswith("❄"):
            upsert(inner, "text")
        else:
            upsert(inner, kind)
    for m in _QUOTE.finditer(text):
        inner = m.group(1).strip()
        if len(inner.split()) <= 12:
            upsert(inner, "text")

    for phrase, kind in _ICON_PHRASES:
        if re.search(phrase, text, re.I):
            upsert(phrase.replace(r"\s+", " "), kind)

    body = _BOLD.sub(lambda m: m.group(1), text)
    body = body.replace("“", '"').replace("”", '"')

    for src, dst, rel in _match_binary(body):
        s = upsert(src, "object" if "symbol" in src.lower() or "arrow" in src.lower() else None)
        d = upsert(dst)
        if s and d:
            edges.append({"src": s, "dst": d, "rel": rel})

    for src, dst, rel in _match_identify(body):
        s = upsert(src)
        d = upsert(dst, "attr" if _norm_query(dst).lower() in ATTR_WORDS else "text")
        if s and d:
            edges.append({"src": s, "dst": d, "rel": rel})

    for src, a, b in _match_between(body):
        s = upsert(src)
        ia = upsert(a)
        ib = upsert(b)
        if s and ia:
            edges.append({"src": s, "dst": ia, "rel": "between"})
        if s and ib:
            edges.append({"src": s, "dst": ib, "rel": "between"})

    graph = canonicalize({"nodes": nodes, "edges": edges})
    if claim_id is not None:
        graph["claim_id"] = int(claim_id)
    return graph


_ICON_PHRASES = [
    (r"fire symbol", "object"),
    (r"snowflake symbol", "object"),
    (r"snowflake", "object"),
    (r"red outlined arrow", "object"),
    (r"orange dotted arrow", "object"),
    (r"blue dotted arrow", "object"),
    (r"black downward arrow", "object"),
    (r"dotted arrow", "object"),
    (r"forward connector", "object"),
    (r"legend", "object"),
]


def _clean_chunk(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" ,.;:)")
    s = re.sub(
        r"^(whereas|while|thus|therefore|and|or|but|so)\s+",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(r"^(the|a|an|its|their)\s+", "", s, flags=re.I)
    # cut trailing verbs / relative clauses
    s = re.split(
        r"\s+(?:is|are|was|were|that|which|who|identifies|reads|labeled)\b",
        s,
        maxsplit=1,
    )[0]
    return s.strip()


def _match_binary(text: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    # standard PREP patterns: A <prep> B
    prep_pat = re.compile(
        rf"(?P<a>.{{3,55}}?)\s+(?P<p>on(?:to)?|inside|within|in|contains|enclosed by|"
        rf"to the left of|left of|to the right of|right of|above|over|below|beneath|under)\s+"
        rf"(?:the |a |an )?(?P<b>.{{3,55}}?)(?=(?:,|;|\.| whereas | while | and |$))",
        re.I,
    )
    prep_map = {
        "on": "on",
        "onto": "on",
        "in": "in",
        "inside": "in",
        "within": "in",
        "contains": "contains",
        "enclosed by": "contains",
        "to the left of": "left_of",
        "left of": "left_of",
        "to the right of": "right_of",
        "right of": "right_of",
        "above": "above",
        "over": "above",
        "below": "below",
        "beneath": "below",
        "under": "below",
    }
    for m in prep_pat.finditer(text):
        a = _clean_chunk(m.group("a"))
        b = _clean_chunk(m.group("b"))
        rel = prep_map.get(m.group("p").lower())
        if rel and a and b:
            # take last 6 words of a (subject sits at the end of a long clause)
            a = " ".join(a.split()[-6:])
            b = " ".join(b.split()[:6])
            if _ok_mention(a) and _ok_mention(b):
                out.append((a, b, rel))

    conn = re.compile(
        r"(?:arrow|connector)\b.{0,40}?\bfrom\s+(?:the |a )?(?P<a>.{3,40}?)"
        r"\s+(?:to|toward|into)\s+(?:the |a )?(?P<b>.{3,40}?)(?=[,.;]|$)",
        re.I,
    )
    for m in conn.finditer(text):
        a, b = _clean_chunk(m.group("a")), _clean_chunk(m.group("b"))
        if _ok_mention(a) and _ok_mention(b):
            out.append((a, b, "connects"))

    points = re.compile(
        r"points?\s+from\s+(?:the |a )?(?P<a>.{3,40}?)\s+(?:to|toward)\s+(?:the |a )?(?P<b>.{3,40}?)(?=[,.;]|$)",
        re.I,
    )
    for m in points.finditer(text):
        a, b = _clean_chunk(m.group("a")), _clean_chunk(m.group("b"))
        if _ok_mention(a) and _ok_mention(b):
            out.append((a, b, "connects"))
    return out


def _match_identify(text: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    # A on B identifies it as C  → A means C (and A on B already from prep)
    p1 = re.compile(
        r"(?P<a>.{3,50}?)\s+on\s+(?:the |a )?(?P<b>.{3,40}?)\s+identifies\s+"
        r"(?:it|them|this)\s+as\s+(?P<c>.{2,40}?)(?=(?:,|;|\.| whereas | while |$))",
        re.I,
    )
    for m in p1.finditer(text):
        a = " ".join(_clean_chunk(m.group("a")).split()[-6:])
        c = _clean_chunk(m.group("c")).strip("*\" ")
        if _ok_mention(a) and _ok_mention(c):
            out.append((a, c, "means"))

    p2 = re.compile(
        r"identifies\s+(?:it|them|this|the (?P<a>.{3,30}?))\s+as\s+(?P<c>.{2,40}?)(?=[,.;]|$)",
        re.I,
    )
    for m in p2.finditer(text):
        a = _clean_chunk(m.group("a") or "")
        c = _clean_chunk(m.group("c"))
        if a and _ok_mention(a) and _ok_mention(c):
            out.append((a, c, "means"))

    p3 = re.compile(
        r"(?P<a>.{3,40}?)\s+(?:denotes|represents|means|named)\s+(?P<c>.{2,40}?)(?=[,.;]|$)",
        re.I,
    )
    for m in p3.finditer(text):
        a = " ".join(_clean_chunk(m.group("a")).split()[-6:])
        c = _clean_chunk(m.group("c"))
        if _ok_mention(a) and _ok_mention(c):
            out.append((a, c, "means"))

    labeled = re.compile(
        r"(?:labeled|reads|titled|named)\s+(?P<c>.{2,60}?)(?=[,.;]|$)",
        re.I,
    )
    for m in labeled.finditer(text):
        c = _clean_chunk(m.group("c").strip(":* "))
        # try to find a nearby subject: "panel is labeled X" / "block labeled X"
        start = max(0, m.start() - 50)
        head = text[start : m.start()]
        subj = re.search(
            r"(legend|panel|block|region|box|title|caption|model|loss)\b[^.]*$",
            head,
            re.I,
        )
        src = subj.group(1) if subj else "label"
        if _ok_mention(c):
            out.append((src, c, "labeled"))
    return out


def _match_between(text: str) -> List[Tuple[str, str, str]]:
    out = []
    pat = re.compile(
        r"(?P<a>.{3,40}?)\s+between\s+(?:the |a )?(?P<b>.{3,30}?)\s+and\s+(?:the |a )?(?P<c>.{3,30}?)(?=[,.;]|$)",
        re.I,
    )
    for m in pat.finditer(text):
        a = " ".join(_clean_chunk(m.group("a")).split()[-6:])
        b, c = _clean_chunk(m.group("b")), _clean_chunk(m.group("c"))
        if _ok_mention(a) and _ok_mention(b) and _ok_mention(c):
            out.append((a, b, c))
    return out


def _ok_mention(s: str) -> bool:
    if not s or len(s) < 2:
        return False
    words = s.split()
    if len(words) > 8:
        return False
    low = s.lower()
    if low in _STOP:
        return False
    if low.startswith(
        ("which ", "that ", "and ", "or ", "from ", "with ", "whereas ", "while ", "thus ")
    ):
        return False
    return True
