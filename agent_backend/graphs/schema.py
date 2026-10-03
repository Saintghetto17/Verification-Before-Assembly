"""Closed scene-graph schema: claim text → nodes + geo/sem edges."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

GEO_RELS = (
    "on",
    "in",
    "contains",
    "left_of",
    "right_of",
    "above",
    "below",
    "between",
    "connects",
)
SEM_RELS = ("means", "labeled", "attr")
REL_CLASS = {r: "geo" for r in GEO_RELS}
REL_CLASS.update({r: "sem" for r in SEM_RELS})
NODE_KINDS = ("region", "object", "text", "attr")

SYSTEM_PROMPT = """\
You convert one visual-spec claim into a scene graph.
Reply with JSON only. No markdown, no extra keys, no commentary.

JSON schema:
{"nodes":[{"id":"n1","query":"...","kind":"..."}],"edges":[{"src":"n1","dst":"n2","rel":"...","class":"..."}]}

NODES
- id: n1, n2, ... in appearance order
- query: short phrase for an open-vocab detector (<= 8 words, no latex, no markdown)
- kind (exactly one):
  region  — labeled block, panel, model, loss box
  object  — icon, arrow, symbol, legend marker
  text    — readable inscription / quoted label
  attr    — property word: trained, frozen, dashed, red, ...

EDGES
- src / dst: node ids already listed in nodes
- class is geo or sem; rel must match that class

geo (checked later on bounding boxes):
  on         A is drawn on / attached to B
  in         A is inside B
  contains   A contains B
  left_of    A is to the left of B
  right_of   A is to the right of B
  above      A is above B
  below      A is below B
  between    A lies between B (use two edges A-between-X and A-between-Y)
  connects   arrow / connector from A to B

sem (checked later on the crop / OCR):
  means      A identifies / denotes / represents B
  labeled    A is labeled / reads / titled B
  attr       A has attribute B

Direction follows English: "A on B" => src=A, rel=on, dst=B.
Do not invent entities that are not in the claim.
Empty nodes or edges is allowed if nothing is grounded.\
"""

# One few-shot: canonical geo+sem pattern (fire on ControlNet means trained).
FEWSHOT_CLAIM = (
    "A fire symbol on the ControlNet identifies it as trained, "
    "whereas a snowflake symbol on the Base DiT identifies it as frozen."
)
FEWSHOT_GRAPH = {
    "nodes": [
        {"id": "n1", "query": "fire symbol", "kind": "object"},
        {"id": "n2", "query": "ControlNet", "kind": "region"},
        {"id": "n3", "query": "trained", "kind": "attr"},
        {"id": "n4", "query": "snowflake symbol", "kind": "object"},
        {"id": "n5", "query": "Base DiT", "kind": "region"},
        {"id": "n6", "query": "frozen", "kind": "attr"},
    ],
    "edges": [
        {"src": "n1", "dst": "n2", "rel": "on", "class": "geo"},
        {"src": "n1", "dst": "n3", "rel": "means", "class": "sem"},
        {"src": "n4", "dst": "n5", "rel": "on", "class": "geo"},
        {"src": "n4", "dst": "n6", "rel": "means", "class": "sem"},
    ],
}


def dump_graph(graph: Dict[str, Any]) -> str:
    """Stable compact JSON for SFT targets."""
    clean = canonicalize(graph)
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


def canonicalize(graph: Dict[str, Any]) -> Dict[str, Any]:
    nodes = []
    seen_q: Dict[str, str] = {}
    for n in graph.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        q = _norm_query(n.get("query") or "")
        if not q:
            continue
        kind = n.get("kind") if n.get("kind") in NODE_KINDS else _guess_kind(q)
        nid = str(n.get("id") or "").strip()
        if q in seen_q:
            continue
        if not nid:
            nid = f"n{len(nodes) + 1}"
        seen_q[q] = nid
        nodes.append({"id": nid, "query": q, "kind": kind})
    id_map = {n["id"]: n["id"] for n in nodes}
    # remap duplicate queries that had different ids
    q_to_id = {n["query"]: n["id"] for n in nodes}
    edges = []
    seen_e = set()
    for e in graph.get("edges") or []:
        if not isinstance(e, dict):
            continue
        rel = str(e.get("rel") or "").strip()
        if rel not in REL_CLASS:
            continue
        src = str(e.get("src") or "").strip()
        dst = str(e.get("dst") or "").strip()
        src = id_map.get(src, src)
        dst = id_map.get(dst, dst)
        if src not in id_map or dst not in id_map or src == dst:
            continue
        cls = REL_CLASS[rel]
        key = (src, dst, rel, cls)
        if key in seen_e:
            continue
        seen_e.add(key)
        edges.append({"src": src, "dst": dst, "rel": rel, "class": cls})
    # drop nodes with no edges only if we have many; keep labeled/text always
    if len(nodes) > 12:
        used = {e["src"] for e in edges} | {e["dst"] for e in edges}
        keep_kind = {"text", "attr", "object"}
        nodes = [n for n in nodes if n["id"] in used or n["kind"] in keep_kind][:12]
        keep_ids = {n["id"] for n in nodes}
        edges = [e for e in edges if e["src"] in keep_ids and e["dst"] in keep_ids]
    # reindex n1..nk for compactness
    remap = {n["id"]: f"n{i+1}" for i, n in enumerate(nodes)}
    nodes = [{"id": remap[n["id"]], "query": n["query"], "kind": n["kind"]} for n in nodes]
    edges = [
        {"src": remap[e["src"]], "dst": remap[e["dst"]], "rel": e["rel"], "class": e["class"]}
        for e in edges
        if e["src"] in remap and e["dst"] in remap
    ]
    return {"nodes": nodes, "edges": edges[:16]}


def parse_graph_json(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    obj = _loads(raw)
    if obj is None:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            obj = _loads(m.group(0))
    if not isinstance(obj, dict):
        return None
    try:
        return canonicalize(obj)
    except Exception:
        return None


def graph_sets(graph: Dict[str, Any]) -> Tuple[set, set]:
    g = canonicalize(graph)
    id_q = {n["id"]: n["query"] for n in g["nodes"]}
    nodes = set(id_q.values())
    edges = set()
    for e in g["edges"]:
        edges.add((id_q[e["src"]], e["rel"], id_q[e["dst"]]))
    return nodes, edges


def prf_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Micro P/R/F1 from TP/FP/FN. Empty gold and empty pred → 1.0."""
    if tp + fp + fn == 0:
        return {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1_val = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1_val),
    }


def f1(pred: Iterable, gold: Iterable) -> float:
    p, g = set(pred), set(gold)
    return prf_from_counts(len(p & g), len(p - g), len(g - p))["f1"]


def _loads(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _norm_query(q: str) -> str:
    q = str(q or "")
    q = re.sub(r"\*+", "", q)
    q = q.replace("“", '"').replace("”", '"').replace("’", "'")
    q = re.sub(r"\\[a-zA-Z]+", " ", q)
    q = re.sub(r"[{}$\\]", " ", q)
    q = re.sub(r"[()\[\]]", " ", q)
    q = re.sub(r"\s+", " ", q).strip(" \t\"'`.,;:")
    words = q.split()
    if len(words) > 8:
        q = " ".join(words[:8])
    return q[:80]


def _guess_kind(q: str) -> str:
    ql = q.lower()
    if ql in {
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
    }:
        return "attr"
    if any(k in ql for k in ("symbol", "arrow", "icon", "snowflake", "fire", "legend")):
        return "object"
    if len(q.split()) >= 4 or q[:1].islower():
        return "text"
    return "region"
