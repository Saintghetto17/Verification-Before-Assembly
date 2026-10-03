"""Prototype: extract a claim graph with spaCy rules.

Usage:
    python -m agent_backend.spacy_graph_test \
        "ControlNet is above the decoder and the legend contains eight bars."
    python -m agent_backend.spacy_graph_test --debug --file claim.txt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import spacy
from spacy.matcher import DependencyMatcher, Matcher
from spacy.tokens import Doc, Span, Token
from tqdm import tqdm

from .graphs.schema import canonicalize


PREP_RELATIONS = {
    "on": "on",
    "onto": "on",
    "in": "in",
    "inside": "in",
    "within": "in",
    "above": "above",
    "over": "above",
    "below": "below",
    "beneath": "below",
    "under": "below",
}

VERB_RELATIONS = {
    "contain": "contains",
    "label": "labeled",
    "title": "labeled",
    "name": "labeled",
    "read": "labeled",
    "mean": "means",
    "represent": "means",
    "denote": "means",
}

ATTR_WORDS = {
    "black",
    "blue",
    "dashed",
    "dotted",
    "fixed",
    "frozen",
    "green",
    "orange",
    "purple",
    "red",
    "solid",
    "trained",
    "trainable",
    "white",
    "yellow",
}


def _entity_text(token: Token) -> str:
    """Return the noun phrase containing a dependency-match token."""
    chunk = next((chunk for chunk in token.doc.noun_chunks if token in chunk), None)
    tokens = list(chunk) if chunk is not None else [token]
    while tokens and tokens[0].dep_ in {"det", "predet"}:
        tokens.pop(0)
    text = " ".join(tok.text for tok in tokens).strip(" \t,.;:\"'`()")
    return text or token.text


def _kind(query: str) -> str:
    low = query.lower()
    if low in ATTR_WORDS:
        return "attr"
    if any(word in low for word in ("arrow", "symbol", "icon", "connector")):
        return "object"
    if any(word in low for word in ("panel", "region", "block", "box", "area")):
        return "region"
    return "text"


class SpacyGraphExtractor:
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model {model!r} is unavailable. Install it with: "
                f"python -m spacy download {model}"
            ) from exc
        if "parser" not in self.nlp.pipe_names:
            raise RuntimeError(f"spaCy model {model!r} has no dependency parser")

        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add(
            "left_of",
            [
                [{"LOWER": "left"}, {"LOWER": "of"}],
                [{"LOWER": "to"}, {"LOWER": "the"}, {"LOWER": "left"}, {"LOWER": "of"}],
            ],
        )
        self.matcher.add(
            "right_of",
            [
                [{"LOWER": "right"}, {"LOWER": "of"}],
                [{"LOWER": "to"}, {"LOWER": "the"}, {"LOWER": "right"}, {"LOWER": "of"}],
            ],
        )
        self.matcher.add(
            "between",
            [[{"LOWER": "between"}]],
        )

        self.dependency_matcher = DependencyMatcher(self.nlp.vocab)
        self.dependency_matcher.add(
            "copular_prep",
            [[
                {
                    "RIGHT_ID": "predicate",
                    "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB"]}},
                },
                {
                    "LEFT_ID": "predicate",
                    "REL_OP": ">",
                    "RIGHT_ID": "source",
                    "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}},
                },
                {
                    "LEFT_ID": "predicate",
                    "REL_OP": ">",
                    "RIGHT_ID": "relation",
                    "RIGHT_ATTRS": {
                        "DEP": "prep",
                        "LOWER": {"IN": list(PREP_RELATIONS)},
                    },
                },
                {
                    "LEFT_ID": "relation",
                    "REL_OP": ">",
                    "RIGHT_ID": "target",
                    "RIGHT_ATTRS": {"DEP": "pobj"},
                },
            ]],
        )
        self.dependency_matcher.add(
            "noun_prep",
            [[
                {
                    "RIGHT_ID": "source",
                    "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}},
                },
                {
                    "LEFT_ID": "source",
                    "REL_OP": ">",
                    "RIGHT_ID": "relation",
                    "RIGHT_ATTRS": {
                        "DEP": "prep",
                        "LOWER": {"IN": list(PREP_RELATIONS)},
                    },
                },
                {
                    "LEFT_ID": "relation",
                    "REL_OP": ">",
                    "RIGHT_ID": "target",
                    "RIGHT_ATTRS": {"DEP": "pobj"},
                },
            ]],
        )
        self.dependency_matcher.add(
            "transitive_relation",
            [[
                {
                    "RIGHT_ID": "relation",
                    "RIGHT_ATTRS": {"LEMMA": {"IN": list(VERB_RELATIONS)}},
                },
                {
                    "LEFT_ID": "relation",
                    "REL_OP": ">",
                    "RIGHT_ID": "source",
                    "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}},
                },
                {
                    "LEFT_ID": "relation",
                    "REL_OP": ">",
                    "RIGHT_ID": "target",
                    "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "attr", "oprd"]}},
                },
            ]],
        )
        self.dependency_matcher.add(
            "connect_from_to",
            [[
                {
                    "RIGHT_ID": "relation",
                    "RIGHT_ATTRS": {"LEMMA": "connect"},
                },
                {
                    "LEFT_ID": "relation",
                    "REL_OP": ">",
                    "RIGHT_ID": "source",
                    "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj"]}},
                },
                {
                    "LEFT_ID": "relation",
                    "REL_OP": ">",
                    "RIGHT_ID": "to",
                    "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "to"},
                },
                {
                    "LEFT_ID": "to",
                    "REL_OP": ">",
                    "RIGHT_ID": "target",
                    "RIGHT_ATTRS": {"DEP": "pobj"},
                },
            ]],
        )

    @staticmethod
    def _nearest_chunks(doc: Doc, marker: Span) -> tuple[Token, Token] | None:
        chunks = list(doc.noun_chunks)
        left = [chunk for chunk in chunks if chunk.end <= marker.start]
        right = [chunk for chunk in chunks if chunk.start >= marker.end]
        if not left or not right:
            return None
        return left[-1].root, right[0].root

    def extract(self, description: str, *, debug: bool = False) -> dict[str, Any]:
        doc = self.nlp(description)
        raw_edges: list[tuple[str, str, str, str]] = []
        matcher_hits: list[dict[str, Any]] = []

        for match_id, start, end in self.matcher(doc):
            rule = self.nlp.vocab.strings[match_id]
            marker = doc[start:end]
            pair = self._nearest_chunks(doc, marker)
            matcher_hits.append({"rule": rule, "text": marker.text})
            if pair is not None and rule in {"left_of", "right_of"}:
                raw_edges.append(
                    (_entity_text(pair[0]), rule, _entity_text(pair[1]), f"matcher:{rule}")
                )

        for match_id, token_ids in self.dependency_matcher(doc):
            rule = self.nlp.vocab.strings[match_id]
            tokens = [doc[index] for index in token_ids]
            if rule == "copular_prep":
                _, source, relation, target = tokens
                rel = PREP_RELATIONS[relation.lower_]
            elif rule == "noun_prep":
                source, relation, target = tokens
                rel = PREP_RELATIONS[relation.lower_]
            elif rule == "transitive_relation":
                relation, source, target = tokens
                rel = VERB_RELATIONS[relation.lemma_.lower()]
            else:
                _, source, _, target = tokens
                rel = "connects"
            raw_edges.append(
                (_entity_text(source), rel, _entity_text(target), f"dependency:{rule}")
            )

        nodes: list[dict[str, str]] = []
        edges: list[dict[str, str]] = []
        query_to_id: dict[str, str] = {}
        seen_edges: set[tuple[str, str, str]] = set()

        def node_id(query: str) -> str:
            key = query.casefold()
            if key not in query_to_id:
                identifier = f"n{len(nodes) + 1}"
                query_to_id[key] = identifier
                nodes.append({"id": identifier, "query": query, "kind": _kind(query)})
            return query_to_id[key]

        for source, relation, target, rule in raw_edges:
            source_id = node_id(source)
            target_id = node_id(target)
            key = (source_id, relation, target_id)
            if source_id == target_id or key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append(
                {
                    "src": source_id,
                    "dst": target_id,
                    "rel": relation,
                    "_rule": rule,
                }
            )

        graph = canonicalize(
            {
                "nodes": nodes,
                "edges": [
                    {key: value for key, value in edge.items() if key != "_rule"}
                    for edge in edges
                ],
            }
        )
        if debug:
            graph["_debug"] = {
                "tokens": [
                    {
                        "text": token.text,
                        "lemma": token.lemma_,
                        "pos": token.pos_,
                        "dep": token.dep_,
                        "head": token.head.text,
                    }
                    for token in doc
                ],
                "matcher_hits": matcher_hits,
                "dependency_edges": [
                    {
                        "src": source,
                        "rel": relation,
                        "dst": target,
                        "rule": rule,
                    }
                    for source, relation, target, rule in raw_edges
                ],
            }
        return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a test scene graph from an English description using spaCy."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("description", nargs="?", help="Description supplied on the command line")
    source.add_argument("--file", type=Path, help="Read description from a UTF-8 text file")
    source.add_argument("--input-jsonl", type=Path, help="Process claims from a JSONL dataset")
    parser.add_argument("--output-jsonl", type=Path, help="Write validation records as JSONL")
    parser.add_argument("--output-json", type=Path, help="Write wrapped validation results as JSON")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extractor = SpacyGraphExtractor(args.model)
    if args.input_jsonl is not None:
        if (args.output_jsonl is None) == (args.output_json is None):
            raise SystemExit(
                "with --input-jsonl, set exactly one of --output-jsonl/--output-json"
            )
        with args.input_jsonl.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        output_path = args.output_jsonl or args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = []
        output = (
            args.output_jsonl.open("w", encoding="utf-8")
            if args.output_jsonl is not None
            else None
        )
        try:
            for row in tqdm(rows, desc="spaCy graph validation", unit="claim"):
                error = None
                prediction_graph = None
                try:
                    prediction_graph = extractor.extract(row["claim"], debug=args.debug)
                    if row.get("claim_id") is not None:
                        prediction_graph["claim_id"] = int(row["claim_id"])
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                prediction_text = (
                    json.dumps(prediction_graph, ensure_ascii=False)
                    if prediction_graph is not None
                    else ""
                )
                result = {
                    "id": row.get("id"),
                    "figure_key": row.get("figure_key"),
                    "claim_id": row.get("claim_id"),
                    "claim": row["claim"],
                    "gold_graph": row.get("graph"),
                    "prediction_text": prediction_text,
                    "prediction_graph": prediction_graph,
                    "error": error,
                }
                results.append(result)
                if output is not None:
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
        finally:
            if output is not None:
                output.close()

        if args.output_json is not None:
            json_ok = sum(row["prediction_graph"] is not None for row in results)
            payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "input": str(args.input_jsonl),
                "extractor": f"spaCy/{args.model}",
                "processed": len(results),
                "total": len(rows),
                "json_ok": json_ok,
                "json_ok_rate": json_ok / len(results) if results else 0.0,
                "results": results,
            }
            args.output_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"Saved {len(rows)} spaCy validation records to {output_path}")
        return 0

    description = (
        args.file.read_text(encoding="utf-8").strip()
        if args.file is not None
        else args.description
    )
    graph = extractor.extract(description, debug=args.debug)
    print(json.dumps(graph, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
