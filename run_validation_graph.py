#!/usr/bin/env python3
"""Generate scene graphs for every claim in the graph validation split."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from agent_backend.graphs.generate import GraphGenerator
from agent_backend.graphs.schema import parse_graph_json


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "graph_creator_model_dataset" / "val.jsonl"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "smollm_graph"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "agent_v6" / "validation_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SmolLM graph generation over the full validation split."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Atomically update the output after this many examples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit; by default processes the full split.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_results(
    path: Path,
    *,
    input_path: Path,
    checkpoint: Path,
    total: int,
    results: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_ok = sum(result["prediction_graph"] is not None for result in results)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "checkpoint": str(checkpoint),
        "processed": len(results),
        "total": total,
        "json_ok": json_ok,
        "json_ok_rate": json_ok / len(results) if results else 0.0,
        "results": results,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    generator = GraphGenerator(
        args.checkpoint,
        base=args.base,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    results: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="graph validation", unit="claim"):
        error = None
        prediction_text = ""
        prediction_graph = None
        try:
            prediction_text = generator.generate_text(row["claim"])
            prediction_graph = parse_graph_json(prediction_text)
            if prediction_graph is not None and row.get("claim_id") is not None:
                prediction_graph["claim_id"] = int(row["claim_id"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        results.append(
            {
                "id": row.get("id"),
                "figure_key": row.get("figure_key"),
                "claim_id": row.get("claim_id"),
                "claim": row["claim"],
                "gold_graph": row.get("graph"),
                "prediction_text": prediction_text,
                "prediction_graph": prediction_graph,
                "error": error,
            }
        )
        if len(results) % max(args.save_every, 1) == 0:
            save_results(
                args.output,
                input_path=args.input,
                checkpoint=args.checkpoint,
                total=len(rows),
                results=results,
            )

    save_results(
        args.output,
        input_path=args.input,
        checkpoint=args.checkpoint,
        total=len(rows),
        results=results,
    )
    print(f"Saved {len(results)} validation generations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
