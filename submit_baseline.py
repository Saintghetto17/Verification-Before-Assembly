#!/usr/bin/env python3
"""Local PyTorch launcher for standalone baseline evaluations.

Runs Qwen-VL / VQAScore baseline scripts on the current machine. This is a
research entrypoint — not a cluster job client.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

QWEN_MODELS = {
    "qwen25_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3_4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen25_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen25_7b_dev": "Qwen/Qwen2.5-VL-7B-Instruct",
}


def _python() -> str:
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file() and os.access(venv_py, os.X_OK):
        return str(venv_py)
    return sys.executable


def launch_baseline(kind: str, *, dry_run: bool = False) -> int:
    """Run one baseline evaluation under the local Python/PyTorch environment."""
    output_dir = ROOT / "outputs" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    if kind in QWEN_MODELS:
        model = QWEN_MODELS[kind]
        cmd = [
            _python(),
            str(ROOT / "baseline_eval.py"),
            "judge",
            "--model",
            model,
            "--output",
            str(output_dir / f"{kind}_golden.jsonl"),
            "--report",
            str(output_dir / f"{kind}_report.json"),
        ]
        if kind.endswith("_dev"):
            cmd.extend(
                [
                    "--input",
                    str(ROOT / "outputs" / "agent_v5_balanced_dev" / "non_golden.jsonl"),
                ]
            )
    elif kind == "vqascore":
        cmd = [
            _python(),
            str(ROOT / "baseline_eval.py"),
            "vqascore",
            "--model",
            "clip-flant5-xxl",
            "--validation-scores",
            str(output_dir / "vqa_validation.jsonl"),
            "--golden-scores",
            str(output_dir / "vqa_golden.jsonl"),
            "--report",
            str(output_dir / "vqa_report.json"),
        ]
    else:
        raise ValueError(
            "unknown baseline; choose from "
            + ", ".join([*QWEN_MODELS, "vqascore"])
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"[launch] baseline={kind}")
    print("[launch]", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch baseline evaluations with local PyTorch / Python."
    )
    parser.add_argument(
        "kind",
        choices=[*QWEN_MODELS.keys(), "vqascore"],
        help="baseline identifier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved command and exit",
    )
    args = parser.parse_args(argv)
    return launch_baseline(args.kind, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
