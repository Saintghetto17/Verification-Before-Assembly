#!/usr/bin/env python3
"""Local PyTorch launcher for verifier training and evaluation.

Runs paper experiment modes on the current machine (single- or multi-GPU via
``torchrun`` when requested). This is a research entrypoint — not a cluster
job client.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODES = (
    "train_router",
    "train_sem",
    "prepare_features",
    "train_arbiter",
    "train_graph",
    "validate_graph",
    "eval",
    "ablate",
    "all",
)


def _python() -> str:
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file() and os.access(venv_py, os.X_OK):
        return str(venv_py)
    return sys.executable


def launch_agent(
    mode: str = "train_sem",
    *,
    config_name: str = "agent_v5",
    exp_name: str = "agent_v5",
    n_gpus: int = 1,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Run one verifier mode under PyTorch locally."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if n_gpus < 1:
        raise ValueError("n_gpus must be >= 1")

    config_name = config_name.removesuffix(".yaml")
    exp_name = Path(exp_name).name
    config = ROOT / "configs" / f"{config_name}.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"missing config: {config}")

    module_args = [
        "-m",
        "agent_backend",
        "--config",
        str(config),
        "--mode",
        mode,
        "--exp-name",
        exp_name,
        "--project-root",
        str(ROOT),
    ]
    if limit is not None:
        module_args.extend(["--limit", str(limit)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("PYTHONUNBUFFERED", "1")

    if n_gpus > 1:
        # Multi-GPU host: torchrun launches one process per GPU.
        cmd = [
            _python(),
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={n_gpus}",
            "--standalone",
            *module_args,
        ]
    else:
        cmd = [_python(), *module_args]

    print(f"[launch] mode={mode} config={config_name} exp={exp_name} n_gpus={n_gpus}")
    print("[launch]", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch Verification Before Assembly training/eval with PyTorch."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="train_sem",
        choices=MODES,
        help="experiment mode (default: train_sem)",
    )
    parser.add_argument("--config-name", default=None, help="YAML stem under configs/")
    parser.add_argument("--exp-name", default=None, help="experiment / output name")
    parser.add_argument("--n-gpus", type=int, default=1, help="GPUs for torchrun (>=1)")
    parser.add_argument("--limit", type=int, default=None, help="optional sample cap")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved command and exit",
    )
    args = parser.parse_args(argv)

    extra: dict = {}
    if args.mode in {"train_graph", "validate_graph"}:
        extra = {"config_name": "agent_v6", "exp_name": "agent_v6"}
    config_name = args.config_name or extra.get("config_name", "agent_v5")
    exp_name = args.exp_name or extra.get("exp_name", config_name)

    return launch_agent(
        mode=args.mode,
        config_name=config_name,
        exp_name=exp_name,
        n_gpus=args.n_gpus,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
