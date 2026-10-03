#!/usr/bin/env python3
"""Build claim→graph JSONL *before* SFT. Training must only read this folder.

Usage:
  python scripts/build_graph_sft_dataset.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_backend.config import AgentConfig
from agent_backend.graphs.dataset import build_graph_sft_dataset


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = AgentConfig.from_yaml(ROOT / "configs" / "agent_v6.yaml")
    figures_dir = ROOT / cfg.paths.figures_dir
    out_dir = ROOT / cfg.graph.dataset_dir
    info = build_graph_sft_dataset(
        figures_dir,
        out_dir,
        val_ratio=cfg.graph.val_ratio,
        seed=cfg.graph.seed,
    )
    print(json.dumps({k: v for k, v in info.items() if not k.endswith("_keys")}, indent=2))
    print(f"wrote {out_dir / 'train.jsonl'}")
    print(f"wrote {out_dir / 'val.jsonl'}")
    print(f"wrote {out_dir / 'split_meta.json'}")


if __name__ == "__main__":
    main()
