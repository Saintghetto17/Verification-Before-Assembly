#!/usr/bin/env python3
"""Download the public Hugging Face dataset into the local repository layout.

Dataset:
  https://huggingface.co/datasets/Saintghetto17/verification-before-assembly

Places files as:
  data/figures/   ← dataset `figures/`
  data_router/    ← dataset `router/`
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ID = "Saintghetto17/verification-before-assembly"
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="optional dataset revision / commit / tag",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from exc

    cache_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
    )
    cache = Path(cache_dir)

    figures_src = cache / "figures"
    router_src = cache / "router"
    if not figures_src.is_dir():
        raise SystemExit(f"figures/ missing in downloaded dataset: {cache}")

    figures_dst = ROOT / "data" / "figures"
    router_dst = ROOT / "data_router"
    figures_dst.parent.mkdir(parents=True, exist_ok=True)

    if figures_dst.exists():
        shutil.rmtree(figures_dst)
    shutil.copytree(figures_src, figures_dst)
    print(f"installed figures -> {figures_dst}")

    if router_src.is_dir():
        if router_dst.exists():
            # keep local README if present, replace contents
            shutil.rmtree(router_dst)
        shutil.copytree(router_src, router_dst)
        print(f"installed router  -> {router_dst}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
