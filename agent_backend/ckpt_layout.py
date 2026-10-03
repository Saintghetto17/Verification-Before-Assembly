"""Checkpoint layout: load from checkpoints/, save to train_ckpt/<name>/."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Files/dirs under checkpoints/ that agents load at inference
PROMOTE_NAMES = (
    "router.pth",
    "sem.pt",
    "arbiter.pkl",
    "scaler.pkl",
    "decision_threshold.json",
)


def train_ckpt_dir(project_root: Path, config_name: str) -> Path:
    return Path(project_root) / "train_ckpt" / config_name


def checkpoints_dir(project_root: Path) -> Path:
    return Path(project_root) / "checkpoints"


def promote_to_checkpoints(
    project_root: Path,
    src_dir: Path,
    *,
    names: Iterable[str] = PROMOTE_NAMES,
) -> list[str]:
    """Copy trained artifacts from train_ckpt into shared checkpoints/."""
    dst_root = checkpoints_dir(project_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in names:
        src = Path(src_dir) / name
        if not src.exists():
            continue
        dst = dst_root / name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        copied.append(name)
        logger.info("promoted %s → %s", src, dst)
    return copied
