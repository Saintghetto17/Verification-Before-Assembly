"""TensorBoard helpers: all runs write under <project_root>/tfevents/."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

TFEVENTS_DIRNAME = "tfevents"


def tfevents_root(project_root: Path | str) -> Path:
    root = Path(project_root) / TFEVENTS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_run_dir(project_root: Path | str, run_name: str) -> Path:
    """Create tfevents/<run_name>/<timestamp>/ for one training job."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = tfevents_root(project_root) / run_name / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_summary_writer(
    project_root: Path | str,
    run_name: str,
) -> Tuple[Optional["SummaryWriter"], Path]:
    """Return (SummaryWriter|None, log_dir). None if tensorboard/torch unavailable."""
    run_dir = make_run_dir(project_root, run_name)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover
        logger.warning("TensorBoard unavailable (%s); skipping TB logs → %s", exc, run_dir)
        return None, run_dir

    writer = SummaryWriter(log_dir=str(run_dir))
    logger.info("TensorBoard logging → %s", run_dir)
    return writer, run_dir
