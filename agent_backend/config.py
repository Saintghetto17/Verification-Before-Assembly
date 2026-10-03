"""YAML config for Verification Agent."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            key, default = match.group(1), match.group(2)
            env_val = os.environ.get(key)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            raise KeyError(f"env var {key} is not set and has no default")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    return value


class PathsConfig(BaseModel):
    figures_dir: str = "data/figures"
    router_data_dir: str = "data_router"
    # Shared inference weights (READ)
    checkpoints_dir: str = "checkpoints"
    router_ckpt: str = "checkpoints/router.pth"
    sem_ckpt: str = "checkpoints/sem.pt"
    arbiter_ckpt: str = "checkpoints/arbiter.pkl"
    scaler_ckpt: str = "checkpoints/scaler.pkl"
    # Per-exp: weights only (WRITE → then promote to checkpoints/)
    train_ckpt_dir: str = "train_ckpt/agent_v3"
    # Per-exp: logs, features.csv, metrics, predictions (WRITE)
    output_dir: str = "outputs/agent_v3"
    features_csv: str = "outputs/agent_v3/features.csv"


class SiglipConfig(BaseModel):
    pretrained: str = "google/siglip-base-patch16-224"
    max_text_length: int = 64
    batch_size: int = 8


class SemTrainConfig(BaseModel):
    epochs: int = 4
    batch_size: int = 16
    lr_head: float = 1.0e-4
    lr_backbone: float = 2.0e-6
    weight_decay: float = 0.01
    test_ratio: float = 0.15
    freeze_backbone_epochs: int = 1
    seed: int = 42
    loss_type: str = "weighted_bce"
    positive_class_weight: Optional[float] = None
    focal_gamma: float = 2.0
    pair_aware_batches: bool = False
    pairwise_loss_weight: float = 0.0
    pairwise_margin: float = 0.2
    refit_all: bool = False


class RouterConfig(BaseModel):
    classes: List[str] = Field(default_factory=lambda: ["byt", "video", "graph", "scheme"])
    image_size: int = 224
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_per_class_train: int = 250
    num_per_class_val: int = 50
    num_video_train: int = 400
    num_video_val: int = 80
    seed: int = 42


class ArbiterConfig(BaseModel):
    test_ratio: float = 0.2
    seed: int = 42
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    decision_threshold: Optional[float] = None
    max_clean_per_corrupted: Optional[float] = 3.0
    feature_log_every_pct: float = 1.0
    feature_log_every_n: int = 50
    # Feature-gen throughput (H100-friendly defaults can override in yaml)
    feature_batch_size: int = 32
    ocr_batch_size: int = 64
    ocr_use_gpu: Optional[bool] = None  # None → auto (cuda if available)


class GraphTrainConfig(BaseModel):
    """Claim → scene-graph SFT (SmolLM2). Detector is inference-only for now."""

    smol_pretrained: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    detector_pretrained: str = "iSEE-Laboratory/llmdet_tiny"
    dataset_dir: str = "graph_creator_model_dataset"
    val_ratio: float = 0.15
    seed: int = 42
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2.0e-5
    weight_decay: float = 0.01
    max_seq_len: int = 1536
    warmup_ratio: float = 0.03
    max_new_tokens: int = 512
    eval_generate_n: int = 64
    lora_r: int = 16


class AgentConfig(BaseModel):
    name: str = "agent_v3"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    siglip: SiglipConfig = Field(default_factory=SiglipConfig)
    sem_train: SemTrainConfig = Field(default_factory=SemTrainConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    arbiter: ArbiterConfig = Field(default_factory=ArbiterConfig)
    graph: GraphTrainConfig = Field(default_factory=GraphTrainConfig)
    device: str = "cuda"
    limit: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**_interpolate(raw))

    def apply_exp_name(self, exp_name: str) -> None:
        """Set outputs/<exp> + train_ckpt/<exp> from a single experiment name."""
        name = exp_name.strip().strip("/")
        if not name:
            raise ValueError("exp_name must be non-empty")
        self.name = name
        self.paths.train_ckpt_dir = f"train_ckpt/{name}"
        self.paths.output_dir = f"outputs/{name}"
        self.paths.features_csv = f"outputs/{name}/features.csv"


def resolve(project_root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root / p


def pick_device(requested: str) -> str:
    if requested == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return requested
