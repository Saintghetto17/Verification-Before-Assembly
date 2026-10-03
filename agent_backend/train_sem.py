"""Leakage-resistant, class-balanced fine-tuning for the semantic judge."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModel, AutoProcessor

from .judges.match_head import (
    MatchHead,
    as_embed as _as_embed,
    resize_text_position_embeddings,
)

logger = logging.getLogger(__name__)


def _source_figure_group(row: Mapping[str, Any]) -> str:
    """Canonical source id that puts orig/fail variants in one split group."""
    raw = str(row.get("source_sample_id") or row.get("sample_id") or "")
    if not raw:
        raw = str(row.get("image_path") or "")
    parts = raw.replace("\\", "/").split("/")
    for index, part in enumerate(parts):
        for suffix in ("_l2_fail", "_orig"):
            if part.endswith(suffix):
                parts[index] = part[: -len(suffix)]
                break
    return "/".join(parts)


def _is_golden_row(row: Mapping[str, Any]) -> bool:
    split = str(row.get("split") or "").lower()
    path_parts = str(row.get("image_path") or "").lower().replace("\\", "/").split("/")
    return split == "golden" or "golden" in path_parts


def _grouped_train_val_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """Fixed stratified group split, selected to best match the requested ratio."""
    from sklearn.model_selection import StratifiedGroupKFold

    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    if any(_is_golden_row(row) for row in rows):
        raise ValueError("Golden rows cannot be used for semantic training or threshold tuning")

    y = np.asarray([1 if row.get("label") == "match" else 0 for row in rows])
    groups = np.asarray([_source_figure_group(row) for row in rows])
    if len(rows) < 4 or np.unique(y).size != 2:
        raise ValueError("Semantic training requires at least four rows and both classes")
    if any(not group for group in groups):
        raise ValueError("Every semantic row needs a sample_id, source_sample_id, or image_path")

    n_splits = min(
        max(2, round(1.0 / test_ratio)),
        len(np.unique(groups)),
        int(np.bincount(y).min()),
    )
    if n_splits < 2:
        raise ValueError("Not enough groups/class examples for grouped validation")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidates = []
    for train_idx, val_idx in splitter.split(np.zeros(len(y)), y, groups):
        if np.unique(y[train_idx]).size != 2 or np.unique(y[val_idx]).size != 2:
            continue
        ratio_error = abs(len(val_idx) / len(rows) - test_ratio)
        prevalence_error = abs(float(y[val_idx].mean()) - float(y.mean()))
        candidates.append(((ratio_error, prevalence_error), train_idx, val_idx))
    if not candidates:
        raise ValueError("Could not create a grouped validation split containing both classes")

    _, train_idx, val_idx = min(candidates, key=lambda item: item[0])
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    if train_groups & val_groups:
        raise RuntimeError("Source figure groups crossed the train/validation boundary")
    return [rows[i] for i in train_idx], [rows[i] for i in val_idx]


def _threshold_metrics(
    labels: Sequence[float],
    probabilities: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    """Metrics treating match and mismatch symmetrically."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("labels and probabilities must be same-length 1D arrays")
    if len(y) == 0:
        raise ValueError("Cannot score an empty validation set")
    pred = (p >= threshold).astype(np.int64)

    def class_metrics(class_id: int) -> Tuple[float, float, float]:
        tp = int(((pred == class_id) & (y == class_id)).sum())
        fp = int(((pred == class_id) & (y != class_id)).sum())
        fn = int(((pred != class_id) & (y == class_id)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    mismatch_precision, mismatch_recall, mismatch_f1 = class_metrics(0)
    match_precision, match_recall, match_f1 = class_metrics(1)
    return {
        "threshold": float(threshold),
        "accuracy": float((pred == y).mean()),
        "precision_match": float(match_precision),
        "recall_match": float(match_recall),
        "f1_match": float(match_f1),
        "precision_mismatch": float(mismatch_precision),
        "recall_mismatch": float(mismatch_recall),
        "f1_mismatch": float(mismatch_f1),
        "macro_f1": float((match_f1 + mismatch_f1) / 2.0),
        "min_f1": float(min(match_f1, mismatch_f1)),
    }


def _optimize_balanced_threshold(
    labels: Sequence[float], probabilities: Sequence[float]
) -> Tuple[float, Dict[str, float]]:
    """Maximize min class F1; use macro F1, accuracy, then proximity to 0.5 as ties."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("labels and probabilities must be non-empty, same-length 1D arrays")
    if np.unique(y).size != 2:
        raise ValueError("Threshold optimization requires both classes")
    if not np.isfinite(p).all():
        raise ValueError("Probabilities must all be finite")

    candidates = np.append(np.unique(p), np.nextafter(float(p.max()), np.inf))
    scored = [_threshold_metrics(y, p, float(threshold)) for threshold in candidates]
    best = max(
        scored,
        key=lambda metric: (
            metric["min_f1"],
            metric["macro_f1"],
            metric["accuracy"],
            -abs(metric["threshold"] - 0.5),
        ),
    )
    return best["threshold"], best


class SemMatchModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: MatchHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def encode(self, enc: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(**enc)
        if getattr(outputs, "image_embeds", None) is not None:
            img = outputs.image_embeds
            txt = outputs.text_embeds
        else:
            img = _as_embed(self.backbone.get_image_features(pixel_values=enc["pixel_values"]))
            txt = _as_embed(
                self.backbone.get_text_features(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                )
            )
        return img, txt

    def forward(self, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        img, txt = self.encode(enc)
        return self.head(img, txt)


class PairDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], processor, max_text_length: int):
        self.rows = list(rows)
        self.processor = processor
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        text = (r.get("premise") or r.get("text_block") or r.get("caption") or "").strip()
        if not text:
            text = "scientific figure"
        with Image.open(r["image_path"]) as im:
            img = im.convert("RGB")
        y = 1.0 if r.get("label") == "match" else 0.0
        return img, text, y, _source_figure_group(r)


class PairAwareBatchSampler(Sampler[List[int]]):
    """Keep source-figure match/mismatch pairs in the same shuffled batch."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], batch_size: int, seed: int):
        if batch_size < 2:
            raise ValueError("pair-aware batches require batch_size >= 2")
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(_source_figure_group(row), []).append(index)
        self.groups = list(grouped.values())
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        batch: List[int] = []
        for group_index in rng.permutation(len(self.groups)):
            indices = list(self.groups[int(group_index)])
            if batch and len(batch) + len(indices) > self.batch_size:
                yield batch
                batch = []
            if len(indices) > self.batch_size:
                for start in range(0, len(indices), self.batch_size):
                    yield indices[start : start + self.batch_size]
            else:
                batch.extend(indices)
        if batch:
            yield batch

    def __len__(self) -> int:
        return int(np.ceil(sum(map(len, self.groups)) / self.batch_size))


def _collate(batch, processor, max_text_length: int):
    """CPU-only collate — never .to(cuda) here (breaks DataLoader workers)."""
    images, texts, ys, groups = zip(*batch)
    enc = processor(
        text=list(texts),
        images=list(images),
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    y = torch.tensor(ys, dtype=torch.float32)
    return enc, y, list(groups)


def _batch_to_device(
    enc: Dict[str, torch.Tensor], y: torch.Tensor, device: torch.device
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    return enc, y.to(device, non_blocking=True)


@torch.inference_mode()
def _eval_split(
    model: SemMatchModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    probs_all: List[float] = []
    y_all: List[float] = []
    for enc, y, _groups in loader:
        enc, y = _batch_to_device(enc, y, device)
        logits = model(enc)
        probs = torch.sigmoid(logits)
        probs_all.extend(probs.detach().cpu().tolist())
        y_all.extend(y.detach().cpu().tolist())
    p = np.asarray(probs_all, dtype=np.float64)
    y = np.asarray(y_all, dtype=np.float64)
    threshold, balanced = _optimize_balanced_threshold(y, p)
    match_mask = y >= 0.5
    mism_mask = ~match_mask
    match_mean = float(p[match_mask].mean()) if match_mask.any() else float("nan")
    mism_mean = float(p[mism_mask].mean()) if mism_mask.any() else float("nan")
    match_gt = float((p[match_mask] > 0.5).mean()) if match_mask.any() else float("nan")
    mism_lt = float((p[mism_mask] < 0.5).mean()) if mism_mask.any() else float("nan")
    # margin: how far from 0.5 on average for correct side
    match_margin = float((p[match_mask] - 0.5).mean()) if match_mask.any() else float("nan")
    mism_margin = float((0.5 - p[mism_mask]).mean()) if mism_mask.any() else float("nan")
    return {
        # Keep the legacy key, but evaluate it at the tuned validation threshold.
        "acc": balanced["accuracy"],
        **balanced,
        "match_mean": match_mean,
        "mismatch_mean": mism_mean,
        "match_gt_0.5": match_gt,
        "mismatch_lt_0.5": mism_lt,
        "match_margin": match_margin,
        "mismatch_margin": mism_margin,
        "n": float(len(y)),
    }


class _FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, *, pos_weight: torch.Tensor, gamma: float):
        super().__init__()
        if gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probability = torch.sigmoid(logits)
        p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
        return (((1.0 - p_t) ** self.gamma) * base).mean()


def _paired_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    groups: Sequence[str],
    *,
    margin: float,
) -> torch.Tensor:
    """Encourage each matched figure to outrank its corrupted paired version."""
    losses = []
    for group in dict.fromkeys(groups):
        indices = [index for index, value in enumerate(groups) if value == group]
        positive = [index for index in indices if float(targets[index]) >= 0.5]
        negative = [index for index in indices if float(targets[index]) < 0.5]
        if positive and negative:
            pos_logit = logits[positive].mean()
            neg_logit = logits[negative].mean()
            losses.append(torch.relu(logits.new_tensor(margin) - pos_logit + neg_logit))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def train_semantic_judge(
    rows: List[Dict[str, Any]],
    *,
    pretrained: str | Path,
    out_ckpt: Path,
    project_root: Path,
    epochs: int = 3,
    batch_size: int = 16,
    lr_head: float = 1e-4,
    lr_backbone: float = 2e-6,
    weight_decay: float = 0.01,
    max_text_length: int = 64,
    test_ratio: float = 0.15,
    seed: int = 42,
    device: str = "cuda",
    freeze_backbone_epochs: int = 1,
    tb_run_name: str = "agent_sem",
    loss_type: str = "weighted_bce",
    positive_class_weight: Optional[float] = None,
    focal_gamma: float = 2.0,
    pair_aware_batches: bool = False,
    pairwise_loss_weight: float = 0.0,
    pairwise_margin: float = 0.2,
    refit_all: bool = False,
) -> Dict[str, Any]:
    from .tb_logging import make_summary_writer

    device_t = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    train_rows_raw, val_rows_raw = _grouped_train_val_split(
        rows, test_ratio=test_ratio, seed=seed
    )
    train_rows = list(rows) if refit_all else list(train_rows_raw)
    val_rows = list(val_rows_raw)

    processor = AutoProcessor.from_pretrained(str(pretrained))
    backbone = AutoModel.from_pretrained(str(pretrained))
    resize_text_position_embeddings(backbone, max_text_length)
    # probe embed dim
    with torch.no_grad():
        dummy = processor(
            text=["test"],
            images=[Image.new("RGB", (224, 224), color=(128, 128, 128))],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_length,
        )
        out = backbone(**dummy)
        if getattr(out, "image_embeds", None) is not None:
            dim = int(out.image_embeds.shape[-1])
        else:
            dim = int(_as_embed(backbone.get_image_features(pixel_values=dummy["pixel_values"])).shape[-1])

    head = MatchHead(dim)
    model = SemMatchModel(backbone, head).to(device_t)
    logger.info(
        "sem model ready device=%s embed_dim=%d n_train=%d n_val=%d batch_size=%d",
        device_t,
        dim,
        len(train_rows),
        len(val_rows),
        batch_size,
    )

    train_ds = PairDataset(train_rows, processor, max_text_length)
    val_ds = PairDataset(val_rows, processor, max_text_length)

    # num_workers=0: avoid CUDA/fork issues on cluster; collate stays on CPU
    def make_loader(ds, shuffle: bool):
        if shuffle and pair_aware_batches:
            return DataLoader(
                ds,
                batch_sampler=PairAwareBatchSampler(ds.rows, batch_size, seed),
                num_workers=0,
                collate_fn=lambda b: _collate(b, processor, max_text_length),
            )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda b: _collate(b, processor, max_text_length),
        )

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)

    writer, tb_dir = make_summary_writer(project_root, tb_run_name)
    n_match = sum(row.get("label") == "match" for row in train_rows)
    n_mismatch = len(train_rows) - n_match
    if not n_match or not n_mismatch:
        raise ValueError("Grouped training partition must contain both classes")
    pos_weight_value = (
        float(positive_class_weight)
        if positive_class_weight is not None
        else float(n_mismatch / n_match)
    )
    if not np.isfinite(pos_weight_value) or pos_weight_value <= 0:
        raise ValueError("positive_class_weight must be finite and greater than zero")
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device_t)
    normalized_loss_type = loss_type.lower().strip()
    if normalized_loss_type == "weighted_bce":
        crit: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif normalized_loss_type == "focal":
        crit = _FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=focal_gamma).to(device_t)
    else:
        raise ValueError("loss_type must be 'weighted_bce' or 'focal'")

    best_selection = (-1.0, -1.0)
    best_metrics: Dict[str, float] = {}
    best_path = Path(out_ckpt)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []

    global_step = 0
    for epoch in range(epochs):
        freeze = epoch < freeze_backbone_epochs
        for p in model.backbone.parameters():
            p.requires_grad = not freeze
        params = [{"params": model.head.parameters(), "lr": lr_head}]
        if not freeze:
            params.append({"params": model.backbone.parameters(), "lr": lr_backbone})
        opt = torch.optim.AdamW(params, weight_decay=weight_decay)

        model.train()
        total_loss = 0.0
        n = 0
        logger.info(
            "sem epoch %d/%d start (freeze_backbone=%s)",
            epoch + 1,
            epochs,
            freeze,
        )
        for enc, y, groups in train_loader:
            enc, y = _batch_to_device(enc, y, device_t)
            opt.zero_grad(set_to_none=True)
            logits = model(enc)
            loss = crit(logits, y)
            if pairwise_loss_weight > 0:
                loss = loss + pairwise_loss_weight * _paired_ranking_loss(
                    logits, y, groups, margin=pairwise_margin
                )
            loss.backward()
            opt.step()
            bs = int(y.shape[0])
            total_loss += float(loss.item()) * bs
            n += bs
            if global_step == 0:
                logger.info("sem first step ok loss=%.4f", float(loss.item()))
            if writer is not None:
                writer.add_scalar("train/loss_step", float(loss.item()), global_step)
            global_step += 1

        train_loss = total_loss / max(n, 1)
        metrics = _eval_split(model, val_loader, device_t)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch + 1
        metrics["freeze_backbone"] = freeze
        history.append(metrics)
        logger.info(
            "sem epoch %d/%d loss=%.4f threshold=%.4f val_acc=%.4f "
            "f1(match/mismatch)=%.4f/%.4f macro=%.4f min=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            metrics["threshold"],
            metrics["acc"],
            metrics["f1_match"],
            metrics["f1_mismatch"],
            metrics["macro_f1"],
            metrics["min_f1"],
        )
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/acc", metrics["acc"], epoch)
            writer.add_scalar("val/threshold", metrics["threshold"], epoch)
            writer.add_scalar("val/f1_match", metrics["f1_match"], epoch)
            writer.add_scalar("val/f1_mismatch", metrics["f1_mismatch"], epoch)
            writer.add_scalar("val/macro_f1", metrics["macro_f1"], epoch)
            writer.add_scalar("val/min_f1", metrics["min_f1"], epoch)
            writer.add_scalar("val/match_mean", metrics["match_mean"], epoch)
            writer.add_scalar("val/mismatch_mean", metrics["mismatch_mean"], epoch)
            writer.add_scalar("val/match_gt_0.5", metrics["match_gt_0.5"], epoch)
            writer.add_scalar("val/mismatch_lt_0.5", metrics["mismatch_lt_0.5"], epoch)
            writer.add_scalar("val/match_margin", metrics["match_margin"], epoch)
            writer.add_scalar("val/mismatch_margin", metrics["mismatch_margin"], epoch)
            writer.flush()

        selection = (metrics["min_f1"], metrics["macro_f1"])
        should_save = epoch == epochs - 1 if refit_all else selection >= best_selection
        if should_save:
            best_selection = selection
            best_metrics = dict(metrics)
            torch.save(
                {
                    "kind": "sem_match_v1",
                    "pretrained": str(pretrained),
                    "embed_dim": dim,
                    "backbone_state_dict": model.backbone.state_dict(),
                    "head_state_dict": model.head.state_dict(),
                    "max_text_length": max_text_length,
                    "val_metrics": metrics,
                    "decision_threshold": metrics["threshold"],
                    "threshold_source": (
                        "training_diagnostic_not_for_selection"
                        if refit_all
                        else "non_golden_grouped_validation_max_min_class_f1"
                    ),
                    "selection_metric": (
                        "frozen_final_epoch" if refit_all
                        else "min(f1_match,f1_mismatch)"
                    ),
                    "refit_all_non_golden": bool(refit_all),
                },
                best_path,
            )

    if writer is not None:
        writer.add_scalar("val/best_min_f1", best_selection[0], epochs - 1)
        writer.close()

    return {
        "best_val_acc": float(best_metrics.get("acc", 0.0)),
        "best_val_min_f1": float(best_selection[0]),
        "best_val_macro_f1": float(best_metrics.get("macro_f1", 0.0)),
        "decision_threshold": float(best_metrics.get("threshold", 0.5)),
        "selection_metric": (
            "frozen_final_epoch" if refit_all else "min(f1_match,f1_mismatch)"
        ),
        "threshold_source": (
            "training_diagnostic_not_for_selection"
            if refit_all
            else "non_golden_grouped_validation"
        ),
        "loss_type": normalized_loss_type,
        "positive_class_weight": pos_weight_value,
        "focal_gamma": float(focal_gamma) if normalized_loss_type == "focal" else None,
        "pair_aware_batches": bool(pair_aware_batches),
        "pairwise_loss_weight": float(pairwise_loss_weight),
        "pairwise_margin": float(pairwise_margin),
        "refit_all_non_golden": bool(refit_all),
        "ckpt": str(best_path),
        "tfevents": str(tb_dir),
        "history": history,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_train_groups": len({_source_figure_group(row) for row in train_rows}),
        "n_val_groups": len({_source_figure_group(row) for row in val_rows}),
    }
