"""Leakage-safe ablations for the verification pipeline.

All model selection and threshold selection use one fixed, stratified split of
the non-golden data.  Golden data is used exactly once, for final reporting.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .agent import VerificationAgent
from .arbiter import classwise_metrics, find_best_f1_threshold
from .config import AgentConfig
from .features import FEATURE_NAMES
from .text_utils import row_texts

logger = logging.getLogger(__name__)

JUDGE_FEATURES = {
    "judge_a": "score_sem",
    "judge_b": "score_struct",
    "judge_c": "score_obj",
}
ROUTER_FEATURES = [
    "p_byt",
    "p_video",
    "p_graph",
    "p_scheme",
    "router_conf",
]
ROUTER_DERIVED_FEATURES = [
    *ROUTER_FEATURES,
    "sem_w",
    "struct_w",
    "obj_w",
]


def variant_feature_names() -> Dict[str, List[str]]:
    """Return explicit feature sets for every XGBoost ablation."""
    all_features = list(FEATURE_NAMES)

    def without(*names: str) -> List[str]:
        excluded = set(names)
        return [name for name in all_features if name not in excluded]

    return {
        "full": all_features,
        "no_router": without(*ROUTER_DERIVED_FEATURES),
        "no_judge_a": without("score_sem", "sem_w"),
        "no_judge_b": without("score_struct", "struct_w"),
        "no_judge_c": without("score_obj", "obj_w"),
        "router_only": list(ROUTER_FEATURES),
    }


def _feature_matrix(
    agent: VerificationAgent,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Compute each row's complete feature vector exactly once."""
    matrices: List[np.ndarray] = []
    labels: List[int] = []
    metadata: List[Dict[str, Any]] = []
    for start in range(0, len(rows), max(batch_size, 1)):
        chunk = rows[start : start + max(batch_size, 1)]
        premises: List[str] = []
        captions: List[str] = []
        for row in chunk:
            premise, caption = row_texts(dict(row))
            premises.append(premise)
            captions.append(caption)
        features = agent.compute_features_batch(
            [row["image_path"] for row in chunk],
            premises,
            captions=captions,
        )
        matrices.extend(
            np.asarray([feature[name] for name in FEATURE_NAMES], dtype=np.float64)
            for feature in features
        )
        for row in chunk:
            label = row.get("label")
            if label not in {"match", "mismatch"}:
                raise ValueError(f"Unlabeled ablation row: {row.get('sample_id')}")
            labels.append(int(label == "match"))
            metadata.append(
                {
                    "sample_id": row.get("sample_id"),
                    "image_path": str(row.get("image_path")),
                    "label": label,
                    "split": row.get("split"),
                }
            )
        logger.info("ablation features: %d/%d", min(start + len(chunk), len(rows)), len(rows))
    return np.vstack(matrices), np.asarray(labels, dtype=np.int64), metadata


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    symmetric = classwise_metrics(y_true, y_pred)
    return {
        **symmetric,
        "f1_match": symmetric["by_class"]["match"]["f1"],
        "f1_mismatch": symmetric["by_class"]["mismatch"]["f1"],
        "roc_auc": (
            float(roc_auc_score(y_true, y_score))
            if np.unique(y_true).size == 2
            else None
        ),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _stratified_bootstrap_indices(
    y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    parts = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        parts.append(rng.choice(indices, size=len(indices), replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, List[float] | None]:
    """Stratified percentile intervals for all five scalar metrics."""
    rng = np.random.default_rng(seed)
    values: Dict[str, List[float]] = {
        name: []
        for name in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f1_match",
            "f1_mismatch",
            "macro_f1",
            "min_class_f1",
            "roc_auc",
        )
    }
    for _ in range(n_bootstrap):
        idx = _stratified_bootstrap_indices(y_true, rng)
        sample = _metrics(y_true[idx], y_pred[idx], y_score[idx])
        for name in values:
            value = sample[name]
            if value is not None:
                values[name].append(float(value))
    return {
        name: (
            [float(x) for x in np.percentile(samples, [2.5, 97.5])]
            if samples
            else None
        )
        for name, samples in values.items()
    }


def paired_bootstrap_deltas(
    y_true: np.ndarray,
    full_pred: np.ndarray,
    full_score: np.ndarray,
    other_pred: np.ndarray,
    other_score: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Dict[str, float | List[float] | None]]:
    """Paired intervals for full minus comparator on identical golden rows."""
    rng = np.random.default_rng(seed)
    names = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_match",
        "f1_mismatch",
        "macro_f1",
        "min_class_f1",
        "roc_auc",
    )
    samples: Dict[str, List[float]] = {name: [] for name in names}
    for _ in range(n_bootstrap):
        idx = _stratified_bootstrap_indices(y_true, rng)
        full = _metrics(y_true[idx], full_pred[idx], full_score[idx])
        other = _metrics(y_true[idx], other_pred[idx], other_score[idx])
        for name in names:
            if full[name] is not None and other[name] is not None:
                samples[name].append(float(full[name]) - float(other[name]))
    point_full = _metrics(y_true, full_pred, full_score)
    point_other = _metrics(y_true, other_pred, other_score)
    return {
        name: {
            "estimate": (
                float(point_full[name]) - float(point_other[name])
                if point_full[name] is not None and point_other[name] is not None
                else None
            ),
            "ci95": (
                [float(x) for x in np.percentile(samples[name], [2.5, 97.5])]
                if samples[name]
                else None
            ),
        }
        for name in names
    }


def _candidate_params(cfg: AgentConfig) -> List[Dict[str, Any]]:
    """Small deterministic tuning grid anchored at configured parameters."""
    raw = [
        (cfg.arbiter.n_estimators, cfg.arbiter.max_depth, cfg.arbiter.learning_rate),
        (
            max(25, cfg.arbiter.n_estimators // 2),
            max(2, cfg.arbiter.max_depth - 2),
            cfg.arbiter.learning_rate,
        ),
        (
            cfg.arbiter.n_estimators,
            max(2, cfg.arbiter.max_depth - 2),
            max(0.01, cfg.arbiter.learning_rate / 2),
        ),
    ]
    return [
        {"n_estimators": int(n), "max_depth": int(d), "learning_rate": float(lr)}
        for n, d, lr in dict.fromkeys(raw)
    ]


def _fit_xgb_variant(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    golden_X: np.ndarray,
    *,
    params_grid: Sequence[Mapping[str, Any]],
    seed: int,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    scaler = StandardScaler()
    train_X = scaler.fit_transform(X[train_idx])
    val_X = scaler.transform(X[val_idx])
    golden_scaled = scaler.transform(golden_X)
    candidates: List[Tuple[Tuple[float, float], Any, Dict[str, Any], np.ndarray]] = []
    for params in params_grid:
        model = XGBClassifier(
            **dict(params),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=4,
        )
        model.fit(train_X, y[train_idx], verbose=False)
        val_score = model.predict_proba(val_X)[:, 1]
        threshold, best_f1, threshold_metrics = find_best_f1_threshold(
            y[val_idx], val_score
        )
        auc = float(roc_auc_score(y[val_idx], val_score))
        details = {
            "params": dict(params),
            "validation_roc_auc": auc,
            "validation_f1": float(best_f1),
            "validation_threshold": float(threshold),
            "validation_metrics_at_threshold": threshold_metrics,
        }
        candidates.append(((auc, best_f1), model, details, val_score))
    _, model, selected, _ = max(
        candidates,
        key=lambda item: (
            item[0][0],
            item[0][1],
            -int(item[2]["params"]["max_depth"]),
        ),
    )
    threshold = float(selected["validation_threshold"])
    golden_score = model.predict_proba(golden_scaled)[:, 1]
    tuning = {
        "selection_metric": "validation_roc_auc_then_f1",
        "selected": selected,
        "candidates": [item[2] for item in candidates],
    }
    return golden_score, threshold, tuning


def _split_fingerprint(indices: np.ndarray, metadata: Sequence[Mapping[str, Any]]) -> str:
    values = [str(metadata[i].get("sample_id")) for i in sorted(indices.tolist())]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _pair_group(meta: Mapping[str, Any]) -> str:
    """Keep orig/fail versions of one figure on the same split side."""
    sample = str(meta.get("sample_id") or "")
    parts = sample.split("/")
    if len(parts) < 3:
        return sample
    family = parts[0]
    for suffix in ("_l2_fail", "_orig"):
        if family.endswith(suffix):
            family = family[: -len(suffix)]
            break
    return "/".join([family, *parts[1:]])


def _sample_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(row.get("sample_id"))
        for row in rows
        if row.get("sample_id") is not None
    }


def _image_paths(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(Path(str(row["image_path"])).resolve())
        for row in rows
        if row.get("image_path") is not None
    }


def run_ablations(
    cfg: AgentConfig,
    project_root: Path,
    train_rows: Sequence[Mapping[str, Any]],
    golden_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    *,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Run all ablations without fitting or selecting anything on golden."""
    if not train_rows or not golden_rows:
        raise ValueError("Both non-golden and golden rows are required")
    overlapping_ids = _sample_ids(train_rows) & _sample_ids(golden_rows)
    golden_source_ids = {
        str(row.get("source_sample_id"))
        for row in golden_rows
        if row.get("source_sample_id") is not None
    }
    overlapping_source_ids = _sample_ids(train_rows) & golden_source_ids
    overlapping_paths = _image_paths(train_rows) & _image_paths(golden_rows)
    if overlapping_ids or overlapping_source_ids or overlapping_paths:
        raise ValueError(
            "Non-golden/golden source overlap detected "
            f"(sample_ids={len(overlapping_ids)}, "
            f"golden_source_ids={len(overlapping_source_ids)}, "
            f"image_paths={len(overlapping_paths)})"
        )
    agent = VerificationAgent(cfg, project_root, load_arbiter=False)
    batch_size = max(int(cfg.arbiter.feature_batch_size), 1)
    train_X, train_y, train_meta = _feature_matrix(
        agent, train_rows, batch_size=batch_size
    )
    golden_X, golden_y, golden_meta = _feature_matrix(
        agent, golden_rows, batch_size=batch_size
    )

    all_indices = np.arange(len(train_y))
    groups = np.asarray([_pair_group(meta) for meta in train_meta])
    requested_splits = max(2, round(1.0 / float(cfg.arbiter.test_ratio)))
    splitter = StratifiedGroupKFold(
        n_splits=requested_splits,
        shuffle=True,
        random_state=cfg.arbiter.seed,
    )
    candidate_splits = list(splitter.split(all_indices, train_y, groups))
    train_idx, val_idx = min(
        candidate_splits,
        key=lambda split: abs(len(split[1]) / len(all_indices) - cfg.arbiter.test_ratio),
    )
    if np.unique(train_y[train_idx]).size != 2 or np.unique(train_y[val_idx]).size != 2:
        raise ValueError("Stratified non-golden split must contain both classes")
    fit_ids = {str(train_meta[i].get("sample_id")) for i in train_idx}
    validation_ids = {str(train_meta[i].get("sample_id")) for i in val_idx}
    duplicate_ids = fit_ids & validation_ids
    if duplicate_ids:
        raise ValueError(
            "Duplicate sample IDs cross the fit/validation boundary: "
            f"{len(duplicate_ids)}"
        )
    fit_groups = set(groups[train_idx])
    validation_groups = set(groups[val_idx])
    duplicate_groups = fit_groups & validation_groups
    if duplicate_groups:
        raise ValueError(
            "Paired orig/fail figure groups cross the fit/validation boundary: "
            f"{len(duplicate_groups)}"
        )

    feature_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    variants = variant_feature_names()
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    results: Dict[str, Dict[str, Any]] = {}
    params_grid = _candidate_params(cfg)

    for offset, (name, features) in enumerate(variants.items()):
        columns = [feature_index[feature] for feature in features]
        score, threshold, tuning = _fit_xgb_variant(
            train_X[:, columns],
            train_y,
            train_idx,
            val_idx,
            golden_X[:, columns],
            params_grid=params_grid,
            seed=cfg.arbiter.seed,
        )
        pred = (score >= threshold).astype(np.int64)
        predictions[name] = (pred, score)
        metric = _metrics(golden_y, pred, score)
        metric["bootstrap_ci95"] = bootstrap_confidence_intervals(
            golden_y,
            pred,
            score,
            n_bootstrap=n_bootstrap,
            seed=cfg.arbiter.seed + offset,
        )
        results[name] = {
            "kind": "xgboost",
            "feature_names": features,
            "decision_threshold": threshold,
            "threshold_source": "fixed_non_golden_validation_max_f1",
            "tuning": tuning,
            "golden_metrics": metric,
        }

    judge_thresholds: Dict[str, float] = {}
    judge_val_predictions: List[np.ndarray] = []
    judge_golden_predictions: List[np.ndarray] = []
    for offset, (name, feature) in enumerate(JUDGE_FEATURES.items(), start=len(results)):
        column = feature_index[feature]
        threshold, _, val_metrics = find_best_f1_threshold(
            train_y[val_idx], train_X[val_idx, column]
        )
        judge_thresholds[name] = threshold
        val_pred = (train_X[val_idx, column] >= threshold).astype(np.int64)
        score = golden_X[:, column]
        pred = (score >= threshold).astype(np.int64)
        judge_val_predictions.append(val_pred)
        judge_golden_predictions.append(pred)
        predictions[name] = (pred, score)
        metric = _metrics(golden_y, pred, score)
        metric["bootstrap_ci95"] = bootstrap_confidence_intervals(
            golden_y,
            pred,
            score,
            n_bootstrap=n_bootstrap,
            seed=cfg.arbiter.seed + offset,
        )
        results[name] = {
            "kind": "direct_judge",
            "feature_names": [feature],
            "decision_threshold": threshold,
            "threshold_source": "fixed_non_golden_validation_max_f1",
            "validation_metrics_at_threshold": val_metrics,
            "golden_metrics": metric,
        }

    majority_score = np.mean(np.vstack(judge_golden_predictions), axis=0)
    majority_pred = (majority_score >= (2.0 / 3.0)).astype(np.int64)
    predictions["majority_vote"] = (majority_pred, majority_score)
    majority_metric = _metrics(golden_y, majority_pred, majority_score)
    majority_metric["bootstrap_ci95"] = bootstrap_confidence_intervals(
        golden_y,
        majority_pred,
        majority_score,
        n_bootstrap=n_bootstrap,
        seed=cfg.arbiter.seed + len(results),
    )
    results["majority_vote"] = {
        "kind": "majority_vote",
        "feature_names": list(JUDGE_FEATURES.values()),
        "judge_thresholds": judge_thresholds,
        "vote_rule": "at_least_2_of_3_predict_match",
        "validation_metrics": _metrics(
            train_y[val_idx],
            (np.mean(np.vstack(judge_val_predictions), axis=0) >= (2.0 / 3.0)).astype(int),
            np.mean(np.vstack(judge_val_predictions), axis=0),
        ),
        "golden_metrics": majority_metric,
    }

    full_pred, full_score = predictions["full"]
    paired = {
        name: paired_bootstrap_deltas(
            golden_y,
            full_pred,
            full_score,
            pred,
            score,
            n_bootstrap=n_bootstrap,
            seed=cfg.arbiter.seed + 1000 + offset,
        )
        for offset, (name, (pred, score)) in enumerate(predictions.items())
        if name != "full"
    }

    prediction_rows: List[Dict[str, Any]] = []
    for i, meta in enumerate(golden_meta):
        prediction_rows.append(
            {
                **meta,
                "label_id": int(golden_y[i]),
                "variants": {
                    name: {
                        "pred_label": "match" if int(pred[i]) else "mismatch",
                        "pred_id": int(pred[i]),
                        "score_match": float(score[i]),
                    }
                    for name, (pred, score) in predictions.items()
                },
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_json = out_dir / "ablation_predictions.json"
    prediction_jsonl = out_dir / "ablation_predictions.jsonl"
    prediction_json.write_text(
        json.dumps(prediction_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with prediction_jsonl.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report: Dict[str, Any] = {
        "protocol": {
            "train_source": "load_train_figures (golden excluded)",
            "evaluation_source": "load_golden",
            "golden_usage": "final evaluation only; never training, tuning, or threshold selection",
            "feature_computation": "one complete matrix per source using one VerificationAgent",
            "source_overlap_rows": 0,
            "split": {
                "type": "fixed_stratified_grouped_non_golden",
                "seed": int(cfg.arbiter.seed),
                "validation_ratio": float(cfg.arbiter.test_ratio),
                "n_fit": int(len(train_idx)),
                "n_validation": int(len(val_idx)),
                "sample_id_overlap": 0,
                "pair_group_overlap": 0,
                "fit_fingerprint_sha256": _split_fingerprint(train_idx, train_meta),
                "validation_fingerprint_sha256": _split_fingerprint(val_idx, train_meta),
            },
            "n_bootstrap": int(n_bootstrap),
            "bootstrap": "class-stratified percentile 95% CI; paired resampling for deltas",
            "positive_class": "match",
        },
        "n_non_golden": int(len(train_y)),
        "n_golden": int(len(golden_y)),
        "all_feature_names": list(FEATURE_NAMES),
        "variants": results,
        "paired_bootstrap_full_minus_each": paired,
        "prediction_files": [str(prediction_json), str(prediction_jsonl)],
    }
    (out_dir / "ablations.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
