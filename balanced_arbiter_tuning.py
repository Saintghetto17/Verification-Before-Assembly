"""Leakage-safe balanced arbiter tuning on non-golden precomputed features.

This module is deliberately standalone: it does not import or modify the
runtime arbiter/training pipeline.  It compares class-weighted XGBoost and
regularized logistic models with repeated stratified group cross-validation,
then fits and serializes the selected model on all development rows.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
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

from agent_backend.features import (
    BALANCED_DERIVED_FEATURE_NAMES,
    add_balanced_derived_features,
)

BASE_FEATURE_NAMES = [
    "score_sem", "score_struct", "score_obj",
    "p_byt", "p_video", "p_graph", "p_scheme",
    "sem_w", "struct_w", "obj_w", "router_conf",
    "ratio", "mean_brightness", "std_brightness", "entropy", "edge_density",
    "color_variance", "texture_contrast", "num_blobs", "avg_blob_area",
    "horizontal_lines", "vertical_lines", "symmetry_score", "saturation_mean",
    "saturation_std", "sharpness", "len_words", "len_chars", "num_digits",
    "num_verbs", "num_nouns", "num_entities", "has_numbers", "has_actions",
    "sentence_complexity", "stopword_ratio", "unique_word_ratio",
]
FEATURE_NAMES = [*BASE_FEATURE_NAMES, *BALANCED_DERIVED_FEATURE_NAMES]
LABEL_MAP = {"mismatch": 0, "match": 1}
DEFAULT_INPUT = Path("outputs/agent_v5_retrain_20260830/features.csv")
DEFAULT_MODEL_DIR = Path("train_ckpt/agent_v5_balanced_dev")
DEFAULT_REPORT_DIR = Path("outputs/agent_v5_balanced_dev")


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    params: Mapping[str, Any]
    components: tuple[str, str] | None = None
    weight: float | None = None


def pair_group_id(sample_id: str) -> str:
    """Map domain_orig/domain_l2_fail versions of a figure to one group."""
    parts = str(sample_id).split("/")
    if len(parts) < 3:
        raise ValueError(f"Malformed sample_id (expected family/path/item): {sample_id!r}")
    family = re.sub(r"_(?:orig|l2_fail)$", "", parts[0])
    return "/".join((family, *parts[1:]))


def derive_domain(sample_id: str) -> str:
    family = str(sample_id).split("/", 1)[0]
    return re.sub(r"_(?:orig|l2_fail)$", "", family)


def load_development_features(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load a development-only feature CSV and fail closed on golden rows."""
    frame = pd.read_csv(path)
    required = {"sample_id", "split", "label", "label_id", *BASE_FEATURE_NAMES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature CSV is missing columns: {missing}")
    unexpected = sorted(set(frame.columns) - required - set(FEATURE_NAMES))
    if unexpected:
        raise ValueError(f"Feature CSV has unexpected columns: {unexpected}")
    golden = (
        frame["sample_id"].astype(str).str.contains("golden", case=False, na=False)
        | frame["split"].astype(str).str.contains("golden", case=False, na=False)
    )
    if golden.any():
        raise ValueError(
            f"Refusing to tune on a CSV containing {int(golden.sum())} golden rows"
        )
    if not frame["label"].isin(LABEL_MAP).all():
        bad = sorted(frame.loc[~frame["label"].isin(LABEL_MAP), "label"].astype(str).unique())
        raise ValueError(f"Unknown labels: {bad}")
    y = frame["label"].map(LABEL_MAP).to_numpy(dtype=np.int64)
    if not np.array_equal(y, frame["label_id"].to_numpy(dtype=np.int64)):
        raise ValueError("label and label_id disagree")
    enriched = []
    for row in frame.to_dict(orient="records"):
        values = add_balanced_derived_features(
            {name: row[name] for name in BASE_FEATURE_NAMES}
        )
        for name in FEATURE_NAMES:
            if name in row:
                values[name] = float(row[name])
        enriched.append(values)
    X = np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in enriched],
        dtype=np.float64,
    )
    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinite values")
    if np.unique(y).size != 2:
        raise ValueError("Both match and mismatch classes are required")
    return frame, X, y


def class_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float] | None = None
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    by_class: dict[str, dict[str, float | int]] = {}
    for label, name in ((0, "mismatch"), (1, "match")):
        by_class[name] = {
            "precision": float(precision_score(y, pred, pos_label=label, zero_division=0)),
            "recall": float(recall_score(y, pred, pos_label=label, zero_division=0)),
            "f1": float(f1_score(y, pred, pos_label=label, zero_division=0)),
            "support": int(np.sum(y == label)),
        }
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "by_class": by_class,
        "macro_f1": float(np.mean([by_class[c]["f1"] for c in ("mismatch", "match")])),
        "min_class_f1": float(min(by_class[c]["f1"] for c in ("mismatch", "match"))),
        "confusion": {
            "true_mismatch_pred_mismatch": int(cm[0, 0]),
            "true_mismatch_pred_match": int(cm[0, 1]),
            "true_match_pred_mismatch": int(cm[1, 0]),
            "true_match_pred_match": int(cm[1, 1]),
        },
    }
    if y_score is not None:
        score = np.asarray(y_score, dtype=np.float64)
        result["roc_auc"] = (
            float(roc_auc_score(y, score)) if np.unique(y).size == 2 else None
        )
    return result


def choose_balanced_threshold(
    y_true: Sequence[int], probabilities: Sequence[float]
) -> tuple[float, dict[str, Any]]:
    """Maximize min class F1, then macro F1, accuracy, and closeness to 0.5."""
    y = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.unique(
        np.concatenate(
            (
                score,
                np.asarray([0.0, 0.5, 1.0, np.nextafter(score.max(), np.inf)]),
            )
        )
    )
    best_key: tuple[float, float, float, float, float] | None = None
    best_threshold = 0.5
    for threshold in thresholds:
        pred = score >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        f1_match = 2 * tp / max(2 * tp + fp + fn, 1)
        f1_mismatch = 2 * tn / max(2 * tn + fp + fn, 1)
        macro_f1 = (f1_match + f1_mismatch) / 2
        accuracy = (tp + tn) / len(y)
        key = (
            min(f1_match, f1_mismatch),
            macro_f1,
            accuracy,
            -abs(float(threshold) - 0.5),
            -float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    best_pred = score >= best_threshold
    return best_threshold, class_metrics(y, best_pred, score)


def base_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        candidates.append(
            Candidate(f"logreg_C{c:g}", "logistic", {"C": c})
        )
    xgb_grid = (
        (120, 2, 0.03, 3.0, 0.8, 0.8, 1.0),
        (200, 2, 0.03, 3.0, 0.8, 1.0, 2.0),
        (120, 2, 0.07, 3.0, 1.0, 0.8, 2.0),
        (200, 2, 0.07, 5.0, 0.8, 0.8, 4.0),
        (120, 3, 0.03, 3.0, 0.8, 0.8, 2.0),
        (200, 3, 0.03, 5.0, 0.8, 1.0, 4.0),
        (120, 3, 0.07, 5.0, 1.0, 0.8, 4.0),
        (200, 3, 0.07, 8.0, 0.8, 0.8, 8.0),
    )
    for i, (n, depth, rate, child, subsample, cols, reg_lambda) in enumerate(xgb_grid):
        candidates.append(
            Candidate(
                f"xgb_{i:02d}", "xgboost",
                {
                    "n_estimators": n, "max_depth": depth, "learning_rate": rate,
                    "min_child_weight": child, "subsample": subsample,
                    "colsample_bytree": cols, "reg_lambda": reg_lambda,
                    "reg_alpha": 0.1,
                },
            )
        )
    return candidates


def _fit_base(
    candidate: Candidate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    *,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, Any, StandardScaler]:
    scaler = StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    valid_scaled = scaler.transform(X_valid)
    if candidate.family == "logistic":
        model = LogisticRegression(
            **candidate.params,
            class_weight="balanced",
            solver="liblinear",
            max_iter=4000,
            random_state=seed,
        )
    else:
        negatives = int(np.sum(y_train == 0))
        positives = int(np.sum(y_train == 1))
        model = XGBClassifier(
            **candidate.params,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negatives / positives,
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            verbosity=0,
        )
    model.fit(train_scaled, y_train)
    return model.predict_proba(valid_scaled)[:, 1], model, scaler


def repeated_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    candidates: Sequence[Candidate],
    *,
    folds: int,
    repeats: int,
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    predictions = {
        candidate.name: np.full((repeats, len(y)), np.nan, dtype=np.float64)
        for candidate in candidates
    }
    split_records: list[dict[str, Any]] = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed + repeat
        )
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups)):
            if set(groups[train_idx]) & set(groups[valid_idx]):
                raise RuntimeError("Group leakage detected")
            split_records.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                    "valid_match": int(y[valid_idx].sum()),
                    "valid_mismatch": int(len(valid_idx) - y[valid_idx].sum()),
                    "_valid_indices": valid_idx.tolist(),
                }
            )
            for candidate in candidates:
                proba, _, _ = _fit_base(
                    candidate,
                    X[train_idx],
                    y[train_idx],
                    X[valid_idx],
                    seed=seed + repeat * folds + fold,
                    n_jobs=n_jobs,
                )
                predictions[candidate.name][repeat, valid_idx] = proba
    for name, matrix in predictions.items():
        if np.isnan(matrix).any():
            raise RuntimeError(f"Incomplete OOF predictions for {name}")
    return predictions, split_records


def _ensemble_candidates(base: Sequence[Candidate]) -> list[Candidate]:
    logistic = [candidate for candidate in base if candidate.family == "logistic"]
    trees = [candidate for candidate in base if candidate.family == "xgboost"]
    ensembles: list[Candidate] = []
    for tree in trees:
        for linear in logistic:
            for weight in (0.25, 0.5, 0.75):
                ensembles.append(
                    Candidate(
                        f"ensemble_{tree.name}_{linear.name}_xgb{weight:g}",
                        "ensemble",
                        {},
                        components=(tree.name, linear.name),
                        weight=weight,
                    )
                )
    return ensembles


def _stability(
    y: np.ndarray,
    prediction_matrix: np.ndarray,
    threshold: float,
    split_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for split in split_records:
        repeat = int(split["repeat"])
        valid_idx = np.asarray(split["_valid_indices"], dtype=np.int64)
        metrics = class_metrics(
            y[valid_idx],
            prediction_matrix[repeat, valid_idx] >= threshold,
            prediction_matrix[repeat, valid_idx],
        )
        rows.append(
            {
                "repeat": repeat,
                "fold": int(split["fold"]),
                "n_valid": int(len(valid_idx)),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "min_class_f1": metrics["min_class_f1"],
                "f1_match": metrics["by_class"]["match"]["f1"],
                "f1_mismatch": metrics["by_class"]["mismatch"]["f1"],
            }
        )
    summary = {}
    for metric in ("accuracy", "macro_f1", "min_class_f1", "f1_match", "f1_mismatch"):
        values = np.asarray([row[metric] for row in rows])
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {"per_fold": rows, "summary": summary}


def group_bootstrap_ci(
    y: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    *,
    n_bootstrap: int = 1000,
    seed: int,
) -> dict[str, list[float]]:
    """Bootstrap source figures while preserving all paired rows."""
    unique_groups = np.unique(groups)
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    samples = {
        name: []
        for name in (
            "accuracy",
            "macro_f1",
            "min_class_f1",
            "f1_match",
            "f1_mismatch",
            "roc_auc",
        )
    }
    for _ in range(n_bootstrap):
        drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_rows[group] for group in drawn])
        metrics = class_metrics(
            y[indices], score[indices] >= threshold, score[indices]
        )
        samples["accuracy"].append(metrics["accuracy"])
        samples["macro_f1"].append(metrics["macro_f1"])
        samples["min_class_f1"].append(metrics["min_class_f1"])
        samples["f1_match"].append(metrics["by_class"]["match"]["f1"])
        samples["f1_mismatch"].append(metrics["by_class"]["mismatch"]["f1"])
        if metrics["roc_auc"] is not None:
            samples["roc_auc"].append(metrics["roc_auc"])
    return {
        name: [float(value) for value in np.percentile(values, [2.5, 97.5])]
        for name, values in samples.items()
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def tune_and_fit(
    input_path: Path,
    model_dir: Path,
    report_dir: Path,
    *,
    folds: int = 5,
    repeats: int = 5,
    seed: int = 20260830,
    n_jobs: int = 4,
) -> dict[str, Any]:
    frame, X, y = load_development_features(input_path)
    groups = frame["sample_id"].map(pair_group_id).to_numpy()
    domains = frame["sample_id"].map(derive_domain).to_numpy()
    if folds < 2 or repeats < 1:
        raise ValueError("folds must be >= 2 and repeats must be >= 1")

    base = base_candidates()
    prediction_matrices, splits = repeated_oof(
        X, y, groups, base, folds=folds, repeats=repeats, seed=seed, n_jobs=n_jobs
    )
    ensembles = _ensemble_candidates(base)
    by_name = {candidate.name: candidate for candidate in [*base, *ensembles]}
    for candidate in ensembles:
        tree_name, linear_name = candidate.components or ("", "")
        prediction_matrices[candidate.name] = (
            candidate.weight * prediction_matrices[tree_name]
            + (1.0 - candidate.weight) * prediction_matrices[linear_name]
        )

    candidate_reports: list[dict[str, Any]] = []
    aggregate_oof: dict[str, np.ndarray] = {}
    for candidate in [*base, *ensembles]:
        probabilities = prediction_matrices[candidate.name].mean(axis=0)
        aggregate_oof[candidate.name] = probabilities
        threshold, metrics = choose_balanced_threshold(y, probabilities)
        candidate_reports.append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "params": dict(candidate.params),
                "components": candidate.components,
                "weight": candidate.weight,
                "threshold": threshold,
                "metrics": metrics,
            }
        )
    candidate_reports.sort(
        key=lambda row: (
            row["metrics"]["min_class_f1"],
            row["metrics"]["macro_f1"],
            row["metrics"]["accuracy"],
            -abs(row["threshold"] - 0.5),
            row["name"],
        ),
        reverse=True,
    )
    best = candidate_reports[0]
    selected = by_name[best["name"]]
    selected_score = aggregate_oof[selected.name]
    selected_pred = (selected_score >= best["threshold"]).astype(np.int64)

    domain_breakdown = {}
    for domain in sorted(np.unique(domains)):
        idx = domains == domain
        domain_breakdown[str(domain)] = class_metrics(
            y[idx], selected_pred[idx], selected_score[idx]
        )
    stability = _stability(
        y, prediction_matrices[selected.name], best["threshold"], splits
    )
    bootstrap_ci = group_bootstrap_ci(
        y,
        selected_score,
        groups,
        best["threshold"],
        seed=seed + 10000,
    )
    public_splits = [
        {key: value for key, value in split.items() if not key.startswith("_")}
        for split in splits
    ]

    final_components: dict[str, Any] = {}
    component_names = (
        selected.components if selected.family == "ensemble" else (selected.name,)
    )
    final_scaler = StandardScaler().fit(X)
    X_scaled = final_scaler.transform(X)
    for component_name in component_names:
        component = by_name[component_name]
        if component.family == "logistic":
            model = LogisticRegression(
                **component.params,
                class_weight="balanced",
                solver="liblinear",
                max_iter=4000,
                random_state=seed,
            )
        else:
            model = XGBClassifier(
                **component.params,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=int(np.sum(y == 0)) / int(np.sum(y == 1)),
                random_state=seed,
                n_jobs=n_jobs,
                tree_method="hist",
                verbosity=0,
            )
        model.fit(X_scaled, y)
        final_components[component_name] = model

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_blob = {
        "kind": selected.family,
        "name": selected.name,
        "components": final_components,
        "component_names": component_names,
        "xgboost_weight": selected.weight,
        "decision_threshold": best["threshold"],
        "feature_names": FEATURE_NAMES,
        "positive_class": "match",
    }
    joblib.dump(model_blob, model_dir / "model.joblib")
    joblib.dump(final_scaler, model_dir / "scaler.joblib")
    (model_dir / "feature_names.json").write_text(
        json.dumps(FEATURE_NAMES, indent=2) + "\n", encoding="utf-8"
    )
    (model_dir / "decision_threshold.json").write_text(
        json.dumps(
            {
                "decision_threshold": best["threshold"],
                "selection_objective": "max_min_class_f1_then_macro_f1_then_accuracy",
                "positive_class": "match",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    oof_frame = frame[["sample_id", "split", "label", "label_id"]].copy()
    oof_frame["pair_group_id"] = groups
    oof_frame["domain"] = domains
    oof_frame["oof_probability_match"] = selected_score
    oof_frame["threshold"] = best["threshold"]
    oof_frame["prediction"] = np.where(selected_pred == 1, "match", "mismatch")
    oof_frame["correct"] = selected_pred == y
    oof_frame.to_csv(report_dir / "oof_predictions.csv", index=False)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "golden_data_used": False,
        "dataset": {
            "rows": int(len(frame)),
            "features": len(FEATURE_NAMES),
            "match": int(y.sum()),
            "mismatch": int(len(y) - y.sum()),
            "pair_groups": int(np.unique(groups).size),
            "paired_groups": int(pd.Series(groups).value_counts().eq(2).sum()),
            "domains": sorted(map(str, np.unique(domains))),
        },
        "cv": {
            "strategy": "Repeated StratifiedGroupKFold",
            "folds": folds,
            "repeats": repeats,
            "seed": seed,
            "group_source": "sample_id family suffix normalization",
            "splits": public_splits,
        },
        "selection": {
            **best,
            "objective": "max min(F1_match,F1_mismatch); tie macro-F1, accuracy",
            "fold_stability": stability,
            "group_bootstrap_ci95": bootstrap_ci,
            "domain_breakdown": domain_breakdown,
        },
        "candidates_ranked": candidate_reports,
        "artifacts": {
            "model": str(model_dir / "model.joblib"),
            "scaler": str(model_dir / "scaler.joblib"),
            "threshold": str(model_dir / "decision_threshold.json"),
            "feature_names": str(model_dir / "feature_names.json"),
            "oof_predictions": str(report_dir / "oof_predictions.csv"),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    report_path = report_dir / "report.json"
    report["artifacts"]["report"] = str(report_path)
    report_path.write_text(
        json.dumps(report, indent=2, default=_jsonable) + "\n", encoding="utf-8"
    )
    return report


def refit_frozen_selection(
    input_path: Path,
    selected_bundle_path: Path,
    model_dir: Path,
) -> dict[str, Any]:
    """Refit frozen hyperparameters on all non-golden rows, preserving OOF threshold."""
    frame = pd.read_csv(input_path)
    golden = (
        frame["sample_id"].astype(str).str.contains("golden", case=False, na=False)
        | frame["split"].astype(str).str.contains("golden", case=False, na=False)
    )
    if golden.any():
        raise ValueError("Refusing to refit on golden rows")
    source = joblib.load(selected_bundle_path)
    feature_names = list(source["feature_names"])
    enriched = []
    for row in frame.to_dict(orient="records"):
        values = add_balanced_derived_features(
            {name: row[name] for name in BASE_FEATURE_NAMES}
        )
        for name in feature_names:
            if name in row:
                values[name] = float(row[name])
        enriched.append(values)
    X = np.asarray(
        [[row[name] for name in feature_names] for row in enriched],
        dtype=np.float64,
    )
    y = frame["label"].map(LABEL_MAP).to_numpy(dtype=np.int64)
    if not np.isfinite(X).all() or np.unique(y).size != 2:
        raise ValueError("Full non-golden refit data are invalid")

    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    components = {
        name: clone(model).fit(scaled, y)
        for name, model in source["components"].items()
    }
    final_bundle = {
        **source,
        "components": components,
        "refit_rows": int(len(frame)),
        "refit_source": str(input_path),
        "threshold_frozen_from": str(selected_bundle_path),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_bundle, model_dir / "model.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    manifest = {
        "golden_data_used": False,
        "rows": int(len(frame)),
        "match": int(y.sum()),
        "mismatch": int(len(y) - y.sum()),
        "feature_names": feature_names,
        "decision_threshold": float(source["decision_threshold"]),
        "selection_bundle": str(selected_bundle_path),
        "model": str(model_dir / "model.joblib"),
        "scaler": str(model_dir / "scaler.joblib"),
    }
    (model_dir / "refit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = tune_and_fit(
        args.input,
        args.model_dir,
        args.report_dir,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    selected = report["selection"]
    metrics = selected["metrics"]
    print(
        json.dumps(
            {
                "selected": selected["name"],
                "threshold": selected["threshold"],
                "min_class_f1": metrics["min_class_f1"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "report": report["artifacts"]["report"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
