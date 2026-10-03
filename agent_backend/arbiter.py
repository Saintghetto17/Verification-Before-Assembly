"""XGBoost arbiter: 37 features → P(keep/match); decision threshold = best F1 on holdout."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import FEATURE_NAMES

logger = logging.getLogger(__name__)


def classwise_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> Dict[str, Any]:
    """Return symmetric metrics while preserving match=1 as headline class."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        zero_division=0,
    )
    by_class = {
        "mismatch": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "match": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": by_class["match"]["precision"],
        "recall": by_class["match"]["recall"],
        "f1": by_class["match"]["f1"],
        "macro_f1": float(np.mean(f1)),
        "min_class_f1": float(np.min(f1)),
        "positive_class": "match",
        "by_class": by_class,
    }


def find_best_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Pick threshold maximizing F1 on (y_true, y_score). Returns (thr, f1, metrics_at_thr)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if grid is None:
        # dense grid + unique scores for stability on small holdouts
        grid = np.unique(
            np.concatenate(
                [np.linspace(0.01, 0.99, 99), y_score, [0.5]]
            )
        )
    best_thr = 0.5
    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}
    for thr in grid:
        pred = (y_score >= thr).astype(int)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_f1 = f1
            best_thr = float(thr)
            best_metrics = {
                "accuracy": float(accuracy_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": f1,
            }
    return best_thr, best_f1, best_metrics


def find_best_balanced_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Maximize the weaker class F1 without privileging either label."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if np.unique(y_true_arr).size != 2:
        raise ValueError("balanced threshold selection requires both classes")
    if grid is None:
        grid = np.unique(
            np.concatenate(([0.0, 0.5, 1.0], y_score_arr, np.linspace(0.01, 0.99, 99)))
        )

    best_threshold = 0.5
    best_metrics: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, float, float, float]] = None
    for threshold in grid:
        metrics = classwise_metrics(y_true_arr, y_score_arr >= threshold)
        # Prefer robust class balance, then overall class balance, accuracy, and
        # finally an operating point closest to 0.5 for deterministic ties.
        key = (
            metrics["min_class_f1"],
            metrics["macro_f1"],
            metrics["accuracy"],
            -abs(float(threshold) - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    best_metrics = {**best_metrics, "decision_threshold": best_threshold}
    return best_threshold, best_metrics


def train_arbiter(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_ratio: float = 0.2,
    seed: int = 42,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    project_root: Path | None = None,
    tb_run_name: str = "agent_arbiter",
) -> Tuple[XGBClassifier, StandardScaler, float, Dict[str, Any], np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric=["logloss", "auc", "error"],
        random_state=seed,
        n_jobs=4,
    )
    model.fit(
        X_train_s,
        y_train,
        eval_set=[(X_train_s, y_train), (X_test_s, y_test)],
        verbose=False,
    )

    writer = None
    tb_dir = None
    if project_root is not None:
        from .tb_logging import make_summary_writer

        writer, tb_dir = make_summary_writer(project_root, tb_run_name)

    evals = model.evals_result()
    train_hist = evals.get("validation_0", {})
    val_hist = evals.get("validation_1", {})
    n_rounds = max(
        (len(v) for v in list(train_hist.values()) + list(val_hist.values())),
        default=0,
    )
    if writer is not None:
        for i in range(n_rounds):
            if "logloss" in train_hist:
                writer.add_scalar("train/logloss", float(train_hist["logloss"][i]), i)
            if "error" in train_hist:
                writer.add_scalar("train/error", float(train_hist["error"][i]), i)
            if "auc" in train_hist:
                writer.add_scalar("train/auc", float(train_hist["auc"][i]), i)
            if "logloss" in val_hist:
                writer.add_scalar("val/logloss", float(val_hist["logloss"][i]), i)
            if "error" in val_hist:
                writer.add_scalar("val/error", float(val_hist["error"][i]), i)
            if "auc" in val_hist:
                writer.add_scalar("val/auc", float(val_hist["auc"][i]), i)

    proba = model.predict_proba(X_test_s)[:, 1]
    thr, thr_f1, thr_metrics = find_best_f1_threshold(y_test, proba)
    pred_05 = (proba >= 0.5).astype(int)

    balanced = classwise_metrics(y_test, (proba >= thr).astype(int))
    metrics: Dict[str, Any] = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "decision_threshold": float(thr),
        "threshold_source": "holdout_max_f1",
        "accuracy": thr_metrics.get("accuracy"),
        "precision": thr_metrics.get("precision"),
        "recall": thr_metrics.get("recall"),
        "f1": thr_metrics.get("f1", thr_f1),
        "macro_f1": balanced["macro_f1"],
        "min_class_f1": balanced["min_class_f1"],
        "positive_class": balanced["positive_class"],
        "by_class": balanced["by_class"],
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else None,
        "at_0.5": {
            "accuracy": float(accuracy_score(y_test, pred_05)),
            "precision": float(precision_score(y_test, pred_05, zero_division=0)),
            "recall": float(recall_score(y_test, pred_05, zero_division=0)),
            "f1": float(f1_score(y_test, pred_05, zero_division=0)),
        },
        "feature_names": FEATURE_NAMES,
    }
    if tb_dir is not None:
        metrics["tfevents"] = str(tb_dir)
    if writer is not None:
        writer.add_scalar("val/best_f1", float(metrics["f1"]), n_rounds)
        writer.add_scalar("val/decision_threshold", float(thr), n_rounds)
        writer.add_scalar("val/accuracy", float(metrics["accuracy"]), n_rounds)
        if metrics["roc_auc"] is not None:
            writer.add_scalar("val/roc_auc", metrics["roc_auc"], n_rounds)
        writer.add_text(
            "hparams",
            f"n_estimators={n_estimators} max_depth={max_depth} lr={learning_rate} "
            f"decision_threshold={thr:.4f} (max F1 on holdout) "
            f"n_train={metrics['n_train']} n_test={metrics['n_test']}",
        )
        writer.flush()
        writer.close()

    logger.info(
        "arbiter holdout: threshold=%.4f (max F1) metrics=%s",
        thr,
        {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
    )
    return model, scaler, float(thr), metrics, y_test, proba


def save_arbiter(
    model,
    scaler,
    threshold: float,
    model_path: Path,
    scaler_path: Path,
    *,
    threshold_meta: Optional[Dict[str, Any]] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "model": model,
        "decision_threshold": float(threshold),
        "threshold_source": "holdout_max_f1",
        "feature_names": list(feature_names or FEATURE_NAMES),
    }
    if threshold_meta:
        blob["threshold_meta"] = threshold_meta
    joblib.dump(blob, model_path)
    joblib.dump(scaler, scaler_path)
    # also write plain JSON for humans / services
    thr_path = model_path.with_name("decision_threshold.json")
    thr_path.write_text(
        json.dumps(
            {
                "decision_threshold": float(threshold),
                "threshold_source": "holdout_max_f1",
                **(threshold_meta or {}),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class ProbabilityEnsemble:
    """Runtime adapter for the frozen logistic/XGBoost tuning bundle."""

    def __init__(
        self,
        components: Dict[str, Any],
        component_names: Sequence[str],
        xgboost_weight: Optional[float],
    ):
        self.components = components
        self.component_names = list(component_names)
        self.xgboost_weight = xgboost_weight

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probabilities = [
            np.asarray(self.components[name].predict_proba(x), dtype=np.float64)
            for name in self.component_names
        ]
        if len(probabilities) == 1:
            return probabilities[0]
        if len(probabilities) != 2 or self.xgboost_weight is None:
            raise ValueError("Unsupported serialized ensemble layout")
        weight = float(self.xgboost_weight)
        return weight * probabilities[0] + (1.0 - weight) * probabilities[1]


def load_arbiter(model_path: Path, scaler_path: Path):
    """Returns (model, scaler, decision_threshold)."""
    blob = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    if isinstance(blob, dict) and "model" in blob:
        thr = float(blob.get("decision_threshold", 0.5))
        model = blob["model"]
        setattr(model, "_agent_feature_names", list(blob.get("feature_names", FEATURE_NAMES)))
        return model, scaler, thr
    if isinstance(blob, dict) and "components" in blob:
        component_names = list(blob.get("component_names") or blob["components"])
        if len(component_names) == 1:
            model = blob["components"][component_names[0]]
        else:
            model = ProbabilityEnsemble(
                blob["components"],
                component_names,
                blob.get("xgboost_weight"),
            )
        setattr(model, "_agent_feature_names", list(blob.get("feature_names", FEATURE_NAMES)))
        return model, scaler, float(blob.get("decision_threshold", 0.5))
    # legacy: raw XGBClassifier, default 0.5
    logger.warning(
        "arbiter.pkl has no decision_threshold (legacy); using 0.5 — re-run train_arbiter"
    )
    return blob, scaler, 0.5


def decide(prob_keep: float, *, threshold: float = 0.5) -> Tuple[str, str]:
    """
    Fixed single threshold (chosen at train time for max F1).
      verdict: PASS | REGENERATE
      pred_label: match | mismatch
    """
    if prob_keep >= threshold:
        return "PASS", "match"
    return "REGENERATE", "mismatch"


class Arbiter:
    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        *,
        decision_threshold: Optional[float] = None,
    ):
        self.model, self.scaler, saved_thr = load_arbiter(model_path, scaler_path)
        self.feature_names = list(
            getattr(self.model, "_agent_feature_names", FEATURE_NAMES)
        )
        # config override wins if explicitly set; else use trained threshold
        self.decision_threshold = (
            float(decision_threshold) if decision_threshold is not None else saved_thr
        )

    def predict_proba(self, feature_row: Dict[str, float]) -> float:
        if any(name not in feature_row for name in self.feature_names):
            from .features import add_balanced_derived_features

            feature_row = add_balanced_derived_features(feature_row)
        x = np.array([[feature_row[n] for n in self.feature_names]], dtype=np.float64)
        xs = self.scaler.transform(x)
        return float(self.model.predict_proba(xs)[0, 1])

    def predict(
        self,
        feature_row: Dict[str, float],
        judge_confidences: Sequence[float] | None = None,
    ) -> Dict[str, Any]:
        del judge_confidences  # unused; kept for call-site compatibility
        prob = self.predict_proba(feature_row)
        verdict, pred_label = decide(prob, threshold=self.decision_threshold)
        return {
            "prob_keep": prob,
            "verdict": verdict,
            "pred_label": pred_label,
            "decision_threshold": self.decision_threshold,
        }


def summarize_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labeled = [r for r in rows if r.get("label") in ("match", "mismatch")]
    y_true = [1 if r["label"] == "match" else 0 for r in labeled]
    y_pred = [1 if r.get("pred_label") == "match" else 0 for r in labeled]
    y_score = [float(r.get("prob_keep", 0.5)) for r in labeled]
    balanced = classwise_metrics(y_true, y_pred) if labeled else None
    out: Dict[str, Any] = {
        "n": len(rows),
        "n_labeled": len(labeled),
        "accuracy": balanced["accuracy"] if balanced else None,
        "precision": balanced["precision"] if balanced else None,
        "recall": balanced["recall"] if balanced else None,
        "f1": balanced["f1"] if balanced else None,
        "macro_f1": balanced["macro_f1"] if balanced else None,
        "min_class_f1": balanced["min_class_f1"] if balanced else None,
        "positive_class": "match",
        "by_class": balanced["by_class"] if balanced else {},
        "roc_auc": (
            float(roc_auc_score(y_true, y_score))
            if labeled and len(set(y_true)) > 1
            else None
        ),
        "by_verdict": {},
        "baseline_siglip_note": "stage2 random-head ~0.52 accuracy on golden",
    }
    from collections import Counter

    out["by_verdict"] = dict(Counter(r.get("verdict") for r in rows))
    out["by_pred_label"] = dict(Counter(r.get("pred_label") for r in rows))
    thr_vals = [r.get("decision_threshold") for r in rows if r.get("decision_threshold") is not None]
    if thr_vals:
        out["decision_threshold"] = float(thr_vals[0])
    return out


def maybe_shap_summary(
    model,
    scaler,
    X: np.ndarray,
    out_path: Path,
    max_samples: int = 64,
) -> Optional[str]:
    try:
        import shap  # type: ignore
    except Exception:
        return None
    Xs = scaler.transform(X[:max_samples])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    arr = np.abs(sv).mean(axis=0)
    ranking = sorted(
        zip(FEATURE_NAMES, arr.tolist()), key=lambda t: -t[1]
    )
    payload = {"mean_abs_shap": [{"feature": f, "value": v} for f, v in ranking]}
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(out_path)
