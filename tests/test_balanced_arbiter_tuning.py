import numpy as np
import pandas as pd
import pytest

from balanced_arbiter_tuning import (
    FEATURE_NAMES,
    choose_balanced_threshold,
    class_metrics,
    derive_domain,
    load_development_features,
    pair_group_id,
)


def test_pair_group_keeps_orig_and_fail_together() -> None:
    assert (
        pair_group_id("ml_orig/document_7/figure_2")
        == pair_group_id("ml_l2_fail/document_7/figure_2")
    )
    assert (
        pair_group_id("ml_orig/document_7/figure_2")
        != pair_group_id("juri_orig/document_7/figure_2")
    )
    assert derive_domain("juri_l2_fail/document_7/figure_2") == "juri"


def test_threshold_optimizes_the_weaker_class() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.05, 0.2, 0.55, 0.45, 0.7, 0.9])

    threshold, metrics = choose_balanced_threshold(y, score)

    assert threshold == 0.45
    assert metrics["by_class"]["match"]["f1"] == 6 / 7
    assert metrics["by_class"]["mismatch"]["f1"] == 0.8
    assert metrics["min_class_f1"] == 0.8


def test_class_metrics_reports_both_classes() -> None:
    metrics = class_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    assert metrics["by_class"]["match"]["f1"] == 2 / 3
    assert metrics["by_class"]["mismatch"]["f1"] == 0.8
    assert metrics["macro_f1"] == (2 / 3 + 0.8) / 2
    assert metrics["min_class_f1"] == 2 / 3


def test_loader_refuses_golden_rows(tmp_path) -> None:
    row = {
        "sample_id": "golden_orig/document_1/figure_1",
        "split": "golden",
        "label": "match",
        "label_id": 1,
        **{name: 0.0 for name in FEATURE_NAMES},
    }
    path = tmp_path / "features.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="golden"):
        load_development_features(path)


def test_loader_validates_exact_feature_schema(tmp_path) -> None:
    rows = []
    for label, label_id, family in (
        ("match", 1, "ml_orig"),
        ("mismatch", 0, "ml_l2_fail"),
    ):
        rows.append(
            {
                "sample_id": f"{family}/document_1/figure_1",
                "split": family,
                "label": label,
                "label_id": label_id,
                **{name: 0.0 for name in FEATURE_NAMES},
            }
        )
    path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    frame, X, y = load_development_features(path)

    assert len(frame) == 2
    assert X.shape == (2, len(FEATURE_NAMES))
    assert y.tolist() == [1, 0]
