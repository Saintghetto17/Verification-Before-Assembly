from agent_backend.arbiter import classwise_metrics, find_best_balanced_threshold
from agent_backend.features import (
    BALANCED_DERIVED_FEATURE_NAMES,
    add_balanced_derived_features,
)


def test_classwise_metrics_reports_both_classes() -> None:
    metrics = classwise_metrics([1, 1, 0, 0], [1, 0, 0, 0])

    assert metrics["positive_class"] == "match"
    assert metrics["by_class"]["match"]["precision"] == 1.0
    assert metrics["by_class"]["match"]["recall"] == 0.5
    assert metrics["by_class"]["mismatch"]["precision"] == 2 / 3
    assert metrics["by_class"]["mismatch"]["recall"] == 1.0
    assert metrics["macro_f1"] == (2 / 3 + 0.8) / 2
    assert metrics["min_class_f1"] == 2 / 3


def test_balanced_threshold_requires_both_classes_and_balances_f1() -> None:
    threshold, metrics = find_best_balanced_threshold(
        [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9]
    )
    assert 0.4 < threshold <= 0.6
    assert metrics["min_class_f1"] == 1.0


def test_balanced_derived_features_are_finite() -> None:
    row = {
        "score_sem": 0.9,
        "score_struct": 0.25,
        "score_obj": 0.5,
        "p_byt": 0.7,
        "p_video": 0.1,
        "p_graph": 0.1,
        "p_scheme": 0.1,
        "has_numbers": 1.0,
    }
    enriched = add_balanced_derived_features(row)
    assert set(BALANCED_DERIVED_FEATURE_NAMES) <= enriched.keys()
    assert enriched["sem_margin"] == 0.4
    assert enriched["ocr_numeric_consistency"] == 0.25
    assert 0.0 <= enriched["router_entropy"] <= 1.0
