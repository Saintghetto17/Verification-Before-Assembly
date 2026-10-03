import numpy as np

from agent_backend.ablation import (
    ROUTER_DERIVED_FEATURES,
    _metrics,
    _pair_group,
    bootstrap_confidence_intervals,
    paired_bootstrap_deltas,
    variant_feature_names,
)
from agent_backend.features import FEATURE_NAMES


def test_variant_feature_sets_remove_all_dependent_features() -> None:
    variants = variant_feature_names()

    assert variants["full"] == FEATURE_NAMES
    assert not set(ROUTER_DERIVED_FEATURES) & set(variants["no_router"])
    assert {"score_sem", "sem_w"}.isdisjoint(variants["no_judge_a"])
    assert {"score_struct", "struct_w"}.isdisjoint(variants["no_judge_b"])
    assert {"score_obj", "obj_w"}.isdisjoint(variants["no_judge_c"])
    assert variants["router_only"] == [
        "p_byt",
        "p_video",
        "p_graph",
        "p_scheme",
        "router_conf",
    ]


def test_pair_group_keeps_orig_and_fail_together() -> None:
    orig = _pair_group({"sample_id": "ml_orig/document_7/figure_2"})
    fail = _pair_group({"sample_id": "ml_l2_fail/document_7/figure_2"})
    other_domain = _pair_group({"sample_id": "juri_orig/document_7/figure_2"})
    assert orig == fail
    assert orig != other_domain


def test_bootstrap_is_deterministic_and_paired_delta_has_expected_sign() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    perfect_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    perfect_pred = (perfect_score >= 0.5).astype(int)
    weak_score = np.full(6, 0.5)
    weak_pred = np.ones(6, dtype=int)

    first = bootstrap_confidence_intervals(
        y, perfect_pred, perfect_score, n_bootstrap=50, seed=7
    )
    second = bootstrap_confidence_intervals(
        y, perfect_pred, perfect_score, n_bootstrap=50, seed=7
    )
    assert first == second
    assert first["accuracy"] == [1.0, 1.0]

    delta = paired_bootstrap_deltas(
        y,
        perfect_pred,
        perfect_score,
        weak_pred,
        weak_score,
        n_bootstrap=50,
        seed=9,
    )
    assert delta["accuracy"]["estimate"] == 0.5
    assert delta["f1"]["estimate"] > 0


def test_metrics_exposes_balanced_objective() -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.9, 0.4])
    metrics = _metrics(y, (score >= 0.5).astype(int), score)

    assert metrics["f1_match"] == metrics["by_class"]["match"]["f1"]
    assert metrics["f1_mismatch"] == metrics["by_class"]["mismatch"]["f1"]
    assert metrics["min_class_f1"] == min(
        metrics["f1_match"], metrics["f1_mismatch"]
    )
    assert "min_class_f1" in bootstrap_confidence_intervals(
        y, (score >= 0.5).astype(int), score, n_bootstrap=10, seed=1
    )
