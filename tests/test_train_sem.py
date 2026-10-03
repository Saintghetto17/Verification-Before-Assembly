import pytest
import torch
import torch.nn as nn

from agent_backend.train_sem import (
    PairAwareBatchSampler,
    _grouped_train_val_split,
    _optimize_balanced_threshold,
    _paired_ranking_loss,
    _source_figure_group,
    _threshold_metrics,
)
from agent_backend.judges.match_head import resize_text_position_embeddings


def _paired_rows(n_groups: int = 8):
    rows = []
    for index in range(n_groups):
        figure = f"document_{index}/figure_1"
        rows.extend(
            [
                {
                    "sample_id": f"ml_orig/{figure}",
                    "split": "ml_orig",
                    "label": "match",
                    "image_path": f"/data/ml_orig/{figure}.png",
                },
                {
                    "sample_id": f"ml_l2_fail/{figure}",
                    "split": "ml_l2_fail",
                    "label": "mismatch",
                    "image_path": f"/data/ml_l2_fail/{figure}.png",
                },
            ]
        )
    return rows


def test_grouped_split_keeps_source_figure_variants_together() -> None:
    train, val = _grouped_train_val_split(_paired_rows(), test_ratio=0.25, seed=17)

    train_groups = {_source_figure_group(row) for row in train}
    val_groups = {_source_figure_group(row) for row in val}
    assert train_groups.isdisjoint(val_groups)
    assert {row["label"] for row in train} == {"match", "mismatch"}
    assert {row["label"] for row in val} == {"match", "mismatch"}


def test_grouped_split_rejects_golden_rows() -> None:
    rows = _paired_rows()
    rows[0] = {**rows[0], "split": "golden"}
    with pytest.raises(ValueError, match="Golden"):
        _grouped_train_val_split(rows, test_ratio=0.25, seed=17)


def test_threshold_optimization_maximizes_worst_class_f1() -> None:
    labels = [0, 0, 0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.3, 0.8, 0.4, 0.7]

    threshold, metrics = _optimize_balanced_threshold(labels, probabilities)

    assert threshold == pytest.approx(0.4)
    assert metrics == _threshold_metrics(labels, probabilities, threshold)
    assert metrics["f1_match"] == pytest.approx(0.8)
    assert metrics["f1_mismatch"] == pytest.approx(6 / 7)
    assert metrics["macro_f1"] == pytest.approx((0.8 + 6 / 7) / 2)
    assert metrics["min_f1"] == pytest.approx(0.8)


def test_pair_aware_sampler_keeps_each_pair_in_one_batch() -> None:
    rows = _paired_rows(4)
    batches = list(PairAwareBatchSampler(rows, batch_size=4, seed=3))
    location = {
        row["sample_id"]: batch_index
        for batch_index, batch in enumerate(batches)
        for row_index in batch
        for row in [rows[row_index]]
    }
    for index in range(4):
        suffix = f"document_{index}/figure_1"
        assert location[f"ml_orig/{suffix}"] == location[f"ml_l2_fail/{suffix}"]


def test_paired_ranking_loss_penalizes_reversed_pair() -> None:
    targets = torch.tensor([1.0, 0.0])
    groups = ["ml/document_1/figure_1"] * 2
    correct = _paired_ranking_loss(
        torch.tensor([1.0, 0.0]), targets, groups, margin=0.2
    )
    reversed_loss = _paired_ranking_loss(
        torch.tensor([0.0, 1.0]), targets, groups, margin=0.2
    )
    assert correct.item() == 0.0
    assert reversed_loss.item() > 1.0


def test_resize_text_positions_supports_longer_context() -> None:
    class Config:
        max_position_embeddings = 4

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Module()
            self.embeddings.position_embedding = nn.Embedding(4, 3)
            self.embeddings.register_buffer(
                "position_ids", torch.arange(4).expand((1, -1))
            )
            self.config = Config()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_model = TextModel()

    model = Model()
    resize_text_position_embeddings(model, 8)
    assert model.text_model.embeddings.position_embedding.num_embeddings == 8
    assert model.text_model.embeddings.position_ids.shape == (1, 8)
    assert model.text_model.config.max_position_embeddings == 8
