"""Tests for depth.utils.metrics."""

import pytest
import torch
from utils.metrics import (
    calculate_presence_metrics,
    calculate_counting_metrics,
    calculate_single_label_metrics,
    apply_counting_transform,
    get_task_metrics,
    SPECIES_NAMES,
    NUM_CLASSES,
)


# ---------------------------------------------------------------------------
# Presence metrics
# ---------------------------------------------------------------------------

class TestPresenceMetrics:
    def test_perfect_predictions(self):
        targets = torch.eye(4)  # (4, 4) multi-hot, one species each
        logits = targets * 10.0  # High positive logits where target=1
        m = calculate_presence_metrics(logits, targets)
        assert m["f1"] == pytest.approx(1.0, abs=1e-4)
        assert m["precision"] == pytest.approx(1.0, abs=1e-4)
        assert m["recall"] == pytest.approx(1.0, abs=1e-4)

    def test_all_wrong_predictions(self):
        targets = torch.eye(4)
        logits = -targets * 10.0  # Predicts the opposite
        m = calculate_presence_metrics(logits, targets)
        assert m["f1"] == pytest.approx(0.0, abs=1e-4)

    def test_all_negative_predictions(self):
        # Predicts nothing — recall should be ~0
        targets = torch.ones(4, 4)
        logits = torch.full((4, 4), -10.0)
        m = calculate_presence_metrics(logits, targets)
        assert m["recall"] == pytest.approx(0.0, abs=1e-4)

    def test_all_positive_predictions(self):
        # Predicts everything — precision should be low when targets are sparse
        targets = torch.eye(4)
        logits = torch.full((4, 4), 10.0)
        m = calculate_presence_metrics(logits, targets)
        assert m["precision"] == pytest.approx(0.25, abs=1e-4)

    def test_output_keys(self):
        logits = torch.zeros(4, 4)
        targets = torch.zeros(4, 4)
        m = calculate_presence_metrics(logits, targets)
        assert "f1" in m and "precision" in m and "recall" in m
        for name in SPECIES_NAMES:
            assert f"{name}_f1" in m
            assert f"{name}_precision" in m
            assert f"{name}_recall" in m

    def test_batch_size_one(self):
        logits = torch.tensor([[5.0, -5.0, -5.0, -5.0]])
        targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        m = calculate_presence_metrics(logits, targets)
        assert m["f1"] == pytest.approx(1.0, abs=1e-4)

    def test_custom_threshold(self):
        # With threshold=0.9, sigmoid(1.0)≈0.73 should NOT trigger
        logits = torch.tensor([[1.0, -1.0, -1.0, -1.0]])
        targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        m_low = calculate_presence_metrics(logits, targets, threshold=0.5)
        m_high = calculate_presence_metrics(logits, targets, threshold=0.9)
        assert m_low["f1"] > m_high["f1"]

    def test_no_nan_in_output(self):
        logits = torch.zeros(8, 4)
        targets = torch.zeros(8, 4)
        m = calculate_presence_metrics(logits, targets)
        for v in m.values():
            assert not (v != v), f"NaN found in metrics: {m}"

    def test_values_in_valid_range(self):
        logits = torch.randn(16, 4)
        targets = (torch.rand(16, 4) > 0.5).float()
        m = calculate_presence_metrics(logits, targets)
        for k, v in m.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


# ---------------------------------------------------------------------------
# Counting metrics
# ---------------------------------------------------------------------------

class TestCountingMetrics:
    def test_perfect_predictions(self):
        targets = torch.tensor([[5.0, 3.0, 0.0, 0.0], [0.0, 2.0, 4.0, 1.0]])
        m = calculate_counting_metrics(targets, targets, scale_method="clamp")
        assert m["mae"] == pytest.approx(0.0, abs=1e-4)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-4)

    def test_mae_larger_than_zero(self):
        preds = torch.zeros(4, 4)
        targets = torch.ones(4, 4) * 5.0
        m = calculate_counting_metrics(preds, targets, scale_method="clamp")
        assert m["mae"] > 0

    def test_output_keys(self):
        preds = torch.zeros(4, 4)
        targets = torch.zeros(4, 4)
        m = calculate_counting_metrics(preds, targets)
        assert "mae" in m and "rmse" in m
        for name in SPECIES_NAMES:
            assert f"{name}_mae" in m
            assert f"{name}_rmse" in m

    def test_rmse_geq_mae(self):
        preds = torch.randn(16, 4).abs()
        targets = torch.randn(16, 4).abs()
        m = calculate_counting_metrics(preds, targets, scale_method="clamp")
        assert m["rmse"] >= m["mae"] - 1e-5

    def test_clamp_method(self):
        # Predictions well above clamp_range[1] should be clamped
        preds = torch.full((4, 4), 100.0)
        targets = torch.full((4, 4), 30.0)
        m = calculate_counting_metrics(preds, targets, clamp_range=(0, 30), scale_method="clamp")
        assert m["mae"] == pytest.approx(0.0, abs=1e-4)

    def test_tanh_method(self):
        preds = torch.zeros(4, 4)
        targets = torch.zeros(4, 4)
        m = calculate_counting_metrics(preds, targets, scale_method="tanh")
        assert m["mae"] == pytest.approx(0.0, abs=1e-4)

    def test_no_nan_output(self):
        preds = torch.randn(8, 4)
        targets = torch.rand(8, 4) * 10
        m = calculate_counting_metrics(preds, targets)
        for v in m.values():
            assert not (v != v), f"NaN in {m}"


# ---------------------------------------------------------------------------
# Single-label metrics
# ---------------------------------------------------------------------------

class TestSingleLabelMetrics:
    def test_all_correct(self):
        logits = torch.eye(4) * 10  # (4,4) — argmax=i for row i
        targets = torch.arange(4)
        m = calculate_single_label_metrics(logits, targets)
        assert m["acc"] == pytest.approx(1.0)

    def test_all_wrong(self):
        logits = torch.tensor([
            [0.0, 10.0, 0.0, 0.0],  # predicts 1, target 0
            [10.0, 0.0, 0.0, 0.0],  # predicts 0, target 1
            [10.0, 0.0, 0.0, 0.0],  # predicts 0, target 2
            [10.0, 0.0, 0.0, 0.0],  # predicts 0, target 3
        ])
        targets = torch.arange(4)
        m = calculate_single_label_metrics(logits, targets)
        assert m["acc"] == pytest.approx(0.0)

    def test_half_correct(self):
        logits = torch.tensor([
            [10.0, 0.0, 0.0, 0.0],  # predicts 0, target 0 ✓
            [10.0, 0.0, 0.0, 0.0],  # predicts 0, target 1 ✗
        ])
        targets = torch.tensor([0, 1])
        m = calculate_single_label_metrics(logits, targets)
        assert m["acc"] == pytest.approx(0.5)

    def test_empty_batch(self):
        logits = torch.zeros(0, 4)
        targets = torch.zeros(0, dtype=torch.long)
        m = calculate_single_label_metrics(logits, targets)
        assert m["acc"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# apply_counting_transform
# ---------------------------------------------------------------------------

class TestApplyCountingTransform:
    def test_clamp_clips_values(self):
        logits = torch.tensor([-5.0, 0.0, 15.0, 50.0])
        result = apply_counting_transform(logits, method="clamp", max_count=30.0)
        assert result.min().item() >= 0.0
        assert result.max().item() <= 30.0

    def test_relu_non_negative(self):
        logits = torch.randn(10)
        result = apply_counting_transform(logits, method="relu")
        assert (result >= 0).all()

    def test_tanh_bounded(self):
        logits = torch.randn(10) * 100
        result = apply_counting_transform(logits, method="tanh", max_count=30.0)
        assert result.min().item() >= -30.0
        assert result.max().item() <= 30.0

    def test_zero_input_zero_output_clamp(self):
        logits = torch.zeros(4)
        result = apply_counting_transform(logits, method="clamp")
        assert (result == 0).all()


# ---------------------------------------------------------------------------
# get_task_metrics dispatcher
# ---------------------------------------------------------------------------

class TestGetTaskMetrics:
    def test_routes_presence(self):
        logits = torch.zeros(4, 4)
        targets = torch.zeros(4, 4)
        m = get_task_metrics("presence", logits, targets)
        assert "f1" in m

    def test_routes_counting(self):
        logits = torch.zeros(4, 4)
        targets = torch.zeros(4, 4)
        m = get_task_metrics("counting", logits, targets)
        assert "mae" in m

    def test_routes_single_label(self):
        logits = torch.zeros(4, 4)
        targets = torch.zeros(4, dtype=torch.long)
        m = get_task_metrics("single_label", logits, targets)
        assert "acc" in m

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task_metrics("unknown_task", torch.zeros(4, 4), torch.zeros(4, 4))
