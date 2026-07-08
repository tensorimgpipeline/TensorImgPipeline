"""Tests for metric pattern logging across backends."""

from __future__ import annotations

import json

import pytest
from polars import DataFrame
from polars.exceptions import InvalidOperationError

from tipi.core.permanences import (
    ACCURACY_CURVE,
    LOSS_CURVE,
    batch_loss,
    test_accuracy,
    test_loss,
)
from tipi.core.permanences.loggers.basic import BasicLogger


class TestMetricPatterns:
    """Test metric patterns with step policies."""

    def test_batch_loss_pattern_increments_step(self, tmp_path):
        """Batch loss should increment global_step (step_policy='advance')."""
        logger = BasicLogger(log_dir=str(tmp_path))

        logger.log_metrics(batch_loss(1.3))
        logger.log_metrics(batch_loss(1.1))

        assert logger.global_step == 2
        entries = [json.loads(line) for line in logger.metrics_file.read_text(encoding="utf-8").splitlines()]
        assert entries[0]["step"] == 0
        assert entries[1]["step"] == 1

    def test_reuse_last_step_policy(self, tmp_path):
        """Test metrics with step_policy='reuse_last' (test/validation metrics)."""
        logger = BasicLogger(log_dir=str(tmp_path))

        logger.log_metrics(batch_loss(1.3))
        logger.log_metrics(batch_loss(1.1))
        logger.log_metrics([test_loss(1.0), test_accuracy(0.65)])

        entries = [json.loads(line) for line in logger.metrics_file.read_text(encoding="utf-8").splitlines()]
        assert entries[0]["step"] == 0
        assert entries[1]["step"] == 1
        assert entries[2]["step"] == 1
        assert entries[3]["step"] == 1
        assert logger.global_step == 2

    def test_pattern_serializes_metadata(self, tmp_path):
        """Pattern metadata should be serialized in JSONL."""
        logger = BasicLogger(log_dir=str(tmp_path))

        logger.log_metrics(batch_loss(1.3))

        entries = [json.loads(line) for line in logger.metrics_file.read_text(encoding="utf-8").splitlines()]
        metric_entry = entries[0]
        assert metric_entry["name"] == "batch_loss"
        assert metric_entry["value"] == 1.3
        assert metric_entry["pattern"] == "loss"
        assert metric_entry["stage"] == "batch"
        assert metric_entry["split"] == "train"
        assert metric_entry["step_policy"] == "advance"

    def test_explicit_step_override(self, tmp_path):
        """Can override step with explicit value."""
        logger = BasicLogger(log_dir=str(tmp_path))

        logger.log_metrics(batch_loss(1.3, step=42))

        entries = [json.loads(line) for line in logger.metrics_file.read_text(encoding="utf-8").splitlines()]
        assert entries[0]["step"] == 42
        assert logger.global_step == 43

    def test_list_of_metrics(self, tmp_path):
        """Can log list of metric records."""
        logger = BasicLogger(log_dir=str(tmp_path))

        logger.log_metrics([batch_loss(1.3), batch_loss(1.1)])

        entries = [json.loads(line) for line in logger.metrics_file.read_text(encoding="utf-8").splitlines()]
        assert entries[0]["step"] == 0
        assert len(entries) == 2
        assert entries[0]["name"] == "batch_loss"
        assert entries[1]["name"] == "batch_loss"


class TestMetricFigureReconstruction:
    """Test rebuilding standard metric figures from history."""

    def test_loss_curve_reconstruction(self, tmp_path):
        """Should reconstruct loss curve from metric history."""
        logger = BasicLogger(log_dir=str(tmp_path))

        logger.log_metrics(batch_loss(1.3))
        logger.log_metrics(batch_loss(1.1))
        logger.log_metrics([test_loss(1.0)])

        logger.log_metric_figure(LOSS_CURVE)

        # Check that a loss figure was created
        figures = list(tmp_path.glob("loss_*.png"))
        assert len(figures) == 1

    def test_accuracy_curve_reconstruction(self, tmp_path):
        """Should reconstruct accuracy curve from metric history."""
        logger = BasicLogger(log_dir=str(tmp_path))

        logger.log_metrics(batch_loss(1.3))
        logger.log_metrics(batch_loss(1.1))
        logger.log_metrics([test_accuracy(0.65)])

        logger.log_metric_figure(ACCURACY_CURVE)

        # Check that an accuracy figure was created
        figures = list(tmp_path.glob("accuracy_*.png"))
        assert len(figures) == 1

    def test_figure_extraction_empty_history(self, tmp_path):
        """Should handle empty metric history gracefully."""
        logger = BasicLogger(log_dir=str(tmp_path))

        logger.log_metric_figure(LOSS_CURVE)

        # No figure should be created
        figures = list(tmp_path.glob("*.png"))
        assert len(figures) == 0

    def test_figure_matches_pattern_by_name_and_pattern(self, tmp_path):
        """Figure should match metrics by pattern field or metric name."""
        logger = BasicLogger(log_dir=str(tmp_path))

        logger.log_metrics(batch_loss(1.3))
        logger.log_metrics([test_loss(1.0)])

        history = logger._read_metric_history_df(LOSS_CURVE)

        # Both metrics should be in the history
        assert isinstance(history, DataFrame)
        assert history.columns == ["Metric", "Step", "Value"]
        assert "batch_loss" in history["Metric"]
        assert "test_loss" in history["Metric"]
        assert len(history[0]["Value"]) == 1
        assert len(history[1]["Value"]) == 1

    def test_figure_rejects_non_numeric_values(self, tmp_path):
        """Should fail when metric history contains non-numeric values."""
        logger = BasicLogger(log_dir=str(tmp_path))

        # Log a batch loss as non numeric value
        logger.log_metrics([batch_loss("one.three")])

        with pytest.raises(InvalidOperationError):
            logger._read_metric_history_df(LOSS_CURVE)
