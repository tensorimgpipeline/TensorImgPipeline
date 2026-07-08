"""Tests for confusion matrix figure patterns and logger integration."""

from __future__ import annotations

import pytest

from tipi.core.permanences.loggers.basic import BasicLogger
from tipi.core.permanences.loggers.patterns import BINARY_CONFUSION_MATRIX, ConfusionMatrixFigurePattern


def test_binary_confusion_matrix_can_be_logged(tmp_path):
    logger = BasicLogger(log_dir=str(tmp_path))

    figure = logger.build_confusion_matrix_figure(
        BINARY_CONFUSION_MATRIX,
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 0, 1],
    )
    logger.log_figure(BINARY_CONFUSION_MATRIX.name, figure)

    figures = list(tmp_path.glob("binary_confusion_matrix_*.png"))
    assert len(figures) == 1


def test_confusion_matrix_pattern_supports_multiclass(tmp_path):
    logger = BasicLogger(log_dir=str(tmp_path))
    multiclass_pattern = ConfusionMatrixFigurePattern(
        name="animal_confusion_matrix",
        title="Animal Classifier",
        class_values=("cat", "dog", "bird"),
        class_labels=("Cat", "Dog", "Bird"),
        annotation_format="d",
    )

    fig = logger.build_confusion_matrix_figure(
        multiclass_pattern,
        y_true=["cat", "dog", "bird", "cat"],
        y_pred=["cat", "dog", "cat", "bird"],
    )
    ax = fig.axes[0]

    assert len(ax.get_xticklabels()) == 3
    assert len(ax.get_yticklabels()) == 3


def test_confusion_matrix_rejects_unknown_labels(tmp_path):
    logger = BasicLogger(log_dir=str(tmp_path))

    with pytest.raises(ValueError, match="unknown predicted label"):
        logger.build_confusion_matrix_figure(
            BINARY_CONFUSION_MATRIX,
            y_true=[0, 1],
            y_pred=[0, 2],
        )
