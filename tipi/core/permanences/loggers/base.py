from __future__ import annotations

import importlib
import logging
from abc import abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from tipi.abstractions import Permanence
from tipi.core.permanences.loggers.patterns import (
    ConfusionMatrixFigurePattern,
    MetricRecord,
    ResolvedMetricRecord,
)
from tipi.paths import get_path_manager

sns: Any = importlib.import_module("seaborn")


class BaseLoggerManager(Permanence):
    """Common logger interface for all logging backends."""

    def __init__(self, log_level: str = "WARNING", log_file: str | None = None) -> None:
        self._initialized = False
        self.log_level = log_level.upper()
        self.global_step = 0
        self._last_logged_step: int | None = None
        default_log_file = get_path_manager().get_cache_dir() / "logs" / f"{self.__class__.__name__.lower()}.log"
        self.log_file = Path(log_file).expanduser() if log_file else default_log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(getattr(logging, self.log_level, logging.WARNING))
        self._logger.propagate = False

        file_handler_exists = any(
            isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == self.log_file
            for handler in self._logger.handlers
        )
        if not file_handler_exists:
            handler = logging.FileHandler(self.log_file, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            self._logger.addHandler(handler)

    def debug(self, message: str, *args: Any) -> None:
        self._logger.debug(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        self._logger.warning(message, *args)

    def error(self, message: str, *args: Any) -> None:
        self._logger.error(message, *args)

    def initialize(self) -> None:
        """Initialize logger backend via the permanence lifecycle."""
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    def supports_sweep(self) -> bool:
        """Whether this logger backend supports hyperparameter sweeps."""
        return False

    def run_sweep(self, run_once: Callable[[], None], sweep_config: dict[str, Any]) -> None:
        """Execute backend-specific sweep flow.

        Called only if supports_sweep() returns True.
        """
        raise RuntimeError(f"{self.__class__.__name__} does not support sweep execution")

    @abstractmethod
    def log_metrics(self, metrics: MetricRecord | list[MetricRecord]) -> None:
        """Log structured metrics using MetricRecord or list of MetricRecord objects."""

    @abstractmethod
    def log_figure(self, name: str, figure: Any) -> None:
        """Log a pre-built figure object for the active backend."""

    def _resolve_metric_records(self, metrics: MetricRecord | list[MetricRecord]) -> list[ResolvedMetricRecord]:
        """Convert MetricRecord(s) to resolved records with step assignments."""

        if isinstance(metrics, MetricRecord):
            records = [metrics]
        elif isinstance(metrics, list) and all(isinstance(m, MetricRecord) for m in metrics):
            records = metrics
        else:
            msg = "metrics must be a MetricRecord or list[MetricRecord]"
            raise TypeError(msg)

        resolved_records: list[ResolvedMetricRecord] = []
        max_advanced_step: int | None = None

        for record in records:
            resolved_step = self._resolve_metric_step(record)
            resolved_records.append(ResolvedMetricRecord(metric=record, step=resolved_step))
            self._last_logged_step = resolved_step
            if record.step is not None:
                self.global_step = max(self.global_step, record.step + 1)
            elif record.step_policy == "advance":
                max_advanced_step = (
                    resolved_step if max_advanced_step is None else max(max_advanced_step, resolved_step)
                )

        if max_advanced_step is not None:
            self.global_step = max(self.global_step, max_advanced_step + 1)

        return resolved_records

    def _resolve_metric_step(self, record: MetricRecord) -> int:
        """Determine the step for a metric record based on step_policy."""
        if record.step is not None:
            return record.step
        if record.step_policy == "reuse_last" and self._last_logged_step is not None:
            return self._last_logged_step
        return self.global_step

    def build_confusion_matrix_figure(
        self,
        figure_pattern: ConfusionMatrixFigurePattern,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
    ) -> Any:
        """Build a confusion matrix figure using the provided pattern blueprint."""
        if len(y_true) != len(y_pred):
            msg = "y_true and y_pred must have the same length"
            raise ValueError(msg)

        class_to_idx = {value: idx for idx, value in enumerate(figure_pattern.class_values)}
        matrix = np.zeros((len(figure_pattern.class_values), len(figure_pattern.class_values)), dtype=np.int64)

        for expected, predicted in zip(y_true, y_pred, strict=True):
            if expected not in class_to_idx:
                msg = f"Found unknown true label: {expected!r}"
                raise ValueError(msg)
            if predicted not in class_to_idx:
                msg = f"Found unknown predicted label: {predicted!r}"
                raise ValueError(msg)
            matrix[class_to_idx[expected], class_to_idx[predicted]] += 1

        heatmap_values: np.ndarray[Any, Any]
        if figure_pattern.normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            heatmap_values = matrix / row_sums
        else:
            heatmap_values = matrix

        fig, ax = plt.subplots()
        sns.heatmap(
            heatmap_values,
            annot=True,
            fmt=figure_pattern.annotation_format,
            cmap=figure_pattern.cmap,
            xticklabels=figure_pattern.class_labels,
            yticklabels=figure_pattern.class_labels,
            ax=ax,
        )
        ax.set_title(figure_pattern.title)
        ax.set_xlabel(figure_pattern.xlabel)
        ax.set_ylabel(figure_pattern.ylabel)
        fig.tight_layout()
        return fig
