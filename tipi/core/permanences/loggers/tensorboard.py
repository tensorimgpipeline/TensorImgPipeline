from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from torch.utils.tensorboard import SummaryWriter

from tipi.core.permanences.loggers.base import BaseLoggerManager
from tipi.core.permanences.loggers.patterns import MetricFigurePattern, MetricRecord

MetricLayoutEntry = list[str | list[str]]

MetricDict = dict[str, MetricLayoutEntry]


class TensorBoardMetricDict(TypedDict):
    Metrics: MetricDict


class TensorBoardLogger(BaseLoggerManager):
    """TensorBoard-backed logger with the same interface as BasicLogger."""

    def __init__(
        self,
        log_dir: str = "logs/tensorboard",
        log_level: str = "WARNING",
        log_file: str | None = None,
        patterns: list[MetricFigurePattern] | None = None,
    ) -> None:
        resolved_log_file = log_file or str(Path(log_dir) / "tensorboard_logger.log")
        super().__init__(log_level=log_level, log_file=resolved_log_file)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer: Any | None = None
        self.patterns = patterns or []

    def initialize(self) -> None:
        super().initialize()
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        layout: TensorBoardMetricDict = self._build_layout()
        self.writer.add_custom_scalars(layout)

    def log_metrics(self, metrics: MetricRecord | list[MetricRecord]) -> None:
        if self.writer is None:
            self.initialize()
        if self.writer is None:
            return

        for record in self._resolve_metric_records(metrics):
            if isinstance(record.value, int | float):
                self.writer.add_scalar(f"{record.pattern}/{record.name}", float(record.value), record.step)
            else:
                self.writer.add_text(record.name, str(record.value), record.step)
        self.writer.flush()

    def log_figure(self, name: str, figure: Any) -> None:
        if self.writer is None:
            self.initialize()
        if self.writer is None:
            return
        self.writer.add_figure(name, figure, global_step=self.global_step)
        self.writer.flush()

    def cleanup(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _build_layout(self) -> TensorBoardMetricDict:
        """Builds the layout for custom scalars in TensorBoard based on the provided patterns.

        based on: https://stackoverflow.com/a/71524389/10985257

        Returns:
            TensorBoardMetricDict: layout for custom scalars
        """
        layout: TensorBoardMetricDict = {"Metrics": {}}
        for pattern in self.patterns:
            layout["Metrics"][pattern.title] = [
                "Multiline",
                [f"{pattern.pattern}/{name}" for name in pattern.metric_names],
            ]
        return layout
