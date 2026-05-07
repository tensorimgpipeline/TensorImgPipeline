from __future__ import annotations

from pathlib import Path
from typing import Any

from tipi.core.loggers.base import BaseLoggerManager


class TensorBoardLogger(BaseLoggerManager):
    """TensorBoard-backed logger with the same interface as BasicLogger."""

    def __init__(
        self,
        log_dir: str = "logs/tensorboard",
        log_level: str = "WARNING",
        log_file: str | None = None,
    ) -> None:
        resolved_log_file = log_file or str(Path(log_dir) / "tensorboard_logger.log")
        super().__init__(log_level=log_level, log_file=resolved_log_file)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.writer: Any | None = None

    def initialize(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            super().initialize()
        except Exception as exc:
            self.warning("TensorBoard unavailable: %s", exc)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if self.writer is None:
            self.initialize()
        if self.writer is None:
            return

        for key, value in metrics.items():
            if isinstance(value, int | float):
                self.writer.add_scalar(key, value, self.global_step)
            else:
                self.writer.add_text(key, str(value), self.global_step)
        self.writer.flush()
        self.global_step += 1

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
