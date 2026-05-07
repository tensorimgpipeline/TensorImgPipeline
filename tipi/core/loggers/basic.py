from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tipi.core.loggers.base import BaseLoggerManager


class BasicLogger(BaseLoggerManager):
    """Filesystem logger.

    - debug/warning/error -> terminal
    - metrics -> JSONL file in log_dir
    - figures -> image files in log_dir
    """

    def __init__(
        self,
        log_dir: str = "logs/basic",
        metrics_filename: str = "metrics.jsonl",
        log_level: str = "WARNING",
        log_file: str | None = None,
    ) -> None:
        resolved_log_file = log_file or str(Path(log_dir) / "basic_logger.log")
        super().__init__(log_level=log_level, log_file=resolved_log_file)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / metrics_filename
        self.global_step = 0

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        payload = {
            "step": self.global_step,
            "metrics": metrics,
        }
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
        self.global_step += 1

    def log_figure(self, name: str, figure: Any) -> None:
        sanitized_name = name.replace(" ", "_").lower()
        figure_path = self.log_dir / f"{sanitized_name}_{self.global_step}.png"
        figure.savefig(figure_path)
