from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from tipi.core.permanences.loggers.base import BaseLoggerManager


@dataclass(frozen=True)
class SeabornTheme:
    """A reusable class to package, store and apply custom Seaborn themes."""

    name: str
    bg_color: str
    grid_color: str
    text_color: str
    palette: list[str]
    style_type: str = "whitegrid"  # or "darkgrid"

    def apply(self, context="talk") -> None:
        sns.set_theme(style=self.style_type, context=context)
        sns.set_palette(sns.color_palette(self.palette))

        is_dark = self.style_type == "darkgrid"

        plt.rcParams.update({
            "figure.facecolor": self.bg_color,
            "axes.facecolor": self.bg_color,
            "axes.edgecolor": self.grid_color if is_dark else self.text_color,
            "axes.labelcolor": self.text_color,
            "grid.color": self.grid_color,
            "text.color": self.text_color,
            "xtick.color": self.text_color,
            "ytick.color": self.text_color,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": not is_dark,
            "axes.spines.bottom": not is_dark,
        })


retro_bright_theme = SeabornTheme(
    name="retro-bright",
    bg_color="#F7F5F0",
    grid_color="#E2E0D9",
    text_color="#2B2D2F",
    palette=[
        "#1A4BDE",  # Electric Cobalt
        "#D9532B",  # Burnt Persimmon
        "#607D67",  # Sage Leaf
        "#D4A33B",  # Muted Ochre
    ],
    style_type="whitegrid",
)

retro_dark_theme = SeabornTheme(
    name="retro-dark",
    bg_color="#161B22",
    grid_color="#2D353F",
    text_color="#E6EDF0",
    style_type="darkgrid",
    palette=[
        "#00F0FF",  # Luminous Turquoise
        "#FF9EBB",  # Blush Quartz
        "#A2FF00",  # Acid Lime
        "#784BA0",  # Deep Amethyst
    ],
)


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
        theme: SeabornTheme = retro_bright_theme,
    ) -> None:
        resolved_log_file = log_file or str(Path(log_dir) / "basic_logger.log")
        super().__init__(log_level=log_level, log_file=resolved_log_file)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / metrics_filename
        theme.apply()

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
