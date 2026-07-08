from __future__ import annotations

import dataclasses
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl

from tipi.core.permanences.loggers.base import BaseLoggerManager
from tipi.core.permanences.loggers.patterns import MetricFigurePattern, MetricRecord

sns: Any = importlib.import_module("seaborn")


@dataclass(frozen=True)
class SeabornTheme:
    """A reusable class to package, store and apply custom Seaborn themes."""

    name: str
    bg_color: str
    grid_color: str
    text_color: str
    palette: list[str]
    style_type: str = "whitegrid"  # or "darkgrid"

    def apply(self, context: str = "talk") -> None:
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

    def log_metrics(self, metrics: MetricRecord | list[MetricRecord]) -> None:
        resolved_records = self._resolve_metric_records(metrics)
        with self.metrics_file.open("a", encoding="utf-8") as f:
            for r in resolved_records:
                record_dict = dataclasses.asdict(r.metric) | {"step": r.step}
                f.write(json.dumps(record_dict, default=str) + "\n")

    def log_metric_figure(self, figure_pattern: MetricFigurePattern) -> None:
        """Rebuild a standard metric figure from logged metric history."""
        history = self._read_metric_history_df(figure_pattern)
        if history.is_empty():
            return

        fig, ax = plt.subplots()
        sns.lineplot(data=history, x="Step", y="Value", hue="Metric", ax=ax)
        ax.set_title(figure_pattern.title)
        ax.set_ylabel(figure_pattern.ylabel)
        self.log_figure(figure_pattern.name, fig)
        plt.close(fig)

    def log_figure(self, name: str, figure: Any) -> None:
        sanitized_name = name.replace(" ", "_").lower()
        figure_path = self.log_dir / f"{sanitized_name}_{self.global_step}.png"
        figure.savefig(figure_path)

    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def _read_metric_history_df(self, figure_pattern: MetricFigurePattern) -> pl.DataFrame:
        """Extract metric history matching a figure pattern from the metrics file and return as a Polars DataFrame."""
        schema = {"Metric": pl.String, "Step": pl.Int64, "Value": pl.Float64}
        if not self.metrics_file.exists():
            return pl.DataFrame(schema=schema)

        return (
            pl.read_ndjson(self.metrics_file)
            .select(["name", "pattern", "step", "value"])
            .filter(pl.col("value").is_not_null())
            .filter(
                pl.col("pattern").eq(figure_pattern.pattern) | pl.col("name").is_in(list(figure_pattern.metric_names))
            )
            .select([
                pl.col("name").alias("Metric"),
                pl.col("step").cast(pl.Int64).alias("Step"),
                pl.col("value").cast(pl.Float64).alias("Value"),
            ])
        )
