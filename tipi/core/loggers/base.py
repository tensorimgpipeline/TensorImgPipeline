from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from tipi.abstractions import Permanence
from tipi.paths import get_path_manager


class BaseLoggerManager(Permanence):
    """Common logger interface for all logging backends."""

    def __init__(self, log_level: str = "WARNING", log_file: str | None = None) -> None:
        self._initialized = False
        self.log_level = log_level.upper()
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
    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log scalar or structured metrics for the active backend."""

    @abstractmethod
    def log_figure(self, name: str, figure: Any) -> None:
        """Log a pre-built figure object for the active backend."""

    def log_seaborn_graph(
        self,
        name: str,
        data: Any,
        x: str | None = None,
        y: str | None = None,
        kind: str = "line",
    ) -> None:
        """Create a seaborn graph and route it through log_figure."""
        fig, ax = plt.subplots()
        if kind == "bar":
            sns.barplot(data=data, x=x, y=y, ax=ax)
        elif kind == "scatter":
            sns.scatterplot(data=data, x=x, y=y, ax=ax)
        else:
            sns.lineplot(data=data, x=x, y=y, ax=ax)
        ax.set_title(name)
        self.log_figure(name, fig)
        plt.close(fig)

    def cleanup(self) -> None:
        return
