from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
from wandb.wandb_run import Run

from tipi.abstractions import Permanence
from tipi.core.permanences.loggers.base import BaseLoggerManager
from tipi.errors import SweepNoConfigError


@dataclass
class NullWandBLogger(Permanence): ...


class WandBLogger(BaseLoggerManager):
    """Weights & Biases logger backend."""

    def __init__(
        self,
        project: str,
        entity: str,
        name: str = "",
        tags: list[str] | None = None,
        notes: str = "",
        count: int = 10,
        log_level: str = "WARNING",
        log_file: str | None = None,
    ) -> None:
        super().__init__(log_level=log_level, log_file=log_file)
        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags
        self.notes = notes
        self.count = count

        self.cache_path = Path.home() / ".cache/wandb_local/sweep_id"
        self.sweep_id: str | None = None
        self.run_ids: list[Run] = []
        self.global_step = 0

    def create_sweep(self, config: dict[str, Any]) -> None:
        if not config:
            raise SweepNoConfigError()
        self._read_cached_sweep_id()
        if not self._is_sweep_active():
            self.sweep_id = wandb.sweep(config, entity=self.entity, project=self.project)
        if self.sweep_id is not None:
            self._write_cache_sweep_id()

    def create_sweep_agent(self, func: Any) -> None:
        if self.sweep_id:
            wandb.agent(self.sweep_id, function=func, entity=self.entity, project=self.project, count=self.count)

    def initialize(self) -> None:
        os.environ["WANDB_SILENT"] = "true"
        name = f"{self.name}_{wandb.util.generate_id()}"
        notes = f"{self.notes} {self.sweep_id}" if self.sweep_id else self.notes
        run_id = wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            tags=self.tags,
            notes=notes,
        )
        self.run_ids.append(run_id)
        super().initialize()

    def supports_sweep(self) -> bool:
        return True

    def run_sweep(self, run_once: Callable[[], None], sweep_config: dict[str, Any]) -> None:
        self.create_sweep(sweep_config)
        self.create_sweep_agent(run_once)

    def _is_sweep_active(self) -> str | bool:
        if self.sweep_id is None:
            return False
        api = wandb.Api()
        try:
            sweep = api.sweep(f"{self.entity}/{self.project}/{self.sweep_id}")
        except wandb.errors.CommError as e:
            self.warning("Error fetching sweep: %s", e)
            return False
        else:
            return sweep.state.lower() in ["running", "pending"]

    def _write_cache_sweep_id(self) -> None:
        if self.sweep_id:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w") as f:
                f.write(self.sweep_id)

    def _read_cached_sweep_id(self) -> None:
        if self.cache_path.exists():
            with self.cache_path.open() as f:
                self.sweep_id = f.read()

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        wandb.log(metrics, step=self.global_step)
        self.global_step += 1

    def log_figure(self, name: str, figure: Any) -> None:
        wandb.log({name: wandb.Image(figure)}, step=self.global_step)
