"""Helper utilities for smooth transition from scripts to pipeline.

This module provides simple, script-friendly functions that work standalone
but automatically integrate with the pipeline when available.

Usage in Scripts:
    from tipi.helpers import progress_bar, logger, device_manager

    # Works standalone - no pipeline needed!
    for epoch in progress_bar(range(10), desc="Training"):
        loss = train_step()
        logger.log({"loss": loss})

Copyright (C) 2025 Matti Kaupenjohann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from rich.progress import track

# Global context that pipeline can set
_pipeline_context: dict[str, Any] | None = None


def set_pipeline_context(context: dict[str, Any]) -> None:
    """Set pipeline context (called by PipelineExecutor).

    Args:
        context: Dictionary with permanences and controller reference.
    """
    global _pipeline_context
    _pipeline_context = context


def clear_pipeline_context() -> None:
    """Clear pipeline context."""
    global _pipeline_context
    _pipeline_context = None


def progress_bar(
    iterable: Iterable,
    desc: str = "Processing",
    total: int | None = None,
) -> Iterable:
    """Progress bar that auto-integrates with pipeline or uses rich.track.

    Works standalone or with pipeline:
    - Standalone: Uses rich.track (tqdm-like)
    - Pipeline: Uses pipeline's ProgressManager if available

    Args:
        iterable: The iterable to track progress for.
        desc: Description for the progress bar.
        total: Total items (auto-detected if not provided).

    Yields:
        Items from the iterable.

    Example:
        # Standalone script
        for epoch in progress_bar(range(10), desc="Epochs"):
            train()
    """
    # Check if running in pipeline context
    if _pipeline_context:
        progress_manager = _pipeline_context.get("progress_manager")
        if progress_manager:
            # Use pipeline's progress manager
            # TODO: Implement pipeline progress integration
            pass

    # Fallback to rich.track (tqdm-like)
    return track(iterable, description=desc, total=total)


class Logger:
    """Logger that auto-integrates with WandB or prints locally.

    Works standalone or with pipeline:
    - Standalone: Prints to console or initializes WandB manually
    - Pipeline: Uses pipeline's WandBManager if available
    """

    def __init__(self) -> None:
        self._wandb_initialized = False
        self._project: str | None = None
        self._entity: str | None = None

    def init(self, project: str, entity: str | None = None, name: str | None = None, **kwargs: Any) -> None:
        """Initialize logger (manual WandB setup or use pipeline's).

        Args:
            project: WandB project name.
            entity: WandB entity name.
            name: Run name.
            **kwargs: Additional WandB init arguments.

        Example:
            logger.init(project="my_project", entity="my_team")
        """
        # Check if running in pipeline context
        if _pipeline_context:
            wandb_manager = _pipeline_context.get("wandb_logger")
            if wandb_manager:
                # Pipeline handles WandB init
                return

        # Standalone: Initialize WandB manually
        try:
            import wandb

            wandb.init(project=project, entity=entity, name=name, **kwargs)
            self._wandb_initialized = True
        except ImportError:
            print("WandB not available, logging to console only")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to WandB or console.

        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number.

        Example:
            logger.log({"loss": 0.5, "accuracy": 0.9})
        """
        # Check if running in pipeline context
        if _pipeline_context:
            wandb_manager = _pipeline_context.get("wandb_logger")
            if wandb_manager:
                wandb_manager.log_metrics(metrics)
                return

        # Standalone: Log to WandB if initialized
        if self._wandb_initialized:
            import wandb

            wandb.log(metrics, step=step)
        else:
            # Fallback to console
            print(f"Metrics: {metrics}")


class DeviceManager:
    """Device manager that auto-integrates with pipeline or selects best device.

    Works standalone or with pipeline:
    - Standalone: Selects best available device
    - Pipeline: Uses pipeline's Device permanence
    """

    def get_device(self) -> torch.device:
        """Get best available device.

        Returns:
            torch.device: The selected device.

        Example:
            device = device_manager.get_device()
            model.to(device)
        """
        # Check if running in pipeline context
        if _pipeline_context:
            device_perm = _pipeline_context.get("device")
            if device_perm:
                return torch.device(device_perm.device)

        # Standalone: Select best device
        if torch.cuda.is_available():
            # Simple selection: first GPU
            return torch.device("cuda:0")
        return torch.device("cpu")


# Global instances
logger = Logger()
device_manager = DeviceManager()


__all__ = [
    "clear_pipeline_context",
    "device_manager",
    "logger",
    "progress_bar",
    "set_pipeline_context",
]
