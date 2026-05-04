"""Decorators for automatic progress tracking in pipeline functions.

This module provides the @progress_task decorator that automatically wraps
iterables with progress tracking, working seamlessly in both standalone
and pipeline modes.

Usage:
    @progress_task(desc="Training Epoch")
    def train_epoch(dataloader, model):
        '''Automatically tracks progress over dataloader!'''
        for batch in dataloader:  # ← automatically wrapped with progress
            loss = train_step(batch, model)
        return loss

Copyright (C) 2026 Matti Kaupenjohann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from tipi import helpers as _tipi_helpers


def progress_task(
    desc: str | None = None, progress_name: str = "overall"
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Auto-advancing progress decorator that works standalone or with pipeline.

    This decorator automatically wraps the first iterable parameter of a function
    with a progress bar. It seamlessly switches between standalone Rich progress
    and pipeline ProgressManager based on context.

    Standalone mode: Creates a Rich Progress context manager with status support
    Pipeline mode: Uses ProgressManager from _pipeline_context

    The decorator inspects the function's parameters to find an iterable (DataLoader,
    range, list, etc.) and wraps it with an auto-advancing iterator that updates
    the progress bar on each iteration.

    Status Updates (Optional):
        Functions can provide status updates by yielding tuples of (item, status_string):
        - yield batch  # Regular iteration, no status
        - yield batch, f"Loss: {loss:.4f}"  # With status update

    Args:
        desc: Task description. If None, uses the function name converted to title case.
        progress_name: Which ProgressManager bar to use in pipeline mode (default: "overall").
                      Only used when running in pipeline context.

    Returns:
        Decorated function that automatically tracks progress.

    Example:
        # Basic usage without status
        @progress_task(desc="Training Epoch")
        def train_epoch(self, dataloader, model):
            for batch in dataloader:  # ← automatically wrapped with progress
                loss = train_step(batch, model)
            return loss

        # With status updates (works in both standalone and pipeline mode)
        @progress_task(desc="Training Epoch", progress_name="train")
        def train_epoch(self, dataloader, model):
            for batch in dataloader:
                loss = train_step(batch, model)
                yield batch, f"Loss: {loss:.4f}"  # ← Status appears in progress bar
            return loss
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Inspect function signature to find iterable parameter
            sig = inspect.signature(func)
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                # If binding fails, run function as-is
                return func(*args, **kwargs)

            # Find first iterable parameter (heuristic: has __iter__ and __len__ or __next__)
            iterable = None
            iterable_param = None

            for param_name, param_value in bound.arguments.items():
                # Check if it's an iterable (but not a string)
                if param_value is not None and hasattr(param_value, "__iter__") and not isinstance(param_value, str):
                    iterable = param_value
                    iterable_param = param_name
                    break

            # If no iterable found, run function as-is
            if iterable is None or iterable_param is None:
                return func(*args, **kwargs)

            # Determine task description
            task_desc = desc or func.__name__.replace("_", " ").title()

            # Try to get total count
            total = None
            with contextlib.suppress(TypeError, AttributeError):
                # Some iterables don't support len()
                total = len(iterable)

            # Check if running in pipeline context
            if _tipi_helpers._pipeline_context:
                progress_mgr = _tipi_helpers._pipeline_context.get("progress_manager")
                if progress_mgr:
                    # Use ProgressManager from pipeline
                    return _run_with_progress_manager(
                        func, bound, iterable, iterable_param, progress_mgr, progress_name, task_desc, total
                    )

            # Standalone: use Rich Progress
            return _run_with_rich_progress(func, bound, iterable, iterable_param, task_desc, total)

        return wrapper

    return decorator


def _core_progress_runner(
    func: Callable[..., Any],
    bound: inspect.BoundArguments,
    iterable: Any,
    iterable_param: str,
    setup_task: Callable[[], Any],
    advance_task: Callable[[Any, str], None],
) -> Any:
    """The core logic for wrapping an iterable and executing a function."""
    task_id = setup_task()

    def advancing_iter() -> Any:
        for item in iterable:
            # Support optional status via tuple unpacking
            # User can yield: (data, "status") or just: data
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str):
                actual_item, status = item
                yield actual_item
                advance_task(task_id, status)
            else:
                yield item
                advance_task(task_id, "")

    # Here happens the magic!
    # we replace the iterable of the decorated function with a generator.
    # This way we are able to plant the advance_task method at the end of the loop.
    # This doesn't touch the original for loop.
    bound.arguments[iterable_param] = advancing_iter()
    return func(*bound.args, **bound.kwargs)


def _run_with_rich_progress(
    func: Callable[..., Any],
    bound: inspect.BoundArguments,
    iterable: Any,
    iterable_param: str,
    desc: str,
    total: int | None,
) -> Any:
    """Run function with standalone Rich Progress."""

    def advance_task(task_id: Any, status: str) -> None:
        progress.advance(task_id, 1)
        if status:
            progress.update(task_id, status=status)

    # Create progress with status column for standalone mode
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TextColumn("•"),
        TextColumn("{task.fields[status]}"),
        "•",
        TimeRemainingColumn(),
    )

    with progress:
        return _core_progress_runner(
            func,
            bound,
            iterable,
            iterable_param,
            setup_task=lambda: progress.add_task(desc, total=total, status=""),
            advance_task=advance_task,
        )


def _run_with_progress_manager(
    func: Callable[..., Any],
    bound: inspect.BoundArguments,
    iterable: Any,
    iterable_param: str,
    progress_mgr: Any,
    progress_name: str,
    desc: str,
    total: int | None,
) -> Any:
    """Run function with pipeline ProgressManager."""
    return _core_progress_runner(
        func,
        bound,
        iterable,
        iterable_param,
        setup_task=lambda: progress_mgr.add_task_to_progress(desc, total=total or 0, visible=True),
        advance_task=lambda tid, status: progress_mgr.advance(progress_name, tid, step=1.0, status=status),
    )


__all__ = [
    "progress_task",
]
