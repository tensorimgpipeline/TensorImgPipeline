"""Decorators for automatic progress tracking in pipeline functions.

This module provides the @progress_task decorator that wraps generator functions
with a progress bar. Signal progress by yielding an ``Update`` object::

    @progress_task(desc="Training Epoch")
    def train_epoch(dataloader, model):
        for batch in dataloader:
            yield Update("forward pass…", advance=0)   # status only
            loss = train_step(batch, model)
            yield Update(f"Loss: {loss:.4f}")          # advance + status
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
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from tipi import helpers as _tipi_helpers


@dataclass
class Update:
    """Progress signal yielded from a ``@progress_task`` generator function.

    Controls both the status message and how far the bar advances.

    Args:
        status: Message shown next to the progress bar. Defaults to no update.
        advance: How many steps to advance the bar. Use ``0`` to update the
                 status without moving the bar forward. Defaults to ``1``.

    Examples::

        yield Update()                          # tick bar, no message
        yield Update(f"loss={loss:.4f}")        # tick bar + show status
        yield Update("forward pass…", advance=0) # status only, bar stays
        yield Update("chunk done", advance=512) # advance by 512 steps
    """

    status: str = ""
    advance: int = field(default=1)


def progress_task(
    desc: str | None = None, progress_name: str = "overall"
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Progress-tracking decorator for generator functions.

    Wraps a generator function so that each ``yield`` advances a progress bar.
    An optional string value emitted by ``yield`` is shown as the status message.
    The ``return`` value of the generator is transparently returned to the caller.

    Non-generator functions are executed unchanged (no progress bar).

    Works in both standalone mode (Rich Progress) and pipeline mode (ProgressManager).

    Args:
        desc: Task description. Defaults to the function name in title case.
        progress_name: ProgressManager bar name in pipeline mode (default: "overall").

    Example:
        # Basic usage — one yield per batch advances the bar
        @progress_task(desc="Training Epoch")
        def train_epoch(dataloader, model):
            for batch in dataloader:
                loss = train_step(batch, model)
                yield Update(f"Loss: {loss:.4f}")
            return loss
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Cache task_id per (progress_manager instance, progress_name).
        # Lets repeated calls (e.g. per-epoch) reuse and reset the same bar
        # instead of creating a new task every time.
        _task_cache: dict[tuple[int, str], int] = {}

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not inspect.isgeneratorfunction(func):
                return func(*args, **kwargs)

            task_desc = desc or func.__name__.replace("_", " ").title()
            total = _infer_total(func, args, kwargs)

            if _tipi_helpers._pipeline_context:
                progress_mgr = _tipi_helpers._pipeline_context.get("progress_manager")
                if progress_mgr:
                    cache_key = (id(progress_mgr), progress_name)
                    if cache_key in _task_cache:
                        task_id = _task_cache[cache_key]
                        progress_mgr.progress_dict[progress_name].reset(task_id, total=total or 0)
                        # _toogle_visability hid the task when it reached N/N on the
                        # previous call.  Rich's reset() doesn't restore visibility, so
                        # we do it explicitly here so the bar is visible from the very
                        # first step of this call rather than only after the first yield.
                        progress_mgr.progress_dict[progress_name].update(task_id, visible=True)
                    else:
                        task_id = progress_mgr.add_task_to_progress(
                            task_desc, total=total or 0, visible=True, progress_name=progress_name
                        )
                        _task_cache[cache_key] = task_id
                    return _drive_with_progress_manager(func, args, kwargs, progress_mgr, progress_name, task_id)

            return _drive_with_rich_progress(func, args, kwargs, task_desc, total)

        return wrapper

    return decorator


def _infer_total(func: Callable[..., Any], args: tuple, kwargs: dict) -> int | None:
    """Return len() of the first iterable argument, or None if unavailable."""
    sig = inspect.signature(func)
    with contextlib.suppress(TypeError):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for val in bound.arguments.values():
            if val is not None and hasattr(val, "__iter__") and not isinstance(val, str):
                with contextlib.suppress(TypeError, AttributeError):
                    return len(val)
    return None


def _drive_with_rich_progress(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    desc: str,
    total: int | None,
) -> Any:
    """Drive a generator function with a standalone Rich Progress bar."""
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
        task_id = progress.add_task(desc, total=total, status="")
        gen = func(*args, **kwargs)
        try:
            while True:
                val = next(gen)
                step = val if isinstance(val, Update) else Update()
                progress.advance(task_id, step.advance)
                if step.status:
                    progress.update(task_id, status=step.status)
        except StopIteration as e:
            return e.value


def _drive_with_progress_manager(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    progress_mgr: Any,
    progress_name: str,
    task_id: int,
) -> Any:
    """Drive a generator function using a pipeline ProgressManager."""
    gen = func(*args, **kwargs)
    try:
        while True:
            val = next(gen)
            step = val if isinstance(val, Update) else Update()
            progress_mgr.advance(progress_name, task_id, step=float(step.advance), status=step.status)
    except StopIteration as e:
        return e.value


__all__ = [
    "Update",
    "progress_task",
]
