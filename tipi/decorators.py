"""Decorators for smooth script-to-pipeline transition.

This module provides decorators that allow regular Python functions to
become pipeline processes with minimal code changes.

Usage:
    @pipeline_process
    def train(epochs: int = 10):
        '''This function can run standalone OR as a pipeline process!'''
        for epoch in range(epochs):
            loss = train_epoch()
        return loss

Copyright (C) 2025 Matti Kaupenjohann

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
from typing import Any, Protocol, cast

from tipi import helpers as _tipi_helpers
from tipi.abstractions import PipelineProcess


class ProcessWrapper(Protocol):
    """Protocol for wrapped functions that have a PipelineProcess class attached."""

    PipelineProcess: type[PipelineProcess]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow the wrapper to be called as a normal function."""
        ...


def pipeline_process(
    name: str | None = None,
    skip_if: Callable[[], bool] | None = None,
) -> Callable[[Callable[..., Any]], ProcessWrapper]:
    """Decorator to convert a function into a pipeline process.

    The decorated function can run standalone or as part of a pipeline.
    When used in a pipeline, it automatically gets access to permanences.

    Args:
        name: Optional name for the process (defaults to function name).
        skip_if: Optional function that returns True if process should skip.

    Example:
        @pipeline_process
        def train(epochs: int = 10):
            '''Can run standalone or in pipeline!'''
            device = helpers.device_manager.get_device()
            model = MyModel().to(device)

            for epoch in helpers.progress_bar(range(epochs), desc="Training"):
                loss = train_step(model)
                helpers.logger.log({"loss": loss})

            return loss

        # Run standalone
        if __name__ == "__main__":
            train(epochs=5)

        # Or register in pipeline config:
        # [processes.training]
        # type = "train"  # Uses function name
        # params = { epochs = 10 }
    """

    def decorator(func: Callable[..., Any]) -> ProcessWrapper:
        process_name = name or func.__name__

        # Get function signature for parameter handling
        sig = inspect.signature(func)
        params = {
            name: param.default if param.default != inspect.Parameter.empty else None
            for name, param in sig.parameters.items()
        }

        # Create a PipelineProcess class dynamically
        class FunctionProcess(PipelineProcess):
            def __init__(self, controller: Any, force: bool = False, **kwargs: Any) -> None:
                self.controller = controller
                self.force = force
                # Store function parameters
                for param_name, default_value in params.items():
                    setattr(self, param_name, kwargs.get(param_name, default_value))

            def execute(self) -> None:
                """Execute the wrapped function."""
                try:
                    # Provide permanences to helpers
                    context = {
                        "progress_manager": self.controller.get_permanence("progress_manager", None),
                        "wandb_logger": self.controller.get_permanence("wandb_logger", None),
                        "device": self.controller.get_permanence("device", None),
                    }
                    _tipi_helpers.set_pipeline_context(context)

                    # Call the original function with stored parameters
                    func_params = {name: getattr(self, name) for name in params}
                    func(**func_params)
                finally:
                    _tipi_helpers.clear_pipeline_context()

            def skip(self) -> bool:
                """Check if process should be skipped."""
                if skip_if:
                    return skip_if()
                return False

        # Preserve function metadata
        FunctionProcess.__name__ = process_name
        FunctionProcess.__doc__ = func.__doc__

        # Allow function to still be called normally
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        # Attach the process class for pipeline registration
        wrapper.PipelineProcess = FunctionProcess  # type: ignore[attr-defined]

        return cast(ProcessWrapper, wrapper)

    return decorator


def pipeline_script(
    project: str | None = None,
    config_file: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to make a script runnable as standalone or pipeline.

    This decorator allows an entire script to be run either:
    1. Directly: python my_script.py
    2. As pipeline: tipi run my_pipeline

    Args:
        project: Project name for WandB (if using).
        config_file: Optional config file path.

    Example:
        @pipeline_script(project="my_project")
        def main():
            '''Main training script'''
            helpers.logger.init(project="my_project")

            model = MyModel()
            for epoch in helpers.progress_bar(range(10)):
                train_step(model)

        if __name__ == "__main__":
            main()
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if running in pipeline mode
            import sys

            if "--pipeline-mode" in sys.argv:
                # Running as part of pipeline
                # Pipeline will handle initialization
                pass
            else:
                # Running standalone
                print(f"Running {func.__name__} in standalone mode...")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def progress_task(
    desc: str | None = None, progress_name: str = "overall"
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Auto-advancing progress decorator that works standalone or with pipeline.

    This decorator automatically wraps the first iterable parameter of a function
    with a progress bar. It seamlessly switches between standalone Rich progress
    and pipeline ProgressManager based on context.

    Standalone mode: Creates a Rich Progress context manager
    Pipeline mode: Uses ProgressManager from _pipeline_context

    The decorator inspects the function's parameters to find an iterable (DataLoader,
    range, list, etc.) and wraps it with an auto-advancing iterator that updates
    the progress bar on each iteration.

    Args:
        desc: Task description. If None, uses the function name converted to title case.
        progress_name: Which ProgressManager bar to use in pipeline mode (default: "overall").
                      Only used when running in pipeline context.

    Returns:
        Decorated function that automatically tracks progress.

    Example:
        # Standalone usage
        @progress_task(desc="Training Epoch")
        def train_epoch(self, dataloader, model):
            for batch in dataloader:  # ← automatically wrapped with progress
                loss = train_step(batch, model)
            return loss

        # Pipeline usage (same code, automatically uses ProgressManager)
        @progress_task(desc="Training Epoch", progress_name="train")
        def train_epoch(self, dataloader, model):
            for batch in dataloader:  # ← uses ProgressManager from pipeline
                loss = train_step(batch, model)
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


def _run_with_rich_progress(
    func: Callable[..., Any],
    bound: inspect.BoundArguments,
    iterable: Any,
    iterable_param: str,
    desc: str,
    total: int | None,
) -> Any:
    """Run function with standalone Rich Progress."""
    from rich.progress import Progress

    with Progress() as progress:
        task_id = progress.add_task(desc, total=total)

        def advancing_iter() -> Any:
            for item in iterable:
                yield item
                progress.advance(task_id, 1)

        # Replace iterable in bound arguments
        bound.arguments[iterable_param] = advancing_iter()

        # Call function with modified arguments
        return func(*bound.args, **bound.kwargs)


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
    # Add task to the named progress bar
    task_id = progress_mgr.add_task_to_progress(desc, total=total or 0, visible=True)

    def advancing_iter() -> Any:
        for item in iterable:
            yield item
            progress_mgr.advance(progress_name, task_id, step=1.0)

    # Replace iterable in bound arguments
    bound.arguments[iterable_param] = advancing_iter()

    # Call function with modified arguments
    return func(*bound.args, **bound.kwargs)


__all__ = [
    "pipeline_process",
    "pipeline_script",
    "progress_task",
]
