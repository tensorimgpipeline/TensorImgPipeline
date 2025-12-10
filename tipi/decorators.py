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

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, Protocol, cast

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
                # Set pipeline context so helpers work
                from tipi.helpers import (
                    clear_pipeline_context,
                    set_pipeline_context,
                )

                try:
                    # Provide permanences to helpers
                    context = {
                        "progress_manager": self.controller.get_permanence("progress_manager", None),
                        "wandb_logger": self.controller.get_permanence("wandb_logger", None),
                        "device": self.controller.get_permanence("device", None),
                    }
                    set_pipeline_context(context)

                    # Call the original function with stored parameters
                    func_params = {name: getattr(self, name) for name in params}
                    func(**func_params)
                finally:
                    clear_pipeline_context()

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


__all__ = [
    "pipeline_process",
    "pipeline_script",
]
