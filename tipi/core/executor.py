from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING

from tipi.abstractions import PipelineProcess
from tipi.errors import BuilderError, ExecutionError

if TYPE_CHECKING:
    from tipi.core.controller import PipelineController


class PipelineExecutor:
    """Executes pipeline processes with visualization.

    Responsibilities:
    - Execute processes in order
    - Apply progress bar decoration
    - Handle nested progress bars
    - Integrate with WandB logging
    - Error handling during execution
    """

    def __init__(self, controller: PipelineController) -> None:
        self.controller = controller
        self.progress_manager = controller.get_permanence("progress_manager", None)
        self.wandb_logger = controller.get_permanence("wandb_logger", None)

    def run(self) -> None:
        """Execute the full pipeline."""
        # Initialize WandB if present
        if self.wandb_logger:
            self.wandb_logger.init_wandb()

        # Execute processes with progress
        if self.progress_manager:
            with self.progress_manager.live:
                self._run_processes()
                self._run_cleanup()
        else:
            self._run_processes()
            self._run_cleanup()

    def _run_processes(self) -> None:
        """Execute all processes with progress decoration."""
        decorator = self._get_progress_decorator()

        @decorator("overall")
        def _execute(task_id: int, progress: object) -> None:
            for _idx, process in self.controller.iterate_processes():
                if not process.skip():
                    try:
                        process.execute()
                    except Exception as error:
                        self._handle_error(process, error)
                if progress:
                    progress.advance(task_id)  # type: ignore[attr-defined]

        _execute(self.controller.get_process_count())  # Passed to decorator for progress bar max

    def _run_cleanup(self) -> None:
        """Cleanup permanences with progress decoration."""
        decorator = self._get_progress_decorator()

        @decorator("cleanup")
        def _cleanup(task_id: int, progress: object) -> None:
            for permanence in self.controller.iterate_permanences():
                permanence.cleanup()
                if progress:
                    progress.advance(task_id)  # type: ignore[attr-defined]

        _cleanup(self.controller.get_permanence_count())  # Passed to decorator for progress bar max

    def _get_progress_decorator(self) -> Callable[..., Callable[..., Callable[..., None]]]:
        """Get progress decorator (or no-op if no progress manager)."""
        if self.progress_manager:
            return self.progress_manager.progress_task  # type: ignore[no-any-return]
        else:

            def empty_decorator(name: str) -> Callable[..., Callable[..., None]]:
                def wrapper(func: Callable[..., None]) -> Callable[..., None]:
                    # Inspect to see what parameters the function accepts
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())

                    @wraps(func)
                    def inner(total: int, *args: object, **kwargs: object) -> None:
                        # Build kwargs based on what the function accepts
                        func_kwargs: dict[str, object] = {}
                        if "total" in params:
                            func_kwargs["total"] = total
                        if "task_id" in params:
                            func_kwargs["task_id"] = 0
                        if "progress" in params:
                            func_kwargs["progress"] = None

                        func(*args, **func_kwargs, **kwargs)

                    return inner

                return wrapper

            return empty_decorator

    def _handle_error(self, process: PipelineProcess, error: Exception) -> None:
        """Handle execution errors."""
        if isinstance(error, BuilderError):
            raise error
        raise ExecutionError(process.__class__.__name__, error)
