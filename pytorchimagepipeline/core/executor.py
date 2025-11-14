from functools import wraps
from typing import Callable

from pytorchimagepipeline.abstractions import PipelineProcess
from pytorchimagepipeline.core.controller import PipelineController
from pytorchimagepipeline.errors import BuilderError, ExecutionError


class PipelineExecutor:
    """Executes pipeline processes with visualization.

    Responsibilities:
    - Execute processes in order
    - Apply progress bar decoration
    - Handle nested progress bars
    - Integrate with WandB logging
    - Error handling during execution
    """

    def __init__(self, controller: PipelineController):
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
        def _execute(task_id, total, progress):
            for _idx, process in self.controller.iterate_processes():
                if not process.skip():
                    try:
                        process.execute()
                    except Exception as error:
                        self._handle_error(process, error)
                if progress:
                    progress.advance(task_id)

        _execute(self.controller.get_process_count())

    def _run_cleanup(self) -> None:
        """Cleanup permanences with progress decoration."""
        decorator = self._get_progress_decorator()

        @decorator("cleanup")
        def _cleanup(task_id, total, progress):
            self.controller.cleanup()
            if progress:
                progress.advance(task_id)

        _cleanup(len(self.controller._permanences))

    def _get_progress_decorator(self) -> Callable:
        """Get progress decorator (or no-op if no progress manager)."""
        if self.progress_manager:
            return self.progress_manager.progress_task
        else:

            def empty_decorator(name):
                def wrapper(func):
                    @wraps(func)
                    def inner(total, *args, **kwargs):
                        return func(0, total, None, *args, **kwargs)

                    return inner

                return wrapper

            return empty_decorator

    def _handle_error(self, process: PipelineProcess, error: Exception) -> None:
        """Handle execution errors."""
        if isinstance(error, BuilderError):
            raise error
        raise ExecutionError(process.__class__.__name__, error)
