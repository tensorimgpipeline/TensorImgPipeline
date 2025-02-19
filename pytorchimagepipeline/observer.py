"""This module defines an Observer responsible for managing a pipeline of processes and handling
potential errors that occur during their execution.

Classes:
    Observer: Manages a pipeline of processes and handles errors that occur during.

Copyright (C) 2025 Matti Kaupenjohann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from pytorchimagepipeline.abstractions import AbstractObserver, Permanence, PipelineProcess
from pytorchimagepipeline.errors import (
    BuilderError,
    ErrorCode,
    ExecutionError,
    PermanenceKeyError,
    SweepNoConfigError,
)


class Observer(AbstractObserver):
    def __init__(self, permanences: dict[str, Permanence]):
        """
        Initializes the Observer with the given permanences.

        Args:
            permanences (dict[str, Permanence]): A dictionary mapping string keys to Permanence objects.
        """
        self._permanences = permanences
        self._processes: list[PipelineProcess] = []
        self._current_process: PipelineProcess | None = None

    def add_process(self, process: PipelineProcess) -> None:
        """Adds a process to the pipeline.

        Args:
            process (PipelineProcess): The process to add.
        """
        self._processes.append(process)

    def run_wandb(self) -> None:
        wandb_logger = self._permanences.get("wandb_logger", None)
        if wandb_logger:
            hyperparams: Permanence | dict[str, Any] = self._permanences.get("hyperparams", {})
            if not hyperparams:
                raise SweepNoConfigError()
            wandb_logger.create_sweep(hyperparams.hyperparams.get("sweep_configuration", {}))
            wandb_logger.create_sweep_agent(self.run)
        else:
            self.run()

    def _get_progress_decorator(self) -> callable:
        def empty_decorator(func):
            @wraps(func)
            def wrapper(total, *args, **kwargs):
                return func(0, total, None, *args, **kwargs)

        progress_manager = self._permanences.get("progress_manager", None)
        progress_decorator = progress_manager.progress_task if progress_manager else empty_decorator
        return progress_decorator

    def _get_inner_run(self):
        progress_decorator = self._get_progress_decorator()

        @progress_decorator("overall")
        def _inner_run(task_id, total, progress) -> None:
            for idx in range(total):
                process = self._processes[idx]
                self._current_process = process
                process_instance = process.get_instance(self)
                if not process_instance.skip():
                    error = process_instance.execute()
                    if error:
                        self._handle_error(error)
                self._current_process = None
                if progress:
                    progress.advance(task_id)

        return _inner_run

    def _get_inner_cleanup(self):
        progress_decorator = self._get_progress_decorator()

        @progress_decorator("cleanup")
        def _inner_cleanup(task_id, total, progress) -> None:
            for idx in range(total):
                progress.advance(task_id)
                permanence_keys = list(self._permanences.keys())
                permanence = self._permanences[permanence_keys[idx]]
                permanence.cleanup()

        return _inner_cleanup

    def run(self) -> None:
        """
        Executes each process in the list of processes.

        Iterates over the processes, sets the current process, and executes it.
        If an error occurs during the execution of a process, it handles the error.
        Resets the current process to None after each execution.

        Returns:
            None
        """

        progress_manager = self._permanences.get("progress_manager", None)
        wandb_logger = self._permanences.get("wandb_logger", None)
        if wandb_logger:
            wandb_logger.init_wandb()

        _inner_run = self._get_inner_run()
        _inner_cleanup = self._get_inner_cleanup()

        if progress_manager:
            with self._permanences["progress_manager"].live:
                _inner_run(len(self._processes))
                _inner_cleanup(len(self._permanences))
        else:
            _inner_run(len(self._processes))
            _inner_cleanup(len(self._permanences))

    def _handle_error(self, error: Exception) -> None:
        """
        Handles errors that occur during the execution of a process.

        Args:
            error (Exception): The exception that was raised.

        Raises:
            BuilderError: If the error is an instance of BuilderError.
            ExecutionError: If the error is not an instance of BuilderError,
                            raises an ExecutionError with the current process name and the original error.
        """
        if isinstance(error, BuilderError):
            raise error

        process_name = self._current_process.__class__.__name__
        raise ExecutionError(process_name, error)

    def get_permanence(self, name: str) -> Any:
        """
        Retrieve the permanence value associated with the given name.

        Args:
            name (str): The key name for which to retrieve the permanence value.
        Returns:
            Any: The permanence value associated with the given name.
        Raises:
            PermanenceKeyError: If the given name is not found in the permanences dictionary.
        """

        if name not in self._permanences:
            raise PermanenceKeyError(ErrorCode.PERMA_KEY, key=name)
        return self._permanences[name]
