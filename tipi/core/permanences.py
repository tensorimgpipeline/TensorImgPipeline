from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from logging import info
from pathlib import Path
from typing import Any

import torch
import wandb
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn
from wandb.wandb_run import Run

from tipi.abstractions import Permanence
from tipi.core.utils import create_color
from tipi.errors import ProgressNoMatch, SweepNoConfigError


class VRAMUsageError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("All devices are using more than 80% of VRAM")


@dataclass
class DeviceWithVRAM:
    device: torch.device
    vram_usage: float


class Device(Permanence):
    """
    Device class for managing CUDA device selection based on VRAM usage.

    This class inherits from the Permanence class and is responsible for calculating
    and setting the best available CUDA device for computation based on VRAM usage.

    Example TOML Config:
        ```toml
        [permanences.network]
        type = "Network"
        params = { model = "deeplabv3_resnet50", num_classes = 21, pretrained = true }
        ```

    Methods:
        __init__() -> None:

        _calculate_best_device() -> None:
            Raises VRAMUsageError if the VRAM usage of the best device exceeds 80%.

        cleanup() -> None:
            Cleans up the device instance. This method doesn't perform any cleanup operations.
    """

    def __init__(self) -> None:
        """
        Initializes the instance and calculates the best device for computation.
        """
        self._calculate_best_device()

    def _calculate_best_device(self) -> None:
        """
        Calculate and set the best available CUDA device based on VRAM usage.

        This method iterates over all available CUDA devices, calculates their VRAM usage,
        and selects the device with the lowest VRAM usage. If the VRAM usage of the best
        device exceeds 80%, a VRAMUsageError is raised.

        Raises:
            VRAMUsageError: If the VRAM usage of the best device exceeds 80%.

        Attributes:
            device (torch.device): The CUDA device with the lowest VRAM usage.
        """
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        vram_usage = [
            DeviceWithVRAM(
                device,
                torch.cuda.memory_reserved(device) / torch.cuda.get_device_properties(device).total_memory,
            )
            for device in devices
        ]

        best_device = min(vram_usage, key=lambda x: x.vram_usage)

        if best_device.vram_usage > 0.8:
            raise VRAMUsageError()

        self.device = best_device.device

    def cleanup(self) -> None:
        """
        Cleans up the device instance.

        This method doesn't perform any cleanup operations for the device instance.

        Returns:
            None: This method doesn't return any value.
        """
        pass


class NullProgressManager(Permanence):
    """
    A null progress manager that does not perform any progress tracking.

    This class is used if a manager does not provide a progress manager.
    """

    ...


class ProgressManager(Permanence):
    """
    Manages progress tracking for tasks using the Rich library.

    Example TOML Config:
        ```toml
        [permanences.progress_manager]
        type = "ProgressManager"
        param
        ```

    Attributes:
        progress_dict (dict): A dictionary to store progress objects.
        live (Live): A Live object to manage live updates.

    Methods:
        __init__(direct=True):
            Initializes the ProgressManager with an optional direct live update.

        _create_progress(color="#F55500"):
            Creates and returns a Progress object with specified color.

        _init_live():
            Initializes the live update group with the progress objects.

        progress_task(task_name, visible=False):
            A decorator to wrap functions for progress tracking.

        cleanup():
            not used
    """

    def __init__(self, console: Console | None = None, direct: bool = False) -> None:
        """
        Initializes the instance of the class.

        Args:
            direct (bool): If True, initializes live progress. Defaults to False.
        """
        self.console = console
        self.progress_dict = {
            "overall": self._create_progress(),
            "cleanup": self._create_progress(color="#FFFF55"),
            "result": self._create_progress(color="#5555FF"),
        }
        self.bar_colors: list[str] = []
        if direct:
            self.init_live()

    def add_progresses(self, progresses: list[dict[str, Any]]) -> None:
        """
        Add multiple progress objects to the progress_dict.
        Also re-initializes the live attribute.

        Args:
            names (dict): A dictionary of task names and their visibility status.
        """
        for progress in progresses:
            self.add_progress(**progress)
        self.init_live()

    def add_progress(self, name: str, with_status: bool = False) -> None:
        """
        Add a progress object to the progress_dict.

        Args:
            with_status (bool): Whether to include a status column. Defaults to False.
            color (str): The color to use for the progress bar. Default is "#F55500".

        Returns:
            Progress: A Progress object configured with the specified color.
        """
        self._get_bar_colors()
        color = create_color(self.bar_colors)
        self.bar_colors.append(color)
        progress = self._create_progress(color, with_status)
        self.progress_dict[name] = progress

    def _get_bar_colors(self) -> None:
        if len(self.bar_colors) == len(self.progress_dict):
            # All Bar colors are allready captured
            return
        self.bar_colors = []
        for progress in self.progress_dict.values():
            if not isinstance(progress.columns[1], BarColumn):
                raise TypeError(f"{progress.columns[1]=}")
            bar_column: BarColumn = progress.columns[1]
            self.bar_colors.append(str(bar_column.complete_style))

    def _create_progress(self, color: str = "#F55500", with_status: bool = False) -> Progress:
        """
        Create a progress bar with specified color.

        Args:
            color (str): The color to use for the progress bar. Default is "#F55500".

        Returns:
            Progress: A Progress object configured with the specified color.
        """
        status = ("•", TextColumn("{task.fields[status]}")) if with_status else []

        return Progress(
            TextColumn(f"[bold{color}]" + "{task.description}"),
            BarColumn(style="#333333", complete_style=color, finished_style="#22FF55"),
            TextColumn("({task.completed}/{task.total})"),
            *status,
            "•",
            TimeRemainingColumn(),
            console=self.console,
        )

    def init_live(self) -> None:
        """
        Initializes the live attribute with a Live object.
        This method creates a Group object using the values from the
        progress_dict attribute and then initializes the live attribute
        with a Live object that takes the Group object as an argument.
        """
        group = Group(*self.progress_dict.values())
        self.live = Live(group, console=self.console)

    def add_task_to_progress(self, task_description: str, total: int, visible: bool = False) -> int:
        """
        Add a task to the specified progress object.
        The string `task_description` will be evaluated by `_get_progress_for_task` to determine a
            matching progress

        Args:
            task_description (str): Description of the task to be added.
                This must match with a progress object name or a split of a progress object name.
            total (int): The total number of steps for the task.
            visible (bool): Define if the task is visible at creation. DEFAULT: False

        Returns:
            int: The task_id of the added task.
        """
        progress = self._get_progress_for_task(task_description)
        return progress.add_task(task_description, total=total, status="", visible=visible)

    def _toogle_visability(self, progress: Progress, task_id: int) -> None:
        visible = not progress._tasks[TaskID(task_id)].finished and progress._tasks[TaskID(task_id)].completed >= 0
        progress.update(TaskID(task_id), visible=visible)

    def advance(self, progress_name: str, task_id: int, step: float = 1.0, status: str = "") -> None:
        if progress_name not in self.progress_dict:
            raise ValueError(f"{progress_name=}")
        progress = self.progress_dict[progress_name]
        self._toogle_visability(progress, task_id)
        progress.advance(TaskID(task_id), step)
        progress.update(TaskID(task_id), status=status)
        self._toogle_visability(progress, task_id)

    def _get_progress_for_task(self, task_description: str) -> Progress:
        for key in self.progress_dict:
            if task_description == key:
                return self.progress_dict[key]
            keys = key.split("-")
            if task_description in keys:
                return self.progress_dict[key]
            else:
                continue
        raise RuntimeError(f"{task_description=}")

    def reset(self, progress_name: str) -> None:
        if progress_name not in self.progress_dict:
            raise ValueError(f"{progress_name=}")
        progress = self.progress_dict[progress_name]
        for task_id in self.progress_dict[progress_name]._tasks:
            progress.reset(task_id)

    def progress_task(self, task_name: str, visible: bool = True) -> Callable[..., Callable[..., Any]]:
        """
        Decorator for wrapping functions with progress tracking.

        The decorated function can receive any of these parameters:
        - total: The value passed when calling the decorated function
        - task_id: Injected by decorator, use with progress.advance(task_id)
        - progress: Injected by decorator, the Progress object for manual control

        The decorator inspects the function signature and only passes parameters it accepts.

        Usage examples:
            # Function that uses total
            @progress_manager.progress_task("processing")
            def process_items(total, task_id, progress):
                for i in range(total):
                    progress.advance(task_id)

            # Function that doesn't need total
            @progress_manager.progress_task("cleanup")
            def cleanup_all(task_id, progress):
                for item in items:  # Has its own iteration
                    progress.advance(task_id)

        Args:
            task_name: Name of the task/progress bar to use
            visible: Whether task should remain visible after completion

        Returns:
            Decorator function that wraps the target function with progress tracking
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            # Inspect function signature to see what parameters it accepts
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            @wraps(func)
            def wrapper(total: int, *args: Any, **kwargs: Any) -> Any:
                # Find matching progress bar
                progress_key = next((key for key in self.progress_dict if task_name.lower() in key.lower()), None)
                if not progress_key:
                    raise ProgressNoMatch(task_name)
                progress_obj = self.progress_dict[progress_key]

                # Add task to progress
                task_id = progress_obj.add_task(task_name, total=total)

                # Build kwargs based on what the function accepts
                func_kwargs: dict[str, Any] = {}
                if "total" in params:
                    func_kwargs["total"] = total
                if "task_id" in params:
                    func_kwargs["task_id"] = task_id
                if "progress" in params:
                    func_kwargs["progress"] = progress_obj

                # Call the function with only the parameters it needs
                result = func(*args, **func_kwargs, **kwargs)

                # Hide task when done if not visible
                progress_obj.update(task_id, visible=visible)
                return result

            return wrapper

        return decorator

    def cleanup(self) -> None:
        self.init_live()


@dataclass
class NullWandBManager(Permanence): ...


@dataclass
class WandBManager(Permanence):
    """
    WandBManager class for managing logging of metrics to Weights and Biases.

    This class inherits from the Permanence class and is responsible for logging metrics
    to Weights and Biases.

    Example TOML Config:
        ```toml
        [permanences.wandb_logger]
        type = "WandbLogger"
        params = {
            project = "my_project",
            entity = "my_entity",
            name = "my_run",
            tags = ["tag1", "tag2"],
            notes = "my_notes"
        }
        ```

    Methods:
        __init__(project, entity):
            Initializes the instance with the specified project and entity.

        log_metrics(metrics):
            Logs the specified metrics to Weights and Biases.

        cleanup():
            Cleans up the wandbLogger instance.
    """

    def __init__(
        self,
        project: str,
        entity: str,
        name: str = "",
        tags: list[str] | None = None,
        notes: str = "",
        count: int = 10,
    ) -> None:
        """
        Initializes the instance with the specified project and entity.

        Example TOML Config:
            ```toml
            [permanences.wandb_logger]
            type = "WandBLogger"
            params = {
                project = "DemoFull",
                entity = "demos",
                name = "<run title>",
                tags = ["tag1", "tag2"],
                notes = "<describe the run>"
            }
            ```

        Args:
            project (str): The project to log the metrics to.
            entity (str): The entity to log the metrics to.
        """

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

    def create_sweep_agent(self, func: Callable[[Any], Any]) -> None:
        if self.sweep_id:
            wandb.agent(self.sweep_id, function=func, entity=self.entity, project=self.project, count=self.count)

    def init_wandb(self) -> None:
        """
        Initializes the Weights and Biases run.

        This method initializes the Weights and Biases run with the specified project, entity,
        name, tags, and notes.
        """
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

    def _is_sweep_active(self) -> str | bool:
        if self.sweep_id is None:
            return False
        api = wandb.Api()
        try:
            sweep = api.sweep(f"{self.entity}/{self.project}/{self.sweep_id}")
        except wandb.errors.CommError as e:
            info(f"Error fetching sweep: {e}")
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

    def log_metrics(self, metrics: dict) -> None:
        """
        Logs the specified metrics to Weights and Biases.

        Args:
            metrics (dict): The metrics to log to Weights and Biases.
        """

        wandb.log(metrics, step=self.global_step)
        self.global_step += 1

    def cleanup(self) -> None:
        """
        Cleans up the wandbLogger instance.

        This method closes the Weights and Biases run.
        """
        # wandb.finish()


if __name__ == "__main__":
    from time import sleep

    progress_manager = ProgressManager()
    progress_manager.add_progress("test1")
    progress_manager.add_progress("test2")
    for progress in progress_manager.progress_dict.values():
        bar_col = progress.columns[1]
        if isinstance(bar_col, BarColumn):
            print(bar_col.complete_style)
    # [
    #     "#F55500",
    #     "#FFFF55",
    #     "#5555FF",
    #     "#55ffaa",
    #     "#80ff55",
    # ]
    progress_manager.add_progress("epoch", with_status=True)
    progress_manager.add_progress("train-val-test", with_status=True)
    task_epoch = TaskID(progress_manager.add_task_to_progress("epoch", total=3, visible=True))
    task_train = TaskID(progress_manager.add_task_to_progress("train", 100))
    task_val = TaskID(progress_manager.add_task_to_progress("val", 50))
    task_test = TaskID(progress_manager.add_task_to_progress("test", 10))
    progress_manager.init_live()
    with progress_manager.live:
        for _ in range(3):
            for _ in range(100):
                progress_manager.advance("train-val-test", task_train)
                sleep(0.1)
            for _ in range(50):
                progress_manager.advance("train-val-test", task_val)
                sleep(0.1)
            for _ in range(10):
                progress_manager.advance("train-val-test", task_test)
                sleep(0.1)
            progress_manager.reset("train-val-test")
            progress_manager.advance("epoch", task_epoch, status="NEW")
