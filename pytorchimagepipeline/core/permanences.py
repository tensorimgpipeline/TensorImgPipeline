import os
from dataclasses import dataclass
from functools import wraps
from logging import info, warning
from pathlib import Path
from typing import Any, Optional

import torch
from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

import wandb
from pytorchimagepipeline.abstractions import Permanence
from pytorchimagepipeline.errors import SweepNoConfigError


class VRAMUsageError(RuntimeError):
    def __init__(self):
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

    def __init__(self, console=None, direct=False):
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
        if direct:
            self._init_live()

    def _create_progress(self, color="#F55500", with_status=False):
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

    def _init_live(self):
        """
        Initializes the live attribute with a Live object.
        This method creates a Group object using the values from the
        progress_dict attribute and then initializes the live attribute
        with a Live object that takes the Group object as an argument.
        """
        group = Group(*self.progress_dict.values())
        self.live = Live(group, console=self.console)

    def progress_task(self, task_name, visible=False):
        """
        A decorator to add a progress tracking task to a function.

        Args:
            task_name (str): The name of the task to be tracked.
            visible (bool, optional): Whether the task should be visible when done. Defaults to False.

        Returns:
            function: The decorated function with progress tracking.

        The decorated function should have the following signature:
            func(task_id, total, progress, *args, **kwargs)

        The decorator will:
            - Create a progress task if it does not already exist.
            - Add the task to the progress tracker.
            - Call the decorated function with the task_id, total, progress, and any additional arguments.
            - Update the task visibility when done.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(total, *args, **kwargs):
                progress_key = next((key for key in self.progress_dict if task_name.lower() in key), None)
                if progress_key is None:
                    raise NotImplementedError(f"Progress for {task_name} not found")
                progress = self.progress_dict[progress_key]
                # Add task to progress
                task_id = progress.add_task(task_name, total=total, status="")

                # Call the function with task_id
                result = func(task_id, total, progress, *args, **kwargs)

                if not progress.finished:
                    warning(UserWarning("Progress not completed, Wrong total provided or advance steps to small"))

                # Hide task when done
                progress.update(task_id, visible=visible)
                return result

            return wrapper

        return decorator

    def cleanup(self):
        self._init_live()


@dataclass
class WandBLogger(Permanence):
    """
    WandbLogger class for managing logging of metrics to Weights and Biases.

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
        tags: Optional[list[str]] = None,
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
                project = "Sam2Segnet",
                entity = "lit-rvc",
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

        self.sweep_id = None

        self.run_ids = []

        self.global_step = 0

    def create_sweep(self, config: dict[str, Any]) -> None:
        if not config:
            raise SweepNoConfigError()
        self._read_cached_sweep_id()
        if not self._is_sweep_active():
            self.sweep_id = wandb.sweep(config, entity=self.entity, project=self.project)
        if self.sweep_id is not None:
            self._write_cache_sweep_id()

    def create_sweep_agent(self, func: callable) -> None:
        self.agent = wandb.agent(
            self.sweep_id, function=func, entity=self.entity, project=self.project, count=self.count
        )

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

    def _is_sweep_active(self):
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
