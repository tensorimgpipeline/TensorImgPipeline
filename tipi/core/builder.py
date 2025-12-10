"""This module provides the implementation of the PipelineBuilder class, which is responsible for
building and configuring a pipeline of processes and permanences for the TensorImgPipeline project.

The PipelineBuilder class allows for the registration of classes, loading of configuration files,
validation of configuration sections, and construction of the complete pipeline. It handles errors
related to configuration loading, class instantiation, and process addition.

Classes:
    PipelineBuilder: A class to build and configure a pipeline of processes and permanences.

Functions:
    get_objects_for_pipeline(pipeline_name: str) -> dict[str, type]: Retrieves and combines objects
        to be registered for a given pipeline.

Usage Example:

#TODO: Add usage example


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

import contextlib
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from tomllib import TOMLDecodeError
from tomllib import load as toml_load
from typing import TYPE_CHECKING, Any, cast

from tipi.abstractions import Permanence, PipelineProcess
from tipi.errors import (
    ConfigInvalidTomlError,
    ConfigNotFoundError,
    ConfigPermissionError,
    ConfigSectionError,
    InstTypeError,
    RegistryError,
)
from tipi.paths import get_path_manager

if TYPE_CHECKING:
    from tipi.core.controller import PipelineController


@dataclass
class ProcessWithParams:
    process: type[PipelineProcess]
    params: dict[str, Any]

    def get_instance(self, controller: PipelineController) -> PipelineProcess:
        if "force" not in self.params:
            self.params["force"] = False
        return self.process(controller, **self.params)


class PipelineBuilder:
    """Builds pipeline components from configuration."""

    def __init__(self) -> None:
        """Initialize the builder with empty registries."""
        self._registry: dict[str, type[Permanence] | type[PipelineProcess]] = {}
        self._config: dict[str, Any] = {}
        self._config_path: Path | None = None

    def build(self) -> tuple[dict[str, Permanence], list[ProcessWithParams]]:
        """Construct permanences and process specifications.

        Returns:
            Tuple of (permanences_dict, process_specs_list)

        Raises:
            ConfigSectionError: If config sections are invalid
            InstTypeError: If permanence/process instantiation fails
            RegistryError: If class not found in registry
        """
        # First, create core permanences from pipeline flags
        permanences = self._build_core_permanences()

        # Then add user-defined permanences (can override core)
        user_permanences = self._build_permanences()
        permanences.update(user_permanences)

        # Build processes
        processes = self._build_processes()
        return permanences, processes

    def register_class(self, name: str, class_type: type) -> None:
        """Register a permanence or process class.

        Args:
            name: Name to register the class under
            class_type: The class to register

        Raises:
            RegistryError: If registration fails or class is invalid
        """
        if not isinstance(class_type, type):
            raise RegistryError(f"Cannot register {name}: {class_type} is not a class")

        # Validate that it's a Permanence or PipelineProcess
        if not (issubclass(class_type, Permanence) or issubclass(class_type, PipelineProcess)):
            raise RegistryError(
                f"Cannot register {name}: {class_type.__name__} must be a subclass of Permanence or PipelineProcess"
            )

        self._registry[name] = class_type

    def load_config(self, path: Path) -> None:
        """Load configuration from file.

        Args:
            path: Path to the TOML configuration file

        Raises:
            ConfigNotFoundError: If config file doesn't exist
            ConfigInvalidTomlError: If TOML parsing fails
            ConfigPermissionError: If file can't be read
        """
        self._config_path = path

        # Check if file exists
        if not path.exists():
            raise ConfigNotFoundError(f"Configuration file not found: {path}")

        # Check if we can read it
        if not os.access(path, os.R_OK):
            raise ConfigPermissionError(f"Cannot read configuration file: {path}")

        # Load and parse TOML
        try:
            with open(path, "rb") as f:
                self._config = toml_load(f)
        except TOMLDecodeError as e:
            raise ConfigInvalidTomlError(f"Invalid TOML in {path}: {e}") from e
        except Exception as e:
            raise ConfigPermissionError(f"Error reading {path}: {e}") from e

        # Validate config structure
        if "permanences" not in self._config and "processes" not in self._config and "pipeline" not in self._config:
            raise ConfigSectionError(
                f"Configuration file {path} must contain at least one of "
                f"'pipeline', 'permanences' or 'processes' sections"
            )

    def _build_core_permanences(self) -> dict[str, Permanence]:
        """Build core framework permanences from pipeline flags.

        Core permanences can be enabled via simple flags in [pipeline] section:
        - enable_progress: Creates a ProgressManager
        - enable_wandb: Creates a WandBLogger

        Returns:
            Dictionary mapping core permanence names to instances

        Raises:
            InstTypeError: If core permanence instantiation fails
            ConfigSectionError: If pipeline section is invalid
        """
        core_permanences: dict[str, Permanence] = {}

        # Get pipeline configuration section
        pipeline_config = self._config.get("pipeline", {})
        if not isinstance(pipeline_config, dict):
            raise ConfigSectionError("'pipeline' section must be a table")

        # Check if user explicitly defined these in permanences section
        user_permanences = self._config.get("permanences", {})

        # Build ProgressManager if enabled and not explicitly overridden
        if (
            pipeline_config.get("enable_progress", False)
            and "progress_manager" not in user_permanences
            and "ProgressManager" in self._registry
        ):
            try:
                # Create ProgressManager with direct=True to initialize .live
                perm_class = cast(type[Permanence], self._registry["ProgressManager"])
                progress_manager = perm_class(direct=True)  # type: ignore[call-arg]
                core_permanences["progress_manager"] = progress_manager
            except Exception as e:
                raise InstTypeError(f"Failed to create ProgressManager: {e}") from e

        # Build WandBLogger if enabled and not explicitly overridden
        if (
            pipeline_config.get("enable_wandb", False)
            and "wandb_logger" not in user_permanences
            and "WandBManager" in self._registry
        ):
            try:
                # WandBManager might need config parameters
                wandb_config = pipeline_config.get("wandb", {})
                perm_class = cast(type[Permanence], self._registry["WandBManager"])
                wandb_logger = perm_class(**wandb_config)
                core_permanences["wandb_logger"] = wandb_logger
            except Exception as e:
                raise InstTypeError(f"Failed to create WandBManager: {e}") from e

        return core_permanences

    def _build_permanences(self) -> dict[str, Permanence]:
        """Build permanence instances from config.

        Returns:
            Dictionary mapping permanence names to instances

        Raises:
            InstTypeError: If permanence instantiation fails
            ConfigSectionError: If permanences section is invalid
        """
        permanences: dict[str, Permanence] = {}

        permanences_config = self._config.get("permanences", {})
        if not isinstance(permanences_config, dict):
            raise ConfigSectionError("'permanences' section must be a dict")

        for name, perm_config in permanences_config.items():
            if not isinstance(perm_config, dict):
                raise ConfigSectionError(f"Permanence '{name}' configuration must be a dict")

            if "type" not in perm_config:
                raise ConfigSectionError(f"Permanence '{name}' missing required 'type' field")

            perm_type_name = perm_config["type"]

            # Look up the class in registry
            if perm_type_name not in self._registry:
                raise RegistryError(
                    f"Permanence type '{perm_type_name}' for '{name}' not registered. "
                    f"Available types: {list(self._registry.keys())}"
                )

            perm_class = self._registry[perm_type_name]

            # Verify it's actually a Permanence
            if not issubclass(perm_class, Permanence):
                raise InstTypeError(f"Registered class '{perm_type_name}' is not a Permanence subclass")

            # Extract constructor parameters (everything except 'type')
            perm_params = {k: v for k, v in perm_config.items() if k != "type"}

            # Instantiate the permanence
            try:
                permanence = perm_class(**perm_params)
                permanences[name] = permanence
            except TypeError as e:
                raise InstTypeError(f"Failed to instantiate permanence '{name}' of type '{perm_type_name}': {e}") from e
            except Exception as e:
                raise InstTypeError(f"Error creating permanence '{name}' of type '{perm_type_name}': {e}") from e

        return permanences

    def _build_processes(self) -> list[ProcessWithParams]:
        """Build process specifications from config.

        Returns:
            List of ProcessWithParams specifications

        Raises:
            InstTypeError: If process instantiation fails
            ConfigSectionError: If processes section is invalid
        """
        process_specs: list[ProcessWithParams] = []

        processes_config = self._config.get("processes", {})
        if not isinstance(processes_config, dict):
            raise ConfigSectionError("'processes' section must be a table")

        for name, proc_config in processes_config.items():
            if not isinstance(proc_config, dict):
                raise ConfigSectionError(f"Process '{name}' configuration must be a table")

            if "type" not in proc_config:
                raise ConfigSectionError(f"Process '{name}' missing required 'type' field")

            proc_type_name = proc_config["type"]

            # Look up the class in registry
            if proc_type_name not in self._registry:
                raise RegistryError(
                    f"Process type '{proc_type_name}' for '{name}' not registered. "
                    f"Available types: {list(self._registry.keys())}"
                )

            proc_class = self._registry[proc_type_name]

            # Verify it's actually a PipelineProcess
            if not issubclass(proc_class, PipelineProcess):
                raise InstTypeError(f"Registered class '{proc_type_name}' is not a PipelineProcess subclass")

            # Extract parameters (everything except 'type')
            proc_params = {k: v for k, v in proc_config.items() if k != "type"}

            # Create ProcessWithParams spec (not instantiated yet)
            process_spec = ProcessWithParams(process=proc_class, params=proc_params)
            process_specs.append(process_spec)

        return process_specs


def get_objects_for_pipeline(
    pipeline_name: str,
) -> dict[str, type[Permanence] | type[PipelineProcess]]:
    """
    Retrieves and combines objects to be registered for a given pipeline.

    Args:
        pipeline_name (str): The name of the pipeline for which to retrieve objects.

    Returns:
        dict[str, type]: A dictionary containing the combined objects from
                         `permanences_to_register` and `processes_to_register`
                         of the specified pipeline module, with both instance names
                         and class names as keys.

    Raises:
        ModuleNotFoundError: If the pipeline module cannot be found.
    """
    # Try built-in pipelines first
    full_module_name = "tipi.pipelines." + pipeline_name
    if pipeline_name == "core":
        full_module_name = "tipi." + pipeline_name

    module = None

    # Attempt 1: Try built-in pipelines
    with contextlib.suppress(ModuleNotFoundError):
        module = importlib.import_module(full_module_name)

    # Attempt 2: Try loading from user projects directory (symlinked projects)
    if module is None:
        path_manager = get_path_manager()
        module = path_manager.import_project_module(pipeline_name)

    # If all attempts failed, raise an error
    if module is None:
        raise ModuleNotFoundError(
            f"Pipeline '{pipeline_name}' not found. "
            f"Tried built-in module '{full_module_name}' and user projects directory."
        )

    # Get the registries
    if not hasattr(module, "permanences_to_register") or not hasattr(module, "processes_to_register"):
        raise AttributeError(
            f"Module '{pipeline_name}' must define 'permanences_to_register' and 'processes_to_register' dictionaries"
        )

    # Combine the registries and also register by class name
    combined: dict[str, type[Permanence] | type[PipelineProcess]] = {}

    # Add permanences with both instance names and class names
    for cls in module.permanences_to_register:
        combined[cls.__name__] = cls  # Class name (e.g., 'ConfigPermanence')

    # Add processes with both instance names and class names
    for cls in module.processes_to_register:
        combined[cls.__name__] = cls  # Class name (e.g., 'LoadDataProcess')

    return combined


# Usage Example # TODO Might be broken. Later. Not important!
if __name__ == "__main__":
    # Example usage of the PipelineBuilder

    # Retrieve objects to be registered for the pipeline
    core_objects = get_objects_for_pipeline("core")
    pipeline_objects = get_objects_for_pipeline("DemoFull")

    objects = core_objects | pipeline_objects

    # Initialize the PipelineBuilder
    builder = PipelineBuilder()

    # Register each class in the builder
    for key in objects:
        builder.register_class(key, objects[key])

    # Load the configuration file
    builder.load_config(Path("DemoFull/pipeline_config.toml"))

    # Build the pipeline
    permanences, process_specs = builder.build()

    # Create controller
    controller = PipelineController(permanences, process_specs)

    # Run the pipeline using executor
    from tipi.core.executor import PipelineExecutor

    executor = PipelineExecutor(controller)
    executor.run()
