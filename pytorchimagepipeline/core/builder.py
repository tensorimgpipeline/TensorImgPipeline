# type: ignore
# ruff: noqa
"""This module provides the implementation of the PipelineBuilder class, which is responsible for
building and configuring a pipeline of processes and permanences for the PytorchImagePipeline project.

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

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tomllib import TOMLDecodeError  # type ignore[import-not-found]
    from tomllib import load as toml_load  # type ignore[import-not-found]
except ImportError:
    try:
        from tomli import TOMLDecodeError  # type: ignore  # noqa: PGH003  # type: ignore[unused-import]
        from tomli import load as toml_load  # type: ignore  # noqa: PGH003 # type: ignore[unused-import]
    except ImportError:
        sys.exit("Error: This program requires either tomllib or tomli but neither is available")

from pytorchimagepipeline.abstractions import Permanence, PipelineProcess
from pytorchimagepipeline.errors import (
    ConfigInvalidTomlError,
    ConfigNotFoundError,
    ConfigPermissionError,
    ConfigSectionError,
    InstTypeError,
    RegistryError,
    RegistryParamError,
)
from pytorchimagepipeline.core.controller import PipelineController


@dataclass
class ProcessWithParams:
    process: PipelineProcess
    params: dict[str, Any]

    def get_instance(self, controller: PipelineController) -> PipelineProcess:
        if "force" not in self.params:
            self.params["force"] = False
        return self.process(controller, **self.params)


class PipelineBuilder:
    """Builds pipeline components from configuration."""

    def build(self) -> tuple[dict[str, Permanence], list[ProcessWithParams]]:
        """Construct permanences and process specifications.

        Returns:
            Tuple of (permanences_dict, process_specs_list)

        Raises:
            Various exceptions from _build_permanences and _build_processes
        """
        permanences = self._build_permanences()
        processes = self._build_processes()
        return permanences, processes

    def register_class(self, name: str, class_type: type) -> None:
        """Register a permanence or process class.

        Raises:
            RegistryError: If registration fails
        """
        raise NotImplementedError

    def load_config(self, path: Path) -> None:
        """Load configuration from file.

        Raises:
            ConfigNotFoundError: If config file doesn't exist
            ConfigInvalidTomlError: If TOML parsing fails
            ConfigPermissionError: If file can't be read
        """
        raise NotImplementedError

    def _build_permanences(self) -> dict[str, Permanence]:
        """Build permanence instances from config.

        Raises:
            InstTypeError: If permanence instantiation fails
        """
        raise NotImplementedError

    def _build_processes(self) -> list[ProcessWithParams]:
        """Build process specifications from config.

        Raises:
            InstTypeError: If process instantiation fails
        """
        raise NotImplementedError


def get_objects_for_pipeline(pipeline_name: str) -> dict[str, type]:
    """
    Retrieves and combines objects to be registered for a given pipeline.

    Args:
        pipeline_name (str): The name of the pipeline for which to retrieve objects.

    Returns:
        dict[str, type]: A dictionary containing the combined objects from
                         `permanences_to_register` and `processes_to_register`
                         of the specified pipeline module.

    Raises:
        ModuleNotFoundError: If the pipeline module cannot be found.
    """
    full_module_name = "pytorchimagepipeline.pipelines." + pipeline_name
    if pipeline_name == "core":
        full_module_name = "pytorchimagepipeline." + pipeline_name
    module = importlib.import_module(full_module_name)
    return module.permanences_to_register | module.processes_to_register


# Usage Example
if __name__ == "__main__":
    # Example usage of the PipelineBuilder

    # Retrieve objects to be registered for the pipeline
    core_objects = get_objects_for_pipeline("core")
    pipeline_objects = get_objects_for_pipeline("sam2segnet")

    objects = core_objects | pipeline_objects

    # Initialize the PipelineBuilder
    builder = PipelineBuilder()

    # Register each class in the builder
    for key in objects:
        builder.register_class(key, objects[key])

    # Load the configuration file
    builder.load_config(Path("sam2segnet/pipeline_config.toml"))

    # Build the pipeline
    permanences, process_specs = builder.build()

    # Create controller
    controller = PipelineController(permanences, process_specs)

    # Run the pipeline
    controller.run_wandb()
