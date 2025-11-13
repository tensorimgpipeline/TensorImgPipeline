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
from pytorchimagepipeline.controller import PipelineController


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

    def build(self) -> tuple[dict[str, Permanence], list[ProcessWithParams], Optional[Exception]]:
        """Construct permanences and process specifications.

        Returns:
            Tuple of (permanences_dict, process_specs_list, error)
        """
        permanences, error = self._build_permanences()
        if error:
            return {}, [], error

        processes, error = self._build_processes()
        if error:
            return permanences, [], error

        return permanences, processes, None


def get_objects_for_pipeline(pipeline_name: str) -> tuple[dict[str, type], None | Exception]:
    """
    Retrieves and combines objects to be registered for a given pipeline.

    Args:
        pipeline_name (str): The name of the pipeline for which to retrieve objects.

    Returns:
        dict[str, type]: A dictionary containing the combined objects from
                         `permanences_to_register` and `processes_to_register`
                         of the specified pipeline module.
    """
    full_module_name = "pytorchimagepipeline.pipelines." + pipeline_name
    if pipeline_name == "core":
        full_module_name = "pytorchimagepipeline." + pipeline_name
    try:
        module = importlib.import_module(full_module_name)
    except ModuleNotFoundError as e:
        return {}, e
    return module.permanences_to_register | module.processes_to_register, None


# Usage Example
if __name__ == "__main__":
    # Example usage of the PipelineBuilder

    # Retrieve objects to be registered for the pipeline
    core_objects, error = get_objects_for_pipeline("core")
    if error:
        raise error
    pipeline_objects, error = get_objects_for_pipeline("sam2segnet")
    if error:
        raise error

    objects = core_objects | pipeline_objects

    # Initialize the PipelineBuilder
    builder = PipelineBuilder()

    # Register each class in the builder
    for key in objects:
        error = builder.register_class(key, objects[key])
        if error:
            raise error

    # Load the configuration file
    error = builder.load_config(Path("sam2segnet/execute_pipeline.toml"))
    if error:
        raise error

    # Build the pipeline
    controller, error = builder.build()
    if error:
        raise error

    # Run the pipeline
    controller.run_wandb()
