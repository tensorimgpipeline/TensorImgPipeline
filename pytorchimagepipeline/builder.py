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
    def __init__(self) -> None:
        """
        Initializes the builder with empty configuration and class registry.

        Attributes:
            _config (dict[str, Any]): A dictionary to store configuration settings.
            _class_registry (dict[str, type]): A dictionary to store class types by name.
        """
        self._config: dict[str, Any] = {}
        self._class_registry: dict[str, type] = {}

    def register_class(self, name: str, cls: type) -> None | Exception:
        """
        Registers a class in the class registry.

        Args:
            name (str): The name to register the class under.
            cls (type): The class type to register.

        Returns:
            Optional[Exception]: Returns a RegistryError if the class is not a subclass
                                 of either Permanence or PipelineProcess, otherwise None.
        """
        if not issubclass(cls, (Permanence, PipelineProcess)):
            return RegistryError(name)
        self._class_registry[name] = cls
        return None

    def load_config(self, config_path: Path) -> None | Exception:
        """
        Loads a configuration file from the specified path.

        Args:
            config_path (Path): The path to the configuration file.

        Returns:
            Optional[Exception]: Returns an exception if an error occurs during loading,
                     otherwise returns None.

        Raises:
            ConfigNotFoundError: If the configuration file does not exist.
            ConfigPermissionError: If the configuration file is not readable.
            ConfigInvalidTomlError: If the configuration file is not a valid TOML file.
        """
        config_path_extended = Path("configs") / config_path
        if not config_path_extended.exists():
            return ConfigNotFoundError(config_path_extended)
        if not os.access(config_path_extended, os.R_OK):
            return ConfigPermissionError(config_path_extended)
        with open(config_path_extended, "rb") as f:
            try:
                self._config |= toml_load(f)
            except TOMLDecodeError:
                return ConfigInvalidTomlError(config_path_extended)
        error = self._validate_config_sections()
        return error

    def _validate_config_sections(self) -> None | Exception:
        """
        Validate required configuration sections.

        This method checks if the required sections are present in the configuration.
        If any required section is missing, it returns a ConfigSectionError for the missing section.

        Returns:
            Optional[Exception]: ConfigSectionError if a required section is missing, otherwise None.
        """
        required_sections = ["permanences", "processes"]
        for section in required_sections:
            if section not in self._config:
                return ConfigSectionError(section)
        return None

    def build(self) -> tuple[PipelineController, None | Exception]:
        """
        Construct the complete pipeline.

        Returns:
            tuple[PipelineController, Optional[Exception]]: A tuple containing the constructed PipelineController object
            and an optional Exception if an error occurred during the construction process.
        """
        permanence, error = self._build_permanences()
        if error:
            return PipelineController(permanences={}), error
        controller = PipelineController(permanences=permanence)
        error = self._build_processes(controller)
        if error:
            return controller, error
        return controller, None

    def _build_permanences(self) -> tuple[dict[str, Permanence], None | Exception]:
        """
        Construct permanence objects with error handling.

        This method iterates over the permanence configurations provided in
        `self._config["permanences"]`, instantiates each permanence object,
        and collects them into a dictionary. If an error occurs during the
        instantiation of any permanence object, the method returns an empty
        dictionary and the encountered error.

        Returns:
            tuple[dict[str, Permanence], Optional[Exception]]: A tuple containing:
                - A dictionary where keys are permanence names and values are
                  the instantiated permanence objects.
                - An optional Exception if an error occurred during instantiation,
                  otherwise None.
        """
        objects: dict[str, Permanence] = {}
        for name, config in self._config["permanences"].items():
            type_and_parmas, error = self._get_type_and_param(name, config)
            if error:
                return {}, error
            cls_name, params = type_and_parmas
            instance, error = self._instatiate_permanence(cls_name, params)
            if error:
                return {}, error
            if isinstance(instance, Permanence):
                objects[name] = instance
        return objects, None

    def _build_processes(self, controller: PipelineController) -> None | Exception:
        """
        Builds and adds processes to the controller, based on the configuration.

        Args:
            controller (PipelineController): The controller to which the processes will be added.

        Returns:
            Optional[Exception]: Returns an exception if an error occurs during the
            instantiation or addition of a process, otherwise returns None.
        """
        for name, config in self._config["processes"].items():
            type_and_params, error = self._get_type_and_param(name, config)
            if error:
                return error
            cls_name, params = type_and_params
            if not issubclass(self._class_registry[cls_name], PipelineProcess):
                return InstTypeError(cls_name)
            controller.add_process(ProcessWithParams(self._class_registry[cls_name], params))
        return None

    def _get_type_and_param(
        self, context: str, config: dict[str, Any]
    ) -> tuple[tuple[str, dict[str, Any]] | None, None | Exception]:
        """
        Retrieve the type and parameters from the configuration dictionary.

        Args:
            context (str): The context in which this function is called, used for error messages.
            config (dict[str, Any]): The configuration dictionary containing the type and parameters.

        Returns:
            tuple: A tuple containing:
                - A tuple of the class name (str) and parameters (dict[str, Any]), or None if the type is not found.
                - An exception (None or Exception) if there is an error, otherwise None.

        Raises:
            InstTypeError: If the "type" key is not present in the config dictionary.
            RegistryError: If the class name is not found in the class registry.
        """

        if "type" not in config:
            return None, InstTypeError(context)
        cls_name = config["type"]
        params = config.get("params", {})
        if cls_name not in self._class_registry:
            return None, RegistryError(f"{context}-{cls_name}")
        return (cls_name, params), None

    def _instatiate_permanence(
        self, cls_name: str, params: dict[str, Any]
    ) -> tuple[Permanence | None, None | Exception]:
        """
        Instantiate a permanence object from the class registry.

        Args:
            cls_name (str): The name of the class to instantiate.
            params (dict[str, Any]): The parameters to pass to the class constructor.

        Returns:
            tuple[Permanence | None, None | Exception]: A tuple containing the instantiated
            permanence object or None if instantiation fails, and None or an exception if
            an error occurs.
        """
        try:
            return self._class_registry[cls_name](**params), None
        except TypeError:
            return None, RegistryParamError(params)


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
