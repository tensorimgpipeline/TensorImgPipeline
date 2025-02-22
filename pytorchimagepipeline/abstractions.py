"""This module defines abstract base classes for the Pytorch image pipeline.

Classes:
    AbstractObserver: Base class for the Observer class.
    Permanence: Base class for objects that persist through the entire pipeline lifecycle.
    PipelineProcess: Abstract base class for pipeline processes.

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

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar, Union, get_args, get_origin, get_type_hints

import tomllib

from pytorchimagepipeline.errors import InvalidConfigError


@dataclass
class AbstractCombinedConfig(ABC):
    config_file: Path | str = field(init=False)

    @abstractmethod
    def __post_init__(self) -> None: ...

    def _read_config(self) -> None:
        if not isinstance(self.config_file, Path):
            self.config_file = Path(self.config_file)
        with self.config_file.open("rb") as content:
            self.config = tomllib.load(content)


class Permanence(ABC):
    """Base class for objects that persist through the entire pipeline lifecycle"""

    @abstractmethod
    def cleanup(self) -> Optional[Exception]:
        """Cleans up data from RAM or VRAM.

        Since the objects are permanent, it might be necessary to call a cleanup.
        This will be executed by the observer.

        Returns:
            Optional[Exception]: An exception if an error occurs during cleanup, otherwise None.
        """
        ...


@dataclass(kw_only=True)
class AbstractObserver(ABC):
    """Base class for the Observer class"""

    config_file: Path | str
    config: AbstractCombinedConfig = field(init=False)

    def __post_init__(self) -> None:
        self.__parse_config__(self.config_file)
        self.__init_permanences__()
        self.__init_processes__()

    @abstractmethod
    def __parse_config__(self, config_file: Path | str) -> None: ...

    @abstractmethod
    def __init_permanences__(self) -> None:
        """Initializes the permanences.

        This method initializes the permanences that are used by the observer.
        The implementation details depend on the concrete observer class.

        Returns:
            None: This method doesn't return any value
        """
        ...

    @abstractmethod
    def __init_processes__(self) -> None:
        """Initializes the processes.

        This method initializes the processes that are used by the observer.
        The implementation details depend on the concrete observer class.

        Returns:
            None: This method doesn't return any value
        """
        ...

    @abstractmethod
    def run(self) -> None:
        """Executes the observer's processes.

        This method runs the specific processes defined by the observer implementation.
        The execution details depend on the concrete observer class. #todo: add different observers

        Returns:
            None: This method doesn't return any value
        """
        ...


class PipelineProcess(ABC):
    """Abstract base class for pipeline processes"""

    def __init__(self, observer: AbstractObserver, force: bool) -> None:
        """
        Initializes the instance with the given observer.

        When overriding this method, make sure to call the super().__init__(observer, force) method.
        In genereal instead of creating a new instance of the observer, the observer should be passed as an argument.
        The same applies to the force parameter.

        Args:
            observer (AbstractObserver): The observer to be assigned to the instance.
        """

        self.observer = observer
        self.force = force

    @abstractmethod
    def execute(self) -> None:
        """Executes the process.

        Args:
            observer (AbstractObserver): The observer instance managing the pipeline.

        Returns:
            Optional[Exception]: An exception if an error occurs during execution, otherwise None.
        """
        ...

    @abstractmethod
    def skip(self) -> bool:
        """Returns whether the process should be skipped.

        Returns:
            bool: True if the process should be skipped, False otherwise.
        """
        ...


@dataclass
class AbstractConfig(ABC):
    def __post_init__(self) -> None:
        self._apply_path()
        self.validate()

    @abstractmethod
    def validate(self) -> None: ...

    def validate_params(self, params: dict[str, Any], cls: type) -> None:
        signature = inspect.signature(cls)  # Get constructor signature

        # Get expected parameters (excluding 'self')
        expected_params = list(signature.parameters.keys())
        if "self" in expected_params:
            expected_params.remove("self")

        # Check if all required parameters are provided
        if not set(params.keys()).issubset(expected_params):
            raise InvalidConfigError(context="params-not-valid", value=str(cls))

    def _apply_path(self) -> None:
        hints = get_type_hints(self.__class__)
        for _field in fields(self):
            field_name = _field.name
            field_type = hints[field_name]
            value = getattr(self, field_name)

            # Check if the field type is a union
            origin = get_origin(field_type)
            if origin is Union:
                args = get_args(field_type)
                # If the field accepts Path and also a string type
                if Path in args and isinstance(value, str):
                    setattr(self, field_name, Path(value))


TConfig = TypeVar("TConfig", bound=AbstractConfig)
TProcess = TypeVar("TProcess", bound=PipelineProcess)
ProcessPlanType = dict[str, tuple[type[TProcess], TConfig]]


@dataclass(kw_only=True)
class AbstractSimpleObserver(AbstractObserver):
    process_plan: ProcessPlanType = field(init=False)

    def run(self) -> None:
        """
        Executes each process in the list of processes.

        Iterates over the processes, sets the current process, and executes it.
        If an error occurs during the execution of a process, it handles the error.
        Resets the current process to None after each execution.

        Returns:
            None
        """
        if not self.process_plan:
            raise ValueError(self.process_plan)  # TODO add custom error

        self._iter_processes()

        self._iter_cleanup()

    def _iter_cleanup(self) -> None:
        for _field in fields(self):
            if isinstance(_field, Permanence):
                _field.cleanup()

    def _iter_processes(self) -> None:
        for step, process_with_config in self.process_plan.items():
            logging.debug(f"Executing Process {step}")
            process: type[PipelineProcess] = process_with_config[0]
            process_config: type[AbstractConfig] = process_with_config[1]
            if not (is_dataclass(process_config) and not isinstance(process_config, type)):
                raise RuntimeError(process_config)
            process_instance = process(self, **asdict(process_config))
            if not process_instance.skip():
                process_instance.execute()
