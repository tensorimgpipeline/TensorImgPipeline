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
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

from pytorchimagepipeline.errors import InvalidConfigError


class AbstractObserver(ABC):
    """Base class for the Observer class"""

    def __init__(self, config):
        self.__parse_config__(config)
        self.__init_permanences__()
        self.__init_processes__()

    @abstractmethod
    def __parse_config__(self, config) -> None: ...

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
    def execute(self) -> Optional[Exception]:
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


ProcessPlanType = dict[str, type(PipelineProcess)]


@dataclass
class AbstractConfig(ABC):
    def __post_init__(self):
        self._apply_path()
        self.validate()

    @abstractmethod
    def validate(self) -> None: ...

    def validate_params(self, params: dict[str, Any], cls: type):
        signature = inspect.signature(cls)  # Get constructor signature

        # Get expected parameters (excluding 'self')
        expected_params = list(signature.parameters.keys())
        if "self" in expected_params:
            expected_params.remove("self")

        # Check if all required parameters are provided
        if not set(params.keys()).issubset(expected_params):
            raise InvalidConfigError(context="params-not-valid", value=str(cls))

    def _apply_path(self):
        hints = get_type_hints(self.__class__)
        for field in fields(self):
            field_name = field.name
            field_type = hints[field_name]
            value = getattr(self, field_name)

            # Check if the field type is a union
            origin = get_origin(field_type)
            if origin is Union:
                args = get_args(field_type)
                # If the field accepts Path and also a string type
                if Path in args and isinstance(value, str):
                    setattr(self, field_name, Path(value))
