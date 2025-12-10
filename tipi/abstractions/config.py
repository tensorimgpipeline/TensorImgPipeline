"""Configuration abstractions for TensorImgPipeline.

Provides abstract base classes for configuration objects used throughout the pipeline.

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
from typing import Any, Union, get_args, get_origin, get_type_hints

from tipi.errors import InvalidConfigError


@dataclass
class AbstractConfig(ABC):
    """Abstract base class for configuration objects.

    Provides common functionality for:
    - Path string to Path object conversion
    - Parameter validation
    - Configuration validation

    Subclasses must implement the validate() method to define
    their specific validation logic.
    """

    def __post_init__(self) -> None:
        """Post-initialization hook.

        Applies path conversions and runs validation.
        """
        self._apply_path()
        self.validate()

    @abstractmethod
    def validate(self) -> None:
        """Validate the configuration.

        Subclasses should implement this method to check that all
        configuration values are valid and meet requirements.

        Raises:
            InvalidConfigError: If configuration is invalid.
        """
        ...

    def validate_params(self, params: dict[str, Any], cls: type) -> None:
        """Validate that parameters match a class constructor signature.

        Args:
            params: Dictionary of parameter names to values.
            cls: The class whose constructor signature to validate against.

        Raises:
            InvalidConfigError: If params contain unexpected keys.
        """
        signature = inspect.signature(cls)  # Get constructor signature

        # Get expected parameters (excluding 'self')
        expected_params = list(signature.parameters.keys())
        if "self" in expected_params:
            expected_params.remove("self")

        # Check if all required parameters are provided
        if not set(params.keys()).issubset(expected_params):
            raise InvalidConfigError(context="params-not-valid", value=str(cls))

    def _apply_path(self) -> None:
        """Convert string fields to Path objects where appropriate.

        Examines all dataclass fields and converts string values to Path
        objects if the field's type hint includes Path in a Union.
        """
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


@dataclass
class ProcessConfig(AbstractConfig):
    """Base configuration for pipeline processes.

    Attributes:
        force: If True, forces execution even if outputs exist.
    """

    force: bool = False

    def validate(self) -> None:
        """Validate the process configuration.

        Raises:
            InvalidConfigError: If force is not a boolean.
        """
        if not isinstance(self.force, bool):
            raise InvalidConfigError(context="invalid-force-type", value=f"{self.force=}")
