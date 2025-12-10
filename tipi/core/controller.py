"""This module defines an PipelineController responsible for managing a pipeline of processes and handling
potential errors that occur during their execution.

Classes:
    PipelineController: Manages a pipeline of processes and handles errors that occur during.

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

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from tipi.abstractions import Permanence, PipelineProcess
from tipi.errors import (
    ErrorCode,
    PermanenceKeyError,
)

if TYPE_CHECKING:
    from tipi.core.builder import ProcessWithParams

# Sentinel value for get_permanence to distinguish between "no default provided" and "default is None"
_MISSING = object()


class PipelineController:
    """Coordinates pipeline permanences and processes.

    Responsibilities:
    - Manage permanence lifecycle
    - Provide permanence access to processes
    - Instantiate processes with their parameters
    - Yield processes for execution
    """

    def __init__(self, permanences: dict[str, Permanence], process_specs: list[ProcessWithParams]) -> None:
        self._permanences = permanences
        self._process_specs = process_specs

    def get_permanence(self, name: str, default: Any = _MISSING) -> Any:
        """Get a permanence by name.

        Args:
            name: Name of the permanence to retrieve
            default: Value to return if permanence not found. If not provided,
                    raises PermanenceKeyError instead. Can be None.

        Returns:
            The permanence instance or default value

        Raises:
            PermanenceKeyError: If permanence not found and no default provided
        """
        if default is not _MISSING:
            return self._permanences.get(name, default)
        if name not in self._permanences:
            raise PermanenceKeyError(ErrorCode.PERMA_KEY, key=name)
        return self._permanences[name]

    def iterate_processes(self) -> Iterator[tuple[int, PipelineProcess]]:
        """Yield (index, process_instance) for execution."""
        for idx, spec in enumerate(self._process_specs):
            process_instance = spec.get_instance(self)
            yield idx, process_instance

    def get_process_count(self) -> int:
        """Get total number of processes."""
        return len(self._process_specs)

    def iterate_permanences(self) -> Iterator[Permanence]:
        """Yield permanence instances for cleanup or inspection."""
        yield from self._permanences.values()

    def get_permanence_count(self) -> int:
        """Get total number of permanences."""
        return len(self._permanences)

    def cleanup(self) -> None:
        """Cleanup all permanences."""
        for permanence in self._permanences.values():
            permanence.cleanup()
