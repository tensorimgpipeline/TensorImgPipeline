"""Controller abstractions for TensorImgPipeline.

Provides abstract base class for pipeline controllers that manage
process execution and progress reporting.

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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tipi.abstractions.process import PipelineProcess


class AbstractController(ABC):
    """Abstract base class for pipeline controllers.

    Controllers manage the execution flow of pipeline processes,
    including progress reporting and process lifecycle management.
    """

    @abstractmethod
    def add_process(self, process: "PipelineProcess") -> None:
        """Add a process to the controller's execution queue.

        Args:
            process: The pipeline process to add.
        """
        ...

    @abstractmethod
    def _get_progress_decorator(self) -> Callable:
        """Get a decorator for progress reporting.

        Returns:
            A decorator function that wraps process execution with
            progress reporting capabilities.
        """
        ...
