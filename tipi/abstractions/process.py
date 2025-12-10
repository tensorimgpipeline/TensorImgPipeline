"""Process abstractions for TensorImgPipeline.

Provides abstract base class for pipeline processes that define
units of work within the pipeline.

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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import for type checking only to avoid circular imports
    # In practice, any object with get_permanence() will work
    pass


class PipelineProcess(ABC):
    """Abstract base class for pipeline processes.

    A process represents a unit of work within the pipeline that:
    - Can access permanences via a controller/manager
    - Can be skipped based on conditions
    - Executes its main logic via execute()
    - Can be forced to run via the force parameter

    Example:
        ```python
        class MyProcess(PipelineProcess):
            def __init__(self, controller, force: bool):
                super().__init__(controller, force)
                self.data = controller.get_permanence("data")

            def skip(self) -> bool:
                return not self.force and self.data.is_cached()

            def execute(self) -> None:
                # Process logic here
                self.data.process()
        ```
    """

    def __init__(self, controller: Any, force: bool) -> None:
        """Initialize the process.

        When overriding this method, make sure to call super().__init__(controller, force).

        Args:
            controller: The controller/manager providing access to permanences.
                       Should have a get_permanence(name: str) method.
            force: If True, process should run even if outputs exist.
        """
        self.controller = controller
        self.force = force

    @abstractmethod
    def execute(self) -> None:
        """Execute the process logic.

        This method should contain the main work of the process.
        It should handle any errors internally or let them propagate.

        Raises:
            Exception: Any exceptions during execution.
        """
        ...

    @abstractmethod
    def skip(self) -> bool:
        """Determine if the process should be skipped.

        Returns:
            True if the process should be skipped, False otherwise.
            Common reasons to skip:
            - Outputs already exist and force=False
            - Required inputs are missing
            - Conditional execution based on config
        """
        ...
