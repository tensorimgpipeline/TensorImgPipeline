from abc import ABC, abstractmethod
from typing import Any


class Permanence(ABC):
    """Base class for objects that persist through the entire pipeline lifecycle.

    Permanences are stateful resources that:
    - Store structured data needed throughout pipeline execution
    - Are accessed by processes via controller.get_permanence(name)
    - Have managed lifecycles with hooks
    - Are extensible through abstraction

    Example:
        ```python
        class MyDataPermanence(Permanence):
            def __init__(self, path: Path):
                self.data = self._load_data(path)

            def initialize(self) -> None:
                # Setup phase - called before any process runs
                self._validate_data()

            def checkpoint(self) -> None:
                # Save intermediate state
                self._save_checkpoint()

            def cleanup(self) -> None:
                # Release resources
                del self.data
        ```
    """

    @abstractmethod
    def cleanup(self) -> None:
        """Cleans up data from RAM or VRAM.

        Called after all processes complete or on error.
        Should release any held resources (memory, file handles, connections).

        Raises:
            Exception: If cleanup fails
        """
        ...

    def initialize(self) -> None:
        """Initialize the permanence before pipeline execution.

        Called once after all permanences are constructed but before
        any process runs. Use for validation, resource allocation, or
        setup that depends on other permanences.

        Raises:
            Exception: If initialization fails
        """
        return

    def checkpoint(self) -> None:
        """Save intermediate state during pipeline execution.

        Called at configurable checkpoints during execution.
        Use for saving progress, creating backups, or logging state.

        Raises:
            Exception: If checkpointing fails
        """
        return

    def validate(self) -> None:
        """Validate the permanence state.

        Called to verify permanence is in valid state.
        Use for health checks, data validation, or consistency checks.

        Raises:
            Exception: If validation fails
        """
        return

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for inspection or debugging.

        Returns a dictionary representation of the permanence state.
        Useful for logging, debugging, or state inspection.

        Returns:
            dict[str, Any]: Dictionary containing permanence state.
        """
        return {
            "type": self.__class__.__name__,
            "initialized": True,
        }
