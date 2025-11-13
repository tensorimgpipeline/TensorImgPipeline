from abc import ABC, abstractmethod
from typing import Any, Optional


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

            def initialize(self) -> Optional[Exception]:
                # Setup phase - called before any process runs
                return self._validate_data()

            def checkpoint(self) -> Optional[Exception]:
                # Save intermediate state
                return self._save_checkpoint()

            def cleanup(self) -> Optional[Exception]:
                # Release resources
                del self.data
                return None
        ```
    """

    @abstractmethod
    def cleanup(self) -> Optional[Exception]:
        """Cleans up data from RAM or VRAM.

        Called after all processes complete or on error.
        Should release any held resources (memory, file handles, connections).

        Returns:
            Optional[Exception]: An exception if cleanup fails, otherwise None.
        """
        ...

    def initialize(self) -> Optional[Exception]:
        """Initialize the permanence before pipeline execution.

        Called once after all permanences are constructed but before
        any process runs. Use for validation, resource allocation, or
        setup that depends on other permanences.

        Returns:
            Optional[Exception]: An exception if initialization fails, otherwise None.
        """
        return None

    def checkpoint(self) -> Optional[Exception]:
        """Save intermediate state during pipeline execution.

        Called at configurable checkpoints during execution.
        Use for saving progress, creating backups, or logging state.

        Returns:
            Optional[Exception]: An exception if checkpointing fails, otherwise None.
        """
        return None

    def validate(self) -> Optional[Exception]:
        """Validate the permanence state.

        Called to verify permanence is in valid state.
        Use for health checks, data validation, or consistency checks.

        Returns:
            Optional[Exception]: An exception if validation fails, otherwise None.
        """
        return None

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
