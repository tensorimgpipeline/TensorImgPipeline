from tipi.abstractions import Permanence, PipelineProcess
from tipi.core.permanences import (
    Device,
    ProgressManager,
    WandBManager,
)

permanences_to_register: set[type[Permanence]] = {
    Device,
    ProgressManager,
    WandBManager,
}

processes_to_register: set[type[PipelineProcess]] = set()

__all__ = [
    "permanences_to_register",
    "processes_to_register",
]
