from tipi.abstractions import Permanence, PipelineProcess
from tipi.core.loggers import BasicLogger, TensorBoardLogger, WandBLogger
from tipi.core.permanences import (
    Device,
    ProgressManager,
)

permanences_to_register: set[type[Permanence]] = {
    BasicLogger,
    Device,
    ProgressManager,
    TensorBoardLogger,
    WandBLogger,
}

processes_to_register: set[type[PipelineProcess]] = set()

__all__ = [
    "permanences_to_register",
    "processes_to_register",
]
