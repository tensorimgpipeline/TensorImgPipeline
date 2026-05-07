from tipi.abstractions import Permanence, PipelineProcess
from tipi.core.permanences import BasicLogger, Device, ProgressManager, TensorBoardLogger, WandBLogger

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
