from tipi.abstractions import Permanence, PipelineProcess
from tipi.core.permanences.device import Device
from tipi.core.permanences.loggers.basic import BasicLogger
from tipi.core.permanences.loggers.tensorboard import TensorBoardLogger
from tipi.core.permanences.loggers.wandb import WandBLogger
from tipi.core.permanences.progress import ProgressManager

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
