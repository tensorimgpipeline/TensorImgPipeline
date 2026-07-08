import importlib

from tipi.abstractions import Permanence, PipelineProcess
from tipi.core.permanences.device import Device
from tipi.core.permanences.loggers.basic import BasicLogger
from tipi.core.permanences.progress import ProgressManager


def import_optional_dependency_based_classes(module_name: str, class_name: str) -> type[Permanence] | None:
    module_path = "tipi.core.permanences.loggers." + module_name

    try:
        module = importlib.import_module(module_path)
        class_obj: type[Permanence] = getattr(module, class_name)
    except ImportError:
        return None
    return class_obj


WandBLogger = import_optional_dependency_based_classes("wandb", "WandBLogger")
TensorBoardLogger = import_optional_dependency_based_classes("tensorboard", "TensorBoardLogger")

permanences_to_register: set[type[Permanence]] = {
    BasicLogger,
    Device,
    ProgressManager,
    *((WandBLogger,) if WandBLogger is not None else ()),
    *((TensorBoardLogger,) if TensorBoardLogger is not None else ()),
}


processes_to_register: set[type[PipelineProcess]] = set()

__all__ = [
    "permanences_to_register",
    "processes_to_register",
]
