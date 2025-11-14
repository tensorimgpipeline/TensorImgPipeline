from pytorchimagepipeline.abstractions import Permanence, PipelineProcess
from pytorchimagepipeline.core.permanences import (
    Device,
    ProgressManager,
    WandBManager,
)

permanences_to_register: dict[str, type[Permanence]] = {
    "device": Device,
    "progress_manager": ProgressManager,
    "wandb_logger": WandBManager,
}

processes_to_register: dict[str, type[PipelineProcess]] = {
    # "result": ResultProcess,
}

__all__ = [
    "permanences_to_register",
    "processes_to_register",
]
