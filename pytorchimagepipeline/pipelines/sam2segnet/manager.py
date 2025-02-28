from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from pytorchimagepipeline.abstractions import AbstractSimpleManager, ProcessPlanType
from pytorchimagepipeline.core.permanences import Device, ProgressManager
from pytorchimagepipeline.pipelines.sam2segnet.config import Sam2SegnetConfig
from pytorchimagepipeline.pipelines.sam2segnet.permanence import (
    Datasets,
    HyperParameters,
    MaskCreator,
    Network,
    TrainingComponents,
)
from pytorchimagepipeline.pipelines.sam2segnet.processes import PredictMasks, TrainModel


@dataclass
class Sam2SegnetManager(AbstractSimpleManager):
    def __parse_config__(self, config_file: Path) -> None:
        self.config = Sam2SegnetConfig(config_file=config_file)

    def __init_permanences__(self) -> None:
        # Core Init Permanences
        self.progress = ProgressManager()
        additional_progresses = [
            {"name": "create_masks", "with_status": True},
            {"name": "epoch", "with_status": True},
            {"name": "train-val-test", "with_status": True},
        ]
        self.progress.add_progresses(additional_progresses)
        # Core Permanences
        self.device = Device().device
        # self.wandb = WandBlogger(**asdict(self.config.wandb_config))
        # Sam2Segnet Permanences
        self.data = Datasets(**asdict(self.config.data_config))
        self.training_components = TrainingComponents(**asdict(self.config.components_config))
        self.hyperparams = HyperParameters(**asdict(self.config.hyperparams_config))
        self.hyperparams.calculate_batch_size(self.device)
        self.network = Network(**asdict(self.config.network_config))
        self.mask_creator = MaskCreator(**asdict(self.config.mask_creator_config))
        # self.progress = Sam2SegnetProgressManager()

    def __init_processes__(self) -> None:
        self.config: Sam2SegnetConfig
        self.process_plan: ProcessPlanType = {
            "predict_masks": (PredictMasks, self.config.predict_masks_config),
            "train_model": (TrainModel, self.config.train_model_config),
        }


if __name__ == "__main__":
    config_file = Path("configs/sam2segnet/execute_pipeline.toml")
    manager = Sam2SegnetManager(config_file=config_file)

    manager.run()
