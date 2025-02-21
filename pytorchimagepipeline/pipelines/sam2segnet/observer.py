from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AnyStr

import tomllib
from permanence import Datasets, HyperParameters, MaskCreator, Network, Sam2SegnetProgressManager, TrainingComponents

from pytorchimagepipeline.abstractions import AbstractObserver, ProcessPlanType
from pytorchimagepipeline.core.config import WandBLoggerConfig
from pytorchimagepipeline.core.permanences import Device, WandBLogger
from pytorchimagepipeline.errors import InvalidConfigError
from pytorchimagepipeline.pipelines.sam2segnet.config import (
    ComponentsConfig,
    DataConfig,
    HyperParamsConfig,
    MaskCreatorConfig,
    NetworkConfig,
    PredictMaskConfig,
    TrainModelConfig,
)
from pytorchimagepipeline.pipelines.sam2segnet.processes import PredictMasks, TrainModel


@dataclass
class Sam2SegnetConfig:
    config_file: Path | AnyStr

    config: dict[str, Any] = field(default_factory=dict)

    wandb_config: WandBLoggerConfig | None = None
    data_config: DataConfig | None = None
    components_config: ComponentsConfig | None = None
    mask_creator_config: MaskCreatorConfig | None = None
    hyperparams_config: HyperParamsConfig | None = None
    network_config: NetworkConfig | None = None

    def __post_init__(self):
        self.config_file = Path(self.config_file)
        if self.config_file.exists():
            self._read_config()
            # Load Permanence configs
            self.wandb_config = WandBLoggerConfig(**self.config.get("wandb", {}))
            self.data_config = DataConfig(**self.config.get("data", {}))
            self.mask_creator_config = MaskCreatorConfig(**self.config.get("mask_creator", {}))
            self.components_config = ComponentsConfig(**self.config.get("components", {}))
            self.hyperparams_config = HyperParamsConfig(**self.config.get("hyperparams", {}))
            self.network_config = NetworkConfig(**self.config.get("network", {}))
            # Load Process configs
            self.predict_mask_config = PredictMaskConfig(**self.config.get("predict_masks", {}))
            self.train_model_config = TrainModelConfig(**self.config.get("train_model", {}))
        else:
            raise InvalidConfigError(context="missing-execution-config", value=self.config)

    def _read_config(self):
        with self.config_file.open("rb") as content:
            self.config = tomllib.load(content)


class Sam2SegnetObserver(AbstractObserver):
    def __parse_config__(self, config):
        self.config = Sam2SegnetConfig(config)

    def __init_permanences__(self):
        # Core Permanences
        self.device = Device().device
        self.wandb = WandBLogger(**self.config.wandb_config)
        # Sam2Segnet Permanences
        self.data = Datasets(**self.config.data_config)
        self.training_components = TrainingComponents(**self.config.components_config)
        self.hyperparams = HyperParameters(**self.config.hyperparams_config)
        self.hyperparams.calculate_batch_size()
        self.network = Network(**self.config.network_config)
        self.mask_creator = MaskCreator(**self.config.mask_creator_config)
        self.progress = Sam2SegnetProgressManager()

    def __init_processes__(self):
        self.process_plan: ProcessPlanType = {
            "predict_masks": PredictMasks,
            "train_model": TrainModel,
        }


if __name__ == "__main__":
    config_file = Path("configs/sam2segnet/execute_pipeline.toml")
    observer = Sam2SegnetConfig(config_file)
