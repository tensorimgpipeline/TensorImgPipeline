from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from permanence import Datasets, HyperParameters, MaskCreator, Network, TrainingComponents

from pytorchimagepipeline.abstractions import AbstractCombinedConfig, AbstractSimpleObserver, ProcessPlanType
from pytorchimagepipeline.core.config import WandBLoggerConfig
from pytorchimagepipeline.core.permanences import Device
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
class Sam2SegnetConfig(AbstractCombinedConfig):
    config_file: Path | str

    config: dict[str, Any] = field(init=False)

    wandb_config: WandBLoggerConfig = field(init=False)
    data_config: DataConfig = field(init=False)
    components_config: ComponentsConfig = field(init=False)
    mask_creator_config: MaskCreatorConfig = field(init=False)
    hyperparams_config: HyperParamsConfig = field(init=False)
    network_config: NetworkConfig = field(init=False)

    def __post_init__(self) -> None:
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
            self.predict_masks_config = PredictMaskConfig(**self.config.get("predict_masks", {}))
            self.train_model_config = TrainModelConfig(**self.config.get("train_model", {}))
        else:
            raise InvalidConfigError(context="missing-execution-config", value=f"{self.config=}")


@dataclass
class Sam2SegnetObserver(AbstractSimpleObserver):
    def __parse_config__(self, config_file: Path) -> None:
        self.config = Sam2SegnetConfig(config_file=config_file)

    def __init_permanences__(self) -> None:
        # Core Permanences
        self.device = Device().device
        # self.wandb = WandBLogger(**asdict(self.config.wandb_config))
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
    observer = Sam2SegnetObserver(config_file=config_file)

    observer.run()
