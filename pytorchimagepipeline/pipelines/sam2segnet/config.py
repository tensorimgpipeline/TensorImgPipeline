from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pytorchimagepipeline.abstractions import AbstractCombinedConfig, AbstractConfig, ProcessConfig
from pytorchimagepipeline.core.config import WandBLoggerConfig
from pytorchimagepipeline.errors import InvalidConfigError
from pytorchimagepipeline.pipelines.sam2segnet import formats


@dataclass
class DataConfig(AbstractConfig):
    root: Path | str
    data_format: str

    def validate(self) -> None:
        if isinstance(self.root, str):
            self.root = Path(self.root)
        if not self.root.exists():
            raise InvalidConfigError(context="root-not-found", value=f"{self.root=}")
        if not self.root.is_dir():
            raise InvalidConfigError(context="root-not-dir", value=f"{self.root=}")
        if not hasattr(formats, self.data_format):
            raise InvalidConfigError(context="format-not-available", value=f"{self.data_format=}")


@dataclass
class ComponentsConfig(AbstractConfig):
    optimizer: str
    scheduler: str
    criterion: str

    optimizer_params: dict[str, Any] = field(default_factory=dict)
    scheduler_params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not hasattr(torch.optim, self.optimizer):
            raise InvalidConfigError(context="optimizer-not-available", value=self.optimizer)
        if self.optimizer_params:
            cls = getattr(torch.optim, self.optimizer)
            self.validate_params(self.optimizer_params, cls)
        if not hasattr(torch.optim.lr_scheduler, self.scheduler):
            raise InvalidConfigError(context="scheduler-not-available", value=self.scheduler)
        if self.scheduler_params:
            cls = getattr(torch.optim.lr_scheduler, self.scheduler)
            self.validate_params(self.scheduler_params, cls)
        if not hasattr(torch.nn, self.criterion):
            raise InvalidConfigError(context="criterion-not-available", value=self.criterion)


@dataclass
class MaskCreatorConfig(AbstractConfig):
    morph_size: int
    border_size: int
    ignore_value: int

    def validate(self) -> None:
        if self.morph_size < 1:
            raise InvalidConfigError(context="non-positive-morph-size", value=f"{self.morph_size=}")
        if self.border_size < 1:
            raise InvalidConfigError(context="non-positive-border-size", value=f"{self.border_size=}")
        if self.ignore_value < -1 or self.ignore_value > 255:
            raise InvalidConfigError(context="ignore-value-uint8-range", value=f"{self.ignore_value=}")


@dataclass
class HyperParamsConfig(AbstractConfig):
    config_file: Path | str

    def validate(self) -> None:
        if isinstance(self.config_file, str):
            self.config_file = Path(self.config_file)
        if not self.config_file.exists():
            raise InvalidConfigError(context="params-not-found", value=f"{self.config_file=}")
        if not self.config_file.is_file():
            raise InvalidConfigError(context="params-not-file", value=f"{self.config_file=}")
        if not self.config_file.suffix == ".toml":
            raise InvalidConfigError(context="params-not-toml", value=f"{self.config_file=}")


@dataclass
class NetworkConfig(AbstractConfig):
    model: str
    num_classes: int
    pretrained: bool

    def validate(self) -> None:
        if self.num_classes < 1:
            raise InvalidConfigError(context="invalid-class-num", value=f"{self.num_classes=}")
        if not isinstance(self.model, str):
            raise InvalidConfigError(context="only-str-model", value=f"{self.model=}")
        if not isinstance(self.pretrained, bool):
            raise InvalidConfigError(context="only-bool-pretrained", value=f"{self.pretrained=}")


@dataclass
class PredictMaskConfig(ProcessConfig): ...


@dataclass
class TrainModelConfig(ProcessConfig): ...


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


if __name__ == "__main__":
    config = HyperParamsConfig(config_file="configs/sam2segnet/hyper_params.toml")
