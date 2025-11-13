from pathlib import Path

import pytest

from pytorchimagepipeline.errors import InvalidConfigError
from pytorchimagepipeline.pipelines.sam2segnet.config import (
    ComponentsConfig,
    DataConfig,
    HyperParamsConfig,
    MaskCreatorConfig,
)


@pytest.fixture(scope="session")
def tmp_root_location(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    root_dir = root / "root"
    root_dir.mkdir()
    root_file = root / "file.txt"
    root_file.touch()
    return root


@pytest.fixture(scope="session")
def tmp_config_file_location(tmp_path_factory):
    config_dir = tmp_path_factory.mktemp("configs")

    hyperconfigfile = config_dir / "hyper.toml"
    hyperconfigfile.touch()
    wronghyperconfigfile = config_dir / "hyper.json"
    wronghyperconfigfile.touch()
    hyperconfigdir = config_dir / "hyper"
    hyperconfigdir.mkdir()
    return config_dir


class TestSam2Segnet:
    @pytest.mark.parametrize(
        "params, raises, expected",
        [
            (
                {"root": "root", "data_format": "PascalVocFormat"},
                False,
                {"root": Path("root"), "format": "PascalVocFormat"},
            ),
            (
                {"root": "nopath", "data_format": "PascalVocFormat"},
                True,
                {"value": Path("nopath"), "context": "root-not-found"},
            ),
            (
                {"root": "file.txt", "data_format": "PascalVocFormat"},
                True,
                {"value": Path("file.txt"), "context": "root-not-dir"},
            ),
            (
                {"root": "root", "data_format": "NoFormat"},
                True,
                {"value": "NoFormat", "context": "format-not-available"},
            ),
        ],
        ids=("ValidConfig", "InvalidRoot", "InvalidRootDir", "InvalidFormat"),
    )
    def test_data_config(self, params, raises, expected, tmp_root_location):
        extended_root = tmp_root_location / params["root"]
        params["root"] = str(extended_root)
        if raises:
            with pytest.raises(InvalidConfigError) as exception:
                DataConfig(**params)
            assert exception.value.context == expected["context"]
            if isinstance(exception.value.value, Path):
                assert exception.value.value == tmp_root_location / expected["value"]
            else:
                assert exception.value.value == expected["value"]
        else:
            config = DataConfig(**params)
            assert Path(config.root.stem) == expected["root"]
            assert config.data_format == expected["format"]

    @pytest.mark.parametrize(
        "params, raises, expected",
        [
            (
                {"optimizer": "SGD", "scheduler": "LRScheduler", "criterion": "CrossEntropyLoss"},
                False,
                ["SGD", "LRScheduler", "CrossEntropyLoss", {}, {}],
            ),
            (
                {
                    "optimizer": "SGD",
                    "scheduler": "LRScheduler",
                    "criterion": "CrossEntropyLoss",
                    "optimizer_params": {"lr": 0.003, "momentum": 0.9, "dampening": 0.1, "weight_decay": 0.1},
                    "scheduler_params": {"last_epoch": 20},
                },
                False,
                [
                    "SGD",
                    "LRScheduler",
                    "CrossEntropyLoss",
                    {"lr": 0.003, "momentum": 0.9, "dampening": 0.1, "weight_decay": 0.1},
                    {"last_epoch": 20},
                ],
            ),
            (
                {
                    "optimizer": "SGA",
                    "scheduler": "LRScheduler",
                    "criterion": "CrossEntropyLoss",
                },
                True,
                ["SGA", "optimizer-not-available"],
            ),
            (
                {
                    "optimizer": "SGD",
                    "scheduler": "Scheduler",
                    "criterion": "CrossEntropyLoss",
                },
                True,
                ["Scheduler", "scheduler-not-available"],
            ),
            (
                {
                    "optimizer": "SGD",
                    "scheduler": "LRScheduler",
                    "criterion": "CrossEntropy",
                },
                True,
                ["CrossEntropy", "criterion-not-available"],
            ),
            (
                {
                    "optimizer": "SGD",
                    "scheduler": "LRScheduler",
                    "criterion": "CrossEntropyLoss",
                    "scheduler_params": {"not_a_param": 0},
                },
                True,
                ["<class 'torch.optim.lr_scheduler.LRScheduler'>", "params-not-valid"],
            ),
        ],
    )
    def test_component_config(self, params, raises, expected):
        if raises:
            with pytest.raises(InvalidConfigError) as exception:
                ComponentsConfig(**params)
            expected_msg = f"Config entry with value {expected[0]} does failed with context: {expected[1]}"
            assert str(exception.value) == expected_msg
        else:
            config = ComponentsConfig(**params)
            assert config.optimizer == expected[0]
            assert config.scheduler == expected[1]
            assert config.criterion == expected[2]
            assert config.optimizer_params == expected[3]
            assert config.scheduler_params == expected[4]

    @pytest.mark.parametrize(
        "params, raises, expected",
        [
            (
                {"morph_size": 1, "border_size": 4, "ignore_value": 255},
                False,
                {"morph_size": 1, "border_size": 4, "ignore_value": 255},
            ),
            (
                {"morph_size": -1, "border_size": 4, "ignore_value": 255},
                True,
                {"value": -1, "context": "non-positive-morph-size"},
            ),
            (
                {"morph_size": 1, "border_size": -1, "ignore_value": 255},
                True,
                {"value": -1, "context": "non-positive-border-size"},
            ),
            (
                {"morph_size": 1, "border_size": 4, "ignore_value": -30},
                True,
                {"value": -30, "context": "ignore-value-uint8-range"},
            ),
            (
                {"morph_size": 1, "border_size": 4, "ignore_value": 300},
                True,
                {"value": 300, "context": "ignore-value-uint8-range"},
            ),
        ],
        ids=("ValidConfig", "InvalidMorph", "InvalidBorder", "TooSmallIgnore", "TooLargeIgnore"),
    )
    def test_mask_creator_config(self, params, raises, expected):
        if raises:
            with pytest.raises(InvalidConfigError) as exception:
                MaskCreatorConfig(**params)
            assert exception.value.context == expected["context"]
            assert exception.value.value == expected["value"]
        else:
            config = MaskCreatorConfig(**params)
            assert config.morph_size == expected["morph_size"]
            assert config.border_size == expected["border_size"]
            assert config.ignore_value == expected["ignore_value"]

    @pytest.mark.parametrize(
        "params, raises, expected",
        [
            (
                {"config_file": "hyper.toml"},
                False,
                {"config_file": Path("hyper.toml")},
            ),
            (
                {"config_file": "nohyper.toml"},
                True,
                {"value": Path("nohyper.toml"), "context": "params-not-found"},
            ),
            (
                {"config_file": "hyper"},
                True,
                {"value": Path("hyper"), "context": "params-not-file"},
            ),
            (
                {"config_file": "hyper.json"},
                True,
                {"value": Path("hyper.json"), "context": "params-not-toml"},
            ),
        ],
        ids=("ValidConfig", "NoParams", "ParamsDir", "ParamsJson"),
    )
    def test_hyper_params_config(self, params, raises, expected, tmp_config_file_location):
        extended_root = tmp_config_file_location / params["config_file"]
        params["config_file"] = str(extended_root)
        if raises:
            with pytest.raises(InvalidConfigError) as exception:
                HyperParamsConfig(**params)
            assert exception.value.context == expected["context"]
            assert Path(exception.value.value.name) == expected["value"]
        else:
            config = HyperParamsConfig(**params)
            assert Path(config.config_file.name) == expected["config_file"]
