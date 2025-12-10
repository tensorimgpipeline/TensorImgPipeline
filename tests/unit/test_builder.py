from __future__ import annotations

import ast
from pathlib import Path
from tomllib import TOMLDecodeError
from unittest.mock import mock_open, patch

import pytest

from tipi.abstractions import Permanence, PipelineProcess
from tipi.core.builder import PipelineBuilder, get_objects_for_pipeline
from tipi.errors import (
    ConfigInvalidTomlError,
    ConfigNotFoundError,
    ConfigPermissionError,
    ConfigSectionError,
    InstTypeError,
    RegistryError,
)


class MockedPermanence(Permanence):
    def cleanup(self):
        print("Running cleanup")
        return None


class MockedPipelineProcess(PipelineProcess):
    def execute(self, controller):
        print(f"Running execute with {controller}")
        return None


class TestPipelineBuilder:
    pipeline_builder: PipelineBuilder

    @pytest.fixture(autouse=True, scope="class")
    def fixture_class(self):
        cls = type(self)
        cls.pipeline_builder = PipelineBuilder()
        yield

    @pytest.mark.parametrize(
        "name, cls, expected_error",
        [
            ("ValidClass1", Permanence, None),  # Valid registration
            ("ValidClass2", PipelineProcess, None),  # Valid registration
            ("InvalidClass1", dict, RegistryError),  # Invalid class
            ("InvalidClass2", str, RegistryError),  # Invalid class
        ],
        ids=("ValidPermance", "ValidPipelineProcess", "InvalidDict", "InvalidStr"),
    )
    def test_register_class(self, name: str, cls: type, expected_error: None | type[Exception]):
        if expected_error is None:
            self.pipeline_builder.register_class(name, cls)
            assert name in self.pipeline_builder._registry
            assert self.pipeline_builder._registry[name] is cls
        else:
            with pytest.raises(expected_error):
                self.pipeline_builder.register_class(name, cls)
            assert name not in self.pipeline_builder._registry

    @pytest.mark.parametrize(
        "config_path, file_exists, readable, file_content, expected_error",
        [
            ("missing_file.toml", False, False, None, ConfigNotFoundError),
            ("unreadable_file.toml", True, False, None, ConfigPermissionError),
            ("invalid_file.toml", True, True, "invalid_toml", ConfigInvalidTomlError),
            ("missing_section.toml", True, True, '{"unrelated_key": "value"}', ConfigSectionError),
            ("valid_file.toml", True, True, '{"permanences": {}, "processes": {}}', None),
        ],
        ids=("MissingConfig", "UnreadableConfig", "InvalidTOML", "MissingSection", "ValidConfig"),
    )
    def test_load_config(self, config_path, file_exists, readable, file_content, expected_error):
        # Mocking file system behavior
        with (
            patch("pathlib.Path.exists", return_value=file_exists),
            patch("os.access", return_value=readable),
            patch("builtins.open", mock_open(read_data=file_content) if file_content else None),
            patch(
                "tipi.core.builder.toml_load",
                side_effect=TOMLDecodeError if file_content == "invalid_toml" else lambda f: ast.literal_eval(f.read()),
            ),
        ):
            if expected_error is None:
                self.pipeline_builder.load_config(Path(config_path))
                assert self.pipeline_builder._config is not None
            else:
                with pytest.raises(expected_error):
                    self.pipeline_builder.load_config(Path(config_path))

    @pytest.mark.parametrize(
        "processes_config, expected_error",
        [
            ({"process1": {"type": "MockedPipelineProcess"}}, None),
            ({"process1": {"type": "UnknownClass"}}, RegistryError),
            ({"process1": {"type": "MockedPermanence"}}, InstTypeError),
        ],
        ids=("ValidProcess", "InvalidProcess", "InvalidPermanence"),
    )
    def test_build_processes(self, processes_config, expected_error):
        self.pipeline_builder.register_class("MockedPipelineProcess", MockedPipelineProcess)
        self.pipeline_builder.register_class("MockedPermanence", MockedPermanence)
        self.pipeline_builder._config = {"processes": processes_config, "permanences": {}}

        if expected_error is None:
            process_specs = self.pipeline_builder._build_processes()
            assert len(process_specs) == len(processes_config)
        else:
            with pytest.raises(expected_error):
                self.pipeline_builder._build_processes()

    @pytest.mark.parametrize(
        "permanences_config, expected_error",
        [
            ({"object1": {"type": "MockedPermanence"}}, None),  # ValidObjects
            ({"object1": {"type": "UnknownClass"}}, RegistryError),  # InvalidClass
            ({"object1": {"type": "MockedPermanence", "invalid": 1}}, InstTypeError),  # InvalidParams
        ],
        ids=("ValidObjects", "InvalidClass", "InvalidParams"),
    )
    def test_build_permanences(self, permanences_config, expected_error):
        self.pipeline_builder.register_class("MockedPermanence", MockedPermanence)
        self.pipeline_builder._config = {"permanences": permanences_config, "processes": {}}

        if expected_error is None:
            objects = self.pipeline_builder._build_permanences()
            assert len(objects) == len(permanences_config)
        else:
            with pytest.raises(expected_error):
                self.pipeline_builder._build_permanences()

    @pytest.mark.parametrize(
        "config, expected_error",
        [
            (
                {
                    "permanences": {"object1": {"type": "MockedPermanence"}},
                    "processes": {"process1": {"type": "MockedPipelineProcess"}},
                },
                None,
            ),  # Valid case
            (
                {
                    "permanences": {"object1": {"type": "UnknownClass"}},
                    "processes": {"process1": {"type": "PipelineProcess"}},
                },
                RegistryError,
            ),  # Invalid permanent object
            (
                {
                    "permanences": {"object1": {"type": "MockedPermanence"}},
                    "processes": {"process1": {"type": "UnknownClass"}},
                },
                RegistryError,
            ),  # Invalid process
        ],
        ids=("ValidBuild", "InvalidPerma", "InvalidProcess"),
    )
    def test_build(self, config, expected_error):
        self.pipeline_builder.register_class("MockedPermanence", MockedPermanence)
        self.pipeline_builder.register_class("MockedPipelineProcess", MockedPipelineProcess)
        self.pipeline_builder._config = config

        if expected_error is None:
            permanences, processes = self.pipeline_builder.build()
            assert len(permanences) == len(config["permanences"])
            assert len(processes) == len(config["processes"])
        else:
            with pytest.raises(expected_error):
                self.pipeline_builder.build()

    @pytest.mark.parametrize(
        "pipeline_name, module_exists, expected_error",
        [
            ("valid_pipeline", True, None),
            ("invalid_pipeline", False, ImportError),  # Now raises ImportError from path_manager
        ],
        ids=("ValidPipeline", "InvalidPipeline"),
    )
    def test_get_objects_for_pipeline(self, pipeline_name, module_exists, expected_error):
        mock_module = type(
            "MockModule",
            (),
            {
                "permanences_to_register": {MockedPermanence},  # Now a set of classes
                "processes_to_register": {MockedPipelineProcess},  # Now a set of classes
            },
        )

        with patch("tipi.core.builder.importlib.import_module") as mock_import:
            if module_exists:
                mock_import.return_value = mock_module
            else:
                mock_import.side_effect = ModuleNotFoundError()

            if expected_error is None:
                objects = get_objects_for_pipeline(pipeline_name)
                assert len(objects) == 2
                assert "MockedPermanence" in objects
                assert "MockedPipelineProcess" in objects
                assert objects["MockedPermanence"] is MockedPermanence
                assert objects["MockedPipelineProcess"] is MockedPipelineProcess
            else:
                with pytest.raises(expected_error):
                    get_objects_for_pipeline(pipeline_name)
