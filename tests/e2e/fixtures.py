from pathlib import Path
from unittest.mock import patch

import pytest
from pyfakefs.fake_filesystem import FakeFilesystem
from typer.testing import CliRunner


@pytest.fixture
def cli_runner():
    """Provide a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def setup_fake_pipeline(fs: FakeFilesystem):
    """
    Set up a fake demo_pipeline project structure in the fake filesystem.

    Creates a minimal demo_pipeline with all necessary files for testing.
    """
    # Create fake demo pipeline structure
    fake_demo_path = Path("/fake_workspace/demo_pipeline")

    # Create package structure
    package_dir = fake_demo_path / "demo_pipeline"
    fs.create_dir(package_dir)

    # Create __init__.py
    fs.create_file(package_dir / "__init__.py", contents='"""Demo Pipeline Package."""\n\n__version__ = "0.1.0"\n')

    # Create permanences.py
    permanences_content = '''"""Demo permanences."""

from pytorchimagepipeline.abstractions import Permanence


class DemoPermanence(Permanence):
    """A demo permanence for testing."""

    def __init__(self, name: str = "demo"):
        """Initialize demo permanence."""
        self.name = name

    def load(self):
        """Load demo data."""
        return {"status": "loaded", "name": self.name}

    def save(self, data):
        """Save demo data."""
        return {"status": "saved", "data": data}
'''
    fs.create_file(package_dir / "permanences.py", contents=permanences_content)

    # Create processes.py
    processes_content = '''"""Demo processes."""

from pytorchimagepipeline.abstractions import Process


class DemoProcess(Process):
    """A demo process for testing."""

    def __init__(self, name: str = "demo"):
        """Initialize demo process."""
        self.name = name

    def execute(self, data):
        """Execute demo processing."""
        return {"processed": True, "input": data, "name": self.name}
'''
    fs.create_file(package_dir / "processes.py", contents=processes_content)

    # Create configs directory
    configs_dir = fake_demo_path / "configs"
    fs.create_dir(configs_dir)

    # Create a sample config file
    config_content = """[pipeline]
name = "demo_pipeline"
version = "0.1.0"
description = "A demo pipeline for testing"

[pipeline.steps]
step1 = "demo_process"
step2 = "demo_permanence"
"""
    fs.create_file(configs_dir / "pipeline_config.toml", contents=config_content)

    # Create pyproject.toml
    pyproject_content = """[project]
name = "demo_pipeline"
version = "0.1.0"
description = "A demo pipeline for testing"
requires-python = ">=3.9"
dependencies = ["PytorchImagePipeline"]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
"""
    fs.create_file(fake_demo_path / "pyproject.toml", contents=pyproject_content)

    # Create README
    fs.create_file(fake_demo_path / "README.md", contents="# Demo Pipeline\n\nA demo pipeline for testing.\n")

    return fake_demo_path


@pytest.fixture
def mock_home_dir(fs: FakeFilesystem):
    """Set up a fake home directory."""
    fake_home = Path("/fake_home/testuser")
    fs.create_dir(fake_home)

    # Create config directories
    config_dir = fake_home / ".config/pytorchimagepipeline"
    fs.create_dir(config_dir / "projects")
    fs.create_dir(config_dir / "configs")
    fs.create_dir(fake_home / ".cache/pytorchimagepipeline")

    # Mock HOME and XDG environment variables to force use of fake home
    env_overrides = {
        "HOME": str(fake_home),
        "XDG_CONFIG_HOME": str(fake_home / ".config"),
        "XDG_CACHE_HOME": str(fake_home / ".cache"),
        "PYTORCHPIPELINE_PROJECTS_DIR": str(config_dir / "projects"),
        "PYTORCHPIPELINE_CONFIG_DIR": str(config_dir / "configs"),
        "PYTORCHPIPELINE_CACHE_DIR": str(fake_home / ".cache/pytorchimagepipeline"),
    }

    with patch.dict("os.environ", env_overrides, clear=False):
        yield fake_home
