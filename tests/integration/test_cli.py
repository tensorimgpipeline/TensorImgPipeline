"""Tests for the CLI module.

Tests all CLI commands and their interactions with the PathManager.
"""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pytorchimagepipeline.cli import app

# Create CLI runner
runner = CliRunner()


@pytest.fixture
def mock_path_manager(tmp_path):
    """Create a mock PathManager with temporary directories."""
    projects_dir = tmp_path / "projects"
    configs_dir = tmp_path / "configs"
    cache_dir = tmp_path / "cache"

    projects_dir.mkdir()
    configs_dir.mkdir()
    cache_dir.mkdir()

    with patch("pytorchimagepipeline.cli.path_manager") as mock_pm:
        mock_pm.get_projects_dir.return_value = projects_dir
        mock_pm.get_configs_dir.return_value = configs_dir
        mock_pm.get_cache_dir.return_value = cache_dir
        mock_pm.get_config_path.return_value = configs_dir / "test" / "execute_pipeline.toml"
        mock_pm.import_project_module.return_value = None
        mock_pm.is_dev_mode.return_value = True
        mock_pm.get_info.return_value = {
            "mode": "development",
            "projects_dir": str(projects_dir),
            "configs_dir": str(configs_dir),
            "cache_dir": str(cache_dir),
            "user_config_dir": str(tmp_path / ".config"),
        }
        yield mock_pm


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_help(self):
        """Test that --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "PytorchImagePipeline CLI" in result.stdout
        assert "run" in result.stdout
        assert "list" in result.stdout
        assert "info" in result.stdout

    def test_cli_help_for_each_command(self):
        """Test that each command has help."""
        commands = ["run", "list", "inspect", "create", "add", "remove", "validate", "info"]
        for cmd in commands:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0


class TestInfoCommand:
    """Test the info command."""

    def test_info_command(self, mock_path_manager):
        """Test info command shows configuration."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "PytorchImagePipeline Configuration" in result.stdout


class TestListCommand:
    """Test the list command."""

    def test_list_empty(self, mock_path_manager):
        """Test list with no pipelines."""
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0


class TestCreateCommand:
    """Test the create command."""

    def test_create_basic_project(self, tmp_path):
        """Test creating a basic project."""
        project_name = "test_project"
        result = runner.invoke(app, ["create", project_name, "--location", str(tmp_path)])

        assert result.exit_code == 0
        assert "created successfully" in result.stdout

        # Check structure
        project_dir = tmp_path / project_name
        assert project_dir.exists()
        assert (project_dir / "README.md").exists()
