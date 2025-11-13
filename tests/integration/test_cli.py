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
        commands = ["run", "list", "inspect", "create", "add", "remove", "clean", "validate", "info"]
        for cmd in commands:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0


class TestInfoCommand:
    """Test the info command."""

    @pytest.mark.usefixtures("mock_path_manager")
    def test_info_command(self):
        """Test info command shows configuration."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "PytorchImagePipeline Configuration" in result.stdout


class TestListCommand:
    """Test the list command."""

    @pytest.mark.usefixtures("mock_path_manager")
    def test_list_empty(self):
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


class TestCleanCommand:
    """Test the clean command."""

    @pytest.mark.usefixtures("mock_path_manager")
    def test_clean_no_broken_links(self):
        """Test clean command when there are no broken links."""
        result = runner.invoke(app, ["clean"])
        assert result.exit_code == 0
        assert "No broken symlinks found" in result.stdout

    def test_clean_with_broken_link(self, mock_path_manager):
        """Test clean command removes broken symlinks."""
        projects_dir = mock_path_manager.get_projects_dir()

        # Create a broken symlink
        broken_link = projects_dir / "broken_pipeline"
        non_existent_target = projects_dir / "non_existent_target"
        broken_link.symlink_to(non_existent_target)

        assert broken_link.exists(follow_symlinks=False)  # Link exists
        assert not broken_link.exists()  # But target doesn't

        # Run clean with dry-run first
        result = runner.invoke(app, ["clean", "--dry-run"])
        assert result.exit_code == 0
        assert "Found 1 broken symlink" in result.stdout
        assert "broken_pipeline" in result.stdout
        assert "Dry run" in result.stdout
        assert broken_link.exists(follow_symlinks=False)  # Link still exists

        # Run clean to actually remove
        result = runner.invoke(app, ["clean"])
        assert result.exit_code == 0
        assert "Found 1 broken symlink" in result.stdout
        assert "Removed: broken_pipeline" in result.stdout
        assert "Cleaned up 1/1 broken symlink" in result.stdout
        assert not broken_link.exists(follow_symlinks=False)  # Link removed

    def test_clean_verbose(self, mock_path_manager, tmp_path):
        """Test clean command with verbose flag shows valid symlinks."""
        projects_dir = mock_path_manager.get_projects_dir()

        # Create a valid symlink
        valid_target = tmp_path / "valid_target"
        valid_target.mkdir()
        valid_link = projects_dir / "valid_pipeline"
        valid_link.symlink_to(valid_target)

        result = runner.invoke(app, ["clean", "--verbose"])
        assert result.exit_code == 0
        assert "Valid symlinks" in result.stdout
        assert "valid_pipeline" in result.stdout
        assert "No broken symlinks found" in result.stdout

    def test_clean_multiple_broken_links(self, mock_path_manager):
        """Test clean command with multiple broken symlinks."""
        projects_dir = mock_path_manager.get_projects_dir()
        configs_dir = mock_path_manager.get_configs_dir()

        # Create broken symlinks in both directories
        broken_project = projects_dir / "broken_project"
        broken_project.symlink_to(projects_dir / "missing1")

        broken_config = configs_dir / "broken_config"
        broken_config.symlink_to(configs_dir / "missing2")

        result = runner.invoke(app, ["clean"])
        assert result.exit_code == 0
        assert "Found 2 broken symlink" in result.stdout
        assert "broken_project" in result.stdout
        assert "broken_config" in result.stdout
        assert "Cleaned up 2/2 broken symlink" in result.stdout
