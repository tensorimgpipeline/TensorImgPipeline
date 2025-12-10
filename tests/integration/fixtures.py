from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_path_manager(tmp_path):
    """Create a mock PathManager with temporary directories."""
    projects_dir = tmp_path / "projects"
    configs_dir = tmp_path / "configs"
    cache_dir = tmp_path / "cache"

    projects_dir.mkdir()
    configs_dir.mkdir()
    cache_dir.mkdir()

    with patch("tipi.cli.path_manager") as mock_pm:
        mock_pm.get_projects_dir.return_value = projects_dir
        mock_pm.get_configs_dir.return_value = configs_dir
        mock_pm.get_cache_dir.return_value = cache_dir
        mock_pm.get_config_path.return_value = configs_dir / "test" / "execute_pipeline.toml"
        mock_pm.import_project_module.return_value = None
        mock_pm.get_info.return_value = {
            "projects_dir": str(projects_dir),
            "configs_dir": str(configs_dir),
            "cache_dir": str(cache_dir),
            "user_config_dir": str(tmp_path / ".config"),
        }
        yield mock_pm


@pytest.fixture
def fs_mock_path_manager(fs):
    """Create a mock PathManager with fakefs directories.

    This fixture is specifically for tests using fakefs to avoid conflicts
    between fakefs and pytest's tmp_path fixture.
    """

    # Use fakefs paths instead of tmp_path
    projects_dir = Path("/fake_test_root/projects")
    configs_dir = Path("/fake_test_root/configs")
    cache_dir = Path("/fake_test_root/cache")

    projects_dir.mkdir(parents=True)
    configs_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    with patch("tipi.cli.path_manager") as mock_pm:
        mock_pm.get_projects_dir.return_value = projects_dir
        mock_pm.get_configs_dir.return_value = configs_dir
        mock_pm.get_cache_dir.return_value = cache_dir
        mock_pm.get_config_path.return_value = configs_dir / "test" / "execute_pipeline.toml"
        mock_pm.import_project_module.return_value = None
        mock_pm.get_info.return_value = {
            "projects_dir": str(projects_dir),
            "configs_dir": str(configs_dir),
            "cache_dir": str(cache_dir),
            "user_config_dir": "/fake_test_root/.config",
        }
        yield mock_pm
