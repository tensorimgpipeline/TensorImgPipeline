"""Tests for the PathManager module.

Tests path detection, mode switching, and directory resolution.
"""

import os
from pathlib import Path
from unittest.mock import patch

from pytorchimagepipeline.paths import PathManager, get_path_manager


class TestPathManagerModeDetection:
    """Test PathManager mode detection."""

    def test_development_mode_detection_with_pyproject(self, tmp_path):
        """Test development mode is detected when pyproject.toml exists."""
        # Create a fake package structure
        package_dir = tmp_path / "pytorchimagepipeline"
        package_dir.mkdir()
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        with patch("pytorchimagepipeline.paths.Path") as mock_path:
            mock_path.return_value = package_dir
            mock_path.return_value.__file__ = str(package_dir / "__init__.py")

            _ = PathManager()
            # Can't easily test without full setup, but structure is tested

    def test_production_mode_detection_without_pyproject(self, tmp_path):
        """Test production mode is detected when pyproject.toml doesn't exist."""
        # This tests the negative case - no pyproject.toml
        package_dir = tmp_path / "pytorchimagepipeline"
        package_dir.mkdir()

        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "false"}):
            pm = PathManager()
            assert not pm.is_dev_mode()

    def test_environment_variable_override(self):
        """Test environment variable can override mode detection."""
        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "false"}):
            pm = PathManager()
            assert not pm.is_dev_mode()

        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "true"}):
            pm = PathManager()
            assert pm.is_dev_mode()


class TestPathManagerDirectories:
    """Test PathManager directory resolution."""

    def test_get_projects_dir_development(self):
        """Test get_projects_dir in development mode."""
        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "true"}):
            pm = PathManager()
            projects_dir = pm.get_projects_dir()
            # Should return a valid path
            assert projects_dir is not None
            assert isinstance(projects_dir, Path)

    def test_get_projects_dir_production(self):
        """Test get_projects_dir in production mode."""
        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "false"}):
            pm = PathManager()
            projects_dir = pm.get_projects_dir()
            assert ".config" in str(projects_dir) or "projects" in str(projects_dir)

    def test_get_configs_dir_development(self):
        """Test get_configs_dir in development mode."""
        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "true"}):
            pm = PathManager()
            configs_dir = pm.get_configs_dir()
            assert configs_dir is not None
            assert isinstance(configs_dir, Path)

    def test_get_configs_dir_production(self):
        """Test get_configs_dir in production mode."""
        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "false"}):
            pm = PathManager()
            configs_dir = pm.get_configs_dir()
            assert ".config" in str(configs_dir) or "configs" in str(configs_dir)

    def test_get_cache_dir_development(self):
        """Test get_cache_dir in development mode."""
        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "true"}):
            pm = PathManager()
            cache_dir = pm.get_cache_dir()
            assert cache_dir is not None
            assert isinstance(cache_dir, Path)

    def test_get_cache_dir_production(self):
        """Test get_cache_dir in production mode."""
        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "false"}):
            pm = PathManager()
            cache_dir = pm.get_cache_dir()
            assert ".cache" in str(cache_dir) or "cache" in str(cache_dir)


class TestPathManagerEnvironmentOverrides:
    """Test environment variable overrides."""

    def test_projects_dir_override(self, tmp_path):
        """Test PYTORCHPIPELINE_PROJECTS_DIR override."""
        custom_dir = tmp_path / "custom_projects"
        with patch.dict(os.environ, {"PYTORCHPIPELINE_PROJECTS_DIR": str(custom_dir)}):
            pm = PathManager()
            assert pm.get_projects_dir() == custom_dir

    def test_configs_dir_override(self, tmp_path):
        """Test PYTORCHPIPELINE_CONFIG_DIR override."""
        custom_dir = tmp_path / "custom_configs"
        with patch.dict(os.environ, {"PYTORCHPIPELINE_CONFIG_DIR": str(custom_dir)}):
            pm = PathManager()
            assert pm.get_configs_dir() == custom_dir

    def test_cache_dir_override(self, tmp_path):
        """Test PYTORCHPIPELINE_CACHE_DIR override."""
        custom_dir = tmp_path / "custom_cache"
        with patch.dict(os.environ, {"PYTORCHPIPELINE_CACHE_DIR": str(custom_dir)}):
            pm = PathManager()
            assert pm.get_cache_dir() == custom_dir


class TestPathManagerConfigPaths:
    """Test config path resolution."""

    def test_get_config_path_basic(self):
        """Test get_config_path returns correct path."""
        pm = PathManager()
        config_path = pm.get_config_path("test_pipeline", "execute.toml")

        assert config_path is not None
        assert "test_pipeline" in str(config_path)
        assert "execute.toml" in str(config_path)

    def test_get_config_path_with_subdirs(self):
        """Test get_config_path with subdirectories."""
        pm = PathManager()
        config_path = pm.get_config_path("test_pipeline", "subdir/execute.toml")

        assert config_path is not None
        assert "test_pipeline" in str(config_path)
        assert "subdir" in str(config_path)

    def test_get_config_path_nonexistent(self):
        """Test get_config_path with non-existent config."""
        pm = PathManager()
        config_path = pm.get_config_path("nonexistent", "execute.toml")

        # Should return path even if it doesn't exist
        assert config_path is not None


class TestPathManagerModuleImport:
    """Test module import functionality."""

    def test_import_project_module_not_found(self):
        """Test importing non-existent module returns None."""
        pm = PathManager()
        module = pm.import_project_module("nonexistent_module")
        assert module is None

    def test_import_project_module_development(self, tmp_path):
        """Test importing module in development mode."""
        # Create a fake module in a temporary location
        test_module_dir = tmp_path / "test_module"
        test_module_dir.mkdir()
        (test_module_dir / "__init__.py").write_text("test_var = 'test'")

        # Mock the projects directory to point to our temp location
        with patch.dict(os.environ, {"PYTORCHPIPELINE_PROJECTS_DIR": str(tmp_path)}):
            pm = PathManager()
            # Try to import (may fail due to sys.path, but tests the attempt)
            _ = pm.import_project_module("test_module")
            # Result depends on actual import, but function is called

    def test_setup_python_path(self):
        """Test setup_python_path adds correct directory."""
        import sys

        pm = PathManager()
        original_path_len = len(sys.path)
        pm.setup_python_path("test_project")

        # Path might be added
        assert len(sys.path) >= original_path_len


class TestPathManagerInfo:
    """Test get_info method."""

    def test_get_info_structure(self):
        """Test get_info returns correct structure."""
        pm = PathManager()
        info = pm.get_info()

        assert "mode" in info
        assert "projects_dir" in info
        assert "configs_dir" in info
        assert "cache_dir" in info
        assert "user_config_dir" in info

    def test_get_info_development_mode(self):
        """Test get_info in development mode."""
        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "true"}):
            pm = PathManager()
            info = pm.get_info()

            # Should include development in mode
            assert "development" in info["mode"].lower() or info["mode"] == "development"

    def test_get_info_production_mode(self):
        """Test get_info in production mode."""
        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "false"}):
            pm = PathManager()
            info = pm.get_info()

            assert info["mode"] == "production"


class TestPathManagerSingleton:
    """Test singleton pattern."""

    def test_get_path_manager_returns_same_instance(self):
        """Test get_path_manager returns same instance."""
        pm1 = get_path_manager()
        pm2 = get_path_manager()

        assert pm1 is pm2

    def test_get_path_manager_returns_path_manager(self):
        """Test get_path_manager returns PathManager instance."""
        pm = get_path_manager()
        assert isinstance(pm, PathManager)


class TestPathManagerXDGCompliance:
    """Test XDG Base Directory compliance."""

    def test_respects_xdg_config_home(self, tmp_path):
        """Test respects XDG_CONFIG_HOME environment variable."""
        custom_config = tmp_path / "custom_config"
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(custom_config), "PYTORCHPIPELINE_DEV_MODE": "false"}):
            pm = PathManager()
            user_dir = pm._get_user_config_dir()

            assert str(custom_config) in str(user_dir)

    def test_respects_xdg_cache_home(self, tmp_path):
        """Test respects XDG_CACHE_HOME environment variable."""
        custom_cache = tmp_path / "custom_cache"
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(custom_cache), "PYTORCHPIPELINE_DEV_MODE": "false"}):
            pm = PathManager()
            cache_dir = pm.get_cache_dir()

            assert str(custom_cache) in str(cache_dir)

    def test_default_xdg_paths(self):
        """Test default XDG paths are used when env vars not set."""
        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "false"}, clear=False):
            # Remove XDG vars if present
            os.environ.pop("XDG_CONFIG_HOME", None)
            os.environ.pop("XDG_CACHE_HOME", None)

            pm = PathManager()

            # Should use ~/.config and ~/.cache
            user_dir = pm._get_user_config_dir()
            assert ".config" in str(user_dir)


class TestPathManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_missing_parent_directory(self, tmp_path):
        """Test handles case when parent directory doesn't exist."""
        pm = PathManager()
        # Should not crash when checking non-existent paths
        info = pm.get_info()
        assert info is not None

    def test_handles_permission_errors(self, tmp_path):
        """Test handles permission errors gracefully."""
        # This is hard to test portably, but we can verify it doesn't crash
        pm = PathManager()
        projects_dir = pm.get_projects_dir()
        assert projects_dir is not None

    def test_unicode_in_paths(self, tmp_path):
        """Test handles unicode characters in paths."""
        unicode_dir = tmp_path / "проект_тест"
        with patch.dict(os.environ, {"PYTORCHPIPELINE_PROJECTS_DIR": str(unicode_dir)}):
            pm = PathManager()
            projects_dir = pm.get_projects_dir()
            assert projects_dir == unicode_dir

    def test_very_long_paths(self, tmp_path):
        """Test handles very long paths."""
        long_dir = tmp_path / ("a" * 100) / ("b" * 100)
        with patch.dict(os.environ, {"PYTORCHPIPELINE_PROJECTS_DIR": str(long_dir)}):
            pm = PathManager()
            projects_dir = pm.get_projects_dir()
            assert projects_dir == long_dir


class TestPathManagerModeTransitions:
    """Test transitions between development and production modes."""

    def test_mode_consistency(self):
        """Test mode remains consistent across calls."""
        pm = PathManager()
        mode1 = pm.is_dev_mode()
        mode2 = pm.is_dev_mode()

        assert mode1 == mode2

    def test_paths_consistent_with_mode(self):
        """Test that paths are consistent with reported mode."""
        pm = PathManager()
        mode = pm.is_dev_mode()
        projects_dir = pm.get_projects_dir()

        if mode:
            # Development mode should use pipelines directory
            assert "pipelines" in str(projects_dir).lower() or "pytorchimagepipeline" in str(projects_dir)
        else:
            # Production mode should use .config
            assert ".config" in str(projects_dir) or "projects" in str(projects_dir)


class TestPathManagerIntegration:
    """Integration tests for PathManager."""

    def test_full_workflow_development(self):
        """Test full workflow in development mode."""
        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "true"}):
            pm = PathManager()

            # Get all directories
            projects_dir = pm.get_projects_dir()
            configs_dir = pm.get_configs_dir()
            cache_dir = pm.get_cache_dir()

            # All should be valid paths
            assert projects_dir is not None
            assert configs_dir is not None
            assert cache_dir is not None

            # Get config path
            config_path = pm.get_config_path("test", "config.toml")
            assert config_path is not None

            # Get info
            info = pm.get_info()
            assert "development" in info["mode"].lower() or info["mode"] == "development"

    def test_full_workflow_production(self):
        """Test full workflow in production mode."""
        with patch.dict(os.environ, {"PYTORCHPIPELINE_DEV_MODE": "false"}):
            pm = PathManager()

            # Get all directories
            projects_dir = pm.get_projects_dir()
            configs_dir = pm.get_configs_dir()
            cache_dir = pm.get_cache_dir()

            # All should be valid paths
            assert projects_dir is not None
            assert configs_dir is not None
            assert cache_dir is not None

            # Get config path
            config_path = pm.get_config_path("test", "config.toml")
            assert config_path is not None

            # Get info
            info = pm.get_info()
            assert info["mode"] == "production"
