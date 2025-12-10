"""Path management for TensorImgPipeline.

Handles environment-aware path resolution for:
- Development mode (editable install)
- Production mode (PyPI install)
- User data directories (XDG Base Directory)

Copyright (C) 2025 Matti Kaupenjohann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import importlib
import os
import sys
import types
from pathlib import Path


class PathManager:
    """Manages paths for both development and production environments.

    All pipelines are sideloaded from user directories:
        - pipelines: ~/.config/tipi/projects
        - configs: ~/.config/tipi/configs
        - cache: ~/.cache/tipi

    Environment variables can override these defaults.
    """

    def __init__(self) -> None:
        self._user_config_dir = self._get_user_config_dir()
        self._ensure_user_directories()

    def _get_user_config_dir(self) -> Path:
        """Get user configuration directory following XDG Base Directory spec.

        Returns:
            Path: User config directory.
        """
        # Check environment variable override
        if override := os.environ.get("TIPI_CONFIG_DIR"):
            return Path(override).expanduser()

        # Use XDG_CONFIG_HOME if set, otherwise default to ~/.config
        config_home = os.environ.get("XDG_CONFIG_HOME")
        base_dir = Path(config_home) if config_home else Path.home() / ".config"

        return base_dir / "tipi"

    def _ensure_user_directories(self) -> None:
        """Ensure user directories exist."""
        # Always create user directories (all pipelines are sideloaded)
        self.get_projects_dir().mkdir(parents=True, exist_ok=True)
        self.get_configs_dir().mkdir(parents=True, exist_ok=True)
        self.get_cache_dir().mkdir(parents=True, exist_ok=True)

    def get_projects_dir(self) -> Path:
        """Get directory for pipeline projects.

        All pipelines are sideloaded from: ~/.config/tipi/projects

        Returns:
            Path: Projects directory.
        """
        # Check environment variable override
        if override := os.environ.get("TIPI_PROJECTS_DIR"):
            return Path(override).expanduser()

        return self._user_config_dir / "projects"

    def get_configs_dir(self) -> Path:
        """Get directory for pipeline configurations.

        All configs are in: ~/.config/tipi/configs

        Returns:
            Path: Configs directory.
        """
        # Check environment variable override
        if override := os.environ.get("TIPI_CONFIG_DIR"):
            return Path(override).expanduser()

        return self._user_config_dir / "configs"

    def get_cache_dir(self) -> Path:
        """Get cache directory for temporary files (git clones).

        Development: ~/.cache/tipi
        Production: ~/.cache/tipi

        Returns:
            Path: Cache directory.
        """
        # Check environment variable override
        if override := os.environ.get("TIPI_CACHE_DIR"):
            return Path(override).expanduser()

        # Use XDG_CACHE_HOME if set, otherwise default to ~/.cache
        cache_home = os.environ.get("XDG_CACHE_HOME")
        base_dir = Path(cache_home) if cache_home else Path.home() / ".cache"

        return base_dir / "tipi"

    def get_project_path(self, project_name: str) -> Path | None:
        """Get path to a specific project.

        Args:
            project_name: Name of the project.

        Returns:
            Path to project if it exists, None otherwise.
        """
        project_path = self.get_projects_dir() / project_name
        if project_path.exists():
            return project_path
        return None

    def get_config_path(self, project_name: str, config_name: str = "pipeline_config.toml") -> Path:
        """Get path to a project's config file.

        Args:
            project_name: Name of the project.
            config_name: Name of the config file.

        Returns:
            Path to config file (may not exist).
        """
        return self.get_configs_dir() / project_name / config_name

    def setup_python_path(self, project_name: str) -> bool:
        """Add project to Python path for imports.

        Args:
            project_name: Name of the project to add to path.

        Returns:
            bool: True if successful, False otherwise.
        """
        project_path = self.get_project_path(project_name)
        if not project_path:
            return False

        # Add parent directory to sys.path so we can import the package
        parent = str(project_path.parent.resolve())
        if parent not in sys.path:
            sys.path.insert(0, parent)

        return True

    def import_project_module(self, project_name: str) -> types.ModuleType:
        """Dynamically import a project module.

        Args:
            project_name: Name of the project to import.

        Returns:
            Module object if successful, None otherwise.
        """
        # Ensure project is on path
        if not self.setup_python_path(project_name):
            raise ImportError(project_name)

        # import the module, which might fail.
        return importlib.import_module(project_name)

    def list_projects(self) -> list[str]:
        """List all available projects.

        Returns:
            List of unique project names.
        """
        projects: list[str] = []
        projects_dir = self.get_projects_dir()

        if not projects_dir.exists():
            return projects

        for item in projects_dir.iterdir():
            # Check if it's a valid Python package (dir with __init__.py, not starting with _)
            if item.is_dir() and not item.name.startswith("_") and (item / "__init__.py").exists():
                projects.append(item.name)

        return sorted(projects)

    def get_info(self) -> dict[str, str]:
        """Get path manager information for debugging.

        Returns:
            Dictionary with path information.
        """
        return {
            "projects_dir": str(self.get_projects_dir()),
            "configs_dir": str(self.get_configs_dir()),
            "cache_dir": str(self.get_cache_dir()),
            "user_config_dir": str(self._user_config_dir),
        }


# Global instance
_path_manager: PathManager | None = None


def get_path_manager() -> PathManager:
    """Get the global PathManager instance.

    Returns:
        PathManager: Global path manager instance.
    """
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager()
    return _path_manager


# Convenience functions
def get_projects_dir() -> Path:
    """Get projects directory."""
    return get_path_manager().get_projects_dir()


def get_configs_dir() -> Path:
    """Get configs directory."""
    return get_path_manager().get_configs_dir()


def get_cache_dir() -> Path:
    """Get cache directory."""
    return get_path_manager().get_cache_dir()
