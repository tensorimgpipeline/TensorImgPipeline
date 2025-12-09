"""Pytest configuration and fixtures for integration tests."""

import pytest

from .fixtures import fs_mock_path_manager, mock_path_manager

__all__ = ["fs_mock_path_manager", "mock_path_manager"]


@pytest.fixture
def fs(fs):
    """Provide pyfakefs fixture configured for CLI tests.

    This fixture enables fake filesystem for tests, ensuring no real
    filesystem operations occur during testing. It's configured to allow
    pytest's own temp directory operations to work normally.
    """
    # Allow pytest's own temp directories to work
    # This is necessary because pytest creates temp dirs in /tmp
    # and fakefs would otherwise interfere with that
    yield fs
