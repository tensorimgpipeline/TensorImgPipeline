import base64
import io
import os
import socket
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from tipi import paths


def is_running_in_container() -> bool:
    """
    Checks if the code is running inside a known container environment.
    """
    # Check for Custom Environment Variable
    if os.environ.get("IS_IN_CONTAINER", "").lower() in ("true", "1", "yes"):
        return True

    # GitHub Actions sets GITHUB_ACTIONS to 'true'
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return True

    # Check for Codespaces/DevContainer environment variables
    if os.environ.get("CODESPACES") == "true":
        return True

    # Check for VS Code Remote Container environment variable
    return os.environ.get("REMOTE_CONTAINERS") == "true"


# Pytest marker to skip tests when not running in container/CI environment
skip_outside_container = pytest.mark.skipif(
    not is_running_in_container(),
    reason="Test only runs in container/CI environment (set IS_IN_CONTAINER=true to run locally)",
)


def is_connected_to_github() -> bool:
    """
    Checks if github.com is reachable and provides a valid ip.
    """
    try:
        socket.gethostbyname("www.github.com")
    except socket.gaierror:
        return False
    else:
        return True


skip_no_network = pytest.mark.skipif(not is_connected_to_github(), reason="Github.com is not reachable.")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "skip_outside_container: mark test to only run in container/CI environment",
    )


@pytest.fixture
def isolated_path_manager(tmp_path, monkeypatch):
    """Fixture that isolates PathManager to use temporary directories.

    This fixture:
    - Sets environment variables to redirect all paths to tmp_path
    - Resets the global PathManager instance before the test
    - Cleans up sys.path and sys.modules after the test
    - Restores the original PathManager after the test

    Usage:
        def test_something(isolated_path_manager):
            path_mgr, dirs = isolated_path_manager
            # dirs.projects, dirs.configs, dirs.cache are the temp directories
            # path_mgr is the isolated PathManager instance
    """

    # Store original state
    original_path_manager = paths._path_manager
    original_sys_path = sys.path.copy()
    original_sys_modules = set(sys.modules.keys())

    # Create isolated directory structure
    projects_dir = tmp_path / "projects"
    configs_dir = tmp_path / "configs"
    cache_dir = tmp_path / "cache"

    projects_dir.mkdir()
    configs_dir.mkdir()
    cache_dir.mkdir()

    # Set environment variables to use temp directories
    monkeypatch.setenv("TIPI_PROJECTS_DIR", str(projects_dir))
    monkeypatch.setenv("TIPI_CONFIG_DIR", str(configs_dir))
    monkeypatch.setenv("TIPI_CACHE_DIR", str(cache_dir))

    # Reset global path manager to pick up new env vars
    paths._path_manager = None

    # Create a simple namespace to hold directory paths
    class Dirs:
        pass

    dirs = Dirs()
    dirs.root = tmp_path
    dirs.projects = projects_dir
    dirs.configs = configs_dir
    dirs.cache = cache_dir

    # Get the new isolated path manager
    path_mgr = paths.get_path_manager()

    yield path_mgr, dirs

    # Teardown: restore original state

    # Remove any modules that were imported during the test
    new_modules = set(sys.modules.keys()) - original_sys_modules
    for module_name in list(new_modules):
        # Remove all newly imported modules to prevent state pollution
        # This includes test pipeline modules that were dynamically loaded
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Restore sys.path
    sys.path[:] = original_sys_path

    # Restore original path manager
    paths._path_manager = original_path_manager


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    # Handle image embedding only for test execution phase
    if call.when == "call":
        # Get the pytest_html plugin (if available)
        pytest_html = item.config.pluginmanager.getplugin("html")

        html_content = '<div style="display: flex; gap: 20px; flex-wrap: wrap;">'
        # Extract images tensor from user_properties
        for prop in item.user_properties:
            if prop[0] == "image_tensor" and len(prop) == 2:
                img_tensor = prop[1]
                html_content += '<figure style="margin: 0;">'
            elif prop[0] == "image_tensor" and len(prop) == 3:
                tensor_name = prop[1]
                img_tensor = prop[2]
                html_content += f'<figure style="margin: 0;"><figcaption>{tensor_name}</figcaption>'
            else:
                break
            if img_tensor is not None and pytest_html:
                # Convert tensor to numpy array
                if hasattr(img_tensor, "cpu"):
                    img_tensor = img_tensor.cpu().detach()
                img_np = img_tensor.numpy() if hasattr(img_tensor, "numpy") else np.array(img_tensor)

                # Handle tensor shape (C, H, W) -> (H, W, C)
                if img_np.shape[0] in [1, 3]:
                    img_np = img_np.transpose(1, 2, 0)

                if img_np.dtype == np.float32 and img_np.max() <= 1:
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)

                # Convert to base64-encoded image
                buffer = io.BytesIO()
                img = Image.fromarray(img_np)
                if len(img_tensor.shape) == 2:
                    img = img.convert("P")
                    cmap = plt.get_cmap("tab20", 22)
                    colors = cmap(np.arange(22))  # Shape (n_classes, 4) [RGBA]

                    # Convert to 8-bit RGB and flatten into a list
                    palette = (colors[:, :3] * 255).astype(np.uint8).flatten().tolist()

                    # Set Background and Ignore colors
                    palette[:3] = [0, 0, 0]
                    palette.extend([0, 0, 0] * (256 - len(palette) // 3))
                    palette[-3:] = [224, 224, 192]
                    img.putpalette(palette)

                img.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                # Add to HTML report using pytest-html's extras system
                html_content += f'<img style="padding: 5px" src="data:image/png;base64,{img_b64}"/></figure>'
            report.extras = [*getattr(report, "extra", []), pytest_html.extras.html(html_content)]


def pytest_html_report_title(report):
    report.title = "Tensor Image Pipeline Test Report"
