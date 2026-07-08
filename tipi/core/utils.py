"""Core utility functions for TensorImgPipeline."""

import colorsys
import importlib
import types

import numpy as np
from matplotlib.colors import hex2color, rgb2hex

INSTALL_MAPPING: dict[str, str] = {}

PACKAGE_EXTRAS = {
    "wandb": "TensorImagePipeline[wandb]",
    "tensorboard": "TensorImagePipeline[tensorboard]",
}


def create_color(hex_colors: list[str]) -> str:
    """Create a new color by finding the largest gap in HSV color space.

    Args:
        hex_colors: List of existing colors in hex format (e.g., ['#FF0000', '#00FF00'])

    Returns:
        str: A new color in hex format that fills the largest gap in hue space
    """
    rgb2hsv = np.vectorize(colorsys.rgb_to_hsv)
    # Convert to RGB
    rgb_colors = [hex2color(color) for color in hex_colors]

    # Convert to numpy array
    colors_array = np.array(rgb_colors)

    # Convert to HSV
    r, g, b = colors_array.T
    h, s, v = rgb2hsv(r, g, b)
    hsv = np.stack([h, s, v], axis=-1)

    # To create a new color we place them in the middle between two colors with the biggest gap
    # based on hue.
    hsv.sort(axis=0)
    distance_between = np.diff(hsv, axis=0)
    selected_distance = np.argmax(distance_between[:, 0])
    selected_color = hsv[selected_distance]
    selected_color[0] = selected_color[0] + (distance_between[selected_distance] / 2)[0]

    return rgb2hex(colorsys.hsv_to_rgb(*selected_color))


def import_optional_dependency(name: str) -> types.ModuleType:
    """Imports optional dependencies of this library.

    It raises with a nice error msg, if users try to launch a depending module, which depends on it.

    Args:
        name (str): name of the module

    Returns:
        types.ModuleType | None: the imported module
    """

    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    msg = (
        f"`Import {install_name}` failed. "
        f"If intended to use this module please install this Package with extras: "
        f"pip install tensorimagepipeline[{install_name}] or pip install tensorimagepipeline[all]"
    )

    try:
        module = importlib.import_module(name)
    except ImportError as err:
        raise ImportError(msg) from err

    return module
