"""Color utilities for plot customization."""

import matplotlib.colors as mcolors
import seaborn as sns


def adjust_brightness(color, factor):
    """
    Adjust the brightness of a color.

    Parameters
    ----------
    color : color-like
        The color to adjust (any format accepted by matplotlib)
    factor : float
        Brightness adjustment factor. Values > 1.0 make it brighter,
        values < 1.0 make it darker.

    Returns
    -------
    tuple
        RGB tuple with adjusted brightness
    """
    # Convert color to RGB
    rgb = mcolors.to_rgb(color)
    # Convert to HSV for brightness adjustment
    hsv = mcolors.rgb_to_hsv(rgb)
    # Adjust brightness (V component)
    hsv[2] = min(1.0, hsv[2] * factor)
    # Convert back to RGB
    rgb_adjusted = mcolors.hsv_to_rgb(hsv)
    return rgb_adjusted


def get_color_palette(n_colors, palette_name="husl"):
    """
    Generate a color palette with high contrast between colors.

    Parameters
    ----------
    n_colors : int
        Number of colors to generate
    palette_name : str, optional
        Name of the seaborn palette to use (default: "husl")

    Returns
    -------
    list
        List of RGB tuples
    """
    return sns.color_palette(palette_name, n_colors)


def create_sample_colors(base_color, std_brightness=1.1, minmax_brightness=0.9):
    """
    Create a color scheme for a sample with variations for mean, std, and min/max.

    Parameters
    ----------
    base_color : color-like
        The base color for the mean line
    std_brightness : float, optional
        Brightness factor for std bands (default: 1.1 = 10% brighter)
    minmax_brightness : float, optional
        Brightness factor for min/max bands (default: 0.9 = 10% darker)

    Returns
    -------
    dict
        Dictionary with 'mean', 'std', and 'minmax' color keys
    """
    return {
        "mean": base_color,
        "std": adjust_brightness(base_color, std_brightness),
        "minmax": adjust_brightness(base_color, minmax_brightness),
    }
