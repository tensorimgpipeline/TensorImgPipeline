"""Core plotting functions for statistical visualization."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline

from tipi.benchmarks.colors import create_sample_colors, get_color_palette


def smooth_data(x, y, num_points=300, log_space=False):
    """
    Smooth data using cubic spline interpolation.

    Parameters
    ----------
    x : array-like
        X coordinates
    y : array-like
        Y coordinates
    num_points : int, optional
        Number of points in the smoothed curve (default: 300)
    log_space : bool, optional
        If True, use logarithmic spacing for x_smooth (default: False)
        Use this when x-values span multiple orders of magnitude

    Returns
    -------
    tuple
        (x_smooth, y_smooth) - smoothed x and y coordinates
    """
    # Use lower order spline for small datasets
    k = min(3, len(x) - 1)

    # Create smoothed x-axis based on spacing type
    if log_space and x.min() > 0:
        x_smooth = np.logspace(np.log10(x.min()), np.log10(x.max()), num_points)
    else:
        x_smooth = np.linspace(x.min(), x.max(), num_points)

    spl = make_interp_spline(x, y, k=k)
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth


def plot_all_samples(data, smooth=True, figsize=(12, 7), palette="husl"):
    """
    Create a line plot with all samples showing mean, std bands, and min/max ranges.

    Parameters
    ----------
    data : dict
        Dictionary with 'mean', 'std', 'min', 'max' keys, each containing
        a dict mapping sample names to lists of values
    smooth : bool, optional
        Whether to apply spline smoothing to the curves (default: True)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 7))
    palette : str, optional
        Seaborn color palette name (default: "husl")

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Set up seaborn style
    sns.set_style("whitegrid")

    # Get all samples dynamically
    samples = list(data["mean"].keys())
    n_samples = len(samples)

    # Use a high-contrast color palette
    base_colors = get_color_palette(n_samples, palette)

    # Create color variations for each sample
    colors = {}
    for i, sample in enumerate(samples):
        colors[sample] = create_sample_colors(base_colors[i])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Detect if we need log spacing for smoothing
    use_log_space = False
    if "sizes" in data and len(data["sizes"]) > 0:
        first_sample = next(iter(data["sizes"].keys()))
        sizes = data["sizes"][first_sample]
        if len(sizes) > 1 and max(sizes) / min(sizes) > 10:
            use_log_space = True

    for sample in samples:
        mean_vals = data["mean"][sample]
        std_vals = data["std"][sample]
        min_vals = data["min"][sample]
        max_vals = data["max"][sample]

        # Use sizes for x-axis if available, otherwise use sequential points
        if "sizes" in data and sample in data["sizes"]:
            x = np.array(data["sizes"][sample])
        else:
            x = np.arange(len(mean_vals))

        # Calculate std bounds
        mean_array = np.array(mean_vals)
        std_array = np.array(std_vals)
        upper_std = mean_array + std_array
        lower_std = mean_array - std_array

        sample_label = sample.replace("_", " ").title()
        color = colors[sample]

        # Prepare data (smooth or original)
        if smooth:
            x_plot, mean_plot = smooth_data(x, mean_array, log_space=use_log_space)
            _, upper_std_plot = smooth_data(x, upper_std, log_space=use_log_space)
            _, lower_std_plot = smooth_data(x, lower_std, log_space=use_log_space)
            _, min_plot = smooth_data(x, np.array(min_vals), log_space=use_log_space)
            _, max_plot = smooth_data(x, np.array(max_vals), log_space=use_log_space)
            marker_kwargs = {}
        else:
            x_plot = x
            mean_plot = mean_array
            upper_std_plot = upper_std
            lower_std_plot = lower_std
            min_plot = np.array(min_vals)
            max_plot = np.array(max_vals)
            marker_kwargs = {"marker": "o", "markersize": 4}

        # Plot mean line
        ax.plot(
            x_plot,
            mean_plot,
            linewidth=2.5,
            label=f"{sample_label} Mean",
            color=color["mean"],
            **({**marker_kwargs, "markersize": 6} if not smooth else {}),
        )

        # Plot std bands with dashed lines
        ax.plot(
            x_plot,
            upper_std_plot,
            linestyle="--",
            linewidth=1.2,
            color=color["std"],
            alpha=0.7,
            **({**marker_kwargs, "marker": "s"} if not smooth else {}),
        )
        ax.plot(
            x_plot,
            lower_std_plot,
            linestyle="--",
            linewidth=1.2,
            color=color["std"],
            alpha=0.7,
            label=f"{sample_label} ± Std",
            **({**marker_kwargs, "marker": "s"} if not smooth else {}),
        )
        ax.fill_between(
            x_plot,
            lower_std_plot,
            upper_std_plot,
            alpha=0.15,
            color=color["std"],
        )

        # Plot min/max with dotted lines
        ax.plot(
            x_plot,
            min_plot,
            linestyle=":",
            linewidth=1.2,
            color=color["minmax"],
            alpha=0.7,
            **({**marker_kwargs, "marker": "^"} if not smooth else {}),
        )
        ax.plot(
            x_plot,
            max_plot,
            linestyle=":",
            linewidth=1.2,
            color=color["minmax"],
            alpha=0.7,
            label=f"{sample_label} Min/Max",
            **({**marker_kwargs, "marker": "v"} if not smooth else {}),
        )
        ax.fill_between(x_plot, min_plot, max_plot, alpha=0.08, color=color["minmax"])

    # Customize plot
    ax.set_xlabel("Loop Size", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("All Samples - Mean with Std and Min/Max Bands", fontsize=14, fontweight="bold")

    # Use log scale for x-axis if sizes span multiple orders of magnitude
    if "sizes" in data and len(data["sizes"]) > 0:
        first_sample = next(iter(data["sizes"].keys()))
        sizes = data["sizes"][first_sample]
        if len(sizes) > 1 and max(sizes) / min(sizes) > 10:
            ax.set_xscale("log")
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sample(sample_name, mean_vals, std_vals, min_vals, max_vals, smooth=True, figsize=(10, 6)):
    """
    Create a line plot for a single sample with mean, std bands, and min/max ranges.

    Parameters
    ----------
    sample_name : str
        Name of the sample
    mean_vals : array-like
        Mean values
    std_vals : array-like
        Standard deviation values
    min_vals : array-like
        Minimum values
    max_vals : array-like
        Maximum values
    smooth : bool, optional
        Whether to apply spline smoothing to the curves (default: True)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 6))

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Set up seaborn style
    sns.set_style("whitegrid")

    # Create x-axis (assuming sequential points)
    x = np.arange(len(mean_vals))

    # Calculate std bounds
    mean_array = np.array(mean_vals)
    std_array = np.array(std_vals)
    upper_std = mean_array + std_array
    lower_std = mean_array - std_array

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data (smooth or original)
    if smooth:
        x_plot, mean_plot = smooth_data(x, mean_array)
        _, upper_std_plot = smooth_data(x, upper_std)
        _, lower_std_plot = smooth_data(x, lower_std)
        _, min_plot = smooth_data(x, np.array(min_vals))
        _, max_plot = smooth_data(x, np.array(max_vals))
        marker_kwargs = {}
    else:
        x_plot = x
        mean_plot = mean_array
        upper_std_plot = upper_std
        lower_std_plot = lower_std
        min_plot = np.array(min_vals)
        max_plot = np.array(max_vals)
        marker_kwargs = {"marker": "o"}

    # Plot mean line
    ax.plot(x_plot, mean_plot, linewidth=2.5, label="Mean", color="#2E86AB", **marker_kwargs)

    # Plot std bands with dashed lines
    ax.plot(
        x_plot,
        upper_std_plot,
        linestyle="--",
        linewidth=1.5,
        color="#A23B72",
        alpha=0.8,
        label="Mean ± Std",
        **({**marker_kwargs, "marker": "s"} if not smooth else {}),
    )
    ax.plot(
        x_plot,
        lower_std_plot,
        linestyle="--",
        linewidth=1.5,
        color="#A23B72",
        alpha=0.8,
        **({**marker_kwargs, "marker": "s"} if not smooth else {}),
    )
    ax.fill_between(x_plot, lower_std_plot, upper_std_plot, alpha=0.2, color="#A23B72")

    # Plot min/max with dotted lines
    ax.plot(
        x_plot,
        min_plot,
        linestyle=":",
        linewidth=1.5,
        color="#F18F01",
        alpha=0.8,
        label="Min/Max Range",
        **({**marker_kwargs, "marker": "^"} if not smooth else {}),
    )
    ax.plot(
        x_plot,
        max_plot,
        linestyle=":",
        linewidth=1.5,
        color="#F18F01",
        alpha=0.8,
        **({**marker_kwargs, "marker": "v"} if not smooth else {}),
    )
    ax.fill_between(x_plot, min_plot, max_plot, alpha=0.1, color="#F18F01")

    # Customize plot
    ax.set_xlabel("Data Point", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(
        f"{sample_name.replace('_', ' ').title()} - Mean with Std and Min/Max Bands",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
