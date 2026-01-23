"""Data loading and preprocessing utilities."""

import json

import numpy as np


def load_data_from_json(json_path):
    """
    Load benchmark data from JSON file and compute statistics.

    Parameters
    ----------
    json_path : str or Path
        Path to the JSON file containing benchmark data

    Returns
    -------
    dict
        Dictionary with 'mean', 'std', 'min', 'max' keys, each containing
        a dict mapping sample names to their computed statistics
    """
    with open(json_path) as f:
        raw_data = json.load(f)

    # Initialize data structure
    data = {
        "mean": {},
        "std": {},
        "min": {},
        "max": {},
    }

    # Process each benchmark
    for benchmark in raw_data["benchmarks"]:
        sample_name = benchmark["metadata"]["name"]

        # Collect all values from all runs
        all_values = []
        for run in benchmark["runs"]:
            if "values" in run:
                all_values.extend(run["values"])

        # Convert to numpy array for statistics
        values_array = np.array(all_values)

        # Store statistics as single values (not lists)
        data["mean"][sample_name] = np.mean(values_array)
        data["std"][sample_name] = np.std(values_array)
        data["min"][sample_name] = np.min(values_array)
        data["max"][sample_name] = np.max(values_array)

    return data


def prepare_plot_data(data):
    """
    Convert the statistical data dictionary into a format suitable for plotting.

    Groups samples by their base name (without size suffix) and creates arrays
    of values for plotting.

    Parameters
    ----------
    data : dict
        Dictionary with 'mean', 'std', 'min', 'max' keys from load_data_from_json

    Returns
    -------
    dict
        Dictionary with 'mean', 'std', 'min', 'max', 'sizes' keys, each containing
        a dict mapping base sample names to lists of values. 'sizes' contains the
        loop sizes extracted from benchmark names.

    Examples
    --------
    Input: {"mean": {"Baseline (size=100)": 1.5, "Baseline (size=1000)": 2.3}}
    Output: {"mean": {"Baseline": [1.5, 2.3]}, "sizes": {"Baseline": [100, 1000]}}
    """
    # Group benchmarks by their base name (Baseline, Decorated, Manual)
    grouped = {}

    for sample_name in data["mean"]:
        # Extract base name (e.g., "Baseline" from "Baseline (size=100)")
        base_name = sample_name.split(" (")[0]

        # Extract size from name (e.g., 100 from "Baseline (size=100)")
        try:
            size = int(sample_name.split("size=")[1].rstrip(")"))
        except (IndexError, ValueError):
            size = 0  # Default if size cannot be extracted

        if base_name not in grouped:
            grouped[base_name] = {
                "samples": [],
                "sizes": [],
                "mean": [],
                "std": [],
                "min": [],
                "max": [],
            }

        grouped[base_name]["samples"].append(sample_name)
        grouped[base_name]["sizes"].append(size)
        grouped[base_name]["mean"].append(data["mean"][sample_name])
        grouped[base_name]["std"].append(data["std"][sample_name])
        grouped[base_name]["min"].append(data["min"][sample_name])
        grouped[base_name]["max"].append(data["max"][sample_name])

    # Convert to plot-ready format
    plot_data = {
        "mean": {},
        "std": {},
        "min": {},
        "max": {},
        "sizes": {},
    }

    for base_name, values in grouped.items():
        # Sort by size to ensure proper x-axis ordering
        sorted_indices = np.argsort(values["sizes"])

        plot_data["sizes"][base_name] = [values["sizes"][i] for i in sorted_indices]
        plot_data["mean"][base_name] = [values["mean"][i] for i in sorted_indices]
        plot_data["std"][base_name] = [values["std"][i] for i in sorted_indices]
        plot_data["min"][base_name] = [values["min"][i] for i in sorted_indices]
        plot_data["max"][base_name] = [values["max"][i] for i in sorted_indices]

    return plot_data
