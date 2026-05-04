from pathlib import Path

from tipi.benchmarks.data import load_data_from_json, prepare_plot_data
from tipi.benchmarks.plotting import plot_all_samples

# Load and prepare data
raw_data = load_data_from_json(Path("benchmarks/results.json"))
data = prepare_plot_data(raw_data)

# Create plot
fig = plot_all_samples(
    data,
    smooth=False,
    figsize=(24, 14),
    palette="husl",
)

# Save plot
fig.savefig("benchmarks/results.png", dpi=600, bbox_inches="tight")
