from __future__ import annotations

from pathlib import Path
from typing import Any

METRIC_SPECS = (
    ("mean_squared_error", "MSE", "cornflowerblue"),
    ("r2_score", r"$R^2$", "darkseagreen"),
    ("mean_cosine_similarity", "Mean cosine", "rosybrown"),
)

TEXT_COLOR = "#2f2f2f"
GRID_COLOR = "#d8d8d8"
SPINE_COLOR = "#777777"


def write_layerwise_metrics_plot(
    path: str | Path,
    metrics: list[dict[str, Any]],
    title: str = "Layerwise Probe Metrics",
) -> None:
    """Write a PNG plot of reconstruction metrics by layer."""

    if not metrics:
        raise ValueError("Cannot plot an empty metrics list.")

    sorted_metrics = sorted(metrics, key=lambda row: int(row["layer_index"]))
    layers = [int(row["layer_index"]) for row in sorted_metrics]

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.edgecolor": SPINE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=15, fontweight="semibold")

    for axis, (metric_name, label, color) in zip(axes, METRIC_SPECS, strict=True):
        values = [_require_float(row, metric_name) for row in sorted_metrics]
        axis.plot(
            layers,
            values,
            color=color,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            linewidth=2.2,
        )
        axis.set_xlabel("Layer")
        axis.set_ylabel(label)
        axis.set_title(label, fontsize=11, pad=8)
        axis.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.65)
        axis.set_xticks(layers)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(SPINE_COLOR)
        axis.spines["bottom"].set_color(SPINE_COLOR)

    fig.savefig(target, dpi=200)
    plt.close(fig)


def _require_float(row: dict[str, Any], metric_name: str) -> float:
    """Return a required metric value as float."""

    if metric_name not in row:
        layer = row.get("layer_index", "<unknown>")
        raise ValueError(f"Metric row for layer {layer} is missing {metric_name}.")
    return float(row[metric_name])
