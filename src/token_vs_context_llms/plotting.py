from __future__ import annotations

from pathlib import Path
from typing import Any

METRIC_SPECS = (
    ("mean_squared_error", "MSE"),
    ("r2_score", r"$R^2$"),
    ("mean_cosine_similarity", "Mean cosine"),
)


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

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    fig.suptitle(title)

    for axis, (metric_name, label) in zip(axes, METRIC_SPECS, strict=True):
        values = [_require_float(row, metric_name) for row in sorted_metrics]
        axis.plot(layers, values, marker="o", linewidth=1.8)
        axis.set_xlabel("Layer")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.set_xticks(layers)

    fig.savefig(target, dpi=200)
    plt.close(fig)


def _require_float(row: dict[str, Any], metric_name: str) -> float:
    """Return a required metric value as float."""

    if metric_name not in row:
        layer = row.get("layer_index", "<unknown>")
        raise ValueError(f"Metric row for layer {layer} is missing {metric_name}.")
    return float(row[metric_name])
