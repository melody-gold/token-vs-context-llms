from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from token_vs_context_llms.diagnostics import ProbeDiagnostics

METRIC_SPECS = (
    ("mean_squared_error", "MSE", "cornflowerblue"),
    ("r2_score", r"$R^2$", "darkseagreen"),
    ("mean_cosine_similarity", "Mean cosine", "rosybrown"),
)

TEXT_COLOR = "#2f2f2f"
GRID_COLOR = "#d8d8d8"
SPINE_COLOR = "#777777"
CONTINUOUS_CMAP = "magma"
LAYER_PALETTE = (
    "cornflowerblue",
    "darkseagreen",
    "rosybrown",
    "mediumvioletred",
    "darkturquoise",
    "goldenrod",
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


def write_diagnostic_plots(
    output_dir: str | Path,
    diagnostics: ProbeDiagnostics,
    title_prefix: str = "Probe Diagnostics",
) -> None:
    """Write optional exploratory figures from per-token diagnostics."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_paper_style(plt)
    _write_cosine_boxplot(target / "cosine_boxplot.png", diagnostics, plt, title_prefix)
    _write_error_by_position(target / "mse_by_position.png", diagnostics, plt, title_prefix)
    _write_baseline_comparison(target / "baseline_comparison.png", diagnostics, plt, title_prefix)
    _write_norm_vs_error(target / "norm_vs_error.png", diagnostics, plt, title_prefix)
    _write_sequence_heatmap(target / "token_layer_heatmap.png", diagnostics, plt, title_prefix)


def _require_float(row: dict[str, Any], metric_name: str) -> float:
    """Return a required metric value as float."""

    if metric_name not in row:
        layer = row.get("layer_index", "<unknown>")
        raise ValueError(f"Metric row for layer {layer} is missing {metric_name}.")
    return float(row[metric_name])


def _apply_paper_style(plt: Any) -> None:
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


def _write_cosine_boxplot(
    path: Path,
    diagnostics: ProbeDiagnostics,
    plt: Any,
    title_prefix: str,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    data = [diagnostics.heldout_cosine[index] for index in range(len(diagnostics.layer_indices))]
    box = axis.boxplot(
        data,
        patch_artist=True,
        labels=[str(int(layer)) for layer in diagnostics.layer_indices],
        medianprops={"color": TEXT_COLOR, "linewidth": 1.4},
        boxprops={"color": "cornflowerblue", "linewidth": 1.4},
        whiskerprops={"color": SPINE_COLOR, "linewidth": 1.2},
        capprops={"color": SPINE_COLOR, "linewidth": 1.2},
        flierprops={
            "marker": ".",
            "markersize": 3,
            "markeredgecolor": "cornflowerblue",
            "markerfacecolor": "cornflowerblue",
        },
    )
    for patch in box["boxes"]:
        patch.set_facecolor("cornflowerblue")
        patch.set_edgecolor("cornflowerblue")
        patch.set_alpha(0.45)
    _finish_axis(axis, "Layer", "Held-out per-token cosine")
    axis.set_title(f"{title_prefix}: cosine distribution by layer")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_error_by_position(
    path: Path,
    diagnostics: ProbeDiagnostics,
    plt: Any,
    title_prefix: str,
) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    test_positions = diagnostics.positions[diagnostics.test_indices]
    colors = _layer_colors(len(diagnostics.layer_indices))

    for local_layer, layer_index in enumerate(diagnostics.layer_indices):
        positions, means = _mean_by_position(test_positions, diagnostics.heldout_mse[local_layer])
        axis.plot(
            positions,
            means,
            color=colors[local_layer],
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=f"Layer {int(layer_index)}",
        )

    _finish_axis(axis, "Token position", "Held-out MSE")
    axis.set_title(f"{title_prefix}: error by token position")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_baseline_comparison(
    path: Path,
    diagnostics: ProbeDiagnostics,
    plt: Any,
    title_prefix: str,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    layers = np.asarray(diagnostics.layer_indices, dtype=int)
    probe_mse = np.mean(diagnostics.heldout_mse, axis=1)
    baseline_mse = np.mean(diagnostics.heldout_baseline_mse, axis=1)

    axis.plot(layers, probe_mse, color="cornflowerblue", marker="o", linewidth=2.2, label="Probe")
    axis.plot(
        layers,
        baseline_mse,
        color="rosybrown",
        marker="o",
        linewidth=2.2,
        label="Mean baseline",
    )
    _finish_axis(axis, "Layer", "Held-out MSE")
    axis.set_title(f"{title_prefix}: probe vs mean baseline")
    axis.set_xticks(layers)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_norm_vs_error(
    path: Path,
    diagnostics: ProbeDiagnostics,
    plt: Any,
    title_prefix: str,
    max_points_per_layer: int = 500,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    colors = _layer_colors(len(diagnostics.layer_indices))

    for local_layer, layer_index in enumerate(diagnostics.layer_indices):
        total = diagnostics.heldout_mse.shape[1]
        if total > max_points_per_layer:
            sample = np.linspace(0, total - 1, max_points_per_layer, dtype=int)
        else:
            sample = np.arange(total)
        axis.scatter(
            diagnostics.heldout_target_norm[local_layer, sample],
            diagnostics.heldout_mse[local_layer, sample],
            color=colors[local_layer],
            s=12,
            alpha=0.5,
            label=f"Layer {int(layer_index)}",
        )

    _finish_axis(axis, "Target activation norm", "Held-out MSE")
    axis.set_title(f"{title_prefix}: activation norm vs error")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_sequence_heatmap(
    path: Path,
    diagnostics: ProbeDiagnostics,
    plt: Any,
    title_prefix: str,
) -> None:
    start, stop = _first_sequence_bounds(diagnostics.positions)
    sequence_scores = diagnostics.all_cosine[:, start:stop]
    tokens = diagnostics.tokens[start:stop]

    fig, axis = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    image = axis.imshow(sequence_scores, aspect="auto", cmap=CONTINUOUS_CMAP, vmin=-1.0, vmax=1.0)
    axis.set_title(f"{title_prefix}: token-layer cosine heatmap")
    axis.set_xlabel("Token in one sequence")
    axis.set_ylabel("Layer")
    axis.set_yticks(np.arange(len(diagnostics.layer_indices)))
    axis.set_yticklabels([str(int(layer)) for layer in diagnostics.layer_indices])

    tick_indices = _token_tick_indices(len(tokens), max_ticks=18)
    axis.set_xticks(tick_indices)
    axis.set_xticklabels(
        [_short_token(tokens[index]) for index in tick_indices],
        rotation=60,
        ha="right",
    )
    fig.colorbar(image, ax=axis, label="Cosine similarity")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _finish_axis(axis: Any, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE_COLOR)
    axis.spines["bottom"].set_color(SPINE_COLOR)


def _mean_by_position(positions: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_positions = np.asarray(sorted(set(int(position) for position in positions)))
    means = np.asarray([np.mean(values[positions == position]) for position in unique_positions])
    return unique_positions, means


def _layer_colors(num_layers: int) -> list[str]:
    return [LAYER_PALETTE[index % len(LAYER_PALETTE)] for index in range(num_layers)]


def _first_sequence_bounds(positions: np.ndarray) -> tuple[int, int]:
    starts = np.flatnonzero(positions == 0)
    if starts.size == 0:
        return 0, min(len(positions), 32)
    start = int(starts[0])
    later_starts = starts[starts > start]
    stop = int(later_starts[0]) if later_starts.size else min(len(positions), start + 32)
    return start, max(start + 1, stop)


def _token_tick_indices(num_tokens: int, max_ticks: int) -> np.ndarray:
    if num_tokens <= max_ticks:
        return np.arange(num_tokens)
    return np.unique(np.linspace(0, num_tokens - 1, max_ticks, dtype=int))


def _short_token(token: str, max_length: int = 12) -> str:
    cleaned = token.replace("Ġ", " ").replace("Ċ", "\\n")
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 1] + "…"
