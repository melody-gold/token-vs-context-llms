from __future__ import annotations

import re
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
FIGURE_TITLE_SIZE = 18
COMPARISON_TITLE_SIZE = 14
COMPARISON_AXIS_TITLE_SIZE = 16
AXIS_TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 9
LEGEND_SIZE = 10
COLORBAR_LABEL_SIZE = 12
COLORBAR_TICK_SIZE = 9
LAYER_PALETTE = (
    "cornflowerblue",
    "darkseagreen",
    "rosybrown",
    "mediumvioletred",
    "darkturquoise",
    "goldenrod",
    "springgreen",
    "slateblue",
    "tomato",
    "slategray",
    "crimson",
    "yellowgreen",
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

    _apply_paper_style(plt)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(_metrics_plot_title(title), fontsize=FIGURE_TITLE_SIZE, fontweight="semibold")

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
        _finish_axis(axis, "Layer", label)
        _set_axis_title(axis, label)
        axis.set_xticks(layers)

    fig.savefig(target, dpi=200)
    plt.close(fig)


def write_r2_model_comparison_plot(
    path: str | Path,
    model_metrics: list[tuple[str, list[dict[str, Any]]]],
    title: str = "Token-Only R^2 Across Model Depth",
) -> None:
    """Write a side-by-side normalized-depth R^2 comparison figure."""

    if not model_metrics:
        raise ValueError("Cannot plot an empty model comparison.")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_paper_style(plt)

    fig, axes = plt.subplots(
        1,
        len(model_metrics),
        figsize=(5.1 * len(model_metrics), 3.9),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=COMPARISON_TITLE_SIZE, fontweight="semibold")

    for axis, (model_label, metrics) in zip(axes, model_metrics, strict=True):
        sorted_metrics = sorted(metrics, key=lambda row: int(row["layer_index"]))
        if len(sorted_metrics) < 2:
            raise ValueError(
                f"{model_label} needs at least two layers for relative-depth plotting."
            )

        layers = np.asarray([int(row["layer_index"]) for row in sorted_metrics])
        relative_depth = np.linspace(0.0, 1.0, len(sorted_metrics))
        r2_values = np.asarray([_require_float(row, "r2_score") for row in sorted_metrics])

        axis.plot(
            relative_depth,
            r2_values,
            color="darkseagreen",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="darkseagreen",
            markeredgewidth=1.5,
            linewidth=2.2,
        )
        _finish_axis(axis, "Relative depth", r"$R^2$")
        axis.set_title(model_label, fontsize=COMPARISON_AXIS_TITLE_SIZE, pad=8)
        axis.set_ylim(0.0, 0.82)
        axis.set_xlim(-0.03, 1.03)
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.set_xticklabels(["0", "0.5", "1"])

        for local_index, (x_value, layer) in enumerate(zip(relative_depth, layers, strict=True)):
            axis.annotate(
                str(layer),
                (x_value, r2_values[local_index]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=TICK_SIZE,
                color=TEXT_COLOR,
            )

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
    _write_lens_prediction_examples(
        target / "lens_prediction_examples.png",
        diagnostics,
        plt,
        title_prefix,
    )
    _write_cosine_by_position_heatmap(
        target / "token_layer_heatmap.png",
        diagnostics,
        plt,
        title_prefix,
    )


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
            "axes.labelsize": LABEL_SIZE,
            "axes.titlecolor": TEXT_COLOR,
            "axes.titlesize": AXIS_TITLE_SIZE,
            "axes.titleweight": "semibold",
            "xtick.color": TEXT_COLOR,
            "xtick.labelsize": TICK_SIZE,
            "ytick.color": TEXT_COLOR,
            "ytick.labelsize": TICK_SIZE,
            "text.color": TEXT_COLOR,
            "legend.fontsize": LEGEND_SIZE,
            "figure.titlesize": FIGURE_TITLE_SIZE,
            "figure.titleweight": "semibold",
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
    colors = _layer_colors(len(diagnostics.layer_indices))
    box = axis.boxplot(
        data,
        patch_artist=True,
        labels=[str(int(layer)) for layer in diagnostics.layer_indices],
        medianprops={"color": TEXT_COLOR, "linewidth": 1.4},
        boxprops={"linewidth": 1.4},
        whiskerprops={"color": SPINE_COLOR, "linewidth": 1.2},
        capprops={"color": SPINE_COLOR, "linewidth": 1.2},
        flierprops={
            "marker": ".",
            "markersize": 3,
            "markeredgecolor": SPINE_COLOR,
            "markerfacecolor": SPINE_COLOR,
        },
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.45)
    _finish_axis(axis, "Layer", "Held-out per-token cosine")
    _set_axis_title(axis, _paper_title(title_prefix, "Held-Out Cosine Similarity by Layer"))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_error_by_position(
    path: Path,
    diagnostics: ProbeDiagnostics,
    plt: Any,
    title_prefix: str,
) -> None:
    fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    test_positions = diagnostics.positions[diagnostics.test_indices]
    positions, mean_mse = _mean_matrix_by_position(test_positions, diagnostics.heldout_mse)
    positive_values = mean_mse[mean_mse > 0]
    epsilon = float(np.min(positive_values)) * 0.5 if positive_values.size else 1e-12
    log_mse = np.log10(mean_mse + epsilon)

    image = axis.imshow(log_mse, aspect="auto", cmap=CONTINUOUS_CMAP, origin="lower")
    _set_axis_title(axis, _paper_title(title_prefix, "Mean Reconstruction Error by Token Position"))
    _finish_heatmap_axis(axis, "Token position", "Layer")
    axis.set_yticks(np.arange(len(diagnostics.layer_indices)))
    axis.set_yticklabels([str(int(layer)) for layer in diagnostics.layer_indices])

    tick_indices = _token_tick_indices(len(positions), max_ticks=16)
    axis.set_xticks(tick_indices)
    axis.set_xticklabels([str(int(positions[index])) for index in tick_indices])
    _add_colorbar(fig, axis, image, r"Mean held-out MSE ($\log_{10}$)")
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
    _set_axis_title(axis, _paper_title(title_prefix, "Probe Error vs. Mean Baseline"))
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
    _set_axis_title(axis, _paper_title(title_prefix, "Activation Norm vs. Reconstruction Error"))
    axis.legend(frameon=False, ncol=2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_lens_prediction_examples(
    path: Path,
    diagnostics: ProbeDiagnostics,
    plt: Any,
    title_prefix: str,
) -> None:
    layer_positions = _representative_layer_positions(len(diagnostics.layer_indices))
    fig, axes = plt.subplots(
        1,
        len(layer_positions),
        figsize=(4.2 * len(layer_positions), 4.1),
        constrained_layout=True,
        squeeze=False,
    )
    targets = diagnostics.lens_example_targets[layer_positions].ravel()
    predictions = diagnostics.lens_example_predictions[layer_positions].ravel()
    lower = float(min(np.min(targets), np.min(predictions)))
    upper = float(max(np.max(targets), np.max(predictions)))
    margin = (upper - lower) * 0.04 or 1.0
    limits = (lower - margin, upper + margin)

    for axis, local_layer in zip(axes.ravel(), layer_positions, strict=True):
        layer_targets = diagnostics.lens_example_targets[local_layer].ravel()
        layer_predictions = diagnostics.lens_example_predictions[local_layer].ravel()
        correlation = np.corrcoef(layer_targets, layer_predictions)[0, 1]
        axis.scatter(
            layer_targets,
            layer_predictions,
            color="cornflowerblue",
            s=10,
            alpha=0.35,
            edgecolors="none",
        )
        axis.plot(limits, limits, color=SPINE_COLOR, linewidth=1.2, linestyle="--")
        _finish_axis(axis, "Actual activation value", "Predicted activation value")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        _set_axis_title(
            axis,
            f"Layer {int(diagnostics.layer_indices[local_layer])} (r = {correlation:.2f})",
        )

    fig.suptitle(
        _paper_title(title_prefix, "Probe Predictions vs. Target Activations"),
        fontsize=FIGURE_TITLE_SIZE,
        fontweight="semibold",
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_cosine_by_position_heatmap(
    path: Path,
    diagnostics: ProbeDiagnostics,
    plt: Any,
    title_prefix: str,
) -> None:
    test_positions = diagnostics.positions[diagnostics.test_indices]
    positions, mean_cosine = _mean_matrix_by_position(test_positions, diagnostics.heldout_cosine)

    fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    image = axis.imshow(
        mean_cosine,
        aspect="auto",
        cmap=CONTINUOUS_CMAP,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    _set_axis_title(axis, _paper_title(title_prefix, "Mean Cosine Similarity by Token Position"))
    _finish_heatmap_axis(axis, "Token position", "Layer")
    axis.set_yticks(np.arange(len(diagnostics.layer_indices)))
    axis.set_yticklabels([str(int(layer)) for layer in diagnostics.layer_indices])

    tick_indices = _token_tick_indices(len(positions), max_ticks=16)
    axis.set_xticks(tick_indices)
    axis.set_xticklabels([str(int(positions[index])) for index in tick_indices])
    _add_colorbar(fig, axis, image, "Mean held-out cosine similarity")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _set_axis_title(axis: Any, title: str) -> None:
    axis.set_title(title, fontsize=AXIS_TITLE_SIZE, fontweight="semibold", pad=8)


def _finish_axis(axis: Any, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.65)
    _finish_spines(axis)


def _finish_heatmap_axis(axis: Any, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    _finish_spines(axis)


def _finish_spines(axis: Any) -> None:
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE_COLOR)
    axis.spines["bottom"].set_color(SPINE_COLOR)


def _add_colorbar(fig: Any, axis: Any, image: Any, label: str) -> None:
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(label, fontsize=COLORBAR_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, colors=TEXT_COLOR)


def _shade_depth_regions(axis: Any) -> None:
    axis.axvspan(0.0, 0.2, color="cornflowerblue", alpha=0.08, linewidth=0)
    axis.axvspan(0.2, 0.8, color="darkseagreen", alpha=0.08, linewidth=0)
    axis.axvspan(0.8, 1.0, color="rosybrown", alpha=0.08, linewidth=0)


def _mean_matrix_by_position(
    positions: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_positions = np.asarray(sorted(set(int(position) for position in positions)))
    means = np.zeros((values.shape[0], unique_positions.size), dtype=float)
    for column, position in enumerate(unique_positions):
        means[:, column] = np.mean(values[:, positions == position], axis=1)
    return unique_positions, means


def _layer_colors(num_layers: int) -> list[str]:
    return [LAYER_PALETTE[index % len(LAYER_PALETTE)] for index in range(num_layers)]


def _representative_layer_positions(num_layers: int) -> list[int]:
    if num_layers <= 3:
        return list(range(num_layers))
    return sorted({0, num_layers // 2, num_layers - 1})


def _token_tick_indices(num_tokens: int, max_ticks: int) -> np.ndarray:
    if num_tokens <= max_ticks:
        return np.arange(num_tokens)
    return np.unique(np.linspace(0, num_tokens - 1, max_ticks, dtype=int))


def _paper_title(prefix: str, topic: str | None = None) -> str:
    formatted_prefix = _format_title_prefix(prefix)
    if topic is None:
        return formatted_prefix
    if not formatted_prefix:
        return topic
    return f"{formatted_prefix}: {topic}"


def _metrics_plot_title(title: str) -> str:
    formatted_title = _format_title_prefix(title)
    if not formatted_title:
        return "Layerwise Probe Metrics"
    if re.search(r"\b(Layerwise|Probe)\b", formatted_title, flags=re.IGNORECASE):
        return "Layerwise Probe Metrics"
    return f"{formatted_title}: Layerwise Probe Metrics"


def _format_title_prefix(prefix: str) -> str:
    cleaned = prefix.strip().replace("_", " ")
    cleaned = re.sub(r"\b(Diagnostics|Metrics)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*,\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :-")
    if not cleaned:
        return ""

    words = []
    for word in cleaned.split():
        lower = word.lower()
        if lower == "pythia":
            words.append("Pythia")
        elif match := re.fullmatch(r"pythia[- ]?(\d+)m", lower):
            words.append(f"Pythia-{match.group(1)}M")
        elif lower == "distilgpt2":
            words.append("DistilGPT-2")
        elif lower in {"pile10k", "pile-10k"}:
            words.append("Pile-10k")
        elif re.fullmatch(r"\d+m", lower):
            words.append(lower.upper())
        elif re.fullmatch(r"\d+k", lower):
            words.append(lower)
        else:
            words.append(word.capitalize())

    if "Pythia" in words and len(words) >= 2 and re.fullmatch(r"\d+M", words[1]):
        words = [f"Pythia-{words[1]}", *words[2:]]

    if len(words) >= 2 and re.fullmatch(r"\d+k", words[-1]) and words[-2] == "Pile-10k":
        token_count = words.pop()
        words[-1] = f"{words[-1]} ({token_count} tokens)"

    if any(word.startswith("Pythia") or word.startswith("DistilGPT") for word in words):
        return ", ".join(words)
    return " ".join(words)
