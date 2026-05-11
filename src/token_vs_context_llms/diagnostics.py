from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from token_vs_context_llms.io import ActivationArtifact
from token_vs_context_llms.probe import fit_affine_probe, train_test_indices


@dataclass(slots=True)
class ProbeDiagnostics:
    """Per-token probe diagnostics for optional exploratory plots."""

    layer_indices: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    positions: np.ndarray
    tokens: np.ndarray
    heldout_cosine: np.ndarray
    heldout_mse: np.ndarray
    heldout_target_norm: np.ndarray
    heldout_baseline_mse: np.ndarray
    all_cosine: np.ndarray
    all_mse: np.ndarray
    lens_example_token_offsets: np.ndarray
    lens_example_dimensions: np.ndarray
    lens_example_targets: np.ndarray
    lens_example_predictions: np.ndarray
    summary: list[dict[str, float | int]]


def compute_probe_diagnostics(
    artifact: ActivationArtifact,
    alpha: float = 0.0,
    test_fraction: float = 0.2,
    random_seed: int = 0,
) -> ProbeDiagnostics:
    """Fit layer probes and keep per-token prediction diagnostics."""

    token_embeddings = np.asarray(artifact.token_embeddings, dtype=np.float64)
    hidden_states = np.asarray(artifact.hidden_states, dtype=np.float64)

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [num_tokens, num_layers, hidden_size].")

    num_tokens, num_layers, hidden_size = hidden_states.shape
    if token_embeddings.shape[0] != num_tokens:
        raise ValueError("token_embeddings and hidden_states must have the same token count.")

    train_indices, test_indices = train_test_indices(
        num_tokens,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )

    heldout_cosine = np.zeros((num_layers, test_indices.size), dtype=np.float64)
    heldout_mse = np.zeros_like(heldout_cosine)
    heldout_target_norm = np.zeros_like(heldout_cosine)
    heldout_baseline_mse = np.zeros_like(heldout_cosine)
    all_cosine = np.zeros((num_layers, num_tokens), dtype=np.float64)
    all_mse = np.zeros_like(all_cosine)
    summary: list[dict[str, float | int]] = []
    rng = np.random.default_rng(random_seed)
    example_token_offsets = _sample_indices(test_indices.size, max_size=128, rng=rng)
    example_dimensions = _sample_indices(hidden_size, max_size=32, rng=rng)
    lens_example_targets = np.zeros(
        (num_layers, example_token_offsets.size, example_dimensions.size),
        dtype=np.float64,
    )
    lens_example_predictions = np.zeros_like(lens_example_targets)

    x_train = token_embeddings[train_indices]
    x_test = token_embeddings[test_indices]

    for local_layer, layer_index in enumerate(artifact.layer_indices):
        targets = hidden_states[:, local_layer, :]
        y_train = targets[train_indices]
        y_test = targets[test_indices]

        probe = fit_affine_probe(x_train, y_train, alpha=alpha)
        pred_test = probe.predict(x_test)
        pred_all = probe.predict(token_embeddings)

        baseline = np.mean(y_train, axis=0, keepdims=True)
        baseline_test = np.repeat(baseline, repeats=test_indices.size, axis=0)

        heldout_cosine[local_layer] = row_cosine_similarity(y_test, pred_test)
        heldout_mse[local_layer] = row_mean_squared_error(y_test, pred_test)
        heldout_target_norm[local_layer] = np.linalg.norm(y_test, axis=1)
        heldout_baseline_mse[local_layer] = row_mean_squared_error(y_test, baseline_test)
        all_cosine[local_layer] = row_cosine_similarity(targets, pred_all)
        all_mse[local_layer] = row_mean_squared_error(targets, pred_all)
        lens_example_targets[local_layer] = y_test[
            np.ix_(example_token_offsets, example_dimensions)
        ]
        lens_example_predictions[local_layer] = pred_test[
            np.ix_(example_token_offsets, example_dimensions)
        ]

        mean_probe_mse = float(np.mean(heldout_mse[local_layer]))
        mean_baseline_mse = float(np.mean(heldout_baseline_mse[local_layer]))
        target_scale = float(np.mean(y_test**2))
        summary.append(
            {
                "layer_index": int(layer_index),
                "mean_cosine_similarity": float(np.mean(heldout_cosine[local_layer])),
                "mean_squared_error": mean_probe_mse,
                "mean_baseline_mse": mean_baseline_mse,
                "mse_improvement_over_baseline": mean_baseline_mse - mean_probe_mse,
                "normalized_mse": mean_probe_mse / max(target_scale, 1e-12),
                "mean_target_norm": float(np.mean(heldout_target_norm[local_layer])),
                "num_train_tokens": int(train_indices.size),
                "num_test_tokens": int(test_indices.size),
            }
        )

    return ProbeDiagnostics(
        layer_indices=np.asarray(artifact.layer_indices, dtype=np.int64),
        train_indices=train_indices,
        test_indices=test_indices,
        positions=np.asarray(artifact.positions),
        tokens=np.asarray(artifact.tokens, dtype=str),
        heldout_cosine=heldout_cosine,
        heldout_mse=heldout_mse,
        heldout_target_norm=heldout_target_norm,
        heldout_baseline_mse=heldout_baseline_mse,
        all_cosine=all_cosine,
        all_mse=all_mse,
        lens_example_token_offsets=example_token_offsets,
        lens_example_dimensions=example_dimensions,
        lens_example_targets=lens_example_targets,
        lens_example_predictions=lens_example_predictions,
        summary=summary,
    )


def row_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return one MSE value per row."""

    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=1)


def row_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return one cosine similarity value per row."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    numerator = np.sum(y_true * y_pred, axis=1)
    denominator = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    return numerator / np.clip(denominator, a_min=1e-12, a_max=None)


def _sample_indices(size: int, max_size: int, rng: np.random.Generator) -> np.ndarray:
    sample_size = min(size, max_size)
    if sample_size == size:
        return np.arange(size)
    return np.sort(rng.choice(size, size=sample_size, replace=False))


def save_probe_diagnostics(output_dir: str | Path, diagnostics: ProbeDiagnostics) -> None:
    """Save diagnostic arrays and a compact summary table."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    np.savez(
        target / "diagnostics.npz",
        layer_indices=diagnostics.layer_indices,
        train_indices=diagnostics.train_indices,
        test_indices=diagnostics.test_indices,
        positions=diagnostics.positions,
        tokens=diagnostics.tokens,
        heldout_cosine=diagnostics.heldout_cosine,
        heldout_mse=diagnostics.heldout_mse,
        heldout_target_norm=diagnostics.heldout_target_norm,
        heldout_baseline_mse=diagnostics.heldout_baseline_mse,
        all_cosine=diagnostics.all_cosine,
        all_mse=diagnostics.all_mse,
        lens_example_token_offsets=diagnostics.lens_example_token_offsets,
        lens_example_dimensions=diagnostics.lens_example_dimensions,
        lens_example_targets=diagnostics.lens_example_targets,
        lens_example_predictions=diagnostics.lens_example_predictions,
    )
    _write_summary_table(target / "diagnostic_summary.md", diagnostics.summary)
    _write_worst_token_table(target / "worst_tokens.md", diagnostics)


def _write_summary_table(path: Path, rows: list[dict[str, float | int]]) -> None:
    lines = [
        "# Probe Diagnostic Summary",
        "",
        "| Layer | MSE | Mean baseline MSE | Normalized MSE | Mean cosine | Mean norm |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["layer_index"])),
                    _format_float(row["mean_squared_error"]),
                    _format_float(row["mean_baseline_mse"]),
                    _format_float(row["normalized_mse"]),
                    _format_float(row["mean_cosine_similarity"]),
                    _format_float(row["mean_target_norm"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_worst_token_table(
    path: Path,
    diagnostics: ProbeDiagnostics,
    rows_per_layer: int = 8,
) -> None:
    lines = [
        "# Worst Held-Out Tokens by Cosine Similarity",
        "",
        "| Layer | Token | Position | Cosine | MSE | Target norm |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    test_positions = diagnostics.positions[diagnostics.test_indices]
    test_tokens = diagnostics.tokens[diagnostics.test_indices]

    for local_layer, layer_index in enumerate(diagnostics.layer_indices):
        order = np.argsort(diagnostics.heldout_cosine[local_layer])[:rows_per_layer]
        for test_offset in order:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(int(layer_index)),
                        _escape_markdown(str(test_tokens[test_offset])),
                        str(int(test_positions[test_offset])),
                        _format_float(diagnostics.heldout_cosine[local_layer, test_offset]),
                        _format_float(diagnostics.heldout_mse[local_layer, test_offset]),
                        _format_float(diagnostics.heldout_target_norm[local_layer, test_offset]),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_float(value: float | int) -> str:
    return f"{float(value):.6g}"


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "\\n")
