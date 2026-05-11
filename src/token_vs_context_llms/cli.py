from __future__ import annotations

import argparse

from token_vs_context_llms.config import ExperimentConfig, ProbeConfig, load_experiment_config
from token_vs_context_llms.diagnostics import compute_probe_diagnostics, save_probe_diagnostics
from token_vs_context_llms.extract import collect_hidden_state_artifact
from token_vs_context_llms.io import load_artifact, save_artifact, save_metrics
from token_vs_context_llms.plotting import (
    write_diagnostic_plots,
    write_layerwise_metrics_plot,
    write_r2_model_comparison_plot,
)
from token_vs_context_llms.probe import evaluate_hidden_state_layers, serialize_metrics
from token_vs_context_llms.summary import load_metrics_json, write_metrics_summary


def main() -> None:
    """Parse command-line arguments and dispatch to extraction or probing

    Returns:
        None. This function runs the requested command and prints the output path
    """

    parser = argparse.ArgumentParser(description="Token-vs-context experiment runner.")
    # CLI stages: extract activations, then probe saved activations
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract token embeddings and hidden states.",
    )
    extract_parser.add_argument("--config", required=True, help="Path to experiment YAML config.")

    probe_parser = subparsers.add_parser(
        "probe",
        help="Fit token-only probes to saved activations.",
    )
    probe_parser.add_argument("--config", help="Optional path to experiment YAML config.")
    probe_parser.add_argument(
        "--artifact",
        help="Override path to the activation artifact .npz file.",
    )
    probe_parser.add_argument("--output", help="Override path to the metrics JSON output.")
    probe_parser.add_argument(
        "--alpha",
        type=float,
        help="Override ridge alpha. Use 0 for the unregularized affine baseline.",
    )
    probe_parser.add_argument("--test-fraction", type=float, help="Override test split fraction.")
    probe_parser.add_argument("--random-seed", type=int, help="Override random seed.")

    summary_parser = subparsers.add_parser(
        "summarize",
        help="Write a Markdown summary from a metrics JSON file.",
    )
    summary_parser.add_argument("--metrics", required=True, help="Path to metrics JSON.")
    summary_parser.add_argument("--output", required=True, help="Path to Markdown summary.")
    summary_parser.add_argument("--title", default="Probe Metrics", help="Summary heading.")

    plot_parser = subparsers.add_parser(
        "plot",
        help="Write a PNG figure from a metrics JSON file.",
    )
    plot_parser.add_argument("--metrics", required=True, help="Path to metrics JSON.")
    plot_parser.add_argument("--output", required=True, help="Path to PNG figure.")
    plot_parser.add_argument("--title", default="Layerwise Probe Metrics", help="Figure title.")

    compare_parser = subparsers.add_parser(
        "compare",
        help="Write a side-by-side R^2 comparison figure from two metrics JSON files.",
    )
    compare_parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="One or more metrics JSON paths.",
    )
    compare_parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Model labels in the same order as --metrics.",
    )
    compare_parser.add_argument("--output", required=True, help="Path to PNG figure.")
    compare_parser.add_argument(
        "--title",
        default=r"Token-Only $R^2$ Across Model Depth",
        help="Figure title.",
    )

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Write per-token diagnostics, exploratory figures, and worst-token tables.",
    )
    diagnose_parser.add_argument("--config", help="Optional path to experiment YAML config.")
    diagnose_parser.add_argument(
        "--artifact",
        help="Override path to the activation artifact .npz file.",
    )
    diagnose_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for diagnostic arrays, tables, and figures.",
    )
    diagnose_parser.add_argument(
        "--alpha",
        type=float,
        help="Override ridge alpha. Use 0 for the unregularized affine baseline.",
    )
    diagnose_parser.add_argument(
        "--test-fraction",
        type=float,
        help="Override test split fraction.",
    )
    diagnose_parser.add_argument("--random-seed", type=int, help="Override random seed.")
    diagnose_parser.add_argument(
        "--title",
        default="Probe Diagnostics",
        help="Figure title prefix.",
    )

    args = parser.parse_args()
    if args.command == "extract":
        run_extract(args.config)
        return

    if args.command == "probe":
        run_probe(
            config_path=args.config,
            artifact_path=args.artifact,
            output_path=args.output,
            alpha=args.alpha,
            test_fraction=args.test_fraction,
            random_seed=args.random_seed,
        )
        return

    if args.command == "summarize":
        run_summarize(args.metrics, args.output, args.title)
        return

    if args.command == "plot":
        run_plot(args.metrics, args.output, args.title)
        return

    if args.command == "compare":
        run_compare(args.metrics, args.labels, args.output, args.title)
        return

    if args.command == "diagnose":
        run_diagnose(
            config_path=args.config,
            artifact_path=args.artifact,
            output_dir=args.output_dir,
            alpha=args.alpha,
            test_fraction=args.test_fraction,
            random_seed=args.random_seed,
            title=args.title,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


def run_extract(config_path: str) -> None:
    """Run activation extraction from a YAML config and save the artifact

    Args:
        config_path: Path to the experiment YAML file

    Returns:
        None. The extraction artifact is written to the configured output path
    """

    config = load_experiment_config(config_path)
    artifact = collect_hidden_state_artifact(config)
    save_artifact(config.extraction.output_path, artifact)
    print(f"Saved activations to {config.extraction.output_path}")


def run_probe(
    config_path: str | None,
    artifact_path: str | None,
    output_path: str | None,
    alpha: float | None,
    test_fraction: float | None,
    random_seed: int | None,
) -> None:
    """Run layerwise probe evaluation and save metrics JSON

    Args:
        config_path: Optional path to the experiment YAML file
        artifact_path: Optional override for the activation artifact path
        output_path: Optional override for the metrics JSON path
        alpha: Optional override for the ridge penalty
        test_fraction: Optional override for the held-out test fraction
        random_seed: Optional override for the train/test split seed

    Returns:
        None. The serialized metrics are written to the configured output path
    """

    config = load_experiment_config(config_path) if config_path else ExperimentConfig()
    # CLI overrides for probe ablations without rerunning LLM extraction
    probe_config = ProbeConfig(
        ridge_alpha=alpha if alpha is not None else config.probe.ridge_alpha,
        test_fraction=test_fraction if test_fraction is not None else config.probe.test_fraction,
        random_seed=random_seed if random_seed is not None else config.probe.random_seed,
        output_path=output_path or config.probe.output_path,
    )

    # probe stage reads saved LLM activations from extract stage
    artifact = load_artifact(artifact_path or config.extraction.output_path)
    metrics = evaluate_hidden_state_layers(
        artifact.token_embeddings,
        artifact.hidden_states,
        artifact.layer_indices,
        alpha=probe_config.ridge_alpha,
        test_fraction=probe_config.test_fraction,
        random_seed=probe_config.random_seed,
    )
    serialized = serialize_metrics(metrics)
    save_metrics(probe_config.output_path, serialized)
    print(f"Saved metrics to {probe_config.output_path}")


def run_summarize(metrics_path: str, output_path: str, title: str) -> None:
    """Write a Markdown experiment summary from serialized metrics.

    Args:
        metrics_path: Path to the layerwise metrics JSON file
        output_path: Path where the Markdown summary should be written
        title: Markdown heading for the summary

    Returns:
        None. The formatted summary is written to `output_path`
    """

    metrics = load_metrics_json(metrics_path)
    write_metrics_summary(output_path, metrics, title=title)
    print(f"Saved summary to {output_path}")


def run_plot(metrics_path: str, output_path: str, title: str) -> None:
    """Write a layerwise metrics figure from serialized metrics.

    Args:
        metrics_path: Path to the layerwise metrics JSON file
        output_path: Path where the PNG figure should be written
        title: Figure title

    Returns:
        None. The plotted figure is written to `output_path`
    """

    metrics = load_metrics_json(metrics_path)
    write_layerwise_metrics_plot(output_path, metrics, title=title)
    print(f"Saved plot to {output_path}")


def run_compare(
    metrics_paths: list[str],
    labels: list[str],
    output_path: str,
    title: str,
) -> None:
    """Write a normalized-depth R^2 comparison figure.

    Args:
        metrics_paths: Paths to the layerwise metrics JSON files
        labels: Model labels matching the metric paths
        output_path: Path where the PNG figure should be written
        title: Figure title

    Returns:
        None. The plotted figure is written to `output_path`
    """

    if len(metrics_paths) != len(labels):
        raise ValueError("--metrics and --labels must have the same number of entries.")

    model_metrics = [
        (label, load_metrics_json(metrics_path))
        for label, metrics_path in zip(labels, metrics_paths, strict=True)
    ]
    write_r2_model_comparison_plot(output_path, model_metrics, title=title)
    print(f"Saved comparison plot to {output_path}")


def run_diagnose(
    config_path: str | None,
    artifact_path: str | None,
    output_dir: str,
    alpha: float | None,
    test_fraction: float | None,
    random_seed: int | None,
    title: str,
) -> None:
    """Write exploratory per-token probe diagnostics.

    Args:
        config_path: Optional path to the experiment YAML file
        artifact_path: Optional override for the activation artifact path
        output_dir: Directory for diagnostic arrays, tables, and figures
        alpha: Optional override for the ridge penalty
        test_fraction: Optional override for the held-out test fraction
        random_seed: Optional override for the train/test split seed
        title: Figure title prefix

    Returns:
        None. Diagnostics are written under `output_dir`
    """

    config = load_experiment_config(config_path) if config_path else ExperimentConfig()
    probe_config = ProbeConfig(
        ridge_alpha=alpha if alpha is not None else config.probe.ridge_alpha,
        test_fraction=test_fraction if test_fraction is not None else config.probe.test_fraction,
        random_seed=random_seed if random_seed is not None else config.probe.random_seed,
        output_path=config.probe.output_path,
    )

    artifact = load_artifact(artifact_path or config.extraction.output_path)
    diagnostics = compute_probe_diagnostics(
        artifact,
        alpha=probe_config.ridge_alpha,
        test_fraction=probe_config.test_fraction,
        random_seed=probe_config.random_seed,
    )
    save_probe_diagnostics(output_dir, diagnostics)
    write_diagnostic_plots(output_dir, diagnostics, title_prefix=title)
    print(f"Saved diagnostics to {output_dir}")
