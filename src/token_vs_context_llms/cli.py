from __future__ import annotations

import argparse

from token_vs_context_llms.config import ExperimentConfig, ProbeConfig, load_experiment_config
from token_vs_context_llms.extract import collect_hidden_state_artifact
from token_vs_context_llms.io import load_artifact, save_artifact, save_metrics
from token_vs_context_llms.probe import evaluate_hidden_state_layers, serialize_metrics


def main() -> None:
    """Parse command-line arguments and dispatch to extraction or probing

    Returns:
        None. This function runs the requested command and prints the output path
    """

    parser = argparse.ArgumentParser(description="Token-vs-context experiment runner.")
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
    # CLI flags win over values loaded from the YAML config
    probe_config = ProbeConfig(
        ridge_alpha=alpha if alpha is not None else config.probe.ridge_alpha,
        test_fraction=test_fraction if test_fraction is not None else config.probe.test_fraction,
        random_seed=random_seed if random_seed is not None else config.probe.random_seed,
        output_path=output_path or config.probe.output_path,
    )

    # artifact path can come from either the config or a direct CLI override
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
