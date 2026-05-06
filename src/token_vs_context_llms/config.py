from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    """Configuration for the Hugging Face causal language model."""

    name: str = "distilgpt2"
    device: str = "cpu"
    max_length: int = 128
    layers: list[int] | None = None


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for the text dataset used during activation extraction."""

    name: str = "wikitext"
    subset: str | None = "wikitext-2-raw-v1"
    split: str = "train[:64]"
    text_column: str = "text"
    max_texts: int = 64


@dataclass(slots=True)
class ExtractionConfig:
    """Configuration for batching and storing token-level activations."""

    batch_size: int = 4
    max_tokens: int = 4096
    output_path: str = "artifacts/extractions/small_debug.npz"


@dataclass(slots=True)
class ProbeConfig:
    """Configuration for train/test splitting and probe metric output."""

    ridge_alpha: float = 0.0
    test_fraction: float = 0.2
    random_seed: int = 0
    output_path: str = "results/generated/small_debug_metrics.json"


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level experiment configuration loaded from YAML."""

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment file into typed config dataclasses.

    Args:
        path: Path to a YAML file with optional `model`, `dataset`, `extraction`,
            and `probe` sections

    Returns:
        An `ExperimentConfig` with defaults filled in for missing sections
    """

    raw = _load_yaml(path)
    # each section is optional so small debug configs can override only what they need
    return ExperimentConfig(
        model=ModelConfig(**raw.get("model", {})),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        extraction=ExtractionConfig(**raw.get("extraction", {})),
        probe=ProbeConfig(**raw.get("probe", {})),
    )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and require a mapping at the top level

    Args:
        path: Path to the YAML file

    Returns:
        The parsed top-level YAML mapping

    Raises:
        ValueError: If the YAML file does not contain a top-level mapping
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    # The config loader expects named sections, so lists/strings at the top level are errors
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at top level of {path}, got {type(loaded)!r}")

    return loaded
