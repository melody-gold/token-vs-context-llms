from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    name: str = "distilgpt2"
    device: str = "cpu"
    max_length: int = 128
    layers: list[int] | None = None


@dataclass(slots=True)
class DatasetConfig:
    name: str = "wikitext"
    subset: str | None = "wikitext-2-raw-v1"
    split: str = "train[:64]"
    text_column: str = "text"
    max_texts: int = 64


@dataclass(slots=True)
class ExtractionConfig:
    batch_size: int = 4
    max_tokens: int = 4096
    output_path: str = "artifacts/extractions/small_debug.npz"


@dataclass(slots=True)
class ProbeConfig:
    ridge_alpha: float = 1.0
    test_fraction: float = 0.2
    random_seed: int = 0
    output_path: str = "results/generated/small_debug_metrics.json"


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = _load_yaml(path)
    return ExperimentConfig(
        model=ModelConfig(**raw.get("model", {})),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        extraction=ExtractionConfig(**raw.get("extraction", {})),
        probe=ProbeConfig(**raw.get("probe", {})),
    )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at top level of {path}, got {type(loaded)!r}")

    return loaded
