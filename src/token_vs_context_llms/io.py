from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ActivationArtifact:
    token_embeddings: np.ndarray
    hidden_states: np.ndarray
    input_ids: np.ndarray
    positions: np.ndarray
    tokens: np.ndarray
    layer_indices: np.ndarray
    metadata: dict[str, Any]


def save_artifact(path: str | Path, artifact: ActivationArtifact) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        token_embeddings=artifact.token_embeddings,
        hidden_states=artifact.hidden_states,
        input_ids=artifact.input_ids,
        positions=artifact.positions,
        tokens=artifact.tokens,
        layer_indices=artifact.layer_indices,
    )
    target.with_suffix(".json").write_text(
        json.dumps(artifact.metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_artifact(path: str | Path) -> ActivationArtifact:
    source = Path(path)
    payload = np.load(source)
    metadata_path = source.with_suffix(".json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return ActivationArtifact(
        token_embeddings=payload["token_embeddings"],
        hidden_states=payload["hidden_states"],
        input_ids=payload["input_ids"],
        positions=payload["positions"],
        tokens=payload["tokens"],
        layer_indices=payload["layer_indices"],
        metadata=metadata,
    )


def save_metrics(path: str | Path, metrics: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
