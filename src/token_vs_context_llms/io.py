from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ActivationArtifact:
    """Token-level extraction output used by the probe pipeline."""

    token_embeddings: np.ndarray
    hidden_states: np.ndarray
    input_ids: np.ndarray
    positions: np.ndarray
    tokens: np.ndarray
    layer_indices: np.ndarray
    metadata: dict[str, Any]


def save_artifact(path: str | Path, artifact: ActivationArtifact) -> None:
    """Save activation arrays to an uncompressed `.npz` file

    Args:
        path: output `.npz` path for the numeric activation arrays
        artifact: activation artifact to write

    Returns:
        None. writes the `.npz` file and a sidecar `.json` metadata file.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Use uncompressed npz because these artifacts are regenerable and speed matters.
    np.savez(
        target,
        token_embeddings=artifact.token_embeddings,
        hidden_states=artifact.hidden_states,
        input_ids=artifact.input_ids,
        positions=artifact.positions,
        tokens=artifact.tokens,
        layer_indices=artifact.layer_indices,
    )
    # Metadata stays in JSON so it can be inspected without loading a large array file.
    target.with_suffix(".json").write_text(
        json.dumps(artifact.metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_artifact(path: str | Path) -> ActivationArtifact:
    """Load activation arrays and optional sidecar metadata from disk

    Args:
        path: path to the saved `.npz` artifact

    Returns:
        An `ActivationArtifact` reconstructed from the array file and sidecar
        JSON metadata when present.
    """

    source = Path(path)
    payload = np.load(source)
    metadata_path = source.with_suffix(".json")
    metadata = {}
    # older or hand-created artifacts may not have metadata, so treat it as optional
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
    """Write serialized layer metrics as formatted JSON

    Args:
        path: output JSON path
        metrics: JSON-serializable metric dictionaries, usually from
            `serialize_metrics`

    Returns:
        None. The function creates parent directories and writes the JSON file.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
