from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ActivationArtifact:
    """Token-level extraction output used by the probe pipeline.

    `token_embeddings` is `embed_activations`, and each
    `hidden_states[:, layer_position, :]` slice is one
    `intermediate_activation`.
    """

    # embed_activations: context-free embedding lookup vectors, [num_tokens, d_model]
    token_embeddings: np.ndarray
    # intermediate activations: contextual block states, [num_tokens, num_layers, d_model]
    hidden_states: np.ndarray
    # later SAE artifact can mirror this layout with feature activations
    # token metadata for qualitative error inspection
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
    # activation artifact: uncompressed npz for faster write/read
    np.savez(
        target,
        token_embeddings=artifact.token_embeddings,
        hidden_states=artifact.hidden_states,
        input_ids=artifact.input_ids,
        positions=artifact.positions,
        tokens=artifact.tokens,
        layer_indices=artifact.layer_indices,
    )
    
    # sidecar metadata: model/dataset/config origin
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
    # metadata is optional; arrays are enough for probe evaluation
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
    # metrics output: small, commit-friendly experiment result
    target.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
