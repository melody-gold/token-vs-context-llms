from __future__ import annotations

from itertools import islice
from typing import Iterable

import numpy as np

from token_vs_context_llms.config import ExperimentConfig
from token_vs_context_llms.io import ActivationArtifact


def collect_hidden_state_artifact(config: ExperimentConfig) -> ActivationArtifact:
    """Extract token embeddings and selected hidden states into one artifact

    Args:
        config: Full experiment configuration describing the model, dataset,
            extraction batch size, token budget, and selected layers

    Returns:
        An `ActivationArtifact` whose arrays are token-level. Padded positions
        are removed, each row corresponds to one observed token, and extraction
        stops at `config.extraction.max_tokens`

    Raises:
        ImportError: If the optional Hugging Face extraction dependencies are
            not installed
        ValueError: If no usable text or no token activations are produced
    """

    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "LLM extraction dependencies are not installed. Run `uv sync --all-extras --dev`."
        ) from exc

    dataset = load_dataset(
        config.dataset.name,
        config.dataset.subset,
        split=config.dataset.split,
    )
    texts = _select_texts(dataset, config.dataset.text_column, config.dataset.max_texts)
    if not texts:
        raise ValueError("No non-empty texts were found in the configured dataset split.")

    # tokenize raw text (token ids: indices into model embedding table)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load causal LM (forward pass returns transformer hidden states)
    model = AutoModelForCausalLM.from_pretrained(config.model.name)
    device = torch.device(config.model.device)
    model.to(device)
    model.eval()

    selected_layers: list[int] | None = config.model.layers
    # probe dataset columns collected from the LLM forward pass
    token_embeddings_rows: list[np.ndarray] = []
    hidden_state_rows: list[np.ndarray] = []
    token_id_rows: list[np.ndarray] = []
    position_rows: list[np.ndarray] = []
    token_strings: list[str] = []
    total_tokens = 0

    for batch in _batched(texts, config.extraction.batch_size):
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.model.max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)
            # hidden_states[0]: embedding stream; hidden_states[1:]: block outputs
            hidden_states = outputs.hidden_states[1:]
            if selected_layers is None:
                selected_layers = list(range(len(hidden_states)))

            # stack selected block outputs as [batch, seq, layer, hidden]
            stacked_layers = torch.stack([hidden_states[index] for index in selected_layers], dim=2)
            # token-only input vectors: embedding lookup before attention/context
            embeddings = model.get_input_embeddings()(encoded["input_ids"])

        attention_mask = encoded["attention_mask"].bool()
        positions = torch.arange(encoded["input_ids"].shape[1], device=device).expand_as(
            encoded["input_ids"]
        )

        # drop pad positions; keep one row per real token
        batch_embeddings = embeddings[attention_mask].cpu().numpy()
        batch_hidden_states = stacked_layers[attention_mask].cpu().numpy()
        batch_input_ids = encoded["input_ids"][attention_mask].cpu().numpy()
        batch_positions = positions[attention_mask].cpu().numpy()

        remaining_tokens = config.extraction.max_tokens - total_tokens
        if remaining_tokens <= 0:
            break

        batch_count = min(remaining_tokens, batch_embeddings.shape[0])
        # truncate final batch to the configured token budget
        token_embeddings_rows.append(batch_embeddings[:batch_count])
        hidden_state_rows.append(batch_hidden_states[:batch_count])
        token_id_rows.append(batch_input_ids[:batch_count])
        position_rows.append(batch_positions[:batch_count])
        # decoded token labels for later qualitative inspection
        token_strings.extend(tokenizer.convert_ids_to_tokens(batch_input_ids[:batch_count].tolist()))
        total_tokens += batch_count

        if total_tokens >= config.extraction.max_tokens:
            break

    if not token_embeddings_rows:
        raise ValueError("Extraction produced no token-level activations.")

    return ActivationArtifact(
        token_embeddings=np.concatenate(token_embeddings_rows, axis=0),
        hidden_states=np.concatenate(hidden_state_rows, axis=0),
        input_ids=np.concatenate(token_id_rows, axis=0),
        positions=np.concatenate(position_rows, axis=0),
        tokens=np.asarray(token_strings, dtype=str),
        layer_indices=np.asarray(selected_layers, dtype=np.int64),
        metadata={
            "model_name": config.model.name,
            "dataset_name": config.dataset.name,
            "dataset_subset": config.dataset.subset,
            "split": config.dataset.split,
            "max_length": config.model.max_length,
            "num_tokens": int(total_tokens),
        },
    )


def _select_texts(dataset: Iterable[dict], text_column: str, max_texts: int) -> list[str]:
    """Collect non-empty text fields from a dataset iterator.

    Args:
        dataset: Iterable of dataset rows, usually from Hugging Face datasets
        text_column: Name of the column containing text
        max_texts: Maximum number of text examples to return

    Returns:
        A list of stripped, non-empty text strings
    """

    texts: list[str] = []
    # scan extra rows because many language-model datasets include blank text rows
    for row in islice(dataset, max_texts * 4):
        value = str(row.get(text_column, "")).strip()
        if value:
            texts.append(value)
        if len(texts) >= max_texts:
            break
    return texts


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    """Yield consecutive batches from a list of text strings

    Args:
        items: Text strings to batch
        batch_size: Maximum number of strings per batch

    Yields:
        Consecutive slices of `items`
    """

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
