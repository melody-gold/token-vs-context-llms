import numpy as np
import pytest

from token_vs_context_llms.probe import (
    evaluate_hidden_state_layers,
    evaluate_hidden_state_layers_repeated_splits,
    evaluate_probe,
    flatten_token_activations,
    serialize_metrics,
)


def test_evaluate_probe_recovers_linear_mapping() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    weights = rng.normal(size=(5, 3))
    bias = np.array([0.1, -0.2, 0.3])
    y = x @ weights + bias

    _, metrics = evaluate_probe(x, y, test_fraction=0.25, random_seed=3)
    assert metrics.mean_squared_error < 1e-12
    assert metrics.r2_score > 0.999999
    assert metrics.mean_cosine_similarity > 0.999999


def test_flatten_token_activations_flattens_batch_and_context_dims() -> None:
    activations = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    flattened = flatten_token_activations(activations)

    assert flattened.shape == (6, 4)
    np.testing.assert_array_equal(flattened[0], activations[0, 0])
    np.testing.assert_array_equal(flattened[-1], activations[-1, -1])


def test_evaluate_probe_accepts_batched_cache_tensors() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8, 5))
    weights = rng.normal(size=(5, 3))
    bias = np.array([0.1, -0.2, 0.3])
    y = x @ weights + bias

    _, metrics = evaluate_probe(x, y, test_fraction=0.25, random_seed=3)

    assert metrics.mean_squared_error < 1e-12
    assert metrics.num_train_tokens == 24
    assert metrics.num_test_tokens == 8


def test_flatten_token_activations_rejects_unexpected_shape() -> None:
    with pytest.raises(ValueError, match="activations must have shape"):
        flatten_token_activations(np.zeros((2, 3, 4, 5)))


def test_evaluate_hidden_state_layers_returns_one_result_per_layer() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 4))
    layer_a = x @ rng.normal(size=(4, 3))
    layer_b = x @ rng.normal(size=(4, 3))
    hidden_states = np.stack([layer_a, layer_b], axis=1)

    metrics = evaluate_hidden_state_layers(
        x,
        hidden_states,
        np.array([2, 5]),
        test_fraction=0.25,
        random_seed=7,
    )

    assert [metric.layer_index for metric in metrics] == [2, 5]
    assert len(metrics) == 2


def test_evaluate_hidden_state_layers_repeated_splits_aggregates_seed_variation() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 4))
    layer_a = x @ rng.normal(size=(4, 3)) + rng.normal(scale=0.05, size=(80, 3))
    layer_b = x @ rng.normal(size=(4, 3)) + rng.normal(scale=0.05, size=(80, 3))
    hidden_states = np.stack([layer_a, layer_b], axis=1)

    metrics = evaluate_hidden_state_layers_repeated_splits(
        x,
        hidden_states,
        np.array([0, 1]),
        random_seeds=[0, 1, 2],
        test_fraction=0.25,
    )

    assert [metric.layer_index for metric in metrics] == [0, 1]
    assert all(metric.num_splits == 3 for metric in metrics)
    assert all(metric.random_seeds == [0, 1, 2] for metric in metrics)
    assert all(metric.r2_score_std >= 0 for metric in metrics)

    serialized = serialize_metrics(metrics)
    assert serialized[0]["num_splits"] == 3
    assert "r2_score_std" in serialized[0]
