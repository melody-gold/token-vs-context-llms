import numpy as np

from token_vs_context_llms.probe import evaluate_hidden_state_layers, evaluate_probe


def test_evaluate_probe_recovers_linear_mapping() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    weights = rng.normal(size=(5, 3))
    bias = np.array([0.1, -0.2, 0.3])
    y = x @ weights + bias

    _, metrics = evaluate_probe(x, y, alpha=1e-8, test_fraction=0.25, random_seed=3)
    assert metrics.mean_squared_error < 1e-12
    assert metrics.r2_score > 0.999999
    assert metrics.mean_cosine_similarity > 0.999999


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
        alpha=1e-8,
        test_fraction=0.25,
        random_seed=7,
    )

    assert [metric.layer_index for metric in metrics] == [2, 5]
    assert len(metrics) == 2
