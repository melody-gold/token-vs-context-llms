import numpy as np

from token_vs_context_llms.metrics import mean_cosine_similarity, mean_squared_error, r2_score


def test_metrics_are_perfect_for_identical_arrays() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert mean_squared_error(values, values) == 0.0
    assert r2_score(values, values) == 1.0
    assert np.isclose(mean_cosine_similarity(values, values), 1.0)


def test_mean_cosine_similarity_handles_scaled_vectors() -> None:
    truth = np.array([[1.0, 0.0], [0.0, 2.0]])
    prediction = np.array([[4.0, 0.0], [0.0, 5.0]])
    assert np.isclose(mean_cosine_similarity(truth, prediction), 1.0)
