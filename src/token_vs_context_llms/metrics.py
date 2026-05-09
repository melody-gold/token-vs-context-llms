from __future__ import annotations

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the average squared difference across all entries.

    Args:
        y_true: target vectors
        y_pred: predicted vectors with the same shape as `y_true`

    Returns:
        MSE as a Python float
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # activation reconstruction error: average squared distance
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return pooled R-squared across all output dimensions.

    Args:
        y_true: target vectors
        y_pred: predicted vectors with the same shape as `y_true`

    Returns:
        pooled R-squared score as a Python float
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    residual = y_true - y_pred
    denominator = np.sum(centered**2)
    numerator = np.sum(residual**2)

    # explained activation variance across all token/layer targets
    if np.isclose(denominator, 0.0):
        return 1.0 if np.isclose(numerator, 0.0) else 0.0

    return float(1.0 - numerator / denominator)


def mean_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean row-wise cosine similarity between target and prediction vectors

    Args:
        y_true: target vectors with shape `[num_examples, hidden_size]`
        y_pred: predicted vectors with the same shape as `y_true`

    Returns:
        average cosine similarity as a Python float
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    true_norms = np.linalg.norm(y_true, axis=1)
    pred_norms = np.linalg.norm(y_pred, axis=1)
    # activation direction agreement, independent of vector norm scale
    denominator = np.clip(true_norms * pred_norms, a_min=1e-12, a_max=None)
    numerators = np.sum(y_true * y_pred, axis=1)
    return float(np.mean(numerators / denominator))
