from __future__ import annotations

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    residual = y_true - y_pred
    denominator = np.sum(centered**2)
    numerator = np.sum(residual**2)

    if np.isclose(denominator, 0.0):
        return 1.0 if np.isclose(numerator, 0.0) else 0.0

    return float(1.0 - numerator / denominator)


def mean_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_norms = np.linalg.norm(y_true, axis=1)
    pred_norms = np.linalg.norm(y_pred, axis=1)
    denominator = np.clip(true_norms * pred_norms, a_min=1e-12, a_max=None)
    numerators = np.sum(y_true * y_pred, axis=1)
    return float(np.mean(numerators / denominator))
