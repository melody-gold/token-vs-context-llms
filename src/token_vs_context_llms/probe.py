from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from token_vs_context_llms.metrics import mean_cosine_similarity, mean_squared_error, r2_score


@dataclass(slots=True)
class LinearProbe:
    """Affine map from token embeddings to a target representation."""

    weights: np.ndarray
    bias: np.ndarray
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Apply the learned affine map to a batch of input vectors.

        Args:
            x: input matrix with shape `[num_examples, input_dim]`

        Returns:
            predicted target matrix with shape `[num_examples, output_dim]`
        """

        return x @ self.weights + self.bias


RidgeProbe = LinearProbe


@dataclass(slots=True)
class LayerMetric:
    """Evaluation metrics for one probed model layer."""

    layer_index: int
    mean_squared_error: float
    r2_score: float
    mean_cosine_similarity: float
    num_train_tokens: int
    num_test_tokens: int


def fit_affine_probe(x: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> LinearProbe:
    """Fit a token-only affine probe from inputs `x` to targets `y`.

    Args:
        x: input token embeddings with shape `[num_examples, input_dim]`
        y: target representations with shape `[num_examples, output_dim]`
        alpha: Ridge penalty. `0.0` uses ordinary least squares; positive
            values use ridge regularization

    Returns:
        a fitted `LinearProbe` containing weights, bias, and the alpha used

    Raises:
        ValueError: if inputs are not 2D, row counts differ, or alpha is negative
    """

    # use float64 for a more stable least-squares solve
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("fit_affine_probe expects 2D arrays for both x and y.")

    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows.")

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    if alpha == 0:
        # add a column of ones so ordinary least squares learns the bias directly
        design = np.column_stack([x, np.ones(x.shape[0], dtype=np.float64)])
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
        weights = solution[:-1]
        bias = solution[-1]
        return LinearProbe(weights=weights, bias=bias, alpha=alpha)

    # centering lets ridge regularize only the weights while the bias stays unpenalized
    x_mean = np.mean(x, axis=0, keepdims=True)
    y_mean = np.mean(y, axis=0, keepdims=True)
    x_centered = x - x_mean
    y_centered = y - y_mean

    gram = x_centered.T @ x_centered
    # diagonal penalty shrinks weights and can help if the solve is ill-conditioned
    regularizer = alpha * np.eye(gram.shape[0], dtype=np.float64)
    weights = np.linalg.solve(gram + regularizer, x_centered.T @ y_centered)
    bias = np.ravel(y_mean - x_mean @ weights)
    return LinearProbe(weights=weights, bias=bias, alpha=alpha)


def fit_ridge_probe(x: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> LinearProbe:
    """Backward-compatible wrapper for fitting an affine probe.

    Args:
        x: input token embeddings with shape `[num_examples, input_dim]`
        y: target representations with shape `[num_examples, output_dim]`
        alpha: ridge penalty. `0.0` gives the simple affine baseline

    Returns:
        a fitted `LinearProbe`
    """

    return fit_affine_probe(x, y, alpha=alpha)


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.2,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle aligned arrays and return train/test splits for probe evaluation.

    Args:
        x: input examples
        y: target examples aligned row-by-row with `x`
        test_fraction: fraction of examples to reserve for testing
        random_seed: seed for the deterministic shuffle

    Returns:
        tuple `(x_train, x_test, y_train, y_test)`

    Raises:
        ValueError: if the split fraction is invalid or too few examples exist
    """

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    num_examples = x.shape[0]
    if num_examples < 2:
        raise ValueError("Need at least 2 examples to create a train/test split.")

    rng = np.random.default_rng(random_seed)
    # shuffle once and use the same indices for x and y so rows stay aligned
    indices = rng.permutation(num_examples)
    num_test = max(1, int(round(num_examples * test_fraction)))
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    if train_indices.size == 0:
        raise ValueError("Train split is empty; decrease test_fraction or add more data.")

    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def evaluate_probe(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.0,
    test_fraction: float = 0.2,
    random_seed: int = 0,
) -> tuple[LinearProbe, LayerMetric]:
    """Fit one probe and compute held-out reconstruction metrics.

    Args:
        x: input token embeddings
        y: target representations for one layer or feature set
        alpha: ridge penalty passed to `fit_affine_probe`
        test_fraction: fraction of rows held out for evaluation
        random_seed: seed for the train/test split

    Returns:
        tuple containing the fitted probe and its held-out metrics
    """

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_fraction=test_fraction, random_seed=random_seed
    )
    model = fit_affine_probe(x_train, y_train, alpha=alpha)
    predictions = model.predict(x_test)
    # layer index is filled in by the layerwise wrapper below
    metric = LayerMetric(
        layer_index=-1,
        mean_squared_error=mean_squared_error(y_test, predictions),
        r2_score=r2_score(y_test, predictions),
        mean_cosine_similarity=mean_cosine_similarity(y_test, predictions),
        num_train_tokens=int(x_train.shape[0]),
        num_test_tokens=int(x_test.shape[0]),
    )
    return model, metric


def evaluate_hidden_state_layers(
    token_embeddings: np.ndarray,
    hidden_states: np.ndarray,
    layer_indices: np.ndarray,
    alpha: float = 0.0,
    test_fraction: float = 0.2,
    random_seed: int = 0,
) -> list[LayerMetric]:
    """Evaluate one token-only probe for each selected hidden-state layer.

    Args:
        token_embeddings: input embedding matrix with one row per token
        hidden_states: hidden-state tensor with shape
            `[num_tokens, num_layers, hidden_size]`
        layer_indices: original model layer index for each hidden-state slice
        alpha: ridge penalty passed to each probe
        test_fraction: fraction of tokens held out for evaluation
        random_seed: seed for the train/test split

    Returns:
        One `LayerMetric` per selected layer

    Raises:
        ValueError: if `hidden_states` or `layer_indices` has an unexpected shape
    """

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [num_tokens, num_layers, hidden_size].")

    if hidden_states.shape[1] != len(layer_indices):
        raise ValueError("layer_indices must match the second hidden_states dimension.")

    metrics: list[LayerMetric] = []
    for local_layer, layer_index in enumerate(layer_indices):
        # use same input embeddings, but change target to this layer's states
        _, metric = evaluate_probe(
            token_embeddings,
            hidden_states[:, local_layer, :],
            alpha=alpha,
            test_fraction=test_fraction,
            random_seed=random_seed,
        )
        metric.layer_index = int(layer_index)
        metrics.append(metric)

    return metrics


def serialize_metrics(metrics: list[LayerMetric]) -> list[dict[str, Any]]:
    """Convert metric dataclasses to JSON-serializable dictionaries.

    Args:
        metrics: layer metrics from probe evaluation

    Returns:
        list of dictionaries that can be passed to `json.dumps`
    """

    return [asdict(metric) for metric in metrics]
