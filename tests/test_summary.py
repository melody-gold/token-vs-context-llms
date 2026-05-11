import json

import pytest

from token_vs_context_llms.summary import format_metrics_summary, load_metrics_json


def test_format_metrics_summary_orders_layers_and_formats_table() -> None:
    metrics = [
        {
            "layer_index": 3,
            "mean_squared_error": 0.125,
            "r2_score": 0.5,
            "mean_cosine_similarity": 0.75,
            "num_train_tokens": 80,
            "num_test_tokens": 20,
        },
        {
            "layer_index": 1,
            "mean_squared_error": 0.25,
            "r2_score": -0.125,
            "mean_cosine_similarity": 0.625,
            "num_train_tokens": 80,
            "num_test_tokens": 20,
        },
    ]

    summary = format_metrics_summary(metrics, title="Debug Run")

    assert summary.startswith("# Debug Run\n")
    assert "| Layer | MSE | R^2 | Mean cosine | Train tokens | Test tokens |" in summary
    assert summary.index("| 1 | 0.25 | -0.125 | 0.625 | 80 | 20 |") < summary.index(
        "| 3 | 0.125 | 0.5 | 0.75 | 80 | 20 |"
    )


def test_format_metrics_summary_includes_repeated_split_uncertainty() -> None:
    metrics = [
        {
            "layer_index": 0,
            "mean_squared_error": 0.125,
            "r2_score": 0.5,
            "mean_cosine_similarity": 0.75,
            "num_train_tokens": 80,
            "num_test_tokens": 20,
            "mean_squared_error_std": 0.01,
            "r2_score_std": 0.02,
            "mean_cosine_similarity_std": 0.03,
            "num_splits": 3,
            "random_seeds": [0, 1, 2],
        }
    ]

    summary = format_metrics_summary(metrics, title="Repeated Run")

    assert (
        "| Layer | MSE | R^2 | Mean cosine | Train tokens | Test tokens | Splits | Seeds |"
        in summary
    )
    assert (
        "| 0 | 0.125 +/- 0.01 | 0.5 +/- 0.02 | 0.75 +/- 0.03 | "
        "80 | 20 | 3 | 0, 1, 2 |"
        in summary
    )


def test_format_metrics_summary_rejects_empty_metrics() -> None:
    with pytest.raises(ValueError, match="empty metrics"):
        format_metrics_summary([])


def test_load_metrics_json_requires_list_of_objects(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"layer_index": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a list"):
        load_metrics_json(metrics_path)
