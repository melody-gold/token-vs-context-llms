from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_METRIC_COLUMNS = (
    "layer_index",
    "mean_squared_error",
    "r2_score",
    "mean_cosine_similarity",
    "num_train_tokens",
    "num_test_tokens",
)


def load_metrics_json(path: str | Path) -> list[dict[str, Any]]:
    """Load serialized layer metrics from a JSON file."""

    source = Path(path)
    loaded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(f"Expected a list of metric rows in {source}.")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(loaded):
        if not isinstance(row, dict):
            raise ValueError(f"Metric row {index} in {source} is not an object.")
        rows.append(row)
    return rows


def write_metrics_summary(path: str | Path, metrics: list[dict[str, Any]], title: str) -> None:
    """Write a compact Markdown summary for layerwise probe metrics."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_metrics_summary(metrics, title=title), encoding="utf-8")


def format_metrics_summary(metrics: list[dict[str, Any]], title: str = "Probe Metrics") -> str:
    """Format layerwise probe metrics as a Markdown table.

    Args:
        metrics: Serialized layer metric dictionaries.
        title: Markdown heading for the summary.

    Returns:
        A Markdown document containing one row per layer.
    """

    if not metrics:
        raise ValueError("Cannot summarize an empty metrics list.")

    for index, row in enumerate(metrics):
        missing_columns = [column for column in REQUIRED_METRIC_COLUMNS if column not in row]
        if missing_columns:
            joined_columns = ", ".join(missing_columns)
            raise ValueError(f"Metric row {index} is missing required columns: {joined_columns}.")

    sorted_metrics = sorted(metrics, key=lambda row: int(row["layer_index"]))
    has_repeated_splits = all("num_splits" in row for row in sorted_metrics)
    if has_repeated_splits:
        lines = [
            f"# {title}",
            "",
            "| Layer | MSE | R^2 | Mean cosine | Train tokens | Test tokens | Splits | Seeds |",
            "|---:|---:|---:|---:|---:|---:|---:|:---|",
        ]
    else:
        lines = [
            f"# {title}",
            "",
            "| Layer | MSE | R^2 | Mean cosine | Train tokens | Test tokens |",
            "|---:|---:|---:|---:|---:|---:|",
        ]

    for row in sorted_metrics:
        metric_values = [
            _format_metric_with_optional_std(row, "mean_squared_error"),
            _format_metric_with_optional_std(row, "r2_score"),
            _format_metric_with_optional_std(row, "mean_cosine_similarity"),
            str(int(row["num_train_tokens"])),
            str(int(row["num_test_tokens"])),
        ]
        if has_repeated_splits:
            metric_values.extend(
                [
                    str(int(row["num_splits"])),
                    _format_seeds(row.get("random_seeds", [])),
                ]
            )

        lines.append(
            "| "
            + " | ".join(
                [str(int(row["layer_index"])), *metric_values]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def _format_float(value: Any) -> str:
    """Format metric values with enough precision for compact comparisons."""

    return f"{float(value):.6g}"


def _format_metric_with_optional_std(row: dict[str, Any], metric_name: str) -> str:
    """Format metric values as mean or mean +/- split standard deviation."""

    mean = _format_float(row[metric_name])
    std_name = f"{metric_name}_std"
    if std_name not in row:
        return mean
    return f"{mean} +/- {_format_float(row[std_name])}"


def _format_seeds(seeds: Any) -> str:
    """Format repeated split seeds for a compact Markdown table cell."""

    if not isinstance(seeds, list):
        return ""
    return ", ".join(str(int(seed)) for seed in seeds)
