"""Tests for dataset-level classification and regression metrics."""

import json

import numpy as np
import pytest

from mlguard import EvaluationReport
from mlguard.evaluation import evaluate_predictions
from mlguard.exceptions import InputValidationError


def test_classification_metrics_include_confusion_matrix_and_per_class_scores():
    report = evaluate_predictions(
        ["cat", "cat", "dog", "dog"],
        ["cat", "dog", "dog", "dog"],
        task="classification",
    )

    assert isinstance(report, EvaluationReport)
    assert report.sample_count == 4
    assert report.metrics["accuracy"] == pytest.approx(0.75)
    assert report.metrics["precision"] == pytest.approx(5 / 6)
    assert report.metrics["recall"] == pytest.approx(0.75)
    assert report.metrics["f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert report.metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert [item["label"] for item in report.metrics["per_class"]] == ["cat", "dog"]
    assert report.metrics["per_class"][0]["support"] == 2


def test_classification_supports_weighted_and_micro_averaging():
    y_true = [0, 0, 0, 1]
    y_pred = [0, 1, 0, 1]

    weighted = evaluate_predictions(
        y_true, y_pred, task="classification", average="weighted"
    )
    micro = evaluate_predictions(
        y_true, y_pred, task="classification", average="micro"
    )

    assert weighted.metrics["average"] == "weighted"
    assert micro.metrics["precision"] == pytest.approx(micro.metrics["accuracy"])
    assert micro.metrics["recall"] == pytest.approx(micro.metrics["accuracy"])
    assert micro.metrics["f1"] == pytest.approx(micro.metrics["accuracy"])


def test_classification_supports_explicit_label_order_and_absent_class():
    report = evaluate_predictions(
        [1, 2, 1],
        [1, 1, 1],
        task="classification",
        labels=[2, 1, 3],
    )

    assert report.metrics["labels"] == [2, 1, 3]
    assert report.metrics["confusion_matrix"] == [
        [0, 1, 0],
        [0, 2, 0],
        [0, 0, 0],
    ]
    assert report.metrics["per_class"][2]["f1"] == 0.0


def test_evaluation_report_is_json_serializable():
    report = evaluate_predictions(
        [0, 1, 1],
        [0, 0, 1],
        task="classification",
        metadata={"dataset": "validation"},
    )

    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["task"] == "classification"
    assert payload["sample_count"] == 3
    assert payload["metadata"]["dataset"] == "validation"
    assert payload["schema_version"] == "1.0"


def test_regression_metrics_compute_common_error_measures():
    report = evaluate_predictions(
        [3.0, -0.5, 2.0, 7.0],
        [2.5, 0.0, 2.0, 8.0],
        task="regression",
    )

    assert report.metrics["mae"] == pytest.approx(0.5)
    assert report.metrics["mse"] == pytest.approx(0.375)
    assert report.metrics["rmse"] == pytest.approx(np.sqrt(0.375))
    assert report.metrics["r2"] == pytest.approx(1.0 - 1.5 / 29.1875)
    assert report.metrics["max_absolute_error"] == pytest.approx(1.0)


def test_r2_is_none_when_only_one_regression_sample_is_provided():
    report = evaluate_predictions([2.0], [2.5], task="regression")

    assert report.sample_count == 1
    assert report.metrics["r2"] is None
    assert report.metrics["mae"] == pytest.approx(0.5)


def test_r2_handles_constant_ground_truth():
    perfect = evaluate_predictions([2.0, 2.0], [2.0, 2.0], task="regression")
    imperfect = evaluate_predictions([2.0, 2.0], [2.0, 3.0], task="regression")

    assert perfect.metrics["r2"] == 1.0
    assert imperfect.metrics["r2"] == 0.0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "kwargs", "message"),
    [
        ([0, 1], [0], {"task": "classification"}, "same number of samples"),
        ([[0], [1]], [[0], [1]], {"task": "classification"}, "one-dimensional"),
        ([], [], {"task": "classification"}, "at least one sample"),
        ([0, 1], [0, 1], {"task": "unknown"}, "task must be"),
        ([0, 1], [0, 1], {"task": "classification", "average": "invalid"}, "average must be"),
        ([0, 1], [0, 1], {"task": "classification", "labels": [0]}, "include every class"),
        ([0, 1], [0, 1], {"task": "classification", "labels": [0, 0, 1]}, "duplicate"),
        ([1, 2], [1, 2], {"task": "regression", "labels": [1, 2]}, "only supported"),
        ([1.0, np.nan], [1.0, 2.0], {"task": "regression"}, "finite numeric"),
        ([1.0, 2.0], [1.0, np.inf], {"task": "regression"}, "finite numeric"),
    ],
)
def test_evaluation_rejects_invalid_inputs(y_true, y_pred, kwargs, message):
    with pytest.raises(InputValidationError, match=message):
        evaluate_predictions(y_true, y_pred, **kwargs)


def test_classification_rejects_non_scalar_labels():
    with pytest.raises(InputValidationError, match="class labels"):
        evaluate_predictions(
            [{"label": "cat"}, {"label": "dog"}],
            [{"label": "cat"}, {"label": "dog"}],
            task="classification",
        )
