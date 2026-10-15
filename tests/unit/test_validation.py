"""Tests for the combined model validation workflow."""

import json

import numpy as np
import pytest

from mlguard import ModelGuard, ValidationReport


class ThresholdModel:
    """Small deterministic classifier for high-level API tests."""

    def predict(self, X):
        values = np.asarray(X, dtype=float)
        return (values[:, 0] > 0.5).astype(int)


def test_validate_combines_reliability_checks_and_classification_metrics():
    X = np.array([[0.1, 0.2], [0.6, 0.3], [0.8, 0.9]])
    y_true = np.array([0, 1, 1])

    report = ModelGuard(ThresholdModel()).validate(
        X,
        y_true,
        task="classification",
        checks=("prediction_stability", "noise_robustness"),
        noise_scale=0.0,
        random_state=11,
        metadata={"model_version": "demo", "split": "validation"},
    )

    assert isinstance(report, ValidationReport)
    assert report.reliability_passed
    assert report.reliability.passed
    assert [result.name for result in report.reliability.results] == [
        "prediction_stability",
        "noise_robustness",
    ]
    assert report.evaluation.task == "classification"
    assert report.evaluation.sample_count == 3
    assert report.evaluation.metrics["accuracy"] == pytest.approx(1.0)
    assert report.metadata == {"model_version": "demo", "split": "validation"}
    assert report.reliability.metadata == report.metadata
    assert report.evaluation.metadata == report.metadata
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["reliability_passed"] is True


def test_validate_supports_regression_metrics():
    class FirstFeatureRegressor:
        def predict(self, X):
            return np.asarray(X, dtype=float)[:, 0]

    report = ModelGuard(FirstFeatureRegressor()).validate(
        [[1.0], [2.0], [3.0]],
        [1.0, 2.5, 2.0],
        task="regression",
        checks=("prediction_stability",),
    )

    assert report.reliability_passed
    assert report.evaluation.task == "regression"
    assert report.evaluation.metrics["mae"] == pytest.approx(0.5)
    assert report.evaluation.metrics["mse"] == pytest.approx(1.25 / 3)


def test_validate_requires_reference_data_for_distribution_shift(constant_model, sample_X):
    with pytest.raises(ValueError, match="reference_X is required"):
        ModelGuard(constant_model).validate(
            sample_X,
            [0] * len(sample_X),
            task="classification",
            checks=("distribution_shift",),
        )


def test_validate_does_not_report_an_unqualified_overall_quality_pass():
    report = ModelGuard(ThresholdModel()).validate(
        [[0.1], [0.9]],
        [0, 0],
        task="classification",
        checks=("prediction_stability",),
    )

    payload = report.to_dict()
    assert report.reliability_passed
    assert "passed" not in payload
    assert payload["evaluation"]["metrics"]["accuracy"] == pytest.approx(0.5)
