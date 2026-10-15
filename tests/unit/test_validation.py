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



def test_metric_thresholds_control_release_gate_status():
    X = [[0.1], [0.6], [0.9]]
    y_true = [0, 0, 1]
    report = ModelGuard(ThresholdModel()).validate(
        X,
        y_true,
        task="classification",
        checks=("prediction_stability",),
        minimum_metrics={"accuracy": 0.8},
        maximum_metrics={"f1": 0.9},
    )

    assert report.evaluation.metrics["accuracy"] == pytest.approx(2 / 3)
    assert report.metric_gates[0].metric == "accuracy"
    assert report.metric_gates[0].operator == ">="
    assert report.metric_gates[0].passed is False
    assert report.performance_passed is False
    assert report.release_passed is False
    payload = report.to_dict()
    assert payload["performance_gate_configured"] is True
    assert payload["release_passed"] is False


def test_metric_thresholds_pass_and_can_be_used_for_regression():
    class FirstFeatureRegressor:
        def predict(self, X):
            return np.asarray(X, dtype=float)[:, 0]

    report = ModelGuard(FirstFeatureRegressor()).validate(
        [[1.0], [2.0], [3.0]],
        [1.0, 2.5, 2.0],
        task="regression",
        checks=("prediction_stability",),
        maximum_metrics={"mae": 0.5, "rmse": 0.75},
    )

    assert report.performance_passed is True
    assert report.release_passed is True
    assert all(gate.passed for gate in report.metric_gates)


def test_metric_thresholds_reject_unknown_metric_and_non_finite_value():
    from mlguard.exceptions import InputValidationError

    guard = ModelGuard(ThresholdModel())
    with pytest.raises(InputValidationError, match="Unsupported metric"):
        guard.validate(
            [[0.1], [0.9]],
            [0, 1],
            task="classification",
            checks=("prediction_stability",),
            minimum_metrics={"loss": 0.1},
        )
    with pytest.raises(InputValidationError, match="finite number"):
        guard.validate(
            [[0.1], [0.9]],
            [0, 1],
            task="classification",
            checks=("prediction_stability",),
            minimum_metrics={"accuracy": float("nan")},
        )


def test_release_status_is_none_when_no_performance_gate_is_configured():
    report = ModelGuard(ThresholdModel()).validate(
        [[0.1], [0.9]],
        [0, 1],
        task="classification",
        checks=("prediction_stability",),
    )

    assert report.performance_passed is None
    assert report.release_passed is None
    assert report.to_dict()["performance_gate_configured"] is False



def test_run_passes_variance_ratio_threshold_to_distribution_check():
    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    candidate = np.array([[-1.5], [1.5], [-1.5], [1.5]])
    report = ModelGuard(ThresholdModel()).run(
        candidate,
        checks=("distribution_shift",),
        reference_X=reference,
        variance_ratio_threshold=1.5,
    )

    assert report.results[0].status == "failed"
    assert report.results[0].metrics["variance_ratio_threshold"] == 1.5
