"""Public high-level API for MLGuard."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from mlguard.adapters.base import ModelAdapter
from mlguard.checks import (
    distribution_shift,
    missing_value_robustness,
    model_comparison,
    noise_robustness,
    prediction_stability,
)
from mlguard.evaluation import Average, Task, evaluate_predictions
from mlguard.exceptions import InputValidationError, ModelExecutionError
from mlguard.results import (
    EvaluationReport,
    MetricGateResult,
    SuiteReport,
    TestResult,
    ValidationReport,
)

_CLASSIFICATION_METRICS = frozenset({"accuracy", "precision", "recall", "f1"})
_REGRESSION_METRICS = frozenset({"mae", "mse", "rmse", "r2", "max_absolute_error"})


def _normalize_metric_thresholds(
    thresholds: Mapping[str, float] | None,
    *,
    task: Task,
    option_name: str,
) -> dict[str, float]:
    """Validate configured metric thresholds before running the model."""
    if task not in {"classification", "regression"}:
        raise InputValidationError("task must be 'classification' or 'regression'.")
    if thresholds is None:
        return {}
    if not isinstance(thresholds, Mapping):
        raise InputValidationError(f"{option_name} must be a mapping of metric names to numbers.")

    supported = _CLASSIFICATION_METRICS if task == "classification" else _REGRESSION_METRICS
    normalized: dict[str, float] = {}
    for metric, threshold in thresholds.items():
        if not isinstance(metric, str) or metric not in supported:
            choices = ", ".join(sorted(supported))
            raise InputValidationError(
                f"Unsupported metric {metric!r} in {option_name} for {task}; "
                f"choose from: {choices}."
            )
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise InputValidationError(f"Threshold for {metric!r} must be a finite number.")
        value = float(threshold)
        if not math.isfinite(value):
            raise InputValidationError(f"Threshold for {metric!r} must be a finite number.")
        normalized[metric] = value
    return normalized


def _metric_gate_results(
    evaluation: EvaluationReport,
    minimum_metrics: Mapping[str, float],
    maximum_metrics: Mapping[str, float],
) -> tuple[MetricGateResult, ...]:
    """Compare the computed metrics with each explicitly configured limit."""
    results: list[MetricGateResult] = []
    for operator, thresholds in ((">=", minimum_metrics), ("<=", maximum_metrics)):
        for metric, threshold in thresholds.items():
            raw_actual = evaluation.metrics[metric]
            actual = float(raw_actual) if isinstance(raw_actual, Real) else None
            passed = actual is not None and (
                actual >= threshold if operator == ">=" else actual <= threshold
            )
            results.append(
                MetricGateResult(
                    metric=metric,
                    operator=operator,
                    threshold=threshold,
                    actual=actual,
                    passed=bool(passed),
                )
            )
    return tuple(results)


class ModelGuard:
    """Run reliability checks against a prediction-compatible model."""

    _CHECKS = {
        "prediction_stability": prediction_stability,
        "noise_robustness": noise_robustness,
        "distribution_shift": distribution_shift,
        "missing_value_robustness": missing_value_robustness,
        "model_comparison": model_comparison,
    }

    @classmethod
    def from_sklearn(cls, estimator: Any) -> ModelGuard:
        """Build a guard for a fitted scikit-learn estimator or Pipeline.

        This convenience constructor uses the optional scikit-learn adapter,
        which validates fitted state and preserves incoming data containers.
        Install the optional dependency with pip install mlguard[sklearn].
        """
        from mlguard.adapters.sklearn import SklearnAdapter

        return cls(SklearnAdapter(estimator))

    def __init__(self, model: ModelAdapter) -> None:
        if not callable(getattr(model, "predict", None)):
            raise TypeError("model must expose a callable predict(X) method")
        self.model = model

    def evaluate(
        self,
        X: Any,
        y_true: Any,
        *,
        task: Task,
        average: Average = "macro",
        labels: Sequence[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Evaluate predictions against ground-truth labels or target values.

        Classification reports include accuracy, precision, recall, F1,
        confusion matrix, and per-class metrics. Regression reports include
        MAE, MSE, RMSE, R-squared, and maximum absolute error.
        """
        try:
            y_pred = self.model.predict(X)
        except Exception as exc:
            raise ModelExecutionError(
                "The model failed while producing evaluation predictions."
            ) from exc

        return evaluate_predictions(
            y_true,
            y_pred,
            task=task,
            average=average,
            labels=labels,
            metadata=metadata,
        )

    def validate(
        self,
        X: Any,
        y_true: Any,
        *,
        task: Task,
        checks: tuple[str, ...] = ("prediction_stability", "noise_robustness"),
        average: Average = "macro",
        labels: Sequence[Any] | None = None,
        noise_scale: float = 0.01,
        random_state: int = 0,
        stability_repeats: int = 3,
        metadata: dict[str, Any] | None = None,
        reference_X: Any | None = None,
        shift_threshold: float = 0.25,
        variance_ratio_threshold: float = 2.0,
        missing_fraction: float = 0.05,
        baseline_model: ModelAdapter | None = None,
        comparison_mode: str = "auto",
        atol: float = 1e-8,
        rtol: float = 1e-5,
        max_changed_fraction: float = 0.0,
        minimum_metrics: Mapping[str, float] | None = None,
        maximum_metrics: Mapping[str, float] | None = None,
    ) -> ValidationReport:
        """Run reliability checks and dataset evaluation as one workflow.

        Metric gates are opt-in. If configured, release_passed is true only
        when all selected reliability checks and all metric gates pass. Metrics
        that cannot be computed (such as R-squared for one sample) fail a gate.
        """
        minimum = _normalize_metric_thresholds(
            minimum_metrics, task=task, option_name="minimum_metrics"
        )
        maximum = _normalize_metric_thresholds(
            maximum_metrics, task=task, option_name="maximum_metrics"
        )
        report_metadata = dict(metadata or {})
        reliability = self.run(
            X,
            checks=checks,
            noise_scale=noise_scale,
            random_state=random_state,
            stability_repeats=stability_repeats,
            metadata=report_metadata,
            reference_X=reference_X,
            shift_threshold=shift_threshold,
            variance_ratio_threshold=variance_ratio_threshold,
            missing_fraction=missing_fraction,
            baseline_model=baseline_model,
            comparison_mode=comparison_mode,
            atol=atol,
            rtol=rtol,
            max_changed_fraction=max_changed_fraction,
        )
        evaluation = self.evaluate(
            X,
            y_true,
            task=task,
            average=average,
            labels=labels,
            metadata=report_metadata,
        )
        metric_gates = _metric_gate_results(evaluation, minimum, maximum)
        return ValidationReport(
            reliability=reliability,
            evaluation=evaluation,
            metric_gates=metric_gates,
            metadata=report_metadata,
        )

    def run(
        self,
        X: Any,
        *,
        checks: tuple[str, ...] = ("prediction_stability", "noise_robustness"),
        noise_scale: float = 0.01,
        random_state: int = 0,
        stability_repeats: int = 3,
        metadata: dict[str, Any] | None = None,
        reference_X: Any | None = None,
        shift_threshold: float = 0.25,
        variance_ratio_threshold: float = 2.0,
        missing_fraction: float = 0.05,
        baseline_model: ModelAdapter | None = None,
        comparison_mode: str = "auto",
        atol: float = 1e-8,
        rtol: float = 1e-5,
        max_changed_fraction: float = 0.0,
    ) -> SuiteReport:
        """Execute requested checks and return a structured report."""
        if not checks:
            raise ValueError("At least one check must be requested.")
        unknown = sorted(set(checks) - self._CHECKS.keys())
        if unknown:
            raise ValueError(f"Unknown check(s): {', '.join(unknown)}")

        if "distribution_shift" in checks and reference_X is None:
            raise ValueError("reference_X is required for the distribution_shift check.")
        if "model_comparison" in checks and baseline_model is None:
            raise ValueError("baseline_model is required for the model_comparison check.")

        results: list[TestResult] = []
        for name in checks:
            check = self._CHECKS[name]
            if name == "prediction_stability":
                result = check(self.model, X, repeats=stability_repeats)
            elif name == "model_comparison":
                result = check(
                    baseline_model,
                    self.model,
                    X,
                    comparison_mode=comparison_mode,
                    atol=atol,
                    rtol=rtol,
                    max_changed_fraction=max_changed_fraction,
                )
            elif name == "distribution_shift":
                result = check(
                    reference_X,
                    X,
                    threshold=shift_threshold,
                    variance_ratio_threshold=variance_ratio_threshold,
                )
            elif name == "missing_value_robustness":
                result = check(
                    self.model,
                    X,
                    missing_fraction=missing_fraction,
                    random_state=random_state,
                )
            elif name == "noise_robustness":
                result = check(
                    self.model, X, noise_scale=noise_scale, random_state=random_state
                )
            else:
                result = check(self.model, X)
            results.append(result)
        return SuiteReport(results=tuple(results), metadata=dict(metadata or {}))
