"""Initial built-in reliability checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlguard._validation import as_array, require_numeric, validate_same_sample_count
from mlguard.adapters.base import ModelAdapter
from mlguard.exceptions import InputValidationError
from mlguard.results import Finding, TestResult


def _predict(model: ModelAdapter, X: Any) -> np.ndarray:
    try:
        return as_array(model.predict(X), name="predictions")
    except InputValidationError:
        raise
    except Exception as exc:
        raise RuntimeError("Model prediction failed.") from exc


def _equal_predictions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compare predictions row-wise, supporting numeric and object labels."""
    if left.shape != right.shape:
        raise InputValidationError(
            f"Prediction shapes differ: {left.shape!r} versus {right.shape!r}."
        )
    if left.ndim == 1:
        return np.asarray(left == right, dtype=bool)
    return np.asarray([np.array_equal(a, b) for a, b in zip(left, right, strict=True)])


def prediction_stability(model: ModelAdapter, X: Any) -> TestResult:
    """Check whether repeated predictions on the same data are deterministic."""
    first = _predict(model, X)
    second = _predict(model, X)
    stable = _equal_predictions(first, second)
    agreement = float(stable.mean())
    findings = () if agreement == 1.0 else (
        Finding(
            code="NON_DETERMINISTIC_PREDICTIONS",
            message="Repeated predictions differed for some samples.",
            details={"changed_samples": int((~stable).sum())},
        ),
    )
    return TestResult(
        name="prediction_stability",
        status="passed" if agreement == 1.0 else "failed",
        summary=f"Prediction agreement across repeated calls: {agreement:.3f}.",
        metrics={"prediction_agreement": agreement, "changed_samples": int((~stable).sum())},
        findings=findings,
    )


def noise_robustness(
    model: ModelAdapter,
    X: Any,
    *,
    noise_scale: float = 0.01,
    random_state: int = 0,
) -> TestResult:
    """Measure prediction agreement after adding seeded Gaussian feature noise.

    Numeric input is required. The function creates a perturbed copy and never
    mutates caller-owned data. ``noise_scale`` is an absolute standard deviation.
    """
    if not np.isfinite(noise_scale) or noise_scale < 0:
        raise InputValidationError("noise_scale must be a finite, non-negative number.")
    original = as_array(X, name="X")
    numeric = require_numeric(original, name="X")
    baseline = _predict(model, X)
    rng = np.random.default_rng(random_state)
    perturbed = numeric + rng.normal(0.0, noise_scale, size=numeric.shape)
    changed_input: Any = perturbed
    noisy_predictions = _predict(model, changed_input)
    validate_same_sample_count(baseline, noisy_predictions)
    same = _equal_predictions(baseline, noisy_predictions)
    agreement = float(same.mean())
    findings = () if agreement == 1.0 else (
        Finding(
            code="PREDICTIONS_CHANGED_UNDER_NOISE",
            message="Some predictions changed after Gaussian input noise was applied.",
            details={"changed_samples": int((~same).sum()), "noise_scale": noise_scale},
        ),
    )
    return TestResult(
        name="noise_robustness",
        status="passed" if agreement == 1.0 else "failed",
        summary=f"Prediction agreement after noise: {agreement:.3f}.",
        metrics={
            "prediction_agreement": agreement,
            "changed_samples": int((~same).sum()),
            "noise_scale": float(noise_scale),
            "random_state": int(random_state),
        },
        findings=findings,
    )


def distribution_shift(
    reference_X: Any,
    candidate_X: Any,
    *,
    threshold: float = 0.25,
) -> TestResult:
    """Compare numeric feature distributions using standardized mean differences.

    reference_X is the reference sample and candidate_X is the newer sample.
    Each row is a sample and each column a feature. This is a descriptive
    screening check, not a statistical hypothesis test.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise InputValidationError("threshold must be a finite, non-negative number.")

    reference = require_numeric(as_array(reference_X, name="reference_X"), name="reference_X")
    candidate = require_numeric(as_array(candidate_X, name="candidate_X"), name="candidate_X")
    if reference.ndim != 2 or candidate.ndim != 2:
        raise InputValidationError("reference_X and candidate_X must be 2D feature matrices.")
    if reference.shape[1] != candidate.shape[1]:
        raise InputValidationError(
            "reference_X and candidate_X must have the same number of features."
        )

    ref_mean = reference.mean(axis=0)
    cand_mean = candidate.mean(axis=0)
    ref_var = (
        reference.var(axis=0, ddof=1)
        if reference.shape[0] > 1
        else np.zeros(reference.shape[1])
    )
    cand_var = (
        candidate.var(axis=0, ddof=1)
        if candidate.shape[0] > 1
        else np.zeros(candidate.shape[1])
    )
    pooled_scale = np.sqrt((ref_var + cand_var) / 2.0)
    differences = np.abs(cand_mean - ref_mean)
    standardized = np.divide(
        differences,
        pooled_scale,
        out=np.where(differences == 0, 0.0, np.inf),
        where=pooled_scale > 0,
    )
    shifted_indices = np.flatnonzero(standardized > threshold)
    shifted = shifted_indices.size > 0
    findings = tuple(
        Finding(
            code="FEATURE_DISTRIBUTION_SHIFT",
            message=f"Feature {int(index)} exceeds the standardized mean-difference threshold.",
            details={
                "feature_index": int(index),
                "standardized_mean_difference": float(standardized[index]),
                "threshold": float(threshold),
            },
        )
        for index in shifted_indices
    )
    return TestResult(
        name="distribution_shift",
        status="failed" if shifted else "passed",
        summary=(
            f"{int(shifted_indices.size)} feature(s) exceeded the shift "
            f"threshold of {threshold:.3f}."
        ),
        metrics={
            "threshold": float(threshold),
            "shifted_feature_count": int(shifted_indices.size),
            "feature_count": int(reference.shape[1]),
            "max_standardized_mean_difference": float(standardized.max(initial=0.0)),
            "standardized_mean_differences": [float(value) for value in standardized],
        },
        findings=findings,
    )


def missing_value_robustness(
    model: ModelAdapter,
    X: Any,
    *,
    missing_fraction: float = 0.05,
    random_state: int = 0,
) -> TestResult:
    """Check whether a model returns valid predictions when features contain NaNs.

    A reproducible subset of cells in a numeric 2D copy of X is replaced with
    NaN. The original input is not mutated. A model that rejects NaN values, or
    emits non-finite predictions after receiving them, fails this check.
    """
    if (
        not np.isfinite(missing_fraction)
        or missing_fraction <= 0
        or missing_fraction > 1
    ):
        raise InputValidationError(
            "missing_fraction must be greater than 0 and at most 1."
        )

    original = require_numeric(as_array(X, name="X"), name="X")
    if original.ndim != 2 or original.shape[1] == 0:
        raise InputValidationError(
            "X must be a 2D feature matrix with at least one feature."
        )

    baseline = _predict(model, X)
    validate_same_sample_count(original, baseline)

    total_cells = int(original.size)
    missing_cells = max(1, int(np.ceil(total_cells * missing_fraction)))
    rng = np.random.default_rng(random_state)
    masked = original.copy()
    masked.flat[rng.choice(total_cells, size=missing_cells, replace=False)] = np.nan

    metrics = {
        "requested_missing_fraction": float(missing_fraction),
        "actual_missing_fraction": float(missing_cells / total_cells),
        "missing_cells": missing_cells,
        "total_cells": total_cells,
        "random_state": int(random_state),
    }
    try:
        predictions = _predict(model, masked)
        validate_same_sample_count(original, predictions)
    except Exception as exc:
        underlying = exc.__cause__ if exc.__cause__ is not None else exc
        error_type = type(underlying).__name__
        return TestResult(
            name="missing_value_robustness",
            status="failed",
            summary=(
                "The model could not return predictions for input containing NaN values."
            ),
            metrics={**metrics, "error_type": error_type},
            findings=(
                Finding(
                    code="MODEL_REJECTED_MISSING_VALUES",
                    message=(
                        "The model raised an exception for input containing missing features."
                    ),
                    details={"error_type": error_type},
                ),
            ),
        )

    if np.issubdtype(predictions.dtype, np.number):
        invalid_predictions = int((~np.isfinite(predictions)).sum())
    elif predictions.dtype.kind == "O":
        invalid_predictions = sum(
            value is None
            or (
                isinstance(value, float | np.floating | complex | np.complexfloating)
                and not np.isfinite(value)
            )
            for value in predictions.flat
        )
    else:
        invalid_predictions = 0

    metrics["invalid_prediction_count"] = int(invalid_predictions)
    if invalid_predictions:
        return TestResult(
            name="missing_value_robustness",
            status="failed",
            summary=(
                "The model returned invalid predictions for input containing NaN values."
            ),
            metrics=metrics,
            findings=(
                Finding(
                    code="INVALID_PREDICTIONS_WITH_MISSING_INPUT",
                    message=(
                        "Some predictions were non-finite or null after missing values "
                        "were introduced."
                    ),
                    details={"invalid_prediction_count": int(invalid_predictions)},
                ),
            ),
        )

    return TestResult(
        name="missing_value_robustness",
        status="passed",
        summary=(
            "The model accepted missing feature values and returned valid predictions."
        ),
        metrics=metrics,
    )



def model_comparison(
    baseline_model: ModelAdapter,
    candidate_model: ModelAdapter,
    X: Any,
    *,
    comparison_mode: str = "auto",
    atol: float = 1e-8,
    rtol: float = 1e-5,
    max_changed_fraction: float = 0.0,
) -> TestResult:
    """Compare a candidate model's predictions against a baseline model.

    Classification predictions are compared exactly. Regression predictions
    are compared using NumPy's absolute and relative tolerances. In auto mode,
    floating-point numeric predictions use regression comparison; other
    prediction types use exact comparison. Set comparison_mode explicitly
    when a classifier emits floating-point labels.
    """
    if comparison_mode not in {"auto", "classification", "regression"}:
        raise InputValidationError(
            "comparison_mode must be 'auto', 'classification', or 'regression'."
        )
    if not np.isfinite(atol) or atol < 0:
        raise InputValidationError("atol must be a finite, non-negative number.")
    if not np.isfinite(rtol) or rtol < 0:
        raise InputValidationError("rtol must be a finite, non-negative number.")
    if (
        not np.isfinite(max_changed_fraction)
        or max_changed_fraction < 0
        or max_changed_fraction > 1
    ):
        raise InputValidationError(
            "max_changed_fraction must be between 0 and 1, inclusive."
        )
    if not callable(getattr(baseline_model, "predict", None)):
        raise InputValidationError("baseline_model must expose a callable predict(X) method.")
    if not callable(getattr(candidate_model, "predict", None)):
        raise InputValidationError("candidate_model must expose a callable predict(X) method.")

    try:
        baseline = _predict(baseline_model, X)
        candidate = _predict(candidate_model, X)
    except Exception as exc:
        error_type = type(exc).__name__
        return TestResult(
            name="model_comparison",
            status="error",
            summary="Could not compare models because a prediction call failed.",
            metrics={"error_type": error_type},
            findings=(
                Finding(
                    code="MODEL_COMPARISON_PREDICTION_ERROR",
                    message="The baseline or candidate model failed to produce predictions.",
                    severity="error",
                    details={"error_type": error_type},
                ),
            ),
        )

    if baseline.shape[0] != candidate.shape[0]:
        return TestResult(
            name="model_comparison",
            status="failed",
            summary="Baseline and candidate models returned different sample counts.",
            metrics={
                "baseline_sample_count": int(baseline.shape[0]),
                "candidate_sample_count": int(candidate.shape[0]),
            },
            findings=(
                Finding(
                    code="PREDICTION_SAMPLE_COUNT_CHANGED",
                    message="The models returned a different number of predictions.",
                    severity="error",
                    details={
                        "baseline_sample_count": int(baseline.shape[0]),
                        "candidate_sample_count": int(candidate.shape[0]),
                    },
                ),
            ),
        )

    if baseline.shape != candidate.shape:
        return TestResult(
            name="model_comparison",
            status="failed",
            summary="Baseline and candidate prediction shapes differ.",
            metrics={
                "baseline_prediction_shape": list(baseline.shape),
                "candidate_prediction_shape": list(candidate.shape),
            },
            findings=(
                Finding(
                    code="PREDICTION_SHAPE_CHANGED",
                    message="The candidate model changed the shape of its predictions.",
                    severity="error",
                    details={
                        "baseline_prediction_shape": list(baseline.shape),
                        "candidate_prediction_shape": list(candidate.shape),
                    },
                ),
            ),
        )

    numeric_kinds = "iuf"
    if comparison_mode == "auto":
        use_regression_comparison = (
            baseline.dtype.kind in numeric_kinds
            and candidate.dtype.kind in numeric_kinds
            and (baseline.dtype.kind == "f" or candidate.dtype.kind == "f")
        )
        effective_mode = "regression" if use_regression_comparison else "classification"
    else:
        effective_mode = comparison_mode

    metrics: dict[str, Any] = {
        "comparison_mode": effective_mode,
        "max_changed_fraction": float(max_changed_fraction),
        "atol": float(atol),
        "rtol": float(rtol),
        "compared_samples": int(baseline.shape[0]),
    }

    if effective_mode == "regression":
        try:
            baseline_numeric = baseline.astype(float)
            candidate_numeric = candidate.astype(float)
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                "Regression comparison requires numeric model predictions."
            ) from exc

        baseline_finite = np.isfinite(baseline_numeric)
        candidate_finite = np.isfinite(candidate_numeric)
        if not baseline_finite.all() or not candidate_finite.all():
            invalid_baseline = int((~baseline_finite).sum())
            invalid_candidate = int((~candidate_finite).sum())
            return TestResult(
                name="model_comparison",
                status="failed",
                summary="One or both models returned non-finite numeric predictions.",
                metrics={
                    **metrics,
                    "invalid_baseline_prediction_count": invalid_baseline,
                    "invalid_candidate_prediction_count": invalid_candidate,
                },
                findings=(
                    Finding(
                        code="NON_FINITE_MODEL_PREDICTIONS",
                        message="Regression comparison requires finite predictions from both models.",
                        severity="error",
                        details={
                            "invalid_baseline_prediction_count": invalid_baseline,
                            "invalid_candidate_prediction_count": invalid_candidate,
                        },
                    ),
                ),
            )

        close = np.isclose(
            baseline_numeric,
            candidate_numeric,
            atol=atol,
            rtol=rtol,
            equal_nan=False,
        )
        changed_rows = ~close.reshape(close.shape[0], -1).all(axis=1)
        absolute_difference = np.abs(candidate_numeric - baseline_numeric)
        metrics["mean_absolute_difference"] = float(absolute_difference.mean())
        metrics["max_absolute_difference"] = float(absolute_difference.max(initial=0.0))
    else:
        same = _equal_predictions(baseline, candidate)
        changed_rows = ~same

    changed_samples = int(changed_rows.sum())
    changed_fraction = float(changed_samples / baseline.shape[0])
    passed = changed_fraction <= max_changed_fraction
    metrics.update({
        "changed_samples": changed_samples,
        "changed_fraction": changed_fraction,
    })
    findings = ()
    if changed_samples:
        findings = (
            Finding(
                code="MODEL_PREDICTIONS_CHANGED",
                message="Some candidate predictions differ from the baseline model.",
                severity="warning",
                details={
                    "changed_samples": changed_samples,
                    "changed_fraction": changed_fraction,
                    "max_changed_fraction": float(max_changed_fraction),
                },
            ),
        )

    return TestResult(
        name="model_comparison",
        status="passed" if passed else "failed",
        summary=(
            f"{changed_samples} of {baseline.shape[0]} predictions differed "
            f"({changed_fraction:.3%}); allowed fraction is {max_changed_fraction:.3%}."
        ),
        metrics=metrics,
        findings=findings,
    )
