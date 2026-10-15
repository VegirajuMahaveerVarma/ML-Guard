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
                isinstance(value, (float, np.floating, complex, np.complexfloating))
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
