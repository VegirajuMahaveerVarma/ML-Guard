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
    ref_var = reference.var(axis=0, ddof=1) if reference.shape[0] > 1 else np.zeros(reference.shape[1])
    cand_var = candidate.var(axis=0, ddof=1) if candidate.shape[0] > 1 else np.zeros(candidate.shape[1])
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
        summary=f"{int(shifted_indices.size)} feature(s) exceeded the shift threshold of {threshold:.3f}.",
        metrics={
            "threshold": float(threshold),
            "shifted_feature_count": int(shifted_indices.size),
            "feature_count": int(reference.shape[1]),
            "max_standardized_mean_difference": float(standardized.max(initial=0.0)),
            "standardized_mean_differences": [float(value) for value in standardized],
        },
        findings=findings,
    )
