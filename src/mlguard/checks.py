"""Initial built-in reliability checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlguard._validation import as_array, require_numeric, validate_same_sample_count
from mlguard.adapters.base import ModelAdapter
from mlguard.exceptions import InputValidationError
from mlguard.results import (
    MAX_FAILURE_CASES_PER_CHECK,
    Finding,
    RegressionCase,
    TestResult,
)


def _predict(model: ModelAdapter, X: Any) -> np.ndarray:
    try:
        return as_array(model.predict(X), name="predictions")
    except InputValidationError:
        raise
    except Exception as exc:
        raise RuntimeError("Model prediction failed.") from exc


def _expected_sample_count(X: Any) -> int | None:
    """Read a batch size without coercing framework-specific inputs to NumPy."""
    shape = getattr(X, "shape", None)
    if shape is not None:
        try:
            if len(shape) > 0:
                return int(shape[0])
        except (TypeError, IndexError):
            pass
    try:
        return len(X)
    except (TypeError, AttributeError):
        return None


def _equal_predictions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compare predictions row-wise, supporting numeric and object labels."""
    if left.shape != right.shape:
        raise InputValidationError(
            f"Prediction shapes differ: {left.shape!r} versus {right.shape!r}."
        )
    if left.ndim == 1:
        return np.asarray(left == right, dtype=bool)
    return np.asarray([np.array_equal(a, b) for a, b in zip(left, right, strict=True)])


def _sample_input_descriptor(
    X: Any, sample_index: int
) -> tuple[Any, str, tuple[Any, ...]]:
    """Capture one input row while retaining DataFrame columns when present."""
    columns = getattr(X, "columns", None)
    iloc = getattr(X, "iloc", None)
    if columns is not None and iloc is not None:
        return iloc[sample_index].tolist(), "dataframe", tuple(columns)
    row = np.asarray(X)[sample_index]
    return (row.tolist() if isinstance(row, np.ndarray) else row), "array", ()


def _sample_regression_case(
    check_name: str,
    scenario: str,
    input_source: Any,
    sample_index: int,
    *,
    reference_source: Any | None = None,
    expected_prediction: Any | None = None,
    observed_prediction: Any | None = None,
    parameters: dict[str, Any] | None = None,
    finding_code: str | None = None,
) -> RegressionCase | None:
    """Build a one-sample case, skipping unsupported custom objects gracefully."""
    try:
        values, input_format, columns = _sample_input_descriptor(input_source, sample_index)
        reference_values = None
        reference_format = "array"
        reference_columns: tuple[Any, ...] = ()
        if reference_source is not None:
            reference_values, reference_format, reference_columns = _sample_input_descriptor(
                reference_source, sample_index
            )
        return RegressionCase.create(
            check_name=check_name,
            scenario=scenario,
            input=values,
            input_layout="sample",
            input_format=input_format,
            columns=columns,
            reference_input=reference_values,
            reference_input_layout="sample",
            reference_input_format=reference_format,
            reference_columns=reference_columns,
            sample_index=int(sample_index),
            expected_prediction=expected_prediction,
            observed_prediction=observed_prediction,
            parameters=dict(parameters or {}),
            finding_code=finding_code,
        )
    except (TypeError, ValueError):
        # Checking must remain usable for model inputs that cannot be stored as JSON.
        return None


def _dataset_regression_case(
    check_name: str,
    scenario: str,
    input_source: Any,
    *,
    expected_prediction: Any | None = None,
    observed_prediction: Any | None = None,
    parameters: dict[str, Any] | None = None,
    finding_code: str | None = None,
) -> RegressionCase | None:
    """Capture a dataset-level case for failures that depend on batch behavior."""
    try:
        columns = getattr(input_source, "columns", None)
        if columns is not None:
            to_numpy = getattr(input_source, "to_numpy", None)
            values = to_numpy().tolist() if callable(to_numpy) else np.asarray(input_source).tolist()
            input_format = "dataframe"
            column_names = tuple(columns)
        else:
            values = np.asarray(input_source).tolist()
            input_format = "array"
            column_names = ()
        return RegressionCase.create(
            check_name=check_name,
            scenario=scenario,
            input=values,
            input_layout="dataset",
            input_format=input_format,
            columns=column_names,
            expected_prediction=expected_prediction,
            observed_prediction=observed_prediction,
            parameters=dict(parameters or {}),
            finding_code=finding_code,
        )
    except (TypeError, ValueError):
        # Large/custom or non-JSON-compatible inputs must not break a check.
        return None


def _preserve_input_container(original_input: Any, values: np.ndarray) -> Any:
    """Keep DataFrame column names and index when perturbing numeric features."""
    columns = getattr(original_input, "columns", None)
    index = getattr(original_input, "index", None)
    if columns is not None and index is not None:
        try:
            return type(original_input)(values, columns=columns, index=index)
        except (TypeError, ValueError):
            # Fall back to NumPy for DataFrame-like containers with other
            # constructor conventions. Plain arrays never enter this branch.
            pass
    return values


def prediction_stability(
    model: ModelAdapter,
    X: Any,
    *,
    repeats: int = 3,
) -> TestResult:
    """Check prediction agreement across repeated calls on identical input.

    At least two calls are required. The result marks a sample as changed if
    any repeat differs from the first prediction for that sample.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise InputValidationError("repeats must be an integer of at least 2.")

    expected_samples = _expected_sample_count(X)
    predictions = []
    for _ in range(repeats):
        prediction = _predict(model, X)
        if expected_samples is not None and prediction.shape[0] != expected_samples:
            raise InputValidationError(
                "The model returned "
                f"{prediction.shape[0]} predictions for {expected_samples} input samples."
            )
        predictions.append(prediction)

    first = predictions[0]
    stable = np.ones(first.shape[0], dtype=bool)
    for prediction in predictions[1:]:
        stable &= _equal_predictions(first, prediction)
    agreement = float(stable.mean())
    changed_samples = int((~stable).sum())
    findings = () if agreement == 1.0 else (
        Finding(
            code="NON_DETERMINISTIC_PREDICTIONS",
            message="Repeated predictions differed for some samples.",
            details={"changed_samples": changed_samples, "repeats": repeats},
        ),
    )
    failure_cases: list[RegressionCase] = []
    for sample_index in np.flatnonzero(~stable)[:MAX_FAILURE_CASES_PER_CHECK]:
        observed_prediction = first[sample_index]
        for repeated_prediction in predictions[1:]:
            if not _equal_predictions(first, repeated_prediction)[sample_index]:
                observed_prediction = repeated_prediction[sample_index]
                break
        case = _sample_regression_case(
            "prediction_stability",
            "repeated_prediction_instability",
            X,
            int(sample_index),
            expected_prediction=first[sample_index],
            observed_prediction=observed_prediction,
            parameters={"repeats": repeats},
            finding_code="NON_DETERMINISTIC_PREDICTIONS",
        )
        if case is not None:
            failure_cases.append(case)
    return TestResult(
        name="prediction_stability",
        status="passed" if agreement == 1.0 else "failed",
        summary=(
            f"Prediction agreement across {repeats} calls: {agreement:.3f}."
        ),
        metrics={
            "prediction_agreement": agreement,
            "changed_samples": changed_samples,
            "repeats": repeats,
            "failure_cases_captured": len(failure_cases),
            "failure_cases_truncated": changed_samples > len(failure_cases),
        },
        findings=findings,
        failure_cases=tuple(failure_cases),
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
    validate_same_sample_count(numeric, baseline)
    rng = np.random.default_rng(random_state)
    perturbed = numeric + rng.normal(0.0, noise_scale, size=numeric.shape)
    changed_input = _preserve_input_container(X, perturbed)
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
    changed_indices = np.flatnonzero(~same)
    failure_cases: list[RegressionCase] = []
    for sample_index in changed_indices[:MAX_FAILURE_CASES_PER_CHECK]:
        case = _sample_regression_case(
            "noise_robustness",
            "gaussian_noise_changed_prediction",
            changed_input,
            int(sample_index),
            reference_source=X,
            expected_prediction=baseline[sample_index],
            observed_prediction=noisy_predictions[sample_index],
            parameters={"noise_scale": float(noise_scale), "random_state": int(random_state)},
            finding_code="PREDICTIONS_CHANGED_UNDER_NOISE",
        )
        if case is not None:
            failure_cases.append(case)
    return TestResult(
        name="noise_robustness",
        status="passed" if agreement == 1.0 else "failed",
        summary=f"Prediction agreement after noise: {agreement:.3f}.",
        metrics={
            "prediction_agreement": agreement,
            "changed_samples": int((~same).sum()),
            "noise_scale": float(noise_scale),
            "random_state": int(random_state),
            "failure_cases_captured": len(failure_cases),
            "failure_cases_truncated": int(changed_indices.size) > len(failure_cases),
        },
        findings=findings,
        failure_cases=tuple(failure_cases),
    )


def distribution_shift(
    reference_X: Any,
    candidate_X: Any,
    *,
    threshold: float = 0.25,
    variance_ratio_threshold: float = 2.0,
    shape_alpha: float = 0.05,
    _shape_correction_feature_count: int | None = None,
) -> TestResult:
    """Screen numeric feature distributions for mean, variance, and shape changes.

    Mean changes are measured with standardized mean differences; spread changes
    use sample-variance ratios. Shape changes are screened using the maximum
    two-sample empirical-CDF distance. Its conservative DKW-style threshold
    depends on both sample sizes and applies a Bonferroni correction across
    features to bound the family-wise false-positive probability by shape_alpha.

    The shape screen is descriptive thresholding, not a returned p-value. It can
    miss subtle shifts when samples are small and should be calibrated to the
    application rather than treated as a universal production guarantee.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise InputValidationError("threshold must be a finite, non-negative number.")
    if not np.isfinite(variance_ratio_threshold) or variance_ratio_threshold <= 1:
        raise InputValidationError(
            "variance_ratio_threshold must be a finite number greater than 1."
        )
    if not np.isfinite(shape_alpha) or not 0 < shape_alpha < 1:
        raise InputValidationError("shape_alpha must be a finite number strictly between 0 and 1.")

    reference = require_numeric(as_array(reference_X, name="reference_X"), name="reference_X")
    candidate = require_numeric(as_array(candidate_X, name="candidate_X"), name="candidate_X")
    if reference.ndim != 2 or candidate.ndim != 2:
        raise InputValidationError("reference_X and candidate_X must be 2D feature matrices.")
    if reference.shape[1] != candidate.shape[1]:
        raise InputValidationError(
            "reference_X and candidate_X must have the same number of features."
        )
    if reference.shape[1] == 0:
        raise InputValidationError("reference_X and candidate_X must contain at least one feature.")

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
    if not (
        np.isfinite(ref_mean).all()
        and np.isfinite(cand_mean).all()
        and np.isfinite(ref_var).all()
        and np.isfinite(cand_var).all()
    ):
        raise InputValidationError(
            "Feature magnitudes are too large to compare safely; rescale the inputs."
        )

    pooled_scale = np.sqrt(ref_var / 2.0 + cand_var / 2.0)
    mean_differences = np.abs(cand_mean - ref_mean)
    standardized = np.divide(
        mean_differences,
        pooled_scale,
        out=np.where(mean_differences == 0, 0.0, np.inf),
        where=pooled_scale > 0,
    )
    mean_shifted = standardized > threshold

    variance_check_available = reference.shape[0] > 1 and candidate.shape[0] > 1
    variance_ratios: list[float | None] = [None] * int(reference.shape[1])
    variance_shifted = np.zeros(reference.shape[1], dtype=bool)
    if variance_check_available:
        for index, (ref_value, candidate_value) in enumerate(
            zip(ref_var, cand_var, strict=True)
        ):
            if ref_value == 0.0 and candidate_value == 0.0:
                ratio = 1.0
            elif ref_value == 0.0 or candidate_value == 0.0:
                # None denotes an unbounded ratio; the separate flag preserves
                # the actionable result without putting Infinity in JSON output.
                ratio = None
                variance_shifted[index] = True
            else:
                raw_ratio = float(max(ref_value, candidate_value) / min(ref_value, candidate_value))
                if np.isfinite(raw_ratio):
                    ratio = raw_ratio
                    variance_shifted[index] = ratio > variance_ratio_threshold
                else:
                    ratio = None
                    variance_shifted[index] = True
            variance_ratios[index] = ratio

    feature_count = int(reference.shape[1])
    shape_correction_feature_count = feature_count
    if _shape_correction_feature_count is not None:
        if (
            isinstance(_shape_correction_feature_count, bool)
            or not isinstance(_shape_correction_feature_count, (int, np.integer))
            or _shape_correction_feature_count < feature_count
        ):
            raise InputValidationError(
                "_shape_correction_feature_count must be an integer at least as large "
                "as the number of supplied features."
            )
        shape_correction_feature_count = int(_shape_correction_feature_count)
    reference_count = int(reference.shape[0])
    candidate_count = int(candidate.shape[0])
    per_feature_alpha = shape_alpha / shape_correction_feature_count
    shape_distance_threshold = float(
        np.sqrt(
            0.5
            * np.log(2.0 / per_feature_alpha)
            * (1.0 / reference_count + 1.0 / candidate_count)
        )
    )
    shape_distances = np.zeros(feature_count, dtype=float)
    for index in range(feature_count):
        ref_values = np.sort(reference[:, index])
        cand_values = np.sort(candidate[:, index])
        pooled_values = np.unique(np.concatenate((ref_values, cand_values)))
        ref_left = np.searchsorted(ref_values, pooled_values, side="left") / reference_count
        cand_left = np.searchsorted(cand_values, pooled_values, side="left") / candidate_count
        ref_right = np.searchsorted(ref_values, pooled_values, side="right") / reference_count
        cand_right = np.searchsorted(cand_values, pooled_values, side="right") / candidate_count
        left_distance = np.abs(ref_left - cand_left)
        right_distance = np.abs(ref_right - cand_right)
        shape_distances[index] = float(
            max(left_distance.max(initial=0.0), right_distance.max(initial=0.0))
        )
    shape_shifted = shape_distances > shape_distance_threshold

    shifted_indices = np.flatnonzero(mean_shifted | variance_shifted | shape_shifted)
    findings = tuple(
        Finding(
            code="FEATURE_DISTRIBUTION_SHIFT",
            message=f"Feature {int(index)} exceeds a configured distribution-shift threshold.",
            details={
                "feature_index": int(index),
                "shift_types": [
                    shift_type
                    for shift_type, is_shifted in (
                        ("mean", bool(mean_shifted[index])),
                        ("variance", bool(variance_shifted[index])),
                        ("shape", bool(shape_shifted[index])),
                    )
                    if is_shifted
                ],
                "standardized_mean_difference": (
                    float(standardized[index]) if np.isfinite(standardized[index]) else None
                ),
                "variance_ratio": variance_ratios[index],
                "empirical_cdf_distance": float(shape_distances[index]),
                "shape_distance_threshold": shape_distance_threshold,
                "shape_alpha": float(shape_alpha),
                "threshold": float(threshold),
                "variance_ratio_threshold": float(variance_ratio_threshold),
            },
        )
        for index in shifted_indices
    )
    finite_standardized = standardized[np.isfinite(standardized)]
    max_standardized = (
        float(finite_standardized.max(initial=0.0))
        if finite_standardized.size == standardized.size
        else None
    )
    variance_shift_count = int(variance_shifted.sum())
    mean_shift_count = int(mean_shifted.sum())
    shape_shift_count = int(shape_shifted.sum())
    shifted_count = int(shifted_indices.size)
    failure_cases = tuple(
        RegressionCase.create(
            check_name="distribution_shift",
            scenario="feature_distribution_shift",
            input=candidate[:, [int(index)]].tolist(),
            input_layout="dataset",
            input_format="array",
            reference_input=reference[:, [int(index)]].tolist(),
            reference_input_layout="dataset",
            reference_input_format="array",
            parameters={
                "threshold": float(threshold),
                "variance_ratio_threshold": float(variance_ratio_threshold),
                "shape_alpha": float(shape_alpha),
                "shape_correction_feature_count": shape_correction_feature_count,
                "feature_index": int(index),
            },
            observed_prediction={
                "shift_types": [
                    shift_type
                    for shift_type, is_shifted in (
                        ("mean", bool(mean_shifted[index])),
                        ("variance", bool(variance_shifted[index])),
                        ("shape", bool(shape_shifted[index])),
                    )
                    if is_shifted
                ]
            },
            finding_code="FEATURE_DISTRIBUTION_SHIFT",
        )
        for index in shifted_indices[:MAX_FAILURE_CASES_PER_CHECK]
    )
    return TestResult(
        name="distribution_shift",
        status="failed" if shifted_count else "passed",
        summary=(
            f"{shifted_count} feature(s) flagged: {mean_shift_count} mean shift(s), "
            f"{variance_shift_count} variance shift(s), {shape_shift_count} shape shift(s)."
        ),
        metrics={
            "threshold": float(threshold),
            "variance_ratio_threshold": float(variance_ratio_threshold),
            "shape_alpha": float(shape_alpha),
            "shape_distance_threshold": shape_distance_threshold,
            "shifted_feature_count": shifted_count,
            "mean_shifted_feature_count": mean_shift_count,
            "variance_shifted_feature_count": variance_shift_count,
            "shape_shifted_feature_count": shape_shift_count,
            "variance_check_available": variance_check_available,
            "feature_count": feature_count,
            "shape_correction_feature_count": shape_correction_feature_count,
            "max_standardized_mean_difference": max_standardized,
            "standardized_mean_differences": [
                float(value) if np.isfinite(value) else None for value in standardized
            ],
            "variance_ratios": variance_ratios,
            "empirical_cdf_distances": [float(value) for value in shape_distances],
            "failure_cases_captured": len(failure_cases),
            "failure_cases_truncated": shifted_count > len(failure_cases),
        },
        findings=findings,
        failure_cases=failure_cases,
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
    missing_flat_indices = rng.choice(total_cells, size=missing_cells, replace=False)
    missing_mask = np.zeros(total_cells, dtype=bool)
    missing_mask[missing_flat_indices] = True
    masked = original.copy()
    masked.flat[missing_flat_indices] = np.nan
    masked_input: Any = masked
    missing_mask_2d = missing_mask.reshape(original.shape)
    missing_rows = np.flatnonzero(missing_mask_2d.any(axis=1))

    metrics = {
        "requested_missing_fraction": float(missing_fraction),
        "actual_missing_fraction": float(missing_cells / total_cells),
        "missing_cells": missing_cells,
        "total_cells": total_cells,
        "random_state": int(random_state),
    }
    try:
        masked_input = _preserve_input_container(X, masked)
        predictions = _predict(model, masked_input)
        validate_same_sample_count(original, predictions)
    except Exception as exc:
        underlying = exc.__cause__ if exc.__cause__ is not None else exc
        error_type = type(underlying).__name__
        failure_cases: list[RegressionCase] = []
        for sample_index in missing_rows[:MAX_FAILURE_CASES_PER_CHECK]:
            case = _sample_regression_case(
                "missing_value_robustness",
                "model_rejected_missing_values",
                masked_input,
                int(sample_index),
                reference_source=X,
                expected_prediction=baseline[sample_index],
                observed_prediction={"error_type": error_type},
                parameters={
                    "missing_fraction": float(missing_fraction),
                    "random_state": int(random_state),
                    "missing_feature_indices": np.flatnonzero(
                        missing_mask_2d[sample_index]
                    ).tolist(),
                },
                finding_code="MODEL_REJECTED_MISSING_VALUES",
            )
            if case is not None:
                failure_cases.append(case)
        return TestResult(
            name="missing_value_robustness",
            status="failed",
            summary=(
                "The model could not return predictions for input containing NaN values."
            ),
            metrics={
                **metrics,
                "error_type": error_type,
                "failure_cases_captured": len(failure_cases),
                "failure_cases_truncated": len(missing_rows) > len(failure_cases),
            },
            findings=(
                Finding(
                    code="MODEL_REJECTED_MISSING_VALUES",
                    message=(
                        "The model raised an exception for input containing missing features."
                    ),
                    details={"error_type": error_type},
                ),
            ),
            failure_cases=tuple(failure_cases),
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
        def row_has_invalid(value: Any) -> bool:
            row = np.asarray(value)
            if np.issubdtype(row.dtype, np.number):
                return not bool(np.isfinite(row).all())
            if row.dtype.kind == "O":
                for item in row.flat:
                    if item is None:
                        return True
                    if isinstance(item, (float, np.floating, complex, np.complexfloating)):
                        if not np.isfinite(item):
                            return True
            return False

        invalid_rows = [
            int(index)
            for index in range(predictions.shape[0])
            if row_has_invalid(predictions[index])
        ]
        failure_cases: list[RegressionCase] = []
        for sample_index in invalid_rows[:MAX_FAILURE_CASES_PER_CHECK]:
            case = _sample_regression_case(
                "missing_value_robustness",
                "invalid_prediction_with_missing_values",
                masked_input,
                sample_index,
                reference_source=X,
                expected_prediction=baseline[sample_index],
                observed_prediction=predictions[sample_index],
                parameters={
                    "missing_fraction": float(missing_fraction),
                    "random_state": int(random_state),
                    "missing_feature_indices": np.flatnonzero(
                        missing_mask_2d[sample_index]
                    ).tolist(),
                },
                finding_code="INVALID_PREDICTIONS_WITH_MISSING_INPUT",
            )
            if case is not None:
                failure_cases.append(case)
        metrics.update({
            "failure_cases_captured": len(failure_cases),
            "failure_cases_truncated": len(invalid_rows) > len(failure_cases),
        })
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
            failure_cases=tuple(failure_cases),
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
    except Exception as exc:
        error_type = type(exc).__name__
        return TestResult(
            name="model_comparison",
            status="error",
            summary="Could not compare models because the baseline prediction call failed.",
            metrics={"error_type": error_type, "failed_model": "baseline"},
            findings=(
                Finding(
                    code="MODEL_COMPARISON_PREDICTION_ERROR",
                    message="The baseline model failed to produce predictions.",
                    severity="error",
                    details={"error_type": error_type, "failed_model": "baseline"},
                ),
            ),
        )

    try:
        candidate = _predict(candidate_model, X)
    except Exception as exc:
        error_type = type(exc).__name__
        case = _dataset_regression_case(
            "model_comparison",
            "candidate_prediction_error",
            X,
            expected_prediction={"shape": list(baseline.shape)},
            observed_prediction={"error_type": error_type},
            parameters={
                "baseline_prediction_shape": list(baseline.shape),
                "comparison_mode": comparison_mode,
                "atol": float(atol),
                "rtol": float(rtol),
                "max_changed_fraction": float(max_changed_fraction),
            },
            finding_code="MODEL_COMPARISON_PREDICTION_ERROR",
        )
        failure_cases = (case,) if case is not None else ()
        return TestResult(
            name="model_comparison",
            status="error",
            summary="Could not compare models because the candidate prediction call failed.",
            metrics={
                "error_type": error_type,
                "failed_model": "candidate",
                "failure_cases_captured": len(failure_cases),
            },
            findings=(
                Finding(
                    code="MODEL_COMPARISON_PREDICTION_ERROR",
                    message="The candidate model failed to produce predictions.",
                    severity="error",
                    details={"error_type": error_type, "failed_model": "candidate"},
                ),
            ),
            failure_cases=failure_cases,
        )

    expected_samples = _expected_sample_count(X)
    if expected_samples is not None and (
        baseline.shape[0] != expected_samples or candidate.shape[0] != expected_samples
    ):
        metrics = {
            "input_sample_count": expected_samples,
            "baseline_sample_count": int(baseline.shape[0]),
            "candidate_sample_count": int(candidate.shape[0]),
        }
        failure_cases: tuple[RegressionCase, ...] = ()
        if baseline.shape[0] == expected_samples and candidate.shape[0] != expected_samples:
            case = _dataset_regression_case(
                "model_comparison",
                "candidate_prediction_shape_changed",
                X,
                expected_prediction={"shape": list(baseline.shape)},
                observed_prediction={"shape": list(candidate.shape)},
                parameters={
                    "comparison_mode": comparison_mode,
                    "atol": float(atol),
                    "rtol": float(rtol),
                    "max_changed_fraction": float(max_changed_fraction),
                },
                finding_code="MODEL_PREDICTION_COUNT_MISMATCH",
            )
            failure_cases = (case,) if case is not None else ()
        return TestResult(
            name="model_comparison",
            status="error",
            summary="One or both models returned a prediction count that does not match the input.",
            metrics={**metrics, "failure_cases_captured": len(failure_cases)},
            findings=(
                Finding(
                    code="MODEL_PREDICTION_COUNT_MISMATCH",
                    message="Each model must return one prediction per input sample.",
                    severity="error",
                    details=metrics,
                ),
            ),
            failure_cases=failure_cases,
        )

    if baseline.shape[0] != candidate.shape[0] or baseline.shape != candidate.shape:
        failure_code = (
            "PREDICTION_SAMPLE_COUNT_CHANGED"
            if baseline.shape[0] != candidate.shape[0]
            else "PREDICTION_SHAPE_CHANGED"
        )
        case = _dataset_regression_case(
            "model_comparison",
            "candidate_prediction_shape_changed",
            X,
            expected_prediction={"shape": list(baseline.shape)},
            observed_prediction={"shape": list(candidate.shape)},
            parameters={
                "comparison_mode": comparison_mode,
                "atol": float(atol),
                "rtol": float(rtol),
                "max_changed_fraction": float(max_changed_fraction),
                "baseline_prediction_shape": list(baseline.shape),
            },
            finding_code=failure_code,
        )
        failure_cases = (case,) if case is not None else ()
        return TestResult(
            name="model_comparison",
            status="failed",
            summary=(
                "Baseline and candidate models returned different sample counts."
                if baseline.shape[0] != candidate.shape[0]
                else "Baseline and candidate prediction shapes differ."
            ),
            metrics={
                "baseline_prediction_shape": list(baseline.shape),
                "candidate_prediction_shape": list(candidate.shape),
                "failure_cases_captured": len(failure_cases),
            },
            findings=(
                Finding(
                    code=failure_code,
                    message="The candidate model changed its prediction count or shape.",
                    severity="error",
                    details={
                        "baseline_prediction_shape": list(baseline.shape),
                        "candidate_prediction_shape": list(candidate.shape),
                    },
                ),
            ),
            failure_cases=failure_cases,
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
            failure_cases: list[RegressionCase] = []
            baseline_finite_rows = baseline_finite.reshape(baseline_finite.shape[0], -1).all(axis=1)
            candidate_finite_rows = candidate_finite.reshape(candidate_finite.shape[0], -1).all(axis=1)
            candidate_invalid_rows = np.flatnonzero(baseline_finite_rows & ~candidate_finite_rows)
            for sample_index in candidate_invalid_rows[:MAX_FAILURE_CASES_PER_CHECK]:
                case = _sample_regression_case(
                    "model_comparison",
                    "candidate_non_finite_prediction",
                    X,
                    int(sample_index),
                    expected_prediction=baseline[sample_index],
                    observed_prediction=candidate[sample_index],
                    parameters={
                        "comparison_mode": "regression",
                        "atol": float(atol),
                        "rtol": float(rtol),
                        "max_changed_fraction": float(max_changed_fraction),
                    },
                    finding_code="NON_FINITE_MODEL_PREDICTIONS",
                )
                if case is not None:
                    failure_cases.append(case)
            return TestResult(
                name="model_comparison",
                status="failed",
                summary="One or both models returned non-finite numeric predictions.",
                metrics={
                    **metrics,
                    "invalid_baseline_prediction_count": invalid_baseline,
                    "invalid_candidate_prediction_count": invalid_candidate,
                    "failure_cases_captured": len(failure_cases),
                    "failure_cases_truncated": len(candidate_invalid_rows) > len(failure_cases),
                },
                findings=(
                    Finding(
                        code="NON_FINITE_MODEL_PREDICTIONS",
                        message=(
                            "Regression comparison requires finite predictions from "
                            "both models."
                        ),
                        severity="error",
                        details={
                            "invalid_baseline_prediction_count": invalid_baseline,
                            "invalid_candidate_prediction_count": invalid_candidate,
                        },
                    ),
                ),
                failure_cases=tuple(failure_cases),
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

    failure_cases: list[RegressionCase] = []
    if not passed:
        for sample_index in np.flatnonzero(changed_rows)[:MAX_FAILURE_CASES_PER_CHECK]:
            case = _sample_regression_case(
                "model_comparison",
                "candidate_prediction_regression",
                X,
                int(sample_index),
                expected_prediction=baseline[sample_index],
                observed_prediction=candidate[sample_index],
                parameters={
                    "comparison_mode": effective_mode,
                    "atol": float(atol),
                    "rtol": float(rtol),
                    "max_changed_fraction": float(max_changed_fraction),
                },
                finding_code="MODEL_PREDICTIONS_CHANGED",
            )
            if case is not None:
                failure_cases.append(case)

    return TestResult(
        name="model_comparison",
        status="passed" if passed else "failed",
        summary=(
            f"{changed_samples} of {baseline.shape[0]} predictions differed "
            f"({changed_fraction:.3%}); allowed fraction is {max_changed_fraction:.3%}."
        ),
        metrics={
            **metrics,
            "failure_cases_captured": len(failure_cases),
            "failure_cases_truncated": (not passed) and changed_samples > len(failure_cases),
        },
        findings=findings,
        failure_cases=tuple(failure_cases),
    )
