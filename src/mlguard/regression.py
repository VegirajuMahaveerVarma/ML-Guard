"""Save, load, and replay MLGuard regression cases."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from mlguard._validation import as_array
from mlguard.checks import _equal_predictions, _predict, distribution_shift, prediction_stability
from mlguard.results import (
    RegressionCase,
    RegressionReplayReport,
    RegressionReplayResult,
    decode_json_safe,
)


def save_regression_cases(
    cases: Iterable[RegressionCase],
    path: str | Path,
) -> Path:
    """Save captured cases to a versioned, strict-JSON regression artifact."""
    normalized = tuple(cases)
    if any(not isinstance(case, RegressionCase) for case in normalized):
        raise TypeError("cases must contain only RegressionCase objects.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "kind": "mlguard-regression-cases",
        "case_count": len(normalized),
        "cases": [case.to_dict() for case in normalized],
    }
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def load_regression_cases(path: str | Path) -> tuple[RegressionCase, ...]:
    """Load and validate cases saved by save_regression_cases."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load regression cases from '{source}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Regression case file must contain a JSON object.")
    if payload.get("kind") != "mlguard-regression-cases":
        raise ValueError("File is not an MLGuard regression-case artifact.")
    if payload.get("schema_version") != "1.0":
        raise ValueError(
            f"Unsupported regression-case schema version: {payload.get('schema_version')!r}."
        )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("Regression case file must contain a cases array.")
    cases = tuple(RegressionCase.from_dict(item) for item in raw_cases)
    if payload.get("case_count") != len(cases):
        raise ValueError("Regression case count does not match the cases array.")
    return cases


def _restore_input(
    values: Any,
    *,
    layout: str,
    input_format: str,
    columns: tuple[Any, ...],
) -> Any:
    """Restore one sample or a dataset, optionally keeping pandas column labels."""
    decoded = decode_json_safe(values)
    batch_values = decoded if layout == "dataset" else [decoded]
    if input_format == "dataframe":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "This case preserves DataFrame columns. Install pandas to replay it."
            ) from exc
        return pd.DataFrame(batch_values, columns=list(columns))
    return np.asarray(batch_values)


def _invalid_predictions(predictions: np.ndarray) -> bool:
    if np.issubdtype(predictions.dtype, np.number):
        return not bool(np.isfinite(predictions).all())
    if predictions.dtype.kind == "O":
        for value in predictions.flat:
            if value is None:
                return True
            if isinstance(value, (float, np.floating, complex, np.complexfloating)):
                if not np.isfinite(value):
                    return True
    return False


def _expected_prediction_matches(
    actual: Any,
    expected: Any,
    *,
    comparison_mode: str,
    atol: float,
    rtol: float,
) -> bool:
    actual_array = np.asarray(actual).reshape(-1)
    expected_array = np.asarray(expected).reshape(-1)
    if actual_array.shape != expected_array.shape:
        return False
    use_numeric = comparison_mode == "regression" or (
        comparison_mode == "auto"
        and actual_array.dtype.kind in "iuf"
        and expected_array.dtype.kind in "iuf"
        and (actual_array.dtype.kind == "f" or expected_array.dtype.kind == "f")
    )
    if use_numeric:
        try:
            return bool(
                np.isclose(
                    actual_array.astype(float),
                    expected_array.astype(float),
                    atol=atol,
                    rtol=rtol,
                    equal_nan=False,
                ).all()
            )
        except (TypeError, ValueError):
            return False
    return bool(np.array_equal(actual_array, expected_array))


def _replay_one(model: Any | None, case: RegressionCase) -> RegressionReplayResult:
    details: dict[str, Any] = {"scenario": case.scenario, "sample_index": case.sample_index}
    try:
        if case.check_name == "distribution_shift":
            if case.reference_input is None:
                raise ValueError("Distribution-shift case is missing reference_input.")
            reference = _restore_input(
                case.reference_input,
                layout=case.reference_input_layout,
                input_format=case.reference_input_format,
                columns=case.reference_columns,
            )
            candidate = _restore_input(
                case.input,
                layout=case.input_layout,
                input_format=case.input_format,
                columns=case.columns,
            )
            result = distribution_shift(
                reference,
                candidate,
                threshold=float(case.parameters.get("threshold", 0.25)),
                variance_ratio_threshold=float(
                    case.parameters.get("variance_ratio_threshold", 2.0)
                ),
                shape_alpha=float(case.parameters.get("shape_alpha", 0.05)),
            )
            reproduced = result.status == "failed"
            details["current_check_status"] = result.status
            details["current_metrics"] = result.metrics
            return RegressionReplayResult(
                case_id=case.case_id,
                check_name=case.check_name,
                status="reproduced" if reproduced else "resolved",
                summary=(
                    "The distribution shift is still detected."
                    if reproduced
                    else "The saved reference/candidate inputs no longer trigger the shift check."
                ),
                details=details,
            )

        if model is None or not callable(getattr(model, "predict", None)):
            raise ValueError(
                f"A model exposing predict(X) is required to replay {case.check_name} cases."
            )

        test_input = _restore_input(
            case.input,
            layout=case.input_layout,
            input_format=case.input_format,
            columns=case.columns,
        )

        if case.check_name == "prediction_stability":
            result = prediction_stability(
                model,
                test_input,
                repeats=int(case.parameters.get("repeats", 3)),
            )
            reproduced = result.status == "failed"
            details["current_metrics"] = result.metrics
            return RegressionReplayResult(
                case_id=case.case_id,
                check_name=case.check_name,
                status="reproduced" if reproduced else "resolved",
                summary=(
                    "Repeated predictions are still unstable."
                    if reproduced
                    else "Repeated predictions are now stable for this saved input."
                ),
                details=details,
            )

        if case.check_name == "noise_robustness":
            if case.reference_input is None:
                raise ValueError("Noise regression case is missing its clean reference input.")
            reference_input = _restore_input(
                case.reference_input,
                layout=case.reference_input_layout,
                input_format=case.reference_input_format,
                columns=case.reference_columns,
            )
            baseline_prediction = _predict(model, reference_input)
            perturbed_prediction = _predict(model, test_input)
            same = _equal_predictions(baseline_prediction, perturbed_prediction)
            reproduced = not bool(same.all())
            details["current_predictions_agree"] = bool(same.all())
            return RegressionReplayResult(
                case_id=case.case_id,
                check_name=case.check_name,
                status="reproduced" if reproduced else "resolved",
                summary=(
                    "The noisy input still changes predictions."
                    if reproduced
                    else "The saved noisy input no longer changes predictions."
                ),
                details=details,
            )

        if case.check_name == "missing_value_robustness":
            try:
                predictions = as_array(model.predict(test_input), name="predictions")
            except Exception as exc:
                details["error_type"] = type(exc).__name__
                return RegressionReplayResult(
                    case_id=case.case_id,
                    check_name=case.check_name,
                    status="reproduced",
                    summary="The model still rejects or errors on the saved missing-value input.",
                    details=details,
                )
            reproduced = _invalid_predictions(predictions)
            details["invalid_predictions"] = reproduced
            return RegressionReplayResult(
                case_id=case.case_id,
                check_name=case.check_name,
                status="reproduced" if reproduced else "resolved",
                summary=(
                    (
                        "The model still produces invalid predictions for the saved missing-value input."
                    )
                    if reproduced
                    else (
                        "The model now returns valid predictions for the saved missing-value input."
                    )
                ),
                details=details,
            )

        if case.check_name == "model_comparison":
            if case.expected_prediction is None:
                raise ValueError(
                    "Model-comparison case is missing its expected baseline prediction."
                )
            prediction = _predict(model, test_input)
            if prediction.shape[0] != 1:
                raise ValueError("The model must return exactly one prediction for a saved sample.")
            matches = _expected_prediction_matches(
                prediction[0],
                case.expected_prediction,
                comparison_mode=str(case.parameters.get("comparison_mode", "auto")),
                atol=float(case.parameters.get("atol", 1e-8)),
                rtol=float(case.parameters.get("rtol", 1e-5)),
            )
            details["expected_prediction_matches"] = matches
            return RegressionReplayResult(
                case_id=case.case_id,
                check_name=case.check_name,
                status="resolved" if matches else "reproduced",
                summary=(
                    "The current model matches the stored baseline prediction."
                    if matches
                    else "The current model still differs from the stored baseline prediction."
                ),
                details=details,
            )

        raise ValueError(f"Unsupported regression check: {case.check_name!r}.")
    except Exception as exc:
        details["error_type"] = type(exc).__name__
        details["error"] = str(exc)
        return RegressionReplayResult(
            case_id=case.case_id,
            check_name=case.check_name,
            status="error",
            summary="The saved regression case could not be replayed.",
            details=details,
        )


def replay_regression_cases(
    model: Any | None,
    cases: Iterable[RegressionCase],
) -> RegressionReplayReport:
    """Replay saved cases and report which previously detected issues remain."""
    normalized = tuple(cases)
    if not normalized:
        raise ValueError("No regression cases were supplied.")
    if any(not isinstance(case, RegressionCase) for case in normalized):
        raise TypeError("cases must contain only RegressionCase objects.")
    if model is None and any(case.check_name != "distribution_shift" for case in normalized):
        raise ValueError("A model is required when replaying prediction-based regression cases.")
    results = tuple(_replay_one(model, case) for case in normalized)
    return RegressionReplayReport(results=results)
