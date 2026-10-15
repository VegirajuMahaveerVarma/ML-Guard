"""JSON-friendly result models for MLGuard checks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np

Status = Literal["passed", "failed", "error", "skipped"]
MAX_FAILURE_CASES_PER_CHECK = 100


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON-compatible data."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        label = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        return {"__mlguard_float__": label}
    if isinstance(value, complex):
        return {"__mlguard_complex__": [_json_safe(value.real), _json_safe(value.imag)]}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Value of type {type(value).__name__} cannot be stored in a regression case. "
        "Convert it to a JSON-compatible value first."
    )


def decode_json_safe(value: Any) -> Any:
    """Restore non-finite floating-point values encoded in regression-case JSON."""
    if isinstance(value, list):
        return [decode_json_safe(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__mlguard_float__"}:
            label = value["__mlguard_float__"]
            if label == "nan":
                return float("nan")
            if label == "inf":
                return float("inf")
            if label == "-inf":
                return float("-inf")
            raise ValueError(f"Unknown encoded floating-point value: {label!r}.")
        if set(value) == {"__mlguard_complex__"}:
            parts = value["__mlguard_complex__"]
            if not isinstance(parts, list) or len(parts) != 2:
                raise ValueError("Malformed encoded complex number.")
            return complex(decode_json_safe(parts[0]), decode_json_safe(parts[1]))
        return {key: decode_json_safe(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class RegressionCase:
    """A compact, serializable input that reproduces a detected reliability issue."""

    case_id: str
    check_name: str
    scenario: str
    input: Any
    input_layout: Literal["sample", "dataset"] = "sample"
    input_format: Literal["array", "dataframe"] = "array"
    columns: tuple[Any, ...] = ()
    reference_input: Any | None = None
    reference_input_layout: Literal["sample", "dataset"] = "sample"
    reference_input_format: Literal["array", "dataframe"] = "array"
    reference_columns: tuple[Any, ...] = ()
    sample_index: int | None = None
    expected_prediction: Any | None = None
    observed_prediction: Any | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    finding_code: str | None = None

    @classmethod
    def create(cls, **kwargs: Any) -> RegressionCase:
        """Create a case with a deterministic ID derived from its content."""
        normalized = _json_safe(kwargs)
        canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
        case_id = "rg-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return cls(case_id=case_id, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return strict JSON-compatible case data, including safe encodings for NaN."""
        return {
            "case_id": self.case_id,
            "check_name": self.check_name,
            "scenario": self.scenario,
            "input": _json_safe(self.input),
            "input_layout": self.input_layout,
            "input_format": self.input_format,
            "columns": _json_safe(self.columns),
            "reference_input": _json_safe(self.reference_input),
            "reference_input_layout": self.reference_input_layout,
            "reference_input_format": self.reference_input_format,
            "reference_columns": _json_safe(self.reference_columns),
            "sample_index": self.sample_index,
            "expected_prediction": _json_safe(self.expected_prediction),
            "observed_prediction": _json_safe(self.observed_prediction),
            "parameters": _json_safe(self.parameters),
            "finding_code": self.finding_code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegressionCase:
        """Validate and construct a regression case loaded from JSON."""
        if not isinstance(data, dict):
            raise ValueError("Each regression case must be a JSON object.")
        required = ("case_id", "check_name", "scenario", "input")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Regression case is missing required field(s): {', '.join(missing)}.")
        allowed_checks = {
            "prediction_stability",
            "noise_robustness",
            "distribution_shift",
            "missing_value_robustness",
            "model_comparison",
        }
        if data["check_name"] not in allowed_checks:
            raise ValueError(f"Unsupported regression check: {data['check_name']!r}.")
        if data.get("input_layout", "sample") not in {"sample", "dataset"}:
            raise ValueError("input_layout must be 'sample' or 'dataset'.")
        if data.get("reference_input_layout", "sample") not in {"sample", "dataset"}:
            raise ValueError("reference_input_layout must be 'sample' or 'dataset'.")
        if data.get("input_format", "array") not in {"array", "dataframe"}:
            raise ValueError("input_format must be 'array' or 'dataframe'.")
        if data.get("reference_input_format", "array") not in {"array", "dataframe"}:
            raise ValueError("reference_input_format must be 'array' or 'dataframe'.")
        return cls(
            case_id=str(data["case_id"]),
            check_name=str(data["check_name"]),
            scenario=str(data["scenario"]),
            input=decode_json_safe(data["input"]),
            input_layout=data.get("input_layout", "sample"),
            input_format=data.get("input_format", "array"),
            columns=tuple(decode_json_safe(data.get("columns", []))),
            reference_input=decode_json_safe(data.get("reference_input")),
            reference_input_layout=data.get("reference_input_layout", "sample"),
            reference_input_format=data.get("reference_input_format", "array"),
            reference_columns=tuple(decode_json_safe(data.get("reference_columns", []))),
            sample_index=data.get("sample_index"),
            expected_prediction=decode_json_safe(data.get("expected_prediction")),
            observed_prediction=decode_json_safe(data.get("observed_prediction")),
            parameters=decode_json_safe(data.get("parameters", {})),
            finding_code=data.get("finding_code"),
        )


@dataclass(frozen=True)
class RegressionReplayResult:
    """Replay outcome for one saved case."""

    case_id: str
    check_name: str
    status: Literal["resolved", "reproduced", "error"]
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible replay result."""
        return {
            "case_id": self.case_id,
            "check_name": self.check_name,
            "status": self.status,
            "summary": self.summary,
            "details": _json_safe(self.details),
        }


@dataclass(frozen=True)
class RegressionReplayReport:
    """Aggregate replay outcome for a saved set of regression cases."""

    results: tuple[RegressionReplayResult, ...]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def reproduced_count(self) -> int:
        """Number of cases whose original reliability issue is still present."""
        return sum(result.status == "reproduced" for result in self.results)

    @property
    def error_count(self) -> int:
        """Number of cases that could not be replayed successfully."""
        return sum(result.status == "error" for result in self.results)

    @property
    def passed(self) -> bool:
        """Whether every saved issue has been resolved and replayed without errors."""
        return bool(self.results) and all(result.status == "resolved" for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible regression replay report."""
        return {
            "schema_version": "1.0",
            "created_at": self.created_at,
            "passed": self.passed,
            "case_count": len(self.results),
            "reproduced_count": self.reproduced_count,
            "error_count": self.error_count,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class Finding:
    """A single actionable observation from a check."""

    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TestResult:
    """Outcome of one model reliability check."""

    # Prevent pytest from treating this public result model as a test class.
    __test__ = False

    name: str
    status: Status
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: tuple[Finding, ...] = ()
    duration_seconds: float = 0.0
    failure_cases: tuple[RegressionCase, ...] = ()

    def to_dict(self, *, include_failure_cases: bool = False) -> dict[str, Any]:
        """Return JSON-compatible check output without raw input rows by default."""
        payload = asdict(self)
        payload.pop("failure_cases", None)
        payload["findings"] = [asdict(item) for item in self.findings]
        payload["failure_case_count"] = len(self.failure_cases)
        if include_failure_cases:
            payload["failure_cases"] = [case.to_dict() for case in self.failure_cases]
        return payload


@dataclass(frozen=True)
class EvaluationReport:
    """Dataset-level performance metrics for a model evaluation."""

    task: Literal["classification", "regression"]
    sample_count: int
    metrics: dict[str, Any]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evaluation report."""
        return {
            "schema_version": "1.0",
            "created_at": self.created_at,
            "task": self.task,
            "sample_count": self.sample_count,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SuiteReport:
    """Report for a collection of reliability checks."""

    results: tuple[TestResult, ...]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether all executed checks passed."""
        return bool(self.results) and all(r.status == "passed" for r in self.results)

    @property
    def failure_cases(self) -> tuple[RegressionCase, ...]:
        """All captured cases, in check order."""
        return tuple(case for result in self.results for case in result.failure_cases)

    def to_dict(self, *, include_failure_cases: bool = False) -> dict[str, Any]:
        """Return a report without raw input rows unless explicitly requested."""
        payload = {
            "schema_version": "1.0",
            "created_at": self.created_at,
            "passed": self.passed,
            "metadata": self.metadata,
            "failure_case_count": len(self.failure_cases),
            "results": [result.to_dict() for result in self.results],
        }
        if include_failure_cases:
            payload["failure_cases"] = [case.to_dict() for case in self.failure_cases]
        return payload


@dataclass(frozen=True)
class MetricGateResult:
    """Outcome of one configured minimum or maximum performance threshold."""

    metric: str
    operator: Literal[">=", "<="]
    threshold: float
    actual: float | None
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible metric threshold result."""
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    """Combined report for reliability checks and dataset performance metrics.

    Reliability results have explicit pass/fail criteria. Evaluation metrics do
    not have a universal pass threshold, so this report intentionally exposes
    reliability_passed rather than a misleading overall quality status.
    """

    reliability: SuiteReport
    evaluation: EvaluationReport
    metric_gates: tuple[MetricGateResult, ...] = ()
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failure_cases(self) -> tuple[RegressionCase, ...]:
        """Regression cases captured by the reliability portion of validation."""
        return self.reliability.failure_cases

    @property
    def reliability_passed(self) -> bool:
        """Whether every selected reliability check passed."""
        return self.reliability.passed

    @property
    def performance_passed(self) -> bool | None:
        """Whether configured performance gates passed, or None if none were set."""
        if not self.metric_gates:
            return None
        return all(result.passed for result in self.metric_gates)

    @property
    def release_passed(self) -> bool | None:
        """Release status when performance gates are configured; otherwise None."""
        performance_passed = self.performance_passed
        if performance_passed is None:
            return None
        return self.reliability_passed and performance_passed

    def to_dict(self, *, include_failure_cases: bool = False) -> dict[str, Any]:
        """Return the combined report; raw input cases are opt-in."""
        payload = {
            "schema_version": "1.0",
            "created_at": self.created_at,
            "reliability_passed": self.reliability_passed,
            "performance_gate_configured": bool(self.metric_gates),
            "performance_passed": self.performance_passed,
            "release_passed": self.release_passed,
            "metric_gates": [result.to_dict() for result in self.metric_gates],
            "metadata": self.metadata,
            "reliability": self.reliability.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "failure_case_count": len(self.failure_cases),
        }
        if include_failure_cases:
            payload["failure_cases"] = [case.to_dict() for case in self.failure_cases]
        return payload
