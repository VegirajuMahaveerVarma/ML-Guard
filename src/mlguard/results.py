"""JSON-friendly result models for MLGuard checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Status = Literal["passed", "failed", "error", "skipped"]


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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["findings"] = [asdict(item) for item in self.findings]
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report without storing raw input data."""
        return {
            "schema_version": "1.0",
            "created_at": self.created_at,
            "passed": self.passed,
            "metadata": self.metadata,
            "results": [result.to_dict() for result in self.results],
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Return the combined report as a JSON-compatible dictionary."""
        return {
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
        }
