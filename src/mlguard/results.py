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
