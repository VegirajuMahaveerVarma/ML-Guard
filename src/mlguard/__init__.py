"""MLGuard: local-first reliability checks for machine-learning models."""

from mlguard.adapters.sklearn import SklearnAdapter
from mlguard.api import ModelGuard
from mlguard.results import (
    EvaluationReport,
    Finding,
    MetricGateResult,
    SuiteReport,
    TestResult,
    ValidationReport,
)

__all__ = [
    "EvaluationReport",
    "Finding",
    "MetricGateResult",
    "ModelGuard",
    "SklearnAdapter",
    "SuiteReport",
    "TestResult",
    "ValidationReport",
]
__version__ = "0.1.0"
