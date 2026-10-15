"""MLGuard: local-first reliability checks for machine-learning models."""

from mlguard.adapters.sklearn import SklearnAdapter
from mlguard.api import ModelGuard
from mlguard.results import (
    EvaluationReport,
    Finding,
    SuiteReport,
    TestResult,
    ValidationReport,
)

__all__ = [
    "EvaluationReport",
    "Finding",
    "ModelGuard",
    "SklearnAdapter",
    "SuiteReport",
    "TestResult",
    "ValidationReport",
]
__version__ = "0.1.0"
