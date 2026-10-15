"""MLGuard: local-first reliability checks for machine-learning models."""

from mlguard.adapters.sklearn import SklearnAdapter
from mlguard.api import ModelGuard
from mlguard.regression import (
    load_regression_cases,
    replay_regression_cases,
    save_regression_cases,
)
from mlguard.results import (
    EvaluationReport,
    Finding,
    MetricGateResult,
    RegressionCase,
    RegressionReplayReport,
    RegressionReplayResult,
    SuiteReport,
    TestResult,
    ValidationReport,
)

__all__ = [
    "EvaluationReport",
    "Finding",
    "MetricGateResult",
    "ModelGuard",
    "RegressionCase",
    "RegressionReplayReport",
    "RegressionReplayResult",
    "SklearnAdapter",
    "SuiteReport",
    "TestResult",
    "ValidationReport",
    "load_regression_cases",
    "replay_regression_cases",
    "save_regression_cases",
]
__version__ = "0.1.0"
