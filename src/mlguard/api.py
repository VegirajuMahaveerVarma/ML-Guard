"""Public high-level API for MLGuard."""

from __future__ import annotations

from typing import Any

from mlguard.adapters.base import ModelAdapter
from mlguard.checks import (
    distribution_shift,
    missing_value_robustness,
    model_comparison,
    noise_robustness,
    prediction_stability,
)
from mlguard.results import SuiteReport, TestResult


class ModelGuard:
    """Run reliability checks against a prediction-compatible model."""

    _CHECKS = {
        "prediction_stability": prediction_stability,
        "noise_robustness": noise_robustness,
        "distribution_shift": distribution_shift,
        "missing_value_robustness": missing_value_robustness,
        "model_comparison": model_comparison,
    }

    @classmethod
    def from_sklearn(cls, estimator: Any) -> ModelGuard:
        """Build a guard for a fitted scikit-learn estimator or Pipeline.

        This convenience constructor uses the optional scikit-learn adapter,
        which validates fitted state and preserves incoming data containers.
        Install the optional dependency with pip install mlguard[sklearn].
        """
        from mlguard.adapters.sklearn import SklearnAdapter

        return cls(SklearnAdapter(estimator))

    def __init__(self, model: ModelAdapter) -> None:
        if not callable(getattr(model, "predict", None)):
            raise TypeError("model must expose a callable predict(X) method")
        self.model = model

    def run(
        self,
        X: Any,
        *,
        checks: tuple[str, ...] = ("prediction_stability", "noise_robustness"),
        noise_scale: float = 0.01,
        random_state: int = 0,
        metadata: dict[str, Any] | None = None,
        reference_X: Any | None = None,
        shift_threshold: float = 0.25,
        missing_fraction: float = 0.05,
        baseline_model: ModelAdapter | None = None,
        comparison_mode: str = "auto",
        atol: float = 1e-8,
        rtol: float = 1e-5,
        max_changed_fraction: float = 0.0,
    ) -> SuiteReport:
        """Execute requested checks and return a structured report."""
        if not checks:
            raise ValueError("At least one check must be requested.")
        unknown = sorted(set(checks) - self._CHECKS.keys())
        if unknown:
            raise ValueError(f"Unknown check(s): {', '.join(unknown)}")

        if "distribution_shift" in checks and reference_X is None:
            raise ValueError("reference_X is required for the distribution_shift check.")
        if "model_comparison" in checks and baseline_model is None:
            raise ValueError("baseline_model is required for the model_comparison check.")

        results: list[TestResult] = []
        for name in checks:
            check = self._CHECKS[name]
            if name == "model_comparison":
                result = check(
                    baseline_model,
                    self.model,
                    X,
                    comparison_mode=comparison_mode,
                    atol=atol,
                    rtol=rtol,
                    max_changed_fraction=max_changed_fraction,
                )
            elif name == "distribution_shift":
                result = check(reference_X, X, threshold=shift_threshold)
            elif name == "missing_value_robustness":
                result = check(
                    self.model,
                    X,
                    missing_fraction=missing_fraction,
                    random_state=random_state,
                )
            elif name == "noise_robustness":
                result = check(
                    self.model, X, noise_scale=noise_scale, random_state=random_state
                )
            else:
                result = check(self.model, X)
            results.append(result)
        return SuiteReport(results=tuple(results), metadata=dict(metadata or {}))
