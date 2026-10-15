"""Public high-level API for MLGuard."""

from __future__ import annotations

from typing import Any

from mlguard.adapters.base import ModelAdapter
from mlguard.checks import noise_robustness, prediction_stability
from mlguard.results import SuiteReport, TestResult


class ModelGuard:
    """Run reliability checks against a prediction-compatible model."""

    _CHECKS = {
        "prediction_stability": prediction_stability,
        "noise_robustness": noise_robustness,
    }

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
    ) -> SuiteReport:
        """Execute requested checks and return a structured report."""
        if not checks:
            raise ValueError("At least one check must be requested.")
        unknown = sorted(set(checks) - self._CHECKS.keys())
        if unknown:
            raise ValueError(f"Unknown check(s): {', '.join(unknown)}")

        results: list[TestResult] = []
        for name in checks:
            check = self._CHECKS[name]
            if name == "noise_robustness":
                result = check(
                    self.model, X, noise_scale=noise_scale, random_state=random_state
                )
            else:
                result = check(self.model, X)
            results.append(result)
        return SuiteReport(results=tuple(results), metadata=dict(metadata or {}))
