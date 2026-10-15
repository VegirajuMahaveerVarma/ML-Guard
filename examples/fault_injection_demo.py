"""Reproducible smoke evaluation using deliberately faulty toy models.

Run after installing MLGuard in editable mode:
    python examples/fault_injection_demo.py
"""

from __future__ import annotations

import json

import numpy as np

from mlguard import ModelGuard


class ThresholdClassifier:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict(self, X):
        values = np.asarray(X, dtype=float)
        return (values[:, 0] > self.threshold).astype(int)


class IntermittentClassifier:
    """Returns stable outputs twice, then changes on the third call."""

    def __init__(self):
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        value = 0 if self.calls < 3 else 1
        return np.full(len(X), value, dtype=int)


class RejectMissingClassifier:
    def predict(self, X):
        values = np.asarray(X, dtype=float)
        if np.isnan(values).any():
            raise ValueError("This model does not accept NaN features.")
        return (values[:, 0] > 0.5).astype(int)


def main() -> int:
    """Run five known-failure scenarios and summarize whether MLGuard flags them."""
    stability = ModelGuard(IntermittentClassifier()).run(
        [[0.1], [0.9]],
        checks=("prediction_stability",),
        stability_repeats=3,
    ).results[0]

    boundary_X = np.array([[0.49], [0.51], [0.2], [0.8]])
    noise = ModelGuard(ThresholdClassifier()).run(
        boundary_X,
        checks=("noise_robustness",),
        noise_scale=0.1,
        random_state=3,
    ).results[0]

    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    candidate = np.array([[-3.0], [3.0], [-3.0], [3.0]])
    variance_shift = ModelGuard(ThresholdClassifier()).run(
        candidate,
        checks=("distribution_shift",),
        reference_X=reference,
    ).results[0]

    missing = ModelGuard(RejectMissingClassifier()).run(
        [[0.1, 0.2], [0.3, 0.4]],
        checks=("missing_value_robustness",),
        missing_fraction=0.25,
        random_state=4,
    ).results[0]

    comparison_X = np.array([[0.1], [0.5], [0.9]])
    model_comparison = ModelGuard(ThresholdClassifier(0.6)).run(
        comparison_X,
        checks=("model_comparison",),
        baseline_model=ThresholdClassifier(0.4),
        comparison_mode="classification",
    ).results[0]

    scenarios = [
        ("intermittent_prediction_instability", stability),
        ("prediction_change_under_noise", noise),
        ("variance_only_distribution_shift", variance_shift),
        ("model_rejects_missing_values", missing),
        ("candidate_model_prediction_regression", model_comparison),
    ]
    results = [
        {
            "scenario": name,
            "detected": result.status == "failed",
            "status": result.status,
            "metrics": result.metrics,
            "finding_codes": [finding.code for finding in result.findings],
        }
        for name, result in scenarios
    ]
    passed = all(item["detected"] for item in results)
    print(
        json.dumps(
            {"all_expected_failures_detected": passed, "scenarios": results},
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
