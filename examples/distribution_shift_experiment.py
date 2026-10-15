"""Repeatable synthetic experiment for the distribution-shift screening check.

Run after installing the project:
    python examples/distribution_shift_experiment.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mlguard.checks import distribution_shift


def _flagged(reference: np.ndarray, candidate: np.ndarray) -> bool:
    """Return whether the configured mean/variance screen flags these samples."""
    return distribution_shift(reference, candidate).status == "failed"


def run_experiment(*, trials: int = 100, sample_size: int = 512, seed: int = 20261015) -> dict:
    """Measure detection on controlled mean/variance shifts and matched controls."""
    if trials < 1:
        raise ValueError("trials must be a positive integer.")
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2.")

    rng = np.random.default_rng(seed)
    counts = {
        "same_distribution_false_positives": 0,
        "mean_shift_detections": 0,
        "variance_shift_detections": 0,
        "shape_only_changes_flagged": 0,
    }

    for _ in range(trials):
        reference = rng.normal(0.0, 1.0, size=(sample_size, 1))
        same_distribution = rng.normal(0.0, 1.0, size=(sample_size, 1))
        mean_shift = rng.normal(0.75, 1.0, size=(sample_size, 1))
        variance_shift = rng.normal(0.0, 2.0, size=(sample_size, 1))
        # Uniform(-sqrt(3), sqrt(3)) has population mean 0 and variance 1,
        # matching the standard normal's first two moments but not its shape.
        shape_only_change = rng.uniform(
            -np.sqrt(3.0), np.sqrt(3.0), size=(sample_size, 1)
        )

        counts["same_distribution_false_positives"] += int(
            _flagged(reference, same_distribution)
        )
        counts["mean_shift_detections"] += int(_flagged(reference, mean_shift))
        counts["variance_shift_detections"] += int(_flagged(reference, variance_shift))
        counts["shape_only_changes_flagged"] += int(
            _flagged(reference, shape_only_change)
        )

    rates = {
        "same_distribution_false_positive_rate": (
            counts["same_distribution_false_positives"] / trials
        ),
        "mean_shift_detection_rate": counts["mean_shift_detections"] / trials,
        "variance_shift_detection_rate": counts["variance_shift_detections"] / trials,
        "shape_only_change_flag_rate": counts["shape_only_changes_flagged"] / trials,
    }
    passed = (
        rates["mean_shift_detection_rate"] >= 0.95
        and rates["variance_shift_detection_rate"] >= 0.95
        and rates["same_distribution_false_positive_rate"] <= 0.10
    )
    return {
        "experiment": "synthetic_distribution_shift_screen",
        "seed": seed,
        "trials": trials,
        "sample_size_per_distribution": sample_size,
        "thresholds": {
            "standardized_mean_difference": 0.25,
            "variance_ratio": 2.0,
            "minimum_injected_shift_detection_rate": 0.95,
            "maximum_same_distribution_false_positive_rate": 0.10,
        },
        "counts": counts,
        "rates": rates,
        "smoke_gate_passed": passed,
        "interpretation": (
            "Synthetic checks only. The shape-only case documents a known blind spot: "
            "matching means and variances can still hide a changed distribution. "
            "Rates are not estimates of production performance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()

    try:
        report = run_experiment(
            trials=args.trials,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["smoke_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
