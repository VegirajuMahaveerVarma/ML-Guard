"""Repeatable synthetic experiment for distribution-shift screening.

Run after installing the project:
    python examples/distribution_shift_experiment.py
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from mlguard.checks import distribution_shift

DEFAULT_MEAN_SHIFTS = (0.25, 0.5, 0.75)
DEFAULT_VARIANCE_SCALES = (1.25, 1.5, 2.0)
DEFAULT_ADDITIONAL_SAMPLE_SIZES = (128, 2048)
GATE_MEAN_SHIFT = 0.75
GATE_VARIANCE_SCALE = 2.0


def _flagged(reference: np.ndarray, candidate: np.ndarray) -> bool:
    """Return whether the configured mean/variance screen flags these samples."""
    return distribution_shift(reference, candidate).status == "failed"


def _rate(numerator: int, denominator: int) -> float:
    """Calculate a rate without silently dividing by zero."""
    return float(numerator / denominator)


def run_experiment(
    *,
    trials: int = 100,
    sample_size: int = 512,
    seed: int = 20261015,
    additional_sample_sizes: tuple[int, ...] = DEFAULT_ADDITIONAL_SAMPLE_SIZES,
    mean_shifts: tuple[float, ...] = DEFAULT_MEAN_SHIFTS,
    variance_scales: tuple[float, ...] = DEFAULT_VARIANCE_SCALES,
) -> dict:
    """Measure false alarms and shift detection across a synthetic parameter grid.

    The canonical smoke gate uses the requested sample size, a mean shift of
    0.75 standard deviations, and a variance scale factor of 2.0. Additional
    conditions measure weaker and stronger effects at other sample sizes.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer.")
    sample_sizes = tuple(sorted(set((sample_size, *additional_sample_sizes))))
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 2
        for size in sample_sizes
    ):
        raise ValueError("all sample sizes must be integers of at least 2.")
    if not mean_shifts or any(
        not math.isfinite(value) or value < 0 for value in mean_shifts
    ):
        raise ValueError("mean_shifts must contain finite, non-negative values.")
    if not variance_scales or any(
        not math.isfinite(value) or value <= 0 for value in variance_scales
    ):
        raise ValueError("variance_scales must contain finite, positive scale factors.")

    effective_mean_shifts = tuple(sorted(set((*mean_shifts, GATE_MEAN_SHIFT))))
    effective_variance_scales = tuple(
        sorted(set((*variance_scales, GATE_VARIANCE_SCALE)))
    )
    rng = np.random.default_rng(seed)
    results_by_size: dict[str, dict] = {}

    for size in sample_sizes:
        counts = {
            "same_distribution_false_positives": 0,
            "shape_only_changes_flagged": 0,
            "mean_shift_detections": {str(value): 0 for value in effective_mean_shifts},
            "variance_shift_detections": {
                str(value): 0 for value in effective_variance_scales
            },
        }
        for _ in range(trials):
            reference = rng.normal(0.0, 1.0, size=(size, 1))
            same_distribution = rng.normal(0.0, 1.0, size=(size, 1))
            shape_only_change = rng.uniform(
                -np.sqrt(3.0), np.sqrt(3.0), size=(size, 1)
            )
            counts["same_distribution_false_positives"] += int(
                _flagged(reference, same_distribution)
            )
            counts["shape_only_changes_flagged"] += int(
                _flagged(reference, shape_only_change)
            )

            for effect in effective_mean_shifts:
                shifted = rng.normal(effect, 1.0, size=(size, 1))
                counts["mean_shift_detections"][str(effect)] += int(
                    _flagged(reference, shifted)
                )
            for scale in effective_variance_scales:
                shifted = rng.normal(0.0, scale, size=(size, 1))
                counts["variance_shift_detections"][str(scale)] += int(
                    _flagged(reference, shifted)
                )

        rates = {
            "same_distribution_false_positive_rate": _rate(
                counts["same_distribution_false_positives"], trials
            ),
            "shape_only_change_flag_rate": _rate(
                counts["shape_only_changes_flagged"], trials
            ),
            "mean_shift_detection_rates": {
                effect: _rate(count, trials)
                for effect, count in counts["mean_shift_detections"].items()
            },
            "variance_shift_detection_rates": {
                scale: _rate(count, trials)
                for scale, count in counts["variance_shift_detections"].items()
            },
        }
        results_by_size[str(size)] = {
            "sample_size_per_distribution": size,
            "trials": trials,
            "counts": counts,
            "rates": rates,
        }

    canonical = results_by_size[str(sample_size)]
    canonical_mean_rate = canonical["rates"]["mean_shift_detection_rates"][
        str(GATE_MEAN_SHIFT)
    ]
    canonical_variance_rate = canonical["rates"]["variance_shift_detection_rates"][
        str(GATE_VARIANCE_SCALE)
    ]
    false_positive_rate = canonical["rates"]["same_distribution_false_positive_rate"]
    passed = (
        canonical_mean_rate >= 0.95
        and canonical_variance_rate >= 0.95
        and false_positive_rate <= 0.10
    )

    return {
        "experiment": "synthetic_distribution_shift_screen",
        "seed": seed,
        "trials_per_configuration": trials,
        "canonical_sample_size": sample_size,
        "sample_sizes": list(sample_sizes),
        "mean_shifts": list(effective_mean_shifts),
        "variance_scales": list(effective_variance_scales),
        "variance_scale_definition": (
            "Values are standard-deviation multipliers; the variance multiplier is their square."
        ),
        "check_thresholds": {
            "standardized_mean_difference": 0.25,
            "variance_ratio": 2.0,
        },
        "smoke_gate": {
            "sample_size": sample_size,
            "mean_shift": GATE_MEAN_SHIFT,
            "standard_deviation_multiplier": GATE_VARIANCE_SCALE,
            "minimum_detection_rate": 0.95,
            "maximum_same_distribution_false_positive_rate": 0.10,
            "mean_shift_detection_rate": canonical_mean_rate,
            "variance_shift_detection_rate": canonical_variance_rate,
            "same_distribution_false_positive_rate": false_positive_rate,
            "passed": passed,
        },
        "smoke_gate_passed": passed,
        "results_by_sample_size": results_by_size,
        "interpretation": (
            "Synthetic experiments only. Results describe these distributions, "
            "thresholds, sample sizes, and seeds; they are not estimates of general "
            "or production performance. The shape-only case documents a known blind "
            "spot: matching means and variances can still hide a changed distribution."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument(
        "--additional-sample-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_ADDITIONAL_SAMPLE_SIZES),
        help="Additional sample sizes to evaluate alongside --sample-size.",
    )
    parser.add_argument(
        "--mean-shifts",
        type=float,
        nargs="+",
        default=list(DEFAULT_MEAN_SHIFTS),
        help="Mean-shift magnitudes, in standard-deviation units.",
    )
    parser.add_argument(
        "--variance-scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_VARIANCE_SCALES),
        help="Standard-deviation multipliers for variance-shift scenarios.",
    )
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()

    try:
        report = run_experiment(
            trials=args.trials,
            sample_size=args.sample_size,
            seed=args.seed,
            additional_sample_sizes=tuple(args.additional_sample_sizes),
            mean_shifts=tuple(args.mean_shifts),
            variance_scales=tuple(args.variance_scales),
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["smoke_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
