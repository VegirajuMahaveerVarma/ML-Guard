"""Tests for reproducible synthetic evaluation helpers."""

import json

import pytest

from examples.distribution_shift_experiment import run_experiment


def test_experiment_reports_multiple_sample_sizes_and_effect_strengths():
    report = run_experiment(
        trials=3,
        sample_size=512,
        seed=17,
        additional_sample_sizes=(128, 256),
        mean_shifts=(0.25, 0.5),
        variance_scales=(1.25, 1.5),
    )

    assert report["sample_sizes"] == [128, 256, 512]
    assert "0.25" in report["results_by_sample_size"]["128"]["rates"][
        "mean_shift_detection_rates"
    ]
    assert "0.5" in report["results_by_sample_size"]["512"]["rates"][
        "mean_shift_detection_rates"
    ]
    assert "1.25" in report["results_by_sample_size"]["256"]["rates"][
        "variance_shift_detection_rates"
    ]
    assert 0.75 in report["mean_shifts"]
    assert 2.0 in report["variance_scales"]
    json.dumps(report, allow_nan=False)


def test_experiment_is_reproducible_for_a_fixed_seed():
    kwargs = {
        "trials": 4,
        "sample_size": 32,
        "seed": 109,
        "additional_sample_sizes": (16,),
    }
    assert run_experiment(**kwargs) == run_experiment(**kwargs)


@pytest.mark.parametrize("sample_size", [0, 1, -2, True])
def test_experiment_rejects_invalid_sample_size(sample_size):
    with pytest.raises(ValueError, match="sample sizes"):
        run_experiment(trials=1, sample_size=sample_size, additional_sample_sizes=())


@pytest.mark.parametrize("trials", [0, -1, True])
def test_experiment_rejects_invalid_trial_count(trials):
    with pytest.raises(ValueError, match="trials"):
        run_experiment(trials=trials, sample_size=4, additional_sample_sizes=())


def test_experiment_rejects_invalid_effect_values():
    with pytest.raises(ValueError, match="mean_shifts"):
        run_experiment(trials=1, sample_size=4, mean_shifts=(float("nan"),))
    with pytest.raises(ValueError, match="variance_scales"):
        run_experiment(trials=1, sample_size=4, variance_scales=(0.0,))



def test_experiment_gate_passes_on_canonical_strong_shifts():
    report = run_experiment(
        trials=20,
        sample_size=512,
        seed=20261015,
        additional_sample_sizes=(),
    )

    assert report["smoke_gate_passed"] is True
    assert report["smoke_gate"]["mean_shift_detection_rate"] >= 0.95
    assert report["smoke_gate"]["variance_shift_detection_rate"] >= 0.95
    assert report["smoke_gate"]["same_distribution_false_positive_rate"] <= 0.10
