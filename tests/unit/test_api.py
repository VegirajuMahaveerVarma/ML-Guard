import pytest

from mlguard import ModelGuard


def test_default_run_returns_both_checks(constant_model, sample_X):
    report = ModelGuard(constant_model).run(sample_X)
    assert [result.name for result in report.results] == [
        "prediction_stability",
        "noise_robustness",
    ]
    assert report.passed


def test_unknown_check_is_rejected(constant_model, sample_X):
    with pytest.raises(ValueError, match="Unknown check"):
        ModelGuard(constant_model).run(sample_X, checks=("not_a_check",))


def test_empty_checks_are_rejected(constant_model, sample_X):
    with pytest.raises(ValueError, match="At least one"):
        ModelGuard(constant_model).run(sample_X, checks=())


def test_report_contains_metadata(constant_model, sample_X):
    report = ModelGuard(constant_model).run(
        sample_X, checks=("prediction_stability",), metadata={"model": "demo"}
    )
    assert report.to_dict()["metadata"] == {"model": "demo"}


def test_distribution_shift_check_requires_reference_data(constant_model, sample_X):
    with pytest.raises(ValueError, match="reference_X is required"):
        ModelGuard(constant_model).run(sample_X, checks=("distribution_shift",))


def test_distribution_shift_can_run_through_model_guard(constant_model):
    import numpy as np
    reference = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    report = ModelGuard(constant_model).run(
        reference.copy(), checks=("distribution_shift",), reference_X=reference
    )
    assert report.results[0].name == "distribution_shift"
    assert report.results[0].status == "passed"


def test_missing_value_robustness_is_exposed_through_model_guard(constant_model, sample_X):
    report = ModelGuard(constant_model).run(
        sample_X,
        checks=("missing_value_robustness",),
        missing_fraction=0.25,
        random_state=7,
    )
    assert report.results[0].name == "missing_value_robustness"
    assert report.results[0].status == "passed"
    assert report.results[0].metrics["missing_cells"] == 2
