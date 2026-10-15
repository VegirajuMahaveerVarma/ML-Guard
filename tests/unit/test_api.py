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
