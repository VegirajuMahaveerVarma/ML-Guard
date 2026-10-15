import numpy as np
import pytest

from mlguard import ModelGuard
from mlguard.exceptions import InputValidationError


class FirstColumnThresholdModel:
    def predict(self, X):
        values = np.asarray(X)
        return (values[:, 0] > 0.5).astype(int)


def test_prediction_stability_passes_for_deterministic_model(constant_model, sample_X):
    result = ModelGuard(constant_model).run(
        sample_X, checks=("prediction_stability",)
    ).results[0]
    assert result.status == "passed"
    assert result.metrics["prediction_agreement"] == 1.0


def test_noise_robustness_passes_for_constant_model(constant_model, sample_X):
    result = ModelGuard(constant_model).run(
        sample_X, checks=("noise_robustness",), noise_scale=0.1
    ).results[0]
    assert result.status == "passed"


def test_noise_robustness_detects_prediction_changes():
    X = np.array([[0.49], [0.51], [0.2], [0.8]])
    result = ModelGuard(FirstColumnThresholdModel()).run(
        X, checks=("noise_robustness",), noise_scale=0.1, random_state=3
    ).results[0]
    assert result.status == "failed"
    assert result.metrics["changed_samples"] > 0


def test_negative_noise_scale_is_rejected(constant_model, sample_X):
    with pytest.raises(InputValidationError, match="noise_scale"):
        ModelGuard(constant_model).run(
            sample_X, checks=("noise_robustness",), noise_scale=-0.1
        )


def test_non_numeric_features_are_rejected(constant_model):
    with pytest.raises(InputValidationError, match="numeric"):
        ModelGuard(constant_model).run(
            [["a"], ["b"]], checks=("noise_robustness",)
        )


def test_distribution_shift_passes_for_matching_distributions():
    reference = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    candidate = reference + np.array([0.1, 0.1])
    from mlguard.checks import distribution_shift
    result = distribution_shift(reference, candidate, threshold=0.25)
    assert result.status == "passed"
    assert result.metrics["shifted_feature_count"] == 0


def test_distribution_shift_flags_shifted_feature():
    reference = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    candidate = reference.copy()
    candidate[:, 1] += 5.0
    from mlguard.checks import distribution_shift
    result = distribution_shift(reference, candidate, threshold=0.25)
    assert result.status == "failed"
    assert result.metrics["shifted_feature_count"] == 1
    assert result.findings[0].details["feature_index"] == 1


def test_distribution_shift_requires_matching_feature_counts():
    from mlguard.checks import distribution_shift
    with pytest.raises(InputValidationError, match="same number of features"):
        distribution_shift([[1.0, 2.0]], [[1.0]])


def test_distribution_shift_rejects_negative_threshold():
    from mlguard.checks import distribution_shift
    with pytest.raises(InputValidationError, match="threshold"):
        distribution_shift([[1.0]], [[2.0]], threshold=-0.1)
