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
