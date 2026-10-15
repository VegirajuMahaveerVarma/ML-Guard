import numpy as np
import pytest

from mlguard import ModelGuard
from mlguard.checks import missing_value_robustness
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


class NaNRejectingModel:
    def predict(self, X):
        values = np.asarray(X, dtype=float)
        if np.isnan(values).any():
            raise ValueError("NaN values are not supported")
        return values[:, 0]


class NaNOutputModel:
    def predict(self, X):
        values = np.asarray(X, dtype=float)
        if np.isnan(values).any():
            return np.full(len(values), np.nan)
        return values[:, 0]


def test_missing_value_robustness_passes_when_model_handles_nan():
    class NaNHandlingModel:
        def predict(self, X):
            values = np.asarray(X, dtype=float)
            return np.nan_to_num(values[:, 0], nan=0.0)

    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    result = missing_value_robustness(
        NaNHandlingModel(), X, missing_fraction=0.25, random_state=4
    )
    assert result.status == "passed"
    assert result.metrics["missing_cells"] == 2
    assert result.metrics["invalid_prediction_count"] == 0


def test_missing_value_robustness_reports_model_rejection():
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = missing_value_robustness(
        NaNRejectingModel(), X, missing_fraction=0.25, random_state=4
    )
    assert result.status == "failed"
    assert result.findings[0].code == "MODEL_REJECTED_MISSING_VALUES"


def test_missing_value_robustness_detects_invalid_predictions():
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = missing_value_robustness(
        NaNOutputModel(), X, missing_fraction=0.25, random_state=4
    )
    assert result.status == "failed"
    assert result.findings[0].code == "INVALID_PREDICTIONS_WITH_MISSING_INPUT"


def test_missing_value_robustness_does_not_mutate_input():
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    original = X.copy()
    result = missing_value_robustness(
        FirstColumnThresholdModel(), X, missing_fraction=0.25, random_state=2
    )
    assert result.status == "passed"
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, np.inf])
def test_invalid_missing_fraction_is_rejected(fraction):
    with pytest.raises(InputValidationError, match="missing_fraction"):
        missing_value_robustness(
            NaNOutputModel(), [[0.1], [0.2]], missing_fraction=fraction
        )


def test_missing_value_robustness_requires_2d_numeric_input():
    with pytest.raises(InputValidationError, match="2D feature matrix"):
        missing_value_robustness(NaNOutputModel(), [0.1, 0.2])
