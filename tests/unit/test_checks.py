import numpy as np
import pytest

from mlguard import ModelGuard
from mlguard.checks import missing_value_robustness, model_comparison
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



class OffsetRegressionModel:
    def __init__(self, offset=0.0):
        self.offset = offset

    def predict(self, X):
        values = np.asarray(X, dtype=float)
        return values[:, 0] + self.offset


class ConfigurableThresholdModel:
    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, X):
        values = np.asarray(X, dtype=float)
        return (values[:, 0] > self.threshold).astype(int)


def test_model_comparison_passes_for_identical_classification_predictions():
    X = np.array([[0.1], [0.5], [0.9]])
    result = model_comparison(
        ConfigurableThresholdModel(0.4),
        ConfigurableThresholdModel(0.4),
        X,
        comparison_mode="classification",
    )
    assert result.status == "passed"
    assert result.metrics["changed_samples"] == 0
    assert result.metrics["comparison_mode"] == "classification"


def test_model_comparison_detects_changed_classification_predictions():
    X = np.array([[0.1], [0.5], [0.9]])
    result = model_comparison(
        ConfigurableThresholdModel(0.4),
        ConfigurableThresholdModel(0.6),
        X,
        comparison_mode="classification",
    )
    assert result.status == "failed"
    assert result.metrics["changed_samples"] == 1
    assert result.metrics["changed_fraction"] == pytest.approx(1 / 3)
    assert result.findings[0].code == "MODEL_PREDICTIONS_CHANGED"


def test_model_comparison_uses_tolerance_for_regression_predictions():
    X = np.array([[0.1], [0.5], [0.9]])
    result = model_comparison(
        OffsetRegressionModel(),
        OffsetRegressionModel(offset=0.001),
        X,
        comparison_mode="regression",
        atol=0.01,
        rtol=0.0,
    )
    assert result.status == "passed"
    assert result.metrics["changed_samples"] == 0
    assert result.metrics["max_absolute_difference"] == pytest.approx(0.001)


def test_model_comparison_fails_when_regression_change_exceeds_tolerance():
    X = np.array([[0.1], [0.5], [0.9]])
    result = model_comparison(
        OffsetRegressionModel(),
        OffsetRegressionModel(offset=1.0),
        X,
        comparison_mode="regression",
        atol=0.01,
        rtol=0.0,
    )
    assert result.status == "failed"
    assert result.metrics["changed_samples"] == 3
    assert result.metrics["mean_absolute_difference"] == pytest.approx(1.0)


def test_model_comparison_can_allow_a_small_changed_fraction():
    X = np.array([[0.1], [0.5], [0.9]])
    result = model_comparison(
        ConfigurableThresholdModel(0.4),
        ConfigurableThresholdModel(0.6),
        X,
        comparison_mode="classification",
        max_changed_fraction=1 / 3,
    )
    assert result.status == "passed"
    assert result.metrics["changed_fraction"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"comparison_mode": "unknown"}, "comparison_mode"),
        ({"atol": -0.1}, "atol"),
        ({"rtol": np.inf}, "rtol"),
        ({"max_changed_fraction": 1.1}, "max_changed_fraction"),
    ],
)
def test_model_comparison_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(InputValidationError, match=message):
        model_comparison(
            OffsetRegressionModel(),
            OffsetRegressionModel(),
            [[0.1], [0.2]],
            **kwargs,
        )


def test_distribution_shift_detects_variance_only_change():
    from mlguard.checks import distribution_shift

    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    candidate = np.array([[-3.0], [3.0], [-3.0], [3.0]])
    result = distribution_shift(reference, candidate, threshold=0.25)

    assert result.status == "failed"
    assert result.metrics["mean_shifted_feature_count"] == 0
    assert result.metrics["variance_shifted_feature_count"] == 1
    assert result.metrics["variance_ratios"] == [9.0]
    assert result.findings[0].details["shift_types"] == ["variance"]


def test_distribution_shift_reports_unbounded_variance_ratio_as_json_null():
    import json

    from mlguard.checks import distribution_shift

    reference = np.zeros((4, 1))
    candidate = np.array([[0.0], [1.0], [0.0], [1.0]])
    result = distribution_shift(reference, candidate)

    assert result.status == "failed"
    assert result.metrics["variance_ratios"] == [None]
    assert result.findings[0].details["variance_ratio"] is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_distribution_shift_skips_variance_comparison_for_single_sample():
    from mlguard.checks import distribution_shift

    result = distribution_shift([[0.0]], [[0.1]])

    assert result.metrics["variance_check_available"] is False
    assert result.metrics["variance_shifted_feature_count"] == 0
    assert result.metrics["variance_ratios"] == [None]


def test_distribution_shift_rejects_variance_ratio_threshold_at_or_below_one():
    from mlguard.checks import distribution_shift

    with pytest.raises(InputValidationError, match="variance_ratio_threshold"):
        distribution_shift([[0.0], [1.0]], [[0.0], [1.0]], variance_ratio_threshold=1.0)


class ThirdCallChangesModel:
    def __init__(self):
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        value = 0 if self.calls < 3 else 1
        return np.full(len(X), value, dtype=int)


def test_prediction_stability_catches_intermitttent_change_on_third_call():
    model = ThirdCallChangesModel()
    result = ModelGuard(model).run([[0.1], [0.9]], checks=("prediction_stability",)).results[0]

    assert result.status == "failed"
    assert result.metrics["repeats"] == 3
    assert result.metrics["changed_samples"] == 2


def test_prediction_stability_rejects_fewer_than_two_repeats():
    with pytest.raises(InputValidationError, match="repeats"):
        ModelGuard(ThirdCallChangesModel()).run(
            [[0.1]], checks=("prediction_stability",), stability_repeats=1
        )
