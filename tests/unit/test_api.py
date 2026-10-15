import pytest

import numpy as np

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



def test_model_comparison_requires_baseline_model(constant_model, sample_X):
    with pytest.raises(ValueError, match="baseline_model is required"):
        ModelGuard(constant_model).run(
            sample_X, checks=("model_comparison",)
        )


def test_model_comparison_is_available_through_model_guard(constant_model, sample_X):
    report = ModelGuard(constant_model).run(
        sample_X,
        checks=("model_comparison",),
        baseline_model=constant_model,
    )
    assert report.results[0].name == "model_comparison"
    assert report.results[0].status == "passed"
    assert report.results[0].metrics["changed_samples"] == 0


def test_model_comparison_detects_candidate_model_changes(sample_X):
    class AlwaysOneModel:
        def predict(self, X):
            import numpy as np
            return np.ones(len(X), dtype=int)

    class AlwaysZeroModel:
        def predict(self, X):
            import numpy as np
            return np.zeros(len(X), dtype=int)

    report = ModelGuard(AlwaysOneModel()).run(
        sample_X,
        checks=("model_comparison",),
        baseline_model=AlwaysZeroModel(),
        comparison_mode="classification",
    )
    assert report.results[0].status == "failed"
    assert report.results[0].metrics["changed_samples"] == len(sample_X)



def test_model_guard_evaluate_computes_classification_metrics(constant_model):
    report = ModelGuard(constant_model).evaluate(
        [[0.1], [0.2], [0.3]],
        [0, 0, 1],
        task="classification",
        metadata={"split": "validation"},
    )

    assert report.task == "classification"
    assert report.sample_count == 3
    assert report.metrics["accuracy"] == pytest.approx(2 / 3)
    assert report.metrics["precision"] == pytest.approx(1 / 3)
    assert report.metrics["recall"] == pytest.approx(0.5)
    assert report.metadata["split"] == "validation"


def test_model_guard_evaluate_computes_regression_metrics():
    class FirstFeatureRegressor:
        def predict(self, X):
            return np.asarray(X, dtype=float)[:, 0]

    X = [[1.0], [2.0], [3.0]]
    report = ModelGuard(FirstFeatureRegressor()).evaluate(
        X,
        [1.0, 2.5, 2.0],
        task="regression",
    )

    assert report.task == "regression"
    assert report.metrics["mae"] == pytest.approx(0.5)
    assert report.metrics["mse"] == pytest.approx(1.25 / 3)
    assert report.metrics["rmse"] == pytest.approx(np.sqrt(1.25 / 3))


def test_model_guard_evaluate_reports_prediction_errors():
    from mlguard.exceptions import ModelExecutionError

    class BrokenModel:
        def predict(self, X):
            raise RuntimeError("deliberate test failure")

    with pytest.raises(ModelExecutionError, match="evaluation predictions"):
        ModelGuard(BrokenModel()).evaluate([[1.0]], [1.0], task="regression")
