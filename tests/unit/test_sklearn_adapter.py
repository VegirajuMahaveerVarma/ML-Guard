"""Tests for scikit-learn estimators, Pipelines, and DataFrame inputs."""

import numpy as np
import pytest

pytest.importorskip("sklearn")
pd = pytest.importorskip("pandas")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlguard import ModelGuard, SklearnAdapter


def _dataset():
    frame = pd.DataFrame(
        {
            "length": [1.0, 1.1, 0.9, 1.2, 1.05, 0.95, 3.0, 3.1, 2.9, 3.2, 3.05, 2.95],
            "width": [0.9, 1.0, 0.8, 1.1, 0.95, 0.85, 2.8, 2.9, 2.7, 3.0, 2.85, 2.75],
        },
        index=[f"sample-{index}" for index in range(12)],
    )
    labels = np.array([0] * 6 + [1] * 6)
    return frame, labels


def _pipeline():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [("numeric", numeric_pipeline, ["length", "width"])],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=0)),
        ]
    )


def test_sklearn_adapter_rejects_unfitted_estimator():
    with pytest.raises(ValueError, match="must be fitted"):
        SklearnAdapter(LogisticRegression())


def test_sklearn_adapter_accepts_fitted_pipeline():
    X, y = _dataset()
    pipeline = _pipeline().fit(X, y)

    adapter = SklearnAdapter(pipeline)
    np.testing.assert_array_equal(adapter.predict(X), pipeline.predict(X))


def test_from_sklearn_supports_dataframe_pipeline_for_stability_and_noise():
    X, y = _dataset()
    pipeline = _pipeline().fit(X, y)
    guard = ModelGuard.from_sklearn(pipeline)
    original = X.copy(deep=True)

    report = guard.run(
        X,
        checks=("prediction_stability", "noise_robustness"),
        noise_scale=1e-6,
        random_state=19,
    )

    assert [item.name for item in report.results] == [
        "prediction_stability",
        "noise_robustness",
    ]
    assert all(item.status == "passed" for item in report.results)
    pd.testing.assert_frame_equal(X, original)


def test_missing_value_check_preserves_dataframe_columns_for_pipeline():
    X, y = _dataset()
    pipeline = _pipeline().fit(X, y)
    guard = ModelGuard.from_sklearn(pipeline)
    original = X.copy(deep=True)

    result = guard.run(
        X,
        checks=("missing_value_robustness",),
        missing_fraction=0.2,
        random_state=4,
    ).results[0]

    assert result.name == "missing_value_robustness"
    assert result.status == "passed"
    pd.testing.assert_frame_equal(X, original)


def test_sklearn_pipeline_can_be_used_for_baseline_comparison():
    X, y = _dataset()
    baseline = _pipeline().fit(X, y)
    candidate = _pipeline().fit(X, 1 - y)

    report = ModelGuard.from_sklearn(candidate).run(
        X,
        checks=("model_comparison",),
        baseline_model=SklearnAdapter(baseline),
        comparison_mode="classification",
    )

    assert report.results[0].status == "failed"
    assert report.results[0].metrics["changed_samples"] == len(X)
