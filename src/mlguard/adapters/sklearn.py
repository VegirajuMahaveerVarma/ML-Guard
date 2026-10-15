"""Optional adapter for estimators implementing the scikit-learn API."""

from __future__ import annotations

from typing import Any

from mlguard.exceptions import ModelExecutionError


class SklearnAdapter:
    """Adapt an estimator exposing ``predict(X)`` to the MLGuard protocol."""

    def __init__(self, estimator: Any) -> None:
        if not callable(getattr(estimator, "predict", None)):
            raise TypeError("estimator must expose a callable predict(X) method")
        self.estimator = estimator

    def predict(self, X: Any) -> Any:
        try:
            return self.estimator.predict(X)
        except Exception as exc:
            raise ModelExecutionError("The model failed while predicting.") from exc
