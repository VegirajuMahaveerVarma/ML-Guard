"""Optional, fitted-estimator adapter for scikit-learn models and pipelines."""

from __future__ import annotations

from typing import Any

from mlguard.exceptions import ModelExecutionError


class SklearnAdapter:
    """Adapt a fitted scikit-learn estimator or Pipeline to MLGuard.

    The scikit-learn dependency is loaded only when this adapter is constructed,
    so users of the framework-neutral API do not need to install scikit-learn.
    Input objects are passed through unchanged to preserve pandas column names
    and other container metadata required by sklearn Pipelines.
    """

    def __init__(self, estimator: Any) -> None:
        if not callable(getattr(estimator, "predict", None)):
            raise TypeError("estimator must expose a callable predict(X) method")

        try:
            from sklearn.exceptions import NotFittedError
            from sklearn.utils.validation import check_is_fitted
        except ImportError as exc:
            raise ImportError(
                "SklearnAdapter requires scikit-learn. Install it with "
                "Install with: python -m pip install -e .[sklearn]."
            ) from exc

        try:
            check_is_fitted(estimator)
        except NotFittedError as exc:
            estimator_name = type(estimator).__name__
            raise ValueError(
                f"{estimator_name} must be fitted before it can be checked by MLGuard."
            ) from exc

        self.estimator = estimator

    def predict(self, X: Any) -> Any:
        """Return estimator predictions without converting or mutating X."""
        try:
            return self.estimator.predict(X)
        except Exception as exc:
            raise ModelExecutionError(
                f"{type(self.estimator).__name__} failed while predicting."
            ) from exc
