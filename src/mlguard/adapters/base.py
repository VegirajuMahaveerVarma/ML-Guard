"""Small adapter interface for model prediction."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    """Interface MLGuard needs from a model.

    An adapter intentionally exposes only prediction. Framework-specific
    integrations can implement this protocol without becoming a hard dependency.
    """

    def predict(self, X: Any) -> Any:
        """Return model predictions for input samples."""
        ...
