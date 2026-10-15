"""Model adapters supported by MLGuard."""

from mlguard.adapters.base import ModelAdapter
from mlguard.adapters.sklearn import SklearnAdapter

__all__ = ["ModelAdapter", "SklearnAdapter"]
