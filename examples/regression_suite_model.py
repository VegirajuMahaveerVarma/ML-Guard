"""Small deterministic model used by MLGuard's persistent replay fixture."""

from __future__ import annotations

import numpy as np


class RegressionSuiteModel:
    """Return the recorded baseline class for the persisted CI regression case."""

    def predict(self, X: object) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("The regression-suite fixture expects one numeric feature.")
        return (values[:, 0] > 0.5).astype(int)


def load_model() -> RegressionSuiteModel:
    """Factory callable used by the CLI in CI."""
    return RegressionSuiteModel()
