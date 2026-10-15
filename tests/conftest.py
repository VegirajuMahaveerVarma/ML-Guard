"""Shared test fixtures."""

import numpy as np
import pytest


class ConstantModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class FirstColumnThresholdModel:
    def predict(self, X):
        values = np.asarray(X)
        return (values[:, 0] > 0.5).astype(int)


@pytest.fixture
def constant_model():
    return ConstantModel()


@pytest.fixture
def sample_X():
    return np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.6]], dtype=float)
