import builtins

import pytest

from mlguard import SklearnAdapter


class DummyModel:
    def predict(self, X):
        return X


def test_sklearn_adapter_shows_installation_message_when_dependency_missing(monkeypatch):
    original_import = builtins.__import__

    def import_without_sklearn(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("No module named sklearn")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_sklearn)

    with pytest.raises(ImportError) as error:
        SklearnAdapter(DummyModel())

    assert str(error.value) == (
        "SklearnAdapter requires scikit-learn. Install the optional dependency "
        "with: python -m pip install scikit-learn."
    )