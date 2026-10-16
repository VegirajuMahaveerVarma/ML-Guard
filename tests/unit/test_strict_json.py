import json

import pytest

from mlguard import load_regression_cases


def test_loader_rejects_nonstandard_json_constants(tmp_path):
    path = tmp_path / "cases.json"
    artifact = {
        "kind": "mlguard-regression-cases",
        "schema_version": "1.0",
        "case_count": 0,
        "cases": [],
        "unexpected": float("nan"),
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="Could not load regression cases"):
        load_regression_cases(path)
