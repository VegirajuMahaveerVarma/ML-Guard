import json

import pytest

from mlguard import load_regression_cases


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_loader_rejects_nonstandard_json_constants(tmp_path, value):
    path = tmp_path / "cases.json"
    artifact = {
        "kind": "mlguard-regression-cases",
        "schema_version": "1.0",
        "case_count": 0,
        "cases": [],
        "unexpected": value,
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="Could not load regression cases"):
        load_regression_cases(path)


def test_loader_rejects_duplicate_json_object_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"kind":"mlguard-regression-cases","kind":"other",'
        '"schema_version":"1.0","case_count":0,"cases":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Could not load regression cases"):
        load_regression_cases(path)
