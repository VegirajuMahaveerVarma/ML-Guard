"""Tests for captured, serializable and replayable regression cases."""

import json
import sys
from types import ModuleType

import numpy as np
import pytest

from mlguard import (
    ModelGuard,
    load_regression_cases,
    replay_regression_cases,
    save_regression_cases,
)
from mlguard.cli import main


class ThresholdModel:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def predict(self, X):
        values = np.asarray(X, dtype=float)
        return (values[:, 0] > self.threshold).astype(int)


class NaNRejectingModel:
    def predict(self, X):
        values = np.asarray(X, dtype=float)
        if np.isnan(values).any():
            raise ValueError("NaN inputs are not supported")
        return (values[:, 0] > 0.5).astype(int)


class ConstantModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_noise_cases_round_trip_and_reproduce_issue(tmp_path):
    model = ThresholdModel()
    X = np.array([[0.49], [0.51], [0.2], [0.8]])
    report = ModelGuard(model).run(
        X,
        checks=("noise_robustness",),
        noise_scale=0.1,
        random_state=3,
    )

    assert report.results[0].status == "failed"
    assert report.failure_cases
    assert report.results[0].metrics["failure_cases_captured"] == len(report.failure_cases)

    destination = tmp_path / "cases.json"
    save_regression_cases(report.failure_cases, destination)
    cases = load_regression_cases(destination)
    replay = replay_regression_cases(model, cases)

    assert len(cases) == len(report.failure_cases)
    assert replay.reproduced_count == len(cases)
    assert replay.passed is False
    assert all(case.reference_input is not None for case in cases)
    json.loads(destination.read_text(encoding="utf-8"))


def test_missing_value_case_preserves_nan_and_replays_rejection(tmp_path):
    model = NaNRejectingModel()
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.7, 0.8], [0.9, 0.6]])
    report = ModelGuard(model).run(
        X,
        checks=("missing_value_robustness",),
        missing_fraction=0.25,
        random_state=7,
    )

    assert report.results[0].status == "failed"
    assert report.failure_cases
    path = tmp_path / "missing-cases.json"
    save_regression_cases(report.failure_cases, path)
    cases = load_regression_cases(path)

    assert any(np.isnan(np.asarray(case.input, dtype=float)).any() for case in cases)
    replay = replay_regression_cases(model, cases)
    assert replay.reproduced_count == len(cases)


def test_model_comparison_cases_can_be_replayed_until_a_regression_is_fixed():
    X = np.array([[0.1], [0.5], [0.9]])
    baseline = ThresholdModel(0.4)
    candidate = ThresholdModel(0.6)
    report = ModelGuard(candidate).run(
        X,
        checks=("model_comparison",),
        baseline_model=baseline,
        comparison_mode="classification",
    )

    assert report.failure_cases
    assert replay_regression_cases(candidate, report.failure_cases).reproduced_count == 1
    fixed_replay = replay_regression_cases(baseline, report.failure_cases)
    assert fixed_replay.passed is True
    assert fixed_replay.reproduced_count == 0


def test_distribution_shift_case_replays_without_a_model():
    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    candidate = reference + 5.0
    report = ModelGuard(ConstantModel()).run(
        candidate,
        checks=("distribution_shift",),
        reference_X=reference,
    )

    assert report.failure_cases
    replay = replay_regression_cases(None, report.failure_cases)
    assert replay.reproduced_count == len(report.failure_cases)


def test_report_does_not_embed_raw_inputs_unless_explicitly_requested():
    report = ModelGuard(ThresholdModel()).run(
        [[0.49], [0.51], [0.2], [0.8]],
        checks=("noise_robustness",),
        noise_scale=0.1,
        random_state=3,
    )
    ordinary_report = report.to_dict()
    explicit_report = report.to_dict(include_failure_cases=True)

    assert ordinary_report["failure_case_count"] > 0
    assert "failure_cases" not in ordinary_report
    assert "failure_cases" in explicit_report
    assert "input" in explicit_report["failure_cases"][0]


def test_case_file_validation_rejects_wrong_artifact_kind(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"kind": "other", "schema_version": "1.0", "cases": []}))
    with pytest.raises(ValueError, match="not an MLGuard"):
        load_regression_cases(path)


def test_cli_can_export_and_replay_cases(tmp_path, monkeypatch, capsys):
    module = ModuleType("mlguard_case_cli_models")
    module.load_model = ThresholdModel
    monkeypatch.setitem(sys.modules, "mlguard_case_cli_models", module)
    features_path = tmp_path / "features.csv"
    cases_path = tmp_path / "regression-cases.json"
    np.savetxt(features_path, [[0.49], [0.51], [0.2], [0.8]], delimiter=",")

    check_exit = main([
        "check",
        "--model", "mlguard_case_cli_models:load_model",
        "--input", str(features_path),
        "--checks", "noise_robustness",
        "--noise-scale", "0.1",
        "--random-state", "3",
        "--cases-output", str(cases_path),
    ])
    check_report = json.loads(capsys.readouterr().out)

    assert check_exit == 1
    assert cases_path.exists()
    assert check_report["failure_case_count"] > 0

    replay_exit = main([
        "replay",
        "--model", "mlguard_case_cli_models:load_model",
        "--cases", str(cases_path),
    ])
    replay_report = json.loads(capsys.readouterr().out)

    assert replay_exit == 1
    assert replay_report["reproduced_count"] > 0
    assert replay_report["passed"] is False


def test_replay_requires_a_model_for_prediction_cases(tmp_path, capsys):
    path = tmp_path / "cases.json"
    report = ModelGuard(ThresholdModel()).run(
        [[0.49], [0.51], [0.2], [0.8]],
        checks=("noise_robustness",),
        noise_scale=0.1,
        random_state=3,
    )
    save_regression_cases(report.failure_cases, path)

    exit_code = main(["replay", "--cases", str(path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "A model is required" in captured.err
