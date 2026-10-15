"""Tests for the command-line interface."""

import json
import sys
from types import ModuleType

import numpy as np

from mlguard.cli import main


def _install_model_module(monkeypatch, name, *, factory, extra=None):
    module = ModuleType(name)
    module.load_model = factory
    for key, value in (extra or {}).items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_list_checks_displays_all_builtin_checks(capsys):
    exit_code = main(["list-checks"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "prediction_stability" in output
    assert "distribution_shift" in output
    assert "missing_value_robustness" in output
    assert "model_comparison" in output


def test_cli_writes_json_report_for_passing_check(tmp_path, monkeypatch, capsys):
    class StableModel:
        def predict(self, X):
            values = np.asarray(X)
            return (values[:, 0] > 0.5).astype(int)

    _install_model_module(monkeypatch, "mlguard_cli_test_models", factory=StableModel)
    input_file = tmp_path / "features.csv"
    output_file = tmp_path / "report.json"
    np.savetxt(input_file, np.array([[0.1, 0.2], [0.9, 0.8]]), delimiter=",")

    exit_code = main(
        [
            "check",
            "--model",
            "mlguard_cli_test_models:load_model",
            "--input",
            str(input_file),
            "--checks",
            "prediction_stability",
            "--metadata-json",
            '{"origin": "cli-test"}',
            "--output",
            str(output_file),
        ]
    )

    stdout_report = json.loads(capsys.readouterr().out)
    saved_report = json.loads(output_file.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert stdout_report == saved_report
    assert stdout_report["passed"] is True
    assert stdout_report["metadata"]["origin"] == "cli-test"
    assert stdout_report["results"][0]["name"] == "prediction_stability"


def test_cli_exits_nonzero_when_a_check_fails(tmp_path, monkeypatch, capsys):
    class ChangingModel:
        def __init__(self):
            self.calls = 0

        def predict(self, X):
            self.calls += 1
            return np.full(len(X), self.calls)

    _install_model_module(monkeypatch, "mlguard_cli_changing_model", factory=ChangingModel)
    input_file = tmp_path / "features.csv"
    np.savetxt(input_file, np.array([[0.1], [0.2]]), delimiter=",")

    exit_code = main(
        [
            "check",
            "--model",
            "mlguard_cli_changing_model:load_model",
            "--input",
            str(input_file),
            "--checks",
            "prediction_stability",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["passed"] is False
    assert report["results"][0]["status"] == "failed"


def test_cli_requires_reference_csv_for_distribution_shift(
    tmp_path, monkeypatch, capsys
):
    class StableModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    _install_model_module(monkeypatch, "mlguard_cli_shift_model", factory=StableModel)
    input_file = tmp_path / "features.csv"
    np.savetxt(input_file, np.array([[0.1], [0.2]]), delimiter=",")

    exit_code = main(
        [
            "check",
            "--model",
            "mlguard_cli_shift_model:load_model",
            "--input",
            str(input_file),
            "--checks",
            "distribution_shift",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "--reference-input is required" in captured.err


def test_cli_can_compare_candidate_and_baseline_models(tmp_path, monkeypatch, capsys):
    class ThresholdModel:
        def __init__(self, threshold):
            self.threshold = threshold

        def predict(self, X):
            values = np.asarray(X)
            return (values[:, 0] > self.threshold).astype(int)

    _install_model_module(
        monkeypatch,
        "mlguard_cli_comparison_models",
        factory=lambda: ThresholdModel(0.6),
        extra={"load_baseline": lambda: ThresholdModel(0.4)},
    )
    input_file = tmp_path / "features.csv"
    np.savetxt(input_file, np.array([[0.1], [0.5], [0.9]]), delimiter=",")

    exit_code = main(
        [
            "check",
            "--model",
            "mlguard_cli_comparison_models:load_model",
            "--baseline-model",
            "mlguard_cli_comparison_models:load_baseline",
            "--input",
            str(input_file),
            "--checks",
            "model_comparison",
            "--comparison-mode",
            "classification",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["results"][0]["metrics"]["changed_samples"] == 1


def test_cli_rejects_non_object_metadata(tmp_path, monkeypatch, capsys):
    class StableModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    _install_model_module(monkeypatch, "mlguard_cli_metadata_model", factory=StableModel)
    input_file = tmp_path / "features.csv"
    np.savetxt(input_file, np.array([[0.1], [0.2]]), delimiter=",")

    exit_code = main(
        [
            "check",
            "--model",
            "mlguard_cli_metadata_model:load_model",
            "--input",
            str(input_file),
            "--checks",
            "prediction_stability",
            "--metadata-json",
            '["not", "an", "object"]',
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "--metadata-json must decode to a JSON object" in captured.err



def test_cli_combined_validation_passes_configured_metric_gate(
    tmp_path, monkeypatch, capsys
):
    class ThresholdModel:
        def predict(self, X):
            values = np.asarray(X, dtype=float)
            return (values[:, 0] > 0.5).astype(int)

    _install_model_module(monkeypatch, "mlguard_cli_validation_pass", factory=ThresholdModel)
    input_file = tmp_path / "features.csv"
    target_file = tmp_path / "targets.csv"
    np.savetxt(input_file, [[0.1], [0.9]], delimiter=",")
    np.savetxt(target_file, [0, 1], delimiter=",")

    exit_code = main([
        "check", "--model", "mlguard_cli_validation_pass:load_model",
        "--input", str(input_file), "--targets", str(target_file),
        "--task", "classification", "--checks", "prediction_stability",
        "--minimum-metric", "accuracy=1.0",
    ])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["evaluation"]["metrics"]["accuracy"] == 1.0
    assert report["metric_gates"][0]["passed"] is True
    assert report["release_passed"] is True


def test_cli_returns_failure_when_performance_gate_is_not_met(
    tmp_path, monkeypatch, capsys
):
    class ThresholdModel:
        def predict(self, X):
            values = np.asarray(X, dtype=float)
            return (values[:, 0] > 0.5).astype(int)

    _install_model_module(monkeypatch, "mlguard_cli_validation_fail", factory=ThresholdModel)
    input_file = tmp_path / "features.csv"
    target_file = tmp_path / "targets.csv"
    np.savetxt(input_file, [[0.1], [0.6], [0.9]], delimiter=",")
    np.savetxt(target_file, [0, 0, 1], delimiter=",")

    exit_code = main([
        "check", "--model", "mlguard_cli_validation_fail:load_model",
        "--input", str(input_file), "--targets", str(target_file),
        "--task", "classification", "--checks", "prediction_stability",
        "--minimum-metric", "accuracy=0.8",
    ])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["reliability_passed"] is True
    assert report["performance_passed"] is False
    assert report["release_passed"] is False


def test_cli_requires_targets_and_task_together(tmp_path, monkeypatch, capsys):
    class ThresholdModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    _install_model_module(monkeypatch, "mlguard_cli_validation_pair", factory=ThresholdModel)
    input_file = tmp_path / "features.csv"
    np.savetxt(input_file, [[0.1], [0.9]], delimiter=",")

    exit_code = main([
        "check", "--model", "mlguard_cli_validation_pair:load_model",
        "--input", str(input_file), "--task", "classification",
    ])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "--targets and --task must be provided together" in captured.err



def test_cli_distribution_shift_accepts_variance_ratio_threshold(
    tmp_path, monkeypatch, capsys
):
    class StableModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    _install_model_module(monkeypatch, "mlguard_cli_variance_model", factory=StableModel)
    input_file = tmp_path / "candidate.csv"
    reference_file = tmp_path / "reference.csv"
    candidate = np.array([[-3.0], [3.0], [-3.0], [3.0]])
    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    np.savetxt(input_file, candidate, delimiter=",")
    np.savetxt(reference_file, reference, delimiter=",")

    exit_code = main(
        [
            "check",
            "--model",
            "mlguard_cli_variance_model:load_model",
            "--input",
            str(input_file),
            "--reference-input",
            str(reference_file),
            "--checks",
            "distribution_shift",
            "--variance-ratio-threshold",
            "10.0",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["results"][0]["status"] == "passed"
    assert report["results"][0]["metrics"]["variance_ratios"] == [9.0]
    assert report["results"][0]["metrics"]["variance_ratio_threshold"] == 10.0



def test_cli_stability_repeats_detects_third_call_change(tmp_path, monkeypatch, capsys):
    class ThirdCallChangesModel:
        def __init__(self):
            self.calls = 0

        def predict(self, X):
            self.calls += 1
            return np.full(len(X), 0 if self.calls < 3 else 1, dtype=int)

    _install_model_module(
        monkeypatch, "mlguard_cli_stability_model", factory=ThirdCallChangesModel
    )
    input_file = tmp_path / "features.csv"
    np.savetxt(input_file, [[0.1], [0.9]], delimiter=",")

    exit_code = main(
        [
            "check",
            "--model",
            "mlguard_cli_stability_model:load_model",
            "--input",
            str(input_file),
            "--checks",
            "prediction_stability",
            "--stability-repeats",
            "3",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["results"][0]["status"] == "failed"
    assert report["results"][0]["metrics"]["repeats"] == 3
