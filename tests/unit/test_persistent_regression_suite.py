"""Integration test for the persisted regression replay fixture used in CI."""

from pathlib import Path

from examples.regression_suite_model import load_model
from mlguard import load_regression_cases, replay_regression_cases


def test_persisted_regression_fixture_replays_as_resolved():
    artifact = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "regression_cases.json"
    )
    cases = load_regression_cases(artifact)
    report = replay_regression_cases(load_model(), cases)

    assert len(cases) == 1
    assert report.passed is True
    assert report.reproduced_count == 0
    assert report.error_count == 0
    assert report.results[0].status == "resolved"
