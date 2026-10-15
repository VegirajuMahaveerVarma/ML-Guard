import json

from mlguard.results import Finding, SuiteReport, TestResult


def test_report_is_json_serializable():
    report = SuiteReport(
        results=(
            TestResult(
                name="example",
                status="passed",
                summary="OK",
                findings=(Finding(code="OK", message="Looks good", severity="info"),),
            ),
        )
    )
    encoded = json.dumps(report.to_dict())
    assert '"schema_version": "1.0"' in encoded
    assert report.passed


def test_empty_report_is_not_considered_passed():
    assert not SuiteReport(results=()).passed
