"""Command-line interface for MLGuard."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mlguard import __version__
from mlguard.api import ModelGuard
from mlguard.regression import (
    load_regression_cases,
    replay_regression_cases,
    save_regression_cases,
)
from mlguard.results import SuiteReport, ValidationReport

CHECK_DESCRIPTIONS = {
    "prediction_stability": "Check repeat-prediction determinism.",
    "noise_robustness": "Check predictions after Gaussian input noise.",
    "distribution_shift": "Compare reference and candidate feature distributions.",
    "missing_value_robustness": "Check model behavior when feature values are missing.",
    "model_comparison": "Compare a candidate model with a baseline model.",
}


def _model_from_spec(spec: str) -> Any:
    """Load a model instance from a trusted 'module:attribute' reference."""
    module_name, separator, attribute_path = spec.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(
            "Model references must use 'module:attribute', for example "
            "'my_models:load_model'."
        )

    module = importlib.import_module(module_name)
    value: Any = module
    for name in attribute_path.split("."):
        value = getattr(value, name)

    if isinstance(value, type):
        value = value()
    elif callable(value) and not callable(getattr(value, "predict", None)):
        value = value()

    if not callable(getattr(value, "predict", None)):
        raise TypeError(
            f"'{spec}' must resolve to a model instance or a zero-argument factory "
            "that returns an object with predict(X)."
        )
    return value


def _load_numeric_csv(path: Path, *, skip_header: bool) -> np.ndarray:
    """Load a numeric feature matrix from a local CSV file."""
    try:
        values = np.loadtxt(
            path,
            delimiter=",",
            dtype=float,
            skiprows=1 if skip_header else 0,
            ndmin=2,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not read '{path}' as a numeric CSV feature matrix: {exc}"
        ) from exc

    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"'{path}' must contain at least one row and one feature column.")
    if not np.isfinite(values).all():
        raise ValueError(
            f"'{path}' contains missing or non-finite values. Supply finite numeric "
            "features; MLGuard injects test perturbations into a copy when needed."
        )
    return values




def _load_numeric_targets(path: Path, *, skip_header: bool) -> np.ndarray:
    """Load one numeric target value per row from a local CSV file."""
    try:
        values = np.loadtxt(
            path,
            delimiter=",",
            dtype=float,
            skiprows=1 if skip_header else 0,
            ndmin=1,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not read '{path}' as a one-column numeric target CSV: {exc}"
        ) from exc

    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"'{path}' must contain exactly one target column and at least one row.")
    if not np.isfinite(values).all():
        raise ValueError(f"'{path}' contains missing or non-finite target values.")
    return values


def _parse_metric_thresholds(values: list[str], *, option_name: str) -> dict[str, float]:
    """Parse repeated CLI values in metric=value form."""
    thresholds: dict[str, float] = {}
    for item in values:
        metric, separator, raw_value = item.partition("=")
        metric = metric.strip()
        if not separator or not metric or not raw_value.strip():
            raise ValueError(f"{option_name} values must use metric=value syntax.")
        if metric in thresholds:
            raise ValueError(f"Metric {metric!r} was provided more than once in {option_name}.")
        try:
            threshold = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Threshold for {metric!r} in {option_name} must be numeric."
            ) from exc
        if not math.isfinite(threshold):
            raise ValueError(f"Threshold for {metric!r} in {option_name} must be finite.")
        thresholds[metric] = threshold
    return thresholds


def _parse_checks(value: str) -> tuple[str, ...]:
    checks = tuple(part.strip() for part in value.split(",") if part.strip())
    if not checks:
        raise ValueError("Provide at least one check using --checks.")
    unknown = sorted(set(checks) - CHECK_DESCRIPTIONS.keys())
    if unknown:
        raise ValueError(
            f"Unknown check(s): {', '.join(unknown)}. "
            f"Available checks: {', '.join(CHECK_DESCRIPTIONS)}."
        )
    return checks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlguard",
        description="Run local reliability checks against machine-learning models.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "list-checks",
        help="List built-in reliability checks.",
        description="Built-in MLGuard reliability checks.",
    )

    check_parser = subparsers.add_parser(
        "check",
        help="Run reliability checks against a model.",
        description=(
            "Run checks against a model referenced as module:attribute. "
            "Imports execute local Python code, so only load trusted modules."
        ),
    )
    check_parser.add_argument(
        "--model",
        required=True,
        help="Trusted module:attribute that resolves to a model or zero-argument factory.",
    )
    check_parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory to add to Python's import path; may be supplied more than once.",
    )
    check_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="CSV",
        help="Path to the numeric CSV feature matrix.",
    )
    check_parser.add_argument(
        "--checks",
        default="prediction_stability,noise_robustness",
        help="Comma-separated check names. Defaults to prediction_stability,noise_robustness.",
    )
    check_parser.add_argument(
        "--skip-header",
        action="store_true",
        help="Skip the first row in the input and reference CSV files.",
    )
    check_parser.add_argument(
        "--reference-input",
        type=Path,
        metavar="CSV",
        help="Reference numeric CSV required by distribution_shift.",
    )
    check_parser.add_argument(
        "--baseline-model",
        help="Trusted module:attribute for the baseline model, required by model_comparison.",
    )
    check_parser.add_argument("--noise-scale", type=float, default=0.01)
    check_parser.add_argument("--random-state", type=int, default=0)
    check_parser.add_argument("--stability-repeats", type=int, default=3)
    check_parser.add_argument("--shift-threshold", type=float, default=0.25)
    check_parser.add_argument("--variance-ratio-threshold", type=float, default=2.0)
    check_parser.add_argument(
        "--shape-alpha",
        type=float,
        default=0.05,
        help="Family-wise false-positive budget for empirical-CDF feature-shape screening.",
    )
    check_parser.add_argument("--missing-fraction", type=float, default=0.05)
    check_parser.add_argument(
        "--targets",
        type=Path,
        metavar="CSV",
        help="One-column numeric ground-truth CSV; enables combined validation with --task.",
    )
    check_parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        help="Evaluation task; must be supplied together with --targets.",
    )
    check_parser.add_argument(
        "--average",
        choices=("macro", "weighted", "micro"),
        default="macro",
        help="Classification averaging mode for combined validation.",
    )
    check_parser.add_argument(
        "--minimum-metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Minimum acceptable metric; repeat for multiple gates, e.g. accuracy=0.9.",
    )
    check_parser.add_argument(
        "--maximum-metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Maximum acceptable metric; repeat for multiple gates, e.g. rmse=1.2.",
    )
    check_parser.add_argument(
        "--comparison-mode",
        choices=("auto", "classification", "regression"),
        default="auto",
    )
    check_parser.add_argument("--atol", type=float, default=1e-8)
    check_parser.add_argument("--rtol", type=float, default=1e-5)
    check_parser.add_argument("--max-changed-fraction", type=float, default=0.0)
    check_parser.add_argument(
        "--metadata-json",
        default="{}",
        help="JSON object to attach as report metadata.",
    )
    check_parser.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Also write the complete JSON report to this file.",
    )
    check_parser.add_argument(
        "--cases-output",
        type=Path,
        metavar="FILE",
        help="Save captured failing inputs to an explicit JSON regression-case artifact.",
    )
    check_parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit successfully even if one or more checks fail.",
    )
    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay saved regression cases against a model.",
        description=(
            "Replay saved failing inputs. Exit 0 when every issue is resolved, "
            "1 when an issue is still reproduced, and 2 for setup errors."
        ),
    )
    replay_parser.add_argument(
        "--cases",
        required=True,
        type=Path,
        metavar="FILE",
        help="Regression-case JSON file produced by --cases-output.",
    )
    replay_parser.add_argument(
        "--model",
        help="Trusted module:attribute resolving to a model or zero-argument factory; "
        "not required for distribution-shift-only case files.",
    )
    replay_parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory to add to Python's import path; may be supplied more than once.",
    )
    replay_parser.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Also write the replay report to this file.",
    )
    return parser


def _run_checks(args: argparse.Namespace) -> int:
    checks = _parse_checks(args.checks)
    has_thresholds = bool(args.minimum_metric or args.maximum_metric)
    if (args.targets is None) != (args.task is None):
        raise ValueError("--targets and --task must be provided together.")
    if has_thresholds and args.targets is None:
        raise ValueError("Metric thresholds require both --targets and --task.")

    minimum_metrics = _parse_metric_thresholds(
        args.minimum_metric, option_name="--minimum-metric"
    )
    maximum_metrics = _parse_metric_thresholds(
        args.maximum_metric, option_name="--maximum-metric"
    )
    for search_path in reversed(args.model_path):
        path = str(Path(search_path).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)

    X = _load_numeric_csv(args.input, skip_header=args.skip_header)
    model = _model_from_spec(args.model)
    guard = ModelGuard(model)

    reference_X = None
    if "distribution_shift" in checks:
        if args.reference_input is None:
            raise ValueError(
                "--reference-input is required when distribution_shift is selected."
            )
        reference_X = _load_numeric_csv(
            args.reference_input,
            skip_header=args.skip_header,
        )

    baseline_model = None
    if "model_comparison" in checks:
        if not args.baseline_model:
            raise ValueError(
                "--baseline-model is required when model_comparison is selected."
            )
        baseline_model = _model_from_spec(args.baseline_model)

    try:
        metadata = json.loads(args.metadata_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--metadata-json is not valid JSON: {exc.msg}.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("--metadata-json must decode to a JSON object.")

    common_options = {
        "checks": checks,
        "noise_scale": args.noise_scale,
        "random_state": args.random_state,
        "stability_repeats": args.stability_repeats,
        "metadata": metadata,
        "reference_X": reference_X,
        "shift_threshold": args.shift_threshold,
        "variance_ratio_threshold": args.variance_ratio_threshold,
        "shape_alpha": args.shape_alpha,
        "missing_fraction": args.missing_fraction,
        "baseline_model": baseline_model,
        "comparison_mode": args.comparison_mode,
        "atol": args.atol,
        "rtol": args.rtol,
        "max_changed_fraction": args.max_changed_fraction,
    }
    if args.targets is not None:
        y_true = _load_numeric_targets(args.targets, skip_header=args.skip_header)
        report: SuiteReport | ValidationReport = guard.validate(
            X,
            y_true,
            task=args.task,
            average=args.average,
            minimum_metrics=minimum_metrics,
            maximum_metrics=maximum_metrics,
            **common_options,
        )
        failed = not report.reliability_passed or report.release_passed is False
    else:
        report = guard.run(X, **common_options)
        failed = not report.passed

    if args.cases_output is not None:
        save_regression_cases(report.failure_cases, args.cases_output)

    rendered = json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

    if failed and not args.allow_failures:
        return 1
    return 0

def _run_replay(args: argparse.Namespace) -> int:
    """Replay a saved case artifact against the requested model."""
    cases = load_regression_cases(args.cases)
    for search_path in reversed(args.model_path):
        path = str(Path(search_path).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    model = _model_from_spec(args.model) if args.model else None
    report = replay_regression_cases(model, cases)
    rendered = json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    """Run the MLGuard CLI and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-checks":
        for name, description in CHECK_DESCRIPTIONS.items():
            print(f"{name}: {description}")
        return 0

    try:
        if args.command == "replay":
            return _run_replay(args)
        return _run_checks(args)
    except Exception as exc:
        print(f"mlguard: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
