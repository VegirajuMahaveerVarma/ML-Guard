# MLGuard

**Local-first reliability checks for machine-learning models.**

MLGuard is an early-stage Python library for testing whether model predictions remain stable and robust under controlled input changes. Its core functionality runs locally; it does not upload models, datasets, or reports.

> **Project status:** Experimental alpha (pre-release). The current implementation includes five checks: repeated-prediction stability, seeded Gaussian-noise robustness, feature-distribution shift screening, missing-value handling, and baseline-versus-candidate model comparison. APIs and reports may evolve; validate regression replay and evaluation behavior for your model and data before using MLGuard as a release gate.

## Features

- Framework-light model interface: any object with a `predict(X)` method can be checked.
- Optional scikit-learn integration for fitted estimators and Pipelines.
- Pandas DataFrame support for named numeric features in perturbation checks.
- Prediction stability checks across three model calls by default, configurable to catch intermittent changes.
- Gaussian-noise robustness checks with reproducible random seeds.
- Reference-versus-candidate distribution screening using standardized mean differences, variance ratios, and empirical-CDF shape distances.
- Missing-value robustness checks that report whether a model rejects NaNs or emits invalid predictions.
- Baseline-versus-candidate prediction comparisons for model updates.
- Dataset-level classification and regression metrics against ground-truth targets.
- Structured report objects with opt-in, reproducible failing-input case capture.
- Save, load, and replay regression cases to check whether previously detected issues remain.
- Local execution; no telemetry or network calls in the library.
- Command-line interface for running checks against local numeric CSV files.

## First public alpha release

MLGuard is in alpha. A GitHub Release is planned, but no public release has been created. After
the `v0.1.0` release is published, download the wheel from the release page
and install it locally:

```bash
python -m pip install ./mlguard_reliability-0.1.0-py3-none-any.whl
mlguard --version
```

The wheel filename above is the expected artifact name for version `0.1.0`
with the proposed distribution name `mlguard-reliability`; the import name
and CLI command remain `mlguard`. This distribution name is not reserved until
a first upload. Confirm its availability on PyPI immediately before publishing;
if unavailable, update metadata and the release filenames before tagging.
No PyPI package has been published. See [CHANGELOG.md](CHANGELOG.md) for the
alpha scope and [docs/RELEASING.md](docs/RELEASING.md) for release steps.

## Install for development

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install -e ".[dev]"
```

For scikit-learn integration, install the optional dependency:

```bash
python -m pip install -e ".[sklearn]"
```

Use `ModelGuard.from_sklearn` for a fitted scikit-learn estimator or Pipeline.
It checks that the estimator is fitted before running and preserves DataFrame
columns for numeric noise and missing-value perturbations, which is important
for Pipelines that select features by column name.

```python
from mlguard import ModelGuard

# pipeline is an already-fitted sklearn Pipeline
guard = ModelGuard.from_sklearn(pipeline)
report = guard.run(
    X_test,
    checks=("prediction_stability", "noise_robustness"),
    noise_scale=0.001,
    random_state=42,
)
print(report.to_dict())
```

MLGuard's noise and missing-value checks currently require numeric features.
Prediction stability and model comparison can also be used with other
scikit-learn-compatible input containers supported by the estimator.

## Quick start

```python
import numpy as np
from mlguard import ModelGuard


class ThresholdModel:
    def predict(self, X):
        X = np.asarray(X)
        return (X[:, 0] > 0.5).astype(int)


X = np.array([[0.1, 0.2], [0.6, 0.3], [0.8, 0.9]])
guard = ModelGuard(ThresholdModel())

report = guard.run(
    X,
    checks=("prediction_stability", "noise_robustness"),
    noise_scale=0.01,
    random_state=42,
    metadata={"model_version": "local-demo"},
)

print(report.to_dict())
```

A result with status `failed` means a check detected a change relative to its test criterion; it does not automatically mean the model is unusable. Interpret each metric in the context of your application.

## Available checks

| Check | What it measures |
|---|---|
| `prediction_stability` | Agreement across repeated predictions on identical input (3 calls by default) |
| `noise_robustness` | Agreement between baseline predictions and predictions after seeded Gaussian noise |
| `distribution_shift` | Mean, variance, and empirical-CDF shape changes between reference and candidate numeric features |
| `missing_value_robustness` | Whether a model accepts missing feature values and returns valid predictions |
| `model_comparison` | Whether a candidate model's predictions differ from a baseline model |

Prediction stability runs the model three times by default; set `stability_repeats` in Python or `--stability-repeats` in the CLI to require more or fewer repeated calls (minimum 2). More repeats can catch intermittent variation but increase inference cost.

Distribution shift screens feature means, sample variances, and empirical-CDF shape differences. The default variance ratio threshold is 2.0: a feature is flagged when one sample variance is more than twice the other. A zero-versus-nonzero variance is treated as an unbounded ratio. The shape screen uses a conservative two-sample DKW-style threshold that depends on the reference and candidate sample sizes, with a Bonferroni correction across features. Its default family-wise false-positive budget is `shape_alpha=0.05`; tune it through the Python API or `--shape-alpha` in the CLI. This is a descriptive reliability screen, not a returned p-value or a guarantee of detecting subtle changes. Small samples can leave it underpowered.

Noise is applied to a numeric copy of the input; the caller's original array is not intentionally mutated. `noise_scale` is the absolute standard deviation of the added noise, not a feature-normalized percentage. The `missing_value_robustness` check also requires a numeric 2D feature matrix and injects NaNs into a copy. Run it explicitly with `checks=("missing_value_robustness",)` and optionally set `missing_fraction` (default `0.05`).

### Compare a candidate model with a baseline

Pass the candidate model to `ModelGuard` and provide the existing model as
`baseline_model`. By default, changed predictions fail the check. Use
`max_changed_fraction` when a known amount of change is acceptable.

```python
from mlguard import ModelGuard

report = ModelGuard(candidate_model).run(
    X,
    checks=("model_comparison",),
    baseline_model=baseline_model,
    comparison_mode="auto",
    atol=1e-8,
    rtol=1e-5,
    max_changed_fraction=0.0,
)
print(report.to_dict())
```

In auto mode, floating-point numeric outputs are compared using numeric
tolerances; integer, boolean, and label-like outputs are compared exactly.
Set `comparison_mode="classification"` for classifiers that return
floating-point class labels, or `comparison_mode="regression"` for numeric
regression outputs. Numeric differences are descriptive and do not directly
measure model quality or accuracy.

## Dataset performance evaluation

Reliability checks inspect model behavior; performance evaluation compares
predictions with known ground-truth targets. MLGuard provides both without
requiring scikit-learn for metric computation.

For classification, the report includes accuracy, precision, recall, F1,
a confusion matrix, and per-class support and scores. The default averaging
mode is macro; use `weighted` or `micro` when appropriate.

```python
from mlguard import ModelGuard

# model can be any fitted object exposing predict(X)
report = ModelGuard(model).evaluate(
    X_test,
    y_test,
    task="classification",
    average="macro",
    metadata={"dataset": "validation"},
)
print(report.to_dict())
```

For regression, evaluation reports MAE, MSE, RMSE, R-squared, and maximum
absolute error:

```python
report = ModelGuard(model).evaluate(
    X_test,
    y_test,
    task="regression",
    metadata={"dataset": "holdout"},
)
print(report.to_dict()["metrics"])
```

Evaluation inputs must contain one target per sample. Regression currently
supports a single numeric target, and classification supports single-label
binary or multiclass targets. R-squared is reported as `null` when fewer
than two regression samples are supplied. Evaluation reports contain aggregate
metrics and metadata, not raw feature rows or targets.


## Combined validation workflow

Use the high-level validation method to run reliability checks and calculate
dataset performance metrics through one entry point. The returned
ValidationReport keeps its reliability and evaluation sections separate.

    from mlguard import ModelGuard

    report = ModelGuard(model).validate(
        X_test,
        y_test,
        task="classification",
        checks=("prediction_stability", "noise_robustness"),
        noise_scale=0.001,
        random_state=42,
        metadata={"model_version": "v2", "dataset": "holdout"},
    )

    print(report.reliability_passed)
    print(report.evaluation.metrics["accuracy"])
    print(report.to_dict())

The reliability_passed property is true only when every selected reliability
check passes. Performance metrics are intentionally separate: MLGuard does
not impose universal accuracy or error thresholds because acceptable quality
depends on the task and deployment context. Set application-specific metric
thresholds in your own evaluation or release process. The combined report
contains aggregate metrics and metadata, not raw samples.

The checks and their options match ModelGuard.run. For example,
distribution_shift requires reference_X, and model_comparison requires
baseline_model.

## Metric thresholds and release gates

Set optional metric limits in `ModelGuard.validate` to turn evaluation scores
into explicit release criteria. Minimum limits use `>=` (for example accuracy
or F1); maximum limits use `<=` (for example MAE or RMSE).

```python
report = ModelGuard(model).validate(
    X_test,
    y_test,
    task="classification",
    minimum_metrics={"accuracy": 0.90, "f1": 0.85},
    maximum_metrics={"recall": 0.95},
)

print(report.performance_passed)  # None when no metric gates were configured
print(report.release_passed)      # True/False when metric gates are configured
print(report.to_dict()["metric_gates"])
```

Only top-level numeric metrics for the selected task can be thresholded:
classification supports accuracy, precision, recall, and F1; regression supports
MAE, MSE, RMSE, R-squared, and maximum absolute error. Each configured threshold
appears in the report with the observed value and pass/fail result. If a metric
cannot be computed (for example, R-squared with fewer than two samples), its
gate fails rather than silently passing. Without any configured metric limits,
`performance_passed` and `release_passed` are `null`; the report does not
imply that an unconfigured performance gate passed.

## Command-line interface

Install MLGuard in the environment used for your model code:

```bash
python -m pip install -e .
mlguard --version
mlguard list-checks
```

The CLI loads your model from a trusted local Python import reference in
`module:attribute` form. The referenced attribute may be a model instance, a
class with a zero-argument constructor, or a zero-argument factory that returns
an object exposing `predict(X)`. Add the directory containing your model module
with `--model-path`.

For example, if `my_models.py` provides a `load_model()` function and
`features.csv` contains only numeric feature columns (no target column):

```bash
mlguard check --model my_models:load_model --model-path . --input features.csv --checks prediction_stability,noise_robustness --output report.json

# Combined validation: numeric target CSV plus explicit release thresholds
mlguard check --model my_models:load_model --model-path . --input features.csv --targets targets.csv --task classification --checks prediction_stability,noise_robustness --minimum-metric accuracy=0.90 --minimum-metric f1=0.85 --output validation-report.json
```

To compare feature distributions, include `--reference-input reference.csv`
and select `distribution_shift`. To compare a candidate model against an
existing model, use `--baseline-model package:load_baseline` with
`model_comparison`. For Python code, wrap an existing fitted baseline with
`SklearnAdapter` and pass it as `baseline_model` to `ModelGuard.from_sklearn(candidate).run(...)`. Use `--skip-header` for CSV files whose first row is a
header. The report is printed as JSON and can also be written with `--output`.

The CLI accepts `--targets targets.csv --task classification|regression` to
run combined validation. The target CSV must contain one numeric column; use
`--minimum-metric NAME=VALUE` and `--maximum-metric NAME=VALUE` (repeatable)
to configure performance gates. The CLI exits with code 1 when selected
reliability checks or configured metric gates fail, code 2 for setup or input
errors, and code 0 when all configured checks pass. Add `--allow-failures` to make failed checks
non-fatal in automation. Model references import and execute local Python code:
only point MLGuard at trusted modules and model artifacts. Input CSV files must
contain finite numeric feature values; missing-value testing introduces NaNs
into a copy of those features.

## Save and replay regression cases

When a reliability check flags a failure, MLGuard captures up to 100 compact
cases per check in memory. Each case preserves the specific input, relevant
reference input, check parameters, and expected/observed predictions when
available. Reports include the case count but do not expose input values by
default. To explicitly save cases:

```python
from mlguard import ModelGuard, load_regression_cases, replay_regression_cases, save_regression_cases

report = ModelGuard(model).run(
    X_test,
    checks=("noise_robustness", "missing_value_robustness"),
    noise_scale=0.02,
    random_state=42,
)
save_regression_cases(report.failure_cases, "mlguard-cases.json")

cases = load_regression_cases("mlguard-cases.json")
replay = replay_regression_cases(model, cases)
print(replay.to_dict())
```

The CLI supports `--cases-output FILE` on `mlguard check`, then
`mlguard replay --model my_models:load_model --cases FILE` to recheck the
captured cases. Replay exits successfully when every saved issue is resolved.
Case files contain real input values, so treat them as potentially sensitive
data; see [Regression cases](docs/REGRESSION_CASES.md) for the full workflow.

## Run tests

```bash
python -m pytest
```

## Controlled failure demo

Run a local smoke evaluation using toy models with deliberately injected failures:

```bash
python examples/fault_injection_demo.py
```

The script checks five scenarios and exits nonzero if an expected failure is not
detected. It uses no remote APIs. Run the repeated synthetic distribution-shift
experiment as well:

```bash
python examples/distribution_shift_experiment.py
```

The experiment reports detection and false-positive rates across multiple sample
sizes, mean shifts, and standard-deviation multipliers for variance changes. It
also evaluates a shape-only change using empirical-CDF distances; sensitivity
depends on sample size and effect magnitude. Adjust the sweep with
--additional-sample-sizes, --mean-shifts, and --variance-scales (standard-deviation
multipliers). These are controlled synthetic checks, not production benchmarks
or proof of general detection accuracy. See
[controlled evaluation](docs/CONTROLLED_EVALUATION.md) and
[competitive positioning](docs/COMPETITIVE_POSITIONING.md).


## Roadmap

- Expand controlled evaluation across more models, dataset sizes, and random seeds.
- Evaluate false-positive behavior on benign and shape-only distribution changes.
- Add further framework adapters only where they improve real developer workflows.

## Privacy

MLGuard is designed to run locally. By default, result reports include metrics, findings, and user-provided metadata—not raw samples. Treat metadata as potentially sensitive and avoid placing secrets or personal data in it.

## License

MIT. See [LICENSE](LICENSE).

## Maintainer

Created and maintained by Mahaveer Varma.
