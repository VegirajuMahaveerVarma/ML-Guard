# MLGuard

**Local-first reliability checks for machine-learning models.**

MLGuard is an early-stage Python library for testing whether model predictions remain stable and robust under controlled input changes. Its core functionality runs locally; it does not upload models, datasets, or reports.

> **Project status:** Alpha. The current implementation includes five checks: repeated-prediction stability, seeded Gaussian-noise robustness, feature-distribution shift screening, missing-value handling, and baseline-versus-candidate model comparison. APIs and reports may evolve.

## Features

- Framework-light model interface: any object with a `predict(X)` method can be checked.
- Optional scikit-learn integration for fitted estimators and Pipelines.
- Pandas DataFrame support for named numeric features in perturbation checks.
- Prediction stability checks using repeated calls.
- Gaussian-noise robustness checks with reproducible random seeds.
- Reference-versus-candidate feature distribution screening using standardized mean differences.
- Missing-value robustness checks that report whether a model rejects NaNs or emits invalid predictions.
- Baseline-versus-candidate prediction comparisons for model updates.
- Dataset-level classification and regression metrics against ground-truth targets.
- Structured report objects that can be converted to JSON-compatible dictionaries.
- Local execution; no telemetry or network calls in the library.
- Command-line interface for running checks against local numeric CSV files.

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
| `prediction_stability` | Agreement between two predictions on identical input |
| `noise_robustness` | Agreement between baseline predictions and predictions after seeded Gaussian noise |
| `distribution_shift` | Per-feature standardized mean differences between reference and candidate samples |
| `missing_value_robustness` | Whether a model accepts missing feature values and returns valid predictions |
| `model_comparison` | Whether a candidate model's predictions differ from a baseline model |

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
```

To compare feature distributions, include `--reference-input reference.csv`
and select `distribution_shift`. To compare a candidate model against an
existing model, use `--baseline-model package:load_baseline` with
`model_comparison`. For Python code, wrap an existing fitted baseline with
`SklearnAdapter` and pass it as `baseline_model` to `ModelGuard.from_sklearn(candidate).run(...)`. Use `--skip-header` for CSV files whose first row is a
header. The report is printed as JSON and can also be written with `--output`.

The CLI exits with code 1 when checks fail, code 2 for setup or input errors,
and code 0 when all checks pass. Add `--allow-failures` to make failed checks
non-fatal in automation. Model references import and execute local Python code:
only point MLGuard at trusted modules and model artifacts. Input CSV files must
contain finite numeric feature values; missing-value testing introduces NaNs
into a copy of those features.

## Run tests

```bash
python -m pytest
```

## Roadmap

- More robust input validation and typed configuration.
- Additional metrics, documentation, and framework adapters.
- Additional CI templates and richer reporting integrations.

## Privacy

MLGuard is designed to run locally. By default, result reports include metrics, findings, and user-provided metadata—not raw samples. Treat metadata as potentially sensitive and avoid placing secrets or personal data in it.

## License

MIT. See [LICENSE](LICENSE).

## Maintainer

Created and maintained by Mahaveer Varma.
