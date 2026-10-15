# MLGuard

**Local-first reliability checks for machine-learning models.**

MLGuard is an early-stage Python library for testing whether model predictions remain stable and robust under controlled input changes. Its core functionality runs locally; it does not upload models, datasets, or reports.

> **Project status:** Alpha. The current implementation includes repeated-prediction stability, seeded Gaussian-noise robustness, and numeric feature-distribution shift screening, and missing-value behavior checks. APIs and reports may evolve.

## Features

- Framework-light model interface: any object with a `predict(X)` method can be checked.
- Optional scikit-learn adapter.
- Prediction stability checks using repeated calls.
- Gaussian-noise robustness checks with reproducible random seeds.
- Reference-versus-candidate feature distribution screening using standardized mean differences.
- Missing-value robustness checks that report whether a model rejects NaNs or emits invalid predictions.
- Structured report objects that can be converted to JSON-compatible dictionaries.
- Local execution; no telemetry or network calls in the library.

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

Noise is applied to a numeric copy of the input; the caller's original array is not intentionally mutated. `noise_scale` is the absolute standard deviation of the added noise, not a feature-normalized percentage. The `missing_value_robustness` check also requires a numeric 2D feature matrix and injects NaNs into a copy. Run it explicitly with `checks=("missing_value_robustness",)` and optionally set `missing_fraction` (default `0.05`).

## Run tests

```bash
python -m pytest
```

## Roadmap

- More robust input validation and typed configuration.
- Baseline-versus-candidate model regression checks.
- Additional metrics, documentation, and framework adapters.
- CLI and CI integrations after the core API stabilizes.

## Privacy

MLGuard is designed to run locally. By default, result reports include metrics, findings, and user-provided metadata—not raw samples. Treat metadata as potentially sensitive and avoid placing secrets or personal data in it.

## License

MIT. See [LICENSE](LICENSE).

## Maintainer

Created and maintained by Mahaveer Varma.
