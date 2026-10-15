# Controlled failure evaluation

This repository includes `examples/fault_injection_demo.py`, a small reproducible
smoke evaluation of five known-failure scenarios. It uses synthetic arrays and
toy estimators only; it does not download a dataset, train a model, or call a
remote API.

Run it after installing the project in editable mode:

```bash
python -m pip install -e .
python examples/fault_injection_demo.py
```

The script exits with code 0 only if each intentionally introduced issue is
flagged: intermittent predictions, sensitivity to seeded Gaussian noise,
variance-only feature shift, rejection of missing values, and a baseline versus
candidate prediction change. The JSON output records each result's status,
metrics, and finding codes. The test workflow runs this script as a smoke gate.

## How to interpret it

This is a fault-injection smoke evaluation, not a broad benchmark. The scenarios
are hand-constructed and cover only a narrow set of behaviors. They do not
estimate general detection rate, specificity, false-positive rate, runtime at
scale, or effectiveness on production datasets. A passing result means the
current implementation caught these particular injected failures; it does not
prove that MLGuard can detect every kind of model degradation.

Before making broad reliability claims, extend this evaluation with:
- more than one model family and dataset size;
- several perturbation magnitudes and random seeds;
- benign distribution changes to assess false positives;
- distribution-shape changes with matching means and variances;
- repeated runs with summarized detection rate, false-positive rate, and runtime.

The variance-shift check intentionally measures per-feature mean and sample
variance changes. It is a descriptive signal rather than a statistical
hypothesis test and may miss shape-only changes. See
[Competitive positioning](COMPETITIVE_POSITIONING.md) for the scope of the
current design relative to established tools.


## Repeated synthetic distribution-shift experiment

`examples/distribution_shift_experiment.py` runs a seeded, repeated experiment
for four cases: two samples from the same distribution, an injected mean shift,
an injected variance change, and a shape change with matching population mean
and variance. The default is 100 trials with 512 rows per sample:

```bash
python examples/distribution_shift_experiment.py
```

The report gives observed false-positive and detection rates for these specific
synthetic settings. Its smoke gate requires at least 95% detection of the
configured mean and variance shifts and no more than 10% flags for same-
distribution controls. These are deliberately simple checks to prevent gross
regressions, not universal acceptance criteria or statistically precise
performance estimates.

The shape-only case is included to make an important limitation visible: a
screen based on means and variances can miss a distribution change that preserves
both. A real evaluation should include more distributions, thresholds, seeds,
sample sizes, and model families before making broad reliability claims.
