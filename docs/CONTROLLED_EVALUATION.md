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

The distribution-shift check measures per-feature mean and sample-variance
changes and also compares empirical cumulative distribution functions (CDFs).
For shape, it uses a conservative two-sample DKW-style distance threshold that
depends on both sample sizes; a Bonferroni correction allocates the configured
family-wise false-positive budget across features. The default `shape_alpha=0.05`
is not a returned p-value, and low sample sizes can produce thresholds too
conservative to detect even visible shape differences. See
[Competitive positioning](COMPETITIVE_POSITIONING.md) for the scope of the
current design relative to established tools.


## Repeated synthetic distribution-shift experiment

`examples/distribution_shift_experiment.py` runs a seeded, repeated experiment
for four cases: two samples from the same distribution, injected mean shifts,
injected variance changes, and a shape change with matching population mean and
variance. By default it evaluates 128, 512, and 2048 samples per distribution,
three mean-shift magnitudes, and three standard-deviation multipliers, with 100
trials per configuration. Each standard-deviation multiplier is squared to get
the generated distribution's variance multiplier:

```bash
python examples/distribution_shift_experiment.py
```

The JSON report gives false-positive and detection rates for each combination
of sample size and shift strength. Its smoke gate is evaluated at the canonical
sample size (512 by default): it requires at least 95% detection for a mean shift
of 0.75 standard deviations and a standard-deviation multiplier of 2.0 (four
times the variance), and no more than 10% flags for same-distribution controls.
These are deliberately simple checks to prevent gross regressions, not universal
acceptance criteria or statistically precise performance estimates.

The shape-only case exercises the empirical-CDF screen on a distribution change
that preserves mean and variance. Detection still depends on sample size and
effect magnitude, so a low shape-only detection rate is evidence of limited
power for this scenario rather than proof of no distribution change. The
experiment accepts --trials, --sample-size, --additional-sample-sizes,
--mean-shifts, --variance-scales (standard-deviation multipliers), and --seed
for reproducible parameter sweeps. It remains a synthetic, descriptive
evaluation; further work should add more distribution families and public
datasets before broad reliability claims.
