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
