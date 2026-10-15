# Competitive positioning

**Status:** internal planning note, reviewed 15 October 2026. This is a feature-
scope comparison based on official project documentation, not an independent
performance benchmark.

## Existing tools

- [Deepchecks](https://deepchecks.com/open-source/) describes a broad open-source
  ML validation offering for data integrity, model evaluation, train/test
  validation, data drift, and CI/CD workflows. Its
  [documentation](https://docs.deepchecks.com/stable/) also covers testing and
  monitoring through the model lifecycle.
- [Evidently](https://docs.evidentlyai.com/) provides data and ML checks including
  prediction quality, input data quality, and data/prediction drift. Its
  [ML quickstart](https://github.com/evidentlyai/docs/blob/main/quickstart_ml.mdx)
  demonstrates local Python reports and structured outputs.
- [NannyML](https://github.com/NannyML/nannyml) focuses on post-deployment model
  performance estimation when targets are delayed or unavailable, drift
  detection, and linking drift alerts to changes in estimated performance.

These are established projects with broader monitoring, statistical methods,
reporting, and/or performance-estimation capabilities than MLGuard currently
implements. MLGuard should not claim feature parity or superior detection
quality.

## A narrower potential position for MLGuard

MLGuard currently emphasizes a small Python API around any model exposing
`predict(X)`, local perturbation checks, explicit baseline/candidate comparisons,
opt-in performance thresholds, and JSON-friendly results. The core dependency
is NumPy; scikit-learn support is optional. No hosted service or external API is
required for the implemented checks.

Those are product design choices, not proof that MLGuard is uniquely local or
faster than the alternatives. The project should test whether this lightweight
approach reduces setup friction for developers who want a small, scriptable
pre-merge reliability gate rather than a full validation or monitoring system.

## Evidence required before stronger claims

A future comparison should use the same synthetic and public datasets for all
tools where comparable interfaces exist, and document versions and setup.
Useful measurements include:

1. time and number of lines needed for a minimal model check;
2. installation footprint and direct dependency count;
3. runtime on fixed sample and feature sizes;
4. detection on a published suite of controlled failures;
5. false-positive behavior on benign changes and distribution-shape changes.

Until such measurements are recorded and independently reproducible, the
defensible positioning is **a small, local-first reliability-check library in
alpha**, not a claim of being more accurate or more comprehensive than the
established alternatives.
