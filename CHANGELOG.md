# Changelog

Notable changes to MLGuard are recorded here.

## [0.1.1] — 2026-10-24

- Improved the optional scikit-learn adapter's missing-dependency error message with a direct installation command.
- Added regression coverage to verify that the adapter provides the correct installation guidance when scikit-learn is unavailable.

## [0.1.0] — 2026-10-15

First public alpha scope:

- Added five model reliability checks: repeated-call prediction stability,
  Gaussian-noise robustness, mean/variance/empirical-CDF distribution-shift screening,
  missing-value robustness, and baseline-versus-candidate prediction comparison.
- Made prediction stability configurable (three calls by default) to catch
  intermittent changes that may not appear across only two calls.
- Added a controlled fault-injection smoke demo and a repeated synthetic
  distribution-shift experiment, including same-distribution controls and
  empirical-CDF shape-change scenarios. These are not substitutes for broad
  external benchmarks.
- Added a documented competitive positioning note based on current feature scope.
- Added classification metrics (accuracy, precision, recall, F1, confusion
  matrix, and per-class metrics) and regression metrics (MAE, MSE, RMSE,
  R-squared, and maximum absolute error).
- Added combined validation reports and optional metric-based release gates.
- Added a local CLI for executing reliability checks against numeric CSV data
  and producing JSON reports.
- Added opt-in failure-case capture, strict JSON save/load, and replay reports
  so developers can track whether previously detected reliability issues remain.
- Added optional scikit-learn estimator and Pipeline integration.
- Added automated tests on Python 3.10–3.13.

Alpha limitations:

- Public APIs may change before a stable 1.0 release.
- CLI features and targets are currently numeric CSV inputs; classification is
  single-label, and regression supports a single numeric target.
- Perturbation checks for noise and missing values require numeric features.
- Captured regression cases are bounded to 100 per check and contain real input
  values when explicitly saved; review them for sensitive data before sharing.
- Distribution-shift screening adds empirical-CDF shape distances with a
  conservative sample-size-aware threshold and a feature-wise Bonferroni
  correction; subtle changes may still be missed, especially with small samples.
- GitHub Release distribution is prepared but has not been published.
- The PyPI distribution name is provisionally set to `mlguard-reliability`;
  its availability must be confirmed before any PyPI upload.
