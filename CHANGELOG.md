# Changelog

Notable changes to MLGuard are recorded here.

## [0.1.0] — 2026-10-15

First public alpha scope:

- Added five model reliability checks: repeated-call prediction stability,
  Gaussian-noise robustness, mean/variance distribution-shift screening,
  missing-value robustness, and baseline-versus-candidate prediction comparison.
- Made prediction stability configurable (three calls by default) to catch
  intermittent changes that may not appear across only two calls.
- Added a controlled fault-injection smoke demo and a repeated synthetic
  distribution-shift experiment, including same-distribution controls and an
  explicit shape-only blind-spot scenario. These are not substitutes for broad
  external benchmarks.
- Added a documented competitive positioning note based on current feature scope.
- Added classification metrics (accuracy, precision, recall, F1, confusion
  matrix, and per-class metrics) and regression metrics (MAE, MSE, RMSE,
  R-squared, and maximum absolute error).
- Added combined validation reports and optional metric-based release gates.
- Added a local CLI for executing reliability checks against numeric CSV data
  and producing JSON reports.
- Added optional scikit-learn estimator and Pipeline integration.
- Added automated tests on Python 3.10–3.13.

Alpha limitations:

- Public APIs may change before a stable 1.0 release.
- CLI features and targets are currently numeric CSV inputs; classification is
  single-label, and regression supports a single numeric target.
- Perturbation checks for noise and missing values require numeric features.
- Distribution-shift screening compares feature means and sample variances;
  it is descriptive, not a statistical hypothesis test, and can miss changes
  in distribution shape that preserve those statistics.
- GitHub Release distribution is prepared but has not been published.
- The PyPI distribution name is provisionally set to `mlguard-reliability`;
  its availability must be confirmed before any PyPI upload.
