# Regression cases: save, replay, and prevent repeat failures

MLGuard captures compact input cases from checks that flag a reliability
problem. Each case includes the check/scenario, a reproducible input (and
reference input when relevant), original expected/observed predictions when
available, check parameters, and a stable case ID. At most 100 cases are
captured per check run; result metrics indicate whether cases were truncated.

## Python workflow

```python
from mlguard import (
    ModelGuard,
    load_regression_cases,
    replay_regression_cases,
    save_regression_cases,
)

report = ModelGuard(model).run(
    X_test,
    checks=("noise_robustness", "missing_value_robustness"),
    noise_scale=0.02,
    random_state=42,
)

# Write only the captured cases, not the whole input dataset.
save_regression_cases(report.failure_cases, "artifacts/mlguard-cases.json")

cases = load_regression_cases("artifacts/mlguard-cases.json")
replay = replay_regression_cases(model, cases)
print(replay.to_dict())
```

A replay result is `reproduced` when the issue still occurs, `resolved` when
the saved condition no longer triggers the issue, or `error` when MLGuard cannot
replay the artifact. The aggregate report passes only when every saved issue
is resolved. For model-comparison cases, replay checks predictions against the
stored baseline prediction. For distribution-shift cases, it re-runs the shift
screen against the saved reference and candidate values.

## Command-line workflow

Capture cases from an ordinary check command:

```bash
mlguard check --model my_models:load_model --model-path . \
  --input features.csv --checks noise_robustness \
  --noise-scale 0.02 --random-state 42 \
  --cases-output artifacts/mlguard-cases.json
```

Replay the artifact against the current model:

```bash
mlguard replay --model my_models:load_model --model-path . \
  --cases artifacts/mlguard-cases.json \
  --output artifacts/replay-report.json
```

The replay command exits with code 0 when all saved issues are resolved, code 1
when one or more issues are still reproduced (or a case cannot be replayed), and
code 2 for setup or artifact errors. If an artifact contains only
`distribution_shift` cases, `--model` is not required.

## Privacy and limitations

A case artifact intentionally contains input values needed to reproduce a
failure. Review it for personal, proprietary, or regulated information before
committing it to a repository or uploading it as a CI artifact. Most cases
contain one sample; model-comparison failures that depend on full-batch errors
or prediction shapes preserve the original input batch, so those artifacts can
be much larger and more sensitive. MLGuard does not put raw inputs in ordinary
JSON reports by default; case contents are included only when explicitly
requested or saved with `save_regression_cases` / `--cases-output`.

Captured cases are capped at 100 per check to keep artifacts bounded. If an
input object cannot be represented safely as JSON, its case is skipped rather
than causing the original reliability check to fail. Replay preserves DataFrame
column labels; DataFrame cases require pandas in the replay environment. A
saved case preserves triggering inputs, but stochastic models or
environment-dependent inference may still produce different outcomes.


## Keep a persistent regression suite in CI

MLGuard includes a small, synthetic fixture at
tests/fixtures/regression_cases.json. The automated workflow replays this
artifact against examples.regression_suite_model:load_model on every change.
It confirms saved cases can be loaded and replayed through the public CLI, and
that the recorded baseline prediction still matches. If the fixture model is
changed so the old prediction returns, the replay command exits nonzero and CI
fails.

For your own project, capture cases from a real failure with --cases-output and
commit only sanitized, non-sensitive examples that you intend to keep as
long-lived regression tests. Point the replay command at a trusted model
factory in your repository. Remember that saved case files include input values.
