# Release process

This repository currently targets a first **GitHub alpha release**. The release
workflow builds and validates the wheel and source distribution, runs the test
suite, smoke-tests the wheel install, computes SHA-256 checksums, and attaches
the artifacts to a GitHub prerelease.

## Release checklist

Before tagging:

1. Confirm the project version in `pyproject.toml` and `src/mlguard/__init__.py`
   agrees with the intended tag.
2. Update `CHANGELOG.md` and the release notes under `docs/releases/`.
3. Confirm the latest Tests workflow is green on `main`.
4. Pull the latest `main` locally and verify the working tree is clean.

For the initial alpha release, run from a clean clone after the release workflow
has landed on `main`:

```bash
git pull origin main
git status --short
git tag -a v0.1.0 -m "MLGuard v0.1.0 alpha"
git push origin v0.1.0
```

Pushing the tag starts `.github/workflows/release.yml`. It refuses to release
if the tag's version does not match `pyproject.toml`. If the build, lint, tests,
or wheel smoke test fail, no release should be considered ready. When the job
succeeds, the workflow creates a GitHub prerelease with wheel, source archive,
and checksums.

Do not re-use a published tag. If the workflow fails, fix the source and make
a new commit. For a replacement release, increment the version and tag a new
version.

## PyPI distribution

PyPI publication is intentionally not automated yet. The project metadata now
uses the proposed distribution name `mlguard-reliability`, while the import
package and CLI remain `mlguard`. PyPI name availability could not be
conclusively verified during this preparation pass, and a name is not reserved
until its first upload. Before any PyPI upload, check the exact project URL and
JSON endpoint, then update the wheel filename examples if a different name is
required. Configure a PyPI trusted publisher only after confirming the name and
deciding to distribute through PyPI. The current release workflow publishes
GitHub Release assets only.
