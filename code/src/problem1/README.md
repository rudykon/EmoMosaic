# Independent feature extraction and word-level alignment

This module extracts text, audio and visual features, word intervals, frame indices
and explicit quality masks from source media. Its outputs must not be substituted
for the official aligned features used by the primary predictor.

See the [project README](../../../README.md) for the separate `.venv-p1`
environment and the data/model download commands. No data or pretrained models
are bundled in GitHub.

From the repository root:

```bash
make align-smoke
bash code/src/problem1/run_all.sh
.venv-p1/bin/python -m unittest discover -s code/src/problem1 -p 'test_*.py'
```

The complete pipeline performs extraction, export, verification, official-feature
auditing and QC. It does not export a manuscript or report chapter. Reproduction
and feature-bundle utilities require outputs previously generated locally.
