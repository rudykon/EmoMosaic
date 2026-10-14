# Prediction, training and explanations

See the [project README](../../../README.md) for setup and resource downloads.
The deployed predictor uses five RM-Net seeds, five BERT LoRA adapters, and frozen
availability-gated fusion rules. These assets are restored from Hugging Face into
`code/models/`; prepared caches and historical experiment outputs are not bundled.

From the project root:

```bash
make predict
.venv/bin/python code/src/p2p3/prepare.py --device cpu
make evaluate
make test
```

`run_p3_fusion.py --stage all --skip-figures` computes numerical explanations in
a fresh output directory. It also requires the separately installed alignment
environment (`COLING_ALIGNMENT_PYTHON` or `.venv-p1/bin/python`) when word-level
alignment must be rebuilt. It does not generate manuscript sections.

Candidate search, historical verification and some plotting commands require
additional local experiment outputs. The original held-out split was inspected
during development and is not an independent blind test.
