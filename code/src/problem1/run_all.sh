#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT_ROOT="$(dirname "$CODE_ROOT")"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Usage: bash code/src/problem1/run_all.sh [--force]\nFull 100-sample extraction, export, verification, and QC.\nSet P1_PYTHON to override the Python interpreter.\n'
  exit 0
fi
if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--force") ]]; then
  printf 'Only --force is accepted; use run.py directly for partial or custom-output runs.\n' >&2
  exit 2
fi
cd "$CODE_ROOT"
export MPLCONFIGDIR="$PWD/.cache/p1-matplotlib"
export NUMBA_CACHE_DIR="$PWD/.cache/p1-numba"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
if [[ -n "${P1_PYTHON:-}" ]]; then
  PYTHON="$P1_PYTHON"
elif [[ -x "$PROJECT_ROOT/.venv-p1/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv-p1/bin/python"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON=python3
fi
"$PYTHON" src/problem1/run.py "$@"
"$PYTHON" src/problem1/export.py
"$PYTHON" src/problem1/verify.py
"$PYTHON" src/problem1/audit_official.py
"$PYTHON" src/problem1/analyze.py
