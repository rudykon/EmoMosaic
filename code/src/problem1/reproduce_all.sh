#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT_ROOT="$(dirname "$CODE_ROOT")"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Usage: bash code/src/problem1/reproduce_all.sh\nForce two complete raw-media extractions and compare all 500 feature/time files.\nSet P1_PYTHON to override the Python interpreter.\n'
  exit 0
fi
if [[ $# -ne 0 ]]; then
  printf 'This command accepts no options other than --help.\n' >&2
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
"$PYTHON" src/problem1/run.py --output outputs/problem1 --force
"$PYTHON" src/problem1/run.py --output outputs/problem1/validation/repro --force
"$PYTHON" src/problem1/reproduce.py
"$PYTHON" src/problem1/export.py
"$PYTHON" src/problem1/verify.py
