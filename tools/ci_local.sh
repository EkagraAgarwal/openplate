#!/usr/bin/env bash
# Same test selection as CI. Run from Git Bash on Windows, or any POSIX shell.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export NUMBA_NUM_THREADS=4
export MPLBACKEND=Agg
export PYVISTA_OFF_SCREEN=true

# Set CI_TEMP_ROOT if pytest's default temp directory is not writable on your machine.
temp_root="${CI_TEMP_ROOT:-${TMPDIR:-/tmp}/openplate-ci}"
mkdir -p "$temp_root"
base="$(mktemp -d "$temp_root/pytest-XXXXXXXX")"
uv run --project . --extra fast --extra ref --extra layout --extra viz --extra spec \
  --with pytest --with httpx python -m pytest -p no:cacheprovider \
  --basetemp="$base" -m "not slow and not gpu" --durations=10 -ra

cd app
if [[ ! -d node_modules ]]; then npm ci; fi
npm test
npx tsc --noEmit -p .
