#!/usr/bin/env bash
# Run every gate a module must pass before it is committed. Fails fast.
set -euo pipefail
cd "$(dirname "$0")"
ruff check .
python3 -m mypy
python3 -m pytest
