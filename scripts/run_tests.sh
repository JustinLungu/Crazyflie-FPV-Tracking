#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Run repository system/integration tests.
cd "$REPO_ROOT"
if command -v uv >/dev/null 2>&1; then
    uv run python -m unittest discover -s tests -p "test_*.py" "$@"
else
    python3 -m unittest discover -s tests -p "test_*.py" "$@"
fi
