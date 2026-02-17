#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Merge labeled sessions into one class dataset.
# Optional args are forwarded to create_dataset.py (e.g. --sessions ... --overwrite).
run_repo_python "data/create_dataset.py" "$@"
