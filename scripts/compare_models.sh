#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Compare multiple models on one split and print a summary table.
run_repo_python "models/compare_models.py" "$@"
