#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Evaluate chosen YOLO weights on val/test split.
run_repo_python "models/test_yolo.py" "$@"
