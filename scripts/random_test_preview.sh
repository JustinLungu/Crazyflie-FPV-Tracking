#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Sample random split images, run YOLO prediction, and save a grid JPG.
run_repo_python "models/random_test_preview.py" "$@"
