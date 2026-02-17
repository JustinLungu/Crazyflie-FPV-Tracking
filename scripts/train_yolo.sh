#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Train YOLO model from models/constants.py.
run_repo_python "models/train_yolo.py" "$@"
