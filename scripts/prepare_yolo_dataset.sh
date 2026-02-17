#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Build YOLO train/val/test split dataset from manifest/session data.
run_repo_python "data/prepare_yolo_dataset.py" "$@"
