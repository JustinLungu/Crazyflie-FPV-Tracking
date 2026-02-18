#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Run trained YOLO model on live drone receiver feed.
run_repo_python "inference/live_inference.py" "$@"
