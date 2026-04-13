#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Run naive YOLO+bbox-depth review using settings in constants.py.
run_repo_python "depth_estimation/naive_bbox_depth/session_depth_review.py"
