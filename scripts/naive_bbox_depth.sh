#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Run naive bbox-based depth estimation.
# Configure mode and image path in depth_estimation/naive_bbox_depth/constants.py.
run_repo_python "depth_estimation/naive_bbox_depth/bbox_dist_estimator.py"
