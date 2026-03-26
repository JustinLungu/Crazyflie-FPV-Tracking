#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Run naive bbox-based depth estimation.
# Usage:
#   ./scripts/naive_bbox_depth.sh                # default image mode
#   ./scripts/naive_bbox_depth.sh <image_path>   # custom image mode
#   ./scripts/naive_bbox_depth.sh --live         # live camera mode
run_repo_python "depth_estimation/naive_bbox_depth/bbox_dist_estimator.py" "$@"
