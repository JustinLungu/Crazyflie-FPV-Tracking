#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Run checkerboard camera calibration.
run_repo_python "depth_estimation/camera_calibration/calibration.py" "$@"
