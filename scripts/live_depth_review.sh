#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Constants-driven live depth review (top-level depth_estimation).
# Configure methods/camera/UI in depth_estimation/constants.py.
run_repo_python "depth_estimation/live_depth_review.py"
