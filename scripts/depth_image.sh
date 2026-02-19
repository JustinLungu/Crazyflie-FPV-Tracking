#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Run UniDepth v2 on a single image using depth_estimation/constants.py.
run_repo_python "depth_estimation/depth_image_inference.py"
