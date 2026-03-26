#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Run MiDaS on a single image using depth_estimation/midas/constants.py.
run_repo_python "depth_estimation/midas/depth_image_inference.py"
