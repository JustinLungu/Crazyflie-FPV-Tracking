#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Run live depth estimation with selectable methods.
# Examples:
#   ./scripts/live_depth.sh --methods naive
#   ./scripts/live_depth.sh --methods unidepth
#   ./scripts/live_depth.sh --methods midas
#   ./scripts/live_depth.sh --methods naive,unidepth
run_repo_python "depth_estimation/live_depth_estimation.py" "$@"
