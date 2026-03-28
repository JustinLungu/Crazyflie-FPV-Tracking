#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Run drone_control + YOLO live inference concurrently.
run_repo_python "flight_vision/main.py"
