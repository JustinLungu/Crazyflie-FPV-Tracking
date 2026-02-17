#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Live camera preview for receiver/device debugging.
run_repo_python "setting_up_camera/get_visual.py" "$@"
