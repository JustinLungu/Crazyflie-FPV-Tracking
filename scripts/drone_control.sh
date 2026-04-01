#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Crazyflie teleoperation/autonomy entrypoint.
run_repo_python "drone_control/start_drone.py" "$@"
