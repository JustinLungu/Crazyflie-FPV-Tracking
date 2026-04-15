#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Demo: autonomous drone following using live depth + takeover-safe teleop.
# Configure behavior in demos/drone_follower/constants.py.
run_repo_python "demos/drone_follower/run_demo.py"
