#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Interactive label-review player (pause/back/forward/delete).
run_repo_python "data/view_labeling.py" "$@"
