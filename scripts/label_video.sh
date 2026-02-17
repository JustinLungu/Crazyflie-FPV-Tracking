#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Tracker-assisted labeling from VIDEO_PATH in data/constants.py.
run_repo_python "data/track_label_video.py" "$@"
