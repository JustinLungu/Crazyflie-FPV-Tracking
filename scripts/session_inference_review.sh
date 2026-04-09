#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Run YOLO inference over one configured session folder with playback controls.
run_repo_python "inference/session_inference_review.py" "$@"
