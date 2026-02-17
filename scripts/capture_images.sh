#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Capture still frames into a new raw_data image session.
run_repo_python "data/images_get_data.py" "$@"
