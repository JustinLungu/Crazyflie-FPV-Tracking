#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Record video into a new raw_data video session.
run_repo_python "data/videos_get_data.py" "$@"
