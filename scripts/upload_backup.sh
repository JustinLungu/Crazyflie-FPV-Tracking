#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Zip raw_data + labels and upload to Google Drive folder from .env.
run_repo_python "data/upload_data_drive.py" "$@"
