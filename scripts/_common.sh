#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

run_repo_python() {
    local entry="$1"
    shift || true

    cd "$REPO_ROOT"
    if command -v uv >/dev/null 2>&1; then
        uv run python "$entry" "$@"
    else
        python3 "$entry" "$@"
    fi
}
