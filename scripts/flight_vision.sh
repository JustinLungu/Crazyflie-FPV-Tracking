#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Reduce noisy OpenCV/Qt font warnings if common system fonts are available.
if [[ -z "${QT_QPA_FONTDIR:-}" ]]; then
    for font_dir in \
        /usr/share/fonts/truetype/dejavu \
        /usr/share/fonts/dejavu \
        /usr/share/fonts/truetype/freefont; do
        if [[ -d "$font_dir" ]]; then
            export QT_QPA_FONTDIR="$font_dir"
            break
        fi
    done
fi

# Run drone_control + YOLO live inference concurrently.
# Optional:
#   --vision-only  Run camera + YOLO only (no Crazyradio/Crazyflie connection).
run_repo_python "flight_vision/main.py" "$@"
