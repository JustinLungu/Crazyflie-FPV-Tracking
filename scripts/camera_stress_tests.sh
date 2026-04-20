#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

print_menu() {
    cat <<'MENU'
Camera Stress Tests Launcher
1) Build test plan CSV
2) Run selected single test (RUN_SELECTED_ID in constants.py)
3) Run all core tests
4) Analyze latest run
5) Analyze all runs
6) Summarize campaign
7) Exit
MENU
}

run_choice() {
    local choice="$1"
    case "$choice" in
        1)
            run_repo_python "setting_up_camera/camera_stress_tests/build_test_matrix.py"
            ;;
        2)
            run_repo_python "setting_up_camera/camera_stress_tests/run_camera_test.py"
            ;;
        3)
            run_repo_python "setting_up_camera/camera_stress_tests/run_all_core_tests.py"
            ;;
        4)
            run_repo_python "setting_up_camera/camera_stress_tests/analyze_camera_test.py"
            ;;
        5)
            run_repo_python "setting_up_camera/camera_stress_tests/analyze_all_runs.py"
            ;;
        6)
            run_repo_python "setting_up_camera/camera_stress_tests/summarize_campaign.py"
            ;;
        7)
            exit 0
            ;;
        *)
            echo "Invalid choice: $choice"
            ;;
    esac
}

while true; do
    print_menu
    read -r -p "Choose [1-7]: " choice
    run_choice "${choice:-}"
    echo
done
