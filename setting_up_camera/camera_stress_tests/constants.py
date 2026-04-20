"""Camera-receiver link stress-test constants (minimal protocol).

Goal: justify camera+link reliability using only camera-drone to receiver distance.
No target-motion, detector, tracker, or control benchmarking is included.
"""

from __future__ import annotations

########################################## Paths ##########################################

CAMERA_STRESS_ROOT = "setting_up_camera/camera_stress_tests"
CAMERA_STRESS_RUNS_ROOT = f"{CAMERA_STRESS_ROOT}/runs"
CAMERA_STRESS_TEST_PLAN_CSV = f"{CAMERA_STRESS_ROOT}/test_plan.csv"


########################################## Camera #########################################

CAMERA_DEVICE = "/dev/video2"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS_HINT = 30.0
CAMERA_BUFFER_SIZE = 1
CAMERA_FOURCC = "MJPG"


########################################## Run Control ####################################

# Select which run to execute from TEST_RUNS.
RUN_SELECTED_ID = "run_01_link_0p5m"

# Optional operator metadata.
RUN_OPERATOR = ""
RUN_NOTES = ""

# Runtime options.
RUN_SHOW_PREVIEW = False
RUN_RECORD_RAW_VIDEO = True

# Batch runner options (`run_all_core_tests.py`).
RUN_ALL_INCLUDE_BONUS = False
RUN_ALL_PROMPT_BETWEEN_RUNS = True
RUN_ALL_STOP_ON_ERROR = True
RUN_ALL_SHOW_PREFLIGHT_PREVIEW = True
RUN_ALL_PREFLIGHT_WINDOW_NAME = "Camera Stress Preflight"
# <= 0 disables no-frame timeout (wait indefinitely for camera stream).
RUN_ALL_PREFLIGHT_NO_FRAME_TIMEOUT_S = 0.0
RUN_ALL_PREFLIGHT_DISPLAY_SCALE = 1.8
RUN_ALL_PREFLIGHT_MIN_WIDTH = 1100
RUN_ALL_PREFLIGHT_MIN_HEIGHT = 800


########################################## Analysis #######################################

# If ANALYZE_RUN_DIR is empty and ANALYZE_USE_LATEST_RUN_IF_EMPTY is True,
# analyzer picks latest run folder under ANALYZE_RUNS_ROOT.
ANALYZE_RUN_DIR = ""
ANALYZE_RUNS_ROOT = CAMERA_STRESS_RUNS_ROOT
ANALYZE_USE_LATEST_RUN_IF_EMPTY = True

# Freeze heuristic: consecutive near-identical frames.
ANALYZE_FREEZE_DIFF_THRESHOLD = 1.0
ANALYZE_FREEZE_MIN_FRAMES = 5

# Blur proxy threshold (variance of Laplacian). Lower => blurrier frame.
ANALYZE_BLUR_LAPLACIAN_THRESHOLD = 60.0

# Pass/fail thresholds for practical camera-link adequacy.
ANALYZE_MIN_FPS_RATIO = 0.85
ANALYZE_MAX_DROP_PCT = 5.0
ANALYZE_MAX_FREEZE_COUNT = 2
ANALYZE_MAX_BLUR_FRAME_PCT = 40.0
ANALYZE_MAX_DT_P95_MS = 80.0


########################################## Campaign #######################################

SUMMARY_RUNS_ROOT = CAMERA_STRESS_RUNS_ROOT
SUMMARY_OUTPUT_DIR = CAMERA_STRESS_ROOT


########################################## Test Set #######################################

# Minimal 4-run protocol:
# - receiver+laptop fixed in one place
# - only change camera-drone to receiver-node distance
# - camera stream captured for same duration each run
TEST_RUNS = [
    {
        "id": "run_01_link_0p5m",
        "title": "Run 1 - Link distance 0.5 m",
        "link_distance_m": 0.5,
        "motion": "static",
        "duration_s": 25.0,
        "purpose": "Baseline link quality at very close distance.",
        "check_focus": "effective fps, frame drops, freezes, stream lag, artifacts",
        "setup_checklist": [
            "Keep receiver+laptop fixed in one place",
            "Place camera drone ~0.5 m from receiver node",
            "Keep camera drone orientation similar across all runs",
            "Keep camera drone mostly static during recording",
        ],
        "observe_realtime": [
            "Actual FPS vs nominal FPS",
            "Frame drops / freezes",
            "Visible stream lag/choppiness",
            "Compression/noise artifacts",
        ],
    },
    {
        "id": "run_02_link_1p0m",
        "title": "Run 2 - Link distance 1.0 m",
        "link_distance_m": 1.0,
        "motion": "static",
        "duration_s": 25.0,
        "purpose": "Link quality at short practical distance.",
        "check_focus": "effective fps, frame drops, freezes, stream lag, artifacts",
        "setup_checklist": [
            "Keep receiver+laptop fixed in exact same place as Run 1",
            "Place camera drone ~1.0 m from receiver node",
            "Keep camera drone orientation similar to Run 1",
            "Keep camera drone mostly static during recording",
        ],
        "observe_realtime": [
            "Actual FPS vs nominal FPS",
            "Frame drops / freezes",
            "Visible stream lag/choppiness",
            "Compression/noise artifacts",
        ],
    },
    {
        "id": "run_03_link_2p0m",
        "title": "Run 3 - Link distance 2.0 m",
        "link_distance_m": 2.0,
        "motion": "static",
        "duration_s": 25.0,
        "purpose": "Link quality at medium distance.",
        "check_focus": "effective fps, frame drops, freezes, stream lag, artifacts",
        "setup_checklist": [
            "Keep receiver+laptop fixed in exact same place as Run 1",
            "Place camera drone ~2.0 m from receiver node",
            "Keep camera drone orientation similar to Run 1",
            "Keep camera drone mostly static during recording",
        ],
        "observe_realtime": [
            "Actual FPS vs nominal FPS",
            "Frame drops / freezes",
            "Visible stream lag/choppiness",
            "Compression/noise artifacts",
        ],
    },
    {
        "id": "run_04_link_3p0m",
        "title": "Run 4 - Link distance 3.0 m",
        "link_distance_m": 3.0,
        "motion": "static",
        "duration_s": 25.0,
        "purpose": "Link quality at farther distance (stress point).",
        "check_focus": "effective fps, frame drops, freezes, stream lag, artifacts",
        "setup_checklist": [
            "Keep receiver+laptop fixed in exact same place as Run 1",
            "Place camera drone ~3.0 m from receiver node",
            "Keep camera drone orientation similar to Run 1",
            "Keep camera drone mostly static during recording",
        ],
        "observe_realtime": [
            "Actual FPS vs nominal FPS",
            "Frame drops / freezes",
            "Visible stream lag/choppiness",
            "Compression/noise artifacts",
        ],
    },
]

# No bonus runs in this minimal protocol.
BONUS_RUNS: list[dict[str, object]] = []
