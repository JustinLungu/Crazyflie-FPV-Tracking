from depth_estimation.constants import (
    DEPTH_LIVE_REVIEW_BUFFER_SIZE,
    DEPTH_LIVE_REVIEW_DEVICE,
    DEPTH_LIVE_REVIEW_FOURCC,
    DEPTH_LIVE_REVIEW_FPS_HINT,
    DEPTH_LIVE_REVIEW_HEIGHT,
    DEPTH_LIVE_REVIEW_WIDTH,
)
from depth_estimation.naive_bbox_depth.constants import KEY_TOGGLE_GATING as NAIVE_KEY_TOGGLE_GATING
from flight_vision.constants import FLIGHT_DRONE_URI
from drone_control.constants import (
    TELEOP_DEFAULT_TARGET_Z,
    TELEOP_INVERT_ROLL,
    TELEOP_INVERT_YAW,
)

# Depth method used by this demo.
# Start with "naive" now; later can switch to "unidepth" once its control metrics are ready.
DEMO_DEPTH_METHOD = "naive"

# Mission behavior.
DEMO_FOLLOW_TARGET_DISTANCE_M = 0.40
DEMO_FOLLOW_TAKEOFF_HEIGHT_M = 0.50
DEMO_FOLLOW_CONTROL_DT = 0.06 # smaller = faster updates
# Number of frames to run CV before engaging flight control.
# This warms up model/camera and ensures preview window is live first.
DEMO_PRECONTROL_CV_WARMUP_FRAMES = 12
# If True, follower sends motion only on fresh accepted detections
# (track_state=tracked + estimate_source=measurement).
DEMO_FOLLOW_ONLY_ON_MEASUREMENT = True

# Forward-distance control.
DEMO_FOLLOW_KP_FORWARD = 0.90
DEMO_FOLLOW_MAX_VX = 0.35
DEMO_FOLLOW_DISTANCE_DEADBAND_M = 0.08 # error zone where drone "close enough" to target distance (avoid oscillations)

# Yaw-centering control.
DEMO_FOLLOW_KP_YAW = 1.80
DEMO_FOLLOW_MAX_YAWRATE_DEG_S = 60.0
DEMO_FOLLOW_YAW_DEADBAND_DEG = 5.0 # error zone where drone "facing close enough" to target (avoid oscillations)

# Runner safety behavior.
DEMO_TAKEOVER_ON_ANY_INPUT = True
DEMO_LAND_AFTER_MISSION_IF_NO_TAKEOVER = True

# Crazyflie URI.
# Reuses flight_vision URI so both runtimes target the same drone link by default.
DEMO_DRONE_URI = FLIGHT_DRONE_URI

# Teleoperation defaults (used by TeleoperationController in this demo).
DEMO_TELEOP_DEFAULT_TARGET_Z = TELEOP_DEFAULT_TARGET_Z
DEMO_TELEOP_INVERT_ROLL = TELEOP_INVERT_ROLL
DEMO_TELEOP_INVERT_YAW = TELEOP_INVERT_YAW

# Camera settings for the depth tracker.
DEMO_CAMERA_DEVICE = DEPTH_LIVE_REVIEW_DEVICE
DEMO_CAMERA_WIDTH = DEPTH_LIVE_REVIEW_WIDTH
DEMO_CAMERA_HEIGHT = DEPTH_LIVE_REVIEW_HEIGHT
DEMO_CAMERA_FPS_HINT = DEPTH_LIVE_REVIEW_FPS_HINT
DEMO_CAMERA_FOURCC = DEPTH_LIVE_REVIEW_FOURCC
DEMO_CAMERA_BUFFER_SIZE = DEPTH_LIVE_REVIEW_BUFFER_SIZE

# Preview window.
DEMO_SHOW_PREVIEW = True
DEMO_PREVIEW_WINDOW_NAME = "Demo: Drone Follower"
DEMO_MISSION_COMPLETE_ON_PREVIEW_QUIT = True
KEY_PREVIEW_QUIT = {ord("q"), 27}
KEY_PREVIEW_TOGGLE_GATING = set(NAIVE_KEY_TOGGLE_GATING)
