# Non-live default input image.
DEFAULT_IMAGE_STEM = "frame_009096"
DRONE_NAME = "brushless_drone"
IMAGE_PATH = "data/labels/" + DRONE_NAME + "/" + DRONE_NAME + "_yolo/images/train/" + DEFAULT_IMAGE_STEM + ".jpg"

# Entry mode for bbox_dist_estimator.py:
# - "image": run single-image inference using IMAGE_PATH
# - "live": run live camera inference
NAIVE_RUN_MODE = "image"

YOLO_CONF_THRESHOLD = 0.5

# Trained brushless model weights.
MODEL_NAME = "brushless_drone_yolo26s_20260322_221846"
#MODEL_PATH = "runs/models/" + MODEL_NAME + "/weights/best.pt"
MODEL_PATH = "runs/models/backup/weights/best.pt"

# Non-live outputs are grouped under output/.
OUTPUT_DIR = "depth_estimation/output/naive_bbox"

# Manual intrinsics fallback (pixels).
FX = 218.867
FY = 217.141
CX = 322.385
CY = 236.242

# Intrinsics source:
# - "calibration_npy": load from camera calibration matrix file
# - "manual": use FX/FY/CX/CY above
NAIVE_INTRINSICS_SOURCE = "calibration_npy"
NAIVE_CAMERA_MATRIX_PATH = "depth_estimation/camera_calibration/calibration_images/camera_matrix.npy"
NAIVE_INTRINSICS_FALLBACK_TO_MANUAL = True

# Crazyflie physical width in meters.
DRONE_WIDTH_M = 0.1

# Relative camera-frame position output:
# - x_rel_m: horizontal offset (meters)
# - y_rel_m: vertical offset (meters)
# - z_rel_m: forward distance (meters)
# - yaw_error_rad: horizontal angle offset from optical axis
NAIVE_ENABLE_RELATIVE_POSITION = True
NAIVE_SHOW_RELATIVE_OVERLAY_ON_FRAME = False

# y-axis convention for y_rel_m:
# - "up": positive y_rel_m means target is above optical center
# - "down": positive y_rel_m means target is below optical center
NAIVE_Y_AXIS_CONVENTION = "up"

# Camera settings (for live mode).
DEVICE = "/dev/video2"
WIDTH, HEIGHT = 640, 480
FPS_HINT = 30
FOURCC = "MJPG"
BUFFER_SIZE = 1


########################################## Temporal Filtering ###########################################

# Filter mode:
# - "none": disable temporal filtering
# - "ema": exponential moving average
# - "kalman": constant-velocity 1D Kalman filter
NAIVE_FILTER_MODE = "ema"

# Signals to filter when mode != "none".
NAIVE_FILTER_DISTANCE = True
NAIVE_FILTER_CENTER = True
NAIVE_FILTER_WIDTH = True

# EMA tuning (smaller alpha = smoother but more lag).
NAIVE_EMA_ALPHA_DISTANCE = 0.25
NAIVE_EMA_ALPHA_CENTER = 0.50
NAIVE_EMA_ALPHA_WIDTH = 0.70

# Kalman tuning (process variance Q / measurement variance R).
# Q = how much you allow state to change between frames.
# higher = more responsive but more jitter, lower = smoother but more lag.
NAIVE_KALMAN_PROCESS_VAR_DISTANCE = 0.03
NAIVE_KALMAN_PROCESS_VAR_CENTER = 3.00
NAIVE_KALMAN_PROCESS_VAR_WIDTH = 2.00
# R = how noisy you think YOLO measurements are.
# higher = trust measurements less (smoother), lower = trust measurements more (faster/jittery).
NAIVE_KALMAN_MEAS_VAR_DISTANCE = 0.20
NAIVE_KALMAN_MEAS_VAR_CENTER = 25.00
NAIVE_KALMAN_MEAS_VAR_WIDTH = 16.00

# Dropout handling:
# - hold filtered estimate for first N missed detections
# - then mark stale
# - after stale timeout, declare target lost
NAIVE_DROPOUT_HOLD_FRAMES = 3
NAIVE_DROPOUT_STALE_FRAMES = 8
NAIVE_RESET_FILTER_ON_LOST = True

########################################## Gating / Sanity Checks ###########################################

# Debug safety layer around YOLO->depth measurements.
# Can be toggled at runtime with KEY_TOGGLE_GATING in live/review windows.
NAIVE_GATING_ENABLED = False

# Only consider the top-K detections by confidence each frame.
# - 1 => only highest-confidence detection
# - 2 => consider top-2, etc.
NAIVE_GATING_MAX_CANDIDATES = 1


# Which checks to apply when gating is enabled.
NAIVE_GATING_CHECK_CONFIDENCE = True
NAIVE_GATING_CHECK_MIN_WIDTH = True
NAIVE_GATING_CHECK_MAX_DISTANCE = True
NAIVE_GATING_CHECK_BORDER = True
NAIVE_GATING_CHECK_DISTANCE_JUMP = True
NAIVE_GATING_CHECK_X_JUMP = False
NAIVE_GATING_CHECK_Y_JUMP = False



# Thresholds (starting points; tune per setup/session).
NAIVE_GATING_MIN_CONF_FOR_CONTROL = 0.60
NAIVE_GATING_MIN_BBOX_WIDTH_PX = 8.0
NAIVE_GATING_MAX_VALID_DISTANCE_M = 2.0
NAIVE_GATING_BORDER_MARGIN_PX = 5
NAIVE_GATING_MAX_DISTANCE_JUMP_M = 0.30
NAIVE_GATING_MAX_X_JUMP_M = 0.50
NAIVE_GATING_MAX_Y_JUMP_M = 0.50

# Draw rejected-detection reason overlay on frame.
NAIVE_GATING_SHOW_REJECTION_OVERLAY = True


####################### Session Review Constants (recorded test session playback) ########################################

# Must point to one session folder containing images/.
NAIVE_REVIEW_SESSION_DIR = "data/labels/brushless_drone/all_data/test/label_session_20260407_164310"
NAIVE_REVIEW_WINDOW_NAME = "Naive Depth Session Review"
NAIVE_REVIEW_START_PAUSED = False
NAIVE_REVIEW_DELAY_S = 0.15
NAIVE_REVIEW_ALLOW_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
NAIVE_REVIEW_WRITE_LOG = True
NAIVE_REVIEW_LOG_DIR = OUTPUT_DIR + "/review_logs"
NAIVE_REVIEW_PRINT_EVERY_N_FRAMES = 50

# Overlay text for session review.
NAIVE_REVIEW_TEXT_ORIGIN = (12, 28)
NAIVE_REVIEW_TEXT_LINE_HEIGHT = 26
NAIVE_REVIEW_TEXT_COLOR = (0, 255, 0)
NAIVE_REVIEW_TEXT_SCALE = 0.7
NAIVE_REVIEW_TEXT_THICKNESS = 2

# Optional right-side telemetry panel (cleaner than text over video).
NAIVE_REVIEW_USE_SIDE_PANEL = True
NAIVE_REVIEW_SIDE_PANEL_WIDTH = 420
NAIVE_REVIEW_SIDE_PANEL_BG_COLOR = (22, 22, 22)
NAIVE_REVIEW_SIDE_PANEL_TEXT_COLOR = (230, 230, 230)
NAIVE_REVIEW_SIDE_PANEL_ACCENT_COLOR = (120, 220, 120)

# Keyboard controls.
KEY_QUIT = {ord("q"), 27}  # q or ESC
KEY_TOGGLE_PLAY = {ord(" ")}
KEY_PREV = {ord("a"), 2424832, 65361}
KEY_NEXT = {ord("d"), 2555904, 65363}
KEY_TOGGLE_GATING = {ord("g"), ord("G")}
