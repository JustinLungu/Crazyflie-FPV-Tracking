# Non-live default input image.
DEFAULT_IMAGE_STEM = "frame_009096"
DRONE_NAME = "brushless_drone"
IMAGE_PATH = "data/labels/" + DRONE_NAME + "/" + DRONE_NAME + "_yolo/images/train/" + DEFAULT_IMAGE_STEM + ".jpg"

YOLO_CONF_THRESHOLD = 0.5

# Trained brushless model weights.
MODEL_NAME = "brushless_drone_yolo26s_20260322_221846"
#MODEL_PATH = "runs/models/" + MODEL_NAME + "/weights/best.pt"
MODEL_PATH = "runs/models/backup/weights/best.pt"

# Non-live outputs are grouped under output/.
OUTPUT_DIR = "depth_estimation/output/naive_bbox"

# Calibrated intrinsics (pixels).
FX = 218.867

# Crazyflie physical width in meters.
DRONE_WIDTH_M = 0.1

# Camera settings (for --live mode).
DEVICE = "/dev/video0"
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

# Keyboard controls.
KEY_QUIT = {ord("q"), 27}  # q or ESC
KEY_TOGGLE_PLAY = {ord(" ")}
KEY_PREV = {ord("a"), 2424832, 65361}
KEY_NEXT = {ord("d"), 2555904, 65363}
