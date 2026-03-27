# Non-live default input image.
DEFAULT_IMAGE_STEM = "frame_009096"
DRONE_NAME = "brushless_drone"
IMAGE_PATH = "data/labels/" + DRONE_NAME + "/" + DRONE_NAME + "_yolo/images/train/" + DEFAULT_IMAGE_STEM + ".jpg"

YOLO_CONF_THRESHOLD = 0.5

# Trained brushless model weights.
MODEL_NAME = "brushless_drone_yolo26s_20260322_221846"
MODEL_PATH = "runs/models/" + MODEL_NAME + "/weights/best.pt"

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
