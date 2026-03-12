NAME = "30_cm"
IMAGE_PATH = "images/depth/drone/" + NAME + ".jpg"
YOLO_CONF_THRESHOLD = 0.5
MODEL_PATH = "yolo/best.pt"
OUTPUT_DIR = "output/" + NAME

# Replace these with your actual calibrated camera intrinsics
FX = 478.62

# Replace these with the real physical dimensions of the Crazyflie in meters
DRONE_WIDTH_M = 0.06

# Camera settings
DEVICE = "/dev/video2"
WIDTH, HEIGHT = 640, 480
FPS_HINT = 30
FOURCC = "MJPG"
BUFFER_SIZE = 1