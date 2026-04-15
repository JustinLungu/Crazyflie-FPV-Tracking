from drone_control.constants import MISSION as DEFAULT_MISSION
from drone_control.joystick.constants import URI as DEFAULT_DRONE_URI
from inference.constants import (
    CAMERA_BUFFER_SIZE as DEFAULT_CAMERA_BUFFER_SIZE,
    CAMERA_DEVICE as DEFAULT_CAMERA_DEVICE,
    CAMERA_FPS_HINT as DEFAULT_CAMERA_FPS_HINT,
    CAMERA_HEIGHT as DEFAULT_CAMERA_HEIGHT,
    CAMERA_WIDTH as DEFAULT_CAMERA_WIDTH,
    INFER_DEVICE as DEFAULT_INFER_DEVICE,
    INFER_MODEL_WEIGHTS as DEFAULT_MODEL_WEIGHTS,
)

########################################## Frequently Switched ############################################

# Drone control target (Crazyradio URI).
# Switch this when you want to control a different drone.
# Format: "radio://<dongle>/<channel>/<rate>/<address>"
# Example: "radio://0/80/2M/E7E7E7E7E7"
#FLIGHT_DRONE_URI = DEFAULT_DRONE_URI
FLIGHT_DRONE_URI = "radio://0/80/250K/E7E7E7E7E3"

# Mission name used by DroneControlApp.
# Common built-in options: "square", "height", "origin_to_point", "roll_pitch_yaw"
# Example:
# FLIGHT_MISSION = "square"
FLIGHT_MISSION = DEFAULT_MISSION

# Camera source settings (USB FPV receiver).
# Switch VISION_CAMERA_DEVICE if receiver appears on another node.
# Example: "/dev/video0" -> "/dev/video1"
# Tip: run `ls /dev/video*` to inspect available devices.
VISION_CAMERA_DEVICE = "/dev/video2"
# Output frame size shown and processed by YOLO.
# Example:
# VISION_CAMERA_WIDTH = 640
# VISION_CAMERA_HEIGHT = 480
VISION_CAMERA_WIDTH = DEFAULT_CAMERA_WIDTH
VISION_CAMERA_HEIGHT = DEFAULT_CAMERA_HEIGHT
# Capture hints for OpenCV/V4L2.
# Lower buffer size reduces latency (usually preferred for live control).
# Example:
# VISION_CAMERA_FPS_HINT = 30
# VISION_CAMERA_BUFFER_SIZE = 1
VISION_CAMERA_FPS_HINT = DEFAULT_CAMERA_FPS_HINT
VISION_CAMERA_BUFFER_SIZE = DEFAULT_CAMERA_BUFFER_SIZE

# YOLO model selection.
# Path can be repo-relative or absolute.
# Example:
# VISION_MODEL_WEIGHTS = "runs/models/brushless_drone_yolo26s_20260322_221846/weights/best.pt"
VISION_MODEL_WEIGHTS = DEFAULT_MODEL_WEIGHTS
# Inference device:
# - 0, 1, ... -> CUDA GPU index
# - "cpu"     -> CPU inference
# Example:
# VISION_INFER_DEVICE = 0
VISION_INFER_DEVICE = DEFAULT_INFER_DEVICE
