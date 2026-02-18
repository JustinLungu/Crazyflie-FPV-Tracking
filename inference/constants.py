########################################## Live Camera Constants ##########################################

# Receiver camera device (Linux /dev/videoX).
CAMERA_DEVICE = "/dev/video2"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS_HINT = 30
CAMERA_BUFFER_SIZE = 1


########################################## YOLO Inference Constants #######################################

# Trained model weights to run on live feed.
INFER_MODEL_WEIGHTS = "runs/detect/runs/detect/black_drone_yolo26s/weights/best.pt"

# Inference behavior.
INFER_IMAGE_SIZE = 960
INFER_CONF_THRESHOLD = 0.4
INFER_IOU_THRESHOLD = 0.9 #lower = suppress overlapping boxes aggressively, higher = multiple overlapping boxes
INFER_MAX_DETECTIONS = 10
INFER_DEVICE = 0  # Set "cpu" to run on CPU, 0 = GPU
INFER_VERBOSE = False


########################################## Visualization Constants ########################################

WINDOW_NAME = "Live Drone Inference"
SHOW_LABELS = True
SHOW_CONFIDENCE = True
BOX_LINE_WIDTH = 2

# Overlay text for runtime feedback.
OVERLAY_FONT_SCALE = 0.7
OVERLAY_THICKNESS = 2
OVERLAY_LINE_HEIGHT = 26
OVERLAY_TEXT_COLOR = (0, 255, 0)
OVERLAY_TEXT_ORIGIN = (12, 28)


########################################## Keyboard Controls ###############################################

KEY_QUIT = {ord("q"), 27}  # q or ESC
