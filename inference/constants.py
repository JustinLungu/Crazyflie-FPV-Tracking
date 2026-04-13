########################################## Live Camera Constants ##########################################

# Receiver camera device (Linux /dev/videoX).
CAMERA_DEVICE = "/dev/video2"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS_HINT = 30
CAMERA_BUFFER_SIZE = 1


########################################## YOLO Inference Constants #######################################

# Trained model weights to run on live feed.
CLASS_NAME = "brushless_drone_yolo26s_20260410_220049"
INFER_MODEL_WEIGHTS = "runs/models/" + CLASS_NAME + "/weights/best.pt"
#INFER_MODEL_WEIGHTS = "runs/models/backup/weights/best.pt"

# Inference behavior.
INFER_IMAGE_SIZE = 1024
INFER_CONF_THRESHOLD = 0.4
INFER_IOU_THRESHOLD = 0.9 #lower = suppress overlapping boxes aggressively, higher = multiple overlapping boxes
INFER_MAX_DETECTIONS = 10
# Extra de-duplication for single-drone use case:
# if two predicted boxes overlap more than this percent
# (intersection divided by the smaller box area),
# only the higher-confidence one is kept.
INFER_OVERLAP_SUPPRESSION_PERCENT = 30.0
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


########################################## Session Inference Viewer #######################################

# Run YOLO inference over a recorded label session folder with playback controls.
# Must point directly to one session dir containing an images/ subfolder.
# Example: data/labels/brushless_drone/all_data/test/label_session_YYYYMMDD_HHMMSS
INFER_REVIEW_SESSION_DIR = "data/labels/brushless_drone/all_data/test/label_session_20260407_164310"
INFER_REVIEW_WINDOW_NAME = "YOLO Session Inference Review"
INFER_REVIEW_START_PAUSED = False
INFER_REVIEW_DELAY_S = 0.15
INFER_REVIEW_ALLOW_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


########################################## Keyboard Controls ###############################################

KEY_QUIT = {ord("q"), 27}  # q or ESC
KEY_TOGGLE_PLAY = {ord(" ")}
KEY_PREV = {ord("a"), 2424832, 65361}
KEY_NEXT = {ord("d"), 2555904, 65363}
