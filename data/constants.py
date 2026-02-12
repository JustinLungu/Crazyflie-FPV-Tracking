################################# Video & Image Capture Constants #################################
DEVICE = "/dev/video1"
TARGET_FPS = 10  # set 5 or 10 for labeling
WIDTH, HEIGHT = 640, 480
FPS_HINT = 30
RAW_DATA_ROOT = "data/raw_data"
FOURCC = "MJPG"
BUFFER_SIZE = 1
VIDEO_FLIE_NAME = "video.avi"


################################ Tracker Labeling Constants ########################################

# Constants for tracker labeling
# Input video to label and output folder for image/label pairs.
VIDEO_PATH = "data/raw_data/black_video_session_20260211_183212/video.avi"
# Root directory where all labeled outputs are stored.
OUT_DIR = "data/labels"

# Folder name for the object class currently being labeled.
# Example: "black_drone", "green_drone"
LABEL_CLASS_NAME = "black_drone"
CLASS_ID = 1

# Each class stores labeled sessions under labels/<class_name>/all_data/.
LABEL_ALL_DATA_DIR = "all_data"
# Each labeling run gets its own timestamped subfolder under all_data/.
LABEL_SESSION_PREFIX = "label_session_"

# Export target FPS for sampled frames from the source video.
EXPORT_FPS = 30.0

TRACKER_TYPE = "CSRT" # CSRT/KCF/MOSSE need opencv-contrib;
STARTUP_PROBE_FRAMES = 30  # skip startup frames that are near-black/blank
# Slow down preview to give operator more reaction time while validating tracking.
TRACK_REVIEW_DELAY_S = 0.5 # 0.5s ~= 2 frames per second in review mode.

#######################################################################################################
