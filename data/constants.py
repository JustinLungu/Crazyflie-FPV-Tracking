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
VIDEO_PATH = "data/raw_data/video_session_20260211_174944/video.avi"
OUT_DIR = "data/labels_session/"
CLASS_ID = 1
# Export target FPS for sampled frames from the source video.
EXPORT_FPS = 10.0
TRACKER_TYPE = "CSRT"  # CSRT is accurate, KCF is faster, MOSSE is fastest
STARTUP_PROBE_FRAMES = 30  # skip startup frames that are near-black/blank
# Slow down preview to give operator more reaction time while validating tracking.
TRACK_REVIEW_DELAY_S = 0.5 # 0.5s ~= 2 frames per second in review mode.

#######################################################################################################