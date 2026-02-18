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




########################################## Labeling Viewer Constants #######################################

# Keyboard mappings from cv2.waitKeyEx across common Linux/OpenCV backends.
KEY_QUIT = {ord("q"), 27}  # q, ESC
KEY_TOGGLE_PLAY = {ord(" ")}
KEY_PREV = {ord("a"), 2424832, 65361}
KEY_NEXT = {ord("d"), 2555904, 65363}
KEY_DELETE = {ord("x"), ord("X")}
KEY_CONFIRM_YES = {ord("y"), ord("Y")}
KEY_CONFIRM_NO = {ord("n"), ord("N"), 27}


########################################## Drive Backup Constants ###########################################

# Name template: <BACKUP_PREFIX>_<ddmmyyyy>_<suffix>.zip
BACKUP_PREFIX = "dataset_backup"
BACKUP_DATE_FORMAT = "%d%m%Y"
BACKUP_RAW_SUFFIX = "raw_data"
BACKUP_LABELS_SUFFIX = "labels"
BACKUP_ARCHIVE_FORMAT = "zip"

# Environment configuration for private Drive target.
ENV_FILE_PATH = ".env"
GDRIVE_FOLDER_URL_ENV_KEY = "GOOGLE_DRIVE_FOLDER_URL"

# OAuth files for Google Drive API.
GDRIVE_CREDENTIALS_PATH = "data/gdrive_credentials.json"
GDRIVE_TOKEN_PATH = "data/gdrive_token.json"
GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
GDRIVE_UPLOAD_MIME_TYPE = "application/zip"
GDRIVE_UPLOAD_RESPONSE_FIELDS = "id,name"
GDRIVE_SUPPORTS_ALL_DRIVES = True

GDRIVE_FOLDER_ID_PATH_REGEX = r"/folders/([A-Za-z0-9_-]+)"
GDRIVE_FOLDER_ID_RAW_REGEX = r"[A-Za-z0-9_-]{10,}"
GDRIVE_FOLDER_QUERY_KEYS = ("id", "folder")

# Temporary directory prefix when building transient archives.
BACKUP_TEMP_DIR_PREFIX = "dataset_backup_"





########################################## YOLO Dataset Prep Constants ####################################

# Class folder and dataset names under data/labels/.
YOLO_TARGET_CLASS_NAME = "green_drone"
YOLO_SOURCE_DATASET_NAME = YOLO_TARGET_CLASS_NAME + "_dataset"
YOLO_OUTPUT_DATASET_NAME = YOLO_TARGET_CLASS_NAME + "_yolo"
YOLO_DATASET_YAML_NAME = "dataset.yaml"
YOLO_OVERWRITE_OUTPUT = True

# Optional filter: leave empty tuple to include all source sessions from manifest.csv.
YOLO_INCLUDED_SESSIONS = ()

# Split configuration for train/val/test.
YOLO_SPLIT_SEED = 42
YOLO_TRAIN_RATIO = 0.70
YOLO_VAL_RATIO = 0.20
YOLO_TEST_RATIO = 0.10

# Prefer session-level split to avoid leakage between temporally-adjacent frames.
YOLO_MIN_SESSIONS_FOR_GROUP_SPLIT = 3
YOLO_FALLBACK_TO_FRAME_SPLIT_IF_FEW_SESSIONS = True

# Single-class remap: force all labels to class 0 in generated YOLO split.
YOLO_SINGLE_CLASS_MODE = True
YOLO_TARGET_CLASS_ID = 0


#######################################################################################################
