########################################## Model Constants ###############################################

# MiDaS model variant loaded via torch.hub from intel-isl/MiDaS.
# Common options: "DPT_Large", "DPT_Hybrid", "MiDaS_small".
MIDAS_MODEL_TYPE = "DPT_Hybrid"

# Device selection:
# - "auto": cuda if available, otherwise cpu
# - "cuda", "cpu", or "mps" (if supported)
MIDAS_DEVICE = "auto"


########################################## Single Image Constants ########################################

# Input image to estimate depth from.
DRONE_NAME = "green_drone"
CUSTOM_PATH_IMG = "images_session_20260211_175008/images/frame_000001"
MIDAS_IMAGE_INPUT_PATH = "data/raw_data/" + CUSTOM_PATH_IMG + ".png"
# If MIDAS_IMAGE_INPUT_PATH does not exist, try these extensions on the same stem.
MIDAS_IMAGE_FALLBACK_EXTENSIONS = ("png", "jpg", "jpeg")

# Outputs for image-depth inference (flat, non-live -> output folder).
MIDAS_OUTPUT_ROOT = "depth_estimation/output/midas/" + DRONE_NAME
MIDAS_IMAGE_OUTPUT_DIR = MIDAS_OUTPUT_ROOT
MIDAS_IMAGE_OUTPUT_NPY_SUFFIX = "_depth.npy"
MIDAS_IMAGE_OUTPUT_VIS_SUFFIX = "_depth_vis.png"


########################################## Video Constants ###############################################

# Input .avi file to estimate depth frame-by-frame.
CUSTOM_PATH_VIDEO = "brushless_session_20260312_171741"
MIDAS_VIDEO_INPUT_PATH = "data/raw_data/" + CUSTOM_PATH_VIDEO + "/video.avi"

# If True, writes a side-by-side video (RGB frame + depth colormap).
MIDAS_VIDEO_WRITE_OUTPUT = True
MIDAS_VIDEO_OUTPUT_PATH = MIDAS_OUTPUT_ROOT + "/" + CUSTOM_PATH_VIDEO + "/video_depth_overlay.avi"

# If True, show a live preview window during processing.
MIDAS_VIDEO_SHOW_PREVIEW = True
MIDAS_VIDEO_WINDOW_NAME = "MiDaS Video Inference"

# Optional frame cap for quick checks. Set to 0 to process all frames.
MIDAS_VIDEO_MAX_FRAMES = 0


########################################## Depth Measurement Constants ###################################

# Center-patch size used for robust center-depth reporting.
# Odd number recommended (3, 5, 7, ...).
MIDAS_CENTER_PATCH_SIZE = 9

# Console logging cadence for video processing.
MIDAS_PRINT_EVERY_N_FRAMES = 10


########################################## Visualization Constants #######################################

# Supported: "turbo", "magma", "inferno", "jet", "viridis".
MIDAS_COLORMAP = "turbo"
# If True, near depth is rendered in warm colors (red/yellow) and far depth in cool colors (blue).
MIDAS_INVERT_COLORMAP = False

# Overlay style for video output.
MIDAS_TEXT_ORIGIN = (14, 28)
MIDAS_TEXT_LINE_HEIGHT = 26
MIDAS_TEXT_COLOR = (0, 255, 0)
MIDAS_TEXT_SCALE = 0.7
MIDAS_TEXT_THICKNESS = 2


########################################## Keyboard Controls #############################################

KEY_QUIT = {ord("q"), 27}  # q or ESC
