########################################## Model Constants ###############################################

# UniDepth v2 resolution setting.
# Range: [0, 10). Lower is faster, higher can preserve more detail.
# Set to None to keep the model default.
DEPTH_RESOLUTION_LEVEL = None


########################################## Single Image Constants ########################################

# Input image to estimate depth from.
CUSTOM_PATH_IMG = "images_session_20260211_175008/images/frame_000001"
DEPTH_IMAGE_INPUT_PATH = "data/raw_data/" + CUSTOM_PATH_IMG + ".png"
# If DEPTH_IMAGE_INPUT_PATH does not exist, try these extensions on the same stem.
DEPTH_IMAGE_FALLBACK_EXTENSIONS = ("png", "jpg", "jpeg")

# Outputs for image-depth inference.
DEPTH_IMAGE_OUTPUT_NPY_PATH = "data/depth/" + CUSTOM_PATH_IMG + "/image_depth.npy"
DEPTH_IMAGE_OUTPUT_VIS_PATH = "data/depth/" + CUSTOM_PATH_IMG + "/image_depth_vis.png"


########################################## Video Constants ###############################################

# Input .avi file to estimate depth frame-by-frame.
CUSTOM_PATH_VIDEO = "depth_black_video_session_20260219_145858"
DEPTH_VIDEO_INPUT_PATH = "data/raw_data/" + CUSTOM_PATH_VIDEO + "/video.avi"

# If True, writes a side-by-side video (RGB frame + depth colormap).
DEPTH_VIDEO_WRITE_OUTPUT = True
DEPTH_VIDEO_OUTPUT_PATH = "data/depth/" + CUSTOM_PATH_VIDEO + "/video_depth_overlay.avi"

# If True, show a live preview window during processing.
DEPTH_VIDEO_SHOW_PREVIEW = True
DEPTH_VIDEO_WINDOW_NAME = "UniDepth V2 Video Inference"

# Optional frame cap for quick checks. Set to 0 to process all frames.
DEPTH_VIDEO_MAX_FRAMES = 0


########################################## Depth Measurement Constants ###################################

# Center-patch size used for robust center-depth reporting.
# Odd number recommended (3, 5, 7, ...).
DEPTH_CENTER_PATCH_SIZE = 9

# Console logging cadence for video processing.
DEPTH_PRINT_EVERY_N_FRAMES = 10


########################################## Visualization Constants #######################################

# Supported: "turbo", "magma", "inferno", "jet", "viridis".
DEPTH_COLORMAP = "turbo"
# If True, near depth is rendered in warm colors (red/yellow) and far depth in cool colors (blue).
DEPTH_INVERT_COLORMAP = True

# Overlay style for video output.
DEPTH_TEXT_ORIGIN = (14, 28)
DEPTH_TEXT_LINE_HEIGHT = 26
DEPTH_TEXT_COLOR = (0, 255, 0)
DEPTH_TEXT_SCALE = 0.7
DEPTH_TEXT_THICKNESS = 2


########################################## Keyboard Controls #############################################

KEY_QUIT = {ord("q"), 27}  # q or ESC
