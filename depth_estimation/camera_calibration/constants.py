################ Check Board Detection ################
import os

CHECKERBOARD = (10, 7)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_PATH = os.path.join(
    SCRIPT_DIR,
    "calibration_images",
    "input_images",
    "frame_1.jpg",
)

################# Calibration Pipeline ################
import cv2

# IMPORTANT:
# This must match your printed checkerboard INTERNAL corners
CHECKERBOARD = (10, 7)

# Real square size in meters
SQUARE_SIZE = 0.025  # 25 mm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder structure for calibration inputs/outputs
CALIBRATION_DIR = os.path.join(SCRIPT_DIR, "calibration_images")
IMAGE_DIR = os.path.join(CALIBRATION_DIR, "input_images")
IMAGE_GLOB = os.path.join(IMAGE_DIR, "**", "*.jpg")
OUTPUT_DIR = CALIBRATION_DIR
CORNER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "corner_detections")

# Corner refinement criteria
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)