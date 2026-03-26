import os
import sys

import cv2

CHECKERBOARD = (10, 7)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_PATH = os.path.join(
    SCRIPT_DIR,
    "calibration_images",
    "input_images",
    "frame_1.jpg",
)


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    print(f"Detected: {ret}")

    cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
    cv2.imshow("corners", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
