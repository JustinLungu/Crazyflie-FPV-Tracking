from pathlib import Path
from datetime import datetime
from constants import *
import cv2


def make_session_dir(root: Path, save_type: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = root / f"{save_type}_session_{ts}"
    (session / save_type).mkdir(parents=True, exist_ok=True)
    return session


def open_camera() -> cv2.VideoCapture:
    # Explicitly use V4L2 backend (important on Linux for /dev/videoX devices)
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)

    # Each frame is compressed independently as a JPEG image.
    # So instead of sending raw pixels, the camera sends JPG frame
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    # Camera capture fps
    cap.set(cv2.CAP_PROP_FPS, FPS_HINT)

    # # Lower buffering = lower latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at {DEVICE}")

    return cap