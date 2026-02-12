from pathlib import Path
from datetime import datetime
import time
from constants import *
import cv2

################################ RECORDING DATASET #################################

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


def estimate_capture_fps(cap: cv2.VideoCapture, probe_seconds: float = 1.5,) -> float:
    # Use a monotonic clock so system clock updates do not affect timing.
    start = time.monotonic()
    frames = 0

    # Read frames for a short probe window to measure actual delivered FPS.
    # This reflects real camera+USB throughput better than driver metadata.
    while time.monotonic() - start < probe_seconds:
        ok, _ = cap.read()
        if not ok:
            # If a frame is missed during probing, briefly back off.
            time.sleep(0.01)
            continue
        frames += 1

    elapsed = time.monotonic() - start
    # Return 0.0 when the probe is unreliable so caller can use fallbacks.
    if frames < 2 or elapsed <= 0:
        return 0.0
    return frames / elapsed


def match_writer_fps(cap: cv2.VideoCapture, min_fps: float = 1.0) -> tuple[float, float, float]:
    # Value reported by backend/driver for the current stream settings.
    driver_fps = float(cap.get(cv2.CAP_PROP_FPS))
    measured_fps = estimate_capture_fps(cap)

    # Priority:
    # 1) measured FPS (best reflection of real delivery rate),
    # 2) driver-reported FPS,
    # 3) configured fallback hint.
    if measured_fps > min_fps:
        writer_fps = measured_fps
    elif driver_fps > min_fps:
        writer_fps = driver_fps
    else:
        writer_fps = float(FPS_HINT)

    return writer_fps, driver_fps, measured_fps


################################ TRACK AND LABEL VIDEO #################################

def make_tracker(tracker_type: str):
    # Tracker choice controls the speed/accuracy tradeoff during labeling.
    t = tracker_type.upper()
    if t == "CSRT":
        return cv2.TrackerCSRT_create()
    if t == "KCF":
        return cv2.TrackerKCF_create()
    if t == "MOSSE":
        return cv2.TrackerMOSSE_create()
    raise ValueError(f"Unknown tracker_type {tracker_type}")


def clamp_bbox(x, y, w, h, W, H):
    # Clamp the predicted box to stay inside the image
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def yolo_line(class_id, x, y, w, h, W, H):
    # Convert pixel bbox (x,y,w,h) to normalized [0,1] cx cy w h YOLO format.
    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    nw = w / W
    nh = h / H
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def is_near_black(frame) -> bool:
    # Analog receivers often output black frames during startup or if the channel is wrong.
    return float(frame.mean()) < 25.0 and (int(frame.max()) - int(frame.min())) < 8
