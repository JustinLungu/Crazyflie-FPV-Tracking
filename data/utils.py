from pathlib import Path
from datetime import datetime
import math
import time
from constants import *
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent

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

def sanitize_class_folder_name(name: str) -> str:
    """Convert class name into a safe folder name for local filesystem use."""
    cleaned = name.strip()
    cleaned = cleaned.replace("/", "_").replace("\\", "_")
    return cleaned if cleaned else "unnamed_class"


def create_unique_label_session_dir(labels_root: Path, class_name: str) -> Path:
    """Create labels/<class_name>/all_data/<label_session_timestamp[_NN]> without overwriting."""
    session_prefix = "label_session_"
    safe_class_name = sanitize_class_folder_name(class_name)
    class_dir = labels_root / safe_class_name
    all_data_dir = class_dir / LABEL_ALL_DATA_DIR
    all_data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = all_data_dir / f"{session_prefix}{timestamp}"
    suffix = 1
    while session_dir.exists():
        session_dir = all_data_dir / f"{session_prefix}{timestamp}_{suffix:02d}"
        suffix += 1

    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


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


def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _candidate_class_tokens(class_name: str) -> list[str]:
    safe = sanitize_class_folder_name(class_name).lower()
    tokens: list[str] = [safe]
    for suffix in ("_drone", "-drone"):
        if safe.endswith(suffix):
            trimmed = safe[: -len(suffix)].rstrip("_-")
            if trimmed:
                tokens.append(trimmed)
    if safe.startswith("drone_"):
        tokens.append(safe[len("drone_"):])
    deduped: list[str] = []
    for token in tokens:
        if token and token not in deduped:
            deduped.append(token)
    return deduped


def _discover_best_pt_in_dir(model_dir: Path) -> Path | None:
    direct = model_dir / "weights" / "best.pt"
    if direct.exists():
        return direct
    flat = model_dir / "best.pt"
    if flat.exists():
        return flat

    candidates = [p for p in model_dir.rglob("best.pt") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_labeling_yolo_weights(class_name: str, models_root: str) -> Path | None:
    root = resolve_repo_path(models_root)
    if not root.exists() or not root.is_dir():
        return None

    tokens = _candidate_class_tokens(class_name)

    # First pass: exact directory name match.
    for token in tokens:
        direct_dir = root / token
        if direct_dir.is_dir():
            weights = _discover_best_pt_in_dir(direct_dir)
            if weights is not None:
                return weights

    # Second pass: partial match in directory name (e.g. class token embedded in run folder name).
    matched_dirs = []
    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        name = subdir.name.lower()
        if any(token in name for token in tokens):
            matched_dirs.append(subdir)

    if not matched_dirs:
        return None

    matched_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for model_dir in matched_dirs:
        weights = _discover_best_pt_in_dir(model_dir)
        if weights is not None:
            return weights
    return None


def load_labeling_yolo_model(weights_path: Path):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is required for YOLO-assisted labeling but could not be imported."
        ) from exc
    return YOLO(str(weights_path))


def yolo_best_detection_xywh(yolo_model, frame_bgr, conf_threshold: float) -> tuple[int, int, int, int, float] | None:
    h, w = frame_bgr.shape[:2]
    results = yolo_model.predict(frame_bgr, conf=float(conf_threshold), verbose=False)
    best: tuple[int, int, int, int, float] | None = None

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue

        xyxy_list = boxes.xyxy.cpu().tolist()
        conf_list = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(xyxy_list)

        for xyxy, conf_raw in zip(xyxy_list, conf_list):
            x1, y1, x2, y2 = map(float, xyxy[:4])
            conf = float(conf_raw)
            x = int(round(x1))
            y = int(round(y1))
            bw = int(round(x2 - x1))
            bh = int(round(y2 - y1))
            x, y, bw, bh = clamp_bbox(x, y, bw, bh, w, h)
            if best is None or conf > best[4]:
                best = (x, y, bw, bh, conf)
    return best


def bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return x + (w / 2.0), y + (h / 2.0)


def bbox_center_jump_ratio(
    prev_box: tuple[int, int, int, int],
    next_box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> float:
    prev_cx, prev_cy = bbox_center(prev_box)
    next_cx, next_cy = bbox_center(next_box)
    dist = math.hypot(next_cx - prev_cx, next_cy - prev_cy)
    diag = math.hypot(float(frame_w), float(frame_h))
    if diag <= 0:
        return 0.0
    return dist / diag


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
