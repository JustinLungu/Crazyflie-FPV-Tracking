from pathlib import Path

import cv2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_yolo_model(model_ref: str) -> YOLO:
    model_path = resolve_repo_path(model_ref)
    if model_path.exists():
        return YOLO(str(model_path))
    # Allow aliases like "yolo26n.pt".
    return YOLO(model_ref)


def open_camera(
    device: str,
    width: int,
    height: int,
    fps_hint: int,
    buffer_size: int,
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_hint)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at {device}")
    return cap
