from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def open_camera(
    camera_device: str,
    width: int,
    height: int,
    fps_hint: float,
    buffer_size: int,
    fourcc: str,
):
    import cv2

    cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)

    if len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps_hint))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))

    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_device)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {camera_device}")

    return cap


def laplacian_variance(frame_bgr) -> float:
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
