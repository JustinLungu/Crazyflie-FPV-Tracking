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


def compute_overlap_ratio_xyxy(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    min_area = min(area_a, area_b)
    if min_area <= 0:
        return 0.0
    # Overlap ratio relative to the smaller box:
    # 1.0 means the smaller box is fully inside the larger box.
    return inter_area / min_area


def suppress_overlapping_detections_indices(
    boxes_xyxy: list[tuple[float, float, float, float]],
    confidences: list[float],
    overlap_threshold: float,
) -> list[int]:
    ordered = sorted(range(len(boxes_xyxy)), key=lambda i: confidences[i], reverse=True)
    kept: list[int] = []

    for idx in ordered:
        current_box = boxes_xyxy[idx]
        has_high_overlap = False
        for kept_idx in kept:
            kept_box = boxes_xyxy[kept_idx]
            if compute_overlap_ratio_xyxy(current_box, kept_box) > overlap_threshold:
                has_high_overlap = True
                break
        if not has_high_overlap:
            kept.append(idx)
    return kept


def apply_overlap_suppression_to_result(result, overlap_threshold: float) -> int:
    if result.boxes is None or len(result.boxes) == 0:
        return 0

    boxes_xyxy = [tuple(map(float, b)) for b in result.boxes.xyxy.cpu().tolist()]
    confidences = [float(c) for c in result.boxes.conf.cpu().tolist()]
    kept_indices = suppress_overlapping_detections_indices(
        boxes_xyxy=boxes_xyxy,
        confidences=confidences,
        overlap_threshold=overlap_threshold,
    )
    result.boxes = result.boxes[kept_indices]
    return len(result.boxes)


def clamp_overlap_threshold_from_percent(overlap_percent: float) -> float:
    return max(0.0, min(1.0, overlap_percent / 100.0))


def resolve_review_session_dir(session_dir_like: str) -> Path:
    candidate = resolve_repo_path(session_dir_like)
    if not candidate.exists():
        raise RuntimeError(f"Review session path not found: {candidate}")

    images_dir = candidate / "images"
    if not images_dir.is_dir():
        raise RuntimeError(
            f"INFER_REVIEW_SESSION_DIR must point to a session folder with images/: {candidate}"
        )
    return candidate


def collect_review_images(
    session_dir: Path,
    allowed_exts: tuple[str, ...],
) -> tuple[Path, list[Path]]:
    images_dir = session_dir / "images"
    if not images_dir.exists():
        raise RuntimeError(f"Missing images directory: {images_dir}")

    normalized_exts = {ext.lower() for ext in allowed_exts}
    image_paths = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in normalized_exts
        ],
        key=lambda p: p.name,
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")
    return images_dir, image_paths


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
