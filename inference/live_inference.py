import time

import cv2

from constants import *
from utils import *


def draw_overlay(frame, detection_count: int, infer_ms: float, display_fps: float) -> None:
    x, y = OVERLAY_TEXT_ORIGIN
    lines = [
        f"detections: {detection_count}",
        f"inference: {infer_ms:.1f} ms",
        f"display fps: {display_fps:.1f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * OVERLAY_LINE_HEIGHT),
            cv2.FONT_HERSHEY_SIMPLEX,
            OVERLAY_FONT_SCALE,
            OVERLAY_TEXT_COLOR,
            OVERLAY_THICKNESS,
            cv2.LINE_AA,
        )


def compute_iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
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
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def suppress_overlapping_detections(
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
            if compute_iou_xyxy(current_box, kept_box) > overlap_threshold:
                has_high_overlap = True
                break
        if not has_high_overlap:
            kept.append(idx)
    return kept


def extract_kept_indices(result, overlap_threshold: float) -> list[int]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes_xyxy = [tuple(map(float, b)) for b in result.boxes.xyxy.cpu().tolist()]
    confidences = [float(c) for c in result.boxes.conf.cpu().tolist()]
    return suppress_overlapping_detections(
        boxes_xyxy=boxes_xyxy,
        confidences=confidences,
        overlap_threshold=overlap_threshold,
    )


def main() -> None:
    model = load_yolo_model(INFER_MODEL_WEIGHTS)
    cap = open_camera(
        device=CAMERA_DEVICE,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps_hint=CAMERA_FPS_HINT,
        buffer_size=CAMERA_BUFFER_SIZE,
    )

    print(f"Loaded model: {INFER_MODEL_WEIGHTS}")
    print(f"Camera: {CAMERA_DEVICE} ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    print(f"Overlap suppression: {INFER_OVERLAP_SUPPRESSION_PERCENT:.1f}% IoU")
    print("Press q or ESC to quit.")

    prev_loop_time = time.perf_counter()
    overlap_threshold = max(0.0, min(1.0, INFER_OVERLAP_SUPPRESSION_PERCENT / 100.0))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Brief backoff avoids a hot loop when receiver drops frames.
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            results = model.predict(
                source=frame,
                imgsz=INFER_IMAGE_SIZE,
                conf=INFER_CONF_THRESHOLD,
                iou=INFER_IOU_THRESHOLD,
                max_det=INFER_MAX_DETECTIONS,
                device=INFER_DEVICE,
                verbose=INFER_VERBOSE,
            )
            infer_ms = (time.perf_counter() - t0) * 1000.0

            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                kept_indices = extract_kept_indices(result, overlap_threshold=overlap_threshold)
                result.boxes = result.boxes[kept_indices]

            annotated = result.plot(
                labels=SHOW_LABELS,
                conf=SHOW_CONFIDENCE,
                line_width=BOX_LINE_WIDTH,
            )
            detection_count = 0 if result.boxes is None else len(result.boxes)
            now = time.perf_counter()
            loop_dt = max(1e-6, now - prev_loop_time)
            prev_loop_time = now
            display_fps = 1.0 / loop_dt
            draw_overlay(annotated, detection_count, infer_ms, display_fps)

            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in KEY_QUIT:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
