import time

import cv2

from constants import *
from tracker_failsafe import (
    DetectionFailsafeTracker,
    draw_detection_candidate,
    extract_yolo_detections,
    rank_detections,
)
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


def main() -> None:
    model = load_yolo_model(INFER_MODEL_WEIGHTS)
    failsafe_tracker = DetectionFailsafeTracker(
        enabled=INFER_FAILSAFE_TRACKER_ENABLED,
        tracker_type=INFER_FAILSAFE_TRACKER_TYPE,
        max_fallback_frames=INFER_FAILSAFE_TRACKER_MAX_FRAMES,
        min_bbox_area_px=INFER_FAILSAFE_TRACKER_MIN_BBOX_AREA_PX,
        max_center_jump_px=INFER_FAILSAFE_TRACKER_MAX_CENTER_JUMP_PX,
        reinitialize_on_detection=INFER_FAILSAFE_TRACKER_REINIT_ON_DETECTION,
    )
    failsafe_label = (
        INFER_FAILSAFE_TRACKER_LABEL.strip()
        or f"{INFER_FAILSAFE_TRACKER_TYPE.upper()} failsafe"
    )
    cap = open_camera(
        device=CAMERA_DEVICE,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps_hint=CAMERA_FPS_HINT,
        buffer_size=CAMERA_BUFFER_SIZE,
    )

    print(f"Loaded model: {INFER_MODEL_WEIGHTS}")
    print(f"Camera: {CAMERA_DEVICE} ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    print(f"Overlap suppression: {INFER_OVERLAP_SUPPRESSION_PERCENT:.1f}% overlap")
    print(
        "Tracker failsafe: "
        f"{'on' if INFER_FAILSAFE_TRACKER_ENABLED else 'off'}"
        + (
            f" ({INFER_FAILSAFE_TRACKER_TYPE.upper()}, "
            f"max {INFER_FAILSAFE_TRACKER_MAX_FRAMES} frames)"
            if INFER_FAILSAFE_TRACKER_ENABLED
            else ""
        )
    )
    print("Press q or ESC to quit.")

    prev_loop_time = time.perf_counter()
    overlap_threshold = clamp_overlap_threshold_from_percent(INFER_OVERLAP_SUPPRESSION_PERCENT)
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
            apply_overlap_suppression_to_result(result, overlap_threshold=overlap_threshold)
            yolo_detections = rank_detections(extract_yolo_detections(result))
            fallback_detection = None
            if yolo_detections:
                failsafe_tracker.initialize(frame, yolo_detections[0])
            else:
                fallback_detection = failsafe_tracker.update(frame)

            annotated = result.plot(
                labels=SHOW_LABELS,
                conf=SHOW_CONFIDENCE,
                line_width=BOX_LINE_WIDTH,
            )
            if fallback_detection is not None:
                draw_detection_candidate(
                    annotated,
                    fallback_detection,
                    color=INFER_FAILSAFE_TRACKER_BOX_COLOR,
                    label=failsafe_label,
                    line_width=BOX_LINE_WIDTH,
                )
            detection_count = 0 if result.boxes is None else len(result.boxes)
            if fallback_detection is not None:
                detection_count = 1
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
