import time

import cv2

from constants import *
from utils import *


def draw_session_overlay(
    frame,
    session_name: str,
    image_name: str,
    index: int,
    total: int,
    playing: bool,
    infer_ms: float,
    detection_count: int,
    delay_s: float,
) -> None:
    x, y = OVERLAY_TEXT_ORIGIN
    status = "PLAY" if playing else "PAUSE"
    lines = [
        f"{status}  frame {index + 1}/{total}  detections: {detection_count}",
        f"inference: {infer_ms:.1f} ms  delay: {delay_s:.2f}s",
        f"session: {session_name}",
        f"image: {image_name}",
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


def run_inference_on_frame(model, frame, overlap_threshold: float):
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
    detection_count = apply_overlap_suppression_to_result(result, overlap_threshold=overlap_threshold)
    annotated = result.plot(
        labels=SHOW_LABELS,
        conf=SHOW_CONFIDENCE,
        line_width=BOX_LINE_WIDTH,
    )
    return annotated, detection_count, infer_ms


def main() -> None:
    model = load_yolo_model(INFER_MODEL_WEIGHTS)
    session_dir = resolve_review_session_dir(
        session_dir_like=INFER_REVIEW_SESSION_DIR,
    )
    _, image_paths = collect_review_images(
        session_dir=session_dir,
        allowed_exts=INFER_REVIEW_ALLOW_IMAGE_EXTS,
    )

    overlap_threshold = clamp_overlap_threshold_from_percent(INFER_OVERLAP_SUPPRESSION_PERCENT)
    delay_s = max(0.0, float(INFER_REVIEW_DELAY_S))
    delay_ms = max(1, int(delay_s * 1000))
    playing = not INFER_REVIEW_START_PAUSED

    print(f"Loaded model: {INFER_MODEL_WEIGHTS}")
    print(f"Review session: {session_dir}")
    print(f"Images: {len(image_paths)}")
    print(f"Overlap suppression: {INFER_OVERLAP_SUPPRESSION_PERCENT:.1f}% IoU")
    print("Controls")
    print("space: play/pause")
    print("a or Left Arrow: previous frame")
    print("d or Right Arrow: next frame")
    print("q or ESC: quit")

    index = 0
    cached_index = -1
    cached_annotated = None
    cached_detection_count = 0
    cached_infer_ms = 0.0

    while True:
        image_path = image_paths[index]
        if cached_index != index or cached_annotated is None:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Warning: could not read image {image_path}. Skipping.")
                image_paths.pop(index)
                if not image_paths:
                    print("No readable images left. Exiting.")
                    break
                index = min(index, len(image_paths) - 1)
                cached_index = -1
                continue

            cached_annotated, cached_detection_count, cached_infer_ms = run_inference_on_frame(
                model=model,
                frame=frame,
                overlap_threshold=overlap_threshold,
            )
            cached_index = index

        display = cached_annotated.copy()
        draw_session_overlay(
            frame=display,
            session_name=session_dir.name,
            image_name=image_path.name,
            index=index,
            total=len(image_paths),
            playing=playing,
            infer_ms=cached_infer_ms,
            detection_count=cached_detection_count,
            delay_s=delay_s,
        )
        cv2.imshow(INFER_REVIEW_WINDOW_NAME, display)
        key = cv2.waitKeyEx(delay_ms if playing else 0)

        if key == -1:
            if playing:
                if index < len(image_paths) - 1:
                    index += 1
                else:
                    playing = False
            continue

        if key in KEY_QUIT:
            break
        if key in KEY_TOGGLE_PLAY:
            playing = not playing
            continue
        if key in KEY_PREV:
            index = max(0, index - 1)
            playing = False
            continue
        if key in KEY_NEXT:
            index = min(len(image_paths) - 1, index + 1)
            playing = False
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
