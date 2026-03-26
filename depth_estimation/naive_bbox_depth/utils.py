from pathlib import Path

import cv2
from ultralytics import YOLO

from constants import *

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = resolve_repo_path(str(path))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def estimate_distance_from_bbox(
    bbox_xyxy,
    fx: float,
    real_width_m: float,
):
    x1, y1, x2, y2 = bbox_xyxy

    bbox_width_px = max(x2 - x1, 1.0)
    z_est = (fx * real_width_m) / bbox_width_px

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    return {
        "center_px": (cx, cy),
        "bbox_width_px": bbox_width_px,
        "z_est_m": z_est,
    }


def yolo_inference(image_path: str, model_path: str, conf_threshold: float):
    image_abs = resolve_repo_path(image_path)
    if not image_abs.exists():
        raise FileNotFoundError(f"Could not read image: {image_abs}")

    model_abs = resolve_repo_path(model_path)
    if not model_abs.exists():
        raise FileNotFoundError(f"Could not read model weights: {model_abs}")

    model = YOLO(str(model_abs))
    results = model.predict(str(image_abs), conf=conf_threshold)
    return results, image_abs


def process_best_detection(results, image_path: str, output_dir: str):
    image_abs = resolve_repo_path(image_path)
    image = cv2.imread(str(image_abs))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_abs}")

    best_detection = None
    best_conf = -1.0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()

            if conf > best_conf:
                best_conf = conf
                best_detection = xyxy

    if best_detection is None:
        print("No detection found.")
        return

    estimate = estimate_distance_from_bbox(
        bbox_xyxy=best_detection,
        fx=FX,
        real_width_m=DRONE_WIDTH_M,
    )

    x1, y1, x2, y2 = map(int, best_detection)
    cx, cy = map(int, estimate["center_px"])

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"Distance: {estimate['z_est_m']:.3f} m (width-based)"

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

    print("Best detection:")
    print(f"  confidence       = {best_conf:.3f}")
    print(f"  bbox width px    = {estimate['bbox_width_px']:.2f}")
    print(f"  distance (width) = {estimate['z_est_m']:.3f} m")

    output_dir_abs = ensure_output_dir(output_dir)
    image_stem = image_abs.stem
    output_path = output_dir_abs / f"{image_stem}_distance_estimate.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"Saved annotated image to: {output_path}")


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_HINT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at {DEVICE}")

    return cap


def live_distance_inference(model_path: str, conf_threshold: float):
    model_abs = resolve_repo_path(model_path)
    if not model_abs.exists():
        raise FileNotFoundError(f"Could not read model weights: {model_abs}")

    yolo_model = YOLO(str(model_abs))
    cap = open_camera()

    print("Live distance inference started. Press ESC to exit.")
    print("Displaying: YOLO detections + Distance Estimation (width-based)")

    frame_count = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            frame_count += 1
            display_frame = frame_bgr.copy()

            yolo_results = yolo_model.predict(frame_bgr, conf=conf_threshold, verbose=False)

            best_detection = None
            best_conf = -1.0

            for result in yolo_results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy()

                    if conf > best_conf:
                        best_conf = conf
                        best_detection = (xyxy, conf)

            if best_detection is not None:
                xyxy, conf = best_detection
                estimate = estimate_distance_from_bbox(
                    bbox_xyxy=xyxy,
                    fx=FX,
                    real_width_m=DRONE_WIDTH_M,
                )

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = map(int, estimate["center_px"])

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"Distance: {estimate['z_est_m']:.3f} m | Conf: {conf:.2f}"
                cv2.putText(
                    display_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.circle(display_frame, (cx, cy), 4, (0, 0, 255), -1)

            else:
                cv2.putText(
                    display_frame,
                    "No detection",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (10, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Live Distance Estimation (Naive BBox)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("Exiting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Live inference finished. Processed {frame_count} frames.")
