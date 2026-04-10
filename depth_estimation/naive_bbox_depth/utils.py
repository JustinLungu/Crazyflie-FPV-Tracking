from pathlib import Path
import math

import cv2
import numpy as np
from ultralytics import YOLO

from depth_estimation.naive_bbox_depth.constants import (
    BUFFER_SIZE,
    DEVICE,
    DRONE_WIDTH_M,
    FOURCC,
    FPS_HINT,
    FX,
    HEIGHT,
    WIDTH,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = resolve_repo_path(str(path))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_intrinsics_from_camera_matrix(camera_matrix_path: str):
    matrix_abs = resolve_repo_path(camera_matrix_path)
    if not matrix_abs.exists():
        raise FileNotFoundError(f"Could not read camera matrix: {matrix_abs}")

    camera_matrix = np.load(str(matrix_abs))
    if camera_matrix.shape != (3, 3):
        raise ValueError(
            f"Camera matrix at {matrix_abs} must have shape (3, 3), got {camera_matrix.shape}"
        )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "camera_matrix_path": str(matrix_abs),
    }


def estimate_relative_position_from_center(
    center_px: tuple[float, float],
    z_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    y_axis_convention: str = "up",
):
    u, v = center_px
    x_rel_m = ((u - cx) / fx) * z_m
    y_img_down_m = ((v - cy) / fy) * z_m

    convention = y_axis_convention.strip().lower()
    if convention not in {"up", "down"}:
        raise ValueError(f"Unsupported y-axis convention: {y_axis_convention}")
    y_rel_m = -y_img_down_m if convention == "up" else y_img_down_m

    yaw_error_rad = math.atan((u - cx) / fx)
    yaw_error_deg = math.degrees(yaw_error_rad)

    return {
        "x_rel_m": float(x_rel_m),
        "y_rel_m": float(y_rel_m),
        "z_rel_m": float(z_m),
        "yaw_error_rad": float(yaw_error_rad),
        "yaw_error_deg": float(yaw_error_deg),
    }


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


def process_best_detection(
    results,
    image_path: str,
    output_dir: str,
    fx: float = FX,
    real_width_m: float = DRONE_WIDTH_M,
):
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
        fx=fx,
        real_width_m=real_width_m,
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
