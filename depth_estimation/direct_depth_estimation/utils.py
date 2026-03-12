import os
import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from constants import *



def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def yolo_inference(image_path, model_path, conf_threshold):
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf_threshold)
    return results



def process_best_detection(results, image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

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

    ensure_output_dir(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, "distance_estimate.jpg")
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to: {output_path}")


def open_camera() -> cv2.VideoCapture:
    """Open camera with V4L2 backend for Linux."""
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_HINT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at {DEVICE}")

    return cap


def live_distance_inference(model_path, conf_threshold):
    """Live feed distance estimation using YOLO detection on bbox width."""
    yolo_model = YOLO(model_path)
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

            # Run YOLO inference
            yolo_results = yolo_model.predict(frame_bgr, conf=conf_threshold, verbose=False)

            best_detection = None
            best_conf = -1.0

            # Find best detection
            for result in yolo_results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy()

                    if conf > best_conf:
                        best_conf = conf
                        best_detection = (xyxy, conf)

            # If we have a detection, estimate distance
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

            # Add frame counter
            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (10, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Live Distance Estimation (Brutal Method)", display_frame)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC to exit
                print("Exiting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Live inference finished. Processed {frame_count} frames.")
