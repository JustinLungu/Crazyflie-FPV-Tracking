from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import TextIO

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_estimation.naive_bbox_depth.constants import (
    CX,
    CY,
    DRONE_WIDTH_M,
    FY,
    FX,
    KEY_NEXT,
    KEY_PREV,
    KEY_QUIT,
    KEY_TOGGLE_PLAY,
    MODEL_PATH,
    NAIVE_CAMERA_MATRIX_PATH,
    NAIVE_ENABLE_RELATIVE_POSITION,
    NAIVE_REVIEW_ALLOW_IMAGE_EXTS,
    NAIVE_REVIEW_DELAY_S,
    NAIVE_REVIEW_LOG_DIR,
    NAIVE_REVIEW_PRINT_EVERY_N_FRAMES,
    NAIVE_REVIEW_SESSION_DIR,
    NAIVE_REVIEW_START_PAUSED,
    NAIVE_REVIEW_TEXT_COLOR,
    NAIVE_REVIEW_TEXT_LINE_HEIGHT,
    NAIVE_REVIEW_TEXT_ORIGIN,
    NAIVE_REVIEW_TEXT_SCALE,
    NAIVE_REVIEW_TEXT_THICKNESS,
    NAIVE_REVIEW_WRITE_LOG,
    NAIVE_REVIEW_WINDOW_NAME,
    NAIVE_INTRINSICS_FALLBACK_TO_MANUAL,
    NAIVE_INTRINSICS_SOURCE,
    NAIVE_Y_AXIS_CONVENTION,
    YOLO_CONF_THRESHOLD,
)
from depth_estimation.naive_bbox_depth.pipeline import NaiveBBoxDepthPipeline
from depth_estimation.naive_bbox_depth.utils import ensure_output_dir, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review a recorded session with YOLO + naive bbox depth estimation."
    )
    parser.add_argument(
        "--session",
        type=str,
        default=NAIVE_REVIEW_SESSION_DIR,
        help="Session folder path containing images/.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=NAIVE_REVIEW_DELAY_S,
        help="Playback delay in seconds between frames while in play mode.",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        default=NAIVE_REVIEW_START_PAUSED,
        help="Start in paused mode.",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default=NAIVE_REVIEW_WINDOW_NAME,
        help="OpenCV window title.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="YOLO model weights path.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=YOLO_CONF_THRESHOLD,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=FX,
        help="Camera focal length in pixels.",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=FY,
        help="Camera focal length in y (pixels).",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=CX,
        help="Principal point x (pixels).",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=CY,
        help="Principal point y (pixels).",
    )
    parser.add_argument(
        "--real-width-m",
        type=float,
        default=DRONE_WIDTH_M,
        help="Real drone width in meters.",
    )
    parser.add_argument(
        "--intrinsics-source",
        type=str,
        default=NAIVE_INTRINSICS_SOURCE,
        choices=("manual", "calibration_npy"),
        help="Intrinsics source selection.",
    )
    parser.add_argument(
        "--camera-matrix-path",
        type=str,
        default=NAIVE_CAMERA_MATRIX_PATH,
        help="Path to camera_matrix.npy when using calibration_npy source.",
    )
    parser.add_argument(
        "--intrinsics-fallback-to-manual",
        action=argparse.BooleanOptionalAction,
        default=NAIVE_INTRINSICS_FALLBACK_TO_MANUAL,
        help="Fallback to manual intrinsics if calibration file is unavailable.",
    )
    parser.add_argument(
        "--enable-relative-position",
        action=argparse.BooleanOptionalAction,
        default=NAIVE_ENABLE_RELATIVE_POSITION,
        help="Compute x/y/z relative position and yaw error.",
    )
    parser.add_argument(
        "--y-axis-convention",
        type=str,
        default=NAIVE_Y_AXIS_CONVENTION,
        choices=("up", "down"),
        help="Sign convention for y relative coordinate.",
    )
    parser.add_argument(
        "--write-log",
        action=argparse.BooleanOptionalAction,
        default=NAIVE_REVIEW_WRITE_LOG,
        help="Write per-frame metrics CSV.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=NAIVE_REVIEW_LOG_DIR,
        help="Directory for per-frame metrics CSV.",
    )
    return parser.parse_args()


def resolve_review_session_dir(session_dir_like: str) -> Path:
    candidate = resolve_repo_path(session_dir_like)
    if not candidate.exists():
        raise RuntimeError(f"Review session path not found: {candidate}")

    images_dir = candidate / "images"
    if not images_dir.is_dir():
        raise RuntimeError(f"Session folder must contain images/: {candidate}")
    return candidate


def collect_review_images(
    session_dir: Path,
    allowed_exts: tuple[str, ...],
) -> list[Path]:
    images_dir = session_dir / "images"
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
    return image_paths


def open_metrics_logger(session_dir: Path, log_dir: str):
    output_dir = ensure_output_dir(log_dir)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"{session_dir.name}_naive_depth_{run_tag}.csv"
    log_file = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "frame_index",
            "image_name",
            "processed_at_iso",
            "infer_ms",
            "track_state",
            "frames_since_detection",
            "estimate_source",
            "is_stale",
            "filter_mode",
            "detection_count",
            "confidence",
            "raw_bbox_width_px",
            "bbox_width_px",
            "raw_bbox_center_x_px",
            "raw_bbox_center_y_px",
            "bbox_center_x_px",
            "bbox_center_y_px",
            "raw_distance_m",
            "distance_m",
            "raw_x_rel_m",
            "raw_y_rel_m",
            "raw_z_rel_m",
            "x_rel_m",
            "y_rel_m",
            "z_rel_m",
            "raw_yaw_error_rad",
            "raw_yaw_error_deg",
            "yaw_error_rad",
            "yaw_error_deg",
        ],
    )
    writer.writeheader()
    return log_path, log_file, writer


def write_metrics_row(
    writer: csv.DictWriter,
    log_file: TextIO,
    frame_index: int,
    image_name: str,
    metrics: dict,
) -> None:
    writer.writerow(
        {
            "frame_index": frame_index,
            "image_name": image_name,
            "processed_at_iso": datetime.now().isoformat(timespec="milliseconds"),
            "infer_ms": metrics.get("infer_ms", ""),
            "track_state": metrics.get("track_state", ""),
            "frames_since_detection": metrics.get("frames_since_detection", ""),
            "estimate_source": metrics.get("estimate_source", ""),
            "is_stale": metrics.get("is_stale", ""),
            "filter_mode": metrics.get("filter_mode", ""),
            "detection_count": metrics.get("detection_count", 0),
            "confidence": metrics.get("confidence", ""),
            "raw_bbox_width_px": metrics.get("raw_bbox_width_px", ""),
            "bbox_width_px": metrics.get("bbox_width_px", ""),
            "raw_bbox_center_x_px": metrics.get("raw_bbox_center_x_px", ""),
            "raw_bbox_center_y_px": metrics.get("raw_bbox_center_y_px", ""),
            "bbox_center_x_px": metrics.get("bbox_center_x_px", ""),
            "bbox_center_y_px": metrics.get("bbox_center_y_px", ""),
            "raw_distance_m": metrics.get("raw_distance_m", ""),
            "distance_m": metrics.get("distance_m", ""),
            "raw_x_rel_m": metrics.get("raw_x_rel_m", ""),
            "raw_y_rel_m": metrics.get("raw_y_rel_m", ""),
            "raw_z_rel_m": metrics.get("raw_z_rel_m", ""),
            "x_rel_m": metrics.get("x_rel_m", ""),
            "y_rel_m": metrics.get("y_rel_m", ""),
            "z_rel_m": metrics.get("z_rel_m", ""),
            "raw_yaw_error_rad": metrics.get("raw_yaw_error_rad", ""),
            "raw_yaw_error_deg": metrics.get("raw_yaw_error_deg", ""),
            "yaw_error_rad": metrics.get("yaw_error_rad", ""),
            "yaw_error_deg": metrics.get("yaw_error_deg", ""),
        }
    )
    log_file.flush()


def draw_session_overlay(
    frame,
    session_name: str,
    image_name: str,
    index: int,
    total: int,
    playing: bool,
    delay_s: float,
    metrics: dict,
) -> None:
    x, y = NAIVE_REVIEW_TEXT_ORIGIN
    status = "PLAY" if playing else "PAUSE"

    track_state = str(metrics.get("track_state", "unknown"))
    detection_count = int(metrics.get("detection_count", 0))
    infer_ms = float(metrics.get("infer_ms", 0.0))
    bbox_width_px = metrics.get("bbox_width_px")

    bbox_text = "bbox_w: n/a"
    if bbox_width_px is not None:
        bbox_text = f"bbox_w: {float(bbox_width_px):.1f}px"

    lines = [
        f"{status}  frame {index + 1}/{total}  detections: {detection_count}  state: {track_state}",
        f"inference: {infer_ms:.1f} ms  delay: {delay_s:.2f}s  {bbox_text}",
        f"session: {session_name}",
        f"image: {image_name}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * NAIVE_REVIEW_TEXT_LINE_HEIGHT),
            cv2.FONT_HERSHEY_SIMPLEX,
            NAIVE_REVIEW_TEXT_SCALE,
            NAIVE_REVIEW_TEXT_COLOR,
            NAIVE_REVIEW_TEXT_THICKNESS,
            cv2.LINE_AA,
        )


def ensure_processed_upto_index(
    target_index: int,
    image_paths: list[Path],
    processed_frames: list,
    processed_metrics: list[dict],
    pipeline: NaiveBBoxDepthPipeline,
    log_writer: csv.DictWriter | None,
    log_file: TextIO | None,
) -> tuple[list, list[dict], list[Path]]:
    while len(processed_frames) <= target_index and len(processed_frames) < len(image_paths):
        frame_index = len(processed_frames)
        image_path = image_paths[frame_index]
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: could not read image {image_path}. Skipping.")
            image_paths.pop(frame_index)
            if not image_paths:
                break
            continue

        output = pipeline.process_live_frame(frame)
        processed_frames.append(output.frame_bgr)
        processed_metrics.append(dict(output.metrics))

        if log_writer is not None and log_file is not None:
            write_metrics_row(
                writer=log_writer,
                log_file=log_file,
                frame_index=frame_index,
                image_name=image_path.name,
                metrics=output.metrics,
            )

        if frame_index % max(1, NAIVE_REVIEW_PRINT_EVERY_N_FRAMES) == 0:
            print(f"Processed {frame_index + 1}/{len(image_paths)} frames for review.")

    return processed_frames, processed_metrics, image_paths


def main() -> None:
    args = parse_args()

    pipeline = NaiveBBoxDepthPipeline(
        model_path=args.model_path,
        conf_threshold=float(args.conf_threshold),
        fx=float(args.fx),
        fy=float(args.fy),
        cx=float(args.cx),
        cy=float(args.cy),
        real_width_m=float(args.real_width_m),
        intrinsics_source=str(args.intrinsics_source),
        camera_matrix_path=str(args.camera_matrix_path),
        intrinsics_fallback_to_manual=bool(args.intrinsics_fallback_to_manual),
        enable_relative_position=bool(args.enable_relative_position),
        y_axis_convention=str(args.y_axis_convention),
    )

    session_dir = resolve_review_session_dir(args.session)
    image_paths = collect_review_images(
        session_dir=session_dir,
        allowed_exts=NAIVE_REVIEW_ALLOW_IMAGE_EXTS,
    )

    delay_s = max(0.0, float(args.delay))
    delay_ms = max(1, int(delay_s * 1000))
    playing = not args.start_paused

    print(f"Review session: {session_dir}")
    print(f"Images: {len(image_paths)}")
    print(f"Model: {args.model_path}")
    print(
        "intrinsics="
        f"fx={pipeline.fx:.3f}, fy={pipeline.fy:.3f}, cx={pipeline.cx:.3f}, cy={pipeline.cy:.3f} "
        f"[{pipeline.intrinsics_loaded_from}]"
    )
    print(
        f"relative_position={pipeline.enable_relative_position}, y_axis={pipeline.y_axis_convention}, "
        f"real_width_m={float(args.real_width_m):.4f}"
    )
    print(
        "filter mode="
        f"{pipeline.filter_mode}, dist={pipeline.filter_distance}, center={pipeline.filter_center}, width={pipeline.filter_width}"
    )
    print(
        f"dropout hold/stale={pipeline.dropout_hold_frames}/{pipeline.dropout_stale_frames}"
    )
    print("Controls")
    print("space: play/pause")
    print("a or Left Arrow: previous frame")
    print("d or Right Arrow: next frame")
    print("q or ESC: quit")

    log_path = None
    log_file = None
    log_writer = None
    if args.write_log:
        log_path, log_file, log_writer = open_metrics_logger(
            session_dir=session_dir,
            log_dir=args.log_dir,
        )
        print(f"Frame log: {log_path}")

    index = 0
    processed_frames: list = []
    processed_metrics: list[dict] = []

    try:
        while True:
            processed_frames, processed_metrics, image_paths = ensure_processed_upto_index(
                target_index=index,
                image_paths=image_paths,
                processed_frames=processed_frames,
                processed_metrics=processed_metrics,
                pipeline=pipeline,
                log_writer=log_writer,
                log_file=log_file,
            )
            if not image_paths:
                print("No readable images left. Exiting.")
                break

            index = min(index, len(image_paths) - 1)
            image_path = image_paths[index]
            display = processed_frames[index].copy()
            frame_metrics = processed_metrics[index]
            draw_session_overlay(
                frame=display,
                session_name=session_dir.name,
                image_name=image_path.name,
                index=index,
                total=len(image_paths),
                playing=playing,
                delay_s=delay_s,
                metrics=frame_metrics,
            )
            cv2.imshow(args.window_name, display)
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
    finally:
        if log_file is not None:
            log_file.close()
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    main()
