from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import TextIO

import cv2
import numpy as np

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
    KEY_TOGGLE_GATING,
    KEY_TOGGLE_PLAY,
    MODEL_PATH,
    NAIVE_CAMERA_MATRIX_PATH,
    NAIVE_ENABLE_RELATIVE_POSITION,
    NAIVE_INTRINSICS_FALLBACK_TO_MANUAL,
    NAIVE_INTRINSICS_SOURCE,
    NAIVE_REVIEW_ALLOW_IMAGE_EXTS,
    NAIVE_REVIEW_DELAY_S,
    NAIVE_REVIEW_LOG_DIR,
    NAIVE_REVIEW_PRINT_EVERY_N_FRAMES,
    NAIVE_REVIEW_SESSION_DIR,
    NAIVE_REVIEW_SIDE_PANEL_ACCENT_COLOR,
    NAIVE_REVIEW_SIDE_PANEL_BG_COLOR,
    NAIVE_REVIEW_SIDE_PANEL_TEXT_COLOR,
    NAIVE_REVIEW_SIDE_PANEL_WIDTH,
    NAIVE_REVIEW_START_PAUSED,
    NAIVE_REVIEW_TEXT_COLOR,
    NAIVE_REVIEW_TEXT_LINE_HEIGHT,
    NAIVE_REVIEW_TEXT_ORIGIN,
    NAIVE_REVIEW_TEXT_SCALE,
    NAIVE_REVIEW_TEXT_THICKNESS,
    NAIVE_REVIEW_USE_SIDE_PANEL,
    NAIVE_REVIEW_WINDOW_NAME,
    NAIVE_REVIEW_WRITE_LOG,
    NAIVE_Y_AXIS_CONVENTION,
    YOLO_CONF_THRESHOLD,
)
from depth_estimation.naive_bbox_depth.pipeline import NaiveBBoxDepthPipeline
from depth_estimation.naive_bbox_depth.utils import ensure_output_dir, resolve_repo_path


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
            "infer_fps",
            "process_ms",
            "process_fps",
            "track_state",
            "frames_since_detection",
            "estimate_source",
            "is_stale",
            "filter_mode",
            "gating_enabled",
            "gating_passed",
            "gating_reasons",
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
            "infer_fps": metrics.get("infer_fps", ""),
            "process_ms": metrics.get("process_ms", ""),
            "process_fps": metrics.get("process_fps", ""),
            "track_state": metrics.get("track_state", ""),
            "frames_since_detection": metrics.get("frames_since_detection", ""),
            "estimate_source": metrics.get("estimate_source", ""),
            "is_stale": metrics.get("is_stale", ""),
            "filter_mode": metrics.get("filter_mode", ""),
            "gating_enabled": metrics.get("gating_enabled", ""),
            "gating_passed": metrics.get("gating_passed", ""),
            "gating_reasons": metrics.get("gating_reasons", ""),
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


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value, decimals: int, unit: str = "") -> str:
    v = _as_float(value)
    if v is None:
        return "n/a"
    return f"{v:.{decimals}f}{unit}"


def compose_review_display(
    frame,
    session_name: str,
    image_name: str,
    index: int,
    total: int,
    playing: bool,
    delay_s: float,
    metrics: dict,
) -> np.ndarray:
    status = "PLAY" if playing else "PAUSE"

    track_state = str(metrics.get("track_state", "unknown"))
    estimate_source = str(metrics.get("estimate_source", "none"))
    detection_count = int(metrics.get("detection_count", 0))

    infer_ms = _as_float(metrics.get("infer_ms"))
    process_fps = _as_float(metrics.get("process_fps"))
    # Dist/conf are shown near bbox on the video frame; keep panel for state/position.
    conf = _as_float(metrics.get("confidence"))
    raw_dist = _as_float(metrics.get("raw_distance_m"))
    filt_dist = _as_float(metrics.get("distance_m"))
    x_rel = _as_float(metrics.get("x_rel_m"))
    y_rel = _as_float(metrics.get("y_rel_m"))
    z_rel = _as_float(metrics.get("z_rel_m"))
    yaw_deg = _as_float(metrics.get("yaw_error_deg"))

    try:
        gating_enabled = int(metrics.get("gating_enabled", 0))
    except (TypeError, ValueError):
        gating_enabled = 0
    try:
        gating_passed = int(metrics.get("gating_passed", -1))
    except (TypeError, ValueError):
        gating_passed = -1

    gating_label = "OFF"
    if gating_enabled:
        if gating_passed == 1:
            gating_label = "PASS"
        elif gating_passed == 0:
            gating_label = "REJECT"
        else:
            gating_label = "ON"

    if not NAIVE_REVIEW_USE_SIDE_PANEL:
        x, y = NAIVE_REVIEW_TEXT_ORIGIN
        lines = [
            f"{status} frame {index + 1}/{total} det: {detection_count} state: {track_state}",
            (
                f"infer: {_format_value(infer_ms, 1, ' ms')} "
                f"fps(no delay): {_format_value(process_fps, 1)} "
                f"delay: {delay_s:.2f}s gate: {gating_label}"
            ),
            (
                f"conf: {_format_value(conf, 2)} "
                f"raw/filt dist: {_format_value(raw_dist, 3)}/{_format_value(filt_dist, 3)} m"
            ),
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
        return frame

    h, w = frame.shape[:2]
    panel_w = max(240, int(NAIVE_REVIEW_SIDE_PANEL_WIDTH))
    canvas = np.zeros((h, w + panel_w, 3), dtype=frame.dtype)
    canvas[:, :w] = frame
    canvas[:, w:] = NAIVE_REVIEW_SIDE_PANEL_BG_COLOR

    px = w + 14
    py = 28
    line_h = 22

    text_color = tuple(int(c) for c in NAIVE_REVIEW_SIDE_PANEL_TEXT_COLOR)
    accent_color = tuple(int(c) for c in NAIVE_REVIEW_SIDE_PANEL_ACCENT_COLOR)

    lines: list[tuple[str, tuple[int, int, int], float]] = [
        ("Telemetry", accent_color, 0.72),
        (f"Mode: {status}", text_color, 0.56),
        (f"Frame: {index + 1}/{total}", text_color, 0.56),
        (f"Track state: {track_state}", text_color, 0.56),
        (f"Estimate src: {estimate_source}", text_color, 0.56),
        (f"Detections: {detection_count}", text_color, 0.56),
        (f"Inference: {_format_value(infer_ms, 1, ' ms')}", text_color, 0.56),
        (f"FPS (no delay): {_format_value(process_fps, 1)}", text_color, 0.56),
        (f"Delay: {delay_s:.2f} s", text_color, 0.56),
        ("", text_color, 0.56),
        (
            f"Gating: {gating_label}",
            accent_color if gating_enabled else text_color,
            0.56,
        ),
    ]

    gating_reasons = str(metrics.get("gating_reasons", "")).strip()
    if gating_reasons:
        for reason in [r for r in gating_reasons.split("|") if r][:3]:
            reason_text = reason if len(reason) <= 34 else reason[:31] + "..."
            lines.append((f" - {reason_text}", text_color, 0.52))

    lines.extend(
        [
            ("", text_color, 0.56),
            (f"X: {_format_value(x_rel, 3, ' m')}", text_color, 0.56),
            (f"Y: {_format_value(y_rel, 3, ' m')}", text_color, 0.56),
            (f"Z: {_format_value(z_rel, 3, ' m')}", text_color, 0.56),
            (f"Yaw err: {_format_value(yaw_deg, 1, ' deg')}", text_color, 0.56),
            ("", text_color, 0.56),
            (f"Session: {session_name}", text_color, 0.50),
            (
                "Image: " + (image_name if len(image_name) <= 30 else image_name[:27] + "..."),
                text_color,
                0.50,
            ),
        ]
    )

    for line, color, scale in lines:
        if line:
            cv2.putText(
                canvas,
                line,
                (px, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(scale),
                color,
                2,
                cv2.LINE_AA,
            )
        py += line_h

    cv2.line(canvas, (w, 0), (w, h - 1), (80, 80, 80), 1)
    return canvas


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
    pipeline = NaiveBBoxDepthPipeline(
        model_path=MODEL_PATH,
        conf_threshold=float(YOLO_CONF_THRESHOLD),
        fx=float(FX),
        fy=float(FY),
        cx=float(CX),
        cy=float(CY),
        real_width_m=float(DRONE_WIDTH_M),
        intrinsics_source=str(NAIVE_INTRINSICS_SOURCE),
        camera_matrix_path=str(NAIVE_CAMERA_MATRIX_PATH),
        intrinsics_fallback_to_manual=bool(NAIVE_INTRINSICS_FALLBACK_TO_MANUAL),
        enable_relative_position=bool(NAIVE_ENABLE_RELATIVE_POSITION),
        y_axis_convention=str(NAIVE_Y_AXIS_CONVENTION),
    )

    session_dir = resolve_review_session_dir(NAIVE_REVIEW_SESSION_DIR)
    image_paths = collect_review_images(
        session_dir=session_dir,
        allowed_exts=NAIVE_REVIEW_ALLOW_IMAGE_EXTS,
    )

    delay_s = max(0.0, float(NAIVE_REVIEW_DELAY_S))
    delay_ms = max(1, int(delay_s * 1000))
    playing = not NAIVE_REVIEW_START_PAUSED

    print(f"Review session: {session_dir}")
    print(f"Images: {len(image_paths)}")
    print(f"Model: {MODEL_PATH}")
    print(
        "intrinsics="
        f"fx={pipeline.fx:.3f}, fy={pipeline.fy:.3f}, cx={pipeline.cx:.3f}, cy={pipeline.cy:.3f} "
        f"[{pipeline.intrinsics_loaded_from}]"
    )
    print(
        f"relative_position={pipeline.enable_relative_position}, y_axis={pipeline.y_axis_convention}, "
        f"real_width_m={float(DRONE_WIDTH_M):.4f}"
    )
    print(
        "filter mode="
        f"{pipeline.filter_mode}, dist={pipeline.filter_distance}, center={pipeline.filter_center}, width={pipeline.filter_width}"
    )
    print(f"dropout hold/stale={pipeline.dropout_hold_frames}/{pipeline.dropout_stale_frames}")
    print(f"gating={'ON' if pipeline.gating_enabled else 'OFF'}")
    print(f"side_panel={'ON' if NAIVE_REVIEW_USE_SIDE_PANEL else 'OFF'}")
    print("Controls")
    print("space: play/pause")
    print("a or Left Arrow: previous frame")
    print("d or Right Arrow: next frame")
    print("g: toggle gating + reprocess timeline")
    print("q or ESC: quit")

    log_path = None
    log_file = None
    log_writer = None
    if NAIVE_REVIEW_WRITE_LOG:
        log_path, log_file, log_writer = open_metrics_logger(
            session_dir=session_dir,
            log_dir=NAIVE_REVIEW_LOG_DIR,
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
            display = compose_review_display(
                frame=processed_frames[index].copy(),
                session_name=session_dir.name,
                image_name=image_path.name,
                index=index,
                total=len(image_paths),
                playing=playing,
                delay_s=delay_s,
                metrics=processed_metrics[index],
            )

            cv2.imshow(NAIVE_REVIEW_WINDOW_NAME, display)
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
            if key in KEY_TOGGLE_GATING:
                new_state = pipeline.toggle_gating()
                pipeline.reset_temporal_state()
                processed_frames.clear()
                processed_metrics.clear()
                playing = False

                if log_file is not None:
                    log_file.close()
                    log_path, log_file, log_writer = open_metrics_logger(
                        session_dir=session_dir,
                        log_dir=NAIVE_REVIEW_LOG_DIR,
                    )
                    print(f"Frame log (new toggle state): {log_path}")

                print(
                    f"[review] gating={'ON' if new_state else 'OFF'}; "
                    f"reprocessing cached timeline up to frame {index + 1}."
                )
                continue
    finally:
        if log_file is not None:
            log_file.close()
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    main()
