from __future__ import annotations

from collections import OrderedDict
import importlib
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_estimation.constants import (
    DEPTH_LIVE_REVIEW_BUFFER_SIZE,
    DEPTH_LIVE_REVIEW_DEVICE,
    DEPTH_LIVE_REVIEW_FOURCC,
    DEPTH_LIVE_REVIEW_FPS_HINT,
    DEPTH_LIVE_REVIEW_HEIGHT,
    DEPTH_LIVE_REVIEW_LINE_HEIGHT,
    DEPTH_LIVE_REVIEW_METHODS,
    DEPTH_LIVE_REVIEW_SIDE_PANEL_ACCENT_COLOR,
    DEPTH_LIVE_REVIEW_SIDE_PANEL_BG_COLOR,
    DEPTH_LIVE_REVIEW_SIDE_PANEL_TEXT_COLOR,
    DEPTH_LIVE_REVIEW_SIDE_PANEL_WIDTH,
    DEPTH_LIVE_REVIEW_TEXT_THICKNESS,
    DEPTH_LIVE_REVIEW_USE_SIDE_PANEL,
    DEPTH_LIVE_REVIEW_WIDTH,
    DEPTH_LIVE_REVIEW_WINDOW_NAME,
    KEY_QUIT,
    KEY_TOGGLE_GATING,
)
from depth_estimation.pipeline_base import LiveDepthPipeline, LiveFrameOutput


PIPELINE_SPECS: dict[str, tuple[str, str]] = {
    "naive": ("depth_estimation.naive_bbox_depth.pipeline", "NaiveBBoxDepthPipeline"),
    "unidepth": ("depth_estimation.unidepth.pipeline", "UniDepthPipeline"),
    "midas": ("depth_estimation.midas.pipeline", "MiDaSPipeline"),
}


def parse_methods(raw_methods: tuple[str, ...] | list[str]) -> list[str]:
    parts = [str(m).strip().lower() for m in raw_methods if str(m).strip()]
    if not parts:
        raise ValueError("DEPTH_LIVE_REVIEW_METHODS cannot be empty.")
    ordered_unique = list(OrderedDict.fromkeys(parts))
    unsupported = [m for m in ordered_unique if m not in PIPELINE_SPECS]
    if unsupported:
        supported = ", ".join(sorted(PIPELINE_SPECS))
        raise ValueError(f"Unsupported method(s): {unsupported}. Supported: {supported}")
    return ordered_unique


def build_pipeline(method: str) -> LiveDepthPipeline:
    module_name, class_name = PIPELINE_SPECS[method]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    pipeline = cls()
    if not isinstance(pipeline, LiveDepthPipeline):
        raise TypeError(f"{class_name} must inherit from LiveDepthPipeline")
    return pipeline


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(DEPTH_LIVE_REVIEW_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*DEPTH_LIVE_REVIEW_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEPTH_LIVE_REVIEW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEPTH_LIVE_REVIEW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, DEPTH_LIVE_REVIEW_FPS_HINT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, DEPTH_LIVE_REVIEW_BUFFER_SIZE)
    if not cap.isOpened():
        cap = cv2.VideoCapture(DEPTH_LIVE_REVIEW_DEVICE)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera at {DEPTH_LIVE_REVIEW_DEVICE}")
    return cap


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    if frame.shape[0] == target_height:
        return frame
    scale = target_height / frame.shape[0]
    new_width = max(1, int(frame.shape[1] * scale))
    return cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_AREA)


def combine_frames(outputs: list[LiveFrameOutput], target_height: int) -> np.ndarray:
    frames = [resize_to_height(out.frame_bgr, target_height) for out in outputs]
    if len(frames) == 1:
        return frames[0]
    return np.hstack(frames)


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value, decimals: int = 2, suffix: str = "") -> str:
    v = _as_float(value)
    if v is None:
        return "n/a"
    return f"{v:.{decimals}f}{suffix}"


def _method_lines(
    out: LiveFrameOutput,
    last_pose: dict[str, float] | None,
    include_method_name: bool,
) -> list[str]:
    m = out.metrics
    lines: list[str] = []
    if include_method_name:
        lines.append(f"Method: {out.method}")

    track_state = str(m.get("track_state", "unknown"))
    estimate_source = str(m.get("estimate_source", "none"))
    lines.append(f"Track state: {track_state}")
    lines.append(f"Estimate src: {estimate_source}")

    det = m.get("detection_count")
    yolo_det = m.get("yolo_detection_count")
    if det is not None:
        if yolo_det is not None:
            lines.append(f"Detections: {det} (YOLO: {yolo_det})")
        else:
            lines.append(f"Detections: {det}")

    infer_ms = _as_float(m.get("infer_ms"))
    if infer_ms is not None:
        lines.append(f"Inference: {infer_ms:.1f} ms")
    process_fps = _as_float(m.get("process_fps"))
    if process_fps is not None:
        lines.append(f"FPS (no delay): {process_fps:.1f}")

    if m.get("gating_enabled") is not None:
        try:
            enabled = int(m.get("gating_enabled", 0))
        except (TypeError, ValueError):
            enabled = 0
        try:
            passed = int(m.get("gating_passed", -1))
        except (TypeError, ValueError):
            passed = -1
        label = "OFF"
        if enabled:
            label = "PASS" if passed == 1 else "REJECT" if passed == 0 else "ON"
        lines.append(f"Gating: {label}")
        reasons = str(m.get("gating_reasons", "")).strip()
        if reasons:
            reason_bits = [r for r in reasons.split("|") if r]
            if reason_bits:
                text = ", ".join(reason_bits[:2])
                if len(reason_bits) > 2:
                    text += ", ..."
                lines.append(f" - {text}")

    x_rel = _as_float(m.get("x_rel_m"))
    y_rel = _as_float(m.get("y_rel_m"))
    z_rel = _as_float(m.get("z_rel_m"))
    yaw_deg = _as_float(m.get("yaw_error_deg"))
    using_last = False
    if (
        x_rel is None
        or y_rel is None
        or z_rel is None
        or yaw_deg is None
    ) and last_pose is not None:
        x_rel = _as_float(last_pose.get("x_rel_m"))
        y_rel = _as_float(last_pose.get("y_rel_m"))
        z_rel = _as_float(last_pose.get("z_rel_m"))
        yaw_deg = _as_float(last_pose.get("yaw_error_deg"))
        using_last = x_rel is not None and y_rel is not None and z_rel is not None and yaw_deg is not None

    if x_rel is not None and y_rel is not None and z_rel is not None and yaw_deg is not None:
        suffix = " (last)" if using_last else ""
        lines.append("")
        lines.append(f"X{suffix}: {x_rel:.3f} m")
        lines.append(f"Y{suffix}: {y_rel:.3f} m")
        lines.append(f"Z{suffix}: {z_rel:.3f} m")
        lines.append(f"Yaw err{suffix}: {yaw_deg:.1f} deg")

    if m.get("distance_m") is not None and (x_rel is None or y_rel is None or z_rel is None):
        lines.append(f"Dist: {_fmt(m.get('distance_m'), 3, ' m')}")

    return lines


def compose_display(
    combined_frame: np.ndarray,
    frame_idx: int,
    loop_fps: float,
    methods: list[str],
    outputs: list[LiveFrameOutput],
    last_pose_by_method: dict[str, dict[str, float]],
) -> np.ndarray:
    if not DEPTH_LIVE_REVIEW_USE_SIDE_PANEL:
        overlay = combined_frame.copy()
        y = 24
        lines = [
            "Live Telemetry",
            f"Mode: LIVE",
            f"Frame: {frame_idx}",
            f"Loop FPS: {loop_fps:.1f}",
            "",
        ]
        if len(methods) > 1:
            lines.insert(4, "Methods: " + ",".join(methods))
        for out in outputs:
            lines.extend(
                _method_lines(
                    out,
                    last_pose_by_method.get(out.method),
                    include_method_name=(len(outputs) > 1),
                )
            )
            lines.append("")
        for line in lines:
            cv2.putText(
                overlay,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                DEPTH_LIVE_REVIEW_TEXT_THICKNESS,
                cv2.LINE_AA,
            )
            y += DEPTH_LIVE_REVIEW_LINE_HEIGHT
        return overlay

    h, w = combined_frame.shape[:2]
    panel_w = max(260, int(DEPTH_LIVE_REVIEW_SIDE_PANEL_WIDTH))
    canvas = np.zeros((h, w + panel_w, 3), dtype=combined_frame.dtype)
    canvas[:, :w] = combined_frame
    canvas[:, w:] = DEPTH_LIVE_REVIEW_SIDE_PANEL_BG_COLOR

    text_color = tuple(int(c) for c in DEPTH_LIVE_REVIEW_SIDE_PANEL_TEXT_COLOR)
    accent_color = tuple(int(c) for c in DEPTH_LIVE_REVIEW_SIDE_PANEL_ACCENT_COLOR)

    x = w + 14
    y = 28

    lines: list[tuple[str, tuple[int, int, int], float]] = [
        ("Telemetry", accent_color, 0.72),
        ("Mode: LIVE", text_color, 0.56),
        (f"Frame: {frame_idx}", text_color, 0.56),
        (f"Loop FPS: {loop_fps:.1f}", text_color, 0.56),
        ("", text_color, 0.56),
    ]
    if len(methods) > 1:
        lines.insert(4, (f"Methods: {','.join(methods)}", text_color, 0.56))
    for out in outputs:
        method_color = accent_color if out.method == "naive" else text_color
        method_lines = _method_lines(
            out,
            last_pose_by_method.get(out.method),
            include_method_name=(len(outputs) > 1),
        )
        for i, line in enumerate(method_lines):
            scale = 0.56 if i == 0 else 0.55
            color = method_color if line.startswith("Method:") else text_color
            lines.append((line, color, scale))
        lines.append(("", text_color, 0.56))

    controls_text = "Controls: q/ESC quit, g toggle gating"

    controls_y = h - 12
    max_text_baseline_y = controls_y - DEPTH_LIVE_REVIEW_LINE_HEIGHT

    truncated = False
    for line, color, scale in lines:
        if y > max_text_baseline_y:
            truncated = True
            break
        if line:
            cv2.putText(
                canvas,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(scale),
                color,
                DEPTH_LIVE_REVIEW_TEXT_THICKNESS,
                cv2.LINE_AA,
            )
        y += DEPTH_LIVE_REVIEW_LINE_HEIGHT

    if truncated and y <= max_text_baseline_y:
        cv2.putText(
            canvas,
            "...",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            DEPTH_LIVE_REVIEW_TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        controls_text,
        (x, controls_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        text_color,
        DEPTH_LIVE_REVIEW_TEXT_THICKNESS,
        cv2.LINE_AA,
    )

    cv2.line(canvas, (w, 0), (w, h - 1), (80, 80, 80), 1)
    return canvas


def toggle_available_gating(pipelines: list[LiveDepthPipeline]) -> list[tuple[str, bool]]:
    toggled: list[tuple[str, bool]] = []
    for pipeline in pipelines:
        toggle_fn = getattr(pipeline, "toggle_gating", None)
        if callable(toggle_fn):
            new_state = bool(toggle_fn())
            toggled.append((pipeline.name, new_state))
    return toggled


def main() -> None:
    methods = parse_methods(DEPTH_LIVE_REVIEW_METHODS)
    pipelines: list[LiveDepthPipeline] = [build_pipeline(m) for m in methods]
    cap = open_camera()

    print("Live depth review started.")
    print("Methods:", ", ".join(methods))
    print("Controls: q/ESC quit, g toggle gating (if supported by loaded method).")

    frame_idx = 0
    last_t = time.perf_counter()
    loop_fps = 0.0
    last_pose_by_method: dict[str, dict[str, float]] = {}

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame_idx += 1
            outputs = [pipeline.process_live_frame(frame_bgr) for pipeline in pipelines]
            for out in outputs:
                m = out.metrics
                x_rel = _as_float(m.get("x_rel_m"))
                y_rel = _as_float(m.get("y_rel_m"))
                z_rel = _as_float(m.get("z_rel_m"))
                yaw_deg = _as_float(m.get("yaw_error_deg"))
                if x_rel is not None and y_rel is not None and z_rel is not None and yaw_deg is not None:
                    last_pose_by_method[out.method] = {
                        "x_rel_m": x_rel,
                        "y_rel_m": y_rel,
                        "z_rel_m": z_rel,
                        "yaw_error_deg": yaw_deg,
                    }
            combined = combine_frames(outputs, target_height=DEPTH_LIVE_REVIEW_HEIGHT)

            now = time.perf_counter()
            dt = max(1e-6, now - last_t)
            inst_fps = 1.0 / dt
            loop_fps = inst_fps if loop_fps <= 0.0 else (0.2 * inst_fps + 0.8 * loop_fps)
            last_t = now

            display = compose_display(
                combined_frame=combined,
                frame_idx=frame_idx,
                loop_fps=loop_fps,
                methods=methods,
                outputs=outputs,
                last_pose_by_method=last_pose_by_method,
            )

            cv2.imshow(DEPTH_LIVE_REVIEW_WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in KEY_QUIT:
                print("Stopped by user.")
                break
            if key in KEY_TOGGLE_GATING:
                toggled = toggle_available_gating(pipelines)
                if not toggled:
                    print("[live-review] no loaded pipeline exposes gating toggle.")
                else:
                    summary = ", ".join([f"{name}={'ON' if state else 'OFF'}" for name, state in toggled])
                    print(f"[live-review] {summary}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for pipeline in pipelines:
            pipeline.close()


if __name__ == "__main__":
    main()
