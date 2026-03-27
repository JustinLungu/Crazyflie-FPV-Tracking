from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_estimation.naive_bbox_depth.constants import BUFFER_SIZE, DEVICE, FOURCC, FPS_HINT, HEIGHT, WIDTH
from depth_estimation.pipeline_base import LiveDepthPipeline, LiveFrameOutput


PIPELINE_SPECS: dict[str, tuple[str, str]] = {
    "naive": ("depth_estimation.naive_bbox_depth.pipeline", "NaiveBBoxDepthPipeline"),
    "unidepth": ("depth_estimation.unidepth.pipeline", "UniDepthPipeline"),
    "midas": ("depth_estimation.midas.pipeline", "MiDaSPipeline"),
}


def parse_methods(raw_methods: str) -> list[str]:
    if not raw_methods.strip():
        raise ValueError("--methods cannot be empty.")

    parts = [m.strip().lower() for m in raw_methods.split(",") if m.strip()]
    if not parts:
        raise ValueError("No valid methods provided.")

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


def open_camera(
    device: str,
    width: int,
    height: int,
    fps_hint: int,
    fourcc: str,
    buffer_size: int,
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_hint)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera at {device}")

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


def format_metric_text(outputs: list[LiveFrameOutput]) -> list[str]:
    lines: list[str] = []
    for out in outputs:
        parts = [out.method]
        if "center_depth" in out.metrics:
            parts.append(f"center={out.metrics['center_depth']:.3f}")
        if "distance_m" in out.metrics:
            parts.append(f"dist={out.metrics['distance_m']:.3f}m")
        if "infer_ms" in out.metrics:
            parts.append(f"{out.metrics['infer_ms']:.1f}ms")
        lines.append(" | ".join(parts))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run live depth estimation with one or multiple methods. "
            "Examples: --methods naive OR --methods naive,unidepth OR --methods midas"
        )
    )
    parser.add_argument("--methods", type=str, default="naive", help="Comma-separated methods: naive,unidepth,midas")
    parser.add_argument("--device", type=str, default=DEVICE, help="Camera device (e.g., /dev/video2)")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--fps", type=int, default=FPS_HINT)
    parser.add_argument("--fourcc", type=str, default=FOURCC)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    parser.add_argument("--window-name", type=str, default="Live Depth Estimation")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    pipelines: list[LiveDepthPipeline] = [build_pipeline(m) for m in methods]

    print("Live depth methods:", ", ".join(methods))

    cap = open_camera(
        device=args.device,
        width=args.width,
        height=args.height,
        fps_hint=args.fps,
        fourcc=args.fourcc,
        buffer_size=args.buffer_size,
    )

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame_idx += 1
            outputs = [pipeline.process_live_frame(frame_bgr) for pipeline in pipelines]
            combined = combine_frames(outputs, target_height=args.height)

            metric_lines = format_metric_text(outputs)
            cv2.putText(
                combined,
                f"frame: {frame_idx}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            y = 50
            for line in metric_lines:
                cv2.putText(
                    combined,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 24

            cv2.imshow(args.window_name, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                print("Stopped by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for pipeline in pipelines:
            pipeline.close()


if __name__ == "__main__":
    main()
