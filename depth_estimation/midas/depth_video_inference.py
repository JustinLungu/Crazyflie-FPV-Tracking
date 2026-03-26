import time

import cv2
import numpy as np

from constants import *
from midas_model import MiDaSModel
from utils import (
    colorize_depth_map,
    compute_center_depth,
    ensure_parent_dir,
    resolve_repo_path,
    resize_depth_to_frame,
)


def draw_overlay(frame: np.ndarray, lines: list[str]) -> None:
    x, y = MIDAS_TEXT_ORIGIN
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * MIDAS_TEXT_LINE_HEIGHT),
            cv2.FONT_HERSHEY_SIMPLEX,
            MIDAS_TEXT_SCALE,
            MIDAS_TEXT_COLOR,
            MIDAS_TEXT_THICKNESS,
            cv2.LINE_AA,
        )


def build_output_writer(width: int, height: int, fps: float) -> cv2.VideoWriter:
    output_path = resolve_repo_path(MIDAS_VIDEO_OUTPUT_PATH)
    ensure_parent_dir(output_path)

    output_size = (width * 2, height)
    if fps <= 1e-6:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_path}")
    return writer


def main() -> None:
    video_path = resolve_repo_path(MIDAS_VIDEO_INPUT_PATH)
    if not video_path.exists():
        raise RuntimeError(
            f"Video not found: {video_path}\n"
            "Set MIDAS_VIDEO_INPUT_PATH in depth_estimation/midas/constants.py."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = MiDaSModel(model_type=MIDAS_MODEL_TYPE, device=MIDAS_DEVICE)
    writer = build_output_writer(width, height, fps) if MIDAS_VIDEO_WRITE_OUTPUT else None

    print("Depth inference started (video, MiDaS).")
    print(f"- input video: {video_path}")
    print(f"- size: {width}x{height} @ {fps:.2f} fps")
    print(f"- frames: {total_frames if total_frames > 0 else 'unknown'}")
    if writer is not None:
        print(f"- output video: {resolve_repo_path(MIDAS_VIDEO_OUTPUT_PATH)}")

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_idx += 1
            t0 = time.perf_counter()

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            depth_map = model.predict(frame_rgb)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            depth_map = resize_depth_to_frame(depth_map, width, height)
            center_depth = compute_center_depth(depth_map, MIDAS_CENTER_PATCH_SIZE)

            depth_color = colorize_depth_map(
                depth_map,
                MIDAS_COLORMAP,
                invert_colormap=MIDAS_INVERT_COLORMAP,
            )
            composed = np.hstack([frame_bgr, depth_color])

            draw_overlay(
                composed,
                [
                    f"frame: {frame_idx}",
                    f"inference: {infer_ms:.1f} ms",
                    f"center depth: {center_depth:.3f}",
                ],
            )

            if writer is not None:
                writer.write(composed)

            if MIDAS_VIDEO_SHOW_PREVIEW:
                cv2.imshow(MIDAS_VIDEO_WINDOW_NAME, composed)
                key = cv2.waitKey(1) & 0xFF
                if key in KEY_QUIT:
                    print("Stopped by user.")
                    break

            if frame_idx % max(1, MIDAS_PRINT_EVERY_N_FRAMES) == 0:
                print(
                    f"frame {frame_idx}: center_depth={center_depth:.3f}, "
                    f"infer_ms={infer_ms:.1f}"
                )

            if MIDAS_VIDEO_MAX_FRAMES > 0 and frame_idx >= MIDAS_VIDEO_MAX_FRAMES:
                print(f"Stopped at MIDAS_VIDEO_MAX_FRAMES={MIDAS_VIDEO_MAX_FRAMES}.")
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print("Depth inference finished (video, MiDaS).")
    print(f"- processed frames: {frame_idx}")


if __name__ == "__main__":
    main()
