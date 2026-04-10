from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_estimation.naive_bbox_depth.constants import (
    DRONE_WIDTH_M,
    FX,
    KEY_NEXT,
    KEY_PREV,
    KEY_QUIT,
    KEY_TOGGLE_PLAY,
    MODEL_PATH,
    NAIVE_REVIEW_ALLOW_IMAGE_EXTS,
    NAIVE_REVIEW_DELAY_S,
    NAIVE_REVIEW_SESSION_DIR,
    NAIVE_REVIEW_START_PAUSED,
    NAIVE_REVIEW_TEXT_COLOR,
    NAIVE_REVIEW_TEXT_LINE_HEIGHT,
    NAIVE_REVIEW_TEXT_ORIGIN,
    NAIVE_REVIEW_TEXT_SCALE,
    NAIVE_REVIEW_TEXT_THICKNESS,
    NAIVE_REVIEW_WINDOW_NAME,
    YOLO_CONF_THRESHOLD,
)
from depth_estimation.naive_bbox_depth.pipeline import NaiveBBoxDepthPipeline
from depth_estimation.naive_bbox_depth.utils import resolve_repo_path


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
        "--real-width-m",
        type=float,
        default=DRONE_WIDTH_M,
        help="Real drone width in meters.",
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

    infer_ms = float(metrics.get("infer_ms", 0.0))
    distance_m = metrics.get("distance_m")
    confidence = metrics.get("confidence")
    bbox_width_px = metrics.get("bbox_width_px")

    depth_text = "dist: n/a"
    if distance_m is not None:
        depth_text = f"dist: {float(distance_m):.3f} m"

    conf_text = "conf: n/a"
    if confidence is not None:
        conf_text = f"conf: {float(confidence):.3f}"

    bbox_text = "bbox_w: n/a"
    if bbox_width_px is not None:
        bbox_text = f"bbox_w: {float(bbox_width_px):.1f}px"

    lines = [
        f"{status}  frame {index + 1}/{total}",
        f"inference: {infer_ms:.1f} ms  delay: {delay_s:.2f}s",
        f"{depth_text}  {conf_text}  {bbox_text}",
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


def main() -> None:
    args = parse_args()

    pipeline = NaiveBBoxDepthPipeline(
        model_path=args.model_path,
        conf_threshold=float(args.conf_threshold),
        fx=float(args.fx),
        real_width_m=float(args.real_width_m),
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
    print(f"fx={float(args.fx):.3f}, real_width_m={float(args.real_width_m):.4f}")
    print("Controls")
    print("space: play/pause")
    print("a or Left Arrow: previous frame")
    print("d or Right Arrow: next frame")
    print("q or ESC: quit")

    index = 0
    cached_index = -1
    cached_frame = None
    cached_metrics = {}

    try:
        while True:
            image_path = image_paths[index]
            if cached_index != index or cached_frame is None:
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

                output = pipeline.process_live_frame(frame)
                cached_frame = output.frame_bgr
                cached_metrics = dict(output.metrics)
                cached_index = index

            display = cached_frame.copy()
            draw_session_overlay(
                frame=display,
                session_name=session_dir.name,
                image_name=image_path.name,
                index=index,
                total=len(image_paths),
                playing=playing,
                delay_s=delay_s,
                metrics=cached_metrics,
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
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    main()
