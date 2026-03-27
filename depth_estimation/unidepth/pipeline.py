from __future__ import annotations

import time

import cv2
import numpy as np

from depth_estimation.pipeline_base import LiveDepthPipeline, LiveFrameOutput
from depth_estimation.unidepth.constants import (
    DEPTH_CENTER_PATCH_SIZE,
    DEPTH_COLORMAP,
    DEPTH_IMAGE_FALLBACK_EXTENSIONS,
    DEPTH_IMAGE_INPUT_PATH,
    DEPTH_IMAGE_OUTPUT_DIR,
    DEPTH_IMAGE_OUTPUT_NPY_SUFFIX,
    DEPTH_IMAGE_OUTPUT_VIS_SUFFIX,
    DEPTH_INVERT_COLORMAP,
    DEPTH_PRINT_EVERY_N_FRAMES,
    DEPTH_RESOLUTION_LEVEL,
    DEPTH_TEXT_COLOR,
    DEPTH_TEXT_LINE_HEIGHT,
    DEPTH_TEXT_ORIGIN,
    DEPTH_TEXT_SCALE,
    DEPTH_TEXT_THICKNESS,
    DEPTH_VIDEO_MAX_FRAMES,
    DEPTH_VIDEO_OUTPUT_PATH,
    DEPTH_VIDEO_SHOW_PREVIEW,
    DEPTH_VIDEO_WINDOW_NAME,
    DEPTH_VIDEO_WRITE_OUTPUT,
    DEPTH_VIDEO_INPUT_PATH,
    KEY_QUIT,
)
from depth_estimation.unidepth.unidepth_v2 import UniDepthV2
from depth_estimation.unidepth.utils import (
    colorize_depth_map,
    compute_center_depth,
    ensure_parent_dir,
    resolve_existing_image_path,
    resolve_repo_path,
    resize_depth_to_frame,
)


class UniDepthPipeline(LiveDepthPipeline):
    name = "unidepth"

    def __init__(self, resolution_level=DEPTH_RESOLUTION_LEVEL):
        self.resolution_level = resolution_level
        self._model: UniDepthV2 | None = None

    def _get_model(self) -> UniDepthV2:
        if self._model is None:
            self._model = UniDepthV2(resolution_level=self.resolution_level)
        return self._model

    def _infer_depth(self, frame_bgr: np.ndarray):
        t0 = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        depth_tensor, intrinsics = self._get_model()(frame_rgb)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        depth_map = depth_tensor.detach().cpu().numpy().squeeze().astype(np.float32)
        return depth_map, intrinsics, infer_ms

    def run_image(self, image_path: str | None = None) -> None:
        image_path_obj = resolve_existing_image_path(
            image_path or DEPTH_IMAGE_INPUT_PATH,
            DEPTH_IMAGE_FALLBACK_EXTENSIONS,
        )

        output_dir = resolve_repo_path(DEPTH_IMAGE_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_stem = image_path_obj.stem
        output_npy = output_dir / f"{image_stem}{DEPTH_IMAGE_OUTPUT_NPY_SUFFIX}"
        output_vis = output_dir / f"{image_stem}{DEPTH_IMAGE_OUTPUT_VIS_SUFFIX}"

        frame_bgr = cv2.imread(str(image_path_obj), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path_obj}")

        depth_map, intrinsics, _infer_ms = self._infer_depth(frame_bgr)

        center_depth = compute_center_depth(depth_map, DEPTH_CENTER_PATCH_SIZE)
        depth_min = float(np.nanmin(depth_map))
        depth_max = float(np.nanmax(depth_map))

        np.save(output_npy, depth_map)
        depth_vis = colorize_depth_map(
            depth_map,
            DEPTH_COLORMAP,
            invert_colormap=DEPTH_INVERT_COLORMAP,
        )
        cv2.imwrite(str(output_vis), depth_vis)

        print("Depth inference complete (single image).")
        print(f"- image: {image_path_obj}")
        print(f"- depth shape: {depth_map.shape}")
        print(f"- depth range: [{depth_min:.4f}, {depth_max:.4f}]")
        print(f"- center depth (median {DEPTH_CENTER_PATCH_SIZE}x{DEPTH_CENTER_PATCH_SIZE}): {center_depth:.4f}")
        print(f"- intrinsics shape: {tuple(intrinsics.shape)}")
        print(f"- depth npy: {output_npy}")
        print(f"- depth visualization: {output_vis}")

    def _draw_overlay(self, frame: np.ndarray, lines: list[str]) -> None:
        x, y = DEPTH_TEXT_ORIGIN
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x, y + i * DEPTH_TEXT_LINE_HEIGHT),
                cv2.FONT_HERSHEY_SIMPLEX,
                DEPTH_TEXT_SCALE,
                DEPTH_TEXT_COLOR,
                DEPTH_TEXT_THICKNESS,
                cv2.LINE_AA,
            )

    def process_live_frame(self, frame_bgr: np.ndarray) -> LiveFrameOutput:
        depth_map, _intrinsics, infer_ms = self._infer_depth(frame_bgr)
        height, width = frame_bgr.shape[:2]
        depth_map = resize_depth_to_frame(depth_map, width, height)
        center_depth = compute_center_depth(depth_map, DEPTH_CENTER_PATCH_SIZE)

        depth_color = colorize_depth_map(
            depth_map,
            DEPTH_COLORMAP,
            invert_colormap=DEPTH_INVERT_COLORMAP,
        )
        composed = np.hstack([frame_bgr, depth_color])

        self._draw_overlay(
            composed,
            [
                f"method: {self.name}",
                f"inference: {infer_ms:.1f} ms",
                f"center depth: {center_depth:.3f}",
            ],
        )

        return LiveFrameOutput(
            method=self.name,
            frame_bgr=composed,
            metrics={
                "infer_ms": round(float(infer_ms), 2),
                "center_depth": round(float(center_depth), 4),
            },
        )

    def run_video(self, video_path: str | None = None) -> None:
        resolved_video_path = resolve_repo_path(video_path or DEPTH_VIDEO_INPUT_PATH)
        if not resolved_video_path.exists():
            raise RuntimeError(
                f"Video not found: {resolved_video_path}\n"
                "Set DEPTH_VIDEO_INPUT_PATH in depth_estimation/unidepth/constants.py."
            )

        cap = cv2.VideoCapture(str(resolved_video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {resolved_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if DEPTH_VIDEO_WRITE_OUTPUT:
            output_path = resolve_repo_path(DEPTH_VIDEO_OUTPUT_PATH)
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

        print("Depth inference started (video).")
        print(f"- input video: {resolved_video_path}")
        print(f"- size: {width}x{height} @ {fps:.2f} fps")
        print(f"- frames: {total_frames if total_frames > 0 else 'unknown'}")
        if writer is not None:
            print(f"- output video: {resolve_repo_path(DEPTH_VIDEO_OUTPUT_PATH)}")

        frame_idx = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_idx += 1
                result = self.process_live_frame(frame_bgr)
                composed = result.frame_bgr

                self._draw_overlay(composed, [f"frame: {frame_idx}"])

                if writer is not None:
                    writer.write(composed)

                if DEPTH_VIDEO_SHOW_PREVIEW:
                    cv2.imshow(DEPTH_VIDEO_WINDOW_NAME, composed)
                    key = cv2.waitKey(1) & 0xFF
                    if key in KEY_QUIT:
                        print("Stopped by user.")
                        break

                if frame_idx % max(1, DEPTH_PRINT_EVERY_N_FRAMES) == 0:
                    print(
                        f"frame {frame_idx}: center_depth={result.metrics['center_depth']:.3f}, "
                        f"infer_ms={result.metrics['infer_ms']:.1f}"
                    )

                if DEPTH_VIDEO_MAX_FRAMES > 0 and frame_idx >= DEPTH_VIDEO_MAX_FRAMES:
                    print(f"Stopped at DEPTH_VIDEO_MAX_FRAMES={DEPTH_VIDEO_MAX_FRAMES}.")
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

        print("Depth inference finished (video).")
        print(f"- processed frames: {frame_idx}")

    def close(self) -> None:
        self._model = None
