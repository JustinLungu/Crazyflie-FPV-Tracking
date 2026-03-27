from __future__ import annotations

import time

import cv2
import numpy as np

from depth_estimation.midas.constants import (
    KEY_QUIT,
    MIDAS_CENTER_PATCH_SIZE,
    MIDAS_COLORMAP,
    MIDAS_DEVICE,
    MIDAS_IMAGE_FALLBACK_EXTENSIONS,
    MIDAS_IMAGE_INPUT_PATH,
    MIDAS_IMAGE_OUTPUT_DIR,
    MIDAS_IMAGE_OUTPUT_NPY_SUFFIX,
    MIDAS_IMAGE_OUTPUT_VIS_SUFFIX,
    MIDAS_INVERT_COLORMAP,
    MIDAS_MODEL_TYPE,
    MIDAS_PRINT_EVERY_N_FRAMES,
    MIDAS_TEXT_COLOR,
    MIDAS_TEXT_LINE_HEIGHT,
    MIDAS_TEXT_ORIGIN,
    MIDAS_TEXT_SCALE,
    MIDAS_TEXT_THICKNESS,
    MIDAS_VIDEO_INPUT_PATH,
    MIDAS_VIDEO_MAX_FRAMES,
    MIDAS_VIDEO_OUTPUT_PATH,
    MIDAS_VIDEO_SHOW_PREVIEW,
    MIDAS_VIDEO_WINDOW_NAME,
    MIDAS_VIDEO_WRITE_OUTPUT,
)
from depth_estimation.midas.midas_model import MiDaSModel
from depth_estimation.midas.utils import (
    colorize_depth_map,
    compute_center_depth,
    ensure_parent_dir,
    resolve_existing_image_path,
    resolve_repo_path,
    resize_depth_to_frame,
)
from depth_estimation.pipeline_base import LiveDepthPipeline, LiveFrameOutput


class MiDaSPipeline(LiveDepthPipeline):
    name = "midas"

    def __init__(self, model_type: str = MIDAS_MODEL_TYPE, device: str = MIDAS_DEVICE):
        self.model_type = model_type
        self.device = device
        self._model: MiDaSModel | None = None

    def _get_model(self) -> MiDaSModel:
        if self._model is None:
            self._model = MiDaSModel(model_type=self.model_type, device=self.device)
        return self._model

    def _infer_depth(self, frame_bgr: np.ndarray):
        t0 = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        depth_map = self._get_model().predict(frame_rgb)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return depth_map, infer_ms

    def run_image(self, image_path: str | None = None) -> None:
        image_path_obj = resolve_existing_image_path(
            image_path or MIDAS_IMAGE_INPUT_PATH,
            MIDAS_IMAGE_FALLBACK_EXTENSIONS,
        )

        output_dir = resolve_repo_path(MIDAS_IMAGE_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_stem = image_path_obj.stem
        output_npy = output_dir / f"{image_stem}{MIDAS_IMAGE_OUTPUT_NPY_SUFFIX}"
        output_vis = output_dir / f"{image_stem}{MIDAS_IMAGE_OUTPUT_VIS_SUFFIX}"

        frame_bgr = cv2.imread(str(image_path_obj), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path_obj}")

        depth_map, _infer_ms = self._infer_depth(frame_bgr)

        center_depth = compute_center_depth(depth_map, MIDAS_CENTER_PATCH_SIZE)
        depth_min = float(np.nanmin(depth_map))
        depth_max = float(np.nanmax(depth_map))

        np.save(output_npy, depth_map)
        depth_vis = colorize_depth_map(
            depth_map,
            MIDAS_COLORMAP,
            invert_colormap=MIDAS_INVERT_COLORMAP,
        )
        cv2.imwrite(str(output_vis), depth_vis)

        print("Depth inference complete (single image, MiDaS).")
        print(f"- image: {image_path_obj}")
        print(f"- depth shape: {depth_map.shape}")
        print(f"- depth range: [{depth_min:.4f}, {depth_max:.4f}]")
        print(f"- center depth (median {MIDAS_CENTER_PATCH_SIZE}x{MIDAS_CENTER_PATCH_SIZE}): {center_depth:.4f}")
        print(f"- depth npy: {output_npy}")
        print(f"- depth visualization: {output_vis}")

    def _draw_overlay(self, frame: np.ndarray, lines: list[str]) -> None:
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

    def process_live_frame(self, frame_bgr: np.ndarray) -> LiveFrameOutput:
        depth_map, infer_ms = self._infer_depth(frame_bgr)
        height, width = frame_bgr.shape[:2]
        depth_map = resize_depth_to_frame(depth_map, width, height)
        center_depth = compute_center_depth(depth_map, MIDAS_CENTER_PATCH_SIZE)

        depth_color = colorize_depth_map(
            depth_map,
            MIDAS_COLORMAP,
            invert_colormap=MIDAS_INVERT_COLORMAP,
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
        resolved_video_path = resolve_repo_path(video_path or MIDAS_VIDEO_INPUT_PATH)
        if not resolved_video_path.exists():
            raise RuntimeError(
                f"Video not found: {resolved_video_path}\n"
                "Set MIDAS_VIDEO_INPUT_PATH in depth_estimation/midas/constants.py."
            )

        cap = cv2.VideoCapture(str(resolved_video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {resolved_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if MIDAS_VIDEO_WRITE_OUTPUT:
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

        print("Depth inference started (video, MiDaS).")
        print(f"- input video: {resolved_video_path}")
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
                result = self.process_live_frame(frame_bgr)
                composed = result.frame_bgr

                self._draw_overlay(composed, [f"frame: {frame_idx}"])

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
                        f"frame {frame_idx}: center_depth={result.metrics['center_depth']:.3f}, "
                        f"infer_ms={result.metrics['infer_ms']:.1f}"
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

    def close(self) -> None:
        self._model = None
