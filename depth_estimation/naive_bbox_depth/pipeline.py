from __future__ import annotations

import time

import cv2
from ultralytics import YOLO

from depth_estimation.naive_bbox_depth.constants import (
    BUFFER_SIZE,
    DEVICE,
    DRONE_WIDTH_M,
    FOURCC,
    FPS_HINT,
    FX,
    HEIGHT,
    IMAGE_PATH,
    KEY_QUIT,
    MODEL_PATH,
    OUTPUT_DIR,
    WIDTH,
    YOLO_CONF_THRESHOLD,
)
from depth_estimation.naive_bbox_depth.utils import (
    ensure_output_dir,
    estimate_distance_from_bbox,
    resolve_repo_path,
)
from depth_estimation.pipeline_base import LiveDepthPipeline, LiveFrameOutput


class NaiveBBoxDepthPipeline(LiveDepthPipeline):
    name = "naive"

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        conf_threshold: float = YOLO_CONF_THRESHOLD,
        fx: float = FX,
        real_width_m: float = DRONE_WIDTH_M,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.fx = fx
        self.real_width_m = real_width_m
        self._model: YOLO | None = None

    def _get_model(self) -> YOLO:
        if self._model is None:
            model_abs = resolve_repo_path(self.model_path)
            if not model_abs.exists():
                raise FileNotFoundError(f"Could not read model weights: {model_abs}")
            self._model = YOLO(str(model_abs))
        return self._model

    def run_image(self, image_path: str | None = None, output_dir: str = OUTPUT_DIR) -> None:
        chosen_image = image_path or IMAGE_PATH
        image_abs = resolve_repo_path(chosen_image)
        if not image_abs.exists():
            raise FileNotFoundError(f"Could not read image: {image_abs}")

        image = cv2.imread(str(image_abs))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_abs}")

        result = self.process_live_frame(image)
        metrics = result.metrics

        output_dir_abs = ensure_output_dir(output_dir)
        output_path = output_dir_abs / f"{image_abs.stem}_distance_estimate.jpg"
        cv2.imwrite(str(output_path), result.frame_bgr)

        if int(metrics.get("detection_count", 0)) == 0:
            print("No detection found.")
        else:
            print("Best detection:")
            print(f"  confidence       = {float(metrics['confidence']):.3f}")
            print(f"  bbox width px    = {float(metrics['bbox_width_px']):.2f}")
            print(
                "  bbox center px   = "
                f"({float(metrics['bbox_center_x_px']):.1f}, {float(metrics['bbox_center_y_px']):.1f})"
            )
            print(f"  distance (width) = {float(metrics['distance_m']):.3f} m")
        print(f"  inference ms     = {float(metrics['infer_ms']):.2f}")
        print(f"Saved annotated image to: {output_path}")

    def _open_camera(
        self,
        device: str = DEVICE,
        width: int = WIDTH,
        height: int = HEIGHT,
        fps_hint: int = FPS_HINT,
        fourcc: str = FOURCC,
        buffer_size: int = BUFFER_SIZE,
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

    def _best_detection(self, results):
        best = None
        best_conf = -1.0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                if conf > best_conf:
                    best_conf = conf
                    best = (xyxy, conf)

        return best

    def _predict(self, source):
        t0 = time.perf_counter()
        results = self._get_model().predict(
            source,
            conf=self.conf_threshold,
            verbose=False,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return results, infer_ms

    def _annotate_best_detection(self, frame_bgr, xyxy, conf: float) -> dict[str, float]:
        estimate = estimate_distance_from_bbox(
            bbox_xyxy=xyxy,
            fx=self.fx,
            real_width_m=self.real_width_m,
        )

        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = map(int, estimate["center_px"])

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Dist: {estimate['z_est_m']:.3f} m | Conf: {conf:.2f}"
        cv2.putText(
            frame_bgr,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.circle(frame_bgr, (cx, cy), 4, (0, 0, 255), -1)

        return {
            "confidence": round(float(conf), 4),
            "distance_m": round(float(estimate["z_est_m"]), 4),
            "bbox_width_px": round(float(estimate["bbox_width_px"]), 2),
            "bbox_center_x_px": round(float(estimate["center_px"][0]), 2),
            "bbox_center_y_px": round(float(estimate["center_px"][1]), 2),
            "detection_count": 1,
        }

    def process_live_frame(self, frame_bgr) -> LiveFrameOutput:
        display_frame = frame_bgr.copy()

        results, infer_ms = self._predict(frame_bgr)

        best = self._best_detection(results)
        metrics = {"infer_ms": round(infer_ms, 2), "detection_count": 0}

        if best is None:
            cv2.putText(
                display_frame,
                "No detection",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return LiveFrameOutput(method=self.name, frame_bgr=display_frame, metrics=metrics)

        xyxy, conf = best
        metrics.update(self._annotate_best_detection(display_frame, xyxy, conf))
        return LiveFrameOutput(method=self.name, frame_bgr=display_frame, metrics=metrics)

    def run_live(
        self,
        device: str = DEVICE,
        width: int = WIDTH,
        height: int = HEIGHT,
        fps_hint: int = FPS_HINT,
        fourcc: str = FOURCC,
        buffer_size: int = BUFFER_SIZE,
        window_name: str = "Live Depth Estimation (Naive BBox)",
    ) -> None:
        cap = self._open_camera(device, width, height, fps_hint, fourcc, buffer_size)
        frame_count = 0

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    print("Failed to read frame from camera")
                    break

                frame_count += 1
                result = self.process_live_frame(frame_bgr)
                display = result.frame_bgr

                cv2.putText(
                    display,
                    f"Frame: {frame_count}",
                    (10, max(display.shape[0] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in KEY_QUIT:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def close(self) -> None:
        self._model = None
