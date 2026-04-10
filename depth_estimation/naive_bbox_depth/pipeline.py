from __future__ import annotations

import time

import cv2
from ultralytics import YOLO

from depth_estimation.naive_bbox_depth.constants import (
    BUFFER_SIZE,
    DEVICE,
    NAIVE_DROPOUT_HOLD_FRAMES,
    NAIVE_DROPOUT_STALE_FRAMES,
    DRONE_WIDTH_M,
    FY,
    CX,
    CY,
    FOURCC,
    FPS_HINT,
    FX,
    HEIGHT,
    IMAGE_PATH,
    NAIVE_EMA_ALPHA_CENTER,
    NAIVE_EMA_ALPHA_DISTANCE,
    NAIVE_EMA_ALPHA_WIDTH,
    NAIVE_ENABLE_RELATIVE_POSITION,
    NAIVE_FILTER_CENTER,
    NAIVE_FILTER_DISTANCE,
    NAIVE_FILTER_MODE,
    NAIVE_FILTER_WIDTH,
    NAIVE_INTRINSICS_FALLBACK_TO_MANUAL,
    NAIVE_INTRINSICS_SOURCE,
    NAIVE_KALMAN_MEAS_VAR_CENTER,
    NAIVE_KALMAN_MEAS_VAR_DISTANCE,
    NAIVE_KALMAN_MEAS_VAR_WIDTH,
    NAIVE_KALMAN_PROCESS_VAR_CENTER,
    NAIVE_KALMAN_PROCESS_VAR_DISTANCE,
    NAIVE_KALMAN_PROCESS_VAR_WIDTH,
    NAIVE_CAMERA_MATRIX_PATH,
    NAIVE_RESET_FILTER_ON_LOST,
    NAIVE_Y_AXIS_CONVENTION,
    KEY_QUIT,
    MODEL_PATH,
    OUTPUT_DIR,
    WIDTH,
    YOLO_CONF_THRESHOLD,
)
from depth_estimation.naive_bbox_depth.filtering import ScalarSignalFilter
from depth_estimation.naive_bbox_depth.utils import (
    estimate_relative_position_from_center,
    ensure_output_dir,
    estimate_distance_from_bbox,
    load_intrinsics_from_camera_matrix,
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
        fy: float = FY,
        cx: float = CX,
        cy: float = CY,
        real_width_m: float = DRONE_WIDTH_M,
        intrinsics_source: str = NAIVE_INTRINSICS_SOURCE,
        camera_matrix_path: str = NAIVE_CAMERA_MATRIX_PATH,
        intrinsics_fallback_to_manual: bool = NAIVE_INTRINSICS_FALLBACK_TO_MANUAL,
        enable_relative_position: bool = NAIVE_ENABLE_RELATIVE_POSITION,
        y_axis_convention: str = NAIVE_Y_AXIS_CONVENTION,
        filter_mode: str = NAIVE_FILTER_MODE,
        filter_distance: bool = NAIVE_FILTER_DISTANCE,
        filter_center: bool = NAIVE_FILTER_CENTER,
        filter_width: bool = NAIVE_FILTER_WIDTH,
        ema_alpha_distance: float = NAIVE_EMA_ALPHA_DISTANCE,
        ema_alpha_center: float = NAIVE_EMA_ALPHA_CENTER,
        ema_alpha_width: float = NAIVE_EMA_ALPHA_WIDTH,
        kalman_process_var_distance: float = NAIVE_KALMAN_PROCESS_VAR_DISTANCE,
        kalman_meas_var_distance: float = NAIVE_KALMAN_MEAS_VAR_DISTANCE,
        kalman_process_var_center: float = NAIVE_KALMAN_PROCESS_VAR_CENTER,
        kalman_meas_var_center: float = NAIVE_KALMAN_MEAS_VAR_CENTER,
        kalman_process_var_width: float = NAIVE_KALMAN_PROCESS_VAR_WIDTH,
        kalman_meas_var_width: float = NAIVE_KALMAN_MEAS_VAR_WIDTH,
        dropout_hold_frames: int = NAIVE_DROPOUT_HOLD_FRAMES,
        dropout_stale_frames: int = NAIVE_DROPOUT_STALE_FRAMES,
        reset_filter_on_lost: bool = NAIVE_RESET_FILTER_ON_LOST,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.real_width_m = real_width_m
        self.intrinsics_source = intrinsics_source.strip().lower()
        self.camera_matrix_path = camera_matrix_path
        self.intrinsics_fallback_to_manual = bool(intrinsics_fallback_to_manual)
        self.enable_relative_position = bool(enable_relative_position)
        self.y_axis_convention = y_axis_convention.strip().lower()
        if self.y_axis_convention not in {"up", "down"}:
            raise ValueError(
                f"Unsupported y-axis convention '{y_axis_convention}'. Use 'up' or 'down'."
            )
        self.intrinsics_loaded_from = "manual"
        self._configure_intrinsics()

        self.filter_mode = filter_mode.strip().lower()
        if self.filter_mode not in {"none", "ema", "kalman"}:
            raise ValueError(
                f"Unsupported filter mode '{filter_mode}'. Use one of: none, ema, kalman."
            )
        self.filter_distance = bool(filter_distance)
        self.filter_center = bool(filter_center)
        self.filter_width = bool(filter_width)

        self.dropout_hold_frames = max(0, int(dropout_hold_frames))
        self.dropout_stale_frames = max(self.dropout_hold_frames, int(dropout_stale_frames))
        self.reset_filter_on_lost = bool(reset_filter_on_lost)

        distance_mode = self.filter_mode if self.filter_distance else "none"
        center_mode = self.filter_mode if self.filter_center else "none"
        width_mode = self.filter_mode if self.filter_width else "none"
        self._distance_filter = ScalarSignalFilter(
            mode=distance_mode,
            ema_alpha=ema_alpha_distance,
            kalman_process_var=kalman_process_var_distance,
            kalman_measurement_var=kalman_meas_var_distance,
        )
        self._center_x_filter = ScalarSignalFilter(
            mode=center_mode,
            ema_alpha=ema_alpha_center,
            kalman_process_var=kalman_process_var_center,
            kalman_measurement_var=kalman_meas_var_center,
        )
        self._center_y_filter = ScalarSignalFilter(
            mode=center_mode,
            ema_alpha=ema_alpha_center,
            kalman_process_var=kalman_process_var_center,
            kalman_measurement_var=kalman_meas_var_center,
        )
        self._width_filter = ScalarSignalFilter(
            mode=width_mode,
            ema_alpha=ema_alpha_width,
            kalman_process_var=kalman_process_var_width,
            kalman_measurement_var=kalman_meas_var_width,
        )

        self._missed_frames = 0
        self._last_estimate_metrics: dict[str, float] | None = None
        self._model: YOLO | None = None

    def _configure_intrinsics(self) -> None:
        if self.intrinsics_source == "manual":
            self.intrinsics_loaded_from = "manual"
            return

        if self.intrinsics_source != "calibration_npy":
            raise ValueError(
                f"Unsupported intrinsics source '{self.intrinsics_source}'. "
                "Use one of: manual, calibration_npy."
            )

        try:
            values = load_intrinsics_from_camera_matrix(self.camera_matrix_path)
        except Exception:
            if not self.intrinsics_fallback_to_manual:
                raise
            self.intrinsics_loaded_from = "manual_fallback"
            return

        self.fx = float(values["fx"])
        self.fy = float(values["fy"])
        self.cx = float(values["cx"])
        self.cy = float(values["cy"])
        self.intrinsics_loaded_from = "calibration_npy"

    def _get_model(self) -> YOLO:
        if self._model is None:
            model_abs = resolve_repo_path(self.model_path)
            if not model_abs.exists():
                raise FileNotFoundError(f"Could not read model weights: {model_abs}")
            self._model = YOLO(str(model_abs))
        return self._model

    def run_image(self, image_path: str | None = None, output_dir: str = OUTPUT_DIR) -> None:
        self.reset_temporal_state()

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
            print(f"  raw bbox width   = {float(metrics['raw_bbox_width_px']):.2f}")
            print(f"  filt bbox width  = {float(metrics['bbox_width_px']):.2f}")
            print(
                "  bbox center px   = "
                f"({float(metrics['bbox_center_x_px']):.1f}, {float(metrics['bbox_center_y_px']):.1f})"
            )
            print(f"  raw distance     = {float(metrics['raw_distance_m']):.3f} m")
            print(f"  filt distance    = {float(metrics['distance_m']):.3f} m")
            if self.enable_relative_position and metrics.get("x_rel_m") is not None:
                print(
                    "  rel xyz (m)      = "
                    f"({float(metrics['x_rel_m']):.3f}, {float(metrics['y_rel_m']):.3f}, {float(metrics['z_rel_m']):.3f})"
                )
                print(
                    "  yaw error        = "
                    f"{float(metrics['yaw_error_rad']):.3f} rad ({float(metrics['yaw_error_deg']):.1f} deg)"
                )
        print(f"  inference ms     = {float(metrics['infer_ms']):.2f}")
        print(
            "  intrinsics       = "
            f"fx={self.fx:.3f}, fy={self.fy:.3f}, cx={self.cx:.3f}, cy={self.cy:.3f} "
            f"[{self.intrinsics_loaded_from}]"
        )
        print(f"  track state      = {metrics.get('track_state', 'unknown')}")
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

    def _raw_measurement_from_detection(self, xyxy, conf: float) -> dict[str, float]:
        estimate = estimate_distance_from_bbox(
            bbox_xyxy=xyxy,
            fx=self.fx,
            real_width_m=self.real_width_m,
        )

        return {
            "confidence": round(float(conf), 4),
            "raw_bbox_width_px": round(float(estimate["bbox_width_px"]), 2),
            "raw_bbox_center_x_px": round(float(estimate["center_px"][0]), 2),
            "raw_bbox_center_y_px": round(float(estimate["center_px"][1]), 2),
            "raw_distance_m": round(float(estimate["z_est_m"]), 4),
        }

    def _filtered_measurement_from_raw(self, raw: dict[str, float]) -> dict[str, float]:
        raw_width = float(raw["raw_bbox_width_px"])
        raw_center_x = float(raw["raw_bbox_center_x_px"])
        raw_center_y = float(raw["raw_bbox_center_y_px"])
        raw_distance = float(raw["raw_distance_m"])

        filtered_width = self._width_filter.update(raw_width)
        width_for_distance = filtered_width if self.filter_width else raw_width
        distance_from_width = (self.fx * self.real_width_m) / max(width_for_distance, 1.0)
        filtered_distance = self._distance_filter.update(distance_from_width)
        filtered_center_x = self._center_x_filter.update(raw_center_x)
        filtered_center_y = self._center_y_filter.update(raw_center_y)

        rel = None
        raw_rel = None
        if self.enable_relative_position:
            raw_rel = estimate_relative_position_from_center(
                center_px=(raw_center_x, raw_center_y),
                z_m=raw_distance,
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
                y_axis_convention=self.y_axis_convention,
            )
            rel = estimate_relative_position_from_center(
                center_px=(filtered_center_x, filtered_center_y),
                z_m=float(filtered_distance),
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
                y_axis_convention=self.y_axis_convention,
            )

        self._missed_frames = 0
        metrics = {
            "confidence": float(raw["confidence"]),
            "raw_bbox_width_px": round(raw_width, 2),
            "bbox_width_px": round(filtered_width, 2),
            "raw_bbox_center_x_px": round(raw_center_x, 2),
            "raw_bbox_center_y_px": round(raw_center_y, 2),
            "bbox_center_x_px": round(filtered_center_x, 2),
            "bbox_center_y_px": round(filtered_center_y, 2),
            "raw_distance_m": round(raw_distance, 4),
            "distance_m": round(float(filtered_distance), 4),
            "detection_count": 1,
            "track_state": "tracked",
            "frames_since_detection": 0,
            "estimate_source": "measurement",
            "is_stale": 0,
            "filter_mode": self.filter_mode,
        }
        if rel is not None and raw_rel is not None:
            metrics.update(
                {
                    "raw_x_rel_m": round(float(raw_rel["x_rel_m"]), 4),
                    "raw_y_rel_m": round(float(raw_rel["y_rel_m"]), 4),
                    "raw_z_rel_m": round(float(raw_rel["z_rel_m"]), 4),
                    "raw_yaw_error_rad": round(float(raw_rel["yaw_error_rad"]), 4),
                    "raw_yaw_error_deg": round(float(raw_rel["yaw_error_deg"]), 2),
                    "x_rel_m": round(float(rel["x_rel_m"]), 4),
                    "y_rel_m": round(float(rel["y_rel_m"]), 4),
                    "z_rel_m": round(float(rel["z_rel_m"]), 4),
                    "yaw_error_rad": round(float(rel["yaw_error_rad"]), 4),
                    "yaw_error_deg": round(float(rel["yaw_error_deg"]), 2),
                }
            )
        self._last_estimate_metrics = metrics
        return metrics

    def _missing_detection_metrics(self) -> dict[str, float | int | str]:
        self._missed_frames += 1
        metrics: dict[str, float | int | str] = {
            "detection_count": 0,
            "frames_since_detection": self._missed_frames,
            "filter_mode": self.filter_mode,
        }

        if self._last_estimate_metrics is None:
            metrics.update(
                {
                    "track_state": "lost",
                    "estimate_source": "none",
                    "is_stale": 1,
                }
            )
            return metrics

        if self._missed_frames <= self.dropout_stale_frames:
            track_state = "held" if self._missed_frames <= self.dropout_hold_frames else "stale"
            metrics.update(self._last_estimate_metrics)
            metrics["detection_count"] = 0
            metrics["track_state"] = track_state
            metrics["frames_since_detection"] = self._missed_frames
            metrics["estimate_source"] = "history"
            metrics["is_stale"] = 1 if track_state == "stale" else 0
            return metrics

        metrics.update(
            {
                "track_state": "lost",
                "estimate_source": "none",
                "is_stale": 1,
            }
        )
        if self.reset_filter_on_lost:
            self.reset_temporal_state()
        return metrics

    def _annotate_best_detection(self, frame_bgr, xyxy, metrics: dict[str, float]) -> None:
        x1, y1, x2, y2 = map(int, xyxy)
        cx = int(float(metrics["bbox_center_x_px"]))
        cy = int(float(metrics["bbox_center_y_px"]))
        raw_cx = int(float(metrics["raw_bbox_center_x_px"]))
        raw_cy = int(float(metrics["raw_bbox_center_y_px"]))

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.circle(frame_bgr, (cx, cy), 4, (0, 0, 255), -1)
        if self.filter_mode != "none" and self.filter_center:
            cv2.circle(frame_bgr, (raw_cx, raw_cy), 3, (255, 0, 0), -1)

    def _annotate_missing_detection(self, frame_bgr, metrics: dict[str, float | int | str]) -> None:
        track_state = str(metrics.get("track_state", "lost")).upper()
        frames_since_detection = int(metrics.get("frames_since_detection", 0))
        distance = metrics.get("distance_m")

        text = "No detection"
        if distance is not None and track_state in {"HELD", "STALE"}:
            text = (
                f"No detection | {track_state} | "
                f"dist {float(distance):.3f} m | miss {frames_since_detection}"
            )

        cv2.putText(
            frame_bgr,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    def _draw_relative_overlay(self, frame_bgr, metrics: dict[str, float | int | str]) -> None:
        if not self.enable_relative_position:
            return
        if metrics.get("x_rel_m") is None:
            return

        lines = [
            f"X: {float(metrics['x_rel_m']):.3f} m",
            f"Y: {float(metrics['y_rel_m']):.3f} m",
            f"Z: {float(metrics['z_rel_m']):.3f} m",
            f"Yaw err: {float(metrics['yaw_error_deg']):.1f} deg",
        ]
        x = 16
        line_h = 24
        y0 = max(20, int(frame_bgr.shape[0]) - (len(lines) * line_h) - 20)
        for i, line in enumerate(lines):
            cv2.putText(
                frame_bgr,
                line,
                (x, y0 + i * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )

    def process_live_frame(self, frame_bgr) -> LiveFrameOutput:
        display_frame = frame_bgr.copy()

        results, infer_ms = self._predict(frame_bgr)

        best = self._best_detection(results)
        metrics: dict[str, float | int | str] = {"infer_ms": round(infer_ms, 2)}

        if best is None:
            metrics.update(self._missing_detection_metrics())
            self._annotate_missing_detection(display_frame, metrics)
            self._draw_relative_overlay(display_frame, metrics)
            return LiveFrameOutput(method=self.name, frame_bgr=display_frame, metrics=metrics)

        xyxy, conf = best
        raw = self._raw_measurement_from_detection(xyxy, conf)
        metrics.update(self._filtered_measurement_from_raw(raw))
        self._annotate_best_detection(display_frame, xyxy, metrics)
        self._draw_relative_overlay(display_frame, metrics)
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
        self.reset_temporal_state()
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

    def reset_temporal_state(self) -> None:
        self._distance_filter.reset()
        self._center_x_filter.reset()
        self._center_y_filter.reset()
        self._width_filter.reset()
        self._missed_frames = 0
        self._last_estimate_metrics = None

    def close(self) -> None:
        self.reset_temporal_state()
        self._model = None
