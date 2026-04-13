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
    NAIVE_GATING_BORDER_MARGIN_PX,
    NAIVE_GATING_CHECK_BORDER,
    NAIVE_GATING_CHECK_CONFIDENCE,
    NAIVE_GATING_CHECK_DISTANCE_JUMP,
    NAIVE_GATING_CHECK_MAX_DISTANCE,
    NAIVE_GATING_CHECK_MIN_WIDTH,
    NAIVE_GATING_CHECK_X_JUMP,
    NAIVE_GATING_CHECK_Y_JUMP,
    NAIVE_GATING_ENABLED,
    NAIVE_GATING_MAX_DISTANCE_JUMP_M,
    NAIVE_GATING_MAX_CANDIDATES,
    NAIVE_GATING_MAX_VALID_DISTANCE_M,
    NAIVE_GATING_MAX_X_JUMP_M,
    NAIVE_GATING_MAX_Y_JUMP_M,
    NAIVE_GATING_MIN_BBOX_WIDTH_PX,
    NAIVE_GATING_MIN_CONF_FOR_CONTROL,
    NAIVE_GATING_SHOW_REJECTION_OVERLAY,
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
    NAIVE_SHOW_RELATIVE_OVERLAY_ON_FRAME,
    NAIVE_Y_AXIS_CONVENTION,
    KEY_QUIT,
    KEY_TOGGLE_GATING,
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
        show_relative_overlay_on_frame: bool = NAIVE_SHOW_RELATIVE_OVERLAY_ON_FRAME,
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
        gating_enabled: bool = NAIVE_GATING_ENABLED,
        gating_check_confidence: bool = NAIVE_GATING_CHECK_CONFIDENCE,
        gating_check_min_width: bool = NAIVE_GATING_CHECK_MIN_WIDTH,
        gating_check_max_distance: bool = NAIVE_GATING_CHECK_MAX_DISTANCE,
        gating_check_border: bool = NAIVE_GATING_CHECK_BORDER,
        gating_check_distance_jump: bool = NAIVE_GATING_CHECK_DISTANCE_JUMP,
        gating_check_x_jump: bool = NAIVE_GATING_CHECK_X_JUMP,
        gating_check_y_jump: bool = NAIVE_GATING_CHECK_Y_JUMP,
        gating_min_conf_for_control: float = NAIVE_GATING_MIN_CONF_FOR_CONTROL,
        gating_min_bbox_width_px: float = NAIVE_GATING_MIN_BBOX_WIDTH_PX,
        gating_max_valid_distance_m: float = NAIVE_GATING_MAX_VALID_DISTANCE_M,
        gating_border_margin_px: int = NAIVE_GATING_BORDER_MARGIN_PX,
        gating_max_distance_jump_m: float = NAIVE_GATING_MAX_DISTANCE_JUMP_M,
        gating_max_candidates: int = NAIVE_GATING_MAX_CANDIDATES,
        gating_max_x_jump_m: float = NAIVE_GATING_MAX_X_JUMP_M,
        gating_max_y_jump_m: float = NAIVE_GATING_MAX_Y_JUMP_M,
        gating_show_rejection_overlay: bool = NAIVE_GATING_SHOW_REJECTION_OVERLAY,
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
        self.show_relative_overlay_on_frame = bool(show_relative_overlay_on_frame)
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
        self.gating_enabled = bool(gating_enabled)
        self.gating_check_confidence = bool(gating_check_confidence)
        self.gating_check_min_width = bool(gating_check_min_width)
        self.gating_check_max_distance = bool(gating_check_max_distance)
        self.gating_check_border = bool(gating_check_border)
        self.gating_check_distance_jump = bool(gating_check_distance_jump)
        self.gating_check_x_jump = bool(gating_check_x_jump)
        self.gating_check_y_jump = bool(gating_check_y_jump)
        self.gating_min_conf_for_control = float(gating_min_conf_for_control)
        self.gating_min_bbox_width_px = float(gating_min_bbox_width_px)
        self.gating_max_valid_distance_m = float(gating_max_valid_distance_m)
        self.gating_border_margin_px = max(0, int(gating_border_margin_px))
        self.gating_max_distance_jump_m = float(gating_max_distance_jump_m)
        self.gating_max_candidates = max(1, int(gating_max_candidates))
        self.gating_max_x_jump_m = float(gating_max_x_jump_m)
        self.gating_max_y_jump_m = float(gating_max_y_jump_m)
        self.gating_show_rejection_overlay = bool(gating_show_rejection_overlay)

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
        self._last_estimate_metrics: dict[str, float | int | str] | None = None
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

    def set_gating_enabled(self, enabled: bool) -> bool:
        self.gating_enabled = bool(enabled)
        return self.gating_enabled

    def toggle_gating(self) -> bool:
        self.gating_enabled = not self.gating_enabled
        return self.gating_enabled

    def _extract_raw_relative(self, raw: dict[str, float]) -> dict[str, float] | None:
        if not self.enable_relative_position:
            return None

        return estimate_relative_position_from_center(
            center_px=(float(raw["raw_bbox_center_x_px"]), float(raw["raw_bbox_center_y_px"])),
            z_m=float(raw["raw_distance_m"]),
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            y_axis_convention=self.y_axis_convention,
        )

    def _evaluate_gating(
        self,
        raw: dict[str, float],
        xyxy,
        frame_shape,
    ) -> tuple[bool, list[str], dict[str, float] | None]:
        if not self.gating_enabled:
            return True, [], None

        reasons: list[str] = []
        conf = float(raw["confidence"])
        raw_width = float(raw["raw_bbox_width_px"])
        raw_distance = float(raw["raw_distance_m"])
        x1, y1, x2, y2 = map(float, xyxy)

        if self.gating_check_confidence and conf < self.gating_min_conf_for_control:
            reasons.append(f"low_conf<{self.gating_min_conf_for_control:.2f}")

        if self.gating_check_min_width and raw_width < self.gating_min_bbox_width_px:
            reasons.append(f"width<{self.gating_min_bbox_width_px:.1f}px")

        if self.gating_check_max_distance and raw_distance > self.gating_max_valid_distance_m:
            reasons.append(f"dist>{self.gating_max_valid_distance_m:.2f}m")

        if self.gating_check_border:
            h, w = frame_shape[:2]
            m = float(self.gating_border_margin_px)
            if x1 <= m or y1 <= m or x2 >= (w - 1 - m) or y2 >= (h - 1 - m):
                reasons.append(f"near_border<{int(m)}px")

        prev = self._last_estimate_metrics
        if prev is not None and self.gating_check_distance_jump and prev.get("distance_m") is not None:
            prev_dist = float(prev["distance_m"])
            if abs(raw_distance - prev_dist) > self.gating_max_distance_jump_m:
                reasons.append(f"dist_jump>{self.gating_max_distance_jump_m:.2f}m")

        raw_rel = self._extract_raw_relative(raw)
        if prev is not None and raw_rel is not None:
            if self.gating_check_x_jump and prev.get("x_rel_m") is not None:
                if abs(float(raw_rel["x_rel_m"]) - float(prev["x_rel_m"])) > self.gating_max_x_jump_m:
                    reasons.append(f"x_jump>{self.gating_max_x_jump_m:.2f}m")
            if self.gating_check_y_jump and prev.get("y_rel_m") is not None:
                if abs(float(raw_rel["y_rel_m"]) - float(prev["y_rel_m"])) > self.gating_max_y_jump_m:
                    reasons.append(f"y_jump>{self.gating_max_y_jump_m:.2f}m")

        return len(reasons) == 0, reasons, raw_rel

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
            if int(metrics.get("gating_passed", 1)) == 0:
                print("Best detection (rejected by gating):")
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
        print(f"  gating           = {'on' if self.gating_enabled else 'off'}")
        if metrics.get("gating_passed") == 0:
            print(f"  gating reasons   = {metrics.get('gating_reasons', '')}")
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

    def _ranked_detections(
        self,
        results,
        max_candidates: int | None = None,
    ) -> tuple[list[tuple], int]:
        candidates = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                candidates.append((xyxy, conf))

        if not candidates:
            return [], 0

        candidates.sort(key=lambda item: item[1], reverse=True)
        total = len(candidates)
        if max_candidates is None:
            return candidates, total
        return candidates[: max(1, int(max_candidates))], total

    def _predict(self, source):
        t0 = time.perf_counter()
        results = self._get_model().predict(
            source,
            conf=self.conf_threshold,
            verbose=False,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return results, infer_ms

    def _attach_runtime_metrics(
        self,
        metrics: dict[str, float | int | str],
        process_t0: float,
    ) -> None:
        process_ms = (time.perf_counter() - process_t0) * 1000.0
        metrics["process_ms"] = round(process_ms, 2)
        metrics["process_fps"] = round(1000.0 / process_ms, 2) if process_ms > 0.0 else 0.0
        infer_ms = float(metrics.get("infer_ms", 0.0))
        metrics["infer_fps"] = round(1000.0 / infer_ms, 2) if infer_ms > 0.0 else 0.0

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

    def _filtered_measurement_from_raw(
        self, raw: dict[str, float]
    ) -> dict[str, float | int | str]:
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
        raw_rel = self._extract_raw_relative(raw)
        if self.enable_relative_position:
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

    def _annotate_best_detection(
        self, frame_bgr, xyxy, metrics: dict[str, float | int | str]
    ) -> None:
        x1, y1, x2, y2 = map(int, xyxy)
        cx = int(float(metrics["bbox_center_x_px"]))
        cy = int(float(metrics["bbox_center_y_px"]))
        raw_cx = int(float(metrics["raw_bbox_center_x_px"]))
        raw_cy = int(float(metrics["raw_bbox_center_y_px"]))

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.circle(frame_bgr, (cx, cy), 4, (0, 0, 255), -1)
        if self.filter_mode != "none" and self.filter_center:
            cv2.circle(frame_bgr, (raw_cx, raw_cy), 3, (255, 0, 0), -1)

        conf = metrics.get("confidence")
        raw_dist = metrics.get("raw_distance_m")
        filt_dist = metrics.get("distance_m")
        lines = []
        if conf is not None:
            lines.append(f"Conf: {float(conf):.2f}")
        if raw_dist is not None:
            lines.append(f"Raw dist: {float(raw_dist):.3f} m")
        if filt_dist is not None:
            lines.append(f"Filt dist: {float(filt_dist):.3f} m")
        if not lines:
            return

        line_h = 18
        text_x = max(8, x1)
        h = int(frame_bgr.shape[0])

        # Keep text off the target: confidence above, distances below.
        conf_line = lines[0] if len(lines) >= 1 else None
        raw_line = lines[1] if len(lines) >= 2 else None
        filt_line = lines[2] if len(lines) >= 3 else None

        # Above position for confidence.
        if conf_line is not None:
            conf_y = y1 - 6
            if conf_y < 14:
                conf_y = min(h - 8, y2 + 18 + (2 * line_h))
            cv2.putText(
                frame_bgr,
                conf_line,
                (text_x, conf_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Below position for raw + filtered distance.
        below_lines = [line for line in [raw_line, filt_line] if line is not None]
        if not below_lines:
            return

        below_start_y = y2 + 18
        needed_below = (len(below_lines) - 1) * line_h
        if below_start_y + needed_below > h - 8:
            below_start_y = max(14, y1 - 6 - needed_below)

        for i, line in enumerate(below_lines):
            cv2.putText(
                frame_bgr,
                line,
                (text_x, below_start_y + i * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    def _annotate_rejected_detection(
        self,
        frame_bgr,
        xyxy,
        reasons: list[str],
        raw: dict[str, float] | None = None,
    ) -> None:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

        conf = None if raw is None else raw.get("confidence")
        raw_dist = None if raw is None else raw.get("raw_distance_m")
        lines = []
        if conf is not None:
            lines.append(f"Conf: {float(conf):.2f}")
        if raw_dist is not None:
            lines.append(f"Raw dist: {float(raw_dist):.3f} m")
        if self.gating_show_rejection_overlay:
            reason_text = " / ".join(reasons[:2]) if reasons else "sanity_reject"
            lines.append(f"Rejected: {reason_text}")

        if not lines:
            return

        line_h = 18
        text_x = max(8, x1)
        h = int(frame_bgr.shape[0])

        # Keep text around the box, not over it.
        conf_line = lines[0] if len(lines) >= 1 else None
        other_lines = lines[1:] if len(lines) >= 2 else []

        if conf_line is not None:
            conf_y = y1 - 6
            if conf_y < 14:
                conf_y = min(h - 8, y2 + 18 + (len(other_lines) * line_h))
            cv2.putText(
                frame_bgr,
                conf_line,
                (text_x, conf_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if not other_lines:
            return

        below_start_y = y2 + 18
        needed_below = (len(other_lines) - 1) * line_h
        if below_start_y + needed_below > h - 8:
            below_start_y = max(14, y1 - 6 - needed_below)

        for i, line in enumerate(other_lines):
            cv2.putText(
                frame_bgr,
                line,
                (text_x, below_start_y + i * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    def _annotate_missing_detection(self, frame_bgr, metrics: dict[str, float | int | str]) -> None:
        track_state = str(metrics.get("track_state", "lost")).upper()
        frames_since_detection = int(metrics.get("frames_since_detection", 0))
        distance = metrics.get("distance_m")
        gating_passed = metrics.get("gating_passed")
        was_rejected = str(gating_passed) == "0"

        text = "Measurement rejected" if was_rejected else "No detection"
        if distance is not None and track_state in {"HELD", "STALE"}:
            prefix = "Rejected" if was_rejected else "No detection"
            text = (
                f"{prefix} | {track_state} | "
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
        if not self.show_relative_overlay_on_frame:
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
        process_t0 = time.perf_counter()
        display_frame = frame_bgr.copy()

        results, infer_ms = self._predict(frame_bgr)

        all_candidates, total_detections = self._ranked_detections(results, max_candidates=None)
        active_candidate_limit = self.gating_max_candidates if self.gating_enabled else 1
        candidates = all_candidates[: max(1, int(active_candidate_limit))]
        metrics: dict[str, float | int | str] = {
            "infer_ms": round(infer_ms, 2),
            "detection_count": int(len(candidates)),
            "yolo_detection_count": int(total_detections),
            "candidate_pool": int(len(candidates)),
            "candidate_limit": int(active_candidate_limit),
        }

        if not candidates:
            metrics.update(self._missing_detection_metrics())
            metrics["gating_enabled"] = 1 if self.gating_enabled else 0
            metrics["gating_passed"] = -1
            metrics["gating_reasons"] = "no_detection"
            self._annotate_missing_detection(display_frame, metrics)
            self._draw_relative_overlay(display_frame, metrics)
            self._attach_runtime_metrics(metrics, process_t0)
            return LiveFrameOutput(method=self.name, frame_bgr=display_frame, metrics=metrics)

        # Fast path: gating disabled -> use top-confidence candidate only.
        if not self.gating_enabled:
            xyxy, conf = candidates[0]
            raw = self._raw_measurement_from_detection(xyxy, conf)
            metrics.update(self._filtered_measurement_from_raw(raw))
            metrics["detection_count"] = int(len(candidates))
            metrics["yolo_detection_count"] = int(total_detections)
            metrics["candidate_pool"] = int(len(candidates))
            metrics["selected_candidate_rank"] = 1
            metrics["gating_enabled"] = 0
            metrics["gating_passed"] = -1
            metrics["gating_reasons"] = "disabled"
            self._annotate_best_detection(display_frame, xyxy, metrics)
            self._draw_relative_overlay(display_frame, metrics)
            self._attach_runtime_metrics(metrics, process_t0)
            return LiveFrameOutput(method=self.name, frame_bgr=display_frame, metrics=metrics)

        rejected: list[tuple[int, object, dict[str, float], list[str], dict[str, float] | None]] = []
        for rank, (xyxy, conf) in enumerate(candidates, start=1):
            raw = self._raw_measurement_from_detection(xyxy, conf)
            gate_ok, gate_reasons, raw_rel = self._evaluate_gating(raw, xyxy, frame_bgr.shape)
            if gate_ok:
                metrics.update(self._filtered_measurement_from_raw(raw))
                metrics["detection_count"] = int(len(candidates))
                metrics["yolo_detection_count"] = int(total_detections)
                metrics["candidate_pool"] = int(len(candidates))
                metrics["selected_candidate_rank"] = int(rank)
                metrics["gating_enabled"] = 1
                metrics["gating_passed"] = 1
                metrics["gating_reasons"] = ""
                self._annotate_best_detection(display_frame, xyxy, metrics)
                self._draw_relative_overlay(display_frame, metrics)
                self._attach_runtime_metrics(metrics, process_t0)
                return LiveFrameOutput(method=self.name, frame_bgr=display_frame, metrics=metrics)
            rejected.append((rank, xyxy, raw, gate_reasons, raw_rel))

        # All top-K candidates failed gating -> hold/stale/lost fallback.
        best_rank, best_xyxy, best_raw, best_reasons, best_raw_rel = rejected[0]
        metrics.update(self._missing_detection_metrics())
        metrics.update(
            {
                "confidence": float(best_raw["confidence"]),
                "raw_bbox_width_px": float(best_raw["raw_bbox_width_px"]),
                "raw_bbox_center_x_px": float(best_raw["raw_bbox_center_x_px"]),
                "raw_bbox_center_y_px": float(best_raw["raw_bbox_center_y_px"]),
                "raw_distance_m": float(best_raw["raw_distance_m"]),
                "detection_count": int(len(candidates)),
                "yolo_detection_count": int(total_detections),
                "candidate_pool": int(len(candidates)),
                "selected_candidate_rank": -1,
                "estimate_source": (
                    "history_rejected" if metrics.get("distance_m") is not None else "none_rejected"
                ),
                "gating_enabled": 1,
                "gating_passed": 0,
                "gating_reasons": "|".join(best_reasons),
                "gating_rejected_candidates": int(len(rejected)),
                "best_rejected_rank": int(best_rank),
            }
        )
        if best_raw_rel is not None:
            metrics.update(
                {
                    "raw_x_rel_m": round(float(best_raw_rel["x_rel_m"]), 4),
                    "raw_y_rel_m": round(float(best_raw_rel["y_rel_m"]), 4),
                    "raw_z_rel_m": round(float(best_raw_rel["z_rel_m"]), 4),
                    "raw_yaw_error_rad": round(float(best_raw_rel["yaw_error_rad"]), 4),
                    "raw_yaw_error_deg": round(float(best_raw_rel["yaw_error_deg"]), 2),
                }
            )

        self._annotate_rejected_detection(display_frame, best_xyxy, best_reasons, raw=best_raw)
        self._annotate_missing_detection(display_frame, metrics)
        self._draw_relative_overlay(display_frame, metrics)
        self._attach_runtime_metrics(metrics, process_t0)
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
        print("Live naive depth started.")
        print("Controls: q/ESC quit, g toggle gating.")

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
                cv2.putText(
                    display,
                    f"Gating: {'ON' if self.gating_enabled else 'OFF'} (g)",
                    (10, max(display.shape[0] - 34, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in KEY_QUIT:
                    break
                if key in KEY_TOGGLE_GATING:
                    new_state = self.toggle_gating()
                    print(f"[live] gating={'ON' if new_state else 'OFF'}")
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
