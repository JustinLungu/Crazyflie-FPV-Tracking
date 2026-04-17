from __future__ import annotations

from collections.abc import Callable
import time

import cv2

from demos.drone_follower.constants import (
    DEMO_CAMERA_BUFFER_SIZE,
    DEMO_CAMERA_DEVICE,
    DEMO_CAMERA_FOURCC,
    DEMO_CAMERA_FPS_HINT,
    DEMO_CAMERA_HEIGHT,
    DEMO_CAMERA_WIDTH,
    DEMO_FOLLOW_CONTROL_DT,
    DEMO_FOLLOW_DISTANCE_DEADBAND_M,
    DEMO_FOLLOW_ENABLE_VERTICAL,
    DEMO_FOLLOW_KP_FORWARD,
    DEMO_FOLLOW_KP_YAW,
    DEMO_FOLLOW_KP_VERTICAL,
    DEMO_FOLLOW_MAX_VX,
    DEMO_FOLLOW_MAX_VZ,
    DEMO_FOLLOW_ONLY_ON_MEASUREMENT,
    DEMO_PRECONTROL_CV_WARMUP_FRAMES,
    DEMO_FOLLOW_MAX_YAWRATE_DEG_S,
    DEMO_FOLLOW_TAKEOFF_HEIGHT_M,
    DEMO_FOLLOW_TARGET_DISTANCE_M,
    DEMO_FOLLOW_VERTICAL_DEADBAND_M,
    DEMO_FOLLOW_YAW_DEADBAND_DEG,
    DEMO_PREVIEW_WINDOW_NAME,
    DEMO_SHOW_PREVIEW,
    KEY_PREVIEW_QUIT,
    KEY_PREVIEW_TOGGLE_GATING,
)
from depth_estimation.live_depth_review import combine_frames, compose_display
from depth_estimation.naive_bbox_depth.pipeline import NaiveBBoxDepthPipeline
from depth_estimation.pipeline_base import LiveDepthPipeline, LiveFrameOutput
from drone_control.autonomous.takeover_runner import AutonomousMission, TakeoverContext


DepthPipelineFactory = Callable[[], LiveDepthPipeline]


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class DroneFollowerMission(AutonomousMission):
    """
    Mission loop:
    - reads live camera frames
    - runs selected live depth tracking pipeline
    - commands forward velocity + yaw rate + vertical velocity
      to keep target centered and at set distance
    - always allows joystick takeover via TakeoverRunner/TakeoverContext
    """

    def __init__(
        self,
        target_distance_m: float = DEMO_FOLLOW_TARGET_DISTANCE_M,
        takeoff_height_m: float = DEMO_FOLLOW_TAKEOFF_HEIGHT_M,
        dt: float = DEMO_FOLLOW_CONTROL_DT,
        kp_forward: float = DEMO_FOLLOW_KP_FORWARD,
        max_vx: float = DEMO_FOLLOW_MAX_VX,
        follow_only_on_measurement: bool = DEMO_FOLLOW_ONLY_ON_MEASUREMENT,
        precontrol_cv_warmup_frames: int = DEMO_PRECONTROL_CV_WARMUP_FRAMES,
        distance_deadband_m: float = DEMO_FOLLOW_DISTANCE_DEADBAND_M,
        enable_vertical: bool = DEMO_FOLLOW_ENABLE_VERTICAL,
        kp_vertical: float = DEMO_FOLLOW_KP_VERTICAL,
        max_vz: float = DEMO_FOLLOW_MAX_VZ,
        vertical_deadband_m: float = DEMO_FOLLOW_VERTICAL_DEADBAND_M,
        kp_yaw: float = DEMO_FOLLOW_KP_YAW,
        max_yawrate_deg_s: float = DEMO_FOLLOW_MAX_YAWRATE_DEG_S,
        yaw_deadband_deg: float = DEMO_FOLLOW_YAW_DEADBAND_DEG,
        show_preview: bool = DEMO_SHOW_PREVIEW,
        pipeline_factory: DepthPipelineFactory | None = None,
    ):
        self.target_distance_m = float(target_distance_m)
        self.takeoff_height_m = float(takeoff_height_m)
        self.dt = float(dt)

        self.kp_forward = float(kp_forward)
        self.max_vx = float(max_vx)
        self.follow_only_on_measurement = bool(follow_only_on_measurement)
        self.precontrol_cv_warmup_frames = max(1, int(precontrol_cv_warmup_frames))
        self.distance_deadband_m = float(distance_deadband_m)
        self.enable_vertical = bool(enable_vertical)
        self.kp_vertical = float(kp_vertical)
        self.max_vz = float(max_vz)
        self.vertical_deadband_m = float(vertical_deadband_m)

        self.kp_yaw = float(kp_yaw)
        self.max_yawrate_deg_s = float(max_yawrate_deg_s)
        self.yaw_deadband_deg = float(yaw_deadband_deg)

        self.show_preview = bool(show_preview)
        self.pipeline_factory = pipeline_factory or NaiveBBoxDepthPipeline

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(DEMO_CAMERA_DEVICE, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*DEMO_CAMERA_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEMO_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEMO_CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, DEMO_CAMERA_FPS_HINT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, DEMO_CAMERA_BUFFER_SIZE)
        if not cap.isOpened():
            cap = cv2.VideoCapture(DEMO_CAMERA_DEVICE)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera at {DEMO_CAMERA_DEVICE}")
        return cap

    def _compute_command(self, metrics: dict) -> tuple[float, float, float, str]:
        track_state = str(metrics.get("track_state", "lost")).lower()
        estimate_source = str(metrics.get("estimate_source", "none")).lower()

        if self.follow_only_on_measurement:
            try:
                detection_count = int(metrics.get("detection_count", 0))
            except (TypeError, ValueError):
                detection_count = 0
            if track_state != "tracked":
                return 0.0, 0.0, 0.0, f"wait_{track_state}"
            if estimate_source != "measurement":
                return 0.0, 0.0, 0.0, f"wait_src_{estimate_source}"
            if detection_count <= 0:
                return 0.0, 0.0, 0.0, "wait_no_detection"

        if track_state != "tracked":
            return 0.0, 0.0, 0.0, f"{track_state}_hold"

        z_rel = _as_float(metrics.get("z_rel_m"))
        y_rel = _as_float(metrics.get("y_rel_m"))
        yaw_err_deg = _as_float(metrics.get("yaw_error_deg"))
        if z_rel is None or yaw_err_deg is None:
            return 0.0, 0.0, 0.0, "missing_pose"
        if self.enable_vertical and y_rel is None:
            return 0.0, 0.0, 0.0, "missing_vertical_pose"

        dist_err = z_rel - self.target_distance_m
        vx = self.kp_forward * dist_err
        if abs(dist_err) < self.distance_deadband_m:
            vx = 0.0
        vx = self._clamp(vx, -self.max_vx, self.max_vx)

        vz = 0.0
        if self.enable_vertical and y_rel is not None:
            vz = self.kp_vertical * y_rel
            if abs(y_rel) < self.vertical_deadband_m:
                vz = 0.0
            vz = self._clamp(vz, -self.max_vz, self.max_vz)

        yawrate = self.kp_yaw * yaw_err_deg
        if abs(yaw_err_deg) < self.yaw_deadband_deg:
            yawrate = 0.0
        yawrate = self._clamp(yawrate, -self.max_yawrate_deg_s, self.max_yawrate_deg_s)

        if self.enable_vertical and y_rel is not None:
            reason = (
                f"tracked z={z_rel:.2f}m target={self.target_distance_m:.2f}m "
                f"y={y_rel:.2f}m yaw={yaw_err_deg:.1f}deg"
            )
        else:
            reason = f"tracked z={z_rel:.2f}m target={self.target_distance_m:.2f}m yaw={yaw_err_deg:.1f}deg"
        return vx, vz, yawrate, reason

    def _update_last_pose(self, output: LiveFrameOutput) -> None:
        m = output.metrics
        x_rel = _as_float(m.get("x_rel_m"))
        y_rel = _as_float(m.get("y_rel_m"))
        z_rel = _as_float(m.get("z_rel_m"))
        yaw_deg = _as_float(m.get("yaw_error_deg"))
        if x_rel is None or y_rel is None or z_rel is None or yaw_deg is None:
            return
        self._last_pose_by_method[output.method] = {
            "x_rel_m": x_rel,
            "y_rel_m": y_rel,
            "z_rel_m": z_rel,
            "yaw_error_deg": yaw_deg,
        }

    def _update_loop_fps(self) -> float:
        now = time.perf_counter()
        dt = max(1e-6, now - self._last_t)
        inst_fps = 1.0 / dt
        self._loop_fps = inst_fps if self._loop_fps <= 0.0 else (0.2 * inst_fps + 0.8 * self._loop_fps)
        self._last_t = now
        return self._loop_fps

    @staticmethod
    def _try_toggle_gating(pipeline: LiveDepthPipeline) -> bool | None:
        toggle = getattr(pipeline, "toggle_gating", None)
        if callable(toggle):
            return bool(toggle())
        return None

    def _render_preview(
        self,
        output: LiveFrameOutput,
        frame_idx: int,
        method_name: str,
        pipeline: LiveDepthPipeline,
    ) -> bool:
        self._update_last_pose(output)
        combined = combine_frames([output], target_height=DEMO_CAMERA_HEIGHT)
        display = compose_display(
            combined_frame=combined,
            frame_idx=frame_idx,
            loop_fps=self._update_loop_fps(),
            methods=[output.method],
            outputs=[output],
            last_pose_by_method=self._last_pose_by_method,
        )
        cv2.imshow(DEMO_PREVIEW_WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key in KEY_PREVIEW_QUIT:
            print("Preview quit requested.")
            return True
        if key in KEY_PREVIEW_TOGGLE_GATING:
            state = self._try_toggle_gating(pipeline)
            if state is None:
                print(f"[demo] gating toggle ignored: method '{method_name}' has no gating.")
            else:
                print(f"[demo] gating={'ON' if state else 'OFF'}")
        return False

    def _warmup_cv(
        self,
        cap: cv2.VideoCapture,
        pipeline: LiveDepthPipeline,
        frame_idx: int,
        method_name: str,
    ) -> tuple[int, bool]:
        print(
            "Initializing CV before control "
            f"({self.precontrol_cv_warmup_frames} frames warm-up)."
        )
        warmed = 0
        while warmed < self.precontrol_cv_warmup_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            frame_idx += 1
            warmed += 1
            output = pipeline.process_live_frame(frame_bgr)

            if self.show_preview:
                should_quit = self._render_preview(output, frame_idx, method_name, pipeline)
                if should_quit:
                    return frame_idx, True
        print("CV ready. Engaging flight control.")
        return frame_idx, False

    def _land_on_preview_quit(self, ctx: TakeoverContext) -> None:
        # Force a graceful stop path: hover-stop then land before full shutdown.
        if not getattr(ctx.teleop, "flying", False):
            return

        print("Preview quit requested: landing before exit.")
        try:
            ctx.stop(0.15)
        except Exception:
            pass
        try:
            ctx.teleop.land()
        except Exception as exc:
            print(f"Warning: land request on preview quit failed: {exc}")

    def run(self, ctx: TakeoverContext) -> bool:
        print("Demo mission: drone_follower")
        print("Safety: touch joystick/button any time for teleop takeover.")

        pipeline = self.pipeline_factory()
        cap = self._open_camera()
        frame_idx = 0
        method_name = str(getattr(pipeline, "name", "unknown"))
        self._last_pose_by_method: dict[str, dict[str, float]] = {}
        self._last_t = time.perf_counter()
        self._loop_fps = 0.0
        has_taken_off = False

        try:
            frame_idx, quit_requested = self._warmup_cv(cap, pipeline, frame_idx, method_name)
            if quit_requested:
                return True

            if ctx.ensure_takeoff(self.takeoff_height_m):
                return False
            has_taken_off = True

            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    print("Camera read failed. Hovering and continuing.")
                    if ctx.stop(self.dt):
                        return False
                    continue

                frame_idx += 1
                output = pipeline.process_live_frame(frame_bgr)
                metrics = output.metrics

                vx_cmd, vz_cmd, yawrate_cmd, reason = self._compute_command(metrics)
                if ctx.command(vx=vx_cmd, vy=0.0, vz=vz_cmd, yawrate=yawrate_cmd, duration_s=self.dt):
                    return False

                if self.show_preview:
                    should_quit = self._render_preview(
                        output=output,
                        frame_idx=frame_idx,
                        method_name=method_name,
                        pipeline=pipeline,
                    )
                    if should_quit:
                        if has_taken_off:
                            self._land_on_preview_quit(ctx)
                        return True
        finally:
            cap.release()
            if self.show_preview:
                try:
                    cv2.destroyWindow(DEMO_PREVIEW_WINDOW_NAME)
                except cv2.error:
                    pass
            pipeline.close()
