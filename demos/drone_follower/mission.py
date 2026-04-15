from __future__ import annotations

from collections.abc import Callable

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
    DEMO_FOLLOW_KP_FORWARD,
    DEMO_FOLLOW_KP_YAW,
    DEMO_FOLLOW_MAX_VX,
    DEMO_FOLLOW_ONLY_ON_MEASUREMENT,
    DEMO_FOLLOW_MAX_YAWRATE_DEG_S,
    DEMO_FOLLOW_TAKEOFF_HEIGHT_M,
    DEMO_FOLLOW_TARGET_DISTANCE_M,
    DEMO_FOLLOW_YAW_DEADBAND_DEG,
    DEMO_MISSION_COMPLETE_ON_PREVIEW_QUIT,
    DEMO_PREVIEW_WINDOW_NAME,
    DEMO_SHOW_PREVIEW,
    DEMO_TEXT_ACCENT_COLOR,
    DEMO_TEXT_COLOR,
    DEMO_TEXT_LINE_HEIGHT,
    DEMO_TEXT_ORIGIN,
    DEMO_TEXT_SCALE,
    DEMO_TEXT_THICKNESS,
    KEY_PREVIEW_TOGGLE_GATING,
    KEY_PREVIEW_QUIT,
)
from depth_estimation.naive_bbox_depth.pipeline import NaiveBBoxDepthPipeline
from depth_estimation.pipeline_base import LiveDepthPipeline
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
    - commands forward velocity + yaw rate to keep target centered and at set distance
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
        distance_deadband_m: float = DEMO_FOLLOW_DISTANCE_DEADBAND_M,
        kp_yaw: float = DEMO_FOLLOW_KP_YAW,
        max_yawrate_deg_s: float = DEMO_FOLLOW_MAX_YAWRATE_DEG_S,
        yaw_deadband_deg: float = DEMO_FOLLOW_YAW_DEADBAND_DEG,
        show_preview: bool = DEMO_SHOW_PREVIEW,
        mission_complete_on_preview_quit: bool = DEMO_MISSION_COMPLETE_ON_PREVIEW_QUIT,
        pipeline_factory: DepthPipelineFactory | None = None,
    ):
        self.target_distance_m = float(target_distance_m)
        self.takeoff_height_m = float(takeoff_height_m)
        self.dt = float(dt)

        self.kp_forward = float(kp_forward)
        self.max_vx = float(max_vx)
        self.follow_only_on_measurement = bool(follow_only_on_measurement)
        self.distance_deadband_m = float(distance_deadband_m)

        self.kp_yaw = float(kp_yaw)
        self.max_yawrate_deg_s = float(max_yawrate_deg_s)
        self.yaw_deadband_deg = float(yaw_deadband_deg)

        self.show_preview = bool(show_preview)
        self.mission_complete_on_preview_quit = bool(mission_complete_on_preview_quit)
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

    def _compute_command(self, metrics: dict) -> tuple[float, float, str]:
        track_state = str(metrics.get("track_state", "lost")).lower()
        estimate_source = str(metrics.get("estimate_source", "none")).lower()

        if self.follow_only_on_measurement:
            try:
                detection_count = int(metrics.get("detection_count", 0))
            except (TypeError, ValueError):
                detection_count = 0
            if track_state != "tracked":
                return 0.0, 0.0, f"wait_{track_state}"
            if estimate_source != "measurement":
                return 0.0, 0.0, f"wait_src_{estimate_source}"
            if detection_count <= 0:
                return 0.0, 0.0, "wait_no_detection"

        if track_state != "tracked":
            return 0.0, 0.0, f"{track_state}_hold"

        z_rel = _as_float(metrics.get("z_rel_m"))
        yaw_err_deg = _as_float(metrics.get("yaw_error_deg"))
        if z_rel is None or yaw_err_deg is None:
            return 0.0, 0.0, "missing_pose"

        dist_err = z_rel - self.target_distance_m
        vx = self.kp_forward * dist_err
        if abs(dist_err) < self.distance_deadband_m:
            vx = 0.0
        vx = self._clamp(vx, -self.max_vx, self.max_vx)

        yawrate = self.kp_yaw * yaw_err_deg
        if abs(yaw_err_deg) < self.yaw_deadband_deg:
            yawrate = 0.0
        yawrate = self._clamp(yawrate, -self.max_yawrate_deg_s, self.max_yawrate_deg_s)

        reason = f"tracked z={z_rel:.2f}m target={self.target_distance_m:.2f}m yaw={yaw_err_deg:.1f}deg"
        return vx, yawrate, reason

    def _draw_demo_overlay(
        self,
        frame_bgr,
        method_name: str,
        frame_idx: int,
        vx_cmd: float,
        yawrate_cmd: float,
        reason: str,
        metrics: dict,
    ) -> None:
        x, y = DEMO_TEXT_ORIGIN
        lines = [
            "Drone Follower Demo",
            f"method: {method_name}",
            f"frame: {frame_idx}",
            f"track_state: {metrics.get('track_state', 'unknown')}",
            f"distance: {metrics.get('distance_m', 'n/a')}",
            f"yaw_err_deg: {metrics.get('yaw_error_deg', 'n/a')}",
            f"vx_cmd: {vx_cmd:.2f} m/s",
            f"yawrate_cmd: {yawrate_cmd:.1f} deg/s",
            f"gating: {'ON' if self._is_gating_enabled(metrics) else 'OFF'}",
            f"mode: {reason}",
            "controls: q/ESC preview-stop, g toggle gating",
            "safety: joystick/button input forces takeover",
        ]
        for i, line in enumerate(lines):
            color = DEMO_TEXT_ACCENT_COLOR if i == 0 else DEMO_TEXT_COLOR
            cv2.putText(
                frame_bgr,
                line,
                (x, y + i * DEMO_TEXT_LINE_HEIGHT),
                cv2.FONT_HERSHEY_SIMPLEX,
                DEMO_TEXT_SCALE,
                color,
                DEMO_TEXT_THICKNESS,
                cv2.LINE_AA,
            )

    @staticmethod
    def _is_gating_enabled(metrics: dict) -> bool:
        raw = metrics.get("gating_enabled", 0)
        try:
            return int(raw) == 1
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _try_toggle_gating(pipeline: LiveDepthPipeline) -> bool | None:
        toggle = getattr(pipeline, "toggle_gating", None)
        if callable(toggle):
            return bool(toggle())
        return None

    def run(self, ctx: TakeoverContext) -> bool:
        print("Demo mission: drone_follower")
        print("Safety: touch joystick/button any time for teleop takeover.")

        if ctx.ensure_takeoff(self.takeoff_height_m):
            return False

        pipeline = self.pipeline_factory()
        cap = self._open_camera()
        frame_idx = 0
        method_name = str(getattr(pipeline, "name", "unknown"))

        try:
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

                vx_cmd, yawrate_cmd, reason = self._compute_command(metrics)
                if ctx.command(vx=vx_cmd, vy=0.0, vz=0.0, yawrate=yawrate_cmd, duration_s=self.dt):
                    return False

                if self.show_preview:
                    display = output.frame_bgr.copy()
                    self._draw_demo_overlay(
                        display,
                        method_name=method_name,
                        frame_idx=frame_idx,
                        vx_cmd=vx_cmd,
                        yawrate_cmd=yawrate_cmd,
                        reason=reason,
                        metrics=metrics,
                    )
                    cv2.imshow(DEMO_PREVIEW_WINDOW_NAME, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in KEY_PREVIEW_QUIT:
                        print("Preview quit requested.")
                        return bool(self.mission_complete_on_preview_quit)
                    if key in KEY_PREVIEW_TOGGLE_GATING:
                        state = self._try_toggle_gating(pipeline)
                        if state is None:
                            print(f"[demo] gating toggle ignored: method '{method_name}' has no gating.")
                        else:
                            print(f"[demo] gating={'ON' if state else 'OFF'}")
        finally:
            cap.release()
            if self.show_preview:
                try:
                    cv2.destroyWindow(DEMO_PREVIEW_WINDOW_NAME)
                except cv2.error:
                    pass
            pipeline.close()
