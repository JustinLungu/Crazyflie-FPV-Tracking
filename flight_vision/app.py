from __future__ import annotations

from threading import Event, Thread
from typing import TYPE_CHECKING

from flight_vision.camera_sources import ReceiverCameraSpec, create_receiver_camera_source
from flight_vision.constants import (
    FLIGHT_DRONE_URI,
    FLIGHT_MISSION,
    VISION_CAMERA_BUFFER_SIZE,
    VISION_CAMERA_DEVICE,
    VISION_CAMERA_FPS_HINT,
    VISION_CAMERA_HEIGHT,
    VISION_CAMERA_WIDTH,
    VISION_INFER_DEVICE,
    VISION_MODEL_WEIGHTS,
)
from flight_vision.vision_runtime import (
    OpenCVPresenter,
    OverlayRenderer,
    VisionRuntime,
    YOLODetector,
)
from inference.constants import (
    BOX_LINE_WIDTH,
    INFER_CONF_THRESHOLD,
    INFER_IMAGE_SIZE,
    INFER_IOU_THRESHOLD,
    INFER_MAX_DETECTIONS,
    INFER_VERBOSE,
    KEY_QUIT,
    OVERLAY_FONT_SCALE,
    OVERLAY_LINE_HEIGHT,
    OVERLAY_TEXT_COLOR,
    OVERLAY_TEXT_ORIGIN,
    OVERLAY_THICKNESS,
    SHOW_CONFIDENCE,
    SHOW_LABELS,
    WINDOW_NAME,
)

if TYPE_CHECKING:
    from drone_control.start_drone import DroneControlApp


class ConcurrentFlightVisionApp:
    """
    Runs drone control and YOLO visualization at the same time.

    Design goals:
    - Keep flight-control code owned by drone_control package.
    - Keep vision runtime independent and replaceable.
    - Keep camera capture as a separate component from detector/runtime logic.
    """

    def __init__(
        self,
        *,
        enable_drone_control: bool = True,
        drone_control_app: DroneControlApp | None = None,
    ) -> None:
        self.enable_drone_control = enable_drone_control
        self._mission_name = FLIGHT_MISSION
        self.drone_control_app: DroneControlApp | None = None

        if drone_control_app is not None:
            self.drone_control_app = drone_control_app
            self._mission_name = None
        elif self.enable_drone_control:
            from drone_control.joystick.teleoperation import TeleoperationController
            from drone_control.start_drone import DroneControlApp

            teleop = TeleoperationController(uri=FLIGHT_DRONE_URI)
            self.drone_control_app = DroneControlApp(
                mission=FLIGHT_MISSION,
                teleop=teleop,
            )

        source_spec = ReceiverCameraSpec(
            camera_device=VISION_CAMERA_DEVICE,
            camera_width=VISION_CAMERA_WIDTH,
            camera_height=VISION_CAMERA_HEIGHT,
            camera_fps_hint=VISION_CAMERA_FPS_HINT,
            camera_buffer_size=VISION_CAMERA_BUFFER_SIZE,
        )
        source = create_receiver_camera_source(source_spec)
        detector = YOLODetector(
            model_weights=VISION_MODEL_WEIGHTS,
            image_size=INFER_IMAGE_SIZE,
            conf_threshold=INFER_CONF_THRESHOLD,
            iou_threshold=INFER_IOU_THRESHOLD,
            max_detections=INFER_MAX_DETECTIONS,
            device=VISION_INFER_DEVICE,
            verbose=INFER_VERBOSE,
            show_labels=SHOW_LABELS,
            show_confidence=SHOW_CONFIDENCE,
            box_line_width=BOX_LINE_WIDTH,
        )
        overlay = OverlayRenderer(
            font_scale=OVERLAY_FONT_SCALE,
            thickness=OVERLAY_THICKNESS,
            line_height=OVERLAY_LINE_HEIGHT,
            text_color=OVERLAY_TEXT_COLOR,
            text_origin=OVERLAY_TEXT_ORIGIN,
        )
        presenter = OpenCVPresenter(
            window_name=WINDOW_NAME,
            key_quit=KEY_QUIT,
        )
        self.vision_runtime = VisionRuntime(
            source=source,
            detector=detector,
            overlay=overlay,
            presenter=presenter,
            frame_poll_backoff_s=0.01,
        )

    def run(self) -> None:
        if not self.enable_drone_control:
            # Vision-only mode: useful for camera/model checks without radio hardware.
            self.vision_runtime.run(stop_event=Event())
            return

        if self.drone_control_app is None:
            raise RuntimeError("Drone control is enabled but DroneControlApp was not initialized.")

        stop_event = Event()
        vision_started_event = Event()
        vision_error: list[BaseException] = []

        def _vision_thread_main() -> None:
            try:
                self.vision_runtime.run(
                    stop_event=stop_event,
                    started_event=vision_started_event,
                )
            except BaseException as exc:  # Surface vision failures after flight exits.
                vision_error.append(exc)
                stop_event.set()
                vision_started_event.set()

        vision_thread = Thread(
            target=_vision_thread_main,
            name="flight-vision-runtime",
            daemon=True,
        )
        vision_thread.start()
        vision_started_event.wait(timeout=5.0)

        if vision_error:
            stop_event.set()
            vision_thread.join(timeout=2.0)
            raise RuntimeError(f"Vision runtime failed: {vision_error[0]}") from vision_error[0]

        try:
            self.drone_control_app.run(mission_name=self._mission_name)
        except Exception as exc:
            if "Crazyradio Dongle" in str(exc):
                raise RuntimeError(
                    "Crazyradio dongle not detected. Plug in the dongle or run with --vision-only."
                ) from exc
            raise
        finally:
            stop_event.set()
            vision_thread.join(timeout=2.0)

        if vision_error:
            raise RuntimeError(f"Vision runtime failed: {vision_error[0]}") from vision_error[0]
