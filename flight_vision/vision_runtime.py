from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event
from typing import Iterable

import cv2

from flight_vision.camera_sources import FrameSource
from inference.utils import load_yolo_model


@dataclass(slots=True, frozen=True)
class DetectionOutput:
    frame: object
    detection_count: int
    inference_ms: float


class YOLODetector:
    def __init__(
        self,
        *,
        model_weights: str,
        image_size: int,
        conf_threshold: float,
        iou_threshold: float,
        max_detections: int,
        device: str | int,
        verbose: bool,
        show_labels: bool,
        show_confidence: bool,
        box_line_width: int,
    ) -> None:
        self.model = load_yolo_model(model_weights)
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = device
        self.verbose = verbose
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.box_line_width = box_line_width

    def detect(self, frame: object) -> DetectionOutput:
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            device=self.device,
            verbose=self.verbose,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0

        result = results[0]
        annotated = result.plot(
            labels=self.show_labels,
            conf=self.show_confidence,
            line_width=self.box_line_width,
        )
        detection_count = 0 if result.boxes is None else len(result.boxes)
        return DetectionOutput(
            frame=annotated,
            detection_count=detection_count,
            inference_ms=infer_ms,
        )


class OverlayRenderer:
    def __init__(
        self,
        *,
        font_scale: float,
        thickness: int,
        line_height: int,
        text_color: tuple[int, int, int],
        text_origin: tuple[int, int],
    ) -> None:
        self.font_scale = font_scale
        self.thickness = thickness
        self.line_height = line_height
        self.text_color = text_color
        self.text_origin = text_origin

    def draw(self, frame: object, detection_count: int, inference_ms: float, display_fps: float) -> None:
        x, y = self.text_origin
        lines = [
            f"detections: {detection_count}",
            f"inference: {inference_ms:.1f} ms",
            f"display fps: {display_fps:.1f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x, y + i * self.line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.text_color,
                self.thickness,
                cv2.LINE_AA,
            )


class OpenCVPresenter:
    def __init__(
        self,
        *,
        window_name: str,
        key_quit: Iterable[int],
    ) -> None:
        self.window_name = window_name
        self.key_quit = frozenset(key_quit)

    def show(self, frame: object) -> bool:
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key not in self.key_quit

    @staticmethod
    def close() -> None:
        cv2.destroyAllWindows()


class VisionRuntime:
    def __init__(
        self,
        source: FrameSource,
        detector: YOLODetector,
        overlay: OverlayRenderer,
        presenter: OpenCVPresenter,
        *,
        frame_poll_backoff_s: float,
    ) -> None:
        self.source = source
        self.detector = detector
        self.overlay = overlay
        self.presenter = presenter
        self.frame_poll_backoff_s = frame_poll_backoff_s

    def run(self, stop_event: Event, started_event: Event | None = None) -> None:
        self.source.open()
        if started_event is not None:
            started_event.set()
        prev_loop_time = time.perf_counter()
        try:
            while not stop_event.is_set():
                ok, frame = self.source.read()
                if not ok:
                    # Avoid hot loop when feed drops.
                    time.sleep(self.frame_poll_backoff_s)
                    continue

                output = self.detector.detect(frame)
                now = time.perf_counter()
                loop_dt = max(1e-6, now - prev_loop_time)
                prev_loop_time = now
                display_fps = 1.0 / loop_dt

                self.overlay.draw(
                    output.frame,
                    detection_count=output.detection_count,
                    inference_ms=output.inference_ms,
                    display_fps=display_fps,
                )
                should_continue = self.presenter.show(output.frame)
                if not should_continue:
                    stop_event.set()
                    break
        finally:
            self.source.close()
            self.presenter.close()
