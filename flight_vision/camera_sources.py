from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cv2


class FrameSource(ABC):
    """
    Abstraction for any frame provider (USB receiver today, custom sources later).
    """

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> tuple[bool, Any]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class OpenCVCaptureSource(FrameSource):
    def __init__(
        self,
        source: str | int,
        *,
        width: int | None = None,
        height: int | None = None,
        fps_hint: int | None = None,
        buffer_size: int | None = None,
        backend: int | None = None,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps_hint = fps_hint
        self.buffer_size = buffer_size
        self.backend = backend
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self._cap is not None:
            return

        if self.backend is None:
            cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(self.source, self.backend)

        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps_hint is not None:
            cap.set(cv2.CAP_PROP_FPS, self.fps_hint)
        if self.buffer_size is not None:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.source}")
        self._cap = cap

    def read(self) -> tuple[bool, Any]:
        if self._cap is None:
            raise RuntimeError("Camera source must be opened before read()")
        return self._cap.read()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


@dataclass(slots=True, frozen=True)
class ReceiverCameraSpec:
    camera_device: str
    camera_width: int
    camera_height: int
    camera_fps_hint: int
    camera_buffer_size: int


def create_receiver_camera_source(spec: ReceiverCameraSpec) -> FrameSource:
    return OpenCVCaptureSource(
        spec.camera_device,
        width=spec.camera_width,
        height=spec.camera_height,
        fps_hint=spec.camera_fps_hint,
        buffer_size=spec.camera_buffer_size,
        backend=cv2.CAP_V4L2,
    )
