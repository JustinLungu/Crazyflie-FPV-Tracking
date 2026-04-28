from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionCandidate:
    xyxy: tuple[float, float, float, float]
    confidence: float
    source: str = "yolo"
    tracker_age_frames: int = 0


def _frame_size(frame_bgr: np.ndarray) -> tuple[int, int]:
    height, width = frame_bgr.shape[:2]
    return int(width), int(height)


def clamp_xyxy_to_frame(
    xyxy: tuple[float, float, float, float],
    frame_bgr: np.ndarray,
) -> tuple[float, float, float, float]:
    width, height = _frame_size(frame_bgr)
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(x1), float(width - 1)))
    y1 = max(0.0, min(float(y1), float(height - 1)))
    x2 = max(0.0, min(float(x2), float(width - 1)))
    y2 = max(0.0, min(float(y2), float(height - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def xyxy_to_xywh(
    xyxy: tuple[float, float, float, float],
    frame_bgr: np.ndarray,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = clamp_xyxy_to_frame(xyxy, frame_bgr)
    x = int(round(x1))
    y = int(round(y1))
    w = max(1, int(round(x2 - x1)))
    h = max(1, int(round(y2 - y1)))
    frame_w, frame_h = _frame_size(frame_bgr)
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    return x, y, max(1, w), max(1, h)


def xywh_to_xyxy(
    xywh: tuple[float, float, float, float],
    frame_bgr: np.ndarray,
) -> tuple[float, float, float, float]:
    x, y, w, h = map(float, xywh)
    return clamp_xyxy_to_frame((x, y, x + w, y + h), frame_bgr)


def bbox_area_xyxy(xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def bbox_center_xyxy(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0


def is_center_inside_expanded_bbox(
    candidate_xyxy: tuple[float, float, float, float],
    reference_xyxy: tuple[float, float, float, float],
    margin_px: float,
) -> bool:
    cx, cy = bbox_center_xyxy(candidate_xyxy)
    x1, y1, x2, y2 = reference_xyxy
    margin = max(0.0, float(margin_px))
    return (x1 - margin) <= cx <= (x2 + margin) and (y1 - margin) <= cy <= (y2 + margin)


def create_opencv_tracker(tracker_type: str):
    tracker_name = tracker_type.strip().upper()
    if not tracker_name:
        raise ValueError("Tracker type cannot be empty.")

    factory_name = f"Tracker{tracker_name}_create"
    class_name = f"Tracker{tracker_name}"
    factories = []

    direct_factory = getattr(cv2, factory_name, None)
    if direct_factory is not None:
        factories.append(direct_factory)

    direct_class = getattr(cv2, class_name, None)
    direct_class_factory = getattr(direct_class, "create", None)
    if direct_class_factory is not None:
        factories.append(direct_class_factory)

    legacy = getattr(cv2, "legacy", None)
    legacy_factory = getattr(legacy, factory_name, None) if legacy is not None else None
    if legacy_factory is not None:
        factories.append(legacy_factory)

    legacy_class = getattr(legacy, class_name, None) if legacy is not None else None
    legacy_class_factory = getattr(legacy_class, "create", None)
    if legacy_class_factory is not None:
        factories.append(legacy_class_factory)

    if not factories:
        available = available_opencv_trackers()
        available_text = ", ".join(available) if available else "none"
        raise RuntimeError(
            f"OpenCV tracker '{tracker_name}' is unavailable. "
            f"Available trackers in this environment: {available_text}. "
            "Install/fix opencv-contrib-python or choose another tracker."
        )

    last_error: Exception | None = None
    for factory in factories:
        try:
            return factory()
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not create OpenCV tracker '{tracker_name}'.") from last_error


def available_opencv_trackers() -> list[str]:
    trackers: set[str] = set()

    def collect(namespace) -> None:
        if namespace is None:
            return
        for name in dir(namespace):
            if not name.startswith("Tracker"):
                continue
            if name.endswith("_create"):
                tracker = name.removeprefix("Tracker").removesuffix("_create")
                if tracker:
                    trackers.add(tracker.upper())
            else:
                tracker_cls = getattr(namespace, name, None)
                if getattr(tracker_cls, "create", None) is not None:
                    tracker = name.removeprefix("Tracker")
                    if tracker:
                        trackers.add(tracker.upper())

    collect(cv2)
    collect(getattr(cv2, "legacy", None))
    return sorted(trackers)


def extract_yolo_detections(results) -> list[DetectionCandidate]:
    if results is None:
        return []

    if isinstance(results, Iterable) and not hasattr(results, "boxes"):
        result_iter = results
    else:
        result_iter = (results,)

    candidates: list[DetectionCandidate] = []
    for result in result_iter:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue

        xyxy_values = boxes.xyxy.cpu().tolist()
        if getattr(boxes, "conf", None) is None:
            conf_values = [0.0] * len(xyxy_values)
        else:
            conf_values = boxes.conf.cpu().tolist()

        for xyxy_raw, conf_raw in zip(xyxy_values, conf_values):
            if len(xyxy_raw) < 4:
                continue
            x1, y1, x2, y2 = map(float, xyxy_raw[:4])
            candidates.append(
                DetectionCandidate(
                    xyxy=(x1, y1, x2, y2),
                    confidence=float(conf_raw),
                    source="yolo",
                )
            )
    return candidates


def rank_detections(
    candidates: list[DetectionCandidate],
    max_candidates: int | None = None,
) -> list[DetectionCandidate]:
    ranked = sorted(candidates, key=lambda item: item.confidence, reverse=True)
    if max_candidates is None:
        return ranked
    return ranked[: max(1, int(max_candidates))]


class DetectionFailsafeTracker:
    def __init__(
        self,
        *,
        enabled: bool,
        tracker_type: str,
        max_fallback_frames: int,
        min_bbox_area_px: float,
        max_center_jump_px: float | None,
        reinitialize_on_detection: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.tracker_type = tracker_type.strip().upper()
        self.max_fallback_frames = max(0, int(max_fallback_frames))
        self.min_bbox_area_px = max(0.0, float(min_bbox_area_px))
        self.max_center_jump_px = (
            None if max_center_jump_px is None else max(0.0, float(max_center_jump_px))
        )
        self.reinitialize_on_detection = bool(reinitialize_on_detection)

        self._tracker = None
        self._fallback_frames = 0
        self._last_confidence = 0.0
        self._last_xyxy: tuple[float, float, float, float] | None = None
        self._last_rejection_reason = ""
        self._unavailable_reason: str | None = None
        self._warned_unavailable = False

    @property
    def is_active(self) -> bool:
        return self._tracker is not None

    @property
    def fallback_age_frames(self) -> int:
        return self._fallback_frames

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    @property
    def last_rejection_reason(self) -> str:
        return self._last_rejection_reason

    def reset(self) -> None:
        self._tracker = None
        self._fallback_frames = 0
        self._last_confidence = 0.0
        self._last_xyxy = None

    def initialize(self, frame_bgr: np.ndarray, detection: DetectionCandidate) -> bool:
        if not self.enabled:
            return False
        if detection.source != "yolo":
            return False
        if self.is_active and not self.reinitialize_on_detection:
            return True

        xywh = xyxy_to_xywh(detection.xyxy, frame_bgr)
        xyxy = xywh_to_xyxy(xywh, frame_bgr)
        if bbox_area_xyxy(xyxy) < self.min_bbox_area_px:
            self.reset()
            return False

        try:
            tracker = create_opencv_tracker(self.tracker_type)
        except RuntimeError as exc:
            self._unavailable_reason = str(exc)
            if not self._warned_unavailable:
                print(f"[tracker-failsafe] {self._unavailable_reason}")
                self._warned_unavailable = True
            self.reset()
            return False

        self._unavailable_reason = None
        ok = tracker.init(frame_bgr, xywh)
        if ok is False:
            self.reset()
            return False

        self._tracker = tracker
        self._fallback_frames = 0
        self._last_confidence = float(detection.confidence)
        self._last_xyxy = xyxy
        self._last_rejection_reason = ""
        return True

    def update(self, frame_bgr: np.ndarray) -> DetectionCandidate | None:
        if not self.enabled or self._tracker is None:
            return None
        if self.max_fallback_frames <= 0 or self._fallback_frames >= self.max_fallback_frames:
            self.reset()
            return None

        ok, xywh = self._tracker.update(frame_bgr)
        if not ok:
            self.reset()
            return None

        xyxy = xywh_to_xyxy(tuple(map(float, xywh)), frame_bgr)
        if bbox_area_xyxy(xyxy) < self.min_bbox_area_px:
            self._last_rejection_reason = f"tracker_area<{self.min_bbox_area_px:.1f}px"
            self.reset()
            return None

        if (
            self.max_center_jump_px is not None
            and self._last_xyxy is not None
            and not is_center_inside_expanded_bbox(
                candidate_xyxy=xyxy,
                reference_xyxy=self._last_xyxy,
                margin_px=self.max_center_jump_px,
            )
        ):
            self._last_rejection_reason = f"tracker_jump>{self.max_center_jump_px:.1f}px"
            self.reset()
            return None

        self._fallback_frames += 1
        self._last_xyxy = xyxy
        self._last_rejection_reason = ""
        return DetectionCandidate(
            xyxy=xyxy,
            confidence=self._last_confidence,
            source=f"tracker_{self.tracker_type.lower()}",
            tracker_age_frames=self._fallback_frames,
        )


def draw_detection_candidate(
    frame_bgr: np.ndarray,
    detection: DetectionCandidate,
    *,
    color: tuple[int, int, int],
    label: str,
    line_width: int,
) -> None:
    x1, y1, x2, y2 = map(int, detection.xyxy)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, line_width)

    text = label
    if detection.tracker_age_frames > 0:
        text = f"{label} ({detection.tracker_age_frames})"

    text_y = y1 - 8
    if text_y < 16:
        text_y = y2 + 18
    cv2.putText(
        frame_bgr,
        text,
        (max(0, x1), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        max(1, line_width),
        cv2.LINE_AA,
    )
