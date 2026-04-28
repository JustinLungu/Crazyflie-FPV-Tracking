import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from inference import tracker_failsafe
from inference.tracker_failsafe import DetectionCandidate, DetectionFailsafeTracker


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)

    def __len__(self):
        return len(self.xyxy.tolist())


class FakeResult:
    def __init__(self, xyxy, conf):
        self.boxes = FakeBoxes(xyxy, conf)


class FakeTracker:
    def __init__(self, updates):
        self.updates = list(updates)
        self.init_bbox = None

    def init(self, _frame, bbox):
        self.init_bbox = bbox
        return True

    def update(self, _frame):
        if not self.updates:
            return False, (0, 0, 0, 0)
        return self.updates.pop(0)


class TrackerFailsafeTests(unittest.TestCase):
    def test_extract_yolo_detections_ranks_by_confidence(self):
        result = FakeResult(
            xyxy=[
                [1.0, 2.0, 5.0, 6.0],
                [3.0, 4.0, 12.0, 14.0],
            ],
            conf=[0.25, 0.9],
        )

        detections = tracker_failsafe.extract_yolo_detections(result)
        ranked = tracker_failsafe.rank_detections(detections)

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].confidence, 0.9)
        self.assertEqual(ranked[0].xyxy, (3.0, 4.0, 12.0, 14.0))
        self.assertEqual(ranked[0].source, "yolo")

    def test_tracker_initializes_from_yolo_and_limits_fallback_frames(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        created_trackers: list[FakeTracker] = []

        def fake_create_tracker(_tracker_type):
            tracker = FakeTracker(updates=[(True, (6, 7, 10, 8))])
            created_trackers.append(tracker)
            return tracker

        original_create_tracker = tracker_failsafe.create_opencv_tracker
        tracker_failsafe.create_opencv_tracker = fake_create_tracker
        try:
            failsafe = DetectionFailsafeTracker(
                enabled=True,
                tracker_type="CSRT",
                max_fallback_frames=1,
                min_bbox_area_px=4.0,
                max_center_jump_px=50.0,
                reinitialize_on_detection=True,
            )
            initialized = failsafe.initialize(
                frame,
                DetectionCandidate(
                    xyxy=(5.0, 6.0, 15.0, 14.0),
                    confidence=0.82,
                    source="yolo",
                ),
            )

            self.assertTrue(initialized)
            self.assertEqual(created_trackers[0].init_bbox, (5, 6, 10, 8))

            fallback = failsafe.update(frame)
            self.assertIsNotNone(fallback)
            self.assertEqual(fallback.source, "tracker_csrt")
            self.assertEqual(fallback.confidence, 0.82)
            self.assertEqual(fallback.tracker_age_frames, 1)
            self.assertEqual(fallback.xyxy, (6.0, 7.0, 16.0, 15.0))

            self.assertIsNone(failsafe.update(frame))
            self.assertFalse(failsafe.is_active)
        finally:
            tracker_failsafe.create_opencv_tracker = original_create_tracker

    def test_tracker_rejects_small_seed_box(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)

        failsafe = DetectionFailsafeTracker(
            enabled=True,
            tracker_type="CSRT",
            max_fallback_frames=3,
            min_bbox_area_px=25.0,
            max_center_jump_px=50.0,
            reinitialize_on_detection=True,
        )

        initialized = failsafe.initialize(
            frame,
            DetectionCandidate(
                xyxy=(5.0, 5.0, 7.0, 7.0),
                confidence=0.9,
                source="yolo",
            ),
        )

        self.assertFalse(initialized)
        self.assertFalse(failsafe.is_active)

    def test_unavailable_tracker_does_not_crash(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)

        def fake_create_tracker(_tracker_type):
            raise RuntimeError("tracker unavailable")

        original_create_tracker = tracker_failsafe.create_opencv_tracker
        tracker_failsafe.create_opencv_tracker = fake_create_tracker
        try:
            failsafe = DetectionFailsafeTracker(
                enabled=True,
                tracker_type="CSRT",
                max_fallback_frames=3,
                min_bbox_area_px=4.0,
                max_center_jump_px=50.0,
                reinitialize_on_detection=True,
            )

            with redirect_stdout(StringIO()):
                initialized = failsafe.initialize(
                    frame,
                    DetectionCandidate(
                        xyxy=(5.0, 6.0, 15.0, 14.0),
                        confidence=0.82,
                        source="yolo",
                    ),
                )

            self.assertFalse(initialized)
            self.assertFalse(failsafe.is_active)
            self.assertEqual(failsafe.unavailable_reason, "tracker unavailable")
            self.assertIsNone(failsafe.update(frame))
        finally:
            tracker_failsafe.create_opencv_tracker = original_create_tracker

    def test_tracker_rejects_update_outside_previous_bbox_margin(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)

        def fake_create_tracker(_tracker_type):
            return FakeTracker(updates=[(True, (220, 120, 20, 20))])

        original_create_tracker = tracker_failsafe.create_opencv_tracker
        tracker_failsafe.create_opencv_tracker = fake_create_tracker
        try:
            failsafe = DetectionFailsafeTracker(
                enabled=True,
                tracker_type="CSRT",
                max_fallback_frames=3,
                min_bbox_area_px=4.0,
                max_center_jump_px=50.0,
                reinitialize_on_detection=True,
            )

            self.assertTrue(
                failsafe.initialize(
                    frame,
                    DetectionCandidate(
                        xyxy=(40.0, 40.0, 60.0, 60.0),
                        confidence=0.9,
                        source="yolo",
                    ),
                )
            )

            self.assertIsNone(failsafe.update(frame))
            self.assertFalse(failsafe.is_active)
            self.assertEqual(failsafe.last_rejection_reason, "tracker_jump>50.0px")
        finally:
            tracker_failsafe.create_opencv_tracker = original_create_tracker


if __name__ == "__main__":
    unittest.main()
