from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LiveFrameOutput:
    method: str
    frame_bgr: np.ndarray
    metrics: dict[str, Any] = field(default_factory=dict)


class DepthPipeline(ABC):
    name: str = "pipeline"

    def close(self) -> None:
        """Optional resource cleanup hook."""
        return


class LiveDepthPipeline(DepthPipeline, ABC):
    @abstractmethod
    def process_live_frame(self, frame_bgr: np.ndarray) -> LiveFrameOutput:
        """Process one BGR frame and return a visualization + metrics."""
        raise NotImplementedError
