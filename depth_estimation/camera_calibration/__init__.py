"""Camera calibration pipeline package."""

from typing import Any

__all__ = ["CameraCalibrationPipeline"]


def __getattr__(name: str) -> Any:
    if name == "CameraCalibrationPipeline":
        from .pipeline import CameraCalibrationPipeline

        return CameraCalibrationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
