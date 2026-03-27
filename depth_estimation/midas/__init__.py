"""MiDaS pipeline package."""

from typing import Any

__all__ = ["MiDaSPipeline"]


def __getattr__(name: str) -> Any:
    if name == "MiDaSPipeline":
        from .pipeline import MiDaSPipeline

        return MiDaSPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
