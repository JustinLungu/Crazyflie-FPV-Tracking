"""Naive bbox depth pipeline package."""

from typing import Any

__all__ = ["NaiveBBoxDepthPipeline"]


def __getattr__(name: str) -> Any:
    if name == "NaiveBBoxDepthPipeline":
        from .pipeline import NaiveBBoxDepthPipeline

        return NaiveBBoxDepthPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
