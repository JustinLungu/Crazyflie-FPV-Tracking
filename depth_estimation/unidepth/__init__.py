"""UniDepth pipeline package."""

from typing import Any

__all__ = ["UniDepthPipeline"]


def __getattr__(name: str) -> Any:
    if name == "UniDepthPipeline":
        from .pipeline import UniDepthPipeline

        return UniDepthPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
