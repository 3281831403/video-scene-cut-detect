"""Small, dependency-free scene-cut detection package."""

from .cuts import (
    SceneError,
    detect_batch,
    detect_cuts,
    edge_change,
    fade_detect,
    hist_diff,
    histogram,
    merge_short,
    validate_shots,
)

__all__ = [
    "SceneError",
    "detect_batch",
    "detect_cuts",
    "edge_change",
    "fade_detect",
    "hist_diff",
    "histogram",
    "merge_short",
    "validate_shots",
]
