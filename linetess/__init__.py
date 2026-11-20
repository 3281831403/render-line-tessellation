"""Screen-space line tessellation package."""

from .tessellator import (
    EPS,
    LineError,
    bevel_join,
    dash_pattern,
    expand_segment,
    miter_join,
    snap_to_pixel,
    tessellate_batch,
    tessellate_bezier,
    validate_mesh,
)

__all__ = [
    "EPS",
    "LineError",
    "bevel_join",
    "dash_pattern",
    "expand_segment",
    "miter_join",
    "snap_to_pixel",
    "tessellate_batch",
    "tessellate_bezier",
    "validate_mesh",
]
