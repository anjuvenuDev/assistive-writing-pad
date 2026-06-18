"""Tablet capture components."""

from assistive_writing_pad.capture.simulator import SimulatedStrokeConfig, StrokeSimulator
from assistive_writing_pad.capture.stroke_io import load_strokes, save_strokes

__all__ = [
    "SimulatedStrokeConfig",
    "StrokeSimulator",
    "load_strokes",
    "save_strokes",
]
