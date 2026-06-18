"""Deterministic stroke simulator for development without tablet hardware."""

from dataclasses import dataclass
from typing import List

from assistive_writing_pad.contracts import StrokePoint


@dataclass(frozen=True)
class SimulatedStrokeConfig:
    start_x: float = 20.0
    baseline_y: float = 80.0
    character_width: float = 18.0
    point_spacing_ms: int = 16
    pressure: float = 0.65


class StrokeSimulator:
    """Create simple stroke paths that can exercise the capture pipeline."""

    def __init__(self, config: SimulatedStrokeConfig = SimulatedStrokeConfig()) -> None:
        self.config = config

    def write_text(self, text: str) -> List[StrokePoint]:
        points: List[StrokePoint] = []
        timestamp_ms = 0
        x = self.config.start_x

        for character in text:
            if character == " ":
                x += self.config.character_width
                timestamp_ms += self.config.point_spacing_ms
                continue

            points.extend(
                [
                    StrokePoint(
                        x=x,
                        y=self.config.baseline_y,
                        timestamp_ms=timestamp_ms,
                        pressure=self.config.pressure,
                    ),
                    StrokePoint(
                        x=x + self.config.character_width * 0.4,
                        y=self.config.baseline_y - 12.0,
                        timestamp_ms=timestamp_ms + self.config.point_spacing_ms,
                        pressure=self.config.pressure,
                    ),
                    StrokePoint(
                        x=x + self.config.character_width * 0.8,
                        y=self.config.baseline_y,
                        timestamp_ms=timestamp_ms + self.config.point_spacing_ms * 2,
                        pressure=self.config.pressure,
                    ),
                ]
            )
            x += self.config.character_width
            timestamp_ms += self.config.point_spacing_ms * 3

        return points
