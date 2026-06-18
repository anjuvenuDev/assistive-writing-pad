import pytest

from assistive_writing_pad.display.web_app import points_from_payload


def test_points_from_payload_parses_strokes() -> None:
    points = points_from_payload(
        {
            "points": [
                {"x": 1, "y": 2, "timestamp_ms": 0, "pressure": 0.5},
                {"x": 3.2, "y": 4.5, "timestamp_ms": 16},
            ]
        }
    )

    assert len(points) == 2
    assert points[0].x == 1.0
    assert points[0].pressure == 0.5
    assert points[1].pressure == 1.0


def test_points_from_payload_rejects_missing_list() -> None:
    with pytest.raises(ValueError, match="points list"):
        points_from_payload({})
