import pytest

from assistive_writing_pad.display.web_app import stroke_groups_from_payload


def test_stroke_groups_from_payload_parses_strokes() -> None:
    points = stroke_groups_from_payload(
        {
            "strokes": [
                [
                    {"x": 1, "y": 2, "timestamp_ms": 0, "pressure": 0.5},
                    {"x": 3.2, "y": 4.5, "timestamp_ms": 16},
                ],
                [{"x": 10, "y": 20, "timestamp_ms": 0}],
            ]
        }
    )

    assert len(points) == 2
    assert len(points[0]) == 2
    assert points[0][0].x == 1.0
    assert points[0][0].pressure == 0.5
    assert points[0][1].pressure == 1.0
    assert points[1][0].x == 10.0


def test_stroke_groups_from_payload_rejects_missing_list() -> None:
    with pytest.raises(ValueError, match="strokes list"):
        stroke_groups_from_payload({})
