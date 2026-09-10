import pytest
import json
import socket
from http.server import ThreadingHTTPServer
from threading import Thread

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import CorrectionResult, RecognitionResult
from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.correction.contextual import ContextualCorrector
from assistive_writing_pad.display.web_app import (
    CAPTURE_HTML,
    HTML,
    RecognitionService,
    _huion_to_normalized,
    make_handler,
    stroke_groups_from_payload,
)


class StubStrokeGroupRecognizer:
    def recognize_stroke_groups(self, stroke_groups, mode="auto") -> RecognitionResult:
        assert len(stroke_groups) == 1
        return RecognitionResult(
            text="teh cat sat on a chaier",
            confidence=0.92,
            metadata={"recognizer": "stub", "mode": mode, "top3": "[]"},
        )


class WarmableCorrector:
    def __init__(self) -> None:
        self.warm_up_count = 0

    def warm_up(self) -> None:
        self.warm_up_count += 1

    def correct(self, text: str) -> CorrectionResult:
        return CorrectionResult(original_text=text, corrected_text=text)


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


def test_recognition_service_returns_realtime_correction_metadata() -> None:
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=ContextualCorrector(model_enabled=False),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )

    result = service.recognize_payload(
        {"strokes": [[{"x": 1, "y": 2, "timestamp_ms": 0}]]}
    )

    assert result["recognized_text"] == "teh cat sat on a chaier"
    assert result["corrected_text"] == "the cat sat on a chair"
    assert result["text"] == "the cat sat on a chair"
    assert result["needs_review"] is False
    assert result["mode"] == "auto"
    assert result["corrections"][0]["original"] == "teh"
    assert result["correction_metadata"] == {}


def test_recognition_service_accepts_legacy_mode_values() -> None:
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=ContextualCorrector(model_enabled=False),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )

    result = service.recognize_payload(
        {"strokes": [[{"x": 1, "y": 2, "timestamp_ms": 0}]], "mode": "word"}
    )

    assert result["mode"] == "word"


def test_recognition_service_corrects_alternative_text() -> None:
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=ContextualCorrector(model_enabled=False),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )

    result = service.correct_payload({"text": "teh cat sat on a chaier"})

    assert result["recognized_text"] == "teh cat sat on a chaier"
    assert result["corrected_text"] == "the cat sat on a chair"
    assert result["corrections"][0]["original"] == "teh"
    assert result["mode"] == "text"


def test_recognition_service_rejects_non_string_correction_text() -> None:
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=ContextualCorrector(model_enabled=False),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )

    with pytest.raises(ValueError, match="text must be a string"):
        service.correct_payload({"text": ["teh"]})


def test_browser_ui_keeps_child_facing_controls_simple() -> None:
    assert 'id="recognize"' in HTML
    assert 'id="captureHuion"' in HTML
    assert 'id="tryNext"' in HTML
    assert 'id="clearScreen"' in HTML
    assert 'id="space"' not in HTML
    assert 'id="backspace"' not in HTML
    assert 'id="clearText"' not in HTML
    assert "scripts/setup_model_env.sh" not in HTML
    assert "Pointer diagnostics" not in HTML
    assert "exportStrokePayload" in HTML
    assert "Save Evaluation Case" not in HTML


def test_huion_capture_payload_uses_reader(monkeypatch) -> None:
    class StubHuionReader:
        def __init__(self, device_path):
            assert device_path == "/dev/input/event4"

        def capture_strokes(self, *, duration_seconds, idle_timeout_seconds):
            assert duration_seconds == 15.0
            assert idle_timeout_seconds == 2.0
            return [[StrokePoint(x=4, y=5, timestamp_ms=6, pressure=7)]]

    monkeypatch.setattr("assistive_writing_pad.display.web_app.HuionEventReader", StubHuionReader)
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=WarmableCorrector(),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )

    result = service.capture_huion_payload()

    assert result["source"] == "huion"
    assert result["strokes"][0][0]["x"] == 4
    assert result["strokes"][0][0]["pressure"] == 7


def test_huion_stream_sends_persistent_event_messages(monkeypatch) -> None:
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=WarmableCorrector(),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )
    service.start_huion_reader = lambda: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = Thread(target=server.handle_request, daemon=True)
    thread.start()
    client = socket.create_connection(("127.0.0.1", server.server_port))
    key = "dGVzdC1rZXk="
    client.sendall(
        (
            "GET /ws/input HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode()
    )
    response = client.recv(4096)
    assert b"101 Switching Protocols" in response
    client.sendall(_masked_ws_frame(json.dumps({
        "type": "stroke_start", "stroke_id": "browser-1"
    }).encode()))
    client.sendall(_masked_ws_frame(json.dumps({
        "type": "stroke_point", "stroke_id": "browser-1",
        "x": 10, "y": 20, "pressure": 1, "timestamp_ms": 1
    }).encode()))
    client.sendall(_masked_ws_frame(json.dumps({
        "type": "stroke_end", "stroke_id": "browser-1"
    }).encode()))
    import time
    time.sleep(0.05)
    client.close()
    server.server_close()
    thread.join(timeout=1)
    assert len(service.stroke_snapshot()) == 1
    assert len(service.stroke_snapshot()[0]) == 1


def _masked_ws_frame(payload: bytes) -> bytes:
    mask = b"\x01\x02\x03\x04"
    encoded = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return bytes([0x81, 0x80 | len(payload)]) + mask + encoded


def test_canonical_stroke_state_handles_huion_clear_and_recognition() -> None:
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=WarmableCorrector(),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )
    service.ingest_stroke_event({"type": "stroke_start", "stroke_id": "h1"}, "huion")
    service.ingest_stroke_event(
        {
            "type": "stroke_point",
            "stroke_id": "h1",
            "x": 10,
            "y": 20,
            "pressure": 80,
            "timestamp_ms": 1,
        },
        "huion",
    )
    service.ingest_stroke_event({"type": "stroke_end", "stroke_id": "h1"}, "huion")

    result = service.recognize_payload({})
    assert result["recognized_text"] == "teh cat sat on a chaier"
    assert len(service.stroke_snapshot()) == 1

    service.clear_strokes()
    assert service.stroke_snapshot() == []
    with pytest.raises(ValueError, match="Write on the pad first"):
        service.recognize_payload({})


def test_huion_events_enter_canonical_canvas_coordinates() -> None:
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=WarmableCorrector(),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )
    service.ingest_stroke_event(
        {"type": "stroke_start", "stroke_id": "h1", "x": 0, "y": 0},
        "huion",
    )
    service.ingest_stroke_event(
        {
            "type": "stroke_point",
            "stroke_id": "h1",
            "x": 32000,
            "y": 20400,
            "pressure": 0,
            "timestamp_ms": 1,
        },
        "huion",
    )
    service.ingest_stroke_event({"type": "stroke_end", "stroke_id": "h1"}, "huion")

    stroke = service.stroke_snapshot()[0]
    assert (stroke[0].x, stroke[0].y) == _huion_to_normalized(0, 0)
    assert (stroke[1].x, stroke[1].y) == _huion_to_normalized(32000, 20400)
    assert 0 <= stroke[1].x <= 1
    assert 0 <= stroke[1].y <= 1


def test_huion_mapping_uses_full_independent_axis_ranges() -> None:
    assert _huion_to_normalized(0, 0) == (0.0, 0.0)
    assert _huion_to_normalized(32000, 0) == (1.0, 0.0)
    assert _huion_to_normalized(0, 20400) == (0.0, 1.0)
    assert _huion_to_normalized(32000, 20400) == (1.0, 1.0)
    assert _huion_to_normalized(16000, 10200) == (0.5, 0.5)


def test_huion_mapping_supports_device_reported_axis_ranges() -> None:
    ranges = ((100.0, 32100.0), (200.0, 20600.0))

    assert _huion_to_normalized(100, 200, axis_ranges=ranges) == (0.0, 0.0)
    assert _huion_to_normalized(32100, 20600, axis_ranges=ranges) == (1.0, 1.0)


def test_huion_snapshot_declares_normalized_coordinate_space() -> None:
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=WarmableCorrector(),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )
    service.ingest_stroke_event(
        {"type": "stroke_start", "stroke_id": "h1", "x": 0, "y": 0},
        "huion",
    )
    service.ingest_stroke_event(
        {"type": "stroke_point", "stroke_id": "h1", "x": 32000, "y": 20400},
        "huion",
    )
    service.ingest_stroke_event({"type": "stroke_end", "stroke_id": "h1"}, "huion")

    snapshot = service.websocket_snapshot()
    assert snapshot["coordinate_space"] == "normalized"
    assert snapshot["strokes"][0][0]["x"] == 0.0
    assert snapshot["strokes"][0][1]["x"] == 1.0
    assert snapshot["strokes"][0][1]["y"] == 1.0


def test_huion_clear_removes_active_and_completed_strokes() -> None:
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=WarmableCorrector(),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )
    service.ingest_stroke_event({"type": "stroke_start", "stroke_id": "h1"}, "huion")
    service.ingest_stroke_event({"type": "clear"}, "huion")

    assert service.stroke_snapshot() == []


def test_websocket_reconnect_does_not_start_duplicate_huion_readers(monkeypatch) -> None:
    starts = []

    class StubReader:
        def __init__(self, device_path):
            starts.append(device_path)

        def iter_stroke_events(self):
            return iter(())

    monkeypatch.setattr(
        "assistive_writing_pad.display.web_app.HuionEventReader",
        StubReader,
    )
    monkeypatch.setattr(
        "assistive_writing_pad.display.web_app.find_huion_device",
        lambda: "/dev/input/event4",
    )
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=WarmableCorrector(),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )

    service.start_huion_reader()
    service.start_huion_reader()
    service._huion_thread.join(timeout=1)

    assert starts == ["/dev/input/event4"]


def test_evaluation_capture_page_has_labeling_controls() -> None:
    assert 'id="caseId"' in CAPTURE_HTML
    assert 'id="category"' in CAPTURE_HTML
    assert 'id="expected"' in CAPTURE_HTML
    assert 'id="save"' in CAPTURE_HTML
    assert "assistiveWritingPadCapture" in CAPTURE_HTML


def test_evaluation_capture_page_subscribes_to_huion_stream() -> None:
    assert 'new WebSocket(protocol + "//" + window.location.host + "/ws/input")' in CAPTURE_HTML
    assert "coordinate_space !== \"normalized\"" in CAPTURE_HTML
    assert "[capture] Huion WebSocket connected" in CAPTURE_HTML
    assert "[capture] Huion stroke start" in CAPTURE_HTML
    assert "[capture] Huion stroke end" in CAPTURE_HTML
    assert "normalizedCanvasPoint(event.x, event.y, event)" in CAPTURE_HTML


def test_browser_huion_mapping_uses_full_tablet_area() -> None:
    assert 'function huionCanvasPoint(event)' in HTML
    assert "canvasPointFromNormalized(nx, ny" in HTML
    assert "coordinate_space === \"normalized\"" in HTML
    assert "Number(event.x) / 32000" not in HTML
    assert "Number(event.y) / 20400" not in HTML
    assert "Math.min(rect.width / tabletWidth, rect.height / tabletHeight)" not in HTML
    assert "const offsetX = (canvas.clientWidth - width * scale) / 2" not in HTML
    assert "const offsetY = (canvas.clientHeight - height * scale) / 2" not in HTML
    assert 'new WebSocket(protocol + "//" + window.location.host + "/ws/input")' in HTML


def test_evaluation_capture_is_disabled_by_default() -> None:
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=ContextualCorrector(model_enabled=False),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )

    with pytest.raises(PermissionError, match="disabled"):
        service.append_evaluation_case_payload(
            {
                "id": "word_001",
                "category": "word",
                "expected": "the",
                "strokes": [[{"x": 1, "y": 2, "timestamp_ms": 0}]],
            }
        )


def test_evaluation_capture_appends_manual_case_when_enabled(tmp_path) -> None:
    manifest = tmp_path / "cases.jsonl"
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=ContextualCorrector(model_enabled=False),
        settings=RuntimeSettings(
            contextual_model_enabled=False,
            evaluation_capture_enabled=True,
            evaluation_manifest_path=manifest,
        ),
    )

    result = service.append_evaluation_case_payload(
        {
            "id": "word_001",
            "category": "word",
            "expected": "the",
            "expected_recognized": "",
            "notes": "manual browser capture",
            "source": "ignored-client-value",
            "strokes": [[{"x": 1, "y": 2, "timestamp_ms": 0}]],
        }
    )

    assert result["id"] == "word_001"
    assert result["source"] == "manual"
    content = manifest.read_text(encoding="utf-8")
    assert '"source":"manual"' in content
    assert "ignored-client-value" not in content


def test_recognition_service_warms_correction_models() -> None:
    corrector = WarmableCorrector()
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=corrector,
        settings=RuntimeSettings(
            preload_ocr_model=False,
            preload_correction_models=True,
        ),
    )

    service._warm_up_correction_models()

    assert corrector.warm_up_count == 1
