"""Browser-based handwriting pad with pretrained OCR endpoint."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.recognition.trocr import RecognitionUnavailable, TrOCRHandwritingRecognizer


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Assistive Writing Pad</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #667085;
      --line: #d0d5dd;
      --accent: #2563eb;
      --accent-dark: #1d4ed8;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      height: 68px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 720;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(340px, 0.8fr);
      gap: 18px;
      padding: 18px;
      min-height: calc(100vh - 68px);
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }
    .section-title {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    h2 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .hint, #status, #confidence {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
    }
    #pad {
      display: block;
      width: 100%;
      height: 460px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      touch-action: none;
      cursor: crosshair;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      min-height: 38px;
      padding: 8px 12px;
      font-size: 14px;
      cursor: pointer;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    button.primary:hover { background: var(--accent-dark); }
    #recognized {
      width: 100%;
      min-height: 280px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      font-size: 24px;
      line-height: 1.35;
      color: var(--ink);
      background: #fbfcfd;
    }
    .setup {
      margin-top: 14px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }
    code {
      background: #eef2f7;
      border-radius: 4px;
      padding: 2px 5px;
      color: #344054;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      #pad { height: 360px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Assistive Writing Pad</h1>
    <div id="status">Ready</div>
  </header>
  <main>
    <section>
      <div class="section-title">
        <h2>Handwriting</h2>
        <div class="hint">Write a word or short line. Recognition runs after you pause.</div>
      </div>
      <canvas id="pad"></canvas>
      <div class="toolbar">
        <button class="primary" id="recognize">Recognize Now</button>
        <button id="clearInk">Clear Ink</button>
        <button id="space">Space</button>
        <button id="backspace">Backspace</button>
        <button id="clearText">Clear Text</button>
      </div>
    </section>
    <section>
      <div class="section-title">
        <h2>Recognized Text</h2>
        <div id="confidence">Confidence: -</div>
      </div>
      <textarea id="recognized" spellcheck="false"></textarea>
      <div class="setup">
        Pretrained OCR uses <code>microsoft/trocr-base-handwritten</code> with line-aware
        segmentation. If recognition reports missing dependencies, create a Python 3.10
        environment and run <code>scripts/setup_model_env.sh</code>.
      </div>
    </section>
  </main>
  <script>
    const canvas = document.getElementById("pad");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const confidenceEl = document.getElementById("confidence");
    const recognizedEl = document.getElementById("recognized");
    let strokes = [];
    let currentStroke = [];
    let drawing = false;
    let last = null;
    let startedAt = 0;
    let recognizeTimer = null;

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const previous = ctx.getImageData(0, 0, canvas.width || 1, canvas.height || 1);
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, rect.width, rect.height);
      try { ctx.putImageData(previous, 0, 0); } catch (_err) {}
      ctx.lineWidth = 4;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#111827";
    }

    function canvasPoint(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
        timestamp_ms: Math.round(performance.now() - startedAt),
        pressure: event.pressure && event.pressure > 0 ? event.pressure : 1.0
      };
    }

    function start(event) {
      drawing = true;
      startedAt = performance.now();
      currentStroke = [];
      last = canvasPoint(event);
      currentStroke.push(last);
      canvas.setPointerCapture(event.pointerId);
      event.preventDefault();
    }

    function move(event) {
      if (!drawing) return;
      const point = canvasPoint(event);
      ctx.beginPath();
      ctx.moveTo(last.x, last.y);
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
      currentStroke.push(point);
      last = point;
      event.preventDefault();
    }

    function finish(event) {
      if (!drawing) return;
      drawing = false;
      currentStroke.push(canvasPoint(event));
      strokes.push(currentStroke);
      currentStroke = [];
      last = null;
      scheduleRecognition();
      event.preventDefault();
    }

    function scheduleRecognition() {
      if (recognizeTimer) clearTimeout(recognizeTimer);
      recognizeTimer = setTimeout(recognize, 800);
    }

    async function recognize() {
      if (!strokes.length) {
        statusEl.textContent = "Write on the pad first.";
        return;
      }
      statusEl.textContent = "Recognizing...";
      confidenceEl.textContent = "Confidence: -";
      try {
        const response = await fetch("/api/recognize", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({strokes})
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Recognition failed");
        recognizedEl.value = result.text || "";
        confidenceEl.textContent = `Confidence: ${Number(result.confidence || 0).toFixed(2)}`;
        statusEl.textContent = result.status || "Recognized";
      } catch (error) {
        statusEl.textContent = error.message;
      }
    }

    function clearInk() {
      strokes = [];
      currentStroke = [];
      ctx.fillStyle = "#ffffff";
      const rect = canvas.getBoundingClientRect();
      ctx.fillRect(0, 0, rect.width, rect.height);
      confidenceEl.textContent = "Confidence: -";
      statusEl.textContent = "Ink cleared";
    }

    window.addEventListener("resize", resizeCanvas);
    canvas.addEventListener("pointerdown", start);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", finish);
    canvas.addEventListener("pointercancel", finish);
    document.getElementById("recognize").addEventListener("click", recognize);
    document.getElementById("clearInk").addEventListener("click", clearInk);
    document.getElementById("space").addEventListener("click", () => recognizedEl.value += " ");
    document.getElementById("backspace").addEventListener("click", () => {
      recognizedEl.value = recognizedEl.value.slice(0, -1);
    });
    document.getElementById("clearText").addEventListener("click", () => {
      recognizedEl.value = "";
    });
    resizeCanvas();
  </script>
</body>
</html>
"""


class RecognitionService:
    def __init__(self) -> None:
        self.recognizer = TrOCRHandwritingRecognizer()

    def recognize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        stroke_groups = stroke_groups_from_payload(payload)
        result = self.recognizer.recognize_stroke_groups(stroke_groups)
        return {
            "text": result.text,
            "confidence": result.confidence,
            "status": "Recognized handwriting with pretrained OCR.",
            "metadata": result.metadata,
        }


def stroke_groups_from_payload(payload: Dict[str, Any]) -> List[List[StrokePoint]]:
    raw_strokes = payload.get("strokes")
    if raw_strokes is None:
        raw_points = payload.get("points")
        if isinstance(raw_points, list):
            raw_strokes = [raw_points]
        else:
            raise ValueError("request must contain a strokes list")

    if not isinstance(raw_strokes, list):
        raise ValueError("request must contain a strokes list")

    stroke_groups: List[List[StrokePoint]] = []
    for stroke in raw_strokes:
        if not isinstance(stroke, list):
            raise ValueError("each stroke must be a list of points")
        points: List[StrokePoint] = []
        for item in stroke:
            if not isinstance(item, dict):
                raise ValueError("each point must be an object")
            points.append(
                StrokePoint(
                    x=float(item["x"]),
                    y=float(item["y"]),
                    timestamp_ms=int(item.get("timestamp_ms", 0)),
                    pressure=float(item.get("pressure", 1.0)),
                )
            )
        stroke_groups.append(points)
    return stroke_groups


def make_handler(service: RecognitionService):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send(HTTPStatus.OK, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/api/recognize":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = service.recognize_payload(payload)
            except RecognitionUnavailable as exc:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    service = RecognitionService()
    server = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"Assistive Writing Pad running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
