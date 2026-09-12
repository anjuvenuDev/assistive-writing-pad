"""Browser-based handwriting pad with pretrained OCR endpoint."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import socket
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from assistive_writing_pad.capture.huion_reader import HuionEventReader
from assistive_writing_pad.capture.huion_probe import find_huion_device, huion_axis_ranges
from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import CorrectionResult, PipelineResult, StrokePoint
from assistive_writing_pad.correction.factory import corrector_from_settings
from assistive_writing_pad.eval.corpus import (
    append_jsonl_record,
    build_end_to_end_case_record,
)
from assistive_writing_pad.pipeline import WritingPipeline
from assistive_writing_pad.recognition.trocr import RecognitionUnavailable, TrOCRHandwritingRecognizer

logger = logging.getLogger(__name__)


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
    #status {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
    }
    /* Confidence badge colours */
    #confidence {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 20px;
      transition: background 0.25s, color 0.25s;
      background: #f0f0f0;
      color: #555;
    }
    #confidence.conf-high  { background: #dcfce7; color: #15803d; }
    #confidence.conf-med   { background: #fef9c3; color: #92400e; }
    #confidence.conf-low   { background: #fee2e2; color: #b91c1c; }
    /* Collapsible debug panel (OCR raw output) */
    details.debug-panel {
      margin-top: 12px;
      font-size: 13px;
      color: var(--muted);
    }
    details.debug-panel summary {
      cursor: pointer;
      user-select: none;
      font-weight: 600;
    }
    #raw-text {
      margin-top: 6px;
      padding: 8px 10px;
      background: #f8f9fb;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-family: monospace;
      white-space: pre-wrap;
      word-break: break-all;
    }
    /* Canvas
     *
     * touch-action: none  -- hand full pointer control to JS; without this
     *                        the browser steals touchmove for scroll/zoom and
     *                        fires pointercancel, aborting the stroke.
     * pointer-events: auto -- must NOT be 'none'; that disables all input.
     * position: relative + z-index: 1 -- ensure the canvas is not buried
     *                        under any sibling overlay element.
     */
    #pad {
      display: block;
      width: 100%;
      height: 460px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      touch-action: none;
      pointer-events: auto;
      cursor: crosshair;
      position: relative;
      z-index: 1;
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
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
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
    /* Pointer input diagnostics panel */
    #pointer-debug {
      margin-top: 10px;
      padding: 7px 10px;
      background: #f1f5f9;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-family: monospace;
      font-size: 12px;
      color: #334155;
      line-height: 1.6;
    }
    #pointer-debug .pdl { display: flex; gap: 12px; flex-wrap: wrap; }
    #pointer-debug .pdl span { white-space: nowrap; }
    #pointer-debug .drawing-yes { color: #15803d; font-weight: 700; }
    #pointer-debug .drawing-no  { color: #64748b; }
    #pointer-log {
      margin-top: 8px;
      padding: 6px 8px;
      background: #0f172a;
      color: #38bdf8;
      border: 1px solid var(--line);
      border-radius: 4px;
      height: 85px;
      overflow-y: auto;
      font-family: monospace;
      font-size: 11px;
      white-space: pre-wrap;
      text-align: left;
    }
    /* Top-3 predictions panel */
    #correction-panel {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      overflow: hidden;
    }
    .correction-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    #correction-confidence {
      text-transform: none;
      letter-spacing: 0;
      font-weight: 600;
    }
    #correction-list {
      display: flex;
      flex-direction: column;
      gap: 0;
    }
    .correction-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-top: 1px solid #eef2f7;
      font-size: 14px;
    }
    .correction-row:first-child { border-top: 0; }
    .correction-word {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .correction-before {
      color: #b42318;
      text-decoration: line-through;
    }
    .correction-after {
      color: #067647;
      font-weight: 700;
    }
    .correction-arrow { color: var(--muted); }
    .correction-empty {
      padding: 10px;
      color: var(--muted);
      font-size: 14px;
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
        <h2>Write</h2>
      </div>
      <canvas id="pad"></canvas>
      <div class="toolbar">
        <button class="primary" id="recognize">Read Again</button>
        <button id="captureHuion">Connect Huion</button>
        <button id="clearScreen">Clear Screen</button>
      </div>
    </section>
    <section>
      <div class="section-title">
        <h2>Result</h2>
        <span id="confidence">&mdash;</span>
      </div>
      <textarea id="recognized" spellcheck="false"></textarea>
      <div id="correction-panel">
        <div class="correction-header">
          <span>Corrections</span>
          <span id="correction-confidence">&mdash;</span>
        </div>
        <div id="correction-list">
          <div class="correction-empty">No corrections yet.</div>
        </div>
      </div>
      <details class="debug-panel">
        <summary>Technical Details</summary>
        <div id="raw-text">No recognition yet.</div>
        <div id="pointer-debug">
          <div class="pdl">
            <span>Type: <b id="pd-type">&mdash;</b></span>
            <span>X: <b id="pd-x">&mdash;</b></span>
            <span>Y: <b id="pd-y">&mdash;</b></span>
            <span>Drawing: <b id="pd-drawing" class="drawing-no">no</b></span>
            <span>Strokes: <b id="pd-strokes">0</b></span>
          </div>
          <div id="pointer-log">No pointer events yet.</div>
        </div>
      </details>
    </section>
  </main>
  <script>
    /* -----------------------------------------------------------------------
     * Element references
     * --------------------------------------------------------------------- */
    const canvas       = document.getElementById("pad");
    const ctx          = canvas.getContext("2d");
    const statusEl     = document.getElementById("status");
    const confidenceEl = document.getElementById("confidence");
    const recognizedEl = document.getElementById("recognized");
    const rawTextEl    = document.getElementById("raw-text");
    const captureHuionEl = document.getElementById("captureHuion");
    const correctionConfidenceEl = document.getElementById("correction-confidence");
    const correctionListEl = document.getElementById("correction-list");
    const pdType       = document.getElementById("pd-type");
    const pdX          = document.getElementById("pd-x");
    const pdY          = document.getElementById("pd-y");
    const pdDrawing    = document.getElementById("pd-drawing");
    const pdStrokes    = document.getElementById("pd-strokes");

    /* -----------------------------------------------------------------------
     * State
     * --------------------------------------------------------------------- */
    let strokes        = [];    // completed strokes sent to OCR
    let currentStroke  = [];    // points in the stroke currently being drawn
    let currentStrokeId = null;
    let drawing        = false;
    let last           = null;  // last canvas-space point {x, y, ...}
    let startedAt      = 0;     // performance.now() at stroke start
    let recognizeTimer = null;
    let currentMode = "auto";
    let strokeRevision = 0;
    let recognitionInFlight = false;
    let recognitionQueued = false;
    let inputSocket = null;
    let inputSocketRetry = null;
    let huionDiagnosticPointCount = 0;
    let huionCanvasDiagnosticShown = false;

    /* -----------------------------------------------------------------------
     * Diagnostics logger
     * --------------------------------------------------------------------- */
    const logEl = document.getElementById("pointer-log");
    function logMsg(msg) {
      const time = new Date().toTimeString().split(' ')[0];
      logEl.innerHTML = "[" + time + "] " + msg + "<br>" + logEl.innerHTML;
      const lines = logEl.innerHTML.split("<br>");
      if (lines.length > 15) {
        logEl.innerHTML = lines.slice(0, 15).join("<br>");
      }
    }
    window.addEventListener("error", (e) => {
      logMsg("JS ERR: " + e.message + " at " + e.filename + ":" + e.lineno);
    });

    /* -----------------------------------------------------------------------
     * Canvas initialisation and resize
     *
     * ROOT CAUSE OF THE REGRESSION — resizeCanvas() bug:
     *
     *   The previous code called:
     *     ctx.setTransform(ratio, 0, 0, ratio, 0, 0);   // set DPR scale
     *     ctx.putImageData(previous, 0, 0);              // restore pixels
     *
     *   putImageData() ignores the current transform and writes pixels at
     *   raw device-pixel offsets.  After the setTransform call, (0,0) in
     *   device pixels is the top-left corner of the backing buffer — correct
     *   only when ratio=1.  On HiDPI screens (MacBook trackpads, Surface,
     *   high-DPI laptop displays) where ratio=2, the restored pixels land in
     *   the wrong position, making the canvas appear blank (all white).
     *   A blank canvas visually looks identical to "input not working" which
     *   is why drawing appeared broken on trackpad-equipped laptops.
     *
     * FIX:
     *   1. Reset transform to identity BEFORE reading/restoring pixel data.
     *   2. Flood-fill white at device-pixel scale (no transform).
     *   3. Restore pixel data at identity (correct position on all DPRs).
     *   4. Re-apply DPR transform for subsequent draw calls.
     *   5. Re-apply drawing style (lineWidth etc.) — canvas.width = N resets
     *      the entire 2D context state including these properties.
     * --------------------------------------------------------------------- */
    function applyDrawingStyle() {
      ctx.lineWidth   = 4;
      ctx.lineCap     = "round";
      ctx.lineJoin    = "round";
      ctx.strokeStyle = "#111827";
      ctx.fillStyle   = "#111827";  // for arc-based fill draw in move()
    }

    function resizeCanvas() {
      const rect  = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;

      // Step 1: reset to identity so getImageData captures device pixels.
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      const prevW = canvas.width  || 1;
      const prevH = canvas.height || 1;
      let previous = null;
      try { previous = ctx.getImageData(0, 0, prevW, prevH); } catch (_) {}

      // Step 2: resize backing buffer to physical pixel dimensions.
      canvas.width  = Math.max(1, Math.floor(rect.width  * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));

      // Step 3: clear at identity (device-pixel space).
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Step 4: restore previous content at identity (correct on all DPRs).
      if (previous) {
        try { ctx.putImageData(previous, 0, 0); } catch (_) {}
      }

      // Step 5: re-apply DPR scale for subsequent draw calls (CSS px space).
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

      // Step 6: re-apply drawing style (reset by canvas.width assignment).
      applyDrawingStyle();
    }

    /* -----------------------------------------------------------------------
     * Coordinate helper
     * --------------------------------------------------------------------- */

    /**
     * Convert a PointerEvent into a canvas-space point.
     * getBoundingClientRect() is called fresh every time so coordinates
     * stay correct even after layout shifts (address bar hiding on mobile,
     * window resize between events, etc.).
     */
    function canvasPoint(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
        timestamp_ms: Math.round(performance.now() - startedAt),
        pressure: (event.pressure != null && event.pressure > 0) ? event.pressure : 1.0
      };
    }

    function canvasPointFromNormalized(nx, ny, metadata = {}) {
      const rect = canvas.getBoundingClientRect();
      const point = {
        x: Math.max(0, Math.min(1, Number(nx))) * rect.width,
        y: Math.max(0, Math.min(1, Number(ny))) * rect.height,
        timestamp_ms: Number(metadata.timestamp_ms || 0),
        pressure: Number(metadata.pressure || 1)
      };
      if (!huionCanvasDiagnosticShown) {
        huionCanvasDiagnosticShown = true;
        console.info(
          "[AWP] Canvas: width=%s height=%s css_width=%s css_height=%s dpr=%s",
          canvas.width,
          canvas.height,
          rect.width.toFixed(1),
          rect.height.toFixed(1),
          window.devicePixelRatio || 1
        );
      }
      return point;
    }

    function huionCanvasPoint(event) {
      if (event.coordinate_space === "normalized") {
        const nx = Math.max(0, Math.min(1, Number(event.x)));
        const ny = Math.max(0, Math.min(1, Number(event.y)));
        const point = canvasPointFromNormalized(nx, ny, {
          timestamp_ms: Number(event.timestamp_ms || 0),
          pressure: Number(event.pressure || 1)
        });
        huionDiagnosticPointCount += 1;
        if (huionDiagnosticPointCount === 1 || huionDiagnosticPointCount % 100 === 0) {
          const rect = canvas.getBoundingClientRect();
          console.debug(
            "[AWP] Huion BROWSER normalized=(%s,%s) canvas=(%s,%s) " +
            "css=%sx%s bitmap=%sx%s dpr=%s",
            nx.toFixed(4), ny.toFixed(4),
            point.x.toFixed(1), point.y.toFixed(1),
            rect.width.toFixed(1), rect.height.toFixed(1),
            canvas.width, canvas.height,
            window.devicePixelRatio || 1
          );
        }
        return point;
      }
      console.warn("[AWP] Ignoring Huion point without normalized coordinate_space");
      return null;
    }

    function drawRemotePoint(point) {
      const radius = 3;
      if (last) {
        ctx.beginPath();
        ctx.moveTo(last.x, last.y);
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    function handleInputEvent(event) {
      if (event.source === "browser") return;
      if (event.type === "test_point") {
        drawRemotePoint({x: Number(event.x), y: Number(event.y)});
        logMsg("Huion test point received");
        return;
      }
      if (event.type === "snapshot") {
        strokes = (event.strokes || []).map(stroke => stroke.map(point => (
          event.coordinate_space === "normalized"
            ? canvasPointFromNormalized(
                Number(point.x),
                Number(point.y),
                point
              )
            : null
        )).filter(Boolean));
        currentStroke = [];
        drawing = false;
        last = null;
        drawCapturedStrokes();
        pdStrokes.textContent = strokes.length;
        logMsg("Huion: Connected / Stroke complete / Points: " + strokes.flat().length);
        return;
      }
      if (event.type === "clear") {
        strokes = [];
        currentStroke = [];
        drawing = false;
        last = null;
        clearCanvasPixels();
        pdStrokes.textContent = "0";
        return;
      }
      if (event.type === "stroke_start") {
        if (currentStroke.length) strokes.push(currentStroke);
        currentStroke = [];
        drawing = true;
        last = null;
        statusEl.textContent = event.source === "huion" ? "Huion pen down" : "Writing";
        if (event.source === "huion") logMsg("Huion: Connected / Stroke active");
        return;
      }
      if (event.type === "stroke_point") {
        const point = event.source === "huion"
          ? huionCanvasPoint(event)
          : {x: Number(event.x), y: Number(event.y),
             pressure: Number(event.pressure || 1),
             timestamp_ms: Number(event.timestamp_ms || 0)};
        if (!point) return;
        if (!drawing) {
          drawing = true;
          currentStroke = [];
        }
        drawRemotePoint(point);
        currentStroke.push(point);
        last = point;
        pdType.textContent = event.source || "browser";
        pdX.textContent = point.x.toFixed(1);
        pdY.textContent = point.y.toFixed(1);
        pdDrawing.textContent = "yes";
        pdDrawing.className = "drawing-yes";
        if (event.source === "huion") {
          statusEl.textContent = "WebSocket: Connected | Huion: Drawing | Points: " + currentStroke.length;
        }
        return;
      }
      if (event.type === "stroke_end") {
        if (currentStroke.length) strokes.push(currentStroke);
        currentStroke = [];
        drawing = false;
        last = null;
        pdDrawing.textContent = "no";
        pdDrawing.className = "drawing-no";
        pdStrokes.textContent = strokes.length;
      }
    }

    function sendInputEvent(event) {
      if (inputSocket && inputSocket.readyState === WebSocket.OPEN) {
        inputSocket.send(JSON.stringify(event));
      }
    }

    function connectInputSocket() {
      if (inputSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(inputSocket.readyState)) {
        return;
      }
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      inputSocket = new WebSocket(protocol + "//" + window.location.host + "/ws/input");
      inputSocket.onopen = () => {
        statusEl.textContent = "WebSocket: Connected | Huion: Waiting";
        logMsg("WebSocket connected");
        console.log("[AWP] Huion WebSocket connected");
      };
      inputSocket.onmessage = (message) => {
        try {
          const event = JSON.parse(message.data);
          console.log("[AWP] Huion WebSocket message:", event.type);
          handleInputEvent(event);
        } catch (err) {
          logMsg("WebSocket message error: " + err.message);
        }
      };
      inputSocket.onerror = () => {
        logMsg("WebSocket connection error");
      };
      inputSocket.onclose = () => {
        logMsg("WebSocket disconnected");
        if (inputSocketRetry) clearTimeout(inputSocketRetry);
        inputSocketRetry = setTimeout(connectInputSocket, 1000);
      };
    }

    /** Refresh the pointer diagnostics panel. */
    function updatePointerDebug(event, isDrawing) {
      const rect = canvas.getBoundingClientRect();
      pdType.textContent    = event.pointerType || "unknown";
      pdX.textContent       = (event.clientX - rect.left).toFixed(1);
      pdY.textContent       = (event.clientY - rect.top).toFixed(1);
      pdDrawing.textContent = isDrawing ? "yes" : "no";
      pdDrawing.className   = isDrawing ? "drawing-yes" : "drawing-no";
      pdStrokes.textContent = strokes.length;
    }

    /* -----------------------------------------------------------------------
     * Pointer event handlers
     *
     * We use the Pointer Events API (W3C Level 2) — the unified model for:
     *   - mouse / left-click drag
     *   - laptop trackpad click-drag
     *   - touchscreen single-finger draw
     *   - stylus / pen (Wacom, Surface Pen, Huion, Apple Pencil via WKWebView)
     *
     * Critical rules observed here:
     *   A. setPointerCapture(id)    in pointerdown  — keeps events arriving
     *      even when pointer leaves canvas bounds during a fast stroke.
     *   B. releasePointerCapture(id) in pointerup AND pointercancel — without
     *      this the capture persists into the next gesture, causing browsers
     *      to silently skip re-issuing gotpointercapture, which on mouse /
     *      trackpad breaks subsequent strokes.  THIS WAS THE PRIMARY BUG.
     *   C. event.preventDefault() in pointerdown + pointermove — suppresses
     *      the browser's touch-scroll and text-selection defaults.
     *      NOT called in pointerup — preventing default there blocks click
     *      events on some browsers (e.g. toolbar buttons right after a stroke).
     *   D. touch-action: none on the canvas (CSS above) — prevents the
     *      browser from starting its own scroll/pinch gesture on the canvas,
     *      which would fire pointercancel and abort the stroke.
     * --------------------------------------------------------------------- */

    function start(event) {
      logMsg("down: type=" + event.pointerType + " btn=" + event.button + " btns=" + event.buttons + " id=" + event.pointerId);
      // Ignore non-primary buttons (right-click, middle-click, eraser end).
      if (event.pointerType === "mouse" && event.button !== 0) {
        logMsg("ignored non-left click down");
        return;
      }

      console.log("[AWP] pointerdown  type:", event.pointerType,
                  " id:", event.pointerId,
                  " x:", event.clientX.toFixed(1),
                  " y:", event.clientY.toFixed(1));

      drawing = true;
      startedAt = performance.now();
      currentStroke = [];
      currentStrokeId = "browser-" + event.pointerId + "-" + Date.now();
      last = canvasPoint(event);
      currentStroke.push(last);
      sendInputEvent({
        type: "stroke_start",
        stroke_id: currentStrokeId,
        x: last.x, y: last.y, pressure: last.pressure, timestamp_ms: last.timestamp_ms
      });

      // Rule A: capture pointer.
      try {
        canvas.setPointerCapture(event.pointerId);
        logMsg("pointer captured: " + event.pointerId);
      } catch (err) {
        logMsg("capture failed: " + err.message);
      }

      updatePointerDebug(event, true);
      event.preventDefault();
    }

    function move(event) {
      // Always update diagnostics so "is the canvas receiving events?" is
      // immediately visible, even when not in a drawing stroke.
      updatePointerDebug(event, drawing);

      if (!drawing) return;

      logMsg("move: x=" + event.clientX.toFixed(0) + " y=" + event.clientY.toFixed(0) + " btns=" + event.buttons);

      console.log("[AWP] pointermove  type:", event.pointerType,
                  " x:", event.clientX.toFixed(1),
                  " y:", event.clientY.toFixed(1));

      const point = canvasPoint(event);

      // Draw a filled circle path between last and current point.
      // Using fillRect-based dots is more reliable across Windows browsers
      // than stroke() which can miss events if the OS throttles them.
      const r = 3;  // half-width in CSS pixels (matching backend render radius)
      const steps = Math.max(1, Math.ceil(
        Math.hypot(point.x - last.x, point.y - last.y) / r
      ));
      for (let s = 0; s <= steps; s++) {
        const t  = s / steps;
        const px = last.x + (point.x - last.x) * t;
        const py = last.y + (point.y - last.y) * t;
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fill();
      }

      currentStroke.push(point);
      last = point;
      sendInputEvent({
        type: "stroke_point",
        stroke_id: currentStrokeId,
        x: point.x, y: point.y, pressure: point.pressure,
        timestamp_ms: point.timestamp_ms
      });

      event.preventDefault();  // Rule C
    }

    function finish(event) {
      logMsg("up/cancel: type=" + event.pointerType + " id=" + event.pointerId);
      console.log("[AWP] pointerup/cancel  type:", event.pointerType,
                  " id:", event.pointerId);

      // Rule B: release pointer capture BEFORE checking drawing state so the
      // capture is freed even if drawing was already false (e.g. a cancel
      // that arrived after a duplicate finish).
      try {
        canvas.releasePointerCapture(event.pointerId);
        logMsg("pointer released: " + event.pointerId);
      } catch (_) {
        // Throws if pointerId is not captured by this element — safe to ignore.
      }

      if (!drawing) return;

      drawing = false;
      const endPoint = canvasPoint(event);
      currentStroke.push(endPoint);
      sendInputEvent({
        type: "stroke_point",
        stroke_id: currentStrokeId,
        x: endPoint.x, y: endPoint.y, pressure: endPoint.pressure,
        timestamp_ms: endPoint.timestamp_ms
      });
      sendInputEvent({type: "stroke_end", stroke_id: currentStrokeId});
      strokes.push(currentStroke);
      strokeRevision += 1;
      currentStroke = [];
      currentStrokeId = null;
      last = null;

      updatePointerDebug(event, false);
      scheduleRecognition();
      // Rule C (inverse): no preventDefault on pointerup — would block clicks.
    }

    /* -----------------------------------------------------------------------
     * OCR recognition
     * --------------------------------------------------------------------- */

    function scheduleRecognition() {
      if (recognizeTimer) clearTimeout(recognizeTimer);
      recognizeTimer = setTimeout(recognize, 350);
    }

    function drawCapturedStrokes() {
      if (!strokes.length) return;
      clearCanvasPixels();
      for (const stroke of strokes) {
        if (!stroke.length) continue;
        ctx.beginPath();
        stroke.forEach((point, index) => {
          if (index === 0) ctx.moveTo(point.x, point.y);
          else ctx.lineTo(point.x, point.y);
        });
        ctx.stroke();
      }
    }

    function clearCanvasPixels() {
      const ratio = window.devicePixelRatio || 1;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      applyDrawingStyle();
    }

    async function captureFromHuion() {
      connectInputSocket();
    }

    async function recognize() {
      if (!strokes.length) {
        statusEl.textContent = "Write on the pad first.";
        return;
      }
      if (recognitionInFlight) {
        recognitionQueued = true;
        return;
      }
      recognizeTimer = null;
      recognitionInFlight = true;
      const requestRevision = strokeRevision;
      statusEl.textContent = "Recognizing\u2026";
      resetConfidenceBadge();
      console.log("[AWP] recognize mode:", currentMode);
      try {
        const response = await fetch("/api/recognize", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({mode: currentMode})
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Recognition failed");
        if (requestRevision === strokeRevision) {
          applyRecognitionResult(result);
        } else {
          recognitionQueued = true;
        }
      } catch (err) {
        if (requestRevision === strokeRevision) {
          statusEl.textContent = err.message;
        }
      } finally {
        recognitionInFlight = false;
        if (recognitionQueued) {
          recognitionQueued = false;
          scheduleRecognition();
        }
      }
    }

    function applyRecognitionResult(result) {
      recognizedEl.value = result.corrected_text || result.text || "";
      updateConfidenceBadge(
        Number(result.confidence || 0),
        Number(result.correction_confidence || 1)
      );
      const usedMode = result.mode || currentMode;
      statusEl.textContent = result.status || "Best match ready.";
      console.log("[AWP] result mode:", usedMode, "conf:", result.confidence, "corrections:", result.corrections);
      updateRawDetails(result);
      pdStrokes.textContent = strokes.length;
      renderCorrections(result.corrections || []);
    }

    function updateRawDetails(result) {
      const meta = result.metadata || {};
      const recognizerName = meta.recognizer || "trocr";
      const lineResults = parseJsonArray(meta.line_results);
      const rawLines = lineResults
        .map(l => "Line " + l.line_index + ": " + (l.raw_text || "(empty)"))
        .join("\\n");
      const rawRecognized = result.recognized_text || meta.raw_text || result.text || "(none)";
      rawTextEl.textContent = "[" + recognizerName + "] " + (rawLines || rawRecognized);
    }

    function parseJsonArray(value) {
      if (!value) return [];
      try {
        const parsed = typeof value === "string" ? JSON.parse(value) : value;
        return Array.isArray(parsed) ? parsed : [];
      } catch (_) {
        return [];
      }
    }

    function resetConfidenceBadge() {
      confidenceEl.textContent = "\u2014";
      confidenceEl.classList.remove("conf-high", "conf-med", "conf-low");
      correctionConfidenceEl.textContent = "\u2014";
    }

    /**
     * Colour-coded confidence badge.
     *  > 0.85  -> green  (high)
     *  0.65-0.85 -> yellow (medium)
     *  < 0.65  -> red    (low — needs review)
     */
    function updateConfidenceBadge(value, correctionValue) {
      const pct = Math.round(value * 100);
      const correctionPct = Math.round(correctionValue * 100);
      confidenceEl.textContent = "Recognition: " + pct + "%";
      correctionConfidenceEl.textContent = "Correction: " + correctionPct + "%";
      confidenceEl.classList.remove("conf-high", "conf-med", "conf-low");
      if (value > 0.85) {
        confidenceEl.classList.add("conf-high");
      } else if (value >= 0.65) {
        confidenceEl.classList.add("conf-med");
      } else {
        confidenceEl.classList.add("conf-low");
      }
    }

    function renderCorrections(corrections) {
      correctionListEl.innerHTML = "";
      if (!corrections.length) {
        const empty = document.createElement("div");
        empty.className = "correction-empty";
        empty.textContent = "No changes.";
        correctionListEl.appendChild(empty);
        return;
      }

      corrections.forEach(item => {
        const row = document.createElement("div");
        row.className = "correction-row";

        const before = document.createElement("span");
        before.className = "correction-word correction-before";
        before.textContent = item.original || "";

        const arrow = document.createElement("span");
        arrow.className = "correction-arrow";
        arrow.textContent = "->";

        const after = document.createElement("span");
        after.className = "correction-word correction-after";
        const pct = Math.round(Number(item.confidence || 0) * 100);
        after.textContent = (item.corrected || "") + "  " + pct + "%";

        row.title = item.reason || "";
        row.appendChild(before);
        row.appendChild(arrow);
        row.appendChild(after);
        correctionListEl.appendChild(row);
      });
    }

    function clearScreen() {
      strokes = [];
      currentStroke = [];
      strokeRevision += 1;
      recognitionQueued = false;
      sendInputEvent({type: "clear"});
      if (recognizeTimer) {
        clearTimeout(recognizeTimer);
        recognizeTimer = null;
      }

      // Clear at device-pixel scale (identity transform), then re-apply DPR.
      clearCanvasPixels();

      recognizedEl.value = "";
      resetConfidenceBadge();
      rawTextEl.textContent = "No recognition yet.";
      statusEl.textContent = "Ready";
      pdStrokes.textContent = "0";
      renderCorrections([]);
    }

    function exportStrokePayload() {
      return {
        schema_version: 1,
        source: "browser",
        mode: currentMode,
        strokes: strokes.map(stroke => stroke.map(point => ({
          x: point.x,
          y: point.y,
          timestamp_ms: point.timestamp_ms,
          pressure: point.pressure
        })))
      };
    }

    window.assistiveWritingPad = {
      exportStrokePayload,
      strokeCount: () => strokes.length
    };

    /* -----------------------------------------------------------------------
     * Event wiring
     * --------------------------------------------------------------------- */
    window.addEventListener("resize", resizeCanvas);

    // Pointer Events API: mouse + trackpad + touch + pen in one model.
    canvas.addEventListener("pointerdown",   start);
    canvas.addEventListener("pointermove",   move);
    canvas.addEventListener("pointerup",     finish);
    canvas.addEventListener("pointercancel", finish);

    // Block wheel/scroll on the canvas so trackpad two-finger scroll
    // never triggers pointercancel on the active drawing stroke.
    canvas.addEventListener("wheel", (e) => e.preventDefault(), { passive: false });

    // Toolbar buttons
    document.getElementById("recognize").addEventListener("click", recognize);
    captureHuionEl.addEventListener("click", connectInputSocket);
    document.getElementById("clearScreen").addEventListener("click", clearScreen);

    // Initial canvas setup.
    resizeCanvas();
    renderCorrections([]);
    connectInputSocket();
  </script>
</body>
</html>
"""


CAPTURE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evaluation Capture</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #667085;
      --line: #d0d5dd;
      --accent: #2563eb;
      --good: #067647;
      --bad: #b42318;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: var(--bg); color: var(--ink); }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 22px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      padding: 18px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
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
    label {
      display: block;
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    input, select, textarea {
      width: 100%;
      margin-top: 5px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      color: var(--ink);
      background: #fff;
      font: inherit;
      font-size: 15px;
    }
    textarea { min-height: 86px; resize: vertical; }
    .toolbar { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      min-height: 38px;
      padding: 8px 12px;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
      font-size: 14px;
    }
    button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    #status { color: var(--muted); font-size: 14px; }
    #huion-status { color: var(--muted); font-size: 12px; margin-top: 6px; }
    #message { margin-top: 12px; font-size: 14px; color: var(--muted); }
    #message.ok { color: var(--good); }
    #message.err { color: var(--bad); }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      #pad { height: 360px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Evaluation Capture</h1>
    <div>
      <div id="status">0 strokes</div>
      <div id="huion-status">Huion: Disconnected</div>
    </div>
  </header>
  <main>
    <section>
      <canvas id="pad"></canvas>
      <div class="toolbar">
        <button id="clear">Clear Screen</button>
      </div>
    </section>
    <section>
      <label>Case ID
        <input id="caseId" autocomplete="off" placeholder="word_the_001">
      </label>
      <label>Category
        <select id="category">
          <option value="word">word</option>
          <option value="sentence">sentence</option>
          <option value="single_character">single_character</option>
        </select>
      </label>
      <label>Expected Corrected Text
        <textarea id="expected"></textarea>
      </label>
      <label>Expected Raw OCR Text
        <textarea id="expectedRecognized"></textarea>
      </label>
      <label>Notes
        <textarea id="notes"></textarea>
      </label>
      <button class="primary" id="save">Save Evaluation Case</button>
      <div id="message"></div>
    </section>
  </main>
  <script>
    const canvas = document.getElementById("pad");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const messageEl = document.getElementById("message");
    let strokes = [];
    let currentStroke = [];
    let drawing = false;
    let last = null;
    let startedAt = 0;
    let inputSocket = null;
    let inputSocketRetry = null;
    let huionPointCount = 0;

    function applyDrawingStyle() {
      ctx.lineWidth = 4;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#111827";
      ctx.fillStyle = "#111827";
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const previous = canvas.width && canvas.height
        ? ctx.getImageData(0, 0, canvas.width, canvas.height)
        : null;
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (previous) {
        try { ctx.putImageData(previous, 0, 0); } catch (_) {}
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      applyDrawingStyle();
    }

    function canvasPoint(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
        timestamp_ms: Math.round(performance.now() - startedAt),
        pressure: (event.pressure != null && event.pressure > 0) ? event.pressure : 1.0
      };
    }

    function normalizedCanvasPoint(x, y, metadata = {}) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(1, Number(x))) * rect.width,
        y: Math.max(0, Math.min(1, Number(y))) * rect.height,
        timestamp_ms: Number(metadata.timestamp_ms || 0),
        pressure: Number(metadata.pressure || 1)
      };
    }

    function drawPoint(point) {
      if (last) {
        ctx.beginPath();
        ctx.moveTo(last.x, last.y);
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 2, 0, Math.PI * 2);
        ctx.fill();
      }
      last = point;
    }

    function startHuionStroke() {
      if (currentStroke.length) strokes.push(currentStroke);
      currentStroke = [];
      drawing = true;
      last = null;
      huionStatusEl.textContent = "Huion: Active";
      console.info("[capture] Huion stroke start");
    }

    function finishHuionStroke() {
      if (currentStroke.length) strokes.push(currentStroke);
      currentStroke = [];
      drawing = false;
      last = null;
      updateStatus();
      huionStatusEl.textContent = "Huion: Connected";
      console.info("[capture] Huion stroke end");
    }

    function handleHuionMessage(event) {
      if (event.type === "snapshot") {
        return;
      }
      if (event.type === "clear") {
        clearScreen();
        return;
      }
      if (event.type === "stroke_start" && event.source === "huion") {
        startHuionStroke();
        return;
      }
      if (event.type === "stroke_end" && event.source === "huion") {
        finishHuionStroke();
        return;
      }
      if (event.type !== "stroke_point" || event.source !== "huion") return;
      if (event.coordinate_space !== "normalized") {
        console.warn("[capture] Ignoring Huion point without normalized coordinates");
        return;
      }
      const point = normalizedCanvasPoint(event.x, event.y, event);
      huionPointCount += 1;
      if (huionPointCount === 1 || huionPointCount % 100 === 0) {
        console.info(
          "[capture] Huion point received x=%s y=%s pressure=%s",
          point.x.toFixed(1), point.y.toFixed(1), point.pressure.toFixed(3)
        );
      }
      if (!drawing) startHuionStroke();
      drawPoint(point);
      currentStroke.push(point);
      huionStatusEl.textContent = "Huion: Active";
    }

    function connectHuionSocket() {
      if (inputSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(inputSocket.readyState)) {
        return;
      }
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      inputSocket = new WebSocket(protocol + "//" + window.location.host + "/ws/input");
      inputSocket.onopen = () => {
        huionStatusEl.textContent = "Huion: Connected";
        console.info("[capture] Huion WebSocket connected");
      };
      inputSocket.onmessage = message => {
        try {
          handleHuionMessage(JSON.parse(message.data));
        } catch (err) {
          console.warn("[capture] malformed WebSocket message", err);
        }
      };
      inputSocket.onerror = () => {
        huionStatusEl.textContent = "Huion: Disconnected";
      };
      inputSocket.onclose = () => {
        huionStatusEl.textContent = "Huion: Disconnected";
        if (inputSocketRetry) clearTimeout(inputSocketRetry);
        inputSocketRetry = setTimeout(connectHuionSocket, 1000);
      };
    }

    function start(event) {
      if (event.pointerType === "mouse" && event.button !== 0) return;
      if (drawing) finishHuionStroke();
      drawing = true;
      startedAt = performance.now();
      currentStroke = [];
      last = canvasPoint(event);
      currentStroke.push(last);
      try { canvas.setPointerCapture(event.pointerId); } catch (_) {}
      event.preventDefault();
    }

    function move(event) {
      if (!drawing) return;
      const point = canvasPoint(event);
      drawPoint(point);
      currentStroke.push(point);
      event.preventDefault();
    }

    function finish(event) {
      try { canvas.releasePointerCapture(event.pointerId); } catch (_) {}
      if (!drawing) return;
      drawing = false;
      const point = canvasPoint(event);
      drawPoint(point);
      currentStroke.push(point);
      strokes.push(currentStroke);
      currentStroke = [];
      last = null;
      updateStatus();
    }

    function updateStatus() {
      statusEl.textContent = strokes.length + (strokes.length === 1 ? " stroke" : " strokes");
    }

    function exportStrokePayload() {
      return {
        schema_version: 1,
        source: "browser_capture",
        strokes: strokes.map(stroke => stroke.map(point => ({
          x: point.x,
          y: point.y,
          timestamp_ms: point.timestamp_ms,
          pressure: point.pressure
        })))
      };
    }

    async function saveCase() {
      messageEl.className = "";
      messageEl.textContent = "Saving...";
      const payload = {
        id: document.getElementById("caseId").value,
        category: document.getElementById("category").value,
        expected: document.getElementById("expected").value,
        expected_recognized: document.getElementById("expectedRecognized").value,
        notes: document.getElementById("notes").value,
        strokes: exportStrokePayload().strokes
      };
      try {
        const response = await fetch("/api/evaluation/cases", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Save failed");
        clearScreen();
        messageEl.className = "ok";
        messageEl.textContent = "Saved " + result.id;
      } catch (err) {
        messageEl.className = "err";
        messageEl.textContent = err.message;
      }
    }

    function clearScreen() {
      strokes = [];
      currentStroke = [];
      drawing = false;
      last = null;
      const ratio = window.devicePixelRatio || 1;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      applyDrawingStyle();
      messageEl.textContent = "";
      updateStatus();
    }

    window.addEventListener("resize", resizeCanvas);
    window.addEventListener("beforeunload", () => {
      if (inputSocket) inputSocket.close();
      if (inputSocketRetry) clearTimeout(inputSocketRetry);
    });
    canvas.addEventListener("pointerdown", start);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", finish);
    canvas.addEventListener("pointercancel", finish);
    canvas.addEventListener("wheel", event => event.preventDefault(), { passive: false });
    document.getElementById("clear").addEventListener("click", clearScreen);
    document.getElementById("save").addEventListener("click", saveCase);
    window.assistiveWritingPadCapture = { exportStrokePayload, strokeCount: () => strokes.length };
    resizeCanvas();
    updateStatus();
    const huionStatusEl = document.getElementById("huion-status");
    connectHuionSocket();
  </script>
</body>
</html>
"""


class RecognitionService:
    def __init__(
        self,
        recognizer: Optional[Any] = None,
        corrector: Optional[Any] = None,
        settings: Optional[RuntimeSettings] = None,
    ) -> None:
        self.settings = settings or RuntimeSettings.from_env()
        self.recognizer = recognizer or TrOCRHandwritingRecognizer(
            device_profile=self.settings.device_profile
        )
        self.pipeline = WritingPipeline(
            recognizer=self.recognizer,
            corrector=corrector or corrector_from_settings(self.settings),
            settings=self.settings,
        )
        self._state_lock = threading.RLock()
        self._strokes: List[List[StrokePoint]] = []
        self._active_strokes: Dict[str, List[StrokePoint]] = {}
        self._websocket_clients: set[socket.socket] = set()
        self._huion_thread: Optional[threading.Thread] = None
        self._huion_started = False
        self._last_input_source = "unknown"
        self._huion_axis_ranges = (
            (0.0, _HUION_MAX_X),
            (0.0, _HUION_MAX_Y),
        )
        self._huion_point_count = 0

    def warm_up_async(self) -> None:
        if self.settings.preload_ocr_model:
            thread = threading.Thread(target=self._warm_up_ocr_model, daemon=True)
            thread.start()
        if self.settings.preload_correction_models:
            thread = threading.Thread(target=self._warm_up_correction_models, daemon=True)
            thread.start()

    def _warm_up_ocr_model(self) -> None:
        warm_up = getattr(self.recognizer, "warm_up", None)
        ensure_loaded = getattr(self.recognizer, "_ensure_loaded", None)
        loader = warm_up if callable(warm_up) else ensure_loaded
        if not callable(loader):
            return
        try:
            loader()
            logger.info("OCR model warmed up")
        except Exception as exc:
            logger.warning("OCR model warm-up failed; first recognition may retry: %s", exc)

    def _warm_up_correction_models(self) -> None:
        warm_up = getattr(self.pipeline.corrector, "warm_up", None)
        if not callable(warm_up):
            return
        try:
            warm_up()
            logger.info("correction models warmed up")
        except Exception as exc:
            logger.warning("correction model warm-up failed; first correction may retry: %s", exc)

    def recognize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "strokes" in payload:
            stroke_groups = stroke_groups_from_payload(payload)
        else:
            stroke_groups = self.stroke_snapshot()
        if not any(stroke_groups):
            raise ValueError("Write on the pad first.")
        point_count = sum(len(stroke) for stroke in stroke_groups)
        all_points = [point for stroke in stroke_groups for point in stroke]
        bbox = (
            min(point.x for point in all_points),
            min(point.y for point in all_points),
            max(point.x for point in all_points),
            max(point.y for point in all_points),
        )
        logger.info(
            "recognition input: source=%s strokes=%d points=%d bbox=(%.1f,%.1f,%.1f,%.1f)",
            self._last_input_source,
            len(stroke_groups),
            point_count,
            *bbox,
        )
        # Accept legacy mode values, but recognition always uses OCR.
        # The browser explicitly requests OCR mode; omitted mode remains a
        # backward-compatible API default for existing callers.
        mode = payload.get("mode", "auto")
        if mode not in ("auto", "character", "word", "ocr"):
            mode = "ocr"
        result = self.recognizer.recognize_stroke_groups(stroke_groups, mode=mode)
        pipeline_result = self.pipeline.process_recognition(result)
        selected_recognition = pipeline_result.recognition
        # Parse top3 from metadata (list of [char, confidence] pairs).
        top3_raw = result.metadata.get("top3", "[]")
        try:
            top3 = json.loads(top3_raw)
        except (json.JSONDecodeError, TypeError):
            top3 = []
        correction = pipeline_result.correction
        return {
            "text": correction.corrected_text,
            "recognized_text": selected_recognition.text,
            "corrected_text": correction.corrected_text,
            "confidence": selected_recognition.confidence,
            "correction_confidence": correction.confidence,
            "corrections": corrections_payload(correction),
            "correction_metadata": correction.metadata,
            "needs_review": pipeline_result.needs_review,
            "review_reason": pipeline_result.review_reason,
            "status": status_message(pipeline_result),
            "metadata": selected_recognition.metadata,
            "top3": top3,
            "mode": selected_recognition.metadata.get("mode", mode),
        }

    def stroke_snapshot(self) -> List[List[StrokePoint]]:
        with self._state_lock:
            strokes = [list(stroke) for stroke in self._strokes]
            strokes.extend(list(stroke) for stroke in self._active_strokes.values() if stroke)
            return strokes

    def clear_strokes(self) -> None:
        with self._state_lock:
            self._strokes.clear()
            self._active_strokes.clear()
        self._broadcast_websocket({"type": "clear"})

    def ingest_stroke_event(self, event: Dict[str, Any], source: str) -> None:
        event_type = str(event.get("type", ""))
        stroke_id = str(event.get("stroke_id") or f"{source}-{uuid.uuid4().hex}")
        outbound = dict(event)
        outbound["source"] = source
        outbound["stroke_id"] = stroke_id
        if (
            source == "huion"
            and event_type in {"stroke_start", "stroke_point"}
            and "x" in event
            and "y" in event
        ):
            self._huion_point_count += 1
            x, y = _huion_to_normalized(
                float(event["x"]),
                float(event["y"]),
                axis_ranges=self._huion_axis_ranges,
            )
            outbound["x"] = x
            outbound["y"] = y
            outbound["coordinate_space"] = "normalized"
            if self._huion_point_count == 1 or self._huion_point_count % 100 == 0:
                logger.info(
                    "[HUION RAW] x=%.1f y=%.1f | [SERVER POINT] x=%.4f y=%.4f | "
                    "[WEBSOCKET POINT] x=%.4f y=%.4f",
                    float(event["x"]),
                    float(event["y"]),
                    x,
                    y,
                    x,
                    y,
                )
            if _near_mapping_edge(x) or _near_mapping_edge(y):
                logger.info(
                    "Huion corner diagnostic: raw=(%.1f,%.1f) normalized=(%.4f,%.4f)",
                    float(event["x"]),
                    float(event["y"]),
                    x,
                    y,
                )
        with self._state_lock:
            self._last_input_source = source
            if event_type == "stroke_start":
                self._active_strokes[stroke_id] = []
                if "x" in event and "y" in event:
                    x, y = (
                        _huion_to_normalized(
                            float(event["x"]),
                            float(event["y"]),
                            axis_ranges=self._huion_axis_ranges,
                        )
                        if source == "huion"
                        else (float(event["x"]), float(event["y"]))
                    )
                    self._active_strokes[stroke_id].append(
                        StrokePoint(
                            x=x,
                            y=y,
                            pressure=float(event.get("pressure", 1.0)),
                            timestamp_ms=int(event.get("timestamp_ms", 0)),
                        )
                    )
            elif event_type == "stroke_point":
                try:
                    x, y = (
                        _huion_to_normalized(
                            float(event["x"]),
                            float(event["y"]),
                            axis_ranges=self._huion_axis_ranges,
                        )
                        if source == "huion"
                        else (float(event["x"]), float(event["y"]))
                    )
                    point = StrokePoint(
                        x=x,
                        y=y,
                        pressure=float(event.get("pressure", 1.0)),
                        timestamp_ms=int(event.get("timestamp_ms", 0)),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("invalid stroke point") from exc
                self._active_strokes.setdefault(stroke_id, []).append(point)
            elif event_type == "stroke_end":
                stroke = self._active_strokes.pop(stroke_id, [])
                if stroke:
                    self._strokes.append(stroke)
            elif event_type == "clear":
                self._strokes.clear()
                self._active_strokes.clear()
            else:
                raise ValueError(f"unsupported stroke event: {event_type}")
        self._broadcast_websocket(outbound)

    def start_huion_reader(self) -> None:
        with self._state_lock:
            if self._huion_started:
                return
            self._huion_started = True
        device_path = os.environ.get("AWP_HUION_DEVICE", "").strip()
        if not device_path:
            device_path = find_huion_device() or "/dev/input/event4"
        try:
            self._huion_axis_ranges = huion_axis_ranges(device_path)
            logger.info(
                "Huion axis ranges: X %.0f..%.0f, Y %.0f..%.0f",
                self._huion_axis_ranges[0][0],
                self._huion_axis_ranges[0][1],
                self._huion_axis_ranges[1][0],
                self._huion_axis_ranges[1][1],
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Could not read Huion axis ranges from %s; using defaults: %s",
                device_path,
                exc,
            )
        logger.info("Huion input device: %s", device_path)
        self._huion_thread = threading.Thread(
            target=self._read_huion,
            args=(device_path,),
            daemon=True,
            name="huion-reader",
        )
        self._huion_thread.start()
        logger.info("Huion reader: STARTED")

    def _read_huion(self, device_path: str) -> None:
        try:
            for event in HuionEventReader(device_path).iter_stroke_events():
                self.ingest_stroke_event(event, source="huion")
        except Exception:
            logger.exception("Huion reader stopped on %s", device_path)
        finally:
            logger.info("Huion reader stopped on %s", device_path)

    def register_websocket(self, connection: socket.socket) -> None:
        with self._state_lock:
            self._websocket_clients.add(connection)

    def unregister_websocket(self, connection: socket.socket) -> None:
        with self._state_lock:
            self._websocket_clients.discard(connection)

    def _broadcast_websocket(self, event: Dict[str, Any]) -> None:
        payload = json.dumps(event, separators=(",", ":")).encode("utf-8")
        dead: List[socket.socket] = []
        with self._state_lock:
            clients = list(self._websocket_clients)
        for client in clients:
            try:
                client.sendall(_websocket_frame(payload))
            except OSError:
                dead.append(client)
        for client in dead:
            self.unregister_websocket(client)

    def websocket_snapshot(self) -> Dict[str, Any]:
        return {
            "type": "snapshot",
            "coordinate_space": "normalized",
            "strokes": [
                [
                    {
                        "x": point.x,
                        "y": point.y,
                        "pressure": point.pressure,
                        "timestamp_ms": point.timestamp_ms,
                    }
                    for point in stroke
                ]
                for stroke in self.stroke_snapshot()
            ],
        }

    def capture_huion_payload(self) -> Dict[str, Any]:
        device_path = os.environ.get("AWP_HUION_DEVICE", "/dev/input/event4").strip()
        duration = _float_env("AWP_HUION_CAPTURE_SECONDS", 15.0)
        idle_timeout = _float_env("AWP_HUION_IDLE_SECONDS", 2.0)
        if not device_path:
            raise ValueError("AWP_HUION_DEVICE must not be empty")
        if duration <= 0 or idle_timeout <= 0:
            raise ValueError("Huion capture timeouts must be positive")

        strokes = HuionEventReader(device_path).capture_strokes(
            duration_seconds=duration,
            idle_timeout_seconds=idle_timeout,
        )
        return {
            "source": "huion",
            "device": device_path,
            "strokes": [
                [
                    {
                        "x": point.x,
                        "y": point.y,
                        "timestamp_ms": point.timestamp_ms,
                        "pressure": point.pressure,
                    }
                    for point in stroke
                ]
                for stroke in strokes
            ],
        }

    def correct_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text", "")
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        correction = self.pipeline.corrector.correct(text)
        return {
            "text": correction.corrected_text,
            "recognized_text": text,
            "corrected_text": correction.corrected_text,
            "confidence": 1.0,
            "correction_confidence": correction.confidence,
            "corrections": corrections_payload(correction),
            "correction_metadata": correction.metadata,
            "needs_review": False,
            "review_reason": None,
            "status": "Corrected alternative.",
            "metadata": {},
            "top3": [],
            "mode": "text",
        }

    def append_evaluation_case_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.settings.evaluation_capture_enabled:
            raise PermissionError("evaluation capture is disabled")

        expected_recognized = payload.get("expected_recognized")
        if isinstance(expected_recognized, str) and not expected_recognized.strip():
            expected_recognized = None
        record = build_end_to_end_case_record(
            case_id=str(payload.get("id", "")),
            category=str(payload.get("category", "")),
            expected=str(payload.get("expected", "")),
            expected_recognized=expected_recognized if isinstance(expected_recognized, str) else None,
            source="manual",
            notes=str(payload.get("notes", "")),
            stroke_payload=payload,
        )
        append_jsonl_record(self.settings.evaluation_manifest_path, record)
        return {
            "id": record["id"],
            "category": record["category"],
            "source": record["source"],
            "manifest": str(self.settings.evaluation_manifest_path),
            "status": "saved",
        }


def corrections_payload(result: CorrectionResult) -> List[Dict[str, Any]]:
    return [
        {
            "original": correction.original,
            "corrected": correction.corrected,
            "confidence": correction.confidence,
            "reason": correction.reason,
        }
        for correction in result.corrections
    ]


def status_message(result: PipelineResult) -> str:
    if result.needs_review:
        return "Best match ready."
    if result.correction.changed:
        return "Writing corrected."
    return "Writing recognized."


_HUION_MAX_X = 32000.0
_HUION_MAX_Y = 20400.0
def _huion_to_normalized(
    x: float,
    y: float,
    *,
    axis_ranges: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, _HUION_MAX_X),
        (0.0, _HUION_MAX_Y),
    ),
) -> tuple[float, float]:
    (min_x, max_x), (min_y, max_y) = axis_ranges
    normalized_x = _clamp_unit((x - min_x) / (max_x - min_x))
    normalized_y = _clamp_unit((y - min_y) / (max_y - min_y))
    logger.debug(
        "Huion mapping raw=(%.1f,%.1f) normalized=(%.4f,%.4f)",
        x,
        y,
        normalized_x,
        normalized_y,
    )
    return normalized_x, normalized_y


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _near_mapping_edge(value: float, tolerance: float = 0.05) -> bool:
    return value <= tolerance or value >= 1.0 - tolerance


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


def _websocket_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    first = 0x80 | (opcode & 0x0F)
    length = len(payload)
    if length < 126:
        return bytes([first, length]) + payload
    if length < 65536:
        return bytes([first, 126]) + length.to_bytes(2, "big") + payload
    return bytes([first, 127]) + length.to_bytes(8, "big") + payload


def _read_websocket_frame(connection: socket.socket) -> Optional[tuple[int, bytes]]:
    header = _recv_exact(connection, 2)
    if not header:
        return None
    first, second = header
    opcode = first & 0x0F
    length = second & 0x7F
    if length == 126:
        length = int.from_bytes(_recv_exact(connection, 2), "big")
    elif length == 127:
        length = int.from_bytes(_recv_exact(connection, 8), "big")
    masked = bool(second & 0x80)
    mask = _recv_exact(connection, 4) if masked else b""
    payload = _recv_exact(connection, length)
    if masked:
        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return opcode, payload


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            return b""
        chunks.extend(chunk)
    return bytes(chunks)


def make_handler(service: RecognitionService):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send(HTTPStatus.OK, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path == "/ws/input":
                self._handle_websocket()
                return
            if self.path == "/capture":
                if not service.settings.evaluation_capture_enabled:
                    self._send_json(
                        HTTPStatus.FORBIDDEN,
                        {"error": "evaluation capture is disabled"},
                    )
                    return
                self._send(
                    HTTPStatus.OK,
                    CAPTURE_HTML.encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _handle_websocket(self) -> None:
            if self.headers.get("Upgrade", "").lower() != "websocket":
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "websocket upgrade required"})
                return
            key = self.headers.get("Sec-WebSocket-Key")
            if not key:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing websocket key"})
                return
            accept = base64.b64encode(
                hashlib.sha1(
                    (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
                ).digest()
            ).decode("ascii")
            self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            connection = self.connection
            service.register_websocket(connection)
            logger.info("WebSocket client connected")
            service.start_huion_reader()
            try:
                connection.sendall(_websocket_frame(json.dumps(service.websocket_snapshot()).encode()))
                if os.environ.get("AWP_HUION_DEBUG_TEST", "").strip() == "1":
                    connection.sendall(
                        _websocket_frame(
                            json.dumps({"type": "test_point", "x": 100, "y": 100}).encode()
                        )
                    )
                while True:
                    frame = _read_websocket_frame(connection)
                    if frame is None:
                        break
                    opcode, payload = frame
                    if opcode == 0x8:
                        connection.sendall(_websocket_frame(b"", opcode=0x8))
                        break
                    if opcode == 0x9:
                        connection.sendall(_websocket_frame(payload, opcode=0xA))
                        continue
                    if opcode != 0x1:
                        continue
                    event = json.loads(payload.decode("utf-8"))
                    if not isinstance(event, dict):
                        raise ValueError("websocket event must be an object")
                    service.ingest_stroke_event(event, source="browser")
            except (ConnectionError, OSError, json.JSONDecodeError, ValueError):
                logger.info("WebSocket input client disconnected")
            finally:
                service.unregister_websocket(connection)

        def do_POST(self) -> None:
            if self.path not in {
                "/api/recognize",
                "/api/correct",
                "/api/huion/capture",
                "/api/evaluation/cases",
            }:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/api/recognize":
                    result = service.recognize_payload(payload)
                elif self.path == "/api/correct":
                    result = service.correct_payload(payload)
                elif self.path == "/api/huion/capture":
                    result = service.capture_huion_payload()
                else:
                    result = service.append_evaluation_case_payload(payload)
            except RecognitionUnavailable as exc:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
                return
            except PermissionError as exc:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    service = RecognitionService()
    service.warm_up_async()
    server = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"Assistive Writing Pad running at http://{host}:{port}")
    if service.settings.evaluation_capture_enabled:
        print(f"Evaluation capture running at http://{host}:{port}/capture")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
