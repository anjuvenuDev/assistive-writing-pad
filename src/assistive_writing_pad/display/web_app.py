"""Browser-based handwriting pad with pretrained OCR endpoint."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import CorrectionResult, PipelineResult, StrokePoint
from assistive_writing_pad.correction.contextual import ContextualCorrector
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
    .hint, #status {
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
    #recognized {
      width: 100%;
      min-height: 180px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      font-size: 24px;
      line-height: 1.35;
      color: var(--ink);
      background: #fbfcfd;
    }
    #recognized-raw {
      width: 100%;
      min-height: 56px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 20px;
      line-height: 1.35;
      color: #334155;
      background: #f8fafc;
      margin-bottom: 10px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .output-label {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin: 10px 0 6px;
    }
    #suggestion-panel {
      margin-top: 12px;
      border: 1px solid #f59e0b;
      background: #fffbeb;
      border-radius: 8px;
      padding: 10px;
      display: none;
    }
    .suggestion-title {
      font-size: 13px;
      font-weight: 700;
      color: #92400e;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .suggestion-main {
      margin-top: 6px;
      font-size: 20px;
      font-weight: 700;
      color: #111827;
    }
    .suggestion-meta {
      margin-top: 4px;
      color: #6b7280;
      font-size: 13px;
    }
    .suggestion-actions {
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .suggestion-actions button {
      min-height: 34px;
      padding: 6px 10px;
      font-size: 13px;
    }
    #suggestion-alternatives {
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .suggestion-alt {
      border: 1px solid #fcd34d;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      color: #92400e;
      background: #fff7d6;
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
    #top3-panel {
      margin-top: 12px;
    }
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
    .top3-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }
    .top3-list {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .top3-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 6px 10px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.12s;
    }
    .top3-item:hover { background: #e8f0fe; border-color: var(--accent); }
    .top3-rank {
      font-size: 11px;
      color: var(--muted);
      width: 14px;
      text-align: center;
      flex-shrink: 0;
    }
    .top3-char {
      font-size: 24px;
      font-weight: 700;
      color: var(--ink);
      width: 32px;
      text-align: center;
      flex-shrink: 0;
      font-family: monospace;
    }
    .top3-bar-wrap {
      flex: 1;
      background: #e2e8f0;
      border-radius: 4px;
      height: 8px;
      overflow: hidden;
    }
    .top3-bar {
      height: 100%;
      border-radius: 4px;
      background: var(--accent);
      transition: width 0.3s ease;
    }
    .top3-bar.rank1 { background: #2563eb; }
    .top3-bar.rank2 { background: #60a5fa; }
    .top3-bar.rank3 { background: #93c5fd; }
    .top3-bar.rank4 { background: #bfdbfe; }
    .top3-bar.rank5 { background: #dbeafe; }
    .top3-pct {
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
      width: 36px;
      text-align: right;
      flex-shrink: 0;
    }
    .top3-item.top3-hidden { display: none; }
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
      <!-- Pointer diagnostics: always visible so input problems are immediately obvious -->
      <div id="pointer-debug">
        <div class="pdl">
          <span>Type: <b id="pd-type">&mdash;</b></span>
          <span>X: <b id="pd-x">&mdash;</b></span>
          <span>Y: <b id="pd-y">&mdash;</b></span>
          <span>Drawing: <b id="pd-drawing" class="drawing-no">no</b></span>
          <span>Strokes: <b id="pd-strokes">0</b></span>
        </div>
        <div id="pointer-log">Canvas event logs will appear here...</div>
      </div>
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
        <h2>Recognition and Correction</h2>
        <span id="confidence">&mdash;</span>
      </div>
      <div class="output-label">Recognized Handwriting</div>
      <div id="recognized-raw">No recognition yet.</div>
      <div class="output-label">Corrected Text</div>
      <textarea id="recognized" spellcheck="false"></textarea>
      <div id="suggestion-panel">
        <div class="suggestion-title">Suggestion</div>
        <div class="suggestion-main" id="suggestion-text">&mdash;</div>
        <div class="suggestion-meta" id="suggestion-confidence">Confidence: &mdash;</div>
        <div id="suggestion-alternatives"></div>
        <div class="suggestion-actions">
          <button id="acceptSuggestion">Accept Correction</button>
          <button id="rejectSuggestion">Reject Correction</button>
          <button id="keepOriginal">Keep Original</button>
        </div>
      </div>
      <div id="correction-panel">
        <div class="correction-header">
          <span>Corrections</span>
          <span id="correction-confidence">&mdash;</span>
        </div>
        <div id="correction-list">
          <div class="correction-empty">No corrections yet.</div>
        </div>
      </div>
      <!-- OCR candidate panel -->
      <div id="top3-panel">
        <div class="top3-title">Top Predictions</div>
        <div class="top3-list" id="top3-list">
          <div class="top3-item top3-hidden" id="top3-0" data-rank="0">
            <span class="top3-rank">1</span>
            <span class="top3-char" id="top3-char-0">&mdash;</span>
            <div class="top3-bar-wrap"><div class="top3-bar rank1" id="top3-bar-0" style="width:0%"></div></div>
            <span class="top3-pct" id="top3-pct-0">0%</span>
          </div>
          <div class="top3-item top3-hidden" id="top3-1" data-rank="1">
            <span class="top3-rank">2</span>
            <span class="top3-char" id="top3-char-1">&mdash;</span>
            <div class="top3-bar-wrap"><div class="top3-bar rank2" id="top3-bar-1" style="width:0%"></div></div>
            <span class="top3-pct" id="top3-pct-1">0%</span>
          </div>
          <div class="top3-item top3-hidden" id="top3-2" data-rank="2">
            <span class="top3-rank">3</span>
            <span class="top3-char" id="top3-char-2">&mdash;</span>
            <div class="top3-bar-wrap"><div class="top3-bar rank3" id="top3-bar-2" style="width:0%"></div></div>
            <span class="top3-pct" id="top3-pct-2">0%</span>
          </div>
          <div class="top3-item top3-hidden" id="top3-3" data-rank="3">
            <span class="top3-rank">4</span>
            <span class="top3-char" id="top3-char-3">&mdash;</span>
            <div class="top3-bar-wrap"><div class="top3-bar rank4" id="top3-bar-3" style="width:0%"></div></div>
            <span class="top3-pct" id="top3-pct-3">0%</span>
          </div>
          <div class="top3-item top3-hidden" id="top3-4" data-rank="4">
            <span class="top3-rank">5</span>
            <span class="top3-char" id="top3-char-4">&mdash;</span>
            <div class="top3-bar-wrap"><div class="top3-bar rank5" id="top3-bar-4" style="width:0%"></div></div>
            <span class="top3-pct" id="top3-pct-4">0%</span>
          </div>
        </div>
      </div>
      <details class="debug-panel">
        <summary>Debug: raw OCR output</summary>
        <div id="raw-text">No recognition yet.</div>
      </details>
      <div class="setup">
        Pretrained OCR uses <code>microsoft/trocr-base-handwritten</code> with line-aware
        segmentation. If recognition reports missing dependencies, create a Python 3.10
        environment and run <code>scripts/setup_model_env.sh</code>.
      </div>
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
    const recognizedRawEl = document.getElementById("recognized-raw");
    const recognizedEl = document.getElementById("recognized");
    const rawTextEl    = document.getElementById("raw-text");
    const correctionConfidenceEl = document.getElementById("correction-confidence");
    const correctionListEl = document.getElementById("correction-list");
    const suggestionPanelEl = document.getElementById("suggestion-panel");
    const suggestionTextEl = document.getElementById("suggestion-text");
    const suggestionConfidenceEl = document.getElementById("suggestion-confidence");
    const suggestionAlternativesEl = document.getElementById("suggestion-alternatives");
    const acceptSuggestionEl = document.getElementById("acceptSuggestion");
    const rejectSuggestionEl = document.getElementById("rejectSuggestion");
    const keepOriginalEl = document.getElementById("keepOriginal");
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
    let drawing        = false;
    let last           = null;  // last canvas-space point {x, y, ...}
    let startedAt      = 0;     // performance.now() at stroke start
    let recognizeTimer = null;
    let currentMode    = "ocr";
    let lastRecognitionResult = null;

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
      last = canvasPoint(event);
      currentStroke.push(last);

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
      currentStroke.push(canvasPoint(event));
      strokes.push(currentStroke);
      currentStroke = [];
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
      recognizeTimer = setTimeout(recognize, 800);
    }

    async function recognize() {
      if (!strokes.length) {
        statusEl.textContent = "Write on the pad first.";
        return;
      }
      statusEl.textContent = "Recognizing\u2026";
      confidenceEl.textContent = "\u2014";
      confidenceEl.classList.remove("conf-high", "conf-med", "conf-low");
      console.log("[AWP] recognize mode:", currentMode);
      try {
        const response = await fetch("/api/recognize", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({strokes, mode: currentMode})
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Recognition failed");
        lastRecognitionResult = result;
        recognizedRawEl.textContent = result.recognized_text || result.text || "";
        recognizedEl.value = result.corrected_text || result.text || "";
        updateConfidenceBadge(
          Number(result.confidence || 0),
          Number(result.correction_confidence || 1)
        );
        const usedMode = result.mode || currentMode;
        const reviewSuffix = result.needs_review ? "  Review" : "";
        statusEl.textContent = (result.status || "Recognized") + "  [" + usedMode + " mode]" + reviewSuffix;
        console.log("[AWP] result mode:", usedMode, "conf:", result.confidence, "corrections:", result.corrections, "top5:", result.top3);
        // Surface raw OCR output / recognizer info in the debug panel.
        const meta = result.metadata || {};
        const recognizerName = meta.recognizer || "trocr";
        const lineResults = meta.line_results ? JSON.parse(meta.line_results) : [];
        const rawLines = lineResults
          .map(l => "Line " + l.line_index + ": " + (l.raw_text || "(empty)"))
          .join("\\n");
        const rawRecognized = result.recognized_text || meta.raw_text || result.text || "(none)";
        rawTextEl.textContent = "[" + recognizerName + "] " + (rawLines || rawRecognized);
        pdStrokes.textContent = strokes.length;
        renderCorrections(result.corrections || []);
        renderSuggestion(result);
        // Render OCR candidates when the recognizer returns them.
        renderTop3(result.top3 || []);
      } catch (err) {
        statusEl.textContent = err.message;
      }
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

    function renderSuggestion(result) {
      if (!result || result.correction_status !== "suggestion" || !result.suggestion_text) {
        suggestionPanelEl.style.display = "none";
        suggestionTextEl.textContent = "\u2014";
        suggestionConfidenceEl.textContent = "Confidence: \u2014";
        suggestionAlternativesEl.innerHTML = "";
        return;
      }

      suggestionPanelEl.style.display = "block";
      suggestionTextEl.textContent = result.suggestion_text;
      const pct = Math.round(Number(result.correction_confidence || 0) * 100);
      suggestionConfidenceEl.textContent = "Confidence: " + pct + "%";

      suggestionAlternativesEl.innerHTML = "";
      const alternatives = Array.isArray(result.alternatives) ? result.alternatives.slice(0, 3) : [];
      alternatives.forEach((item) => {
        const chip = document.createElement("span");
        chip.className = "suggestion-alt";
        const altPct = Math.round(Number(item.confidence || 0) * 100);
        chip.textContent = (item.text || "") + " (" + altPct + "%)";
        suggestionAlternativesEl.appendChild(chip);
      });
    }

    async function sendCorrectionFeedback(decision) {
      if (!lastRecognitionResult) {
        return;
      }

      const original = String(lastRecognitionResult.recognized_text || "");
      const selected = decision === "accept"
        ? String(lastRecognitionResult.suggestion_text || "")
        : original;

      if (decision === "accept") {
        recognizedEl.value = selected;
      } else {
        recognizedEl.value = original;
      }

      suggestionPanelEl.style.display = "none";
      try {
        await fetch("/api/correction-feedback", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({decision, original, selected}),
        });
      } catch (_) {
        // Feedback is best-effort and should never block writing.
      }
    }

    /**
     * Render up to 5 candidate predictions.
     * OCR returns optional candidate text/confidence pairs.
     * Clicking any prediction card inserts that candidate into the recognized textarea.
     */
    function renderTop3(top3) {
      const MAX_SLOTS = 5;
      for (let i = 0; i < MAX_SLOTS; i++) {
        const item   = document.getElementById("top3-" + i);
        const charEl = document.getElementById("top3-char-" + i);
        const barEl  = document.getElementById("top3-bar-" + i);
        const pctEl  = document.getElementById("top3-pct-" + i);
        if (!item) continue;  // slot may not exist yet
        if (i < top3.length) {
          const [ch, conf] = top3[i];
          charEl.textContent = ch || "?";
          const pct = Math.min(100, Math.max(0, Math.round((conf || 0) * 100)));
          barEl.style.width  = pct + "%";
          pctEl.textContent  = pct + "%";
          item.classList.remove("top3-hidden");
          // Click: replace the last character in textarea with this prediction.
          item.onclick = () => {
            const val = recognizedEl.value;
            // If recognizedEl already ends with the top-1 char, replace it; otherwise append.
            recognizedEl.value = (val.trimEnd() || "");
            if (recognizedEl.value.length > 0) recognizedEl.value += " ";
            recognizedEl.value += ch;
          };
        } else {
          item.classList.add("top3-hidden");
          item.onclick = null;
        }
      }
    }

    function clearInk() {
      strokes = [];
      currentStroke = [];
      lastRecognitionResult = null;

      // Clear at device-pixel scale (identity transform), then re-apply DPR.
      const ratio = window.devicePixelRatio || 1;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      applyDrawingStyle();

      confidenceEl.textContent = "\u2014";
      confidenceEl.classList.remove("conf-high", "conf-med", "conf-low");
      correctionConfidenceEl.textContent = "\u2014";
      recognizedRawEl.textContent = "No recognition yet.";
      rawTextEl.textContent = "No recognition yet.";
      statusEl.textContent = "Ink cleared";
      pdStrokes.textContent = "0";
      renderCorrections([]);
      renderSuggestion(null);
      // Clear top-3 panel.
      renderTop3([]);
    }

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
    document.getElementById("clearInk").addEventListener("click", clearInk);
    document.getElementById("space").addEventListener("click", () => { recognizedEl.value += " "; });
    document.getElementById("backspace").addEventListener("click", () => {
      recognizedEl.value = recognizedEl.value.slice(0, -1);
    });
    document.getElementById("clearText").addEventListener("click", () => {
      recognizedEl.value = "";
    });
    acceptSuggestionEl.addEventListener("click", () => sendCorrectionFeedback("accept"));
    rejectSuggestionEl.addEventListener("click", () => sendCorrectionFeedback("reject"));
    keepOriginalEl.addEventListener("click", () => sendCorrectionFeedback("keep_original"));

    // Initial canvas setup.
    resizeCanvas();
    renderCorrections([]);
    renderSuggestion(null);
    renderTop3([]);  // initialise top-3 panel in hidden state
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
        self.recognizer = recognizer or TrOCRHandwritingRecognizer()
        self.pipeline = WritingPipeline(
            recognizer=self.recognizer,
            corrector=corrector or ContextualCorrector.from_settings(self.settings),
            settings=self.settings,
        )

    def warm_up_async(self) -> None:
        if not self.settings.preload_ocr_model:
            return
        thread = threading.Thread(target=self._warm_up_ocr_model, daemon=True)
        thread.start()

    def _warm_up_ocr_model(self) -> None:
        ensure_loaded = getattr(self.recognizer, "_ensure_loaded", None)
        if not callable(ensure_loaded):
            return
        try:
            ensure_loaded()
            logger.info("OCR model warmed up")
        except Exception as exc:
            logger.warning("OCR model warm-up failed; first recognition may retry: %s", exc)

    def recognize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        stroke_groups = stroke_groups_from_payload(payload)
        # Accept legacy mode values, but recognition always uses OCR.
        mode = payload.get("mode", "ocr")
        if mode not in ("auto", "character", "word", "ocr"):
            mode = "ocr"
        result = self.recognizer.recognize_stroke_groups(stroke_groups, mode=mode)
        pipeline_result = self.pipeline.process_recognition(result)
        # Parse top3 from metadata (list of [char, confidence] pairs).
        top3_raw = result.metadata.get("top3", "[]")
        try:
            top3 = json.loads(top3_raw)
        except (json.JSONDecodeError, TypeError):
            top3 = []
        correction = pipeline_result.correction
        suggestion_text = ""
        if correction.status == "suggestion" and correction.corrections:
            suggestion_text = correction.corrections[0].corrected
        return {
            "text": correction.corrected_text,
            "recognized_text": result.text,
            "corrected_text": correction.corrected_text,
            "suggestion_text": suggestion_text,
            "confidence": result.confidence,
            "correction_confidence": correction.confidence,
            "correction_status": correction.status,
            "correction_method": correction.method,
            "alternatives": [
                {
                    "text": alt.text,
                    "confidence": alt.confidence,
                    "reason": alt.reason,
                }
                for alt in correction.alternatives
            ],
            "corrections": corrections_payload(correction),
            "needs_review": pipeline_result.needs_review,
            "review_reason": pipeline_result.review_reason,
            "status": status_message(pipeline_result),
            "metadata": result.metadata,
            "top3": top3,
            "mode": result.metadata.get("mode", mode),
        }

    def record_correction_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        decision = str(payload.get("decision", "")).strip().lower()
        original = str(payload.get("original", "")).strip()
        selected = str(payload.get("selected", "")).strip()
        if decision not in {"accept", "reject", "keep_original"}:
            raise ValueError("decision must be accept, reject, or keep_original")
        if decision == "accept" and original and selected:
            recorder = getattr(self.pipeline.corrector, "record_feedback", None)
            if callable(recorder):
                recorder(original, selected)
        return {"ok": True, "decision": decision}


def corrections_payload(result: CorrectionResult) -> List[Dict[str, Any]]:
    return [
        {
            "original": correction.original,
            "corrected": correction.corrected,
            "confidence": correction.confidence,
            "reason": correction.reason,
            "edit_distance": correction.edit_distance,
            "automatic": correction.automatic,
            "status": correction.status,
            "alternatives": [
                {
                    "text": alt.text,
                    "confidence": alt.confidence,
                    "reason": alt.reason,
                }
                for alt in correction.alternatives
            ],
        }
        for correction in result.corrections
    ]


def status_message(result: PipelineResult) -> str:
    if result.needs_review:
        return "Recognized and corrected handwriting; review recommended."
    if result.correction.status == "suggestion":
        return "Recognized handwriting with correction suggestions."
    if result.correction.changed:
        return "Recognized and corrected handwriting."
    return "Recognized handwriting."


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
            if self.path == "/api/correction-feedback":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    result = service.record_correction_feedback(payload)
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, result)
                return

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
    service.warm_up_async()
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
