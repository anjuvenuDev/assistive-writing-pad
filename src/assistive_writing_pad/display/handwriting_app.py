"""Tkinter interface for handwriting capture and live recognition."""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import List, Optional

from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.recognition.trocr import RecognitionUnavailable, TrOCRHandwritingRecognizer
from assistive_writing_pad.recognition.template import (
    DEFAULT_TEMPLATE_PATH,
    TemplateGlyphRecognizer,
    count_samples_by_label,
)


INK = "#1f2937"
PAPER = "#ffffff"
BG = "#f3f4f6"
ACCENT = "#2563eb"
MUTED = "#6b7280"


class HandwritingApp:
    """Writing pad UI with pretrained recognition and local fallback learning."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Assistive Writing Pad")
        self.root.geometry("1100x680")
        self.root.minsize(940, 560)

        self.pretrained_recognizer = TrOCRHandwritingRecognizer()
        self.template_recognizer = TemplateGlyphRecognizer()
        self.current_stroke: List[StrokePoint] = []
        self.last_stroke: List[StrokePoint] = []
        self.all_points: List[StrokePoint] = []
        self.stroke_started_at: Optional[float] = None
        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None
        self.recognition_job: Optional[str] = None
        self.recognized_text = tk.StringVar(value="")
        self.status = tk.StringVar(value="Pretrained OCR ready to load on first recognition.")
        self.confidence = tk.StringVar(value="Confidence: -")

        self._configure_style()
        self._build_layout()
        self._bind_canvas()

    def run(self) -> None:
        self.root.mainloop()

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PAPER)
        style.configure("TLabel", background=BG, foreground=INK, font=("TkDefaultFont", 11))
        style.configure("Title.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("TButton", font=("TkDefaultFont", 10), padding=(12, 8))
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff")

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(header, text="Assistive Writing Pad", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.status, style="Muted.TLabel").pack(side=tk.RIGHT)

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        canvas_panel = ttk.Frame(body, style="Panel.TFrame", padding=14)
        canvas_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        canvas_panel.rowconfigure(1, weight=1)
        canvas_panel.columnconfigure(0, weight=1)

        ttk.Label(canvas_panel, text="Handwriting", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )
        self.canvas = tk.Canvas(
            canvas_panel,
            background=PAPER,
            highlightthickness=1,
            highlightbackground="#d1d5db",
            cursor="pencil",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew")

        controls = ttk.Frame(canvas_panel, style="Panel.TFrame")
        controls.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(controls, text="Clear Ink", command=self.clear_ink).pack(side=tk.LEFT)
        ttk.Button(controls, text="Recognize Now", command=self.recognize_all_ink).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Button(controls, text="Space", command=self.add_space).pack(side=tk.LEFT)
        ttk.Button(controls, text="Backspace", command=self.backspace).pack(side=tk.LEFT, padx=8)
        ttk.Button(controls, text="Clear Text", command=self.clear_text).pack(side=tk.LEFT, padx=8)

        result_panel = ttk.Frame(body, style="Panel.TFrame", padding=14)
        result_panel.grid(row=0, column=1, sticky="nsew")
        result_panel.columnconfigure(0, weight=1)
        result_panel.rowconfigure(2, weight=1)

        ttk.Label(result_panel, text="Recognized Text", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(result_panel, textvariable=self.confidence, style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 10)
        )

        self.text_box = tk.Text(
            result_panel,
            wrap=tk.WORD,
            height=10,
            font=("TkDefaultFont", 24),
            relief=tk.FLAT,
            background="#f9fafb",
            foreground=INK,
            padx=14,
            pady=14,
        )
        self.text_box.grid(row=2, column=0, sticky="nsew")
        self._sync_text_box()

        fallback = ttk.Frame(result_panel, style="Panel.TFrame")
        fallback.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        fallback.columnconfigure(1, weight=1)
        ttk.Label(fallback, text="Fallback label").grid(row=0, column=0, sticky="w")
        self.label_entry = ttk.Entry(fallback)
        self.label_entry.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(fallback, text="Save Template", command=self.save_sample).grid(row=0, column=2)

    def _bind_canvas(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._continue_stroke)
        self.canvas.bind("<ButtonRelease-1>", self._finish_stroke)

    def _start_stroke(self, event: tk.Event) -> None:
        self.current_stroke = []
        self.stroke_started_at = time.monotonic()
        self.last_x = float(event.x)
        self.last_y = float(event.y)
        self._add_point(event)

    def _continue_stroke(self, event: tk.Event) -> None:
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                fill=INK,
                width=4,
                capstyle=tk.ROUND,
                smooth=True,
            )
        self.last_x = float(event.x)
        self.last_y = float(event.y)
        self._add_point(event)

    def _finish_stroke(self, event: tk.Event) -> None:
        self._add_point(event)
        self.last_stroke = list(self.current_stroke)
        self.all_points.extend(self.last_stroke)
        self.current_stroke = []
        self.last_x = None
        self.last_y = None
        self._schedule_recognition()

    def _add_point(self, event: tk.Event) -> None:
        started_at = self.stroke_started_at or time.monotonic()
        timestamp_ms = int((time.monotonic() - started_at) * 1000)
        self.current_stroke.append(
            StrokePoint(x=float(event.x), y=float(event.y), timestamp_ms=timestamp_ms, pressure=1.0)
        )

    def _schedule_recognition(self) -> None:
        if self.recognition_job is not None:
            self.root.after_cancel(self.recognition_job)
        self.recognition_job = self.root.after(700, self.recognize_all_ink)

    def recognize_all_ink(self) -> None:
        self.recognition_job = None
        if not self.all_points:
            self.status.set("Write on the pad to recognize handwriting.")
            self.confidence.set("Confidence: -")
            return

        self.status.set("Recognizing handwriting...")
        self.root.update_idletasks()
        try:
            result = self.pretrained_recognizer.recognize(self.all_points)
        except RecognitionUnavailable as exc:
            self.status.set(str(exc))
            self._recognize_last_stroke_with_template()
            return
        except Exception as exc:
            self.status.set(f"Pretrained OCR failed: {exc}")
            self._recognize_last_stroke_with_template()
            return

        self.confidence.set(f"Confidence: {result.confidence:.2f}")
        self.recognized_text.set(result.text)
        self._sync_text_box()
        if result.text:
            self.status.set("Recognized handwriting with pretrained OCR.")
        else:
            self.status.set("Pretrained OCR returned no text. Try writing larger or clearer.")

    def _recognize_last_stroke_with_template(self) -> None:
        result = self.template_recognizer.recognize(self.last_stroke)
        if result.metadata.get("reason") == "no_templates":
            return
        self.confidence.set(f"Confidence: {result.confidence:.2f}")
        if result.text and result.confidence >= 0.60:
            self.recognized_text.set(self.recognized_text.get() + result.text)
            self._sync_text_box()
            self.status.set(f"Fallback template recognized '{result.text}' from the last stroke.")
        elif result.text:
            self.status.set(f"Low fallback confidence for '{result.text}'.")

    def save_sample(self) -> None:
        label = self.label_entry.get()
        try:
            sample = self.template_recognizer.learn(label, self.last_stroke)
        except ValueError as exc:
            self.status.set(str(exc))
            return

        self.label_entry.delete(0, tk.END)
        self.status.set(f"Saved sample for '{sample.label}'. {self._template_status()}")

    def clear_ink(self) -> None:
        self.canvas.delete("all")
        self.all_points.clear()
        self.current_stroke.clear()
        self.last_stroke.clear()
        self.confidence.set("Confidence: -")

    def add_space(self) -> None:
        self.recognized_text.set(self.recognized_text.get() + " ")
        self._sync_text_box()

    def backspace(self) -> None:
        self.recognized_text.set(self.recognized_text.get()[:-1])
        self._sync_text_box()

    def clear_text(self) -> None:
        self.recognized_text.set("")
        self._sync_text_box()

    def _sync_text_box(self) -> None:
        self.text_box.configure(state=tk.NORMAL)
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert("1.0", self.recognized_text.get())
        self.text_box.configure(state=tk.DISABLED)

    def _template_status(self) -> str:
        counts = count_samples_by_label(self.template_recognizer.store.samples)
        if not counts:
            return f"Templates: 0 ({DEFAULT_TEMPLATE_PATH})"
        summary = ", ".join(f"{label}:{count}" for label, count in counts.items())
        return f"Templates: {len(self.template_recognizer.store.samples)} ({summary})"


def main() -> None:
    root = tk.Tk()
    HandwritingApp(root).run()


if __name__ == "__main__":
    main()
