"""Tkinter interface for handwriting capture, recognition, and correction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Any, List, Optional, Sequence

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import (
    CorrectionResult,
    PipelineResult,
    RecognitionResult,
    StrokePoint,
)
from assistive_writing_pad.correction.factory import corrector_from_settings
from assistive_writing_pad.pipeline import WritingPipeline
from assistive_writing_pad.recognition.trocr import RecognitionUnavailable, TrOCRHandwritingRecognizer


INK = "#1f2937"
PAPER = "#ffffff"
BG = "#f3f4f6"
ACCENT = "#2563eb"
MUTED = "#6b7280"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextAlternative:
    text: str
    confidence: float


def flatten_strokes(stroke_groups: Sequence[Sequence[StrokePoint]]) -> List[StrokePoint]:
    return [point for stroke in stroke_groups for point in stroke]


def collect_text_alternatives(
    recognition: RecognitionResult,
    correction: CorrectionResult,
    *,
    limit: int = 5,
) -> List[TextAlternative]:
    alternatives: List[TextAlternative] = []
    seen: set[str] = set()

    add_text_alternative(alternatives, seen, recognition.text, recognition.confidence)
    for item in parse_json_list(recognition.metadata.get("top3")):
        if isinstance(item, (list, tuple)) and item:
            text = item[0]
            confidence = item[1] if len(item) > 1 else 0.0
            add_text_alternative(alternatives, seen, text, confidence)
        elif isinstance(item, dict):
            add_text_alternative(
                alternatives,
                seen,
                item.get("text"),
                item.get("confidence", 0.0),
            )

    for stage in parse_json_list(correction.metadata.get("stages")):
        if not isinstance(stage, dict):
            continue
        for item in stage.get("alternatives", []):
            if isinstance(item, dict):
                add_text_alternative(
                    alternatives,
                    seen,
                    item.get("text"),
                    item.get("confidence", 0.0),
                )

    return alternatives[:limit]


def add_text_alternative(
    alternatives: List[TextAlternative],
    seen: set[str],
    text: object,
    confidence: object,
) -> None:
    value = str(text or "").strip()
    if not value:
        return
    key = value.lower()
    if key in seen:
        return
    seen.add(key)
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        score = 0.0
    alternatives.append(TextAlternative(text=value, confidence=score))


def parse_json_list(value: object) -> List[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def format_correction_lines(result: CorrectionResult) -> str:
    if not result.corrections:
        return "No changes."
    lines = []
    for correction in result.corrections:
        before = correction.original or "[insert]"
        after = correction.corrected or "[remove]"
        confidence = round(correction.confidence * 100)
        lines.append(f"{before} -> {after}  {confidence}%")
    return "\n".join(lines)


class HandwritingApp:
    """Writing pad UI backed by pretrained OCR and model correction."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        recognizer: Optional[Any] = None,
        corrector: Optional[Any] = None,
        settings: Optional[RuntimeSettings] = None,
    ) -> None:
        self.root = root
        self.root.title("Assistive Writing Pad")
        self.root.geometry("1100x680")
        self.root.minsize(940, 560)

        self.settings = settings or RuntimeSettings.from_env()
        self.recognizer = recognizer or TrOCRHandwritingRecognizer()
        self.corrector = corrector or corrector_from_settings(self.settings)
        self.pipeline = WritingPipeline(
            recognizer=self.recognizer,
            corrector=self.corrector,
            settings=self.settings,
        )

        self.current_stroke: List[StrokePoint] = []
        self.strokes: List[List[StrokePoint]] = []
        self.stroke_started_at: Optional[float] = None
        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None
        self.recognition_job: Optional[str] = None
        self.recognition_sequence = 0
        self.recognition_lock = threading.Lock()
        self.alternatives: List[TextAlternative] = []
        self.alternative_index = 0
        self.rendering_alternatives = False

        self.recognized_text = tk.StringVar(value="")
        self.status = tk.StringVar(value="Ready")
        self.recognition_confidence = tk.StringVar(value="Recognition: -")
        self.correction_confidence = tk.StringVar(value="Correction: -")

        self._configure_style()
        self._build_layout()
        self._bind_canvas()
        self._warm_up_async()

    def run(self) -> None:
        self.root.mainloop()

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PAPER)
        style.configure("TLabel", background=BG, foreground=INK, font=("TkDefaultFont", 11))
        style.configure("Panel.TLabel", background=PAPER, foreground=INK)
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

        ttk.Label(canvas_panel, text="Write", style="Title.TLabel").grid(
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
        self.recognize_button = ttk.Button(
            controls,
            text="Recognize",
            command=self.recognize_all_ink,
            style="Accent.TButton",
        )
        self.recognize_button.pack(side=tk.LEFT)
        self.try_next_button = ttk.Button(controls, text="Try Next", command=self.try_next)
        self.try_next_button.pack(side=tk.LEFT, padx=8)
        self.clear_button = ttk.Button(controls, text="Clear Screen", command=self.clear_screen)
        self.clear_button.pack(side=tk.LEFT)
        self.try_next_button.state(["disabled"])

        result_panel = ttk.Frame(body, style="Panel.TFrame", padding=14)
        result_panel.grid(row=0, column=1, sticky="nsew")
        result_panel.columnconfigure(0, weight=1)
        result_panel.rowconfigure(2, weight=1)

        ttk.Label(result_panel, text="Result", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        confidence_row = ttk.Frame(result_panel, style="Panel.TFrame")
        confidence_row.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        ttk.Label(confidence_row, textvariable=self.recognition_confidence, style="Panel.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Label(confidence_row, textvariable=self.correction_confidence, style="Panel.TLabel").pack(
            side=tk.LEFT,
            padx=(14, 0),
        )

        self.text_box = tk.Text(
            result_panel,
            wrap=tk.WORD,
            height=9,
            font=("TkDefaultFont", 24),
            relief=tk.FLAT,
            background="#f9fafb",
            foreground=INK,
            padx=14,
            pady=14,
        )
        self.text_box.grid(row=2, column=0, sticky="nsew")

        ttk.Label(result_panel, text="Corrections", style="Panel.TLabel").grid(
            row=3, column=0, sticky="w", pady=(12, 4)
        )
        self.corrections_box = tk.Text(
            result_panel,
            wrap=tk.WORD,
            height=4,
            font=("TkDefaultFont", 12),
            relief=tk.FLAT,
            background="#f9fafb",
            foreground=INK,
            padx=10,
            pady=8,
        )
        self.corrections_box.grid(row=4, column=0, sticky="ew")

        ttk.Label(result_panel, text="Alternatives", style="Panel.TLabel").grid(
            row=5, column=0, sticky="w", pady=(12, 4)
        )
        self.alternatives_list = tk.Listbox(
            result_panel,
            height=5,
            activestyle="dotbox",
            background="#f9fafb",
            foreground=INK,
            highlightthickness=1,
            highlightbackground="#d1d5db",
            relief=tk.FLAT,
        )
        self.alternatives_list.grid(row=6, column=0, sticky="ew")
        self.alternatives_list.bind("<<ListboxSelect>>", self._select_alternative)

        self._sync_text_box()
        self._render_corrections(CorrectionResult(original_text="", corrected_text=""))
        self._render_alternatives()

    def _bind_canvas(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._continue_stroke)
        self.canvas.bind("<ButtonRelease-1>", self._finish_stroke)

    def _warm_up_async(self) -> None:
        if self.settings.preload_ocr_model:
            threading.Thread(target=self._warm_up_ocr_model, daemon=True).start()
        if self.settings.preload_correction_models:
            threading.Thread(target=self._warm_up_correction_models, daemon=True).start()

    def _warm_up_ocr_model(self) -> None:
        ensure_loaded = getattr(self.recognizer, "_ensure_loaded", None)
        if not callable(ensure_loaded):
            return
        try:
            ensure_loaded()
            logger.info("Tk OCR model warmed up")
        except Exception as exc:
            logger.warning("Tk OCR warm-up failed; first recognition may retry: %s", exc)

    def _warm_up_correction_models(self) -> None:
        warm_up = getattr(self.corrector, "warm_up", None)
        if not callable(warm_up):
            return
        try:
            warm_up()
            logger.info("Tk correction models warmed up")
        except Exception as exc:
            logger.warning("Tk correction warm-up failed; first correction may retry: %s", exc)

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
        if self.current_stroke:
            self.strokes.append(list(self.current_stroke))
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
        if not self.strokes:
            self.status.set("Write on the pad first.")
            self.recognition_confidence.set("Recognition: -")
            self.correction_confidence.set("Correction: -")
            return

        sequence = self._next_sequence()
        stroke_snapshot = tuple(tuple(stroke) for stroke in self.strokes if stroke)
        self.status.set("Recognizing...")
        self._set_busy(True)
        threading.Thread(
            target=self._recognize_worker,
            args=(sequence, stroke_snapshot),
            daemon=True,
        ).start()

    def _recognize_worker(
        self,
        sequence: int,
        stroke_snapshot: Sequence[Sequence[StrokePoint]],
    ) -> None:
        try:
            with self.recognition_lock:
                recognition = self._recognize_stroke_groups(stroke_snapshot)
                pipeline_result = self.pipeline.process_recognition(recognition)
                alternatives = collect_text_alternatives(
                    recognition,
                    pipeline_result.correction,
                    limit=self.settings.max_correction_candidates,
                )
        except RecognitionUnavailable as exc:
            self.root.after(0, self._apply_error, sequence, str(exc))
            return
        except Exception as exc:
            logger.exception("Tk recognition failed")
            self.root.after(0, self._apply_error, sequence, f"Recognition failed: {exc}")
            return

        self.root.after(0, self._apply_pipeline_result, sequence, pipeline_result, alternatives)

    def _recognize_stroke_groups(
        self,
        stroke_snapshot: Sequence[Sequence[StrokePoint]],
    ) -> RecognitionResult:
        recognize_groups = getattr(self.recognizer, "recognize_stroke_groups", None)
        if callable(recognize_groups):
            return recognize_groups(stroke_snapshot, mode="ocr")
        return self.recognizer.recognize(flatten_strokes(stroke_snapshot))

    def _apply_pipeline_result(
        self,
        sequence: int,
        result: PipelineResult,
        alternatives: Sequence[TextAlternative],
    ) -> None:
        if sequence != self.recognition_sequence:
            return
        self.alternatives = list(alternatives)
        self.alternative_index = 0
        self.recognized_text.set(result.correction.corrected_text)
        self.recognition_confidence.set(
            f"Recognition: {round(result.recognition.confidence * 100)}%"
        )
        self.correction_confidence.set(f"Correction: {round(result.correction.confidence * 100)}%")
        self.status.set("Review handwriting." if result.needs_review else "Recognized.")
        self._sync_text_box()
        self._render_corrections(result.correction)
        self._render_alternatives()
        self._set_busy(False)

    def _apply_error(self, sequence: int, message: str) -> None:
        if sequence != self.recognition_sequence:
            return
        self.status.set(message)
        self._set_busy(False)

    def try_next(self) -> None:
        if len(self.alternatives) < 2:
            self.status.set("No other option yet.")
            return
        next_index = (self.alternative_index + 1) % len(self.alternatives)
        self._use_alternative(next_index)

    def _select_alternative(self, _event: tk.Event) -> None:
        if self.rendering_alternatives:
            return
        selected = self.alternatives_list.curselection()
        if not selected:
            return
        self._use_alternative(int(selected[0]))

    def _use_alternative(self, index: int) -> None:
        if not 0 <= index < len(self.alternatives):
            return
        alternative = self.alternatives[index]
        self.alternative_index = index
        self._render_alternatives()
        sequence = self._next_sequence()
        self.status.set(f"Correcting option {index + 1}...")
        self._set_busy(True)
        threading.Thread(
            target=self._correct_alternative_worker,
            args=(sequence, alternative),
            daemon=True,
        ).start()

    def _correct_alternative_worker(self, sequence: int, alternative: TextAlternative) -> None:
        try:
            with self.recognition_lock:
                correction = self.corrector.correct(alternative.text)
        except Exception as exc:
            logger.exception("Tk alternative correction failed")
            self.root.after(0, self._apply_error, sequence, f"Correction failed: {exc}")
            return
        self.root.after(0, self._apply_alternative_result, sequence, alternative, correction)

    def _apply_alternative_result(
        self,
        sequence: int,
        alternative: TextAlternative,
        correction: CorrectionResult,
    ) -> None:
        if sequence != self.recognition_sequence:
            return
        self.recognized_text.set(correction.corrected_text)
        self.recognition_confidence.set(f"Recognition: {round(alternative.confidence * 100)}%")
        self.correction_confidence.set(f"Correction: {round(correction.confidence * 100)}%")
        self.status.set(f"Using option {self.alternative_index + 1} of {len(self.alternatives)}.")
        self._sync_text_box()
        self._render_corrections(correction)
        self._render_alternatives()
        self._set_busy(False)

    def clear_screen(self) -> None:
        if self.recognition_job is not None:
            self.root.after_cancel(self.recognition_job)
            self.recognition_job = None
        self.recognition_sequence += 1
        self.canvas.delete("all")
        self.current_stroke.clear()
        self.strokes.clear()
        self.alternatives = []
        self.alternative_index = 0
        self.recognized_text.set("")
        self.status.set("Ready")
        self.recognition_confidence.set("Recognition: -")
        self.correction_confidence.set("Correction: -")
        self._sync_text_box()
        self._render_corrections(CorrectionResult(original_text="", corrected_text=""))
        self._render_alternatives()
        self._set_busy(False)

    def _next_sequence(self) -> int:
        self.recognition_sequence += 1
        return self.recognition_sequence

    def _set_busy(self, busy: bool) -> None:
        self.recognize_button.state(["disabled"] if busy else ["!disabled"])
        can_try_next = (not busy) and len(self.alternatives) > 1
        self.try_next_button.state(["!disabled"] if can_try_next else ["disabled"])

    def _sync_text_box(self) -> None:
        self._replace_text(self.text_box, self.recognized_text.get())

    def _render_corrections(self, correction: CorrectionResult) -> None:
        self._replace_text(self.corrections_box, format_correction_lines(correction))

    def _render_alternatives(self) -> None:
        self.rendering_alternatives = True
        try:
            self.alternatives_list.delete(0, tk.END)
            for index, alternative in enumerate(self.alternatives):
                confidence = round(alternative.confidence * 100)
                self.alternatives_list.insert(
                    tk.END,
                    f"{index + 1}. {alternative.text}  {confidence}%",
                )
            if self.alternatives:
                self.alternatives_list.selection_set(self.alternative_index)
                self.alternatives_list.activate(self.alternative_index)
        finally:
            self.rendering_alternatives = False
        self._set_busy(False)

    def _replace_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    HandwritingApp(root).run()


if __name__ == "__main__":
    main()
