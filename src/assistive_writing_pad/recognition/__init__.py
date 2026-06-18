"""Handwriting recognition components."""

from assistive_writing_pad.recognition.template import TemplateGlyphRecognizer, TemplateStore
from assistive_writing_pad.recognition.trocr import TrOCRHandwritingRecognizer

__all__ = ["TemplateGlyphRecognizer", "TemplateStore", "TrOCRHandwritingRecognizer"]
