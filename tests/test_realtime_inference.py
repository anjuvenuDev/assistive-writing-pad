from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
import time

import numpy as np

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import CorrectionResult, RecognitionResult, StrokePoint
from assistive_writing_pad.display.web_app import RecognitionService
from assistive_writing_pad.recognition.trocr import (
    TrOCRHandwritingRecognizer, _OCRCandidate, _decoded_candidates, _draw_dot,
    _looks_like_single_character_input,
)


class CountingRecognizer:
    def __init__(self):
        self.calls = 0
        self.active = 0
        self.peak = 0

    def recognize_stroke_groups(self, groups, mode='auto'):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        time.sleep(0.01)
        self.active -= 1
        return RecognitionResult(text='hello', confidence=0.99)


class Corrector:
    def correct(self, text):
        return CorrectionResult(original_text=text, corrected_text=text, confidence=1)


def payload(x=10):
    return {'strokes': [[{'x': x, 'y': 10}, {'x': 20, 'y': 40}]], 'mode': 'ocr'}


def test_line_ocr_preserves_numbers_punctuation_and_mixed_tokens():
    text = 'I am in 3rd year. Pi4 costs 50!'
    assert _decoded_candidates([text], [0.9])[0].text == text


def test_failed_correction_is_not_cached_as_a_successful_reading():
    class RecoveringCorrector:
        calls = 0
        def correct(self, text):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError('temporarily unavailable')
            return CorrectionResult(original_text=text, corrected_text=text)
    corrector = RecoveringCorrector()
    service = RecognitionService(CountingRecognizer(), corrector, RuntimeSettings())
    first = service.recognize_payload(payload())
    second = service.recognize_payload(payload())
    assert first['needs_review']
    assert not second['needs_review']
    assert corrector.calls == 2


def test_concurrent_identical_snapshots_compute_once_and_return_independent_results():
    recognizer = CountingRecognizer()
    service = RecognitionService(recognizer, Corrector(), RuntimeSettings())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: service.recognize_payload(payload()), range(4)))
    assert recognizer.calls == 1
    results[0]['metadata']['modified'] = True
    assert 'modified' not in service.recognize_payload(payload())['metadata']
    service.recognize_payload(payload(11))
    assert recognizer.calls == 2


def test_different_snapshots_never_run_models_concurrently():
    recognizer = CountingRecognizer()
    service = RecognitionService(recognizer, Corrector(), RuntimeSettings())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda x: service.recognize_payload(payload(x)), range(4)))
    assert recognizer.calls == 4
    assert recognizer.peak == 1


def test_tablet_sampling_density_does_not_change_character_routing():
    sparse = [StrokePoint(x=10, y=0, timestamp_ms=0), StrokePoint(x=10, y=50, timestamp_ms=20)]
    dense = [StrokePoint(x=10, y=i / 20, timestamp_ms=i) for i in range(1001)]
    assert _looks_like_single_character_input([sparse])
    assert _looks_like_single_character_input([dense])


def test_vectorized_dot_preserves_original_pixels_at_edges():
    for x, y in [(0, 0), (15, 9), (4, 5)]:
        image = np.full((10, 16, 3), 255, dtype=np.uint8)
        expected = image.copy()
        for row in range(max(0, y-3), min(10, y+4)):
            for col in range(max(0, x-3), min(16, x+4)):
                expected[row, col] = 0
        _draw_dot(image, x, y, 3)
        np.testing.assert_array_equal(image, expected)


def test_unchanged_line_reuses_candidates_and_changed_ink_invalidates_cache():
    model = TrOCRHandwritingRecognizer()
    calls = []
    model._run_ocr_batch = lambda images: calls.append(len(images)) or [
        [_OCRCandidate('hello', 0.9)] for _ in images
    ]
    groups = [([[
        StrokePoint(x=10, y=10, timestamp_ms=0),
        StrokePoint(x=20, y=50, timestamp_ms=1),
    ]], 'line0')]
    def save(*args, **kwargs):
        pass
    first = model._recognize_group_images(groups, save)
    assert model._recognize_group_images(deepcopy(groups), save) == first
    assert calls == [1]
    groups[0][0][0].append(StrokePoint(x=30, y=40, timestamp_ms=2))
    model._recognize_group_images(groups, save)
    assert calls == [1, 1]
    model.num_beams += 1
    model._recognize_group_images(groups, save)
    assert calls == [1, 1, 1]


def test_blank_batch_input_never_invokes_ocr():
    model = TrOCRHandwritingRecognizer()
    model._run_ocr_batch = lambda images: (_ for _ in ()).throw(AssertionError('blank OCR'))
    assert model._recognize_group_images([([], 'blank')], lambda *a, **k: None) == [[]]


def test_warmup_and_requests_share_model_lock():
    recognizer = CountingRecognizer()
    entered = threading.Event()
    release = threading.Event()
    def warm_up():
        entered.set()
        assert release.wait(2)
    recognizer.warm_up = warm_up
    service = RecognitionService(recognizer, Corrector(), RuntimeSettings())
    with ThreadPoolExecutor(max_workers=2) as pool:
        warm = pool.submit(service._warm_up_ocr_model)
        assert entered.wait(2)
        request = pool.submit(service.recognize_payload, payload())
        assert recognizer.calls == 0
        release.set()
        warm.result()
        request.result()
    assert recognizer.calls == 1
