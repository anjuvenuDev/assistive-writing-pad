"""Preprocessing pipeline from tablet strokes to model-ready arrays."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.preprocessing.image_ops import normalize_unit, pad_to_square
from assistive_writing_pad.preprocessing.rasterize import RasterizerConfig, rasterize_strokes

logger = logging.getLogger("assistive_writing_pad.preprocessing.pipeline")


def _debug_preprocessing_enabled() -> bool:
    raw = os.environ.get("AWP_DEBUG_PREPROCESSING", "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PreprocessingConfig:
    rasterizer: RasterizerConfig = RasterizerConfig()
    output_size: Tuple[int, int] = (28, 28)


@dataclass(frozen=True)
class PreprocessedImage:
    image: np.ndarray
    original_point_count: int
    output_size: Tuple[int, int]


class StrokePreprocessor:
    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()) -> None:
        self.config = config

    def _com(self, img: np.ndarray) -> Tuple[float, float]:
        total_mass = img.sum()
        if total_mass == 0:
            return img.shape[0] / 2.0, img.shape[1] / 2.0
        h, w = img.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        y_com = (y_coords * img).sum() / total_mass
        x_com = (x_coords * img).sum() / total_mass
        return float(y_com), float(x_com)

    def preprocess(self, points: Sequence[StrokePoint]) -> PreprocessedImage:
        import cv2
        from PIL import Image

        debug_enabled = _debug_preprocessing_enabled()
        stages_data = []  # List of tuples: (name, img, stats)

        # Helper to compute stats for a stage
        def get_stage_stats(img: np.ndarray) -> dict:
            h, w = img.shape
            fg_mask = img > 0.05
            fg_count = fg_mask.sum()
            rows, cols = np.where(fg_mask)
            if rows.size > 0 and cols.size > 0:
                bbox = (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
                aspect_ratio = float((cols.max() - cols.min() + 1) / (rows.max() - rows.min() + 1))
            else:
                bbox = None
                aspect_ratio = 1.0

            y_com, x_com = self._com(img)
            return {
                "width": w,
                "height": h,
                "fg_count": int(fg_count),
                "bbox": bbox,
                "aspect_ratio": aspect_ratio,
                "com": (float(x_com), float(y_com))
            }

        def add_debug_stage(name: str, img: np.ndarray) -> None:
            if debug_enabled:
                stages_data.append((name, img, get_stage_stats(img)))

        # 0. Stage 00: Canvas
        raster = rasterize_strokes(points, self.config.rasterizer)
        add_debug_stage("00_canvas", raster)

        # 1. Stage 01: Binary
        binary = (raster > 0.05).astype(np.float32)
        add_debug_stage("01_binary", binary)

        # 2. Stage 02: Blurred
        # Apply a mild Gaussian blur
        blurred = cv2.GaussianBlur(binary, (3, 3), 0)
        add_debug_stage("02_blurred", blurred)

        # 3. Stage 03: Bounding Box (Cropped)
        rows, cols = np.where(blurred > 0.05)
        if rows.size == 0 or cols.size == 0:
            cropped = np.zeros((1, 1), dtype=np.float32)
            min_r, max_r, min_c, max_c = 0, 0, 0, 0
        else:
            min_r, max_r = rows.min(), rows.max()
            min_c, max_c = cols.min(), cols.max()
            cropped = blurred[min_r : max_r + 1, min_c : max_c + 1].copy()
        add_debug_stage("03_bounding_box", cropped)

        # 4. Stage 04: Square Canvas
        squared = pad_to_square(cropped)
        add_debug_stage("04_square_canvas", squared)

        # 5. Stage 05: Before Resize
        before_resize = squared.copy()
        add_debug_stage("05_before_resize", before_resize)

        # 6. Stage 06: After Resize
        h_s, w_s = before_resize.shape
        if h_s > w_s:
            h_new = 20
            w_new = int(round(20 * w_s / h_s))
            w_new = max(1, w_new)
        else:
            w_new = 20
            h_new = int(round(20 * h_s / w_s))
            h_new = max(1, h_new)

        pil_cropped = Image.fromarray(before_resize)
        pil_resized = pil_cropped.resize((w_new, h_new), Image.Resampling.BILINEAR)
        resized_20 = np.array(pil_resized).astype(np.float32)
        add_debug_stage("06_after_resize", resized_20)

        # 7. Stage 07: After Centering
        canvas_28 = np.zeros((28, 28), dtype=np.float32)
        total_mass = resized_20.sum()
        if total_mass > 0:
            y_coords, x_coords = np.mgrid[0:h_new, 0:w_new]
            y_com = (y_coords * resized_20).sum() / total_mass
            x_com = (x_coords * resized_20).sum() / total_mass
        else:
            y_com = h_new / 2.0
            x_com = w_new / 2.0

        y_top = int(round(13.5 - y_com))
        x_left = int(round(13.5 - x_com))

        y_start = max(0, y_top)
        y_end = min(28, y_top + h_new)
        x_start = max(0, x_left)
        x_end = min(28, x_left + w_new)

        crop_y_start = max(0, -y_top)
        crop_y_end = crop_y_start + (y_end - y_start)
        crop_x_start = max(0, -x_left)
        crop_x_end = crop_x_start + (x_end - x_start)

        canvas_28[y_start:y_end, x_start:x_end] = resized_20[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        add_debug_stage("07_after_centering", canvas_28)

        # 8. Stage 08: After Rotation
        normalized = normalize_unit(canvas_28)
        rotated = np.rot90(normalized, k=2).copy()
        add_debug_stage("08_after_rotation", rotated)

        # 9. Stage 09: Final Tensor
        final_tensor = rotated.copy()
        add_debug_stage("09_final_tensor", final_tensor)

        if debug_enabled:
            import time

            debug_dir = Path("data/debug/stages")
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            logger.info("Preprocessing debug run: %s", ts)
            for idx in range(len(stages_data)):
                name, img, stats = stages_data[idx]

                filename = f"{name}_{ts}.png"
                Image.fromarray((img * 255.0).clip(0, 255).astype(np.uint8)).save(
                    debug_dir / filename
                )

                mse_str = "N/A"
                if idx < len(stages_data) - 1:
                    next_img = stages_data[idx + 1][1]
                    if img.shape != next_img.shape:
                        img_resized = cv2.resize(
                            img,
                            (next_img.shape[1], next_img.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        img_resized = img
                    mse = float(((img_resized - next_img) ** 2).mean())
                    mse_str = f"{mse:.6f}"

                logger.info("Stage %02d: %s", idx, name)
                logger.info("  Width x Height      : %sx%s", stats["width"], stats["height"])
                logger.info("  Foreground Pixels   : %s", stats["fg_count"])
                logger.info("  Bounding Box        : %s", stats["bbox"])
                logger.info("  Aspect Ratio        : %.4f", stats["aspect_ratio"])
                logger.info("  Center of Mass      : (%.2f, %.2f)", stats["com"][0], stats["com"][1])
                logger.info("  MSE (this -> next)  : %s", mse_str)

                if idx < len(stages_data) - 1:
                    next_stats = stages_data[idx + 1][2]
                    fg_ratio = next_stats["fg_count"] / max(stats["fg_count"], 1)
                    ar_diff = abs(next_stats["aspect_ratio"] - stats["aspect_ratio"])
                    if fg_ratio < 0.1 and stats["fg_count"] > 10:
                        logger.info(
                            "  Debug warning: foreground pixels dropped sharply in next stage "
                            "(ratio %.4f)",
                            fg_ratio,
                        )
                    if ar_diff > 0.5:
                        logger.info(
                            "  Debug warning: aspect ratio changed sharply in next stage "
                            "(diff %.4f)",
                            ar_diff,
                        )

        return PreprocessedImage(
            image=final_tensor.astype(np.float32),
            original_point_count=len(points),
            output_size=self.config.output_size,
        )
