"""Preprocessing pipeline from tablet strokes to model-ready arrays."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.preprocessing.image_ops import (
    crop_to_content,
    normalize_unit,
    pad_to_square,
    resize_nearest,
)
from assistive_writing_pad.preprocessing.rasterize import RasterizerConfig, rasterize_strokes

logger = logging.getLogger("assistive_writing_pad.preprocessing.pipeline")


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
        import time
        import cv2
        from PIL import Image, ImageFilter

        ts = int(time.time() * 1000)
        debug_dir = Path("data/debug/stages")
        debug_dir.mkdir(parents=True, exist_ok=True)

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

        stages_data = []  # List of tuples: (name, img, stats)

        # 0. Stage 00: Canvas
        raster = rasterize_strokes(points, self.config.rasterizer)
        stages_data.append(("00_canvas", raster, get_stage_stats(raster)))

        # 1. Stage 01: Binary
        binary = (raster > 0.05).astype(np.float32)
        stages_data.append(("01_binary", binary, get_stage_stats(binary)))

        # 2. Stage 02: Blurred
        # Apply a mild Gaussian blur
        blurred = cv2.GaussianBlur(binary, (3, 3), 0)
        stages_data.append(("02_blurred", blurred, get_stage_stats(blurred)))

        # 3. Stage 03: Bounding Box (Cropped)
        rows, cols = np.where(blurred > 0.05)
        if rows.size == 0 or cols.size == 0:
            cropped = np.zeros((1, 1), dtype=np.float32)
            min_r, max_r, min_c, max_c = 0, 0, 0, 0
        else:
            min_r, max_r = rows.min(), rows.max()
            min_c, max_c = cols.min(), cols.max()
            cropped = blurred[min_r : max_r + 1, min_c : max_c + 1].copy()
        stages_data.append(("03_bounding_box", cropped, get_stage_stats(cropped)))

        # 4. Stage 04: Square Canvas
        squared = pad_to_square(cropped)
        stages_data.append(("04_square_canvas", squared, get_stage_stats(squared)))

        # 5. Stage 05: Before Resize
        before_resize = squared.copy()
        stages_data.append(("05_before_resize", before_resize, get_stage_stats(before_resize)))

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
        stages_data.append(("06_after_resize", resized_20, get_stage_stats(resized_20)))

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
        stages_data.append(("07_after_centering", canvas_28, get_stage_stats(canvas_28)))

        # 8. Stage 08: After Rotation
        normalized = normalize_unit(canvas_28)
        rotated = np.rot90(normalized, k=2).copy()
        stages_data.append(("08_after_rotation", rotated, get_stage_stats(rotated)))

        # 9. Stage 09: Final Tensor
        final_tensor = rotated.copy()
        stages_data.append(("09_final_tensor", final_tensor, get_stage_stats(final_tensor)))

        # Save images and print stats
        logger.info(f"\n================ PREPROCESSING DEBUG RUN: {ts} ================")
        for idx in range(len(stages_data)):
            name, img, stats = stages_data[idx]

            # Save stage image (Do NOT overwrite)
            filename = f"{name}_{ts}.png"
            Image.fromarray((img * 255.0).clip(0, 255).astype(np.uint8)).save(debug_dir / filename)

            # Compute MSE with next stage
            mse_str = "N/A"
            if idx < len(stages_data) - 1:
                next_img = stages_data[idx+1][1]
                if img.shape != next_img.shape:
                    img_resized = cv2.resize(img, (next_img.shape[1], next_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    img_resized = img
                mse = float(((img_resized - next_img) ** 2).mean())
                mse_str = f"{mse:.6f}"

            logger.info(f"Stage {idx:02d}: {name}")
            logger.info(f"  Width x Height      : {stats['width']}x{stats['height']}")
            logger.info(f"  Foreground Pixels   : {stats['fg_count']}")
            logger.info(f"  Bounding Box        : {stats['bbox']}")
            logger.info(f"  Aspect Ratio        : {stats['aspect_ratio']:.4f}")
            logger.info(f"  Center of Mass      : ({stats['com'][0]:.2f}, {stats['com'][1]:.2f})")
            logger.info(f"  MSE (this -> next)  : {mse_str}")

            # Highlight dramatic structural changes
            if idx < len(stages_data) - 1:
                next_stats = stages_data[idx+1][2]
                fg_ratio = next_stats['fg_count'] / max(stats['fg_count'], 1)
                ar_diff = abs(next_stats['aspect_ratio'] - stats['aspect_ratio'])
                if fg_ratio < 0.1 and stats['fg_count'] > 10:
                    logger.info(f"  ⚠️ DETECTED CORRUPTION: Dramatic reduction in foreground pixels in next stage! (Ratio: {fg_ratio:.4f})")
                if ar_diff > 0.5:
                    logger.info(f"  ⚠️ DETECTED CORRUPTION: Dramatic change in aspect ratio in next stage! (Diff: {ar_diff:.4f})")
        logger.info("================================================================\n")

        return PreprocessedImage(
            image=final_tensor.astype(np.float32),
            original_point_count=len(points),
            output_size=self.config.output_size,
        )
