import sys
import os
import gzip
import struct
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add project src/ to path
_SRC = Path("c:/Users/Nikhil/Downloads/IFP/assistive-writing-pad/src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from assistive_writing_pad.recognition.emnist import EMNISTCharacterRecognizer, EMNIST_LABELS, _CACHE_DIR
from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.preprocessing.rasterize import rasterize_strokes, RasterizerConfig

def get_char_strokes(char):
    # Generates strokes for characters in a 120x120 canvas.
    import math
    pts = []
    if char == 'O' or char == '0':
        # Draw a nice circular/oval 'O'
        cx, cy, rx, ry = 60, 60, 35, 45
        for i in range(41):
            angle = 2 * math.pi * i / 40
            pts.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
    return [StrokePoint(x=float(x), y=float(y), timestamp_ms=i*16, pressure=1.0) for i, (x, y) in enumerate(pts)]

def get_real_emnist_sample(label_char):
    # Find a sample of label_char from test ubyte files
    test_img_path  = _CACHE_DIR / "emnist-byclass-test-images-idx3-ubyte.gz"
    test_lbl_path  = _CACHE_DIR / "emnist-byclass-test-labels-idx1-ubyte.gz"
    if not test_img_path.exists():
        return None
    with gzip.open(str(test_img_path), "rb") as f:
        struct.unpack(">4I", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    imgs = data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    
    with gzip.open(str(test_lbl_path), "rb") as f:
        struct.unpack(">2I", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    target_lbl = EMNIST_LABELS.index(label_char)
    for idx, lbl in enumerate(labels):
        if int(lbl) == target_lbl:
            # Apply standard EMNIST orientation fix
            raw_img = imgs[idx]
            fixed_img = np.rot90(raw_img, k=1)
            fixed_img = np.flip(fixed_img, axis=1).copy()
            return fixed_img
    return None

def compute_center_of_mass(img):
    total_mass = img.sum()
    if total_mass == 0:
        return img.shape[0] / 2, img.shape[1] / 2
    
    # Grid coordinates
    h, w = img.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    y_com = (y_coords * img).sum() / total_mass
    x_com = (x_coords * img).sum() / total_mass
    return y_com, x_com

def save_image_helper(img, path):
    # Convert float [0.0, 1.0] to uint8 [0, 255]
    img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)

def debug_pipeline(char='O'):
    print(f"\n==========================================")
    print(f"DEBUG PREPROCESSING FOR CHARACTER: '{char}'")
    print(f"==========================================")
    
    # 1. Generate strokes
    strokes = get_char_strokes(char)
    
    # STAGE 0: Original Canvas Rasterization
    config = RasterizerConfig(line_radius_px=2) # standard thickness
    raster = rasterize_strokes(strokes, config)
    
    debug_dir = Path("data/debug/stages")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original canvas
    save_image_helper(raster, debug_dir / "0_original_canvas.png")
    
    # Print metrics
    r_com_y, r_com_x = compute_center_of_mass(raster)
    fg_pixels = (raster > 0.1).sum()
    print("STAGE 0: Original Canvas")
    print(f"  Width x Height     : {raster.shape[1]}x{raster.shape[0]}")
    print(f"  Center of mass     : ({r_com_x:.2f}, {r_com_y:.2f})")
    print(f"  Foreground pixels  : {fg_pixels}")
    
    # STAGE 1: Bounding Box Cropping
    rows, cols = np.where(raster > 0.0)
    if rows.size == 0 or cols.size == 0:
        print("Empty canvas!")
        return
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    cropped = raster[min_r:max_r+1, min_c:max_c+1].copy()
    
    save_image_helper(cropped, debug_dir / "1_cropped.png")
    c_com_y, c_com_x = compute_center_of_mass(cropped)
    print("\nSTAGE 1: Cropping to content")
    print(f"  Width x Height     : {cropped.shape[1]}x{cropped.shape[0]}")
    print(f"  Bounding box       : y:[{min_r}, {max_r}], x:[{min_c}, {max_c}]")
    print(f"  Center of mass     : ({c_com_x:.2f}, {c_com_y:.2f})")
    print(f"  Foreground pixels  : {(cropped > 0.1).sum()}")
    
    # STAGE 2: Resizing to 20x20 (preserving aspect ratio, bilinear interpolation)
    h_c, w_c = cropped.shape
    if h_c > w_c:
        h_new = 20
        w_new = int(round(20 * w_c / h_c))
        w_new = max(1, w_new)
        scale_factor = 20 / h_c
    else:
        w_new = 20
        h_new = int(round(20 * h_c / w_c))
        h_new = max(1, h_new)
        scale_factor = 20 / w_c
        
    # Bilinear resize using PIL
    pil_cropped = Image.fromarray(cropped)
    pil_resized = pil_cropped.resize((w_new, h_new), Image.Resampling.BILINEAR)
    resized_20 = np.array(pil_resized).astype(np.float32)
    
    save_image_helper(resized_20, debug_dir / "2_resized_20x20.png")
    res_com_y, res_com_x = compute_center_of_mass(resized_20)
    print("\nSTAGE 2: Resize to fit 20x20 box (bilinear)")
    print(f"  Width x Height     : {resized_20.shape[1]}x{resized_20.shape[0]}")
    print(f"  Scale factor       : {scale_factor:.4f}")
    print(f"  Center of mass     : ({res_com_x:.2f}, {res_com_y:.2f})")
    print(f"  Foreground pixels  : {(resized_20 > 0.1).sum()}")
    
    # STAGE 3: Centering in 28x28 Canvas using Center of Mass
    canvas_28 = np.zeros((28, 28), dtype=np.float32)
    
    # Target center of mass is 13.5 (center of 28x28)
    y_top = int(round(13.5 - res_com_y))
    x_left = int(round(13.5 - res_com_x))
    
    # Ensure it fits within bounds or slice if needed
    y_start = max(0, y_top)
    y_end = min(28, y_top + h_new)
    x_start = max(0, x_left)
    x_end = min(28, x_left + w_new)
    
    crop_y_start = max(0, -y_top)
    crop_y_end = crop_y_start + (y_end - y_start)
    crop_x_start = max(0, -x_left)
    crop_x_end = crop_x_start + (x_end - x_start)
    
    canvas_28[y_start:y_end, x_start:x_end] = resized_20[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    
    save_image_helper(canvas_28, debug_dir / "3_centered_28x28.png")
    final_com_y, final_com_x = compute_center_of_mass(canvas_28)
    print("\nSTAGE 3: Centering in 28x28 Canvas using COM")
    print(f"  Width x Height     : {canvas_28.shape[1]}x{canvas_28.shape[0]}")
    print(f"  Target translation : dy: {y_top}, dx: {x_left}")
    print(f"  Actual center of COM: ({final_com_x:.2f}, {final_com_y:.2f})")
    
    # STAGE 4: Normalization (ensure [0, 1])
    # The EMNIST/MNIST images are normalized [0.0, 1.0].
    # Since our canvas_28 values are already derived from PIL bilinear resize of [0.0, 1.0],
    # we just clip/normalize it.
    normalized = canvas_28.clip(0.0, 1.0)
    
    save_image_helper(normalized, debug_dir / "4_normalized_visualization.png")
    
    # STAGE 5: Overlay with Real EMNIST sample and Heatmap
    real_sample = get_real_emnist_sample(char)
    if real_sample is not None:
        save_image_helper(real_sample, debug_dir / "5_real_emnist_sample.png")
        
        # Calculate pixel difference heatmap
        difference = np.abs(normalized - real_sample)
        
        # Save difference image
        save_image_helper(difference, debug_dir / "6_difference_heatmap.png")
        
        mae = difference.mean()
        mse = (difference ** 2).mean()
        print("\nSTAGE 5: EMNIST Comparison")
        print(f"  Mean Absolute Error: {mae:.4f}")
        print(f"  Mean Squared Error : {mse:.4f}")
        
        # Run CNN prediction on both
        rec = EMNISTCharacterRecognizer()
        rec._ensure_loaded()
        
        # Preprocessed Canvas Image
        t_canvas = torch.from_numpy(normalized.copy()).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            probs_canvas = F.softmax(rec._model(t_canvas), dim=1)[0]
            pred_canvas = EMNIST_LABELS[probs_canvas.argmax().item()]
            conf_canvas = probs_canvas.max().item()
            
        # Real EMNIST sample
        t_real = torch.from_numpy(real_sample.copy()).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            probs_real = F.softmax(rec._model(t_real), dim=1)[0]
            pred_real = EMNIST_LABELS[probs_real.argmax().item()]
            conf_real = probs_real.max().item()
            
        print(f"\nPrediction results:")
        print(f"  Real EMNIST sample predicts : '{pred_real}' with confidence {conf_real:.4f}")
        print(f"  Canvas preprocessed predicts: '{pred_canvas}' with confidence {conf_canvas:.4f}")
    else:
        print("\nReal EMNIST sample not found.")

if __name__ == "__main__":
    debug_pipeline()
