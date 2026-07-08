import sys
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

from assistive_writing_pad.recognition.emnist import (
    EMNIST_LABELS,
    _CACHE_DIR,
    _WEIGHTS_FILENAME,
    _build_cnn,
    EMNISTCharacterRecognizer
)
from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.preprocessing.rasterize import rasterize_strokes, RasterizerConfig
from assistive_writing_pad.preprocessing.image_ops import crop_to_content, pad_to_square, resize_nearest, normalize_unit

def interpolate(p1, p2, steps=15):
    x1, y1 = p1
    x2, y2 = p2
    return [(x1 + (x2 - x1) * t / steps, y1 + (y2 - y1) * t / steps) for t in range(steps + 1)]

def get_upright_f_strokes():
    # Capital F
    pts = interpolate((35, 20), (35, 75)) + interpolate((35, 20), (65, 20)) + interpolate((35, 47), (55, 47))
    return [StrokePoint(x=float(x), y=float(y), timestamp_ms=i*16, pressure=1.0) for i, (x, y) in enumerate(pts)]

def get_lowercase_f_strokes():
    # Lowercase f: stem (50, 75) to (50, 30), hook at top (50, 30) to (65, 30), crossbar (40, 45) to (60, 45)
    pts = interpolate((50, 75), (50, 30)) + interpolate((50, 30), (65, 30)) + interpolate((40, 45), (60, 45))
    return [StrokePoint(x=float(x), y=float(y), timestamp_ms=i*16, pressure=1.0) for i, (x, y) in enumerate(pts)]

def print_ascii(img, title):
    print(f"\n--- {title} (28x28) ---")
    for r in range(28):
        line = ""
        for c in range(28):
            val = img[r, c]
            if val > 0.5:
                line += "#"
            elif val > 0.1:
                line += "."
            else:
                line += " "
        print(line)

def compare():
    # Load model
    weights_path = _CACHE_DIR / _WEIGHTS_FILENAME
    model = _build_cnn(num_classes=len(EMNIST_LABELS))
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    
    # 1. Load real EMNIST test image of 'F' (sample 32)
    test_img_path = _CACHE_DIR / "emnist-byclass-test-images-idx3-ubyte.gz"
    with gzip.open(str(test_img_path), "rb") as f:
        struct.unpack(">4I", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    imgs_raw = data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    
    real_f_raw = imgs_raw[32]
    # Fixed image used in benchmark / model training:
    real_f_fixed = np.flip(np.rot90(real_f_raw, k=1), axis=1)
    
    # 2. Preprocess our canvas 'f' strokes normally (upright)
    strokes = get_lowercase_f_strokes()
    config = RasterizerConfig(line_radius_px=4)
    raster = rasterize_strokes(strokes, config)
    cropped = crop_to_content(raster)
    squared = pad_to_square(cropped)
    resized = resize_nearest(squared, (28, 28))
    canvas_f_upright = normalize_unit(resized)
    
    # Print the two images step-by-step
    print("STEP 1: VISUAL COMPARISON")
    print_ascii(real_f_fixed, "Real EMNIST Fixed Sample (Recognized as 'F')")
    print_ascii(canvas_f_upright, "Canvas Upright Sample (Normally Drawn)")
    
    # 3. Compare center of gravity / top-heavy vs bottom-heavy
    # Upright 'f' has loop/crossbar near the top (top-heavy).
    # EMNIST Fixed sample of 'f' has loop/crossbar near the bottom (bottom-heavy).
    real_top_mass = real_f_fixed[:14, :].sum()
    real_bot_mass = real_f_fixed[14:, :].sum()
    
    canvas_top_mass = canvas_f_upright[:14, :].sum()
    canvas_bot_mass = canvas_f_upright[14:, :].sum()
    
    print("\nSTEP 2: PIXEL MASS DISTRIBUTION COMPARISON")
    print(f"Real EMNIST Fixed Sample - Top half mass: {real_top_mass:.2f}, Bottom half mass: {real_bot_mass:.2f}")
    print(f"Canvas Upright Sample     - Top half mass: {canvas_top_mass:.2f}, Bottom half mass: {canvas_bot_mass:.2f}")
    
    if real_top_mass < real_bot_mass and canvas_top_mass > canvas_bot_mass:
        print("-> CONFIRMED MISMATCH: EMNIST sample is bottom-heavy (upside-down), Canvas sample is top-heavy (upright).")
    
    # 4. Apply rot180 to Canvas upright sample
    canvas_f_rot180 = np.rot90(canvas_f_upright, k=2)
    print_ascii(canvas_f_rot180, "Canvas Sample Rotated 180 degrees")
    
    canvas_rot_top_mass = canvas_f_rot180[:14, :].sum()
    canvas_rot_bot_mass = canvas_f_rot180[14:, :].sum()
    print(f"Canvas Rotated Sample    - Top half mass: {canvas_rot_top_mass:.2f}, Bottom half mass: {canvas_rot_bot_mass:.2f}")
    
    # Run CNN on all three
    print("\nSTEP 3: CNN CLASSIFICATION RESULTS")
    for name, img in [("Real EMNIST Fixed", real_f_fixed), ("Canvas Upright", canvas_f_upright), ("Canvas Rotated 180", canvas_f_rot180)]:
        t = torch.from_numpy(img.copy()).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            probs = F.softmax(model(t), dim=1)[0]
            top_idx = probs.argmax().item()
            pred = EMNIST_LABELS[top_idx]
            conf = probs[top_idx].item()
        print(f"{name:22} predicts: '{pred}' with confidence {conf:.4f}")

if __name__ == "__main__":
    compare()
