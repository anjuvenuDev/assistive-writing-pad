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

from assistive_writing_pad.recognition.emnist import EMNISTCharacterRecognizer, EMNIST_LABELS
from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.preprocessing.rasterize import rasterize_strokes
from assistive_writing_pad.preprocessing.image_ops import crop_to_content, pad_to_square, resize_nearest, normalize_unit

# ---------------------------------------------------------------------------
# Dense Stroke Generation Helpers
# ---------------------------------------------------------------------------
def interpolate(p1, p2, steps=15):
    x1, y1 = p1
    x2, y2 = p2
    return [(x1 + (x2 - x1) * t / steps, y1 + (y2 - y1) * t / steps) for t in range(steps + 1)]

def circle_points(cx, cy, rx, ry, num_pts=30):
    import math
    pts = []
    for i in range(num_pts + 1):
        angle = 2 * math.pi * i / num_pts
        pts.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
    return pts

def get_char_strokes(char):
    # Generates dense strokes for characters in a 100x100 canvas.
    pts = []
    c = char
    if c == 'a':
        pts.extend(circle_points(50, 60, 15, 15))
        pts.extend(interpolate((65, 45), (65, 75)))
    elif c == 'b':
        pts.extend(interpolate((35, 20), (35, 75)))
        pts.extend(circle_points(50, 60, 15, 15))
    elif c == 'c':
        pts.extend(interpolate((65, 45), (45, 45)))
        pts.extend(interpolate((45, 45), (45, 75)))
        pts.extend(interpolate((45, 75), (65, 75)))
    elif c == 'd':
        pts.extend(circle_points(50, 60, 15, 15))
        pts.extend(interpolate((65, 20), (65, 75)))
    elif c == 'e':
        pts.extend(interpolate((35, 55), (65, 55)))
        pts.extend(interpolate((65, 55), (65, 40)))
        pts.extend(interpolate((65, 40), (40, 40)))
        pts.extend(interpolate((40, 40), (40, 75)))
        pts.extend(interpolate((40, 75), (65, 75)))
    elif c == 'f':
        pts.extend(interpolate((50, 75), (50, 30)))
        pts.extend(interpolate((50, 30), (65, 30)))
        pts.extend(interpolate((40, 45), (60, 45)))
    elif c == 'g':
        pts.extend(circle_points(50, 45, 15, 15))
        pts.extend(interpolate((65, 30), (65, 75)))
        pts.extend(interpolate((65, 75), (45, 75)))
    elif c == 'h':
        pts.extend(interpolate((35, 20), (35, 75)))
        pts.extend(interpolate((35, 45), (50, 45)))
        pts.extend(interpolate((50, 45), (50, 75)))
    elif c == 'i':
        pts.extend(interpolate((50, 40), (50, 75)))
        pts.extend(interpolate((50, 25), (50, 26))) # Dot
    elif c == 'j':
        pts.extend(interpolate((60, 40), (60, 75)))
        pts.extend(interpolate((60, 75), (45, 75)))
        pts.extend(interpolate((60, 25), (60, 26))) # Dot
    elif c == 'A':
        pts.extend(interpolate((50, 20), (25, 75)))
        pts.extend(interpolate((50, 20), (75, 75)))
        pts.extend(interpolate((35, 55), (65, 55)))
    elif c == 'B':
        pts.extend(interpolate((35, 20), (35, 75)))
        pts.extend(interpolate((35, 20), (60, 20)))
        pts.extend(interpolate((60, 20), (60, 47)))
        pts.extend(interpolate((60, 47), (35, 47)))
        pts.extend(interpolate((35, 47), (65, 47)))
        pts.extend(interpolate((65, 47), (65, 75)))
        pts.extend(interpolate((65, 75), (35, 75)))
    elif c == 'C':
        pts.extend(interpolate((65, 25), (40, 25)))
        pts.extend(interpolate((40, 25), (40, 75)))
        pts.extend(interpolate((40, 75), (65, 75)))
    elif c == 'D':
        pts.extend(interpolate((35, 20), (35, 75)))
        pts.extend(interpolate((35, 20), (65, 25)))
        pts.extend(interpolate((65, 25), (65, 70)))
        pts.extend(interpolate((65, 70), (35, 75)))
    elif c == 'E':
        pts.extend(interpolate((35, 20), (35, 75)))
        pts.extend(interpolate((35, 20), (65, 20)))
        pts.extend(interpolate((35, 47), (55, 47)))
        pts.extend(interpolate((35, 75), (65, 75)))
    elif c == 'F':
        pts.extend(interpolate((35, 20), (35, 75)))
        pts.extend(interpolate((35, 20), (65, 20)))
        pts.extend(interpolate((35, 47), (55, 47)))
    elif c == 'G':
        pts.extend(interpolate((65, 25), (40, 25)))
        pts.extend(interpolate((40, 25), (40, 75)))
        pts.extend(interpolate((40, 75), (65, 75)))
        pts.extend(interpolate((65, 75), (65, 50)))
        pts.extend(interpolate((65, 50), (52, 50)))
    elif c == 'H':
        pts.extend(interpolate((35, 20), (35, 75)))
        pts.extend(interpolate((65, 20), (65, 75)))
        pts.extend(interpolate((35, 47), (65, 47)))
    elif c == 'I':
        pts.extend(interpolate((50, 20), (50, 75)))
        pts.extend(interpolate((35, 20), (65, 20)))
        pts.extend(interpolate((35, 75), (65, 75)))
    elif c == 'J':
        pts.extend(interpolate((35, 20), (65, 20)))
        pts.extend(interpolate((50, 20), (50, 70)))
        pts.extend(interpolate((50, 70), (35, 70)))
    elif c == '0':
        pts.extend(circle_points(50, 50, 20, 28))
    elif c == '1':
        pts.extend(interpolate((40, 30), (50, 20)))
        pts.extend(interpolate((50, 20), (50, 75)))
        pts.extend(interpolate((35, 75), (65, 75)))
    elif c == '2':
        pts.extend(interpolate((35, 30), (50, 20)))
        pts.extend(interpolate((50, 20), (65, 30)))
        pts.extend(interpolate((65, 30), (35, 75)))
        pts.extend(interpolate((35, 75), (65, 75)))
    elif c == '3':
        pts.extend(interpolate((35, 25), (60, 25)))
        pts.extend(interpolate((60, 25), (48, 47)))
        pts.extend(interpolate((48, 47), (65, 47)))
        pts.extend(interpolate((65, 47), (65, 70)))
        pts.extend(interpolate((65, 70), (35, 75)))
    elif c == '4':
        pts.extend(interpolate((35, 20), (35, 50)))
        pts.extend(interpolate((35, 50), (65, 50)))
        pts.extend(interpolate((55, 20), (55, 75)))
    elif c == '5':
        pts.extend(interpolate((65, 20), (40, 20)))
        pts.extend(interpolate((40, 20), (40, 45)))
        pts.extend(interpolate((40, 45), (65, 45)))
        pts.extend(interpolate((65, 45), (65, 75)))
        pts.extend(interpolate((65, 75), (35, 75)))
    elif c == '6':
        pts.extend(interpolate((60, 25), (40, 45)))
        pts.extend(circle_points(50, 60, 15, 15))
    elif c == '7':
        pts.extend(interpolate((35, 20), (65, 20)))
        pts.extend(interpolate((65, 20), (40, 75)))
    elif c == '8':
        pts.extend(circle_points(50, 35, 15, 15))
        pts.extend(circle_points(50, 65, 17, 17))
    elif c == '9':
        pts.extend(circle_points(50, 38, 15, 15))
        pts.extend(interpolate((65, 38), (65, 75)))
        pts.extend(interpolate((65, 75), (45, 75)))

    return [StrokePoint(x=float(x), y=float(y), timestamp_ms=i*16, pressure=1.0) for i, (x, y) in enumerate(pts)]

# ---------------------------------------------------------------------------
# Preprocessing and Orientation Helpers
# ---------------------------------------------------------------------------
def apply_orientation(img, name):
    if name == "original":
        return img
    elif name == "rot90":
        return np.rot90(img, k=1)
    elif name == "rot180":
        return np.rot90(img, k=2)
    elif name == "rot270":
        return np.rot90(img, k=3)
    elif name == "flip_h":
        return np.flip(img, axis=1)
    elif name == "flip_v":
        return np.flip(img, axis=0)
    elif name == "transpose":
        return img.T
    elif name == "transpose_flip":
        return np.flip(np.flip(img.T, axis=0), axis=1)
    else:
        raise ValueError(name)

# Equivalence classes to handle EMNIST classification ambiguities
EQUIVALENCE = {
    '0': {'0', 'O', 'o', 'Q', 'q'},
    '1': {'1', 'I', 'i', 'l', 'L', 't', '|'},
    '2': {'2', 'Z', 'z'},
    '3': {'3', 'E', 's'},
    '4': {'4'},
    '5': {'5', 'S', 's'},
    '6': {'6', 'b', 'G'},
    '7': {'7', 'Z', 'z'},
    '8': {'8', 'B'},
    '9': {'9', 'g', 'q'},
    'a': {'a', 'A', 'o'},
    'b': {'b', 'B', '6', 'h'},
    'c': {'c', 'C', 'o'},
    'd': {'d', 'D', '0'},
    'e': {'e', 'E', 'c'},
    'f': {'f', 'F', 't'},
    'g': {'g', 'G', '9', 'q'},
    'h': {'h', 'H', 'n'},
    'i': {'i', 'I', '1', 'l'},
    'j': {'j', 'J', 'i'},
    'A': {'A', 'a'},
    'B': {'B', 'b', '8'},
    'C': {'C', 'c'},
    'D': {'D', 'd', '0', 'O'},
    'E': {'E', 'e'},
    'F': {'F', 'f'},
    'G': {'G', 'g', '6'},
    'H': {'H', 'h'},
    'I': {'I', 'i', '1', 'l'},
    'J': {'J', 'j'}
}

def is_correct(predicted, expected):
    if predicted == expected:
        return True
    # Check lowercase/uppercase match
    if predicted.upper() == expected.upper():
        return True
    # Check equivalence classes
    if expected in EQUIVALENCE and predicted in EQUIVALENCE[expected]:
        return True
    return False

# ---------------------------------------------------------------------------
# Diagnostics & Verification Runner
# ---------------------------------------------------------------------------
def run_diagnostic():
    rec = EMNISTCharacterRecognizer()
    rec._ensure_loaded()
    
    # 1. Target Character 't' diagnostic (User's specific example)
    print("\n--- RUNNING DIAGNOSTIC ON CHARACTER 't' ---")
    # normal t: stem (50, 15) to (50, 75), crossbar (30, 35) to (70, 35)
    pts_t = interpolate((50, 15), (50, 75)) + interpolate((30, 35), (70, 35))
    strokes_t = [StrokePoint(x=float(x), y=float(y), timestamp_ms=i*16, pressure=1.0) for i, (x, y) in enumerate(pts_t)]
    
    from assistive_writing_pad.preprocessing.rasterize import RasterizerConfig
    config = RasterizerConfig(line_radius_px=4)
    raster = rasterize_strokes(strokes_t, config)
    cropped = crop_to_content(raster)
    squared = pad_to_square(cropped)
    resized = resize_nearest(squared, (28, 28))
    normalized = normalize_unit(resized)
    
    orientations = ["original", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "transpose_flip"]
    
    print("\nOrientation | Prediction | Confidence")
    print("-" * 40)
    for name in orientations:
        rotated_img = apply_orientation(normalized, name).copy()
        
        # Task 2: Save debug images for all orientation variants of this character
        debug_dir = Path("data/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        img_uint8 = (rotated_img * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(debug_dir / f"{name}.png")
        
        tensor = torch.from_numpy(rotated_img).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            logits = rec._model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            top_val, top_idx = torch.topk(probs, 1)
            pred_char = EMNIST_LABELS[top_idx[0].item()]
            confidence = top_val[0].item()
        print(f"{name:15} | {pred_char:10} | {confidence:.4f}")

    # 2. Automatically determine best orientation across 30 characters
    print("\n--- AUTOMATICALLY DETERMINING BEST ORIENTATION ---")
    char_set = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    ]
    
    orientation_scores = {name: 0.0 for name in orientations}
    orientation_correct_counts = {name: 0 for name in orientations}
    
    for char in char_set:
        strokes = get_char_strokes(char)
        raster = rasterize_strokes(strokes, config)
        cropped = crop_to_content(raster)
        squared = pad_to_square(cropped)
        resized = resize_nearest(squared, (28, 28))
        normalized = normalize_unit(resized)
        
        for name in orientations:
            rotated_img = apply_orientation(normalized, name).copy()
            tensor = torch.from_numpy(rotated_img).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                logits = rec._model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                top_idx = probs.argmax().item()
                pred = EMNIST_LABELS[top_idx]
                conf = probs[top_idx].item()
                
            orientation_scores[name] += conf
            if is_correct(pred, char):
                orientation_correct_counts[name] += 1
                
    best_orientation = max(orientations, key=lambda name: (orientation_correct_counts[name], orientation_scores[name]))
    
    print("\nOrientation     | Correct Count | Avg Confidence")
    print("-" * 48)
    for name in orientations:
        avg_conf = orientation_scores[name] / len(char_set)
        correct_cnt = orientation_correct_counts[name]
        indicator = "★" if name == best_orientation else " "
        print(f"{name:15} | {correct_cnt:13d}/{len(char_set)} | {avg_conf:14.4f} {indicator}")
        
    print(f"\nMathematically/Empirically Best Orientation: {best_orientation}")

    # 3. Before vs After Accuracy Comparison
    print("\n--- BEFORE VS AFTER ACCURACY ON 30 CHARACTERS ---")
    
    # Before (no transform/original)
    before_correct = 0
    print("\nBEFORE Fix (Original Upright):")
    print(f"{'Expected':10} | {'Predicted':10} | {'Confidence':10} | {'Correct'}")
    print("-" * 50)
    for char in char_set:
        strokes = get_char_strokes(char)
        raster = rasterize_strokes(strokes, config)
        cropped = crop_to_content(raster)
        squared = pad_to_square(cropped)
        resized = resize_nearest(squared, (28, 28))
        normalized = normalize_unit(resized)
        
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            probs = F.softmax(rec._model(tensor), dim=1)[0]
            top_idx = probs.argmax().item()
            pred = EMNIST_LABELS[top_idx]
            conf = probs[top_idx].item()
            
        correct = is_correct(pred, char)
        if correct:
            before_correct += 1
        ok_sym = "✓" if correct else "✗"
        print(f"{char:10} | {pred:10} | {conf:10.4f} | {ok_sym}")
        
    # After (with best_orientation transform)
    after_correct = 0
    print(f"\nAFTER Fix (Applying {best_orientation}):")
    print(f"{'Expected':10} | {'Predicted':10} | {'Confidence':10} | {'Correct'}")
    print("-" * 50)
    for char in char_set:
        strokes = get_char_strokes(char)
        raster = rasterize_strokes(strokes, config)
        cropped = crop_to_content(raster)
        squared = pad_to_square(cropped)
        resized = resize_nearest(squared, (28, 28))
        normalized = normalize_unit(resized)
        
        fixed_img = apply_orientation(normalized, best_orientation).copy()
        tensor = torch.from_numpy(fixed_img).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            probs = F.softmax(rec._model(tensor), dim=1)[0]
            top_idx = probs.argmax().item()
            pred = EMNIST_LABELS[top_idx]
            conf = probs[top_idx].item()
            
        correct = is_correct(pred, char)
        if correct:
            after_correct += 1
        ok_sym = "✓" if correct else "✗"
        print(f"{char:10} | {pred:10} | {conf:10.4f} | {ok_sym}")
        
    before_acc = before_correct / len(char_set) * 100
    after_acc = after_correct / len(char_set) * 100
    print(f"\nBefore Fix Accuracy : {before_correct}/{len(char_set)} = {before_acc:.1f}%")
    print(f"After Fix Accuracy  : {after_correct}/{len(char_set)} = {after_acc:.1f}%")

if __name__ == "__main__":
    run_diagnostic()
