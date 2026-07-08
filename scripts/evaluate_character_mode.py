import sys
import gzip
import struct
import random
from pathlib import Path
from collections import defaultdict
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
    _build_cnn
)

def evaluate():
    cache_dir = _CACHE_DIR
    test_img_path  = cache_dir / "emnist-byclass-test-images-idx3-ubyte.gz"
    test_lbl_path  = cache_dir / "emnist-byclass-test-labels-idx1-ubyte.gz"
    
    if not test_img_path.exists() or not test_lbl_path.exists():
        print(f"Error: test dataset not found in {cache_dir}.")
        sys.exit(1)
        
    print("Reading EMNIST test split …")
    
    # 1. Load EMNIST test split
    with gzip.open(str(test_img_path), "rb") as f:
        struct.unpack(">4I", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    imgs_raw = data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    
    # Apply standard EMNIST orientation fix (preprocessed format fed to CNN)
    imgs_fixed = np.rot90(imgs_raw, k=1, axes=(1, 2))
    imgs_fixed = np.flip(imgs_fixed, axis=2).copy()
    
    with gzip.open(str(test_lbl_path), "rb") as f:
        struct.unpack(">2I", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    # Group indices by class label
    label_to_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        label_to_indices[int(lbl)].append(idx)
        
    # Sample 100 images per class (with deterministic seed)
    random.seed(42)
    sampled_pairs = [] # (img_idx, label)
    for lbl in range(62):
        indices = label_to_indices[lbl]
        if len(indices) >= 100:
            selected = random.sample(indices, 100)
        else:
            selected = indices.copy()
        for idx in selected:
            sampled_pairs.append((idx, lbl))
            
    print(f"Sampled {len(sampled_pairs)} test images across 62 classes.")
    
    # 2. Load model
    weights_path = cache_dir / _WEIGHTS_FILENAME
    model = _build_cnn(num_classes=62)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    
    # 3. Evaluate samples and build confusion matrix
    # C[expected_idx, predicted_idx]
    confusion_matrix = np.zeros((62, 62), dtype=int)
    
    # Track the first image index for each (expected, predicted) confusion pair
    confusion_first_image = {} # (exp_lbl, pred_lbl) -> img_idx
    
    print("Classifying samples with CNN model …")
    for sample_idx, (img_idx, expected_lbl) in enumerate(sampled_pairs):
        img28 = imgs_fixed[img_idx]
        tensor = torch.from_numpy(img28).unsqueeze(0).unsqueeze(0).float()
        
        with torch.no_grad():
            logits = model(tensor)
            pred_lbl = logits.argmax(dim=1).item()
            
        confusion_matrix[expected_lbl, pred_lbl] += 1
        
        if expected_lbl != pred_lbl:
            pair = (expected_lbl, pred_lbl)
            if pair not in confusion_first_image:
                confusion_first_image[pair] = img_idx
                
    # 4. Compute per-class Precision, Recall, and Accuracy
    per_class_stats = [] # dict of metrics per label
    total_tp = 0
    
    for i in range(62):
        tp = confusion_matrix[i, i]
        total_tp += tp
        
        # Row sum: actual instances of class i
        actual_count = confusion_matrix[i, :].sum()
        # Column sum: predicted instances of class i
        predicted_count = confusion_matrix[:, i].sum()
        
        accuracy = tp / actual_count * 100 if actual_count > 0 else 0.0
        recall = tp / actual_count * 100 if actual_count > 0 else 0.0
        precision = tp / predicted_count * 100 if predicted_count > 0 else 0.0
        
        per_class_stats.append({
            "label": i,
            "char": EMNIST_LABELS[i],
            "tp": tp,
            "actual": actual_count,
            "predicted": predicted_count,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        })
        
    overall_accuracy = total_tp / len(sampled_pairs) * 100
    print(f"Overall model accuracy: {overall_accuracy:.2f}%")
    
    # 5. List 20 most confused character pairs
    confusions = [] # list of (expected_lbl, predicted_lbl, count)
    for exp in range(62):
        for pred in range(62):
            if exp != pred and confusion_matrix[exp, pred] > 0:
                confusions.append((exp, pred, confusion_matrix[exp, pred]))
    confusions.sort(key=lambda x: -x[2])
    
    top_20 = confusions[:20]
    print("\nTop 20 Confused Pairs:")
    for idx, (exp, pred, count) in enumerate(top_20):
        exp_char = EMNIST_LABELS[exp]
        pred_char = EMNIST_LABELS[pred]
        print(f"{idx+1:2d}. {exp_char} -> {pred_char} ({count}x)")
        
    # 6. Save confusion pair images
    confusion_dir = Path("data/debug/confusion_pairs")
    confusion_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving preprocessed 28x28 images of the top 20 confused pairs...")
    for exp, pred, count in top_20:
        exp_char = EMNIST_LABELS[exp]
        pred_char = EMNIST_LABELS[pred]
        
        # Get the first image index for this confusion pair
        img_idx = confusion_first_image.get((exp, pred))
        if img_idx is not None:
            img28 = imgs_fixed[img_idx]
            # Convert float32 [0.0, 1.0] to uint8 [0, 255]
            img_uint8 = (img28 * 255.0).clip(0, 255).astype(np.uint8)
            img_name = f"{exp_char}_to_{pred_char}.png"
            # Ensure filenames are safe on Windows (case sensitivity or special chars)
            # Use ASCII hex representation for expected/predicted chars to avoid file conflicts
            safe_name = f"exp_{exp}_pred_{pred}_{exp_char}_to_{pred_char}.png"
            Image.fromarray(img_uint8).save(confusion_dir / safe_name)
            
    print(f"Saved images to: {confusion_dir}")
    
    # 7. Generate markdown report confusion_analysis.md in the artifact directory
    artifact_dir = Path("C:/Users/Nikhil/.gemini/antigravity-ide/brain/d52d4103-dabb-4fad-8bf4-cb382b5f4c06")
    report_path = artifact_dir / "confusion_analysis.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# EMNIST Character Mode 62-Class Evaluation Report\n\n")
        f.write(f"This report presents the metrics and confusion analysis for the pre-trained EMNIST ByClass CNN model across all 62 character classes. We sampled 100 test images per class, resulting in {len(sampled_pairs)} evaluation samples.\n\n")
        
        f.write("## Overall Metrics\n")
        f.write(f"- **Overall Accuracy**: {overall_accuracy:.2f}%\n")
        f.write(f"- **Total Samples**: {len(sampled_pairs)}\n")
        f.write(f"- **True Positives**: {total_tp}\n\n")
        
        f.write("## Top 20 Most Confused Character Pairs\n")
        f.write("These are the pairs of characters that were most frequently misclassified. The corresponding preprocessed 28x28 input images have been saved to [data/debug/confusion_pairs/](file:///c:/Users/Nikhil/Downloads/IFP/assistive-writing-pad/data/debug/confusion_pairs/).\n\n")
        f.write("| Rank | Expected | Predicted | Count | Saved Image Name |\n")
        f.write("|---|---|---|---|---|\n")
        for idx, (exp, pred, count) in enumerate(top_20):
            exp_char = EMNIST_LABELS[exp]
            pred_char = EMNIST_LABELS[pred]
            safe_name = f"exp_{exp}_pred_{pred}_{exp_char}_to_{pred_char}.png"
            f.write(f"| {idx+1} | `{exp_char}` | `{pred_char}` | {count} | [`{safe_name}`](file:///c:/Users/Nikhil/Downloads/IFP/assistive-writing-pad/data/debug/confusion_pairs/{safe_name}) |\n")
        f.write("\n")
        
        f.write("## Per-Class Metrics\n")
        f.write("This table shows the True Positives (TP), actual dataset counts, prediction counts, Accuracy, Precision, and Recall for each class.\n\n")
        f.write("| Class | Char | TP | Actual Count | Predicted Count | Accuracy (%) | Precision (%) | Recall (%) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for stat in per_class_stats:
            f.write(f"| {stat['label']:2d} | `{stat['char']}` | {stat['tp']} | {stat['actual']} | {stat['predicted']} | {stat['accuracy']:.2f} | {stat['precision']:.2f} | {stat['recall']:.2f} |\n")
        f.write("\n")
        
        f.write("## Confusion Matrix (62x62)\n")
        f.write("Rows represent the expected ground truth classes, and columns represent the predicted classes.\n\n")
        
        # Create a header row with EMNIST_LABELS
        header = "|   | " + " | ".join(f"`{c}`" for c in EMNIST_LABELS) + " |\n"
        separator = "|---|" + "|".join("---" for _ in EMNIST_LABELS) + "|\n"
        f.write(header)
        f.write(separator)
        for i in range(62):
            row_str = f"| `{EMNIST_LABELS[i]}` | " + " | ".join(str(confusion_matrix[i, j]) for j in range(62)) + " |\n"
            f.write(row_str)
            
    print(f"Report successfully saved to: {report_path}")

if __name__ == "__main__":
    evaluate()
