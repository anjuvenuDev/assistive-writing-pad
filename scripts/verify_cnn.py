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

def verify():
    # Tasks 1 & 2: Print model info, output classes, checkpoint metadata, check classes length
    weights_path = _CACHE_DIR / _WEIGHTS_FILENAME
    print("\n==========================================")
    print("TASK 1: MODEL & CHECKPOINT INFO")
    print("==========================================")
    print(f"Checkpoint file : {weights_path}")
    print(f"Exists          : {weights_path.exists()}")
    
    # Instantiate the CNN
    model = _build_cnn(num_classes=len(EMNIST_LABELS))
    print(f"Architecture    :\n{model}")
    
    # Load state dict
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    is_state_dict = True
    metadata = {}
    if isinstance(state, dict) and not all(isinstance(k, str) and (k.startswith("features") or k.startswith("classifier")) for k in state.keys()):
        # It's not just a state_dict, or has metadata
        is_state_dict = False
        print("Checkpoint type : Dictionary (custom dict)")
        print(f"Keys            : {list(state.keys())}")
        for k in list(state.keys()):
            if k != "state_dict":
                metadata[k] = state[k]
    else:
        print("Checkpoint type : State Dictionary (standard weights only)")
        
    print(f"Metadata        : {metadata if metadata else 'None stored in state dict'}")
    print(f"Training dataset: EMNIST ByClass (62 classes: 0-9, A-Z, a-z)")
    print(f"Validation acc  : Not stored in state dict (checkpoint is standard PyTorch state_dict)")
    
    # Output classes
    # Check classifier output features
    last_layer = model.classifier[-1]
    num_classes = last_layer.out_features
    print(f"Output classes  : {num_classes}")
    
    print("\n==========================================")
    print("TASK 2: LABEL VERIFICATION")
    print("==========================================")
    # len(id2label) == model output classes
    print(f"len(EMNIST_LABELS)      : {len(EMNIST_LABELS)}")
    print(f"model output classes    : {num_classes}")
    assert len(EMNIST_LABELS) == num_classes, f"Mismatch: {len(EMNIST_LABELS)} labels vs {num_classes} model outputs!"
    print("Verification result     : len(id2label) == model output classes (MATCH)")
    
    print("\n==========================================")
    print("TASK 3: FIRST 20 LABELS")
    print("==========================================")
    for i in range(20):
        print(f"{i:2d} -> {EMNIST_LABELS[i]}")
        
    # Task 5: Load a known EMNIST test image and run prediction
    print("\n==========================================")
    print("TASK 5: FEED KNOWN EMNIST TEST IMAGE")
    print("==========================================")
    test_img_path = _CACHE_DIR / "emnist-byclass-test-images-idx3-ubyte.gz"
    test_lbl_path = _CACHE_DIR / "emnist-byclass-test-labels-idx1-ubyte.gz"
    
    if not test_img_path.exists():
        print(f"Error: test dataset not found at {test_img_path}. Run scripts/train_emnist.py to download.")
        return
        
    with gzip.open(str(test_img_path), "rb") as f:
        struct.unpack(">4I", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    imgs_raw = data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    
    with gzip.open(str(test_lbl_path), "rb") as f:
        struct.unpack(">2I", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    # We choose a sample, e.g. index 32 which is 'F' (label 15)
    sample_idx = 32
    raw_img = imgs_raw[sample_idx]
    gt_label = int(labels[sample_idx])
    gt_char = EMNIST_LABELS[gt_label]
    
    # Apply standard EMNIST fix to raw test image to get the image EMNIST uses
    fixed_img = np.rot90(raw_img, k=1)
    fixed_img = np.flip(fixed_img, axis=1).copy()
    
    # Run prediction on the fixed image
    model.load_state_dict(state)
    model.eval()
    
    # Task 4: Save exact tensor passed to CNN and print statistics
    print("\n==========================================")
    print("TASK 4: CNN INPUT TENSOR STATISTICS")
    print("==========================================")
    # img28: fixed_img
    tensor = torch.from_numpy(fixed_img).unsqueeze(0).unsqueeze(0).float()
    print(f"Shape           : {list(tensor.shape)}")
    print(f"Dtype           : {tensor.dtype}")
    print(f"Min             : {tensor.min().item():.6f}")
    print(f"Max             : {tensor.max().item():.6f}")
    print(f"Mean            : {tensor.mean().item():.6f}")
    print(f"Std             : {tensor.std().item():.6f}")
    
    # Save the exact 28x28 tensor as emnist_input.png
    debug_dir = Path("data/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    img_uint8 = (fixed_img * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(debug_dir / "emnist_input.png")
    print(f"Saved exact 28x28 tensor to: {debug_dir / 'emnist_input.png'}")
    
    # Feed to CNN
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        top_idx = probs.argmax().item()
        pred_char = EMNIST_LABELS[top_idx]
        confidence = probs[top_idx].item()
        
    print(f"Ground truth    : {gt_char} (label {gt_label})")
    print(f"Prediction      : {pred_char}")
    print(f"Confidence      : {confidence:.4f}")

if __name__ == "__main__":
    verify()
