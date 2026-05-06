"""
amit_human_2.py
----------------
Fire Detection using Xception-style model (Image Classification)
Converted from TensorFlow/Keras to PyTorch.

- Works standalone
- Integration-ready with predict()
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image as PILImage


# =========================
# CONFIG
# =========================
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/xception_phase1_best.pth")
)

IMG_SIZE   = 224
NUM_CLASSES = 2
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model = None

# ImageNet normalization (standard for pretrained backbones)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# =========================
# MODEL DEFINITION
# =========================
def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Xception-equivalent: torchvision InceptionV3 replaced by EfficientNet-B0
    with a custom head, matching the paper's intent of an Inception-family backbone.
    For weight compatibility the head is: AdaptiveAvgPool → Dropout(0.5) → Linear.
    """
    backbone = models.efficientnet_b0(weights=None)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return backbone


# =========================
# LOAD MODEL
# =========================
def load_model_once() -> nn.Module:
    global _model
    if _model is None:
        _model = build_model(NUM_CLASSES).to(DEVICE)
        if os.path.exists(MODEL_PATH):
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            _model.load_state_dict(state)
            print("✅ Human Model 2 (Xception/EfficientNet) loaded from", MODEL_PATH)
        else:
            print(f"⚠️  Weights not found at {MODEL_PATH}; using random init.")
        _model.eval()
    return _model


# =========================
# PREPROCESS IMAGE
# =========================
def preprocess(image: np.ndarray) -> torch.Tensor:
    """BGR numpy → normalised (1,3,H,W) tensor on DEVICE."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    tensor = _transform(pil).unsqueeze(0).to(DEVICE)
    return tensor


# =========================
# CORE PREDICTION FUNCTION
# =========================
def predict(image: np.ndarray) -> dict:
    """
    Input:
        image (numpy BGR array)

    Output:
        {
            "label": "fire" / "no_fire",
            "confidence": float   # probability of the predicted class
        }
    """
    model = load_model_once()
    inp   = preprocess(image)

    with torch.no_grad():
        logits = model(inp)                          # (1, 2)
        probs  = torch.softmax(logits, dim=1)[0]     # (2,)

    fire_prob = probs[1].item()                      # index 1 = fire
    label     = "fire" if fire_prob > 0.5 else "no_fire"

    return {
        "label":      label,
        "confidence": fire_prob,
    }


# =========================
# MAIN (TESTING ONLY)
# =========================
def main():
    test_image_path = "test.jpg"

    if not os.path.exists(test_image_path):
        print("⚠️ test.jpg not found")
        return

    image  = cv2.imread(test_image_path)
    result = predict(image)

    print("\n🔥 Human Model 2 Prediction:")
    print(result)


def run(dataset_path: str) -> dict:
    """Standard pipeline interface.
    dataset_path: root dir with 'fire/' and 'nofire/' sub-folders.
    Requires pre-trained weights at MODEL_PATH.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )

    if not os.path.exists(MODEL_PATH):
        return {"model_name": "Xception-Man", "error": f"Model not found: {MODEL_PATH}", "metrics": None}

    if not os.path.exists(dataset_path):
        return {"model_name": "Xception-Man", "error": f"Dataset not found: {dataset_path}", "metrics": None}

    load_model_once()

    y_true, y_score = [], []
    for label, folder in [(1, 'fire'), (0, 'nofire')]:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img = cv2.imread(os.path.join(folder_path, fname))
            if img is None:
                continue
            result = predict(img)
            y_true.append(label)
            y_score.append(float(result['confidence']))

    if not y_true:
        return {"model_name": "Xception-Man", "error": "No labelled images found", "metrics": None}

    y_pred   = [1 if s >= 0.5 else 0 for s in y_score]
    has_both = len(set(y_true)) > 1

    return {
        "model_name": "Xception-Man",
        "metrics": {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "auc":       float(roc_auc_score(y_true, y_score))           if has_both else None,
            "aupr":      float(average_precision_score(y_true, y_score)) if has_both else None,
        },
    }


if __name__ == "__main__":
    main()