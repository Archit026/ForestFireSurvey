"""
amit_human_2.py
----------------
Fire Detection using Xception (Image Classification)

- Works standalone
- Integration-ready with predict()
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# =========================
# CONFIG
# =========================
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/xception_phase1_best.h5")
)

IMG_SIZE = 224
_model = None


# =========================
# LOAD MODEL
# =========================
def load_model_once():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
        print("✅ Human Model 2 (Xception) loaded")
    return _model


# =========================
# PREPROCESS IMAGE
# =========================
def preprocess(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return np.expand_dims(image, axis=0)


# =========================
# CORE PREDICTION FUNCTION
# =========================
def predict(image):
    """
    Input:
        image (numpy array)

    Output:
        {
            "label": "fire" / "no_fire",
            "confidence": float
        }
    """
    model = load_model_once()

    img = preprocess(image)
    pred = model.predict(img)[0][0]

    label = "fire" if pred > 0.5 else "no_fire"

    return {
        "label": label,
        "confidence": float(pred)
    }


# =========================
# MAIN (TESTING ONLY)
# =========================
def main():
    test_image_path = "test.jpg"

    if not os.path.exists(test_image_path):
        print("⚠️ test.jpg not found")
        return

    image = cv2.imread(test_image_path)

    result = predict(image)

    print("\n🔥 Human Model 2 Prediction:")
    print(result)


def run(dataset_path):
    """Standard pipeline interface.
    dataset_path: root dir with 'fire/' and 'nofire/' sub-folders.
    Requires pre-trained Xception weights at MODEL_PATH.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )

    if not os.path.exists(MODEL_PATH):
        return {"model_name": "Xception-Man", "error": f"Model not found: {MODEL_PATH}", "metrics": None}

    if not os.path.exists(dataset_path):
        return {"model_name": "Xception-Man", "error": f"Dataset not found: {dataset_path}", "metrics": None}

    load_model_once()  # warm-up singleton

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
            # Raw sigmoid output: >0.5 means fire
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