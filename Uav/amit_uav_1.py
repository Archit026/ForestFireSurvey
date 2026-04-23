"""
amit_uav_1.py
----------------
Fire Segmentation using UAV (U-Net .h5 model)
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# =========================
# CONFIG
# =========================
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/fire_unet_final.h5")
)
IMG_SIZE = 256

_model = None


# =========================
# LOAD MODEL
# =========================
def load_uav_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
        print("✅ UAV U-Net model loaded")
    return _model


# =========================
# PREPROCESS IMAGE
# =========================
def preprocess(image):
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_norm = image_resized / 255.0
    return np.expand_dims(image_norm, axis=0)


# =========================
# PREDICT FUNCTION (IMPORTANT)
# =========================
def predict(image):
    """
    Output:
        {
            "label": "fire" / "none",
            "confidence": float,
            "area": float
        }
    """
    model = load_uav_model()

    input_img = preprocess(image)
    pred_mask = model.predict(input_img)[0]

    # Convert mask to binary
    mask = (pred_mask > 0.5).astype(np.uint8)

    fire_pixels = np.sum(mask)
    total_pixels = mask.size

    fire_ratio = fire_pixels / total_pixels

    return {
        "label": "fire" if fire_ratio > 0.01 else "none",
        "confidence": float(np.max(pred_mask)),
        "area": float(fire_ratio)
    }


# =========================
# VISUALIZATION
# =========================
def draw_mask(image):
    model = load_uav_model()

    input_img = preprocess(image)
    pred_mask = model.predict(input_img)[0]

    mask = (pred_mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 2] = mask  # red overlay

    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

    return overlay


# =========================
# MAIN (TEST ONLY)
# =========================
def main():
    test_image_path = "test.jpg"

    if not os.path.exists(test_image_path):
        print("⚠️ test.jpg not found")
        return

    image = cv2.imread(test_image_path)

    result = predict(image)

    print("\n🔥 UAV Prediction:")
    print(result)

    output = draw_mask(image)
    cv2.imshow("UAV Fire Segmentation", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run(dataset_path):
    """Standard pipeline interface.
    dataset_path: root dir containing 'fire/' and 'nofire/' sub-folders.
    Requires the pre-trained U-Net model at MODEL_PATH.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )

    if not os.path.exists(MODEL_PATH):
        return {"model_name": "UAV-UNet", "error": f"Model not found: {MODEL_PATH}", "metrics": None}

    if not os.path.exists(dataset_path):
        return {"model_name": "UAV-UNet", "error": f"Dataset not found: {dataset_path}", "metrics": None}

    # Load model once
    load_uav_model()

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
            # score = probability of fire
            score = result['confidence'] if result['label'] == 'fire' else (1.0 - result['confidence'])
            y_score.append(float(score))

    if not y_true:
        return {"model_name": "UAV-UNet", "error": "No labelled images found", "metrics": None}

    y_pred = [1 if s >= 0.5 else 0 for s in y_score]
    has_both = len(set(y_true)) > 1

    return {
        "model_name": "UAV-UNet",
        "metrics": {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "auc":       float(roc_auc_score(y_true, y_score))  if has_both else None,
            "aupr":      float(average_precision_score(y_true, y_score)) if has_both else None,
        },
    }


if __name__ == "__main__":
    main()