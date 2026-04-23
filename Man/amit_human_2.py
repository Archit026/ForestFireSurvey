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


if __name__ == "__main__":
    main()