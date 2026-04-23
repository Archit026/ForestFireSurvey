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


if __name__ == "__main__":
    main()