# """
# amit_human_1.py
# ----------------
# Fire & Smoke Detection using YOLOv8

# - Can run standalone
# - Also exposes predict() for team integration
# """

import os
import cv2
import pandas as pd
from ultralytics import YOLO


# =========================
# CONFIG
# =========================
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/best.pt")
)
CONF_THRESHOLD = 0.25


# =========================
# LOAD MODEL (singleton style)
# =========================
_model = None


def load_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
        print("✅ YOLO model loaded")
    return _model


# =========================
# CORE PREDICTION FUNCTION (IMPORTANT)
# =========================
def predict(image):
    """
    Input:
        image (numpy array)

    Output:
        {
            "label": "fire" / "smoke" / "none",
            "confidence": float
        }
    """
    model = load_model()

    results = model(image, conf=CONF_THRESHOLD)

    best_label = "none"
    best_conf = 0.0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf > best_conf:
                best_conf = conf
                best_label = label

    return {
        "label": best_label,
        "confidence": best_conf
    }


# =========================
# OPTIONAL: DRAW BOXES
# =========================
def draw_boxes(image):
    model = load_model()
    results = model(image, conf=CONF_THRESHOLD)

    output = image.copy()

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            color = (0, 255, 0) if label == "fire" else (255, 0, 0)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return output


# =========================
# PROCESS IMAGE FOLDER
# =========================
def process_folder(folder_path):
    model = load_model()
    results_data = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, file)
            image = cv2.imread(path)

            pred = predict(image)

            results_data.append({
                "image": file,
                "label": pred["label"],
                "confidence": pred["confidence"]
            })

    return pd.DataFrame(results_data)


# =========================
# MAIN (for standalone run)
# =========================
def main():
    # Test with one image
    test_image_path = "test.jpg"  # change if needed

    if not os.path.exists(test_image_path):
        print("⚠️ test.jpg not found")
        return

    image = cv2.imread(test_image_path)

    result = predict(image)

    print("\n🔥 Prediction Result:")
    print(result)

    # Show detection
    output = draw_boxes(image)
    cv2.imshow("Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run(dataset_path):
    """Standard pipeline interface.
    dataset_path: root dir with 'fire/' and 'nofire/' sub-folders.
    Requires pre-trained YOLOv8 weights at MODEL_PATH.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )
    import numpy as np

    if not os.path.exists(MODEL_PATH):
        return {"model_name": "YOLOv8-Man", "error": f"Model not found: {MODEL_PATH}", "metrics": None}

    if not os.path.exists(dataset_path):
        return {"model_name": "YOLOv8-Man", "error": f"Dataset not found: {dataset_path}", "metrics": None}

    load_model()  # warm-up singleton

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
            # score = confidence that image is fire
            score = result['confidence'] if result['label'] == 'fire' else 0.0
            y_score.append(float(score))

    if not y_true:
        return {"model_name": "YOLOv8-Man", "error": "No labelled images found", "metrics": None}

    y_pred   = [1 if s >= CONF_THRESHOLD else 0 for s in y_score]
    has_both = len(set(y_true)) > 1

    return {
        "model_name": "YOLOv8-Man",
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