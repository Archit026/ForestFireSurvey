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


if __name__ == "__main__":
    main()
    
    
    
    # Your teammates can now do
    
    # # from Man.amit_human_1 import predict
    # # result = predict(image)