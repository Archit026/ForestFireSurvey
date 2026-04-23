# Forest Fire Survey — Unified Pipeline Summary

## What Was Done

Every model file received a `run(dataset_path)` function added **at the bottom** (no existing logic was changed). `Combine/main.py` orchestrates all of them.

---

## Files Modified

| File | Change |
|---|---|
| `Uav/Uav_dnn(himanshu).py` | Added `run()` — trains CNN, returns AUC/AUPR from softmax scores |
| `Uav/amit_uav_1.py` | Added `run()` — evaluates U-Net on labelled folder; requires pre-trained `.h5` |
| `Uav/forest_fire_mobilenet(Archit).py` | Added `run()` — trains MobileNetV2, evaluates on val split |
| `Uav/uav_deepfire(himanshu).py` | Added `run()` — trains VGG19-TL, AUC/AUPR from sigmoid probs |
| `Uav/uav_fufdet(himanshu).py` | Added `run()` — full FLAME-2 pipeline; AP maps to auc/aupr |
| `Satellite/satellite_modis20(himanshu).py` | Added `run()` — pixel detection; metrics = None (no image-GT) |
| `Man/Alok_forest_fire_pipeline.py` | Added `run()` — trains HOG+AdaBoost + CNN+SVM two-stage |
| `Man/amit_human_1.py` | Added `run()` — YOLOv8 inference on labelled folder; requires pre-trained `.pt` |
| `Man/amit_human_2.py` | Added `run()` — Xception inference on labelled folder; requires pre-trained `.h5` |
| `Man/cf_yolo_pipeline(Archit).py` | Added `run()` — subprocess train+eval; parses COCO JSON |
| `Man/fire_detection_ycbcr(Archit).py` | Added `run()` — rule-based CV; AUC/AUPR = None (no probs) |
| `Man/ft_resnet50_pipeline(Archit).py` | Added `run()` — trains FT-ResNet50; AUC/AUPR from softmax |

## Files Created

| File | Purpose |
|---|---|
| `Combine/main.py` | Orchestrator — runs all models, prints sorted comparison table |
| `Combine/config.json` | Dataset path template |

---

## Standard Interface

Every model now exposes:

```python
run(dataset_path: str) -> {
    "model_name": str,
    "metrics": {
        "accuracy":  float | None,
        "precision": float | None,
        "recall":    float | None,
        "f1":        float | None,
        "auc":       float | None,
        "aupr":      float | None,
    },
    # optional:
    "error": str,   # present if the model failed
    "note":  str,   # present for informational remarks
}
```

---

## Dataset Path Requirements Per Model

| Model | `dataset_path` must contain |
|---|---|
| UAV-DNN | `Testing/fire/`, `Testing/nofire/`, `Training and Validation/fire/`, `Training and Validation/nofire/` |
| UAV-UNet | `fire/`, `nofire/` sub-dirs **+ pre-trained** `models/fire_unet_final.h5` |
| MobileNetV2-UAV | `fire/`, `non_fire_images/` |
| DeepFire-VGG19 | Same Mendeley format as UAV-DNN |
| FuFDet | FLAME-2 root (`254p RGB Images/`, `Frame_Pair_Labels.txt`) |
| MODIS-Satellite | FIRMS `.csv` file **+** `MOD021KM` `.hdf` granule files |
| HOG-AdaBoost+CNN-SVM | `fire/`, `nofire/` (64×64 RGB images) |
| YOLOv8-Man | `fire/`, `nofire/` **+ pre-trained** `models/best.pt` |
| Xception-Man | `fire/`, `nofire/` **+ pre-trained** `models/xception_phase1_best.h5` |
| CF-YOLO | `images/{train,val,test}/`, `labels/{train,val,test}/` (YOLO format) |
| YCbCr-FireDetection | `fire/`, `non_fire/` sequential image sequences |
| FT-ResNet50 | `train/`, `val/`, `test/` (PyTorch `ImageFolder` layout) |

---

## How to Run

**Option A — Config file (recommended):**
```bash
# 1. Edit Combine/config.json — fill in all dataset paths
# 2. Run:
cd ForestFireSurvey/Combine
python main.py --config config.json
```

**Option B — CLI flags:**
```bash
python main.py \
  --uav_dataset       /path/to/mendeley_dataset \
  --man_dataset       /path/to/ground_camera \
  --ft_resnet_dataset /path/to/flame_split \
  ...
```

**Skip slow/unavailable models:**
```bash
python main.py --config config.json --skip fufdet cf_yolo modis_satellite
```

---

## Sample Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FOREST FIRE MODEL COMPARISON  —  sorted by F1 descending
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL                    SOURCE     ACCURACY  PRECISION  RECALL      F1      AUC    AUPR
FT-ResNet50              man          97.20%    96.80%   97.50%   97.15%  99.10%  98.70%
DeepFire-VGG19           uav          95.10%    94.50%   95.70%   95.10%  98.60%  98.00%
...
MODIS-Satellite          satellite      N/A       N/A      N/A      N/A     N/A     N/A
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> [!NOTE]
> Models that display `N/A` for all metrics are pixel-level detectors (MODIS) or detection models without image-level ground truth. They still run and report what they can.

> [!IMPORTANT]
> Models requiring **pre-trained weights** (YOLOv8, Xception, UAV-UNet) return an error if the weight file is missing. Place weights at the paths defined in each file's `MODEL_PATH` constant.
