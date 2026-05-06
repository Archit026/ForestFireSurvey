"""
amit_uav_1.py  —  Fire Segmentation using UAV U-Net (PyTorch)
Converted from TensorFlow/Keras to PyTorch.
"""

import os, cv2, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image as PILImage

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/fire_unet_final.pth"))
IMG_SIZE = 256
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model   = None


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.encoders, self.pools, self.upconvs, self.decoders = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        ch = in_ch
        for f in features:
            self.encoders.append(_DoubleConv(ch, f)); self.pools.append(nn.MaxPool2d(2)); ch = f
        self.bottleneck = _DoubleConv(ch, ch * 2); ch = ch * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.decoders.append(_DoubleConv(f * 2, f)); ch = f
        self.final_conv = nn.Conv2d(ch, out_ch, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape != skip.shape: x = F.interpolate(x, size=skip.shape[2:])
            x = dec(torch.cat([skip, x], dim=1))
        return torch.sigmoid(self.final_conv(x))


def load_uav_model():
    global _model
    if _model is None:
        _model = UNet().to(DEVICE)
        if os.path.exists(MODEL_PATH):
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("✅ UAV U-Net model loaded")
        else:
            print(f"⚠️  Weights not found at {MODEL_PATH}; using random init.")
        _model.eval()
    return _model


def preprocess(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def predict(image):
    model = load_uav_model()
    with torch.no_grad():
        pred = model(preprocess(image))[0, 0].cpu().numpy()
    mask       = (pred > 0.5).astype(np.uint8)
    fire_ratio = mask.sum() / mask.size
    return {"label": "fire" if fire_ratio > 0.01 else "none",
            "confidence": float(pred.max()), "area": float(fire_ratio)}


def draw_mask(image):
    model = load_uav_model()
    with torch.no_grad():
        pred = model(preprocess(image))[0, 0].cpu().numpy()
    mask = cv2.resize((pred > 0.5).astype(np.uint8) * 255, (image.shape[1], image.shape[0]))
    colored = np.zeros_like(image); colored[:, :, 2] = mask
    return cv2.addWeighted(image, 0.7, colored, 0.3, 0)


def main():
    if not os.path.exists("test.jpg"): print("⚠️ test.jpg not found"); return
    print("\n🔥 UAV Prediction:"); print(predict(cv2.imread("test.jpg")))


def run(dataset_path):
    """Standard pipeline interface. dataset_path: dir with 'fire/' and 'nofire/' sub-folders."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    if not os.path.exists(MODEL_PATH):
        return {"model_name": "UAV-UNet", "error": f"Model not found: {MODEL_PATH}", "metrics": None}
    if not os.path.exists(dataset_path):
        return {"model_name": "UAV-UNet", "error": f"Dataset not found: {dataset_path}", "metrics": None}
    load_uav_model()
    y_true, y_score = [], []
    for label, folder in [(1, 'fire'), (0, 'nofire')]:
        fp = os.path.join(dataset_path, folder)
        if not os.path.exists(fp): continue
        for fname in sorted(os.listdir(fp)):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            img = cv2.imread(os.path.join(fp, fname))
            if img is None: continue
            r = predict(img); y_true.append(label)
            y_score.append(r['confidence'] if r['label'] == 'fire' else 1.0 - r['confidence'])
    if not y_true: return {"model_name": "UAV-UNet", "error": "No labelled images found", "metrics": None}
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]; has_both = len(set(y_true)) > 1
    return {"model_name": "UAV-UNet", "metrics": {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "auc":       float(roc_auc_score(y_true, y_score))           if has_both else None,
        "aupr":      float(average_precision_score(y_true, y_score)) if has_both else None,
    }}


if __name__ == "__main__":
    main()