"""
forest_fire_mobilenet(Archit).py  —  MobileNetV2 Forest Fire Detection (PyTorch)
Converted from TensorFlow/Keras to PyTorch.
"""

import os, cv2, numpy as np, argparse
from pathlib import Path
import torch, torch.nn as nn, torchvision.models as models, torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image as PILImage

from Uav.uav_common import model_results_dir, save_binary_confusion_matrix, save_binary_training_curves, save_results_summary

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (224, 224)


# =========================
# DATASET
# =========================
class FireFolderDataset(Dataset):
    """Loads images from dataset_dir/{fire,non_fire_images}/ folders."""
    CLASS_ALIASES = [
        ('fire', ['fire', 'Fire']),
        ('non_fire_images', ['non_fire_images', 'No_Fire', 'NoFire', 'nofire', 'no_fire', 'non_fire']),
    ]

    def __init__(self, dataset_dir, transform=None, split='train', val_fraction=0.2):
        self.transform = transform
        self.samples   = []
        rng = np.random.RandomState(42)
        for cls_idx, (_, folder_names) in enumerate(self.CLASS_ALIASES):
            files = []
            for folder_name in folder_names:
                candidates = [
                    os.path.join(dataset_dir, folder_name),
                    os.path.join(dataset_dir, 'Train', folder_name),
                    os.path.join(dataset_dir, 'Training', folder_name),
                    os.path.join(dataset_dir, 'Testing', folder_name),
                    os.path.join(dataset_dir, 'Test', folder_name),
                    os.path.join(dataset_dir, 'Training and Validation', folder_name),
                ]
                for folder in candidates:
                    if os.path.exists(folder):
                        files.extend(
                            os.path.join(folder, f)
                            for f in sorted(os.listdir(folder))
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                        )
            rng.shuffle(files)
            n_val  = max(1, int(len(files) * val_fraction))
            if split == 'train':
                chosen = files[n_val:]
            else:
                chosen = files[:n_val]
            for path in chosen:
                self.samples.append((path, cls_idx))

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = PILImage.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def _get_transforms(augment=True):
    if augment:
        return T.Compose([
            T.Resize(IMG_SIZE), T.RandomHorizontalFlip(), T.RandomRotation(20),
            T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize(IMG_SIZE), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# =========================
# MODEL
# =========================
def build_model(num_classes=2, freeze_base=True):
    """MobileNetV2 with custom binary head."""
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    except Exception as exc:
        print(f"Warning: pretrained MobileNetV2 weights unavailable ({exc}); using random init.")
        model = models.mobilenet_v2(weights=None)
    if freeze_base:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_features, num_classes),
    )
    return model.to(DEVICE)


# =========================
# TRAIN / EVAL
# =========================
def train_model(model, train_loader, val_loader, epochs=30, save_path='./fire_model.pth'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    best_acc, best_state = 0.0, None
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    log_every = max(1, len(train_loader) // 5)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_ok = 0
        train_total = 0
        for batch_idx, (imgs, lbls) in enumerate(train_loader, start=1):
            batch_t0 = time.time()
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * lbls.size(0)
            train_ok += (logits.argmax(1) == lbls).sum().item()
            train_total += lbls.size(0)
            running_loss = train_loss / train_total if train_total else 0.0
            running_acc = train_ok / train_total if train_total else 0.0
            batch_time = time.time() - batch_t0
            eta = batch_time * (len(train_loader) - batch_idx)
            print(
                f"  [Epoch {epoch}/{epochs}] batch {batch_idx}/{len(train_loader)} "
                f"| batch_time={batch_time:.1f}s | eta={eta:.0f}s | train_loss={running_loss:.4f} | train_acc={running_acc:.4f}"
            )

        # Validation
        model.eval(); correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs)
                preds = logits.argmax(1)
                val_loss += criterion(logits, lbls).item() * lbls.size(0)
                correct += (preds == lbls).sum().item(); total += lbls.size(0)
        acc = correct / total if total > 0 else 0
        history['loss'].append(train_loss / train_total if train_total else 0.0)
        history['accuracy'].append(train_ok / train_total if train_total else 0.0)
        history['val_loss'].append(val_loss / total if total else 0.0)
        history['val_accuracy'].append(acc)
        print(f"Epoch {epoch}/{epochs}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, save_path)
        print(f"Model saved to {save_path}")
    return model, history


def run_inference(model_path, test_dir):
    if not os.path.exists(model_path): print(f"Model not found at {model_path}"); return
    model = build_model(2, freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    tf = _get_transforms(augment=False)
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not images: print("No images found in test folder."); return
    path = os.path.join(test_dir, images[0])
    img  = PILImage.open(path).convert('RGB')
    with torch.no_grad():
        pred = model(tf(img).unsqueeze(0).to(DEVICE))
    cls = pred.argmax(1).item()
    print(f"Prediction: {'FIRE 🔥' if cls == 0 else 'NON FIRE 🌲'}")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 Forest Fire Detection (PyTorch)")
    default_root = Path(__file__).resolve().parents[1] / 'dataset' / 'uav' / 'FLAME'
    parser.add_argument('--dataset_dir', default=str(default_root))
    parser.add_argument('--test_dir',    default=str(default_root / 'Test'))
    parser.add_argument('--model_path',  default='./fire_model.pth')
    parser.add_argument('--epochs',      type=int, default=30)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--mode', choices=['train', 'infer', 'all'], default='all')
    args = parser.parse_args()

    if args.mode in ['train', 'all']:
        train_ds = FireFolderDataset(args.dataset_dir, _get_transforms(True),  'train')
        val_ds   = FireFolderDataset(args.dataset_dir, _get_transforms(False), 'val')
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=0)
        model = build_model(); train_model(model, train_loader, val_loader, args.epochs, args.model_path)

    if args.mode in ['infer', 'all']:
        run_inference(args.model_path, args.test_dir)


def run(dataset_path, epochs=30, output_dir=None):
    """Standard pipeline interface. dataset_path: FLAME-style dir."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    if not os.path.exists(dataset_path):
        return {"model_name": "MobileNetV2-UAV", "error": f"Dataset not found: {dataset_path}", "metrics": None}
    try:
        train_ds = FireFolderDataset(dataset_path, _get_transforms(True),  'train')
        val_ds   = FireFolderDataset(dataset_path, _get_transforms(False), 'val')
        train_loader = DataLoader(train_ds, 32, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   32, shuffle=False, num_workers=0)
        model = build_model()
        save_dir = output_dir or str(model_results_dir('mobilenet_uav'))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'mobilenet_fire.pth')
        model, history = train_model(model, train_loader, val_loader, epochs, save_path)

        model.eval()
        y_score_raw, y_true_raw = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                probs = torch.softmax(model(imgs.to(DEVICE)), dim=1)[:, 0].cpu().numpy()
                y_score_raw.extend(probs); y_true_raw.extend(lbls.numpy())

        y_true_raw  = np.array(y_true_raw)
        y_score_raw = np.array(y_score_raw)
        # class 0 = fire in FireFolderDataset
        y_true_bin   = (y_true_raw == 0).astype(int)
        y_score_fire = y_score_raw
        y_pred   = (y_score_fire >= 0.5).astype(int)
        has_both = len(np.unique(y_true_bin)) > 1
        metrics = {
            "accuracy":  float(accuracy_score(y_true_bin, y_pred)),
            "precision": float(precision_score(y_true_bin, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true_bin, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true_bin, y_pred, zero_division=0)),
            "auc":       float(roc_auc_score(y_true_bin, y_score_fire)) if has_both else None,
            "aupr":      float(average_precision_score(y_true_bin, y_score_fire)) if has_both else None,
        }

        save_binary_training_curves(history, os.path.join(save_dir, 'training_curves.png'), 'MobileNetV2 UAV Training History')
        save_binary_confusion_matrix(y_true_bin, y_pred, os.path.join(save_dir, 'confusion_matrix.png'), 'MobileNetV2 UAV Confusion Matrix')
        save_results_summary(os.path.join(save_dir, 'results_summary.txt'), 'MobileNetV2 UAV Final Test Results', metrics)

        return {"model_name": "MobileNetV2-UAV", "metrics": metrics}
    except Exception as exc:
        return {"model_name": "MobileNetV2-UAV", "error": str(exc), "metrics": None}


if __name__ == "__main__":
    main()
