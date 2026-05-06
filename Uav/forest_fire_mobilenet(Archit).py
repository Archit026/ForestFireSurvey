"""
forest_fire_mobilenet(Archit).py  —  MobileNetV2 Forest Fire Detection (PyTorch)
Converted from TensorFlow/Keras to PyTorch.
"""

import os, time, cv2, numpy as np, argparse
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import torch, torch.nn as nn, torchvision.models as models, torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image as PILImage

from Uav.uav_common import model_results_dir, save_binary_confusion_matrix, save_binary_training_curves, save_results_summary

psutil = import_module("psutil") if find_spec("psutil") is not None else None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (128, 128)
BATCH_SIZE = 8


def log(msg):
    print(f"[INFO] {msg}", flush=True)


def log_memory():
    if psutil is None:
        return
    mem = psutil.virtual_memory()
    log(f"RAM Used: {mem.percent}% | Available: {mem.available / (1024**3):.2f} GB")


def log_startup_context():
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(min(4, cpu_count))
    log("===================================================")
    log(f"Device: {DEVICE}")
    log(f"CPU Count: {cpu_count}")
    log(f"Image Size: {IMG_SIZE if 'IMG_SIZE' in globals() else 'custom'}")
    log(f"Batch Size: {BATCH_SIZE if 'BATCH_SIZE' in globals() else 'custom'}")
    log(f"PyTorch threads set to: {torch.get_num_threads()}")
    log("===================================================")


# =========================
# DATASET
# =========================
class FireFolderDataset(Dataset):
    """Loads images from dataset_dir/{fire,non_fire_images}/ folders."""
    CLASS_ALIASES = [
        ('fire', ['Fire']),
        ('non_fire_images', ['No_Fire']),
    ]

    def __init__(self, dataset_dir, transform=None, split='train', val_fraction=0.2):
        log(f"Building {split} dataset...")
        self.transform = transform
        self.samples   = []
        rng = np.random.RandomState(42)
        for cls_idx, (_, folder_names) in enumerate(self.CLASS_ALIASES):
            files = set()
            for folder_name in folder_names:
                log(f"Scanning class: {folder_name}")
                candidates = [
                    os.path.join(dataset_dir, folder_name),
                    os.path.join(dataset_dir, 'Train', folder_name),
                    os.path.join(dataset_dir, 'Training', folder_name),
                    os.path.join(dataset_dir, 'Training and Validation', folder_name),
                ]
                for folder in candidates:
                    if os.path.exists(folder):
                        files.update(
                            os.path.join(folder, f)
                            for f in sorted(os.listdir(folder))
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                        )
            files = list(files)
            log(f"Collected {len(files)} files for class {folder_name}")
            rng.shuffle(files)
            n_val  = max(1, int(len(files) * val_fraction))
            if split == 'train':
                chosen = files[n_val:]
            else:
                chosen = files[:n_val]
            for path in chosen:
                self.samples.append((path, cls_idx))
        log(f"{split.upper()} dataset size: {len(self.samples)}")

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = PILImage.open(path).convert('RGB')
        except Exception as e:
            log(f"Failed loading image: {path} | {e}")
            return self.__getitem__((idx + 1) % len(self.samples))
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
    log(f"Starting MobileNet training for {epochs} epochs...")
    train_start = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    best_acc, best_state = 0.0, None
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(1, epochs + 1):
        log(f"----- Epoch {epoch}/{epochs} -----")
        epoch_t0 = time.time()
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
            log(
                f"[Epoch {epoch}/{epochs}] "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss={running_loss:.4f} | "
                f"Acc={running_acc:.4f} | "
                f"BatchTime={batch_time:.2f}s | "
                f"ETA={eta:.0f}s"
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
        log(
            f"Epoch {epoch} Complete | "
            f"Train Loss={history['loss'][-1]:.4f} | "
            f"Train Acc={history['accuracy'][-1]:.4f} | "
            f"Val Loss={history['val_loss'][-1]:.4f} | "
            f"Val Acc={history['val_accuracy'][-1]:.4f}"
        )
        log(f"Epoch time: {time.time()-epoch_t0:.2f}s")
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            log(f"New best MobileNet model saved with val_acc={acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, save_path)
        log(f"Model saved to {save_path}")
    log("Training finished successfully")
    log(f"Total training time: {time.time()-train_start:.2f}s")
    log_memory()
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
    log_startup_context()
    parser = argparse.ArgumentParser(description="MobileNetV2 Forest Fire Detection (PyTorch)")
    default_root = Path(__file__).resolve().parents[1] / 'dataset' / 'uav' / 'FLAME'
    parser.add_argument('--dataset_dir', default=str(default_root))
    parser.add_argument('--test_dir',    default=str(default_root / 'Test'))
    parser.add_argument('--model_path',  default='./fire_model.pth')
    parser.add_argument('--epochs',      type=int, default=30)
    parser.add_argument('--batch_size',  type=int, default=BATCH_SIZE)
    parser.add_argument('--mode', choices=['train', 'infer', 'all'], default='all')
    args = parser.parse_args()

    if args.mode in ['train', 'all']:
        log("Creating MobileNet DataLoaders...")
        train_ds = FireFolderDataset(args.dataset_dir, _get_transforms(True),  'train')
        val_ds   = FireFolderDataset(args.dataset_dir, _get_transforms(False), 'val')
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        log(f"Train batches: {len(train_loader)}")
        log(f"Validation batches: {len(val_loader)}")
        log("Building MobileNetV2 model...")
        model = build_model()
        log("Model build complete")
        train_model(model, train_loader, val_loader, args.epochs, args.model_path)

    if args.mode in ['infer', 'all']:
        run_inference(args.model_path, args.test_dir)


def run(dataset_path, epochs=30, output_dir=None):
    """Standard pipeline interface. dataset_path: FLAME-style dir."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    log_startup_context()
    if not os.path.exists(dataset_path):
        return {"model_name": "MobileNetV2-UAV", "error": f"Dataset not found: {dataset_path}", "metrics": None}
    try:
        log("Creating MobileNet DataLoaders...")
        train_ds = FireFolderDataset(dataset_path, _get_transforms(True),  'train')
        val_ds   = FireFolderDataset(dataset_path, _get_transforms(False), 'val')
        train_loader = DataLoader(train_ds, 8, shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   8, shuffle=False, num_workers=0, pin_memory=False)
        log(f"Train batches: {len(train_loader)}")
        log(f"Validation batches: {len(val_loader)}")
        log("Building MobileNetV2 model...")
        model = build_model()
        log("Model build complete")
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
