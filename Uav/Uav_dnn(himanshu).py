"""
Uav_dnn(himanshu).py  —  UAV Fire Detection CNN (PyTorch)
Converted from TensorFlow/Keras to PyTorch.
Replicates the custom CNN with Separable-Conv blocks, BN, Dropout.
"""

import os, time, argparse, cv2, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from Uav.uav_common import model_results_dir, save_binary_confusion_matrix, save_binary_training_curves, save_results_summary

try:
    import psutil
except Exception:
    psutil = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
DATA_DTYPE = np.float16


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
# DATA LOADING
# =========================
def load_data(data_dir, img_size=IMG_SIZE):
    """Loads images from FLAME-style split folders."""
    log("Starting dataset loading...")
    t0 = time.time()
    images, labels = [], []

    def load_specific(folder, label):
        if not os.path.exists(folder):
            return
        files = os.listdir(folder)
        log(f"Scanning folder: {folder}")
        log(f"Found {len(files)} files")
        for idx, fname in enumerate(files, start=1):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(folder, fname))
                if img is not None:
                    img = cv2.cvtColor(cv2.resize(img, img_size), cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(label)
            if idx % 500 == 0:
                log(f"Loaded {idx}/{len(files)} images from {folder}")

    base_roots = [Path(data_dir)]
    split_names = ['', 'Training', 'Train', 'Training and Validation', 'Testing', 'Test']
    seen = set()

    for base in base_roots:
        if not base.exists():
            continue
        for split_name in split_names:
            split_dir = base if split_name == '' else base / split_name
            split_key = str(split_dir.resolve())
            if split_key in seen or not split_dir.exists():
                continue
            seen.add(split_key)
            candidate_folders = {}
            for folder in split_dir.iterdir():
                if not folder.is_dir():
                    continue
                name = folder.name.lower()
                if name == 'fire':
                    candidate_folders[str(folder.resolve())] = 1
                elif name in {'no_fire', 'nofire', 'non_fire', 'non_fire_images'}:
                    candidate_folders[str(folder.resolve())] = 0

            for folder_path, label in candidate_folders.items():
                load_specific(folder_path, label)

    log(f"Dataset loading complete")
    log(f"Total images loaded: {len(images)}")
    log(f"Loading took {time.time()-t0:.2f}s")
    log_memory()
    return images, labels


def augment_and_normalize(images, labels, augment=True):
    log("Starting augmentation + normalization..." if augment else "Starting normalization...")
    t0 = time.time()
    aug_images, aug_labels = [], []
    for idx, (img, lbl) in enumerate(zip(images, labels), start=1):
        aug_images.append(img); aug_labels.append(lbl)
        if augment:
            aug_images.append(cv2.flip(img, 1)); aug_labels.append(lbl)
        if idx % 1000 == 0:
            log(f"Processed {idx}/{len(images)} images")
    log("Converting augmented images to NumPy array...")
    X = np.asarray(aug_images, dtype=DATA_DTYPE) / np.float16(255.0)
    y = np.array(aug_labels)
    log(f"Final dataset shape: {X.shape}")
    log(f"Processing took {time.time()-t0:.2f}s")
    log_memory()
    return X, y


def _to_model_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(X.transpose(0, 3, 1, 2)).float()


# =========================
# MODEL — Separable-Conv CNN
# =========================
class _SeparableConv(nn.Module):
    """Depthwise-separable conv: depthwise + pointwise."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class FireCNN(nn.Module):
    """
    Replicates the Keras model:
      Conv(16) → Pool(3) → SepConv(32) → Pool(3) → SepConv(32) → Pool(3)
      → SepConv(64)×3 → BN → Pool(2) → Flatten → FC(128) → BN → Drop(0.3)
      → FC(128) → BN → Drop(0.3) → FC(64) → BN → Drop(0.2) → FC(2)
    Input: (B, 3, 128, 128)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(3),
            # Block 2
            _SeparableConv(16, 32), nn.MaxPool2d(3),
            _SeparableConv(32, 32), nn.MaxPool2d(3),
            # Block 3
            _SeparableConv(32, 64), _SeparableConv(64, 64), _SeparableConv(64, 64),
            nn.BatchNorm2d(64), nn.MaxPool2d(2),
        )
        # 128 → 42 → 14 → 4 → 2  (after the pooling stack)
        self._flat_size = self._infer_flat(IMG_SIZE[0])
        self.classifier = nn.Sequential(
            nn.Linear(self._flat_size, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.30),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.30),
            nn.Linear(128,  64), nn.BatchNorm1d( 64), nn.ReLU(inplace=True), nn.Dropout(0.20),
            nn.Linear(64, num_classes),
        )

    def _infer_flat(self, img_size):
        dummy = torch.zeros(1, 3, img_size, img_size)
        return self.features(dummy).view(1, -1).shape[1]

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def build_model(input_shape=(128, 128, 3)):
    return FireCNN(num_classes=2).to(DEVICE)


# =========================
# TRAINING
# =========================
def train_model(model, X_train, y_train, X_test, y_test,
                epochs=150, batch_size=BATCH_SIZE, save_path='best_model.pth'):
    # Convert one-hot back to class indices if needed
    if y_train.ndim == 2: y_train = y_train.argmax(1)
    if y_test.ndim  == 2: y_test  = y_test.argmax(1)

    log("Converting NumPy arrays to PyTorch tensors...")
    t0 = time.time()
    X_tr = _to_model_tensor(X_train)
    y_tr = torch.from_numpy(y_train).long()
    X_te = _to_model_tensor(X_test)
    y_te = torch.from_numpy(y_test).long()
    log("Tensor conversion complete")
    log(f"Tensor conversion took {time.time()-t0:.2f}s")
    log_memory()

    log("Creating DataLoaders...")
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(TensorDataset(X_te, y_te), batch_size, shuffle=False, num_workers=0, pin_memory=False)
    log(f"Train batches: {len(train_loader)}")
    log(f"Validation batches: {len(val_loader)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc, best_state = 0.0, None
    t0 = time.time()
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    log(f"Starting training for {epochs} epochs...")
    train_start = time.time()

    for ep in range(1, epochs + 1):
        log(f"----- Epoch {ep}/{epochs} -----")
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        train_ok = 0
        train_total = 0
        for batch_idx, (imgs, lbls) in enumerate(train_loader, start=1):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            batch_t0 = time.time()
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * lbls.size(0)
            train_ok += (logits.argmax(1) == lbls).sum().item()
            train_total += lbls.size(0)
            running_loss = train_loss / train_total if train_total else 0.0
            running_acc = train_ok / train_total if train_total else 0.0
            batch_time = time.time() - batch_t0
            eta = batch_time * (len(train_loader) - batch_idx)
            log(
                f"[Epoch {ep}/{epochs}] "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss={running_loss:.4f} | "
                f"Acc={running_acc:.4f} | "
                f"BatchTime={batch_time:.2f}s | "
                f"ETA={eta:.0f}s"
            )

        # Val accuracy
        model.eval(); correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs)
                preds = logits.argmax(1)
                val_loss += criterion(logits, lbls).item() * lbls.size(0)
                correct += (preds == lbls).sum().item(); total += lbls.size(0)
        acc = correct / total if total else 0.0
        history['loss'].append(train_loss / train_total if train_total else 0.0)
        history['accuracy'].append(train_ok / train_total if train_total else 0.0)
        history['val_loss'].append(val_loss / total if total else 0.0)
        history['val_accuracy'].append(acc)
        log(
            f"Epoch {ep} Complete | "
            f"Train Loss={history['loss'][-1]:.4f} | "
            f"Train Acc={history['accuracy'][-1]:.4f} | "
            f"Val Loss={history['val_loss'][-1]:.4f} | "
            f"Val Acc={history['val_accuracy'][-1]:.4f}"
        )
        log(f"Epoch time: {time.time()-epoch_t0:.2f}s")
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            log(f"New best model saved with val_acc={acc:.4f}")
        if ep % 10 == 0:
            log(f"Epoch {ep}/{epochs}  val_acc={acc:.4f}")
        log_memory()

    log("Training complete")
    log(f"Total training time: {time.time()-train_start:.2f}s")
    return model, history


# =========================
# EVALUATION
# =========================
def evaluate_model(model, X_test, y_test):
    log("Starting evaluation...")
    t0 = time.time()
    if y_test.ndim == 2: y_test = y_test.argmax(1)
    X_te = _to_model_tensor(X_test)
    loader = DataLoader(TensorDataset(X_te), 64, shuffle=False, num_workers=2)
    model.eval(); proba = []
    with torch.no_grad():
        for (imgs,) in loader:
            proba.append(torch.softmax(model(imgs.to(DEVICE)), 1).cpu().numpy())
    proba = np.vstack(proba)
    y_pred = proba.argmax(1)
    kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    log("Evaluation complete")
    log(f"Prediction shape: {proba.shape}")
    log(f"Evaluation took {time.time()-t0:.2f}s")
    log_memory()
    print(f"QWK (Cohen's Kappa): {kappa*100:.2f}%\n{classification_report(y_test, y_pred)}")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Fire','Fire'], yticklabels=['No Fire','Fire'])
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()
    return proba


def perform_inference(model_path, sample_img):
    model = build_model()
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Failed to load: {e}"); return
    model.eval()
    if sample_img.ndim == 3: sample_img = sample_img[np.newaxis]
    x = torch.from_numpy(sample_img.transpose(0,3,1,2)).to(DEVICE)
    with torch.no_grad():
        pred = torch.softmax(model(x), 1).cpu().numpy()
    status = "FIRE DETECTED" if pred.argmax(1)[0] == 1 else "Clear"
    print(f"Inference Result: {status} | Probabilities: {pred}")


# =========================
# MAIN
# =========================
def main():
    log_startup_context()
    parser = argparse.ArgumentParser(description="UAV Fire Detection CNN (PyTorch)")
    parser.add_argument('--dataset', default=str(Path(__file__).resolve().parents[1] / 'dataset' / 'uav' / 'FLAME'),
                        help='Root FLAME dataset directory')
    args = parser.parse_args()

    data_dir = args.dataset
    if not os.path.exists(data_dir): print(f"Data directory '{data_dir}' not found."); return
    log("Loading data..."); images, labels = load_data(data_dir)
    if not images: print("No images found."); return
    log_memory()
    log("Splitting dataset into train/test...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        images, labels, test_size=0.20, random_state=42, stratify=labels
    )
    log(f"Train samples: {len(X_train_raw)}")
    log(f"Test samples: {len(X_test_raw)}")
    X_train, y_train = augment_and_normalize(X_train_raw, y_train_raw, augment=True)
    X_test, y_test = augment_and_normalize(X_test_raw, y_test_raw, augment=False)
    y_train = np.eye(2)[np.asarray(y_train).astype(int)]
    y_test = np.eye(2)[np.asarray(y_test).astype(int)]
    model = build_model(); model.summary = lambda: print(model)
    train_model(model, X_train, y_train, X_test, y_test, epochs=150, batch_size=BATCH_SIZE, save_path='best_model.pth')
    best_model = build_model()
    best_model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    evaluate_model(best_model, X_test, y_test)
    perform_inference('best_model.pth', X_test[0:1])


def run(dataset_path, epochs=150, output_dir=None):
    """Standard pipeline interface. dataset_path: FLAME-style dir."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    log_startup_context()
    if not os.path.exists(dataset_path):
        return {"model_name": "UAV-DNN", "error": f"Dataset not found: {dataset_path}", "metrics": None}
    images, labels = load_data(dataset_path)
    if not images:
        return {"model_name": "UAV-DNN", "error": "No images found", "metrics": None}

    log("Splitting dataset into train/test...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        images, labels, test_size=0.20, random_state=42, stratify=labels
    )
    log(f"Train samples: {len(X_train_raw)}")
    log(f"Test samples: {len(X_test_raw)}")
    X_train, y_train = augment_and_normalize(X_train_raw, y_train_raw, augment=True)
    X_test, y_test = augment_and_normalize(X_test_raw, y_test_raw, augment=False)
    y_train = np.eye(2)[np.asarray(y_train).astype(int)]
    y_test = np.eye(2)[np.asarray(y_test).astype(int)]

    model = build_model()
    save_dir = output_dir or str(model_results_dir('uav_dnn'))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'uav_dnn_best.pth')
    model, history = train_model(model, X_train, y_train, X_test, y_test, epochs=epochs, batch_size=BATCH_SIZE, save_path=save_path)

    best_model = build_model()
    best_model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    best_model.eval()

    y_true_cls = y_test.argmax(1)
    X_te   = _to_model_tensor(X_test)
    loader = DataLoader(TensorDataset(X_te), 64, shuffle=False, num_workers=2)
    proba  = []
    with torch.no_grad():
        for (imgs,) in loader:
            proba.append(torch.softmax(best_model(imgs.to(DEVICE)), 1).cpu().numpy())
    proba = np.vstack(proba); y_pred_cls = proba.argmax(1)

    tp = np.sum((y_pred_cls==1)&(y_true_cls==1)); tn = np.sum((y_pred_cls==0)&(y_true_cls==0))
    fp = np.sum((y_pred_cls==1)&(y_true_cls==0)); fn = np.sum((y_pred_cls==0)&(y_true_cls==1))
    total = tp+tn+fp+fn
    acc  = (tp+tn)/total if total>0 else 0.0; prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0;  f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    has_both = len(np.unique(y_true_cls)) > 1
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_true_cls, proba[:,1])) if has_both else None,
        "aupr": float(average_precision_score(y_true_cls, proba[:,1])) if has_both else None,
    }

    save_binary_training_curves(history, os.path.join(save_dir, 'training_curves.png'), 'UAV DNN Training History')
    save_binary_confusion_matrix(y_true_cls, y_pred_cls, os.path.join(save_dir, 'confusion_matrix.png'), 'UAV DNN Confusion Matrix')
    save_results_summary(os.path.join(save_dir, 'results_summary.txt'), 'UAV DNN Final Test Results', metrics)

    return {"model_name": "UAV-DNN", "metrics": metrics}


if __name__ == '__main__':
    main()
