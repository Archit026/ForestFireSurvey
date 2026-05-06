"""
uav_deepfire(himanshu).py  —  DeepFire UAV Detection (PyTorch)
Converted from TensorFlow/Keras to PyTorch.
Uses VGG19 Transfer Learning + traditional ML classifiers.
"""

import os, time, argparse, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

import torch, torch.nn as nn, torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

from Uav.uav_common import model_results_dir, save_binary_confusion_matrix, save_binary_training_curves, save_results_summary

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================
# DATA
# =========================
def check_dataset_exists(data_dir: str) -> bool:
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found at {data_dir}!"); return False
    return True


def load_and_preprocess_data(data_dir: str, img_size=(128, 128), test_size=0.2):
    """Load images with ROI crop, resize and train/test split."""
    images, labels = [], []

    def add_folder(folder: str, label: int):
        if not os.path.exists(folder):
            return
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            mh, mw = int(h * 0.1), int(w * 0.1)
            img = img[mh:h - mh, mw:w - mw]
            img = cv2.cvtColor(cv2.resize(img, img_size), cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)

    base_roots = [Path(data_dir)]
    split_names = ['', 'Training', 'Train', 'Training and Validation', 'Testing', 'Test']
    fire_names = ['fire', 'Fire']
    nofire_names = ['nofire', 'No_Fire', 'NoFire', 'no_fire', 'non_fire', 'non_fire_images']
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
            for name in fire_names:
                add_folder(str(split_dir / name), 1)
            for name in nofire_names:
                add_folder(str(split_dir / name), 0)

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels)
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


# =========================
# TRADITIONAL ML
# =========================
def train_ml_classifiers(X_train, y_train, X_test):
    """Train KNN, NB, SVM, RF, LR on flattened features."""
    Xtr = X_train.reshape(X_train.shape[0], -1)
    Xte = X_test.reshape(X_test.shape[0], -1)
    predictions = {}
    for name, clf in [
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('NB',  GaussianNB()),
        ('SVM', SVC(kernel='linear', probability=True, random_state=42)),
        ('RF',  RandomForestClassifier(n_estimators=100, random_state=42)),
        ('LR',  LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)),
    ]:
        clf.fit(Xtr, y_train); predictions[name] = clf.predict(Xte)
    return predictions


# =========================
# VGG19 TRANSFER LEARNING
# =========================
class _VGG19FireModel(nn.Module):
    def __init__(self, img_size=(128, 128)):
        super().__init__()
        try:
            base = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        except Exception as exc:
            print(f"Warning: pretrained VGG19 weights unavailable ({exc}); using random init.")
            base = models.vgg19(weights=None)
        for p in base.features.parameters():
            p.requires_grad = False
        self.features = base.features
        # Compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size[0], img_size[1])
            flat_sz = self.features(dummy).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_sz, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256),     nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.features(x))


def _np_to_tensor(X: np.ndarray) -> torch.Tensor:
    """(N,H,W,C) float32 [0,1]  →  (N,C,H,W) tensor."""
    return torch.from_numpy(X.transpose(0, 3, 1, 2))


def build_and_train_vgg19(X_train, y_train, X_test, y_test,
                           img_size=(128, 128), epochs=50, batch_size=64, save_path='deepfire_vgg19.pth'):
    model = _VGG19FireModel(img_size).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01, momentum=0.9
    )

    # Augmentation via random horizontal flip only (matches original)
    Xtr_t = _np_to_tensor(X_train); ytr_t = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    Xte_t = _np_to_tensor(X_test);  yte_t = torch.from_numpy(y_test.astype(np.float32)).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(TensorDataset(Xte_t, yte_t), batch_size, shuffle=False, pin_memory=True)

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    for ep in range(1, epochs + 1):
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
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * lbls.size(0)
            train_ok += ((torch.sigmoid(logits) >= 0.5).float() == lbls).sum().item()
            train_total += lbls.size(0)
            running_loss = train_loss / train_total if train_total else 0.0
            running_acc = train_ok / train_total if train_total else 0.0
            batch_time = time.time() - batch_t0
            eta = batch_time * (len(train_loader) - batch_idx)
            print(
                f"  [Epoch {ep}/{epochs}] batch {batch_idx}/{len(train_loader)} "
                f"| batch_time={batch_time:.1f}s | eta={eta:.0f}s | train_loss={running_loss:.4f} | train_acc={running_acc:.4f}"
            )

        model.eval()
        def _acc(loader):
            ok = tot = 0
            with torch.no_grad():
                for imgs, lbls in loader:
                    preds = (torch.sigmoid(model(imgs.to(DEVICE))) >= 0.5).float()
                    ok += (preds == lbls.to(DEVICE)).sum().item(); tot += lbls.size(0)
            return ok / tot if tot else 0.0
        tr_acc, vl_acc = _acc(train_loader), _acc(val_loader)
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs)
                val_loss += criterion(logits, lbls).item() * lbls.size(0)
                val_total += lbls.size(0)
        history['loss'].append(train_loss / train_total if train_total else 0.0)
        history['val_loss'].append(val_loss / val_total if val_total else 0.0)
        history['accuracy'].append(tr_acc); history['val_accuracy'].append(vl_acc)
        if ep % 10 == 0:
            print(f"Epoch {ep}/{epochs}  train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    return model, history


def _get_vgg_probs(model, X: np.ndarray, batch_size=64) -> np.ndarray:
    model.eval(); probs = []
    loader = DataLoader(TensorDataset(_np_to_tensor(X)), batch_size, shuffle=False)
    with torch.no_grad():
        for (imgs,) in loader:
            probs.append(torch.sigmoid(model(imgs.to(DEVICE))).cpu().numpy().flatten())
    return np.concatenate(probs)


# =========================
# EVALUATION & VISUALIZATIONS
# =========================
def evaluate_model(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc  = (tp+tn)/(tp+tn+fp+fn)
    prec = tp/(tp+fp)  if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn)  if (tp+fn) > 0 else 0
    er   = (fp+fn)/(tp+tn+fp+fn)
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    return {'Model': name, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'ER': er, 'F1': f1}


def generate_visualizations(y_test, predictions_dict, vgg_probs, history):
    for name, pred in predictions_dict.items():
        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(4,3)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}'); plt.show()

    fpr, tpr, _ = roc_curve(y_test, vgg_probs)
    plt.figure(); plt.plot(fpr, tpr, label=f'AUC={auc(fpr,tpr):.4f}')
    plt.plot([0,1],[0,1],'k--'); plt.title('ROC'); plt.legend(); plt.show()

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, vgg_probs)
    plt.figure(); plt.plot(rec_arr, prec_arr, label=f'AP={average_precision_score(y_test,vgg_probs):.4f}')
    plt.title('PR Curve'); plt.legend(); plt.show()

    if history:
        plt.figure()
        plt.plot(history['accuracy'], label='Train Acc')
        plt.plot(history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy'); plt.legend(); plt.show()


def simulate_uav_detection(model, image_path, threshold=0.5):
    if not os.path.exists(image_path): return "File not found", 0.0
    img = cv2.cvtColor(cv2.resize(cv2.imread(image_path), (128,128)), cv2.COLOR_BGR2RGB)
    X   = np.array([img], dtype=np.float32) / 255.0
    prob = float(_get_vgg_probs(model, X)[0])
    decision = 'FIRE DETECTED' if prob >= threshold else 'NO FIRE'
    print(f'Image: {image_path} | Prob: {prob:.4f} | Decision: {decision}')
    if decision == 'FIRE DETECTED': print('ACTION: Broadcast to nearby UAVs!')
    return decision, prob


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="DeepFire UAV Detection")
    parser.add_argument('--dataset', default=str(Path(__file__).resolve().parents[1] / 'dataset' / 'uav' / 'FLAME'),
                        help='Root FLAME dataset directory')
    args = parser.parse_args()

    data_dir = args.dataset
    if not check_dataset_exists(data_dir):
        return

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_dir)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    print("Training ML Classifiers...")
    ml_predictions = train_ml_classifiers(X_train, y_train, X_test)

    print("Building and training VGG19...")
    vgg_model, vgg_history = build_and_train_vgg19(X_train, y_train, X_test, y_test)

    vgg_probs = _get_vgg_probs(vgg_model, X_test)
    vgg_pred  = (vgg_probs >= 0.5).astype(int)
    ml_predictions['VGG19-TL'] = vgg_pred

    results = [evaluate_model(name, y_test, pred) for name, pred in ml_predictions.items()]
    df_results = pd.DataFrame(results).round(4)
    print("\n--- Model Comparison Summary ---"); print(df_results.to_string(index=False))
    df_results.to_csv('results_comparison_script.csv', index=False)

    generate_visualizations(y_test, ml_predictions, vgg_probs, vgg_history)


def run(dataset_path, epochs=50, output_dir=None):
    """Standard pipeline interface. Reports VGG19-TL as primary model."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    if not check_dataset_exists(dataset_path):
        return {"model_name": "DeepFire-VGG19", "error": f"Dataset not found: {dataset_path}", "metrics": None}
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
        save_dir = output_dir or str(model_results_dir('deepfire_vgg19'))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'deepfire_vgg19.pth')
        vgg_model, history = build_and_train_vgg19(X_train, y_train, X_test, y_test, epochs=epochs, save_path=save_path)
        vgg_probs = _get_vgg_probs(vgg_model, X_test)
        vgg_pred  = (vgg_probs >= 0.5).astype(int)
        stats = evaluate_model('VGG19-TL', y_test, vgg_pred)
        metrics = {
            "accuracy":  float(stats['Accuracy']),
            "precision": float(stats['Precision']),
            "recall":    float(stats['Recall']),
            "f1":        float(stats['F1']),
            "auc":       float(roc_auc_score(y_test, vgg_probs)) if len(np.unique(y_test)) > 1 else None,
            "aupr":      float(average_precision_score(y_test, vgg_probs)) if len(np.unique(y_test)) > 1 else None,
        }
        save_binary_training_curves(history, os.path.join(save_dir, 'training_curves.png'), 'DeepFire VGG19 Training History')
        save_binary_confusion_matrix(y_test, vgg_pred, os.path.join(save_dir, 'confusion_matrix.png'), 'DeepFire VGG19 Confusion Matrix')
        save_results_summary(os.path.join(save_dir, 'results_summary.txt'), 'DeepFire VGG19 Final Test Results', metrics)
        return {"model_name": "DeepFire-VGG19", "metrics": metrics}
    except Exception as exc:
        return {"model_name": "DeepFire-VGG19", "error": str(exc), "metrics": None}


if __name__ == '__main__':
    main()
