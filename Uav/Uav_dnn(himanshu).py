"""
Uav_dnn(himanshu).py  —  UAV Fire Detection CNN (PyTorch)
Converted from TensorFlow/Keras to PyTorch.
Replicates the custom CNN with Separable-Conv blocks, BN, Dropout.
"""

import os, time, cv2, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================
# DATA LOADING
# =========================
def load_data(data_dir, img_size=(250, 250)):
    """Loads images from 'Testing' and 'Training and Validation' subdirs."""
    images, labels = [], []

    def load_specific(folder, label):
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} not found."); return
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(folder, fname))
                if img is not None:
                    img = cv2.cvtColor(cv2.resize(img, img_size), cv2.COLOR_BGR2RGB)
                    images.append(img); labels.append(label)

    load_specific(os.path.join(data_dir, 'Testing', 'fire'),   1)
    load_specific(os.path.join(data_dir, 'Testing', 'nofire'), 0)
    load_specific(os.path.join(data_dir, 'Training and Validation', 'fire'),   1)
    load_specific(os.path.join(data_dir, 'Training and Validation', 'nofire'), 0)
    return images, labels


def augment_and_normalize(images, labels):
    aug_images, aug_labels = [], []
    for img, lbl in zip(images, labels):
        aug_images.append(img); aug_labels.append(lbl)
        aug_images.append(cv2.flip(img, 1)); aug_labels.append(lbl)
    X = np.array(aug_images, dtype=np.float32) / 255.0
    y = np.array(aug_labels)
    return X, y


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
    Input: (B, 3, 250, 250)
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
        # 250 → 83 → 27 → 9 → 4  (≈4 after three /3 pools and one /2)
        self._flat_size = self._infer_flat(250)
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


def build_model(input_shape=(250, 250, 3)):
    return FireCNN(num_classes=2).to(DEVICE)


# =========================
# TRAINING
# =========================
def train_model(model, X_train, y_train, X_test, y_test,
                epochs=150, batch_size=32, save_path='best_model.pth'):
    # Convert one-hot back to class indices if needed
    if y_train.ndim == 2: y_train = y_train.argmax(1)
    if y_test.ndim  == 2: y_test  = y_test.argmax(1)

    X_tr = torch.from_numpy(X_train.transpose(0,3,1,2))
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test.transpose(0,3,1,2))
    y_te = torch.from_numpy(y_test).long()

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(TensorDataset(X_te, y_te), batch_size, shuffle=False, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc, best_state = 0.0, None
    t0 = time.time()
    history = {'accuracy': [], 'val_accuracy': []}

    for ep in range(1, epochs + 1):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(imgs), lbls).backward()
            optimizer.step()

        # Val accuracy
        model.eval(); correct = total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                preds = model(imgs).argmax(1)
                correct += (preds == lbls).sum().item(); total += lbls.size(0)
        acc = correct / total if total else 0.0
        history['val_accuracy'].append(acc)
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
        if ep % 10 == 0:
            print(f"Epoch {ep}/{epochs}  val_acc={acc:.4f}")

    print(f"Training Time: {time.time()-t0:.2f}s")
    return history


# =========================
# EVALUATION
# =========================
def evaluate_model(model, X_test, y_test):
    if y_test.ndim == 2: y_test = y_test.argmax(1)
    X_te = torch.from_numpy(X_test.transpose(0,3,1,2))
    loader = DataLoader(TensorDataset(X_te), 64, shuffle=False)
    model.eval(); proba = []
    with torch.no_grad():
        for (imgs,) in loader:
            proba.append(torch.softmax(model(imgs.to(DEVICE)), 1).cpu().numpy())
    proba = np.vstack(proba)
    y_pred = proba.argmax(1)
    kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')
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
    data_dir = 'mendeley_dataset'
    if not os.path.exists(data_dir): print(f"Data directory '{data_dir}' not found."); return
    print("Loading data..."); images, labels = load_data(data_dir)
    if not images: print("No images found."); return
    X_norm, y_aug = augment_and_normalize(images, labels)
    y_cat = np.eye(2)[y_aug.astype(int)]
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_cat, test_size=0.20, random_state=42, stratify=y_cat)
    model = build_model(); model.summary = lambda: print(model)
    train_model(model, X_train, y_train, X_test, y_test, epochs=150, batch_size=32, save_path='best_model.pth')
    best_model = build_model()
    best_model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    evaluate_model(best_model, X_test, y_test)
    perform_inference('best_model.pth', X_test[0:1])


def run(dataset_path, epochs=150):
    """Standard pipeline interface. dataset_path: Mendeley-style dir."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    if not os.path.exists(dataset_path):
        return {"model_name": "UAV-DNN", "error": f"Dataset not found: {dataset_path}", "metrics": None}
    images, labels = load_data(dataset_path)
    if not images:
        return {"model_name": "UAV-DNN", "error": "No images found", "metrics": None}

    X_norm, y_aug = augment_and_normalize(images, labels)
    y_cat = np.eye(2)[y_aug.astype(int)]
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_cat, test_size=0.20, random_state=42, stratify=y_cat)

    model = build_model()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uav_dnn_best.pth')
    train_model(model, X_train, y_train, X_test, y_test, epochs=epochs, batch_size=32, save_path=save_path)

    best_model = build_model()
    best_model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    best_model.eval()

    y_true_cls = y_test.argmax(1)
    X_te   = torch.from_numpy(X_test.transpose(0,3,1,2))
    loader = DataLoader(TensorDataset(X_te), 64, shuffle=False)
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

    return {"model_name": "UAV-DNN", "metrics": {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "auc":  float(roc_auc_score(y_true_cls, proba[:,1])),
        "aupr": float(average_precision_score(y_true_cls, proba[:,1])),
    }}


if __name__ == '__main__':
    main()
