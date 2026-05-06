"""
forest_fire_pipeline.py  —  Forest Fire Recognition Pipeline (single-file)

Implements the two-stage fire detection system from the paper:
    "Forest fire recognition based on deep learning methods"
    Optics & Lasers in Engineering  (S0030402620313279)

Dataset type: Hand-held / ground-based camera images (64×64 RGB)
    fire/    — fire scene images
    nofire/  — background images (rain, shine, sunrise categories)

Pipeline stages:
    1. GAN augmentation  — DCGAN synthesises extra fire / nofire images
    2. HOG + AdaBoost    — fast high-recall Stage-1 screening
    3. CNN + SVM         — high-precision Stage-2 confirmation
    4. Two-stage eval    — combined pipeline metrics on the test set

Usage:
    python forest_fire_pipeline.py                        # full pipeline
    python forest_fire_pipeline.py --mode gan             # GAN only
    python forest_fire_pipeline.py --mode hog             # HOG+AdaBoost only
    python forest_fire_pipeline.py --mode cnn             # CNN+SVM only
    python forest_fire_pipeline.py --mode eval            # evaluation only
    python forest_fire_pipeline.py --mode all --use_gan   # use GAN images
    python forest_fire_pipeline.py --mode all --gan_epochs 50 --cnn_epochs 50
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import argparse
import time
from pathlib import Path


# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')                       # non-interactive backend
import matplotlib.pyplot as plt

from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from skimage.feature import hog

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS AND HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Data ─────────────────────────────────────────────────────────────────────
IMG_SIZE    = 64
FIRE_LABEL  = 1
NOFIRE_LABEL = 0
VALID_EXTS  = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

# ── GAN (Table 1) ────────────────────────────────────────────────────────────
GAN_BATCH_SIZE = 64
GAN_NOISE_DIM  = 100
GAN_EPOCHS     = 500
GAN_LR         = 0.0001
GAN_DROPOUT    = 0.10      # keep_prob=0.9 → dropout=0.1

# ── HOG features ─────────────────────────────────────────────────────────────
HOG_PARAMS = dict(
    orientations    = 9,
    pixels_per_cell = (8, 8),
    cells_per_block = (2, 2),
    block_norm      = 'L2-Hys',
    channel_axis    = -1,
)

# ── AdaBoost ─────────────────────────────────────────────────────────────────
ADA_N_ESTIMATORS = 200
ADA_LR           = 1.0

# ── CNN (Table 2) ────────────────────────────────────────────────────────────
CNN_EPOCHS      = 500
CNN_BATCH_SIZE  = 32
CNN_INITIAL_LR  = 0.005
CNN_LR_DECAY    = 0.2
CNN_DROPOUT     = 0.1
CNN_FEATURE_DIM = 1024

# ── SVM ──────────────────────────────────────────────────────────────────────
SVM_KERNEL = 'rbf'
SVM_C      = 1.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════════════════════
#  1. DATA UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _load_images_from_dir(directory: str, label: int, max_images=None):
    """Load all valid images from a directory and return (images, labels)."""
    directory = Path(directory)
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in VALID_EXTS)
    if max_images:
        paths = paths[:max_images]

    images, labels, skipped = [], [], 0
    for p in paths:
        try:
            img = PILImage.open(p).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(label)
        except Exception:
            skipped += 1

    if skipped:
        print(f"  [warn] Skipped {skipped} unreadable files in {directory.name}/")
    return images, labels


def load_dataset(
    base_dir: str,
    fire_subdir:    str = 'fire',
    nofire_subdir:  str = 'nofire',
    max_fire:       int = None,
    max_nofire:     int = None,
    also_generated: bool = False,
):
    """
    Load fire and nofire images from disk.

    Returns
    -------
    X : np.ndarray (N, 64, 64, 3)  float32 [0,1]
    y : np.ndarray (N,)            int32  {0=nofire, 1=fire}
    """
    base_dir = Path(base_dir)
    if nofire_subdir == 'nofire' and not (base_dir / nofire_subdir).exists():
        for candidate in ('non_fire_images', 'non_fire', 'no_fire'):
            if (base_dir / candidate).exists():
                nofire_subdir = candidate
                break
    print("=" * 55)
    print("Loading dataset …")

    fire_imgs,   fire_lbs   = _load_images_from_dir(base_dir / fire_subdir,   FIRE_LABEL,   max_fire)
    nofire_imgs, nofire_lbs = _load_images_from_dir(base_dir / nofire_subdir, NOFIRE_LABEL, max_nofire)
    print(f"  Fire images   : {len(fire_imgs)}")
    print(f"  NonFire images: {len(nofire_imgs)}")

    if also_generated:
        for subdir, label, tag in [
            ('generated/fire',   FIRE_LABEL,   'fire'),
            ('generated/nofire', NOFIRE_LABEL, 'nofire'),
        ]:
            gen_dir = base_dir / subdir
            if gen_dir.exists():
                imgs, lbs = _load_images_from_dir(gen_dir, label)
                fire_imgs   += imgs if label == FIRE_LABEL   else []
                nofire_imgs += imgs if label == NOFIRE_LABEL else []
                fire_lbs    += lbs  if label == FIRE_LABEL   else []
                nofire_lbs  += lbs  if label == NOFIRE_LABEL else []
                print(f"  + Generated {tag}: {len(imgs)}")

    X = np.array(fire_imgs + nofire_imgs, dtype=np.float32)
    y = np.array(fire_lbs + nofire_lbs,  dtype=np.int32)
    print(f"  Total: {len(X)}  (fire={int(y.sum())}, nofire={int((y==0).sum())})")
    print("=" * 55)
    return X, y


def split_dataset(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split into train / validation / test sets (stratified).

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=rel_val, random_state=random_state, stratify=y_tv
    )
    print(f"Split → train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════════════════════════════════════════════════════════════
#  2. GAN — DCGAN FOR DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """FC -> reshape (4x4x256) -> ConvTranspose stack -> 64x64x3 tanh."""
    def __init__(self, noise_dim: int = GAN_NOISE_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 4 * 4 * 256, bias=False),
            nn.BatchNorm1d(4 * 4 * 256),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 256, 4, 4)
        return self.deconv(x)


class Discriminator(nn.Module):
    """Standard DCGAN discriminator with Dropout(0.1)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(GAN_DROPOUT),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(GAN_DROPOUT),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(GAN_DROPOUT),
            nn.Flatten(),
            nn.Linear(8 * 8 * 256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _build_generator(noise_dim: int = GAN_NOISE_DIM) -> Generator:
    return Generator(noise_dim).to(DEVICE)


def _build_discriminator() -> Discriminator:
    return Discriminator().to(DEVICE)


_cross_entropy = nn.BCELoss()


def _gan_train_step(real_images, generator, discriminator, g_opt, d_opt):
    """Single PyTorch DCGAN training step."""
    batch_size = real_images.size(0)
    noise = torch.randn(batch_size, GAN_NOISE_DIM, device=DEVICE)

    d_opt.zero_grad()
    fake_images = generator(noise).detach()
    real_out = discriminator(real_images)
    fake_out = discriminator(fake_images)
    d_real_loss = _cross_entropy(real_out, torch.full_like(real_out, 0.9))
    d_fake_loss = _cross_entropy(fake_out, torch.full_like(fake_out, 0.1))
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    d_opt.step()

    g_opt.zero_grad()
    fake_images = generator(noise)
    fake_out = discriminator(fake_images)
    g_loss = _cross_entropy(fake_out, torch.ones_like(fake_out))
    g_loss.backward()
    g_opt.step()

    return g_loss.item(), d_loss.item(), d_real_loss.item(), d_fake_loss.item()


def train_gan(
    real_images: np.ndarray,
    label_name:  str,
    save_dir:    Path,
    model_dir:   Path,
    results_dir: Path,
    n_generate:  int = 500,
    epochs:      int = GAN_EPOCHS,
):
    """
    Train a DCGAN on `real_images`, save the generator, and generate
    `n_generate` synthetic images to `save_dir`.

    real_images : (N, 64, 64, 3) float32 in [0, 1]
    """
    print(f"\n{'='*55}")
    print(f"GAN [{label_name}]  images={len(real_images)}, epochs={epochs}")
    print(f"{'='*55}")

    imgs_scaled = (real_images * 2.0) - 1.0   # [0,1] → [-1,1] for tanh
    tensor = torch.from_numpy(imgs_scaled.transpose(0, 3, 1, 2)).float()
    dataset = DataLoader(
        TensorDataset(tensor),
        batch_size=GAN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    generator     = _build_generator()
    discriminator = _build_discriminator()
    g_opt = torch.optim.Adam(generator.parameters(), lr=GAN_LR, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=GAN_LR, betas=(0.5, 0.999))

    history = {'g': [], 'd': [], 'd_real': [], 'd_fake': []}
    for epoch in range(1, epochs + 1):
        g_ls, d_ls, dr_ls, df_ls = [], [], [], []
        for (batch,) in dataset:
            batch = batch.to(DEVICE)
            gl, dl, drl, dfl = _gan_train_step(batch, generator, discriminator, g_opt, d_opt)
            g_ls.append(gl); d_ls.append(dl)
            dr_ls.append(drl); df_ls.append(dfl)

        history['g'].append(np.mean(g_ls));      history['d'].append(np.mean(d_ls))
        history['d_real'].append(np.mean(dr_ls)); history['d_fake'].append(np.mean(df_ls))

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"G={history['g'][-1]:.4f}  D={history['d'][-1]:.4f}  "
                  f"D_real={history['d_real'][-1]:.4f}  D_fake={history['d_fake'][-1]:.4f}")

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), str(model_dir / f'gan_generator_{label_name}.pth'))
    print(f"Generator saved -> models/gan_generator_{label_name}.pth")

    # Plot training curves
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for key, color, lbl in [('d',      'blue',   'Discriminator Total'),
                             ('d_real', 'orange', 'Discriminator Real'),
                             ('d_fake', 'green',  'Discriminator Fake'),
                             ('g',      'red',    'Generator')]:
        plt.plot(range(1, epochs+1), history[key], label=lbl, color=color)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'GAN Training Losses [{label_name}]')
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(str(results_dir / f'gan_loss_{label_name}.png'), dpi=150)
    plt.close()

    # Generate synthetic images
    save_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    noise = torch.randn(n_generate, GAN_NOISE_DIM, device=DEVICE)
    with torch.no_grad():
        generated = generator(noise).cpu().numpy().transpose(0, 2, 3, 1)
    generated = ((generated + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    for i, img_arr in enumerate(generated):
        PILImage.fromarray(img_arr).save(str(save_dir / f'gen_{i:05d}.png'))
    print(f"Generated {n_generate} images → {save_dir}")

    return history


# ═══════════════════════════════════════════════════════════════════════════════
#  3. HOG + ADABOOST  (Stage 1 — fast screening)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """Extract HOG feature vectors from (N, 64, 64, 3) images → (N, D) float32."""
    return np.array([hog(img, **HOG_PARAMS) for img in images], dtype=np.float32)


def train_hog_adaboost(X_train, y_train, X_val, y_val, model_dir: Path, results_dir: Path):
    """
    Extract HOG features, train AdaBoost, save model and evaluation plots.

    Returns clf, scaler, scores_dict
    """
    model_dir.mkdir(parents=True,   exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*55)
    print("Stage 1: HOG + AdaBoost")
    print("="*55)

    print("Extracting HOG features …")
    X_tr_hog = extract_hog_features(X_train)
    X_v_hog  = extract_hog_features(X_val)
    print(f"  Feature dim: {X_tr_hog.shape[1]}")

    scaler   = StandardScaler()
    X_tr_hog = scaler.fit_transform(X_tr_hog)
    X_v_hog  = scaler.transform(X_v_hog)

    clf = AdaBoostClassifier(
        estimator     = DecisionTreeClassifier(max_depth=1),
        n_estimators  = ADA_N_ESTIMATORS,
        learning_rate = ADA_LR,
        random_state  = 42,
    )
    print(f"Training AdaBoost (n_estimators={ADA_N_ESTIMATORS}) …")
    clf.fit(X_tr_hog, y_train)

    y_pred = clf.predict(X_v_hog)
    acc    = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=['NoFire', 'Fire'], digits=4)
    print(f"\nValidation Accuracy: {acc:.4f}\n{report}")

    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['NoFire', 'Fire']).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('HOG + AdaBoost — Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig(str(results_dir / 'hog_adaboost_cm.png'), dpi=150)
    plt.close()

    joblib.dump(clf,    str(model_dir / 'hog_adaboost.pkl'))
    joblib.dump(scaler, str(model_dir / 'hog_scaler.pkl'))
    print("Model saved → models/hog_adaboost.pkl")

    return clf, scaler, {'accuracy': acc, 'report': report}


def load_hog_adaboost(model_dir):
    """Load saved AdaBoost classifier and scaler."""
    model_dir = Path(model_dir)
    return (joblib.load(str(model_dir / 'hog_adaboost.pkl')),
            joblib.load(str(model_dir / 'hog_scaler.pkl')))


def predict_hog(images: np.ndarray, clf, scaler) -> np.ndarray:
    """HOG+AdaBoost inference → (N,) int {0=nofire, 1=fire}."""
    feats = extract_hog_features(images)
    return clf.predict(scaler.transform(feats))


# ═══════════════════════════════════════════════════════════════════════════════
#  4. CNN + SVM  (Stage 2 — high-precision confirmation)
# ═══════════════════════════════════════════════════════════════════════════════

class CNNBackbone(nn.Module):
    """CNN backbone from the paper with a 1024-d feature layer."""
    def __init__(self, feature_dim: int = CNN_FEATURE_DIM, dropout: float = CNN_DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.fc_features = nn.Sequential(
            nn.Linear(64 * 16 * 16, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.softmax_head = nn.Linear(feature_dim, 2)

    def forward(self, x, return_features: bool = False):
        x = self.features(x)
        feats = self.fc_features(x)
        if return_features:
            return feats
        return self.softmax_head(feats)


def _to_torch_images(images: np.ndarray) -> torch.Tensor:
    """Convert NHWC float images to NCHW torch tensors."""
    return torch.from_numpy(images.transpose(0, 3, 1, 2)).float()


def build_cnn_backbone():
    """Return the PyTorch CNN used for pre-training and feature extraction."""
    return CNNBackbone().to(DEVICE)


def extract_cnn_features(feature_model, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Extract 1024-d CNN features from NHWC image arrays."""
    feature_model.eval()
    tensor = _to_torch_images(images)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for (imgs,) in loader:
            feats.append(feature_model(imgs.to(DEVICE), return_features=True).cpu().numpy())
    return np.vstack(feats)


def train_cnn(X_train, y_train, X_val, y_val, model_dir: Path, results_dir: Path, epochs=CNN_EPOCHS):
    """
    Pre-train CNN backbone with cross-entropy.

    Returns feature_model (1024-d extractor), history
    """
    model_dir.mkdir(parents=True,   exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*55)
    print("Stage 2a: CNN pre-training")
    print("="*55)
    print(f"  Epochs={epochs}, Batch={CNN_BATCH_SIZE}, LR={CNN_INITIAL_LR}, Dropout={CNN_DROPOUT}")

    model = build_cnn_backbone()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CNN_LR_DECAY, patience=30, min_lr=1e-6
    )

    X_tr = _to_torch_images(X_train)
    y_tr = torch.from_numpy(y_train).long()
    X_va = _to_torch_images(X_val)
    y_va = torch.from_numpy(y_val).long()
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=CNN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=CNN_BATCH_SIZE, shuffle=False)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    best_state = None
    patience = 60
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_loss = train_loss / train_total if train_total else 0.0
        train_acc = train_correct / train_total if train_total else 0.0
        val_loss = val_loss / val_total if val_total else 0.0
        val_acc = val_correct / val_total if val_total else 0.0
        scheduler.step(val_loss)

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, str(model_dir / 'cnn_best.pth'))
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}/{epochs}  loss={train_loss:.4f}  acc={train_acc:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best val_loss={best_val_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), str(model_dir / 'cnn_backbone.pth'))

    for metric, title, fname in [
        ('loss',     'CNN Training Loss',     'cnn_loss.png'),
        ('accuracy', 'CNN Training Accuracy', 'cnn_accuracy.png'),
    ]:
        plt.figure(figsize=(8, 4))
        plt.plot(history[metric], label=f'Train {metric.capitalize()}')
        plt.plot(history[f'val_{metric}'], label=f'Val   {metric.capitalize()}')
        plt.xlabel('Epoch'); plt.ylabel(metric.capitalize())
        plt.title(title); plt.legend(); plt.tight_layout()
        plt.savefig(str(results_dir / fname), dpi=150)
        plt.close()
    print("Training curves saved -> results/cnn_loss.png, cnn_accuracy.png")
    print("CNN backbone saved -> models/cnn_backbone.pth")

    return model, history


def train_svm_on_features(feature_model, X_train, y_train, X_val, y_val,
                          model_dir: Path, results_dir: Path):
    """
    Extract CNN features, train SVM classifier, save model.

    Returns svm, scaler, scores_dict
    """
    print("\n" + "="*55)
    print("Stage 2b: SVM on CNN features")
    print("="*55)

    print("Extracting CNN features ...")
    feat_train = extract_cnn_features(feature_model, X_train, batch_size=64)
    feat_val   = extract_cnn_features(feature_model, X_val,   batch_size=64)
    print(f"  Feature dim: {feat_train.shape[1]}")

    scaler     = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_val   = scaler.transform(feat_val)

    print(f"Training SVM (kernel={SVM_KERNEL}, C={SVM_C}) ...")
    svm = SVC(kernel=SVM_KERNEL, C=SVM_C, probability=True, random_state=42)
    svm.fit(feat_train, y_train)

    y_pred = svm.predict(feat_val)
    acc    = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=['NoFire', 'Fire'], digits=4)
    print(f"\nValidation Accuracy: {acc:.4f}\n{report}")

    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['NoFire', 'Fire']).plot(ax=ax, colorbar=False, cmap='Oranges')
    ax.set_title('CNN + SVM - Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig(str(results_dir / 'cnn_svm_cm.png'), dpi=150)
    plt.close()

    joblib.dump(svm,    str(model_dir / 'svm_model.pkl'))
    joblib.dump(scaler, str(model_dir / 'cnn_svm_scaler.pkl'))
    print("SVM saved -> models/svm_model.pkl")

    return svm, scaler, {'accuracy': acc, 'report': report}


def load_cnn_svm(model_dir):
    """Load saved PyTorch CNN backbone and SVM."""
    model_dir = Path(model_dir)
    feature_model = build_cnn_backbone()
    feature_model.load_state_dict(torch.load(str(model_dir / 'cnn_backbone.pth'), map_location=DEVICE))
    feature_model.eval()
    svm    = joblib.load(str(model_dir / 'svm_model.pkl'))
    scaler = joblib.load(str(model_dir / 'cnn_svm_scaler.pkl'))
    return feature_model, svm, scaler


def predict_cnn_svm(images: np.ndarray, feature_model, svm, scaler) -> np.ndarray:
    """CNN+SVM inference -> (N,) int {0=nofire, 1=fire}."""
    feats = extract_cnn_features(feature_model, images, batch_size=32)
    return svm.predict(scaler.transform(feats))

# ═══════════════════════════════════════════════════════════════════════════════
#  5. TWO-STAGE INFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def two_stage_predict(images, clf_hog, scaler_hog, feature_model, svm, scaler_svm, verbose=False):
    """
    Two-stage pipeline (Fig. 5):
      Stage 1 (HOG+AdaBoost) — screens all images, high recall
      Stage 2 (CNN+SVM)      — re-evaluates Stage-1 positives, high precision
    An image is FIRE only if both stages agree.

    Returns final_preds (N,) int, stats dict
    """
    N = len(images)
    final_preds  = np.zeros(N, dtype=np.int32)
    stage1_preds = predict_hog(images, clf_hog, scaler_hog)
    positive_idx = np.where(stage1_preds == 1)[0]

    if verbose:
        print(f"Stage 1: {len(positive_idx)} positives, {N - len(positive_idx)} negatives")

    if len(positive_idx) > 0:
        final_preds[positive_idx] = predict_cnn_svm(
            images[positive_idx], feature_model, svm, scaler_svm
        )

    stats = {
        'stage1_positives': int(len(positive_idx)),
        'stage1_negatives': int(N - len(positive_idx)),
        'final_fire':       int(final_preds.sum()),
        'final_nofire':     int((final_preds == 0).sum()),
    }
    return final_preds, stats


def evaluate_pipeline(X_test, y_test, clf_hog, scaler_hog,
                      feature_model, svm, scaler_svm, results_dir: Path):
    """Evaluate two-stage pipeline on test set, save confusion matrix and report."""
    print("\n" + "="*55)
    print("Two-Stage Pipeline — Test Set Evaluation")
    print("="*55)

    y_pred, stats = two_stage_predict(
        X_test, clf_hog, scaler_hog, feature_model, svm, scaler_svm, verbose=True
    )
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['NoFire', 'Fire'], digits=4)
    print(f"\nFinal Test Accuracy: {acc:.4f}\n{report}")
    print(f"Stats: {stats}")

    results_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['NoFire', 'Fire']).plot(ax=ax, colorbar=False, cmap='Greens')
    ax.set_title('Two-Stage Pipeline — Test Confusion Matrix')
    plt.tight_layout()
    plt.savefig(str(results_dir / 'pipeline_confusion_matrix.png'), dpi=150)
    plt.close()

    with open(str(results_dir / 'classification_report.txt'), 'w') as f:
        f.write(f"Two-Stage Pipeline Test Results\n{'='*40}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n{report}\nStage stats: {stats}\n")
    print("Report saved → results/classification_report.txt")

    return acc, report


def predict_single_image(img_path: str, model_dir: Path) -> str:
    """Load models and predict a single image. Returns 'Fire' or 'No Fire'."""
    clf_hog, scaler_hog = load_hog_adaboost(model_dir)
    feature_model, svm, scaler_svm = load_cnn_svm(model_dir)

    img = PILImage.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)[np.newaxis] / 255.0   # (1,64,64,3)

    preds, stats = two_stage_predict(
        arr, clf_hog, scaler_hog, feature_model, svm, scaler_svm, verbose=True
    )
    label = 'Fire' if preds[0] == 1 else 'No Fire'
    print(f"\nImage     : {img_path}\nPrediction: {label}")
    return label


# ═══════════════════════════════════════════════════════════════════════════════
#  6. MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Forest Fire Recognition Pipeline (paper: S0030402620313279)'
    )
    parser.add_argument('--base_dir',   default='.', help='Project root directory')
    parser.add_argument('--mode',
                        choices=['all', 'gan', 'hog', 'cnn', 'eval'],
                        default='all', help='Pipeline step(s) to run')
    parser.add_argument('--use_gan',    action='store_true',
                        help='Include GAN-generated images when training HOG/CNN')
    parser.add_argument('--gan_epochs', type=int, default=GAN_EPOCHS,
                        help=f'GAN training epochs (default {GAN_EPOCHS})')
    parser.add_argument('--cnn_epochs', type=int, default=CNN_EPOCHS,
                        help=f'CNN training epochs (default {CNN_EPOCHS})')
    parser.add_argument('--image',      default=None,
                        help='Path to a single image to classify (skips training)')
    args = parser.parse_args()

    base        = Path(args.base_dir)
    model_dir   = base / 'models'
    results_dir = base / 'results'
    model_dir.mkdir(parents=True,   exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Single-image inference shortcut
    if args.image:
        predict_single_image(args.image, model_dir)
        return

    print("\n╔" + "═"*53 + "╗")
    print("║  Forest Fire Recognition — Full Pipeline           ║")
    print("╚" + "═"*53 + "╝")
    print(f"Base dir : {base.resolve()}")
    print(f"Mode     : {args.mode}  |  Use GAN: {args.use_gan}")
    t0_total = time.time()

    # ── 1. Load dataset ────────────────────────────────────────────────────────
    X, y = load_dataset(str(base), also_generated=(args.use_gan and args.mode != 'gan'))
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # ── 2. GAN augmentation ────────────────────────────────────────────────────
    if args.mode in ('gan', 'all'):
        t0 = time.time()
        train_gan(X[y == 1], 'fire',   base / 'generated/fire',
                  model_dir, results_dir, n_generate=500,  epochs=args.gan_epochs)
        train_gan(X[y == 0], 'nofire', base / 'generated/nofire',
                  model_dir, results_dir, n_generate=1500, epochs=args.gan_epochs)
        print(f"\n[GAN] Done in {(time.time()-t0)/60:.1f} min")

        if args.use_gan:
            print("\nReloading dataset with generated images …")
            X, y = load_dataset(str(base), also_generated=True)
            X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # ── 3. HOG + AdaBoost ──────────────────────────────────────────────────────
    if args.mode in ('hog', 'all'):
        t0 = time.time()
        train_hog_adaboost(X_train, y_train, X_val, y_val, model_dir, results_dir)
        print(f"\n[HOG+AdaBoost] Done in {(time.time()-t0)/60:.1f} min")

    # ── 4. CNN + SVM ───────────────────────────────────────────────────────────
    if args.mode in ('cnn', 'all'):
        t0 = time.time()
        feature_model, _ = train_cnn(X_train, y_train, X_val, y_val,
                                     model_dir, results_dir, epochs=args.cnn_epochs)
        train_svm_on_features(feature_model, X_train, y_train, X_val, y_val,
                              model_dir, results_dir)
        print(f"\n[CNN+SVM] Done in {(time.time()-t0)/60:.1f} min")

    # ── 5. Two-stage evaluation ────────────────────────────────────────────────
    if args.mode in ('eval', 'all'):
        t0 = time.time()
        clf_hog, scaler_hog = load_hog_adaboost(model_dir)
        feature_model, svm, scaler_svm = load_cnn_svm(model_dir)
        evaluate_pipeline(X_test, y_test, clf_hog, scaler_hog,
                          feature_model, svm, scaler_svm, results_dir)
        print(f"\n[Evaluation] Done in {(time.time()-t0)/60:.1f} min")

    print(f"\n{'='*55}")
    print(f"Total time : {(time.time()-t0_total)/60:.1f} min")
    print(f"Results    : {results_dir.resolve()}")
    print(f"Models     : {model_dir.resolve()}")
    print("Pipeline complete ✓")


def run(dataset_path, epochs=CNN_EPOCHS):
    """Standard pipeline interface.
    dataset_path: root dir with 'fire/' and 'nofire/' sub-folders (64x64 images).
    """
    from pathlib import Path
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )
    import numpy as np

    base        = Path(dataset_path)
    model_dir   = base / 'models'
    results_dir = base / 'results'
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        X, y = load_dataset(str(base))
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

        # Stage 1: HOG + AdaBoost
        clf_hog, scaler_hog, _ = train_hog_adaboost(
            X_train, y_train, X_val, y_val, model_dir, results_dir
        )
        # Stage 2: CNN backbone + SVM
        feature_model, _ = train_cnn(
            X_train, y_train, X_val, y_val, model_dir, results_dir, epochs=epochs
        )
        svm, scaler_svm, _ = train_svm_on_features(
            feature_model, X_train, y_train, X_val, y_val, model_dir, results_dir
        )

        # Two-stage prediction on test set
        y_pred, _ = two_stage_predict(
            X_test, clf_hog, scaler_hog, feature_model, svm, scaler_svm
        )

        # SVM probability scores for AUC / AUPR
        feat_test = extract_cnn_features(feature_model, X_test, batch_size=64)
        y_score   = svm.predict_proba(scaler_svm.transform(feat_test))[:, 1]
        has_both  = len(np.unique(y_test)) > 1

        return {
            "model_name": "HOG-AdaBoost+CNN-SVM",
            "metrics": {
                "accuracy":  float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
                "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
                "auc":       float(roc_auc_score(y_test, y_score))           if has_both else None,
                "aupr":      float(average_precision_score(y_test, y_score)) if has_both else None,
            },
        }
    except Exception as exc:
        return {"model_name": "HOG-AdaBoost+CNN-SVM", "error": str(exc), "metrics": None}


if __name__ == '__main__':
    main()
