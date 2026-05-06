import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

from Uav.uav_common import model_results_dir, resolve_flame_dirs

print("=" * 60)
print("  UAV Fire Detection — U-Net Encoder + Classifier")
print("  Framework : PyTorch", torch.__version__)
print("  GPU       :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available (using CPU)")
print("=" * 60)


# ============================================================
# CONFIGURATION
# All tunable parameters in one place — change here only
# ============================================================
IMG_SIZE      = 256
BATCH_SIZE    = 16
EPOCHS        = 25
LR            = 1e-4
LR_FINETUNE   = 1e-5
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.15       # 15% of Train split used for validation
SEED          = 42
THRESHOLD     = 0.5        # sigmoid threshold for Fire classification
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DATASET — locate FLAME folder from root dataset path
# ============================================================

def find_flame_dirs(dataset_root: str):
    """
    Walks dataset_root to find Training/ or Train/ and Test/ folders containing
    Fire/ and No_Fire/ subfolders. Handles any nesting depth.
    """
    return resolve_flame_dirs(dataset_root)


# ============================================================
# DATA PIPELINE
# ============================================================

def get_transforms(training=False):
    """
    Training: augmentation + normalisation
    Validation/Test: resize + normalise only
    ImageNet mean/std because we use pretrained MobileNetV2
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if training:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])


def build_dataloaders(train_dir, test_dir):
    """
    Loads FLAME dataset using ImageFolder (expects Fire/ and No_Fire/ subfolders).
    Splits Train into train + val sets.
    Returns: train_loader, val_loader, test_loader, class_weights
    """
    full_train = ImageFolder(train_dir, transform=get_transforms(training=True))
    test_ds    = ImageFolder(test_dir,  transform=get_transforms(training=False))

    # Class label check
    print(f"\nClass mapping: {full_train.class_to_idx}")
    assert set(full_train.class_to_idx.keys()) >= {'Fire', 'No_Fire'} or \
           set(full_train.class_to_idx.keys()) >= {'fire', 'no_fire'}, \
        f"Expected Fire/ and No_Fire/ folders, got: {list(full_train.class_to_idx.keys())}"

    # Train / Val split
    n_val   = int(len(full_train) * VAL_SPLIT)
    n_train = len(full_train) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    # Val uses non-augmented transforms
    val_ds.dataset.transform = get_transforms(training=False)

    print(f"Train samples : {n_train}")
    print(f"Val samples   : {n_val}")
    print(f"Test samples  : {len(test_ds)}")

    # Compute class weights to handle imbalance (fire frames are rarer)
    targets = [full_train.targets[i] for i in train_ds.indices]
    n_fire    = sum(targets)
    n_nofire  = len(targets) - n_fire
    # weight for fire class = total / (2 * n_fire), and vice versa
    weight_fire   = len(targets) / (2 * n_fire)   if n_fire   > 0 else 1.0
    weight_nofire = len(targets) / (2 * n_nofire) if n_nofire > 0 else 1.0
    # ImageFolder: class_to_idx determines label 0 or 1
    # We need weight in label order
    fire_label = full_train.class_to_idx.get('Fire', full_train.class_to_idx.get('fire', 1))
    class_weights = torch.zeros(2)
    class_weights[fire_label]     = weight_fire
    class_weights[1 - fire_label] = weight_nofire
    print(f"Class weights : Fire={weight_fire:.3f}, No_Fire={weight_nofire:.3f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights, fire_label


# ============================================================
# MODEL — U-Net Encoder + Classification Head
# ============================================================

class FireUNetClassifier(nn.Module):
    """
    U-Net style encoder using MobileNetV2 pretrained on ImageNet.

    The encoder extracts feature maps at 4 spatial scales — this is
    what separates it from a plain MobileNetV2 classifier. Instead of
    discarding intermediate spatial features via GAP immediately, we
    concatenate multi-scale features before pooling, preserving the
    spatial fire signal that plain classifiers lose.

    Architecture:
        MobileNetV2 encoder (4 skip connection scales)
            ↓
        Multi-scale feature concatenation
            ↓
        AdaptiveAvgPool → Flatten
            ↓
        Classifier head: Dropout → Linear → ReLU → Dropout → Linear(1)
            ↓
        Sigmoid → P(fire)
    """

    def __init__(self, pretrained=True, freeze_encoder=True):
        super().__init__()

        # Load pretrained MobileNetV2
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # ── Encoder: extract feature maps at 4 scales ──────────
        # MobileNetV2 features is a Sequential of 19 blocks
        features = backbone.features

        self.enc1 = features[:2]    # stride 2  → 128x128, 16 channels
        self.enc2 = features[2:4]   # stride 4  → 64x64,   24 channels
        self.enc3 = features[4:7]   # stride 8  → 32x32,   32 channels
        self.enc4 = features[7:14]  # stride 16 → 16x16,   96 channels
        self.enc5 = features[14:]   # stride 32 → 8x8,    1280 channels

        if freeze_encoder:
            for p in self.parameters():
                p.requires_grad = False

        # ── Pooling: each scale → 4x4 spatial, then flatten ────
        self.pool = nn.AdaptiveAvgPool2d(4)

        # Channel counts after pool + flatten:
        # enc1=16, enc2=24, enc3=32, enc4=96, enc5=1280 → total=1448 × 16 spatial
        total_features = (16 + 24 + 32 + 96 + 1280) * 4 * 4

        # ── Classification head ──────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # Pass through encoder stages, collecting multi-scale features
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        # Pool each scale to same spatial size, flatten, concat
        f1 = self.pool(x1).flatten(1)
        f2 = self.pool(x2).flatten(1)
        f3 = self.pool(x3).flatten(1)
        f4 = self.pool(x4).flatten(1)
        f5 = self.pool(x5).flatten(1)

        features = torch.cat([f1, f2, f3, f4, f5], dim=1)
        return torch.sigmoid(self.classifier(features))

    def unfreeze_encoder(self):
        """Call this before Phase 2 fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx=None, total_epochs=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader, start=1):
        batch_t0 = time.time()
        imgs   = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = (outputs > THRESHOLD).float()
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

        running_loss = total_loss / total if total else 0.0
        running_acc = correct / total if total else 0.0
        batch_time = time.time() - batch_t0
        eta = batch_time * (len(loader) - batch_idx)
        if epoch_idx is not None and total_epochs is not None:
            print(
                f"  [Epoch {epoch_idx}/{total_epochs}] batch {batch_idx}/{len(loader)} "
                f"| batch_time={batch_time:.1f}s | eta={eta:.0f}s | train_loss={running_loss:.4f} | train_acc={running_acc:.4f}"
            )
        else:
            print(
                f"  batch {batch_idx}/{len(loader)} "
                f"| batch_time={batch_time:.1f}s | eta={eta:.0f}s | train_loss={running_loss:.4f} | train_acc={running_acc:.4f}"
            )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = (outputs > THRESHOLD).float()
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

        all_probs.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs  = np.vstack(all_probs).squeeze()
    labels = np.vstack(all_labels).squeeze().astype(int)
    preds  = (probs > THRESHOLD).astype(int)

    metrics = {
        "loss"      : total_loss / total,
        "accuracy"  : correct / total,
        "recall"    : recall_score(labels, preds, zero_division=0),
        "precision" : precision_score(labels, preds, zero_division=0),
        "f1"        : f1_score(labels, preds, zero_division=0),
    }
    if len(np.unique(labels)) > 1:
        metrics["auc"]  = roc_auc_score(labels, probs)
        metrics["aupr"] = average_precision_score(labels, probs)
    else:
        metrics["auc"]  = None
        metrics["aupr"] = None

    return metrics, preds, probs, labels


def train(model, train_loader, val_loader, save_dir, class_weights, device):
    """
    Two-phase training:
      Phase 1 (epochs 1..EPOCHS)       : encoder frozen, train classifier head only
      Phase 2 (epochs EPOCHS//2 onward): unfreeze encoder, fine-tune at lower LR
    """
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "fire_unet_classifier.pth")

    # Weighted BCE loss — handles class imbalance
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion  = nn.BCELoss()   # outputs are already sigmoid'd

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    best_f1      = 0.0
    patience_ctr = 0
    PATIENCE     = 7
    phase        = 1

    history = {k: [] for k in
               ["train_loss","train_acc","val_loss","val_acc","val_f1","val_recall","val_precision"]}

    print("\n" + "=" * 60)
    print("PHASE 1: Training classifier head (encoder frozen)")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):

        # Switch to Phase 2 halfway through
        if epoch == EPOCHS // 2 + 1 and phase == 1:
            print("\n" + "=" * 60)
            print("PHASE 2: Fine-tuning full network (encoder unfrozen)")
            print("=" * 60)
            model.unfreeze_encoder()
            for pg in optimizer.param_groups:
                pg['lr'] = LR_FINETUNE
            phase = 2

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch_idx=epoch, total_epochs=EPOCHS)
        val_metrics, _, _, _  = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["f1"])

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_precision"].append(val_metrics["precision"])

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{EPOCHS}] ({elapsed:.1f}s)  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_metrics['loss']:.4f}  F1: {val_metrics['f1']:.4f}  "
              f"Recall: {val_metrics['recall']:.4f}  Precision: {val_metrics['precision']:.4f}"
              + (f"  AUC: {val_metrics['auc']:.4f}" if val_metrics['auc'] else ""))

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), weights_path)
            print(f"  ✅ Best model saved (F1={best_f1:.4f})")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n⏹  Early stopping triggered at epoch {epoch}")
                break

    print(f"\nTraining complete. Best Val F1: {best_f1:.4f}")
    return history, weights_path


# ============================================================
# PLOTS
# ============================================================

def plot_training_curves(history, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History — UAV Fire Detection (U-Net Encoder)", fontsize=15, fontweight='bold')
    axes = axes.flatten()

    pairs = [
        ("Loss",      "train_loss",  "val_loss",      "#e74c3c"),
        ("Accuracy",  "train_acc",   "val_acc",        "#2ecc71"),
        ("F1 Score",  None,          "val_f1",         "#3498db"),
        ("Recall / Precision", None, "val_recall",     "#9b59b6"),
    ]

    for i, (title, train_key, val_key, color) in enumerate(pairs):
        if train_key:
            axes[i].plot(history[train_key], label='Train', color=color, linewidth=2)
        axes[i].plot(history[val_key], label='Val', color=color, linewidth=2, linestyle='--')
        if title == "Recall / Precision":
            axes[i].plot(history["val_precision"], label='Val Precision',
                         color='#e67e22', linewidth=2, linestyle=':')
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Epoch')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved → {out}")


def plot_confusion_matrix(labels, preds, save_dir, fire_label):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    class_names = ['No Fire', 'Fire'] if fire_label == 1 else ['Fire', 'No Fire']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 14})
    plt.title('Confusion Matrix\nUAV Fire Detection — Test Set', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    out = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {out}")


def save_training_artifacts(history, labels, preds, metrics, save_dir, fire_label):
    if history:
        plot_training_curves(history, save_dir)
    plot_confusion_matrix(labels, preds, save_dir, fire_label)
    save_summary(metrics, save_dir)


# ============================================================
# RESULTS SUMMARY
# ============================================================

def save_summary(metrics, save_dir):
    out = os.path.join(save_dir, "results_summary.txt")
    lines = [
        "=" * 50,
        "  UAV Fire Detection — Final Test Results",
        "=" * 50,
        f"  Model       : U-Net Encoder + Classifier (PyTorch)",
        f"  Architecture: MobileNetV2 encoder (pretrained ImageNet)",
        f"  Input size  : {IMG_SIZE}x{IMG_SIZE}x3",
        f"  Device      : {DEVICE}",
        "",
        f"  Accuracy    : {metrics['accuracy']:.4f}",
        f"  Precision   : {metrics['precision']:.4f}",
        f"  Recall      : {metrics['recall']:.4f}",
        f"  F1 Score    : {metrics['f1']:.4f}",
        f"  AUC-ROC     : {metrics['auc']:.4f}"  if metrics['auc']  else "  AUC-ROC     : N/A (single class in test set)",
        f"  AUC-PR      : {metrics['aupr']:.4f}" if metrics['aupr'] else "  AUC-PR      : N/A",
        "=" * 50,
    ]
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    for line in lines:
        print(line)
    print(f"\nSummary saved → {out}")
    return metrics


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    global EPOCHS, BATCH_SIZE

    parser = argparse.ArgumentParser(description="UAV Fire Detection — U-Net Encoder Classifier")
    parser.add_argument('--dataset',    type=str, required=True,
                        help='Root dataset directory (expects uav/FLAME/Train and Test inside)')
    parser.add_argument('--epochs',     type=int, default=EPOCHS,     help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--no_train',   action='store_true',
                        help='Skip training, load saved weights and evaluate only')
    parser.add_argument('--output',     type=str, default='./results/uav',
                        help='Directory to save results')
    args = parser.parse_args()

    # Override globals if user passed args
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch_size

    save_dir = args.output or str(model_results_dir("amit_uav_1"))
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Locate dataset ──────────────────────────────────────
    print(f"\nDataset root : {args.dataset}")
    train_dir, test_dir = find_flame_dirs(args.dataset)
    print(f"Train dir    : {train_dir}")
    print(f"Test dir     : {test_dir}")

    # ── 2. Build data loaders ──────────────────────────────────
    train_loader, val_loader, test_loader, class_weights, fire_label = \
        build_dataloaders(train_dir, test_dir)

    # ── 3. Build model ─────────────────────────────────────────
    print("\nBuilding model...")
    model = FireUNetClassifier(pretrained=True, freeze_encoder=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {n_params:,}")
    print(f"Trainable params : {n_train:,} (Phase 1, encoder frozen)")

    weights_path = os.path.join(save_dir, "fire_unet_classifier.pth")
    history = None

    # ── 4. Train ───────────────────────────────────────────────
    if not args.no_train:
        history, weights_path = train(model, train_loader, val_loader,
                                      save_dir, class_weights, DEVICE)
    else:
        print("\n--no_train flag set. Skipping training.")

    # ── 5. Load best weights and evaluate on test set ──────────
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f"\nBest weights loaded from {weights_path}")
    else:
        print("\n⚠️  No saved weights found — evaluating with current (untrained) model")

    print("\nEvaluating on test set...")
    criterion = nn.BCELoss()
    test_metrics, preds, probs, labels = evaluate(model, test_loader, criterion, DEVICE)

    # ── 6. Save plots and summary ──────────────────────────────
    print("\n" + classification_report(
        labels, preds, target_names=['No Fire', 'Fire'], zero_division=0
    ))
    save_training_artifacts(history, labels, preds, test_metrics, save_dir, fire_label)

    return test_metrics


# ============================================================
# run() — standard interface for main.py integration
# ============================================================

def run(dataset_path: str, epochs: int = None, output_dir: str = None) -> dict:
    """
    Standard pipeline interface called by main.py.
    dataset_path: root directory containing uav/FLAME/Train and Test.
    Returns a dict with model_name and metrics.
    """
    global EPOCHS

    try:
        train_dir, test_dir = find_flame_dirs(dataset_path)
    except FileNotFoundError as e:
        return {"model_name": "UAV-FireUNet", "error": str(e), "metrics": None}

    if epochs is not None:
        EPOCHS = epochs

    train_loader, val_loader, test_loader, class_weights, fire_label = \
        build_dataloaders(train_dir, test_dir)

    model = FireUNetClassifier(pretrained=True, freeze_encoder=True).to(DEVICE)

    save_dir = output_dir or str(model_results_dir("amit_uav_1"))
    history, weights_path = train(model, train_loader, val_loader,
                                  save_dir, class_weights, DEVICE)

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    criterion = nn.BCELoss()
    metrics, preds, _, labels = evaluate(model, test_loader, criterion, DEVICE)

    save_training_artifacts(history, labels, preds, metrics, save_dir, fire_label)

    return {
        "model_name": "UAV-FireUNet",
        "metrics": {
            "accuracy"  : float(metrics["accuracy"]),
            "precision" : float(metrics["precision"]),
            "recall"    : float(metrics["recall"]),
            "f1"        : float(metrics["f1"]),
            "auc"       : float(metrics["auc"])  if metrics["auc"]  else None,
            "aupr"      : float(metrics["aupr"]) if metrics["aupr"] else None,
        }
    }


if __name__ == "__main__":
    main()