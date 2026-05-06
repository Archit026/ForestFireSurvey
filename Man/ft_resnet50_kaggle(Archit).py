# # FT-ResNet50 — Forest Fire Detection (Kaggle)
# **Author:** Archit  
# ResNet-50 fine-tuned with Mish activations, Focal Loss, and Mixup augmentation.
# Dataset expected: `/kaggle/input/<dataset>/` with `train/`, `val/`, `test/` sub-folders (ImageFolder layout).

# ── 1. Install extras ────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'scikit-learn'], check=True)

# ── 2. Imports ───────────────────────────────────────────────────────────────
import os, random, time, copy, warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score
)

warnings.filterwarnings('ignore')
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

# ── 3. Configuration ─────────────────────────────────────────────────────────
# ⚠ Edit DATA_ROOT to match your Kaggle dataset path
DATA_ROOT   = '/kaggle/input/flame-dataset'   # must contain train/ val/ test/
SAVE_PATH   = '/kaggle/working/ft_resnet50_best.pth'

CFG = dict(
    img_size     = 254,
    batch_size   = 32,
    num_workers  = 2,
    epochs       = 40,
    lr           = 1e-3,
    beta1        = 0.9,
    beta2        = 0.999,
    eps          = 1e-8,
    focal_alpha  = 1.0,
    focal_gamma  = 2.0,
    mixup_alpha  = 0.5,
    seed         = 42,
    save_path    = SAVE_PATH,
)
print('Config:', CFG)

# ── 4. Reproducibility & Device ───────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(CFG['seed'])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# ── 5. Data Loaders ───────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SZ = CFG['img_size']

train_tf = transforms.Compose([
    transforms.Resize((SZ, SZ)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((SZ, SZ)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_ds = ImageFolder(os.path.join(DATA_ROOT, 'train'), transform=train_tf)
val_ds   = ImageFolder(os.path.join(DATA_ROOT, 'val'),   transform=eval_tf)
test_ds  = ImageFolder(os.path.join(DATA_ROOT, 'test'),  transform=eval_tf)

BS = CFG['batch_size']
NW = CFG['num_workers']
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,  num_workers=NW, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=NW, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BS, shuffle=False, num_workers=NW, pin_memory=True)

CLASS_NAMES = train_ds.classes
NUM_CLASSES = len(CLASS_NAMES)
print(f'Classes: {CLASS_NAMES}  |  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}')

# ── 6. Model Components ───────────────────────────────────────────────────────
class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class FocalLoss(nn.Module):
    """Focal Loss — down-weights easy examples to focus on hard ones."""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(inputs, targets, reduction='none')
        pt   = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


def _replace_relu_mish(module: nn.Module) -> nn.Module:
    """Recursively replace every ReLU with Mish in-place."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, Mish())
        else:
            _replace_relu_mish(child)
    return module


def build_ft_resnet50(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    FT-ResNet50 (Scheme 3):
      - conv1, bn1, layer1, layer2  → frozen
      - layer3, layer4              → ReLU → Mish, unfrozen
      - fc head                     → Linear(2048, num_classes)
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model   = models.resnet50(weights=weights)

    # Freeze early layers
    frozen = {'conv1', 'bn1', 'layer1', 'layer2'}
    for name, param in model.named_parameters():
        if any(name.startswith(f) for f in frozen):
            param.requires_grad = False

    # Swap ReLU → Mish in trainable stages
    _replace_relu_mish(model.layer3)
    _replace_relu_mish(model.layer4)

    # New classification head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


model = build_ft_resnet50(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Parameters — trainable: {trainable:,}  /  total: {total:,}')

# ── 7. Mixup Helpers ──────────────────────────────────────────────────────────
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.5):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0), device=x.device)
    mix_x = lam * x + (1 - lam) * x[idx]
    return mix_x, y, y[idx], lam


def mixup_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)

# ── 8. Train / Evaluate Loops ─────────────────────────────────────────────────
criterion = FocalLoss(alpha=CFG['focal_alpha'], gamma=CFG['focal_gamma'])
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG['lr'], betas=(CFG['beta1'], CFG['beta2']), eps=CFG['eps']
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)


def train_epoch(model, loader):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        imgs, ya, yb, lam = mixup_data(imgs, lbls, CFG['mixup_alpha'])
        optimizer.zero_grad()
        out  = model(imgs)
        loss = mixup_criterion(criterion, out, ya, yb, lam)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        preds     = out.argmax(1)
        correct  += (lam*(preds==ya).float() + (1-lam)*(preds==yb).float()).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == lbls).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, correct / total

# ── 9. Training Loop ──────────────────────────────────────────────────────────
best_val_acc  = 0.0
best_weights  = None
history       = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(1, CFG['epochs'] + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(model, train_loader)
    vl_loss, vl_acc = eval_epoch(model, val_loader)
    scheduler.step(vl_loss)

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(vl_acc)

    flag = ''
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(best_weights, CFG['save_path'])
        flag = '  ★ BEST'

    print(f"[{epoch:02d}/{CFG['epochs']}] "
          f"Train loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
          f"Val loss={vl_loss:.4f} acc={vl_acc*100:.2f}% | "
          f"LR={optimizer.param_groups[0]['lr']:.2e} "
          f"({time.time()-t0:.1f}s){flag}")

print(f'\nBest Validation Accuracy: {best_val_acc*100:.2f}%')
if best_weights:
    model.load_state_dict(best_weights)
    print(f'Best weights loaded from: {CFG["save_path"]}')

# ── 10. Test Evaluation ───────────────────────────────────────────────────────
@torch.no_grad()
def test_model(model, loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for imgs, lbls in loader:
        logits = model(imgs.to(DEVICE))
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.numpy())
    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


all_probs, all_preds, all_labels = test_model(model, test_loader)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix:\n', cm)

if cm.size == 4:
    TN, FP, FN, TP = cm.ravel()
else:
    TN = FP = FN = TP = 0

total  = TP + TN + FP + FN
acc    = (TP + TN) / total          if total  > 0 else 0.0
prec   = TP / (TP + FP)             if (TP+FP) > 0 else 0.0
rec    = TP / (TP + FN)             if (TP+FN) > 0 else 0.0
spec   = TN / (TN + FP)             if (TN+FP) > 0 else 0.0
f1     = 2*prec*rec / (prec+rec)    if (prec+rec) > 0 else 0.0

# Determine fire class index (ImageFolder sorts alphabetically)
fire_idx   = CLASS_NAMES.index('fire') if 'fire' in CLASS_NAMES else 1
fire_probs = all_probs[:, fire_idx]
y_bin      = (all_labels == fire_idx).astype(int)
has_both   = len(np.unique(y_bin)) > 1

auc  = float(roc_auc_score(y_bin, fire_probs))            if has_both else None
aupr = float(average_precision_score(y_bin, fire_probs))  if has_both else None

print('\n========== FT-ResNet50 Test Results ==========')
print(f'  Accuracy   : {acc*100:.2f}%')
print(f'  Precision  : {prec*100:.2f}%')
print(f'  Recall     : {rec*100:.2f}%')
print(f'  Specificity: {spec*100:.2f}%')
print(f'  F1 Score   : {f1*100:.2f}%')
print(f'  AUC-ROC    : {auc:.4f}'  if auc  is not None else '  AUC-ROC    : N/A')
print(f'  AUPR       : {aupr:.4f}' if aupr is not None else '  AUPR       : N/A')
print('==============================================')

# ── 11. Training Curves ───────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, CFG['epochs'] + 1)

ax1.plot(epochs_range, history['train_loss'], label='Train Loss')
ax1.plot(epochs_range, history['val_loss'],   label='Val Loss')
ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()

ax2.plot(epochs_range, [a*100 for a in history['train_acc']], label='Train Acc')
ax2.plot(epochs_range, [a*100 for a in history['val_acc']],   label='Val Acc')
ax2.set_title('Accuracy (%)'); ax2.set_xlabel('Epoch'); ax2.legend()

plt.tight_layout()
plt.savefig('/kaggle/working/ft_resnet50_training_curves.png', dpi=150)
plt.show()
print('Curves saved.')

# ── 12. Save Final Metrics to JSON ────────────────────────────────────────────
import json

results = {
    'model_name': 'FT-ResNet50',
    'metrics': {
        'accuracy' : round(acc,  4),
        'precision': round(prec, 4),
        'recall'   : round(rec,  4),
        'f1'       : round(f1,   4),
        'auc'      : round(auc,  4) if auc  is not None else None,
        'aupr'     : round(aupr, 4) if aupr is not None else None,
    }
}

out_path = '/kaggle/working/ft_resnet50_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved → {out_path}')
print(json.dumps(results, indent=2))
