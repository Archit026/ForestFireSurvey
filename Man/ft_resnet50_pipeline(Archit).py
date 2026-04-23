import os
import random
import time
import copy
import warnings
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Configuration and Setup
# ---------------------------------------------------------
def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Return the available device (CUDA or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------
# Data Pipeline & Augmentation
# ---------------------------------------------------------
def get_dataloaders(data_root, img_size=254, batch_size=32, num_workers=4):
    """Create and return train, val, and test dataloaders."""
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_dir = os.path.join(data_root, 'test')
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=eval_transform)
    test_dataset = ImageFolder(root=test_dir, transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_dataset.classes

def mixup_data(x, y, alpha=0.5):
    """Returns mixed inputs, pairs of targets, and mixing coefficient."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup-aware loss: weighted combination of two label losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ---------------------------------------------------------
# Model Components
# ---------------------------------------------------------
class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))."""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard samples."""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()

def replace_relu_with_mish(module):
    """Recursively replace all ReLU activations with Mish."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, Mish())
        else:
            replace_relu_with_mish(child)
    return module

def build_ft_resnet50(num_classes=2, pretrained=True):
    """
    Builds the FT-ResNet50 model:
      - Stages 1 & 2 frozen
      - Stages 3, 4, 5: ReLU replaced with Mish + unfrozen
      - FC head replaced: 2048 -> num_classes
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze conv1, bn1, layer1, layer2
    frozen_layers = ['conv1', 'bn1', 'layer1', 'layer2']
    for name, param in model.named_parameters():
        if any(name.startswith(fl) for fl in frozen_layers):
            param.requires_grad = False

    # Replace ReLU with Mish in Stages 3, 4, 5
    replace_relu_with_mish(model.layer3)
    replace_relu_with_mish(model.layer4)
    if hasattr(model, 'layer5'):
        replace_relu_with_mish(model.layer5)

    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ---------------------------------------------------------
# Training and Evaluation Logic
# ---------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()
        total += images.size(0)
        
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        
    return running_loss / total, correct / total

def train_model(model, train_loader, val_loader, device, config):
    criterion = FocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'], betas=(config['beta1'], config['beta2']), eps=config['eps']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    best_weights = None
    save_path = config['save_path']
    epochs = config['epochs']
    mixup_alpha = config['mixup_alpha']
    
    print(f"Starting training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, mixup_alpha)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            flag = ' * BEST *'
        else:
            flag = ''
            
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
              f"LR: {lr_now:.2e} ({elapsed:.1f}s){flag}")
              
    print(f"Training complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    if best_weights:
        model.load_state_dict(best_weights)
    return model

@torch.no_grad()
def test_model(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in test_loader:
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    cm = confusion_matrix(all_labels, all_preds)
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    else:
        TN, FP, FN, TP = 0, 0, 0, 0
        
    acc = (TP + TN) / (TP + FN + FP + TN) if (TP + FN + FP + TN) > 0 else 0
    pre = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    spe = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    
    print("\n=== FT-ResNet50 Test Results ===")
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Precision:   {pre*100:.2f}%")
    print(f"Recall:      {rec*100:.2f}%")
    print(f"Specificity: {spe*100:.2f}%")
    print(f"F1 Score:    {f1*100:.2f}%")
    
    return acc, pre, rec, spe, f1, cm

def main():
    parser = argparse.ArgumentParser(description="FT-ResNet50 Pipeline for Forest Fire Detection")
    parser.add_argument('--data_root', type=str, default='/kaggle/input/flame-dataset', help='Path to dataset root containing train/val/test folders')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='./ft_resnet50_best.pth', help='Path to save best model weights')
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Check data path
    if not os.path.exists(args.data_root):
        print(f"Warning: Data root {args.data_root} does not exist. Update the path or script may fail on data load.")

    try:
        # Data
        print("Loading data...")
        train_loader, val_loader, test_loader, class_names = get_dataloaders(
            args.data_root, batch_size=args.batch_size
        )

        # Model
        print("Building FT-ResNet50...")
        model = build_ft_resnet50(num_classes=len(class_names), pretrained=True)
        model = model.to(device)

        config = {
            'epochs': args.epochs,
            'lr': args.lr,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'mixup_alpha': 0.5,
            'save_path': args.save_path
        }

        # Train
        model = train_model(model, train_loader, val_loader, device, config)

        # Evaluate
        print("Evaluating on test set...")
        test_model(model, test_loader, device, class_names)
        
    except Exception as e:
        print(f"Pipeline encountered an error: {e}")

def run(dataset_path):
    """Standard pipeline interface.
    dataset_path: root dir with train/ val/ test/ sub-folders (ImageFolder structure).
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    import numpy as np

    if not os.path.exists(dataset_path):
        return {"model_name": "FT-ResNet50", "error": f"Dataset not found: {dataset_path}", "metrics": None}

    try:
        set_seed(42)
        device = get_device()

        train_loader, val_loader, test_loader, class_names = get_dataloaders(
            dataset_path, batch_size=32
        )
        model = build_ft_resnet50(num_classes=len(class_names), pretrained=True).to(device)

        config = {
            'epochs': 40, 'lr': 0.001,
            'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8,
            'focal_alpha': 1.0, 'focal_gamma': 2.0, 'mixup_alpha': 0.5,
            'save_path': os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'ft_resnet50_best.pth'),
        }
        model = train_model(model, train_loader, val_loader, device, config)

        # Collect logits for AUC/AUPR
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                logits = model(images.to(device))
                probs  = torch.softmax(logits, dim=1)[:, 1]  # fire class probability
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)

        # fire class index: ImageFolder sorts alphabetically; determine fire index
        fire_idx = class_names.index('fire') if 'fire' in class_names else 1
        # re-orient probabilities so they represent P(fire)
        fire_probs = all_probs if fire_idx == 1 else (1.0 - all_probs)
        y_true_bin = (all_labels == fire_idx).astype(int)

        acc, pre, rec, _, f1, _ = test_model(model, test_loader, device, class_names)
        has_both = len(np.unique(y_true_bin)) > 1

        return {
            "model_name": "FT-ResNet50",
            "metrics": {
                "accuracy":  float(acc),
                "precision": float(pre),
                "recall":    float(rec),
                "f1":        float(f1),
                "auc":       float(roc_auc_score(y_true_bin, fire_probs))           if has_both else None,
                "aupr":      float(average_precision_score(y_true_bin, fire_probs)) if has_both else None,
            },
        }
    except Exception as exc:
        return {"model_name": "FT-ResNet50", "error": str(exc), "metrics": None}


if __name__ == '__main__':
    main()
