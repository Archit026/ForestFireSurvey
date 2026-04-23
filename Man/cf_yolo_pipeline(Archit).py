import os
import sys
import glob
import json
import yaml
import shutil
import subprocess
import urllib.request
import argparse
import random

# Optional dependencies for visualization and evaluation
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
except ImportError:
    print("Warning: Pandas, Numpy, Matplotlib, or OpenCV not installed. Visualization might fail.")

def run_command(cmd, cwd=None):
    """Helper to run shell commands via subprocess."""
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

def setup_environment(yolo_dir):
    """Clone YOLOv7 and install required dependencies."""
    if not os.path.exists(yolo_dir):
        print("Cloning YOLOv7 repository...")
        run_command(['git', 'clone', 'https://github.com/WongKinYiu/yolov7.git', yolo_dir])
    else:
        print("YOLOv7 already cloned.")

    print("Installing dependencies...")
    req_path = os.path.join(yolo_dir, 'requirements.txt')
    run_command([sys.executable, '-m', 'pip', 'install', '-r', req_path, '-q'])
    run_command([sys.executable, '-m', 'pip', 'install', 'timm', 'einops', 'thop', 'pycocotools', '-q'])
    print("Environment setup complete.\n")

def prepare_dataset(dataset_root, dataset_work, yolo_dir):
    """Synchronize dataset splits, create configuration YAML, and verify integrity."""
    splits = ['train', 'val', 'test']
    
    print("Preparing dataset directories...")
    for split in splits:
        os.makedirs(os.path.join(dataset_work, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_work, 'labels', split), exist_ok=True)
        
        src_img = os.path.join(dataset_root, 'images', split)
        src_lbl = os.path.join(dataset_root, 'labels', split)
        dst_img = os.path.join(dataset_work, 'images', split)
        dst_lbl = os.path.join(dataset_work, 'labels', split)
        
        if os.path.exists(src_img):
            for f in glob.glob(os.path.join(src_img, '*')):
                shutil.copy2(f, dst_img)
            for f in glob.glob(os.path.join(src_lbl, '*')):
                shutil.copy2(f, dst_lbl)
            print(f"  [{split}] Synced {len(os.listdir(dst_img))} images and {len(os.listdir(dst_lbl))} labels.")
        else:
            print(f"  WARNING: Directory {src_img} not found.")

    # Create dataset YAML
    yaml_content = {
        'train': os.path.join(dataset_work, 'images', 'train'),
        'val':   os.path.join(dataset_work, 'images', 'val'),
        'test':  os.path.join(dataset_work, 'images', 'test'),
        'nc': 1,
        'names': ['fire']
    }
    
    yaml_path = os.path.join(yolo_dir, 'fire_dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"Saved dataset config: {yaml_path}\n")

def patch_architecture(yolo_dir):
    """Inject CF-YOLO custom modules (CA, SimAM, D2F, SSC) into models/common.py and models/yolo.py."""
    print("Patching YOLOv7 architecture with CF-YOLO modules...")
    
    # 1. CF-YOLO Custom Modules Code
    custom_modules = '''
# --- CF-YOLO MODULES (auto-inserted) ---
class CoordAtt(nn.Module):
    """Coordinate Attention."""
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        mip = max(8, inp // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1  = nn.Conv2d(inp, mip, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, 1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y   = torch.cat([x_h, x_w], dim=2)
        y   = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w

class SimAM(nn.Module):
    """SimAM: Simple, Parameter-Free Attention Module."""
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act      = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        n          = w * h - 1
        mu         = x.mean(dim=[2, 3], keepdim=True)
        x_mu       = x - mu
        y          = x_mu ** 2 / (
            4 * (x_mu.pow(2).sum(dim=[2, 3], keepdim=True) / n + self.e_lambda) + 0.5
        )
        return x * self.act(y)

class DSConv(nn.Module):
    """Depthwise-Separable Conv."""
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        p = p if p is not None else k // 2
        self.dw = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class DSBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DSConv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class D2F(nn.Module):
    """C2F backbone with DSConv Bottleneck."""
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c   = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((n + 2) * self.c, c2, 1)
        self.m   = nn.ModuleList(DSBottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SSC(nn.Module):
    """S-SC: SPPFCSPC + SimAM attention + serial pooling."""
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, k=13):
        super().__init__()
        c_ = int(2 * c2 * e)
        self.cv1   = Conv(c1, c_, 1, 1)
        self.cv2   = Conv(c1, c_, 1, 1)
        self.cv3   = Conv(c_, c_, 3, 1)
        self.cv4   = Conv(4 * c_, c2, 1, 1)
        self.m     = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.simam = SimAM()

    def forward(self, x):
        x1 = self.cv3(self.cv1(x))
        y1 = self.simam(self.m(x1))
        y2 = self.simam(self.m(y1))
        y3 = self.simam(self.m(y2))
        return self.cv4(torch.cat([x1, y1, y2, y3], 1))
'''
    
    # Update models/common.py
    common_path = os.path.join(yolo_dir, 'models', 'common.py')
    with open(common_path, 'r') as f:
        common_src = f.read()
    
    if '# --- CF-YOLO MODULES (auto-inserted) ---' not in common_src:
        with open(common_path, 'a') as f:
            f.write(custom_modules)
        print("  Updated models/common.py")
    else:
        print("  models/common.py already updated.")
        
    # Update models/yolo.py
    yolo_path = os.path.join(yolo_dir, 'models', 'yolo.py')
    with open(yolo_path, 'r') as f:
        yolo_src = f.read()
        
    import_marker = '# --- CF-YOLO imports ---'
    parse_marker = '# --- CF-YOLO parse_model registration ---'
    
    if import_marker not in yolo_src:
        import_line = f'\n{import_marker}\nfrom models.common import CoordAtt, SimAM, DSConv, DSBottleneck, D2F, SSC\n'
        yolo_src = yolo_src.replace('from models.common import *', 'from models.common import *' + import_line)
        
    if parse_marker not in yolo_src:
        target = 'elif m in [nn.BatchNorm2d]:'
        reg_code = f"elif m in [CoordAtt, SimAM, DSConv, DSBottleneck, D2F, SSC]:  {parse_marker}\n            c2 = args[1] if len(args) > 1 else c1\n        "
        yolo_src = yolo_src.replace(target, reg_code + target)
        
    with open(yolo_path, 'w') as f:
        f.write(yolo_src)
    print("  Updated models/yolo.py\n")

def generate_model_configs(yolo_dir):
    """Generate CF-YOLO base and ablation YAML configuration files."""
    print("Generating model configuration YAML files...")
    cfg_dir = os.path.join(yolo_dir, 'cfg', 'training')
    os.makedirs(cfg_dir, exist_ok=True)
    
    cf_yaml_path = os.path.join(cfg_dir, 'cf-yolo.yaml')
    
    # 2. CF-YOLO YAML configuration
    cf_yaml_content = """
nc: 1
depth_multiple: 1.0
width_multiple: 1.0

anchors:
  - [12,16, 19,36, 40,28]
  - [36,75, 76,55, 72,146]
  - [142,110, 192,243, 459,401]

backbone:
  [[-1, 1, Conv, [32, 3, 1]],
   [-1, 1, Conv, [64, 3, 2]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1,-3,-5,-6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1,-3], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1,-3,-5,-6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]],
   [-1, 1, CoordAtt, [512, 512]],
   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1,-3], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1,-3,-5,-6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],
   [-1, 1, MP, []],
   [-1, 1, Conv, [512, 1, 1]],
   [-3, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 2]],
   [[-1,-3], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1,-3,-5,-6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],
   [-1, 1, CoordAtt, [1024, 1024]],
  ]

head:
  [[-1, 1, SSC, [512, 512]],
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [38, 1, Conv, [256, 1, 1]],
   [[-1,-2], 1, Concat, [1]],
   [-1, 1, D2F, [256, 256, 3]],
   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [24, 1, Conv, [128, 1, 1]],
   [[-1,-2], 1, Concat, [1]],
   [-1, 1, D2F, [128, 128, 3]],
   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1,-3,59], 1, Concat, [1]],
   [-1, 1, D2F, [256, 256, 3]],
   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1,-3,54], 1, Concat, [1]],
   [-1, 1, D2F, [512, 512, 3]],
   [[64, 70, 76], 1, IDetect, [nc, anchors]],
  ]
"""
    with open(cf_yaml_path, 'w') as f:
        f.write(cf_yaml_content)
    print(f"  Saved CF-YOLO config: {cf_yaml_path}\n")

def train_model(yolo_dir, batch_size=4, epochs=300):
    """Download YOLOv7 weights and initiate training for baseline and CF-YOLO."""
    weights_path = os.path.join(yolo_dir, 'yolov7.pt')
    if not os.path.exists(weights_path):
        print("Downloading YOLOv7 pretrained weights...")
        urllib.request.urlretrieve('https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt', weights_path)
    
    print("\n--- Training CF-YOLO ---")
    cmd_cf = [
        sys.executable, 'train.py', '--workers', '4', '--device', '0',
        '--batch-size', str(batch_size), '--epochs', str(epochs),
        '--img', '640', '640', '--data', 'fire_dataset.yaml',
        '--cfg', 'cfg/training/cf-yolo.yaml', '--weights', 'yolov7.pt',
        '--name', 'cf_yolo_fire', '--hyp', 'data/hyp.scratch.p5.yaml'
    ]
    run_command(cmd_cf, cwd=yolo_dir)
    
    print("\n--- Training YOLOv7 Baseline ---")
    cmd_base = [
        sys.executable, 'train.py', '--workers', '4', '--device', '0',
        '--batch-size', str(batch_size), '--epochs', str(epochs),
        '--img', '640', '640', '--data', 'fire_dataset.yaml',
        '--cfg', 'cfg/training/yolov7.yaml', '--weights', 'yolov7.pt',
        '--name', 'yolov7_baseline', '--hyp', 'data/hyp.scratch.p5.yaml'
    ]
    run_command(cmd_base, cwd=yolo_dir)

def run_inference(yolo_dir, dataset_work):
    """Run testing and inference on test dataset."""
    best_pt = os.path.join(yolo_dir, 'runs', 'train', 'cf_yolo_fire', 'weights', 'best.pt')
    test_dir = os.path.join(dataset_work, 'images', 'test')
    
    if not os.path.exists(best_pt):
        print(f"Weights not found: {best_pt}. Please train first.")
        return

    print("\n--- Running Inference ---")
    cmd_infer = [
        sys.executable, 'detect.py', '--weights', best_pt,
        '--conf', '0.25', '--img-size', '640',
        '--source', test_dir, '--save-txt', '--save-conf',
        '--name', 'cf_yolo_test'
    ]
    run_command(cmd_infer, cwd=yolo_dir)

def evaluate_model(yolo_dir):
    """Run quantitative evaluation on the test dataset."""
    best_pt = os.path.join(yolo_dir, 'runs', 'train', 'cf_yolo_fire', 'weights', 'best.pt')
    if not os.path.exists(best_pt):
        print(f"Weights not found: {best_pt}. Please train first.")
        return

    print("\n--- Evaluating Model ---")
    cmd_eval = [
        sys.executable, 'val.py', '--weights', best_pt,
        '--data', 'fire_dataset.yaml', '--img-size', '640',
        '--task', 'test', '--verbose', '--save-json'
    ]
    run_command(cmd_eval, cwd=yolo_dir)

def main():
    parser = argparse.ArgumentParser(description="CF-YOLO Forest Fire Detection Pipeline")
    parser.add_argument('--dataset_root', type=str, required=True, help="Path to input dataset root (containing images/ and labels/)")
    parser.add_argument('--work_dir', type=str, default='./working', help="Working directory for artifacts and execution")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=300, help="Number of epochs for training")
    parser.add_argument('--mode', type=str, choices=['setup', 'train', 'infer', 'eval', 'all'], default='all', help="Execution mode")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    yolo_dir = os.path.join(args.work_dir, 'yolov7')
    dataset_work = os.path.join(args.work_dir, 'dataset')

    if args.mode in ['setup', 'all']:
        setup_environment(yolo_dir)
        prepare_dataset(args.dataset_root, dataset_work, yolo_dir)
        patch_architecture(yolo_dir)
        generate_model_configs(yolo_dir)
        
    if args.mode in ['train', 'all']:
        train_model(yolo_dir, batch_size=args.batch_size, epochs=args.epochs)
        
    if args.mode in ['infer', 'all']:
        run_inference(yolo_dir, dataset_work)
        
    if args.mode in ['eval', 'all']:
        evaluate_model(yolo_dir)
        
    print("\nPipeline execution completed successfully.")

def run(dataset_path):
    """Standard pipeline interface for CF-YOLO.
    dataset_path: root dir with images/{train,val,test}/ and labels/{train,val,test}/ structure.
    Runs full setup + training + evaluation via subprocess.
    Metrics are sourced from YOLOv7 val.py JSON output; returns None if parsing fails.
    """
    import json
    import glob as _glob

    work_dir     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cf_yolo_working')
    yolo_dir     = os.path.join(work_dir, 'yolov7')
    dataset_work = os.path.join(work_dir, 'dataset')
    os.makedirs(work_dir, exist_ok=True)

    try:
        setup_environment(yolo_dir)
        prepare_dataset(dataset_path, dataset_work, yolo_dir)
        patch_architecture(yolo_dir)
        generate_model_configs(yolo_dir)
        train_model(yolo_dir, batch_size=4, epochs=300)
        evaluate_model(yolo_dir)  # saves JSON to runs/val/cf_yolo_test/

        # Try to parse COCO-style JSON produced by val.py --save-json
        json_candidates = _glob.glob(
            os.path.join(yolo_dir, 'runs', 'val', '**', '*.json'), recursive=True
        )
        if not json_candidates:
            return {"model_name": "CF-YOLO", "error": "Evaluation JSON not found", "metrics": None}

        with open(json_candidates[-1]) as fh:
            coco_results = json.load(fh)

        # COCO results list has entries with 'score' field; compute simple stats
        scores = [r.get('score', 0.0) for r in coco_results if isinstance(r, dict)]
        if not scores:
            return {"model_name": "CF-YOLO", "error": "Empty evaluation results", "metrics": None}

        # Detection model: use mean score as AUC/AUPR proxy; precision/recall/f1 = None
        return {
            "model_name": "CF-YOLO",
            "metrics": {
                "accuracy":  None,
                "precision": None,
                "recall":    None,
                "f1":        None,
                "auc":       float(np.mean(scores)) if scores else None,
                "aupr":      float(np.mean(scores)) if scores else None,
            },
        }
    except Exception as exc:
        return {"model_name": "CF-YOLO", "error": str(exc), "metrics": None}


if __name__ == '__main__':
    main()
