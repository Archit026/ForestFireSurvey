# # CF-YOLO Forest Fire Detection — Kaggle Notebook
# **Author:** Archit  
# YOLOv7 + custom CF-YOLO modules: CoordAtt, SimAM, D2F, SSC.
# 
# Dataset layout expected:
# ```
# /kaggle/input/<dataset>/
#   images/train/  images/val/  images/test/
#   labels/train/  labels/val/  labels/test/
# ```

# ── Cell 1: Install dependencies ──────────────────────────────────────────────
import subprocess, sys
def pip(*pkgs): subprocess.run([sys.executable,'-m','pip','install','-q',*pkgs], check=True)
pip('timm','einops','thop','pycocotools')
print('Done.')


# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import os, sys, glob, json, shutil, yaml, urllib.request
import numpy as np
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
DEVICE = '0' if torch.cuda.is_available() else 'cpu'


# ── Cell 3: Configuration ─────────────────────────────────────────────────────
# ⚠ Edit DATASET_ROOT to your Kaggle input path
DATASET_ROOT = '/kaggle/input/fire-detection-yolo-format'
WORK_DIR     = '/kaggle/working'
YOLO_DIR     = os.path.join(WORK_DIR, 'yolov7')
DATASET_WORK = os.path.join(WORK_DIR, 'dataset')
EPOCHS       = 100
BATCH_SIZE   = 16
IMG_SIZE     = 640
print(f'Dataset: {DATASET_ROOT}')
print(f'YOLOv7 dir: {YOLO_DIR}')


# ── Cell 4: Clone YOLOv7 ─────────────────────────────────────────────────────
if not os.path.exists(YOLO_DIR):
    os.system(f'git clone https://github.com/WongKinYiu/yolov7.git {YOLO_DIR}')
else:
    print('YOLOv7 already cloned.')
os.system(f'{sys.executable} -m pip install -q -r {YOLO_DIR}/requirements.txt')
print('YOLOv7 ready.')


# ── Cell 5: Prepare Dataset ───────────────────────────────────────────────────
for split in ['train','val','test']:
    os.makedirs(f'{DATASET_WORK}/images/{split}', exist_ok=True)
    os.makedirs(f'{DATASET_WORK}/labels/{split}', exist_ok=True)
    for kind in ['images','labels']:
        src = f'{DATASET_ROOT}/{kind}/{split}'
        dst = f'{DATASET_WORK}/{kind}/{split}'
        if os.path.exists(src):
            for f in glob.glob(f'{src}/*'):
                shutil.copy2(f, dst)
            print(f'[{split}] {kind}: {len(os.listdir(dst))} files')
        else:
            print(f'WARNING: {src} not found')

yaml_cfg = {
    'train': f'{DATASET_WORK}/images/train',
    'val'  : f'{DATASET_WORK}/images/val',
    'test' : f'{DATASET_WORK}/images/test',
    'nc'   : 1,
    'names': ['fire']
}
yaml_path = f'{YOLO_DIR}/fire_dataset.yaml'
with open(yaml_path,'w') as f:
    yaml.dump(yaml_cfg, f, default_flow_style=False)
print(f'Dataset YAML saved: {yaml_path}')


# ── Cell 6: Patch common.py with CF-YOLO Modules ─────────────────────────────
CUSTOM_MODULES = '''
# --- CF-YOLO MODULES (auto-inserted) ---
class CoordAtt(nn.Module):
    """Coordinate Attention for spatial-channel awareness."""
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
        n, c, h, w = x.shape
        xh = self.pool_h(x)
        xw = self.pool_w(x).permute(0,1,3,2)
        y  = self.act(self.bn1(self.conv1(torch.cat([xh,xw],dim=2))))
        xh, xw = torch.split(y,[h,w],dim=2)
        return x * self.conv_h(xh).sigmoid() * self.conv_w(xw.permute(0,1,3,2)).sigmoid()

class SimAM(nn.Module):
    """SimAM: Simple Parameter-Free Attention Module."""
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
    def forward(self, x):
        b,c,h,w = x.size()
        n = w*h - 1
        mu = x.mean(dim=[2,3],keepdim=True)
        xm = x - mu
        y  = xm**2 / (4*(xm.pow(2).sum(dim=[2,3],keepdim=True)/n + self.e_lambda) + 0.5)
        return x * torch.sigmoid(y)

class DSConv(nn.Module):
    """Depthwise-Separable Convolution."""
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        p = p if p is not None else k//2
        self.dw  = nn.Conv2d(c1,c1,k,s,p,groups=c1,bias=False)
        self.pw  = nn.Conv2d(c1,c2,1,1,0,bias=False)
        self.bn  = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class DSBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2*e)
        self.cv1 = Conv(c1,c_,1,1)
        self.cv2 = DSConv(c_,c2,3,1)
        self.add = shortcut and c1==c2
    def forward(self, x):
        return x+self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class D2F(nn.Module):
    """C2F backbone with DSConv Bottleneck."""
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c   = int(c2*e)
        self.cv1 = Conv(c1,2*self.c,1,1)
        self.cv2 = Conv((n+2)*self.c,c2,1)
        self.m   = nn.ModuleList(DSBottleneck(self.c,self.c,shortcut) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2,1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y,1))

class SSC(nn.Module):
    """SPPFCSPC + SimAM serial pooling."""
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, k=13):
        super().__init__()
        c_ = int(2*c2*e)
        self.cv1   = Conv(c1,c_,1,1)
        self.cv2   = Conv(c1,c_,1,1)
        self.cv3   = Conv(c_,c_,3,1)
        self.cv4   = Conv(4*c_,c2,1,1)
        self.m     = nn.MaxPool2d(kernel_size=k,stride=1,padding=k//2)
        self.simam = SimAM()
    def forward(self, x):
        x1 = self.cv3(self.cv1(x))
        y1 = self.simam(self.m(x1))
        y2 = self.simam(self.m(y1))
        y3 = self.simam(self.m(y2))
        return self.cv4(torch.cat([x1,y1,y2,y3],1))
'''

common_path = f'{YOLO_DIR}/models/common.py'
with open(common_path,'r') as f:
    src = f.read()
if '# --- CF-YOLO MODULES (auto-inserted) ---' not in src:
    with open(common_path,'a') as f:
        f.write(CUSTOM_MODULES)
    print('common.py patched.')
else:
    print('common.py already patched.')


# ── Cell 7: Patch yolo.py parse_model ────────────────────────────────────────
yolo_path = f'{YOLO_DIR}/models/yolo.py'
with open(yolo_path,'r') as f:
    src = f.read()

import_marker = '# --- CF-YOLO imports ---'
parse_marker  = '# --- CF-YOLO parse_model registration ---'

if import_marker not in src:
    imp_line = f'\n{import_marker}\nfrom models.common import CoordAtt,SimAM,DSConv,DSBottleneck,D2F,SSC\n'
    src = src.replace('from models.common import *', 'from models.common import *' + imp_line)

if parse_marker not in src:
    target   = 'elif m in [nn.BatchNorm2d]:'
    reg_code = (f'elif m in [CoordAtt,SimAM,DSConv,DSBottleneck,D2F,SSC]:  {parse_marker}\n'
                f'            c2 = args[1] if len(args)>1 else c1\n        ')
    src = src.replace(target, reg_code + target)

with open(yolo_path,'w') as f:
    f.write(src)
print('yolo.py patched.')


# ── Cell 8: Generate CF-YOLO Config YAML ─────────────────────────────────────
cfg_dir = f'{YOLO_DIR}/cfg/training'
os.makedirs(cfg_dir, exist_ok=True)

cf_yaml = """nc: 1
depth_multiple: 1.0
width_multiple: 1.0

anchors:
  - [12,16, 19,36, 40,28]
  - [36,75, 76,55, 72,146]
  - [142,110, 192,243, 459,401]

backbone:
  [[-1,1,Conv,[32,3,1]],[-1,1,Conv,[64,3,2]],[-1,1,Conv,[64,3,1]],
   [-1,1,Conv,[128,3,2]],[-1,1,Conv,[64,1,1]],[-2,1,Conv,[64,1,1]],
   [-1,1,Conv,[64,3,1]],[-1,1,Conv,[64,3,1]],[-1,1,Conv,[64,3,1]],
   [-1,1,Conv,[64,3,1]],[[-1,-3,-5,-6],1,Concat,[1]],
   [-1,1,Conv,[256,1,1]],[-1,1,MP,[]],[-1,1,Conv,[128,1,1]],
   [-3,1,Conv,[128,1,1]],[-1,1,Conv,[128,3,2]],[[-1,-3],1,Concat,[1]],
   [-1,1,Conv,[128,1,1]],[-2,1,Conv,[128,1,1]],[-1,1,Conv,[128,3,1]],
   [-1,1,Conv,[128,3,1]],[-1,1,Conv,[128,3,1]],[-1,1,Conv,[128,3,1]],
   [[-1,-3,-5,-6],1,Concat,[1]],[-1,1,Conv,[512,1,1]],
   [-1,1,CoordAtt,[512,512]],[-1,1,MP,[]],[-1,1,Conv,[256,1,1]],
   [-3,1,Conv,[256,1,1]],[-1,1,Conv,[256,3,2]],[[-1,-3],1,Concat,[1]],
   [-1,1,Conv,[256,1,1]],[-2,1,Conv,[256,1,1]],[-1,1,Conv,[256,3,1]],
   [-1,1,Conv,[256,3,1]],[-1,1,Conv,[256,3,1]],[-1,1,Conv,[256,3,1]],
   [[-1,-3,-5,-6],1,Concat,[1]],[-1,1,Conv,[1024,1,1]],[-1,1,MP,[]],
   [-1,1,Conv,[512,1,1]],[-3,1,Conv,[512,1,1]],[-1,1,Conv,[512,3,2]],
   [[-1,-3],1,Concat,[1]],[-1,1,Conv,[256,1,1]],[-2,1,Conv,[256,1,1]],
   [-1,1,Conv,[256,3,1]],[-1,1,Conv,[256,3,1]],[-1,1,Conv,[256,3,1]],
   [-1,1,Conv,[256,3,1]],[[-1,-3,-5,-6],1,Concat,[1]],
   [-1,1,Conv,[1024,1,1]],[-1,1,CoordAtt,[1024,1024]]]

head:
  [[-1,1,SSC,[512,512]],[-1,1,Conv,[256,1,1]],
   [-1,1,nn.Upsample,[None,2,'nearest']],[38,1,Conv,[256,1,1]],
   [[-1,-2],1,Concat,[1]],[-1,1,D2F,[256,256,3]],
   [-1,1,Conv,[128,1,1]],[-1,1,nn.Upsample,[None,2,'nearest']],
   [24,1,Conv,[128,1,1]],[[-1,-2],1,Concat,[1]],[-1,1,D2F,[128,128,3]],
   [-1,1,MP,[]],[-1,1,Conv,[128,1,1]],[-3,1,Conv,[128,1,1]],
   [-1,1,Conv,[128,3,2]],[[-1,-3,59],1,Concat,[1]],[-1,1,D2F,[256,256,3]],
   [-1,1,MP,[]],[-1,1,Conv,[256,1,1]],[-3,1,Conv,[256,1,1]],
   [-1,1,Conv,[256,3,2]],[[-1,-3,54],1,Concat,[1]],[-1,1,D2F,[512,512,3]],
   [[64,70,76],1,IDetect,[nc,anchors]]]
"""

cf_yaml_path = f'{cfg_dir}/cf-yolo.yaml'
with open(cf_yaml_path,'w') as f:
    f.write(cf_yaml)
print(f'CF-YOLO config saved: {cf_yaml_path}')


# ── Cell 9: Download Pretrained Weights ───────────────────────────────────────
import urllib.request
weights_path = f'{YOLO_DIR}/yolov7.pt'
if not os.path.exists(weights_path):
    print('Downloading yolov7.pt ...')
    urllib.request.urlretrieve(
        'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt',
        weights_path
    )
    print('Downloaded.')
else:
    print('Weights already present.')


# ── Cell 10: Train CF-YOLO ────────────────────────────────────────────────────
train_cmd = (
    f'{sys.executable} {YOLO_DIR}/train.py '
    f'--workers 4 --device {DEVICE} '
    f'--batch-size {BATCH_SIZE} --epochs {EPOCHS} '
    f'--img {IMG_SIZE} {IMG_SIZE} '
    f'--data {YOLO_DIR}/fire_dataset.yaml '
    f'--cfg {YOLO_DIR}/cfg/training/cf-yolo.yaml '
    f'--weights {YOLO_DIR}/yolov7.pt '
    f'--name cf_yolo_fire '
    f'--hyp {YOLO_DIR}/data/hyp.scratch.p5.yaml '
    f'--project {WORK_DIR}/runs/train'
)
print('Running:', train_cmd)
ret = os.system(train_cmd)
print('Train exit code:', ret)


# ── Cell 11: Evaluate on Test Set ────────────────────────────────────────────
best_pt  = f'{WORK_DIR}/runs/train/cf_yolo_fire/weights/best.pt'
assert os.path.exists(best_pt), f'Weights not found: {best_pt}. Run training first.'

val_cmd = (
    f'{sys.executable} {YOLO_DIR}/val.py '
    f'--weights {best_pt} '
    f'--data {YOLO_DIR}/fire_dataset.yaml '
    f'--img-size {IMG_SIZE} '
    f'--task test --verbose --save-json '
    f'--project {WORK_DIR}/runs/val '
    f'--name cf_yolo_test '
    f'--device {DEVICE}'
)
print('Running:', val_cmd)
ret = os.system(val_cmd)
print('Val exit code:', ret)


# ── Cell 12: Parse Metrics from val.py output ─────────────────────────────────
# YOLOv7 val.py --verbose prints per-class P, R, mAP50, mAP50-95
# We also parse the COCO JSON for AUC proxy

import re

val_log_dir = f'{WORK_DIR}/runs/val/cf_yolo_test'
results_txt  = f'{WORK_DIR}/runs/train/cf_yolo_fire/results.txt'

precision = recall = f1 = mAP50 = mAP5095 = None

# Try results.txt (last line has final epoch metrics)
if os.path.exists(results_txt):
    with open(results_txt) as f:
        lines = [l.strip() for l in f if l.strip()]
    if lines:
        # Format: epoch/total  gpu_mem  box  obj  cls  total  labels  img_size
        #         P  R  mAP@.5  mAP@.5:.95  val_box  val_obj  val_cls
        last = lines[-1].split()
        try:
            # Columns 8..11 = P, R, mAP@.5, mAP@.5:.95
            precision, recall, mAP50, mAP5095 = float(last[8]), float(last[9]), float(last[10]), float(last[11])
            f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        except (IndexError, ValueError):
            print('Could not parse results.txt, trying JSON fallback.')

# JSON fallback for AUC proxy
json_files = glob.glob(f'{val_log_dir}/**/*.json', recursive=True)
mean_score = None
if json_files:
    with open(json_files[-1]) as f:
        coco = json.load(f)
    scores = [r.get('score',0.0) for r in coco if isinstance(r,dict)]
    mean_score = float(np.mean(scores)) if scores else None

print('\n========== CF-YOLO Test Results ==========')
print(f'  Precision : {precision:.4f}' if precision is not None else '  Precision : N/A')
print(f'  Recall    : {recall:.4f}'    if recall    is not None else '  Recall    : N/A')
print(f'  F1        : {f1:.4f}'        if f1        is not None else '  F1        : N/A')
print(f'  mAP@0.5   : {mAP50:.4f}'    if mAP50     is not None else '  mAP@0.5   : N/A')
print(f'  mAP@.5:.95: {mAP5095:.4f}'  if mAP5095   is not None else '  mAP@.5:.95: N/A')
print(f'  AUC proxy : {mean_score:.4f}' if mean_score is not None else '  AUC proxy : N/A')
print('==========================================')


# ── Cell 13: Save Results to JSON ─────────────────────────────────────────────
results = {
    'model_name': 'CF-YOLO',
    'metrics': {
        'precision': round(float(precision), 4) if precision is not None else None,
        'recall'   : round(float(recall),    4) if recall    is not None else None,
        'f1'       : round(float(f1),        4) if f1        is not None else None,
        'mAP50'    : round(float(mAP50),     4) if mAP50     is not None else None,
        'mAP5095'  : round(float(mAP5095),   4) if mAP5095   is not None else None,
        'auc_proxy': round(float(mean_score),4) if mean_score is not None else None,
    }
}
out_path = '/kaggle/working/cf_yolo_results.json'
with open(out_path,'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved → {out_path}')
print(json.dumps(results, indent=2))


# ── Cell 14: Run Inference on Test Images (Visual check) ──────────────────────
best_pt  = f'{WORK_DIR}/runs/train/cf_yolo_fire/weights/best.pt'
test_img_dir = f'{DATASET_WORK}/images/test'

infer_cmd = (
    f'{sys.executable} {YOLO_DIR}/detect.py '
    f'--weights {best_pt} '
    f'--conf 0.25 --img-size {IMG_SIZE} '
    f'--source {test_img_dir} '
    f'--save-txt --save-conf '
    f'--project {WORK_DIR}/runs/detect '
    f'--name cf_yolo_test '
    f'--device {DEVICE}'
)
print('Running inference...')
ret = os.system(infer_cmd)
print('Inference exit code:', ret)
print(f'Results at: {WORK_DIR}/runs/detect/cf_yolo_test/')

