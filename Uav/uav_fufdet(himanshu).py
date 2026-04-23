import os
import re
import sys
import math
import time
import random
import shutil
import json
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import albumentations as A

# ── Primary device (GPU 0 used for inference/eval) ───────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()

# ── Reproducibility (paper does not specify seed; using 42) ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Paths ─────────────────────────────────────────────────────
BASE       = '/kaggle/input/datasets/aryankashyapnaveen/flame-2/254p_Frame_Pairs'
RGB_DIR    = os.path.join(BASE, '254p RGB Images')
WORK       = '/kaggle/working'
SORTED_DIR = os.path.join(WORK, 'sorted')
CLEAN_DIR  = os.path.join(WORK, 'cleaned')
ANNOT_DIR  = os.path.join(WORK, 'annotated')
AUG_DIR    = os.path.join(WORK, 'augmented')
SPLIT_DIR  = os.path.join(WORK, 'splits')
CKPT_DIR   = os.path.join(WORK, 'checkpoints')

# Label ranges from Frame_Pair_Labels.txt
LABEL_RANGES = [
    (1,     13700, 'N'), (13701, 14699, 'Y'), (14700, 15980, 'Y'),
    (15981, 19802, 'Y'), (19803, 19899, 'Y'), (19900, 27183, 'Y'),
    (27184, 27514, 'Y'), (27515, 31294, 'Y'), (31295, 31509, 'Y'),
    (31510, 33597, 'Y'), (33598, 33929, 'Y'), (33930, 36550, 'Y'),
    (36551, 38030, 'N'), (38031, 38153, 'Y'), (38154, 41642, 'N'),
    (41642, 45279, 'Y'), (45280, 51206, 'N'), (51207, 52286, 'Y'),
    (52287, 53451, 'N'),
]


def setup_directories():
    for d in [SORTED_DIR, CLEAN_DIR, ANNOT_DIR, AUG_DIR, SPLIT_DIR, CKPT_DIR]:
        os.makedirs(d, exist_ok=True)
    print('Directories ready.')


def get_fire_label(n):
    for s, e, lbl in LABEL_RANGES:
        if s <= n <= e:
            return lbl
    return None


def sort_dataset():
    print('Phase 2: Sorting dataset...')
    for cls in ['Fire', 'NoFire']:
        os.makedirs(os.path.join(SORTED_DIR, cls), exist_ok=True)

    pattern = re.compile(r'\((\d+)\)')
    if not os.path.exists(RGB_DIR):
        print(f"Directory {RGB_DIR} not found. Skipping sorting.")
        return

    all_files = sorted(os.listdir(RGB_DIR))
    counts = {'Fire': 0, 'NoFire': 0, 'Skipped': 0}

    for fname in all_files:
        if not fname.endswith('.jpg'):
            counts['Skipped'] += 1
            continue
        m = pattern.search(fname)
        if not m:
            counts['Skipped'] += 1
            continue
        lbl = get_fire_label(int(m.group(1)))
        if lbl is None:
            counts['Skipped'] += 1
            continue
        folder = 'Fire' if lbl == 'Y' else 'NoFire'
        shutil.copy2(os.path.join(RGB_DIR, fname),
                     os.path.join(SORTED_DIR, folder, fname))
        counts[folder] += 1

    print(f"Sorting complete -> Fire: {counts['Fire']:,}, NoFire: {counts['NoFire']:,}")


def histogram_similarity_cv(img1, img2):
    h1 = cv2.calcHist([img1], [0,1,2], None, [32,32,32], [0,256,0,256,0,256])
    h2 = cv2.calcHist([img2], [0,1,2], None, [32,32,32], [0,256,0,256,0,256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


def deduplicate_worker(args):
    src_folder, dst_folder, threshold, window, gpu_id = args
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    os.makedirs(dst_folder, exist_ok=True)
    if not os.path.exists(src_folder): return []

    files = sorted(
        [f for f in os.listdir(src_folder) if f.endswith('.jpg')],
        key=lambda x: int(re.search(r'\((\d+)\)', x).group(1)) if re.search(r'\((\d+)\)', x) else 0
    )
    n = len(files)
    discard = set()

    for i in range(n):
        if i in discard: continue
        img_a = cv2.imread(os.path.join(src_folder, files[i]))
        if img_a is None: continue
        for j in range(1, window + 1):
            nxt = i + j
            if nxt >= n or nxt in discard: continue
            img_b = cv2.imread(os.path.join(src_folder, files[nxt]))
            if img_b is None: continue
            if histogram_similarity_cv(img_a, img_b) > threshold:
                discard.add(nxt)

    kept = []
    for i, fname in enumerate(files):
        if i not in discard:
            kept.append(fname)
            shutil.copy2(os.path.join(src_folder, fname),
                         os.path.join(dst_folder, fname))
    return kept


def clean_dataset(threshold=0.98, window=4):
    print('Phase 3: Data Cleaning (Histogram Deduplication)...')
    fire_args   = (os.path.join(SORTED_DIR,'Fire'),   os.path.join(CLEAN_DIR,'Fire'),   threshold, window, 0)
    nofire_args = (os.path.join(SORTED_DIR,'NoFire'), os.path.join(CLEAN_DIR,'NoFire'), threshold, window, 1 if torch.cuda.device_count()>1 else 0)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(deduplicate_worker, [fire_args, nofire_args]))
    print(f'Deduplication complete -> Fire kept: {len(results[0])}, NoFire kept: {len(results[1])}')
    return results


def detect_fire_bbox_yolo(img_bgr, min_area=15):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_bgr.shape[:2]
    m1 = cv2.inRange(hsv, ( 0, 50, 50), (15, 255,255))
    m2 = cv2.inRange(hsv, (15, 50, 50), (40, 255,255))
    m3 = cv2.inRange(hsv, (160,50, 50), (180,255,255))
    mask = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3))
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE,k), cv2.MORPH_OPEN,k)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for c in cnts:
        if cv2.contourArea(c) < min_area: continue
        x,y,bw,bh = cv2.boundingRect(c)
        px,py = int(bw*.1), int(bh*.1)
        x1,y1 = max(0,x-px), max(0,y-py)
        x2,y2 = min(w,x+bw+px), min(h,y+bh+py)
        bboxes.append(( ((x1+x2)/2)/w, ((y1+y2)/2)/h,
                        (x2-x1)/w,     (y2-y1)/h ))
    
    if not bboxes:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thr  = max(int(gray.mean() + 2.0*gray.std()), 150)
        _,bm = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        cnts2,_ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts2:
            if cv2.contourArea(c) < min_area: continue
            x,y,bw,bh = cv2.boundingRect(c)
            px,py = int(bw*.1), int(bh*.1)
            x1,y1 = max(0,x-px), max(0,y-py)
            x2,y2 = min(w,x+bw+px), min(h,y+bh+py)
            bboxes.append(( ((x1+x2)/2)/w, ((y1+y2)/2)/h,
                            (x2-x1)/w,     (y2-y1)/h ))
    return bboxes

def annotate_dataset():
    print('Phase 4: Bounding Box Annotation...')
    img_out   = os.path.join(ANNOT_DIR, 'images')
    label_out = os.path.join(ANNOT_DIR, 'labels')
    os.makedirs(img_out,   exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    fire_dir = os.path.join(CLEAN_DIR, 'Fire')
    if os.path.exists(fire_dir):
        for fname in os.listdir(fire_dir):
            img = cv2.imread(os.path.join(fire_dir, fname))
            if img is None: continue
            bboxes = detect_fire_bbox_yolo(img)
            if not bboxes: continue
            stem = os.path.splitext(fname)[0]
            shutil.copy2(os.path.join(fire_dir,fname), os.path.join(img_out,fname))
            with open(os.path.join(label_out, stem+'.txt'),'w') as f:
                for cx,cy,bw,bh in bboxes:
                    f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')

    nofire_dir = os.path.join(CLEAN_DIR, 'NoFire')
    if os.path.exists(nofire_dir):
        for fname in os.listdir(nofire_dir):
            stem = os.path.splitext(fname)[0]
            shutil.copy2(os.path.join(nofire_dir,fname), os.path.join(img_out,fname))
            open(os.path.join(label_out, stem+'.txt'),'w').close()
            
    print(f'Annotation complete.')

def read_yolo(path):
    bboxes, labels = [], []
    if os.path.exists(path) and os.path.getsize(path)>0:
        with open(path) as f:
            for line in f:
                p = line.strip().split()
                if len(p)==5:
                    labels.append(int(p[0]))
                    bboxes.append([float(x) for x in p[1:]])
    return bboxes, labels

def write_yolo(path, bboxes, labels):
    with open(path,'w') as f:
        for lbl,(cx,cy,bw,bh) in zip(labels,bboxes):
            f.write(f'{lbl} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')

def augment_dataset():
    print('Phase 5: Data Augmentation...')
    aug_img_dir   = os.path.join(AUG_DIR,'images')
    aug_label_dir = os.path.join(AUG_DIR,'labels')
    os.makedirs(aug_img_dir,   exist_ok=True)
    os.makedirs(aug_label_dir, exist_ok=True)

    bp = A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.1)
    AUGMENTATIONS = [
        ('noise',     A.Compose([A.GaussNoise(var_limit=(25,625), p=1.0)],              bbox_params=bp)),
        ('crop',      A.Compose([A.RandomResizedCrop(size=(254,254), scale=(0.9,1.0), p=1.0)], bbox_params=bp)),
        ('flip',      A.Compose([A.HorizontalFlip(p=1.0)],                             bbox_params=bp)),
        ('rotate',    A.Compose([A.Rotate(limit=15, p=1.0)],                           bbox_params=bp)),
        ('translate', A.Compose([A.ShiftScaleRotate(shift_limit=0.1,scale_limit=0,rotate_limit=0,p=1.0)], bbox_params=bp)),
    ]

    img_out = os.path.join(ANNOT_DIR, 'images')
    label_out = os.path.join(ANNOT_DIR, 'labels')
    if not os.path.exists(img_out): return

    all_base = os.listdir(img_out)
    for fname in all_base:
        if not fname.endswith('.jpg'): continue
        stem = os.path.splitext(fname)[0]
        img  = cv2.cvtColor(cv2.imread(os.path.join(img_out,fname)), cv2.COLOR_BGR2RGB)
        bboxes, labels = read_yolo(os.path.join(label_out, stem+'.txt'))
        
        shutil.copy2(os.path.join(img_out,   fname),  os.path.join(aug_img_dir,   fname))
        shutil.copy2(os.path.join(label_out, stem+'.txt'), os.path.join(aug_label_dir, stem+'.txt'))
        
        for aug_name, transform in AUGMENTATIONS:
            try:
                res = transform(image=img, bboxes=bboxes, labels=labels)
            except Exception:
                continue
            out_stem = f'{stem}_{aug_name}'
            cv2.imwrite(os.path.join(aug_img_dir,   out_stem+'.jpg'),
                        cv2.cvtColor(res['image'], cv2.COLOR_RGB2BGR))
            write_yolo(os.path.join(aug_label_dir, out_stem+'.txt'),
                       res['bboxes'], res['labels'])
    print(f'Augmentation complete.')


def split_dataset():
    print('Phase 6: Dataset Splits...')
    aug_img_dir   = os.path.join(AUG_DIR,'images')
    aug_label_dir = os.path.join(AUG_DIR,'labels')
    
    for split in ['train','val','test']:
        os.makedirs(os.path.join(SPLIT_DIR,split,'images'), exist_ok=True)
        os.makedirs(os.path.join(SPLIT_DIR,split,'labels'), exist_ok=True)

    if not os.path.exists(aug_img_dir): return
    all_aug = sorted([f for f in os.listdir(aug_img_dir) if f.endswith('.jpg')])
    
    if len(all_aug) == 0: return

    def is_fire(fname):
        p = os.path.join(aug_label_dir, os.path.splitext(fname)[0]+'.txt')
        return os.path.exists(p) and os.path.getsize(p)>0

    strat = [1 if is_fire(f) else 0 for f in all_aug]

    train_f, temp_f, _, temp_s = train_test_split(
        all_aug, strat, test_size=0.19, random_state=SEED, stratify=strat)
    val_f, test_f = train_test_split(
        temp_f, test_size=0.526, random_state=SEED, stratify=temp_s)

    def copy_split(files, split):
        for fname in files:
            stem = os.path.splitext(fname)[0]
            shutil.copy2(os.path.join(aug_img_dir,   fname),
                         os.path.join(SPLIT_DIR,split,'images',fname))
            shutil.copy2(os.path.join(aug_label_dir, stem+'.txt'),
                         os.path.join(SPLIT_DIR,split,'labels',stem+'.txt'))

    copy_split(train_f, 'train')
    copy_split(val_f,   'val')
    copy_split(test_f,  'test')
    print(f"Splits complete -> Train: {len(train_f)}, Val: {len(val_f)}, Test: {len(test_f)}")


# ── Architecture (Phase 7) ───────────────────────────────────

class ECA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        t = int(abs(math.log2(channels)/2 + 0.5))
        k = t if t%2 else t+1
        self.avg  = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,k,padding=k//2,bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self,x):
        y = self.avg(x).squeeze(-1).transpose(-1,-2)
        y = self.conv(y).transpose(-1,-2).unsqueeze(-1)
        return x*self.sig(y)

class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()
    def forward(self,x):
        b,c,h,w = x.size()
        n   = h*w-1
        mu  = x.mean(dim=[2,3],keepdim=True)
        var = (x-mu).pow(2).sum(dim=[2,3],keepdim=True)/n
        e_t = (x-mu).pow(2)/(4*(var+self.e_lambda))+0.5
        return x*self.act(e_t)

class AAFRM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.eca   = ECA(channels)
        self.simam = SimAM()
        self.cbr   = nn.Sequential(
            nn.Conv2d(channels,channels,1),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.w = nn.Parameter(torch.ones(3))
    def forward(self,x):
        w = torch.softmax(self.w, dim=0)
        return self.cbr(w[0]*self.eca(x) + w[1]*self.simam(x) + w[2]*x)

class RECAB(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.eca = ECA(in_ch)
        self.conv = (nn.ConvTranspose2d(in_ch,out_ch,4,stride=2,padding=1)
                     if upsample else
                     nn.Conv2d(in_ch,out_ch,3,padding=1))
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(self.eca(x)+x)))

class CoordAttention(nn.Module):
    def __init__(self, in_ch, reduction=32):
        super().__init__()
        mid = max(8, in_ch//reduction)
        self.conv_hw = nn.Conv2d(in_ch,mid,1)
        self.bn      = nn.BatchNorm2d(mid)
        self.act     = nn.Hardswish()
        self.conv_h  = nn.Conv2d(mid,in_ch,1)
        self.conv_w  = nn.Conv2d(mid,in_ch,1)
    def forward(self,x):
        B,C,H,W = x.shape
        xh = x.mean(dim=3,keepdim=True)
        xw = x.mean(dim=2,keepdim=True).permute(0,1,3,2)
        y  = self.act(self.bn(self.conv_hw(torch.cat([xh,xw],dim=2))))
        yh,yw = torch.split(y,[H,W],dim=2)
        return x*torch.sigmoid(self.conv_h(yh))*torch.sigmoid(self.conv_w(yw.permute(0,1,3,2)))

class CAHead(nn.Module):
    def __init__(self, in_ch=64, num_classes=1):
        super().__init__()
        self.ca = CoordAttention(in_ch)
        def branch(oc):
            return nn.Sequential(
                nn.Conv2d(in_ch,in_ch,3,padding=1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
                nn.Conv2d(in_ch,oc,1))
        self.heatmap = branch(num_classes)
        self.size    = branch(2)
        self.offset  = branch(2)
    def forward(self,x):
        x = self.ca(x)
        return torch.sigmoid(self.heatmap(x)), self.size(x), self.offset(x)

class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w    = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=w)
        self.c1 = nn.Sequential(base.conv1,base.bn1,base.relu,base.maxpool) 
        self.c2 = base.layer1   
        self.c3 = base.layer2   
        self.c4 = base.layer3   
        self.c5 = base.layer4   
    def forward(self,x):
        c1=self.c1(x); c2=self.c2(c1); c3=self.c3(c2); c4=self.c4(c3); c5=self.c5(c4)
        return c1,c2,c3,c4,c5

class FuFDet(nn.Module):
    def __init__(self, pretrained=True, use_aafrm=True, use_recab=True, use_cahead=True):
        super().__init__()
        self.encoder = Encoder(pretrained)
        def mk_aafrm(ch): return AAFRM(ch) if use_aafrm else nn.Identity()
        self.aafrm1=mk_aafrm(64); self.aafrm2=mk_aafrm(256)
        self.aafrm3=mk_aafrm(512); self.aafrm4=mk_aafrm(1024)
        
        def mk_dec(ic,oc,up=True):
            if use_recab: return RECAB(ic,oc,upsample=up)
            conv = (nn.ConvTranspose2d(ic,oc,4,stride=2,padding=1)
                    if up else nn.Conv2d(ic,oc,3,padding=1))
            return nn.Sequential(conv,nn.BatchNorm2d(oc),nn.ReLU(inplace=True))
            
        self.dec5 = mk_dec(2048,1024)          
        self.dec4 = mk_dec(2048, 512)          
        self.dec3 = mk_dec(1024, 256)          
        self.dec2 = mk_dec( 512, 128, up=False)
        self.dec1 = mk_dec( 192,  64, up=False)
        self.head = CAHead(64) if use_cahead else self._plain_head()

    def _plain_head(self):
        class H(nn.Module):
            def __init__(s):
                super().__init__()
                def b(oc): return nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,oc,1))
                s.heatmap=b(1); s.size=b(2); s.offset=b(2)
            def forward(s,x): return torch.sigmoid(s.heatmap(x)),s.size(x),s.offset(x)
        return H()

    def forward(self,x):
        c1,c2,c3,c4,c5 = self.encoder(x)
        f1=self.aafrm1(c1); f2=self.aafrm2(c2)
        f3=self.aafrm3(c3); f4=self.aafrm4(c4)
        d5=self.dec5(c5)
        d4=self.dec4(torch.cat([d5,f4],1))
        d3=self.dec3(torch.cat([d4,f3],1))
        d2=self.dec2(torch.cat([d3,f2],1))
        d1=self.dec1(torch.cat([d2,f1],1))
        return self.head(d1)


# ── Training and Evaluation ───────────────────────────────────

INPUT_SIZE  = 512
HEATMAP_RES = 128   
SIGMA       = 2

def gaussian2d(shape, sigma=1):
    m,n = [(s-1.)/2. for s in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    return np.exp(-(x*x+y*y)/(2*sigma*sigma))

def draw_gaussian(hm, cx, cy, r):
    d = 2*r+1
    g = gaussian2d((d,d), sigma=d/6)
    H,W = hm.shape
    x0,y0 = int(cx),int(cy)
    l=min(x0,r); r2=min(W-x0,r+1)
    t=min(y0,r); b=min(H-y0,r+1)
    np.maximum(hm[y0-t:y0+b,x0-l:x0+r2], g[r-t:r+b,r-l:r+r2],
               out=hm[y0-t:y0+b,x0-l:x0+r2])

class FireDataset(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.imgs    = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.tf = T.Compose([
            T.Resize((INPUT_SIZE,INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        fname = self.imgs[idx]
        stem  = os.path.splitext(fname)[0]
        img_t = self.tf(Image.open(os.path.join(self.img_dir,fname)).convert('RGB'))
        H=W=HEATMAP_RES
        hm  = np.zeros((1,H,W),np.float32)
        szm = np.zeros((2,H,W),np.float32)
        om  = np.zeros((2,H,W),np.float32)
        msk = np.zeros((H,W),  np.float32)
        lp  = os.path.join(self.lbl_dir, stem+'.txt')
        if os.path.exists(lp) and os.path.getsize(lp)>0:
            with open(lp) as f:
                for line in f:
                    p = line.strip().split()
                    if len(p)!=5: continue
                    _,cx_n,cy_n,bw_n,bh_n = map(float,p)
                    cx_h,cy_h = cx_n*W, cy_n*H
                    ix,iy = int(cx_h),int(cy_h)
                    if 0<=ix<W and 0<=iy<H:
                        draw_gaussian(hm[0],cx_h,cy_h,SIGMA)
                        szm[0,iy,ix]=bw_n*W; szm[1,iy,ix]=bh_n*H
                        om[0,iy,ix]=cx_h-ix; om[1,iy,ix]=cy_h-iy
                        msk[iy,ix]=1.0
        return (img_t, torch.from_numpy(hm), torch.from_numpy(szm),
                torch.from_numpy(om), torch.from_numpy(msk))

def focal_loss(pred, gt, alpha=2, beta=4):
    pos = (gt==1).float(); neg = (gt<1).float()
    lp  = -(1-pred).pow(alpha)*torch.log(pred.clamp(1e-6))*pos
    ln  = -(1-gt).pow(beta)*pred.pow(alpha)*torch.log((1-pred).clamp(1e-6))*neg
    return (lp+ln).sum()/pos.sum().clamp(min=1)

def reg_l1(pred, gt, mask):
    mask = mask.unsqueeze(1).expand_as(pred)
    return F.l1_loss(pred*mask, gt*mask, reduction='sum')/(mask.sum()+1e-6)

def fufdet_loss(hm_p,sz_p,off_p, hm_g,sz_g,off_g, mask, lam_size=0.1, lam_off=1.0):
    lk  = focal_loss(hm_p, hm_g)
    lsz = reg_l1(sz_p,  sz_g,  mask)
    lo  = reg_l1(off_p, off_g, mask)
    return lk + lam_size*lsz + lam_off*lo, lk.item(), lsz.item(), lo.item()

def to_dev(*tensors):
    return [t.to(device) for t in tensors]

def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total = 0.0
    for batch in loader:
        imgs,hm_g,sz_g,off_g,msk = to_dev(*batch)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            hm_p,sz_p,off_p = model(imgs)
            loss,*_ = fufdet_loss(hm_p,sz_p,off_p, hm_g,sz_g,off_g, msk)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer); scaler.update()
        total += loss.item()
    return total/max(len(loader), 1)

@torch.no_grad()
def validate(model, loader):
    model.eval()
    total = 0.0
    for batch in loader:
        imgs,hm_g,sz_g,off_g,msk = to_dev(*batch)
        hm_p,sz_p,off_p = model(imgs)
        loss,*_ = fufdet_loss(hm_p,sz_p,off_p, hm_g,sz_g,off_g, msk)
        total += loss.item()
    return total/max(len(loader), 1)

def get_base(m):
    return m.module if isinstance(m, torch.nn.DataParallel) else m

def decode_centernet(hm, sz, off, conf_thresh=0.3, nms_k=100):
    B,C,H,W = hm.shape
    nms_hm = F.max_pool2d(hm,3,stride=1,padding=1)
    keep   = (nms_hm==hm).float()*hm
    results = []
    for b in range(B):
        flat = keep[b,0].view(-1)
        topk = min(nms_k, int((flat>conf_thresh).sum()))
        if topk==0: results.append([]); continue
        scores,inds = flat.topk(topk)
        ys=(inds//W).float(); xs=(inds%W).float()
        ox=off[b,0].view(-1)[inds]; oy=off[b,1].view(-1)[inds]
        xs=(xs+ox)/W; ys=(ys+oy)/H
        bw=sz[b,0].view(-1)[inds]/W; bh=sz[b,1].view(-1)[inds]/H
        results.append([[float(cx),float(cy),float(abs(w_)),float(abs(h_)),float(s)]
                        for s,cx,cy,w_,h_ in zip(scores.cpu().numpy(),
                            xs.cpu().numpy(),ys.cpu().numpy(),
                            bw.cpu().numpy(),bh.cpu().numpy()) if s>conf_thresh])
    return results

def iou_cxcy(b1,b2):
    x1a,y1a=b1[0]-b1[2]/2,b1[1]-b1[3]/2
    x2a,y2a=b1[0]+b1[2]/2,b1[1]+b1[3]/2
    x1b,y1b=b2[0]-b2[2]/2,b2[1]-b2[3]/2
    x2b,y2b=b2[0]+b2[2]/2,b2[1]+b2[3]/2
    iw=max(0,min(x2a,x2b)-max(x1a,x1b))
    ih=max(0,min(y2a,y2b)-max(y1a,y1b))
    inter=iw*ih
    return inter/(b1[2]*b1[3]+b2[2]*b2[3]-inter+1e-6)

def evaluate_model(model, loader, iou_thresh=0.5, conf_thresh=0.3):
    eval_model = get_base(model)
    eval_model.eval()
    all_preds,all_gts=[],[]
    t0=time.time(); n_imgs=0
    with torch.no_grad():
        for imgs,hm_g,sz_g,off_g,msk in loader:
            imgs=imgs.to(device)
            hm_p,sz_p,off_p = eval_model(imgs)
            preds = decode_centernet(hm_p,sz_p,off_p,conf_thresh)
            for b in range(imgs.size(0)):
                ys,xs=np.where(msk[b].numpy()>0)
                gts=[[ (x+off_g[b,0,y,x].item())/HEATMAP_RES,
                        (y+off_g[b,1,y,x].item())/HEATMAP_RES,
                        sz_g[b,0,y,x].item()/HEATMAP_RES,
                        sz_g[b,1,y,x].item()/HEATMAP_RES ] for y,x in zip(ys,xs)]
                all_gts.append(gts); all_preds.append(preds[b])
            n_imgs+=imgs.size(0)
    fps=n_imgs/(time.time()-t0)
    all_sc,all_tp,all_fp=[],[],[]
    total_gt=0
    for prs,gts in zip(all_preds,all_gts):
        total_gt+=len(gts)
        matched=[False]*len(gts)
        for det in sorted(prs,key=lambda x:-x[4]):
            bi,bv=0,-1
            for j,g in enumerate(gts):
                v=iou_cxcy(det[:4],g)
                if v>bv: bv,bi=v,j
            if bv>=iou_thresh and not matched[bi]:
                matched[bi]=True; all_tp.append(1); all_fp.append(0)
            else:
                all_tp.append(0); all_fp.append(1)
            all_sc.append(det[4])
    if not all_sc:
        return {'AP':0,'Precision':0,'Recall':0,'F1':0,'FPS':round(fps,1)}
    ord_=np.argsort(-np.array(all_sc))
    tp_c=np.cumsum(np.array(all_tp)[ord_])
    fp_c=np.cumsum(np.array(all_fp)[ord_])
    pc=tp_c/(tp_c+fp_c+1e-6)
    rc=tp_c/(total_gt+1e-6)
    ap=float(np.trapz(pc,rc))
    f1c=2*pc*rc/(pc+rc+1e-6)
    bi=int(np.argmax(f1c))
    return {'AP':round(ap*100,2),'Precision':round(float(pc[bi])*100,2),
            'Recall':round(float(rc[bi])*100,2),'F1':round(float(f1c[bi]),4),
            'FPS':round(fps,1)}


def train_pipeline():
    print("Phase 8 & 9: Training and Evaluation")
    train_dir = os.path.join(SPLIT_DIR,'train','images')
    if not os.path.exists(train_dir):
        print("Data not found. Skipping training.")
        return
        
    train_ds = FireDataset(os.path.join(SPLIT_DIR,'train','images'), os.path.join(SPLIT_DIR,'train','labels'))
    val_ds   = FireDataset(os.path.join(SPLIT_DIR,'val',  'images'), os.path.join(SPLIT_DIR,'val',  'labels'))
    test_ds  = FireDataset(os.path.join(SPLIT_DIR,'test', 'images'), os.path.join(SPLIT_DIR,'test', 'labels'))

    loader_16   = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
    loader_8    = DataLoader(train_ds, batch_size=8,  shuffle=True,  num_workers=4, pin_memory=True)
    val_loader  = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    FROZEN_EP   = 50
    UNFROZEN_EP = 150
    INIT_LR     = 5e-4
    MIN_LR      = 5e-6

    model = FuFDet(pretrained=True).to(device)
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=FROZEN_EP+UNFROZEN_EP, eta_min=MIN_LR)
    scaler    = torch.cuda.amp.GradScaler()

    best_val = float('inf')
    CKPT_PATH = os.path.join(CKPT_DIR,'best_fufdet.pth')

    print('Phase 1: Frozen Encoder')
    for p in get_base(model).encoder.parameters(): p.requires_grad = False
    
    for ep in range(1, FROZEN_EP+1):
        tl = train_one_epoch(model, loader_16, optimizer, scaler)
        vl = validate(model, val_loader)
        scheduler.step()
        if vl < best_val:
            best_val = vl
            torch.save(get_base(model).state_dict(), CKPT_PATH)
        if ep % 5 == 0:
            print(f' Epoch {ep} | Train: {tl:.4f} | Val: {vl:.4f}')

    print('Phase 2: Fine-Tuning')
    for p in get_base(model).encoder.parameters(): p.requires_grad = True
    
    for ep in range(FROZEN_EP+1, FROZEN_EP+UNFROZEN_EP+1):
        tl = train_one_epoch(model, loader_8, optimizer, scaler)
        vl = validate(model, val_loader)
        scheduler.step()
        if vl < best_val:
            best_val = vl
            torch.save(get_base(model).state_dict(), CKPT_PATH)
        if ep % 10 == 0:
            print(f' Epoch {ep} | Train: {tl:.4f} | Val: {vl:.4f}')

    print("Evaluating Best Model:")
    get_base(model).load_state_dict(torch.load(CKPT_PATH, map_location=device))
    metrics = evaluate_model(model, test_loader)
    print(metrics)

def run(dataset_path):
    """Standard pipeline interface for FuFDet.
    dataset_path: root of the FLAME-2 dataset (containing '254p RGB Images/' etc.).
    AP (Average Precision) is used as the AUPR proxy; it is also reported as AUC.
    accuracy is N/A for this object-detection model.
    """
    global BASE, RGB_DIR

    if dataset_path and os.path.exists(dataset_path):
        BASE    = dataset_path
        RGB_DIR = os.path.join(BASE, '254p RGB Images')

    try:
        setup_directories()
        sort_dataset()
        clean_dataset()
        annotate_dataset()
        augment_dataset()
        split_dataset()

        train_dir = os.path.join(SPLIT_DIR, 'train', 'images')
        if not os.path.exists(train_dir):
            return {"model_name": "FuFDet", "error": "Splits not found after preparation", "metrics": None}

        train_pipeline()  # trains and saves best checkpoint

        # Evaluate best checkpoint
        test_ds     = FireDataset(os.path.join(SPLIT_DIR, 'test', 'images'),
                                  os.path.join(SPLIT_DIR, 'test', 'labels'))
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
        m = FuFDet(pretrained=False).to(device)
        ckpt = os.path.join(CKPT_DIR, 'best_fufdet.pth')
        if os.path.exists(ckpt):
            m.load_state_dict(torch.load(ckpt, map_location=device))
        det_metrics = evaluate_model(m, test_loader)

        ap   = det_metrics['AP']        / 100.0
        prec = det_metrics['Precision'] / 100.0
        rec  = det_metrics['Recall']    / 100.0
        f1   = det_metrics['F1']

        return {
            "model_name": "FuFDet",
            "metrics": {
                "accuracy":  None,   # object-detection model — no image-level accuracy
                "precision": float(prec),
                "recall":    float(rec),
                "f1":        float(f1),
                "auc":       float(ap),   # AP used as ROC-AUC proxy
                "aupr":      float(ap),
            },
        }
    except Exception as exc:
        return {"model_name": "FuFDet", "error": str(exc), "metrics": None}


if __name__ == '__main__':
    # Set this to true to run data preparing phase
    prepare_data = False

    if prepare_data:
        setup_directories()
        sort_dataset()
        clean_dataset()
        annotate_dataset()
        augment_dataset()
        split_dataset()

    # Set this parameter to true if you are ready to train on a GPU env
    do_train = False
    if do_train:
        train_pipeline()
    else:
        print("Data preperation and training flags are set to False.")

