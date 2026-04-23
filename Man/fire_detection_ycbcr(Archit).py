import os
import glob
import math
import warnings
from collections import deque

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = '/kaggle/input/fire-dataset'  # Update to local path if needed
FLAT_FOLDER  = False                         # True if no fire/non_fire sub-dirs
IMG_EXTS     = ('*.png', '*.jpg', '*.jpeg', '*.bmp')

# Operational thresholds (paper defaults)
THR_NON_REF  = 45          # |R - Y| threshold for non-reflection (Eq. 7)
THR_REF      = 37          # |R - Y| threshold for reflection     (Eq. 6)
W_MAIN       = 0.9         # weight for largest connected region (seed)
W_OTHER      = 0.1         # weight for peripheral regions
ENTROPY_RANGE= range(10, 51) # candidate thr_gray values
FRAMES_MIN   = 30          # minimum frames per clip for feature stats

# Decision thresholds (paper Table 2 / Fig 13-15)
CS_THRESH    = 27.63       # area variation coefficient
DC_THRESH    = 130.0       # centroid dispersion
R_THRESH     = 0.3         # circularity upper bound

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def collect_images(root, label, exts=IMG_EXTS):
    """Recursively collect image paths under root/label/."""
    paths = []
    folder = os.path.join(root, label) if not FLAT_FOLDER else root
    for ext in exts:
        paths += glob.glob(os.path.join(folder, '**', ext), recursive=True)
    paths.sort()
    return paths

def group_into_clips(paths, clip_size=30):
    """Organise paths into clips (groups of FRAMES_MIN consecutive images)."""
    clips = []
    for i in range(0, len(paths), clip_size):
        chunk = paths[i:i + clip_size]
        if len(chunk) >= 10:   # at least 10 frames
            clips.append(chunk)
    return clips

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: INITIAL SEGMENTATION (IMPROVED YCBCR)
# ─────────────────────────────────────────────────────────────────────────────
def bgr_to_ycbcr(bgr):
    """
    Convert BGR image to YCbCr using the paper's Eq. (1).
    Returns separate Y, Cb, Cr channels as float32.
    """
    bgr = bgr.astype(np.float32)
    B, G, R = bgr[:,:,0], bgr[:,:,1], bgr[:,:,2]
    Y  = 16  + 0.257*R + 0.504*G + 0.098*B
    Cb = 128 - 0.148*R - 0.291*G + 0.439*B
    Cr = 128 + 0.439*R - 0.368*G - 0.071*B
    return Y, Cb, Cr

def segment_fire(bgr_frame, thr_ref=THR_REF, thr_non_ref=THR_NON_REF):
    """
    Returns:
      mask_ref     – fire mask from reflection model (Eq. 6)
      mask_non_ref – fire mask from non-reflection model (Eq. 7)
      selected     – chosen mask ('ref' or 'non_ref')
      mask_final   – the selected binary mask (uint8, 0/255)
    """
    Y, Cb, Cr = bgr_to_ycbcr(bgr_frame)
    R_ch = bgr_frame[:,:,2].astype(np.float32)

    Y_mean  = np.mean(Y)
    Cr_mean = np.mean(Cr)

    # High-intensity rule: Y > 220 (Eq. 4)
    hi = (Y > 220) & (Cr > Cb) & (Y > Y_mean) & (Cr > Cr_mean)

    # Base conditions shared by Eq. 6 & 7
    base = (Y > Cb) & (Cr > Cb) & (Y > Y_mean) & (Cr > Cr_mean)

    diff_RY = np.abs(R_ch - Y)

    # Reflection model (Eq. 6): |R - Y| <= thr_ref
    mask_ref_normal     = base & (diff_RY <= thr_ref) & (Y <= 220)
    mask_ref            = (hi | mask_ref_normal).astype(np.uint8) * 255

    # Non-reflection model (Eq. 7): |R - Y| >= thr_non_ref
    mask_non_ref_normal = base & (diff_RY >= thr_non_ref) & (Y <= 220)
    mask_non_ref        = (hi | mask_non_ref_normal).astype(np.uint8) * 255

    # Select by area comparison (Section 3.3)
    area_ref     = np.count_nonzero(mask_ref)
    area_non_ref = np.count_nonzero(mask_non_ref)

    if area_ref < area_non_ref:
        selected, mask_final = 'ref', mask_ref
    else:
        selected, mask_final = 'non_ref', mask_non_ref

    return mask_ref, mask_non_ref, selected, mask_final

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: FINE SEGMENTATION (REGION GROWING)
# ─────────────────────────────────────────────────────────────────────────────
def get_weighted_seed(binary_mask):
    """
    Analyse connected regions of binary_mask.
    Returns (x_seed, y_seed), list_of_region_dicts, label_image.
    """
    label_img = measure.label(binary_mask > 0, connectivity=2)
    regions   = measure.regionprops(label_img)

    if not regions:
        h, w = binary_mask.shape
        return (w // 2, h // 2), [], label_img

    # Sort by area descending
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)

    region_list = []
    sum_w, sum_wx, sum_wy = 0.0, 0.0, 0.0

    for i, reg in enumerate(regions_sorted):
        cy, cx = reg.centroid       # skimage: (row, col)
        w_j = W_MAIN if i == 0 else W_OTHER
        sum_w  += w_j
        sum_wx += w_j * cx
        sum_wy += w_j * cy
        region_list.append({'label': reg.label, 'area': reg.area,
                            'cx': cx, 'cy': cy, 'weight': w_j})

    x_seed = int(round(sum_wx / sum_w))
    y_seed = int(round(sum_wy / sum_w))

    # Clamp to image bounds
    H, W = binary_mask.shape
    x_seed = max(0, min(x_seed, W - 1))
    y_seed = max(0, min(y_seed, H - 1))

    return (x_seed, y_seed), region_list, label_img

def region_grow(gray_img, seed_xy, thr_gray):
    """
    BFS-based 8-connected region growing.
    seed_xy : (col, row) tuple
    Returns a binary mask (uint8, 0/255) of the grown region.
    """
    H, W = gray_img.shape
    x0, y0 = seed_xy

    visited = np.zeros((H, W), dtype=bool)
    fire_mask = np.zeros((H, W), dtype=np.uint8)

    queue = deque()
    queue.append((y0, x0))
    visited[y0, x0] = True

    seed_val = float(gray_img[y0, x0])
    pixel_count = 0
    pixel_sum   = seed_val

    neighbours = [(-1,-1),(-1,0),(-1,1),
                  ( 0,-1),        ( 0,1),
                  ( 1,-1),( 1,0),( 1,1)]

    while queue:
        r, c = queue.popleft()
        fire_mask[r, c] = 255
        pixel_count += 1
        pixel_sum   += gray_img[r, c]
        h_tilde = pixel_sum / (pixel_count + 1)   # Eq. 13: running mean

        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                visited[nr, nc] = True
                if abs(float(gray_img[nr, nc]) - h_tilde) <= thr_gray:
                    queue.append((nr, nc))

    return fire_mask

def compute_entropy(gray_img, seed_xy, thr_gray):
    """
    Grow region, then compute binary Shannon entropy H.
    Uses the paper's formulation from Equations 14-21.
    """
    grown = region_grow(gray_img, seed_xy, thr_gray)
    label_img = measure.label(grown > 0, connectivity=2)
    regions   = measure.regionprops(label_img, intensity_image=gray_img)

    if not regions:
        return 0.0, grown

    # Eq. 14: mean and variance for each region
    D_vals = [np.std(gray_img[label_img == r.label])**2 for r in regions]
    max_D  = max(D_vals) if max(D_vals) > 0 else 1.0

    # Eq. 15: normalised uniformity M_i = D_i / max(D)
    M_vals = [d / max_D for d in D_vals]

    # Eq. 16: inter-region contrast G_i = |mean_i - mean_neighbour|
    means = [np.mean(gray_img[label_img == r.label]) for r in regions]
    if len(means) > 1:
        G_vals = [abs(means[i] - np.mean(means[:i] + means[i+1:]))
                  for i in range(len(means))]
    else:
        G_vals = [0.0]
    max_G  = max(G_vals) if max(G_vals) > 0 else 1.0

    # Eq. 17: normalised inter-region contrast T_i
    T_vals = [g / max_G for g in G_vals]

    # Eq. 19: rescale M_i and T_i to [0, 1/e]
    e_inv = 1.0 / math.e
    M_scaled = [m * e_inv for m in M_vals]
    T_scaled = [t * e_inv for t in T_vals]

    # Eq. 21: binary entropy H
    H_val = 0.0
    for m, t in zip(M_scaled, T_scaled):
        val_max = max(m, t) if max(m, t) > 0 else 1e-9
        val_min = min(m, t) if min(m, t) > 0 else 1e-9
        H_val += -(val_max * math.log(val_max) + val_min * math.log(val_min))

    return H_val, grown

def find_optimal_threshold(gray_img, seed_xy, candidates=ENTROPY_RANGE):
    """
    Sweep candidate thresholds and select the one maximising entropy H.
    Returns (best_thr, best_mask, entropy_curve).
    """
    best_H, best_thr, best_mask = -1, candidates[0], None
    curve = []

    for thr in candidates:
        H, mask = compute_entropy(gray_img, seed_xy, thr)
        curve.append((thr, H))
        if H > best_H:
            best_H, best_thr, best_mask = H, thr, mask

    return best_thr, best_mask, curve

def stage2_fine_segment(bgr_frame):
    """
    Returns:
      seed_xy     – weighted seed (col, row)
      best_thr    – optimal gray threshold
      fine_mask   – final binary fire mask (uint8)
      entropy_curve
    """
    # Stage 1
    _, _, _, coarse_mask = segment_fire(bgr_frame)

    # Morphological post-processing (fill small holes, remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    coarse_mask = cv2.morphologyEx(coarse_mask, cv2.MORPH_CLOSE, kernel)
    coarse_mask = cv2.morphologyEx(coarse_mask, cv2.MORPH_OPEN,  kernel)

    # Connected component → weighted seed
    seed_xy, _, _ = get_weighted_seed(coarse_mask)

    # Convert to grayscale for region growing
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    # Entropy-based optimal threshold
    best_thr, fine_mask, entropy_curve = find_optimal_threshold(gray, seed_xy)

    return seed_xy, best_thr, fine_mask, entropy_curve


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: FIRE IDENTIFICATION (MULTI-FEATURE FUSION)
# ─────────────────────────────────────────────────────────────────────────────
def extract_frame_features(bgr_frame):
    """
    Returns dict with keys:
      area      – fire pixel count
      cx, cy    – centroid (col, row)
      circularity – Eq. 33: R = 4*pi*area / perimeter^2
      fine_mask
    """
    _, _, fine_mask, _ = stage2_fine_segment(bgr_frame)

    area = int(np.count_nonzero(fine_mask))
    cx, cy = 0, 0
    circ   = 0.0

    if area > 0:
        contours, _ = cv2.findContours(fine_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M_mom   = cv2.moments(largest)
            if M_mom['m00'] > 0:
                cx = M_mom['m10'] / M_mom['m00']
                cy = M_mom['m01'] / M_mom['m00']
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 0:
                circ = (4 * math.pi * area) / (perimeter ** 2)   # Eq. 33

    return {'area': area, 'cx': cx, 'cy': cy,
            'circularity': circ, 'fine_mask': fine_mask}

def identify_clip(image_paths, pixel_area_cm2=1.0, verbose=False):
    """
    Process a list of image paths (one video clip).
    Returns classification dict with all quantitative indicators.
    """
    n = min(len(image_paths), 100)   # cap at 100 frames for speed
    areas, cx_list, cy_list, circ_list = [], [], [], []

    for path in image_paths[:n]:
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        feat = extract_frame_features(bgr)
        areas.append(feat['area'] * pixel_area_cm2)
        cx_list.append(feat['cx'])
        cy_list.append(feat['cy'])
        circ_list.append(feat['circularity'])

    if len(areas) < FRAMES_MIN:
        if verbose:
            print(f'  WARNING: only {len(areas)} frames; need {FRAMES_MIN} for reliable stats.')

    areas_arr = np.array(areas)
    mu_S  = np.mean(areas_arr) if len(areas_arr) > 0 else 1e-9
    std_S = np.std(areas_arr)  if len(areas_arr) > 0 else 0.0
    C_S   = std_S / mu_S if mu_S > 0 else 0.0   # Eq. 27

    sigma_X = np.std(cx_list) if cx_list else 0.0
    sigma_Y = np.std(cy_list) if cy_list else 0.0
    d_C     = sigma_X * sigma_Y                  # Eq. 32

    mean_R  = np.mean(circ_list) if circ_list else 1.0  # Eq. 33 averaged

    # Decision logic (Section 5.4)
    is_fire = (C_S > CS_THRESH) and (d_C > DC_THRESH) and (mean_R < R_THRESH)

    result = {
        'n_frames':      len(areas),
        'C_S':           round(C_S, 4),
        'd_C':           round(d_C, 4),
        'mean_R':        round(mean_R, 4),
        'prediction':    'FIRE' if is_fire else 'NON-FIRE',
        'cs_pass':       C_S > CS_THRESH,
        'dc_pass':       d_C > DC_THRESH,
        'r_pass':        mean_R < R_THRESH,
    }

    if verbose:
        print(f"  C_S={C_S:.2f} ({'✓' if result['cs_pass'] else '✗'} > {CS_THRESH}) | "
              f"d_C={d_C:.2f} ({'✓' if result['dc_pass'] else '✗'} > {DC_THRESH}) | "
              f"R={mean_R:.3f} ({'✓' if result['r_pass'] else '✗'} < {R_THRESH}) "
              f"→ {result['prediction']}")

    return result

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION & UTILS
# ─────────────────────────────────────────────────────────────────────────────
def segmentation_accuracy(gt_mask, pred_mask):
    """
    Eq. (22): C = |A ∩ B| / max(|A|, |B|)
    Both masks should be binary (0/255 or bool).
    """
    A = (gt_mask > 0).astype(np.uint8)
    B = (pred_mask > 0).astype(np.uint8)
    intersection = np.count_nonzero(A & B)
    denom = max(np.count_nonzero(A), np.count_nonzero(B))
    return intersection / denom if denom > 0 else 0.0

def evaluate_dataset(fire_clips, non_fire_clips, max_clips=999):
    """
    Run identification on all clips and return precision, recall, and F1 score.
    """
    results = []

    print('Processing FIRE clips...')
    for i, clip in enumerate(fire_clips[:max_clips]):
        print(f'  Clip {i+1}/{min(len(fire_clips), max_clips)}', end=' ')
        res = identify_clip(clip, verbose=True)
        res['gt'] = 'FIRE'
        results.append(res)

    print('\nProcessing NON-FIRE clips...')
    for i, clip in enumerate(non_fire_clips[:max_clips]):
        print(f'  Clip {i+1}/{min(len(non_fire_clips), max_clips)}', end=' ')
        res = identify_clip(clip, verbose=True)
        res['gt'] = 'NON-FIRE'
        results.append(res)

    df = pd.DataFrame(results)

    TP = len(df[(df.gt == 'FIRE')     & (df.prediction == 'FIRE')])
    FP = len(df[(df.gt == 'NON-FIRE') & (df.prediction == 'FIRE')])
    FN = len(df[(df.gt == 'FIRE')     & (df.prediction == 'NON-FIRE')])
    TN = len(df[(df.gt == 'NON-FIRE') & (df.prediction == 'NON-FIRE')])

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return df, {'precision': precision, 'recall': recall, 'f1': f1, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Initializing Fire Detection System...")

    if not FLAT_FOLDER:
        fire_paths     = collect_images(DATASET_ROOT, 'fire')
        non_fire_paths = collect_images(DATASET_ROOT, 'non_fire')
    else:
        all_paths      = collect_images(DATASET_ROOT, '')
        fire_paths, non_fire_paths = all_paths, []

    print(f'Fire images    : {len(fire_paths)}')
    print(f'Non-fire images: {len(non_fire_paths)}')
    print(f'Total          : {len(fire_paths) + len(non_fire_paths)}')

    if not fire_paths and not non_fire_paths:
        print("No images found. Please verify DATASET_ROOT.")
        return

    fire_clips     = group_into_clips(fire_paths)
    non_fire_clips = group_into_clips(non_fire_paths)

    print(f'Fire clips     : {len(fire_clips)}')
    print(f'Non-fire clips : {len(non_fire_clips)}')

    # Set to a smaller number for testing, or None for full dataset
    MAX_CLIPS_TO_PROCESS = 5

    if fire_clips or non_fire_clips:
        print(f"\nEvaluating dataset (max {MAX_CLIPS_TO_PROCESS} clips per class for demonstration)...")
        df, metrics = evaluate_dataset(fire_clips, non_fire_clips, max_clips=MAX_CLIPS_TO_PROCESS)

        print('\n' + '=' * 50)
        print(f"  TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics['FN']}  TN={metrics['TN']}")
        print(f"  Precision : {metrics['precision']*100:.2f}%  (paper: 98.3%)")
        print(f"  Recall    : {metrics['recall']*100:.2f}%  (paper: 95.0%)")
        print(f"  F1-Score  : {metrics['f1']*100:.2f}%  (paper: 96.0%)")
        print('=' * 50)

        # Output results to CSV
        output_file = 'identification_results.csv'
        df.to_csv(output_file, index=False)
        print(f'Results saved -> {output_file}')

def run(dataset_path):
    """Standard pipeline interface.
    dataset_path: root dir with 'fire/' and 'non_fire/' sub-folders of sequential images.
    This is a traditional rule-based CV method; no probability scores exist,
    so AUC and AUPR are returned as None.
    """
    if not os.path.exists(dataset_path):
        return {"model_name": "YCbCr-FireDetection", "error": f"Dataset not found: {dataset_path}", "metrics": None}

    try:
        fire_paths     = collect_images(dataset_path, 'fire')
        non_fire_paths = collect_images(dataset_path, 'non_fire')

        if not fire_paths and not non_fire_paths:
            return {"model_name": "YCbCr-FireDetection", "error": "No images found", "metrics": None}

        fire_clips     = group_into_clips(fire_paths)
        non_fire_clips = group_into_clips(non_fire_paths)

        _, metrics = evaluate_dataset(fire_clips, non_fire_clips)

        total = metrics['TP'] + metrics['TN'] + metrics['FP'] + metrics['FN']
        acc   = (metrics['TP'] + metrics['TN']) / total if total > 0 else 0.0

        return {
            "model_name": "YCbCr-FireDetection",
            "metrics": {
                "accuracy":  float(acc),
                "precision": float(metrics['precision']),
                "recall":    float(metrics['recall']),
                "f1":        float(metrics['f1']),
                "auc":       None,   # rule-based method, no probability scores
                "aupr":      None,
            },
        }
    except Exception as exc:
        return {"model_name": "YCbCr-FireDetection", "error": str(exc), "metrics": None}


if __name__ == "__main__":
    main()
