"""
main.py  —  Unified Forest Fire Model Comparison Pipeline
==========================================================
Discovers and runs every model's run(dataset_path) across the
Uav/, Satellite/, and Man/ folders, collects the standard metric
dict, and prints a sorted comparison table.

Usage
-----
    python main.py --config config.json

    # OR pass paths directly:
    python main.py \\
        --uav_dataset      /path/to/mendeley_dataset \\
        --satellite_dataset /path/to/modis_data \\
        --man_dataset       /path/to/human_camera_dataset \\
        --flame_dataset     /path/to/flame2_dataset \\
        --mobilenet_dataset /path/to/mobilenet_fire_dataset \\
        --ft_resnet_dataset /path/to/flame_train_val_test \\
        --ycbcr_dataset     /path/to/ycbcr_fire_dataset \\
        --cfyolo_dataset    /path/to/cfyolo_dataset

Config JSON format (alternative to CLI flags):
{
    "uav_dataset":       "...",
    "satellite_dataset": "...",
    "man_dataset":       "...",
    "flame_dataset":     "...",
    "mobilenet_dataset": "...",
    "ft_resnet_dataset": "...",
    "ycbcr_dataset":     "...",
    "cfyolo_dataset":    "..."
}
"""

import os
import sys
import json
import time
import argparse
import importlib.util
import traceback
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ForestFireSurvey/

def _load_module(module_name: str, file_path: str):
    """Dynamically import a Python file as a module."""
    spec   = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _safe_run(module_name: str, file_path: str, dataset_path: str) -> Dict[str, Any]:
    """Load a module and call its run(dataset_path), catching all errors."""
    print(f"\n{'='*60}")
    print(f"  Running: {module_name}")
    print(f"  File   : {os.path.relpath(file_path, ROOT)}")
    print(f"  Data   : {dataset_path}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        mod    = _load_module(module_name, file_path)
        result = mod.run(dataset_path)
    except Exception:
        tb = traceback.format_exc()
        print(f"[ERROR] {module_name} raised an exception:\n{tb}")
        result = {"model_name": module_name, "error": tb.splitlines()[-1], "metrics": None}
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return result


def _fmt(val: Optional[float], pct: bool = True) -> str:
    """Format a metric value for table display."""
    if val is None:
        return "   N/A  "
    return f"{val*100:6.2f}%" if pct else f"{val:.4f}"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_registry(args) -> list:
    """
    Returns a list of (source, module_name, file_path, dataset_path) tuples.
    Each entry maps one model file to the dataset it should receive.
    """
    p = lambda *parts: os.path.join(ROOT, *parts)

    registry = [
        # ── UAV ──────────────────────────────────────────────────────────────
        (
            "uav",
            "uav_dnn",
            p("Uav", "Uav_dnn(himanshu).py"),
            args.uav_dataset,          # Mendeley: Testing/ + Training and Validation/
        ),
        (
            "uav",
            "uav_unet",
            p("Uav", "amit_uav_1.py"),
            args.uav_dataset,          # expects fire/ nofire/ sub-dirs + pre-trained model
        ),
        (
            "uav",
            "mobilenet_uav",
            p("Uav", "forest_fire_mobilenet(Archit).py"),
            args.mobilenet_dataset,    # fire/ non_fire_images/
        ),
        (
            "uav",
            "deepfire_vgg19",
            p("Uav", "uav_deepfire(himanshu).py"),
            args.uav_dataset,          # Mendeley format
        ),
        (
            "uav",
            "fufdet",
            p("Uav", "uav_fufdet(himanshu).py"),
            args.flame_dataset,        # FLAME-2 dataset root
        ),

        # ── Satellite ────────────────────────────────────────────────────────
        (
            "satellite",
            "modis_satellite",
            p("Satellite", "satellite_modis20(himanshu).py"),
            args.satellite_dataset,    # dir with FIRMS CSV + HDF granules
        ),

        # ── Man (hand-held / ground-based) ───────────────────────────────────
        (
            "man",
            "hog_cnn_svm",
            p("Man", "Alok_forest_fire_pipeline.py"),
            args.man_dataset,          # fire/ nofire/ (64×64 images)
        ),
        (
            "man",
            "yolov8_man",
            p("Man", "amit_human_1.py"),
            args.man_dataset,          # fire/ nofire/ + pre-trained best.pt
        ),
        (
            "man",
            "xception_man",
            p("Man", "amit_human_2.py"),
            args.man_dataset,          # fire/ nofire/ + pre-trained xception h5
        ),
        (
            "man",
            "cf_yolo",
            p("Man", "cf_yolo_pipeline(Archit).py"),
            args.cfyolo_dataset,       # images/{train,val,test}/ + labels/…
        ),
        (
            "man",
            "ycbcr_fire",
            p("Man", "fire_detection_ycbcr(Archit).py"),
            args.ycbcr_dataset,        # fire/ non_fire/ sequential images
        ),
        (
            "man",
            "ft_resnet50",
            p("Man", "ft_resnet50_pipeline(Archit).py"),
            args.ft_resnet_dataset,    # train/ val/ test/ (ImageFolder)
        ),
    ]

    # Filter out entries where the file doesn't exist (graceful skip)
    valid = []
    for source, name, fpath, dpath in registry:
        if not os.path.isfile(fpath):
            print(f"[SKIP] File not found: {fpath}")
            continue
        if not dpath:
            print(f"[SKIP] No dataset path provided for {name}. Pass the flag or add it to config.json.")
            continue
        valid.append((source, name, fpath, dpath))
    return valid


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

COLS = ["model_name", "source", "accuracy", "precision", "recall", "f1", "auc", "aupr"]
COL_W = [26, 11, 10, 10, 9, 9, 9, 9]

def _header() -> str:
    parts = [f"{'MODEL':<{COL_W[0]}}",
             f"{'SOURCE':<{COL_W[1]}}",
             f"{'ACCURACY':>{COL_W[2]}}",
             f"{'PRECISION':>{COL_W[3]}}",
             f"{'RECALL':>{COL_W[4]}}",
             f"{'F1':>{COL_W[5]}}",
             f"{'AUC':>{COL_W[6]}}",
             f"{'AUPR':>{COL_W[7]}}"]
    sep = "  ".join("─" * w for w in COL_W)
    return "  ".join(parts) + "\n" + sep


def _row(r: dict) -> str:
    m    = r.get("metrics") or {}
    note = f"  ⚠ {r['error']}" if "error" in r else (f"  ℹ {r['note']}" if "note" in r else "")
    line = "  ".join([
        f"{r.get('model_name','?'):<{COL_W[0]}}",
        f"{r.get('source','?'):<{COL_W[1]}}",
        f"{_fmt(m.get('accuracy')):>{COL_W[2]}}",
        f"{_fmt(m.get('precision')):>{COL_W[3]}}",
        f"{_fmt(m.get('recall')):>{COL_W[4]}}",
        f"{_fmt(m.get('f1')):>{COL_W[5]}}",
        f"{_fmt(m.get('auc')):>{COL_W[6]}}",
        f"{_fmt(m.get('aupr')):>{COL_W[7]}}",
    ])
    return line + note


def _f1_sort_key(r: dict) -> float:
    """Sort key: models with metrics rank above failed ones."""
    m = r.get("metrics") or {}
    return m.get("f1") or -1.0


def print_table(results: list):
    """Print results sorted by F1 descending."""
    sorted_r = sorted(results, key=_f1_sort_key, reverse=True)
    print("\n\n" + "━" * 100)
    print("  FOREST FIRE MODEL COMPARISON  —  sorted by F1 ↓")
    print("━" * 100)
    print(_header())
    for r in sorted_r:
        print(_row(r))
    print("━" * 100)
    print(f"  Total models run: {len(results)}  |  "
          f"Successful: {sum(1 for r in results if r.get('metrics'))}")


def save_json(results: list, path: str):
    """Persist raw results to JSON."""
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nFull results saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Forest Fire Unified Model Comparison Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Path to JSON config file with dataset paths.")
    p.add_argument("--uav_dataset",       default=None,
                   help="Mendeley-style dataset root (Testing/ + Training and Validation/).")
    p.add_argument("--satellite_dataset", default=None,
                   help="Directory with FIRMS CSV + MOD021KM HDF granules.")
    p.add_argument("--man_dataset",       default=None,
                   help="Ground-camera dataset root with fire/ and nofire/ sub-dirs.")
    p.add_argument("--flame_dataset",     default=None,
                   help="FLAME-2 dataset root for FuFDet (254p RGB Images/ etc.).")
    p.add_argument("--mobilenet_dataset", default=None,
                   help="Dataset root for MobileNetV2 with fire/ and non_fire_images/.")
    p.add_argument("--ft_resnet_dataset", default=None,
                   help="ImageFolder root with train/ val/ test/ for FT-ResNet50.")
    p.add_argument("--ycbcr_dataset",     default=None,
                   help="Sequential image root with fire/ and non_fire/ for YCbCr model.")
    p.add_argument("--cfyolo_dataset",    default=None,
                   help="YOLO-format dataset root for CF-YOLO.")
    p.add_argument("--output_json",       default="results.json",
                   help="Path to save full results JSON (default: results.json).")
    p.add_argument("--skip",              nargs="*", default=[],
                   help="Module names to skip, e.g. --skip fufdet cf_yolo.")
    return p


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """Override argparse defaults with values from --config JSON if provided."""
    if args.config and os.path.isfile(args.config):
        with open(args.config) as fh:
            cfg = json.load(fh)
        for key, val in cfg.items():
            if getattr(args, key, None) is None:
                setattr(args, key, val)
    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()
    args   = merge_config(args)

    registry = _build_registry(args)

    if not registry:
        print("No models to run. Provide at least one dataset path.")
        return

    results = []
    total_t0 = time.time()

    for source, module_name, file_path, dataset_path in registry:
        if module_name in (args.skip or []):
            print(f"[SKIP] {module_name} (--skip flag)")
            continue

        result = _safe_run(module_name, file_path, dataset_path)
        result["source"]     = source
        result["model_name"] = result.get("model_name", module_name)
        results.append(result)

    print(f"\nAll models finished in {time.time() - total_t0:.1f}s")

    print_table(results)

    # Save to JSON next to this script
    out_path = args.output_json if os.path.isabs(args.output_json) else \
               os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_json)
    save_json(results, out_path)


if __name__ == "__main__":
    main()
