"""
main.py  —  Unified Forest Fire Model Comparison Pipeline
==========================================================
Discovers and runs every UAV model's run(dataset_path) across the
Uav/ folder, collects the standard metric dict, and prints a sorted
comparison table.

Usage
-----
    python main.py --config config.json

    # OR pass paths directly:
    python ./Combine/main.py --dataset ./dataset/uav/FLAME --epochs 3

    # PowerShell multiline example:
    python ./Combine/main.py `
        --dataset ./dataset/uav/FLAME `
        --epochs 3

Config JSON format (alternative to CLI flags):
{
    "dataset": "..."
}
"""

import os
import sys
import json
import time
import argparse
import importlib.util
import inspect
import traceback
from typing import Optional, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ForestFireSurvey/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".matplotlib_cache"))
os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, ".torch_cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(ROOT, ".cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

def _load_module(module_name: str, file_path: str):
    """Dynamically import a Python file as a module."""
    spec   = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _safe_run(module_name: str, file_path: str, dataset_path: str, epochs: Optional[int] = None) -> Dict[str, Any]:
    """Load a module and call its run(dataset_path), catching all errors."""
    print(f"\n{'='*60}")
    print(f"  Running: {module_name}")
    print(f"  File   : {os.path.relpath(file_path, ROOT)}")
    print(f"  Data   : {dataset_path}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        mod = _load_module(module_name, file_path)
        run_kwargs = {}
        if epochs is not None and "epochs" in inspect.signature(mod.run).parameters:
            run_kwargs["epochs"] = epochs
        result = mod.run(dataset_path, **run_kwargs)
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
            "uav_fire_unet",
            p("Uav", "amit_uav_1.py"),
            args.dataset,
        ),
        (
            "uav",
            "uav_dnn",
            p("Uav", "Uav_dnn(himanshu).py"),
            args.dataset,
        ),
        (
            "uav",
            "mobilenet_uav",
            p("Uav", "forest_fire_mobilenet(Archit).py"),
            args.dataset,
        ),
        # (
        #     "uav",
        #     "deepfire_vgg19",
        #     p("Uav", "uav_deepfire(himanshu).py"),
        #     args.dataset,
        # ),
    ]

    # Filter out entries where the file doesn't exist (graceful skip)
    valid = []
    for source, name, fpath, dpath in registry:
        if args.sources and source not in set(args.sources):
            continue
        if args.only and name not in set(args.only):
            continue
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
        description="UAV Forest Fire Unified Model Comparison Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Path to JSON config file with dataset paths.")
    p.add_argument("--dataset", default=None,
                   help="UAV FLAME dataset root (dataset/uav/FLAME).")
    p.add_argument("--output_json",       default="results.json",
                   help="Path to save full results JSON (default: results.json).")
    p.add_argument("--epochs",            type=int, default=None,
                   help="Override training epochs for model run() functions that support it.")
    p.add_argument("--sources",           nargs="*", default=None,
                   choices=["uav"],
                   help="Only run selected source groups. UAV-only pipeline.")
    p.add_argument("--only",              nargs="*", default=None,
                   help="Only run these module names, e.g. --only hog_cnn_svm uav_dnn.")
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
    if getattr(args, "dataset", None) is None and getattr(args, "uav_dataset", None) is not None:
        args.dataset = args.uav_dataset
    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()
    args   = merge_config(args)

    registry = _build_registry(args)
    if args.sources:
        wanted_sources = set(args.sources)
        registry = [entry for entry in registry if entry[0] in wanted_sources]
    if args.only:
        wanted_modules = set(args.only)
        registry = [entry for entry in registry if entry[1] in wanted_modules]

    if not registry:
        print("No models to run. Provide at least one dataset path.")
        return

    results = []
    total_t0 = time.time()

    for source, module_name, file_path, dataset_path in registry:
        if module_name in (args.skip or []):
            print(f"[SKIP] {module_name} (--skip flag)")
            continue

        result = _safe_run(module_name, file_path, dataset_path, epochs=args.epochs)
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
