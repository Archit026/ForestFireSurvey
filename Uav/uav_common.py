from pathlib import Path
from typing import Optional, Sequence, Tuple


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def uav_results_root() -> Path:
    path = repo_root() / "results" / "uav"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_results_dir(model_slug: str) -> Path:
    path = uav_results_root() / model_slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_subdir_casefold(parent: Path, candidates: Sequence[str]) -> Optional[Path]:
    if not parent.exists():
        return None

    children = {child.name.casefold(): child for child in parent.iterdir() if child.is_dir()}
    for candidate in candidates:
        match = children.get(candidate.casefold())
        if match is not None:
            return match
    return None


def resolve_flame_dirs(dataset_root: str) -> Tuple[str, str]:
    root = Path(dataset_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    base_candidates = [root]
    split_pairs = [
        ("Training", "Test"),
        ("Train", "Test"),
        ("Training", "Testing"),
        ("Train", "Testing"),
        ("Training and Validation", "Test"),
        ("Training and Validation", "Testing"),
    ]

    for base in base_candidates:
        if not base.exists():
            continue
        for train_name, test_name in split_pairs:
            train_dir = base / train_name
            test_dir = base / test_name
            if train_dir.exists() and test_dir.exists():
                return str(train_dir), str(test_dir)

    raise FileNotFoundError(
        f"Could not find FLAME-style Train/Test folders under: {dataset_root}\n"
        f"Expected one of: Training/Test, Train/Test, Training/Testing, Train/Testing, Training and Validation/Test"
    )


def save_binary_training_curves(history: dict, save_path: str, title: str) -> None:
    if not history:
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if "loss" in history:
        axes[0].plot(history.get("loss", []), label="Train Loss", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(history.get("val_loss", []), label="Val Loss", linewidth=2, linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "accuracy" in history:
        axes[1].plot(history.get("accuracy", []), label="Train Accuracy", linewidth=2)
    if "val_accuracy" in history:
        axes[1].plot(history.get("val_accuracy", []), label="Val Accuracy", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_binary_confusion_matrix(labels, preds, save_path: str, title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["No Fire", "Fire"],
        yticklabels=["No Fire", "Fire"],
        annot_kws={"size": 14},
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_results_summary(save_path: str, title: str, metrics: dict, extra_lines: Optional[Sequence[str]] = None) -> None:
    lines = [
        "=" * 60,
        f"  {title}",
        "=" * 60,
        f"  Accuracy    : {metrics.get('accuracy'):.4f}" if metrics.get("accuracy") is not None else "  Accuracy    : N/A",
        f"  Precision   : {metrics.get('precision'):.4f}" if metrics.get("precision") is not None else "  Precision   : N/A",
        f"  Recall      : {metrics.get('recall'):.4f}" if metrics.get("recall") is not None else "  Recall      : N/A",
        f"  F1 Score    : {metrics.get('f1'):.4f}" if metrics.get("f1") is not None else "  F1 Score    : N/A",
        f"  AUC-ROC     : {metrics.get('auc'):.4f}" if metrics.get("auc") is not None else "  AUC-ROC     : N/A",
        f"  AUC-PR      : {metrics.get('aupr'):.4f}" if metrics.get("aupr") is not None else "  AUC-PR      : N/A",
    ]
    if extra_lines:
        lines.extend([""] + list(extra_lines))
    lines.append("=" * 60)

    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
