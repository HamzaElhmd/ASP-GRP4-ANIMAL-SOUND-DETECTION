from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

try:
    from train_yamnet import DEVICE, OUTPUT_DIR
    from yamnet_data import ANIMAL_CLASSES, build_dataset, load_embedding_cache
    from yamnet_model import YamnetClassifierHead
except ModuleNotFoundError:
    from src.train_yamnet import DEVICE, OUTPUT_DIR
    from src.yamnet_data import ANIMAL_CLASSES, build_dataset, load_embedding_cache
    from src.yamnet_model import YamnetClassifierHead

CHECKPOINT = OUTPUT_DIR / "yamnet_head.pt"


def load_trained_model(checkpoint: Path = CHECKPOINT) -> YamnetClassifierHead:
    model = YamnetClassifierHead().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    return model


def run_inference(model: YamnetClassifierHead, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    all_logits, all_targets = [], []
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(DEVICE))
            all_logits.append(logits)
            all_targets.append(targets)
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def evaluate_frame_level(model: YamnetClassifierHead, embedding_cache, threshold: float = 0.5, split_name: str = "test") -> None:
    """Defaults to test, not val -- val is for tuning (threshold
    calibration), test stays untouched until this final report, same
    discipline as the sequential track after its split-leak fix."""
    eval_ds = build_dataset(split_name, embedding_cache)
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, num_workers=0)

    logits, targets = run_inference(model, eval_loader)
    preds = (torch.sigmoid(logits) > threshold).float()

    preds_flat = preds.reshape(-1, preds.shape[-1]).numpy()
    targets_flat = targets.reshape(-1, targets.shape[-1]).numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        targets_flat, preds_flat, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets_flat, preds_flat, average="macro", zero_division=0
    )

    print(f"{'class':10s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>10s}")
    for i, cls in enumerate(ANIMAL_CLASSES):
        print(f"{cls:10s} {precision[i]:10.3f} {recall[i]:10.3f} {f1[i]:10.3f} {int(support[i]):10d}")
    print(f"{'macro avg':10s} {macro_p:10.3f} {macro_r:10.3f} {macro_f1:10.3f} {'':>10s}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "frame_level_metrics.txt", "w") as f:
        f.write(f"split: {split_name}\nthreshold: {threshold}\n\n")
        f.write(f"{'class':10s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>10s}\n")
        for i, cls in enumerate(ANIMAL_CLASSES):
            f.write(f"{cls:10s} {precision[i]:10.3f} {recall[i]:10.3f} {f1[i]:10.3f} {int(support[i]):10d}\n")
        f.write(f"{'macro avg':10s} {macro_p:10.3f} {macro_r:10.3f} {macro_f1:10.3f} {'':>10s}\n")


def calibrate_thresholds(model: YamnetClassifierHead, embedding_cache, thresholds=None) -> dict:
    """Per-class threshold that maximizes that class's F1 on val -- no
    retraining, same approach used on the sequential track."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]

    val_ds = build_dataset("val", embedding_cache)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    logits, targets = run_inference(model, val_loader)
    probs = torch.sigmoid(logits).reshape(-1, logits.shape[-1]).numpy()
    targets_flat = targets.reshape(-1, targets.shape[-1]).numpy()

    best_thresholds = {}
    print(f"{'class':10s} {'best_thresh':>12s} {'f1_at_0.5':>10s} {'f1_calibrated':>14s}")
    for i, cls in enumerate(ANIMAL_CLASSES):
        best_f1, best_t = -1.0, 0.5
        f1_at_half = None
        for t in thresholds:
            preds = (probs[:, i] > t).astype(float)
            _, _, f1, _ = precision_recall_fscore_support(
                targets_flat[:, i], preds, average="binary", zero_division=0
            )
            if abs(t - 0.5) < 1e-9:
                f1_at_half = f1
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[cls] = best_t
        print(f"{cls:10s} {best_t:12.2f} {f1_at_half:10.3f} {best_f1:14.3f}")

    return best_thresholds


if __name__ == "__main__":
    cache = load_embedding_cache()
    model = load_trained_model()
    print("=== Frame-level precision/recall/F1 (test set) ===")
    evaluate_frame_level(model, cache)
    print("\n=== Val-based threshold calibration ===")
    calibrate_thresholds(model, cache)
