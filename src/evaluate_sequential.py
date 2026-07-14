from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

try:
    from eda import N_MELS
    from sequential_data import ANIMAL_CLASSES, build_dataset
    from sequential_model import SequentialEventDetector
    from train_sequential import DEVICE, OUTPUT_DIR
except ModuleNotFoundError:
    from src.eda import N_MELS
    from src.sequential_data import ANIMAL_CLASSES, build_dataset
    from src.sequential_model import SequentialEventDetector
    from src.train_sequential import DEVICE, OUTPUT_DIR

CHECKPOINT = OUTPUT_DIR / "sequential_model.pt"
QUALITATIVE_DIR = OUTPUT_DIR / "qualitative"


def load_trained_model(checkpoint: Path = CHECKPOINT) -> SequentialEventDetector:
    model = SequentialEventDetector(input_size=N_MELS, hidden_size=64, num_layers=1).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    return model


def run_inference(model: SequentialEventDetector, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    all_logits, all_targets = [], []
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(DEVICE))
            all_logits.append(logits)
            all_targets.append(targets)
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def evaluate_frame_level(model: SequentialEventDetector, threshold: float = 0.5) -> None:
    val_ds = build_dataset("val")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    logits, targets = run_inference(model, val_loader)
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
        f.write(f"threshold: {threshold}\n\n")
        f.write(f"{'class':10s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>10s}\n")
        for i, cls in enumerate(ANIMAL_CLASSES):
            f.write(f"{cls:10s} {precision[i]:10.3f} {recall[i]:10.3f} {f1[i]:10.3f} {int(support[i]):10d}\n")
        f.write(f"{'macro avg':10s} {macro_p:10.3f} {macro_r:10.3f} {macro_f1:10.3f} {'':>10s}\n")


def plot_qualitative_examples(model: SequentialEventDetector, n_examples: int = 4) -> None:
    val_ds = build_dataset("val")
    rows = val_ds.rows

    seen_labels = set()
    chosen_indices = []
    for label in ANIMAL_CLASSES + ["background"]:
        matches = rows[rows["label"] == label]
        if len(matches) == 0:
            continue
        chosen_indices.append(matches.index[0])
        seen_labels.add(label)
        if len(chosen_indices) >= n_examples:
            break

    QUALITATIVE_DIR.mkdir(parents=True, exist_ok=True)

    for idx in chosen_indices:
        row = rows.iloc[idx]
        features, target = val_ds[idx]
        with torch.no_grad():
            logits = model(features.unsqueeze(0).to(DEVICE))
            probs = torch.sigmoid(logits).squeeze(0).numpy()

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        axes[0].imshow(features.transpose(0, 1).numpy(), origin="lower", aspect="auto")
        axes[0].set_title(f"Mel spectrogram -- true label: {row['label']}")

        for c, cls in enumerate(ANIMAL_CLASSES):
            axes[1].plot(probs[:, c], label=cls)
        axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="threshold")
        axes[1].set_title("Predicted sigmoid probability per class over time")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        out_path = QUALITATIVE_DIR / f"{row['label']}_{Path(row['filepath']).stem}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == "__main__":
    model = load_trained_model()
    print("=== Frame-level precision/recall/F1 (val set) ===")
    evaluate_frame_level(model)

    print("\n=== Qualitative examples ===")
    plot_qualitative_examples(model)
