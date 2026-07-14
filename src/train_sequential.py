import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

try:
    from eda import N_MELS
    from sequential_data import ANIMAL_CLASSES, build_dataset
    from sequential_model import SequentialEventDetector
except ModuleNotFoundError:
    from src.eda import N_MELS
    from src.sequential_data import ANIMAL_CLASSES, build_dataset
    from src.sequential_model import SequentialEventDetector

OUTPUT_DIR = Path("runs/sequential")
DEVICE = torch.device("cpu")


def frame_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    return (preds == targets).float().mean().item()


def compute_class_pos_weight(train_ds) -> torch.Tensor:
    """pos_weight per class for BCEWithLogitsLoss: (n_negative / n_positive)
    counted directly from the train split. Classes with fewer positive
    examples (rooster, sheep, cow) get a higher weight, so a missed rooster
    costs the loss more than a missed cat -- directly counters the ~4x
    cat-vs-sheep imbalance. Tried and found to help some classes (sheep)
    but not clearly beat the plain baseline within 6 epochs on others --
    see runs/sequential/SUMMARY.md. Kept as an option, not the default,
    since it isn't a proven win yet.
    """
    counts = torch.zeros(len(ANIMAL_CLASSES))
    for label in train_ds.rows["label"]:
        if label in ANIMAL_CLASSES:
            counts[ANIMAL_CLASSES.index(label)] += 1
    total = len(train_ds)
    return (total - counts) / counts


def build_balanced_sampler(train_ds) -> WeightedRandomSampler:
    """Per-sample weight = 1 / (count of that sample's class), so each class
    gets roughly equal expected exposure per epoch. Same caveat as
    compute_class_pos_weight -- combining both at full strength overcorrected
    (see SUMMARY.md), use at most one at a time."""
    labels = train_ds.rows["label"].tolist()
    class_counts = train_ds.rows["label"].value_counts().to_dict()
    weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def overfit_sanity_check(n_clips: int = 16, epochs: int = 150, lr: float = 1e-2) -> Tuple[SequentialEventDetector, List[float], List[float]]:
    """Trains on a tiny fixed subset until it can near-perfectly memorize it.
    Confirms the pipeline (label alignment, loss, gradients) is correct
    before trusting a full run -- per the spec's overfitting sanity check."""
    train_ds = build_dataset("train")
    subset = Subset(train_ds, list(range(n_clips)))
    loader = DataLoader(subset, batch_size=n_clips, shuffle=True)

    model = SequentialEventDetector(input_size=N_MELS, hidden_size=64, num_layers=1).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses, accuracies = [], []
    for epoch in range(epochs):
        for features, targets in loader:
            optimizer.zero_grad()
            logits = model(features.to(DEVICE))
            loss = criterion(logits, targets.to(DEVICE))
            loss.backward()
            optimizer.step()
        acc = frame_accuracy(logits, targets.to(DEVICE))
        losses.append(loss.item())
        accuracies.append(acc)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"[overfit check] epoch {epoch:3d}: loss={loss.item():.4f} frame_acc={acc:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(losses)
    axes[0].set_title(f"Overfit check: loss ({n_clips} clips)")
    axes[0].set_xlabel("epoch")
    axes[1].plot(accuracies)
    axes[1].set_title("Overfit check: frame accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overfit_check.png")
    plt.close(fig)

    return model, losses, accuracies


def full_training_run(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 1,
    num_workers: int = 4,
    use_pos_weight: bool = False,
    use_balanced_sampler: bool = False,
) -> Tuple[SequentialEventDetector, List[float], List[float]]:
    train_ds = build_dataset("train")
    val_ds = build_dataset("val")

    sampler = build_balanced_sampler(train_ds) if use_balanced_sampler else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SequentialEventDetector(input_size=N_MELS, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
    pos_weight = compute_class_pos_weight(train_ds) if use_pos_weight else None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        start = time.time()
        model.train()
        running_loss, n_batches = 0.0, 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            logits = model(features.to(DEVICE))
            loss = criterion(logits, targets.to(DEVICE))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        avg_train_loss = running_loss / n_batches
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss, n_val_batches = 0.0, 0
        with torch.no_grad():
            for features, targets in val_loader:
                logits = model(features.to(DEVICE))
                loss = criterion(logits, targets.to(DEVICE))
                running_val_loss += loss.item()
                n_val_batches += 1
        avg_val_loss = running_val_loss / n_val_batches
        val_losses.append(avg_val_loss)

        elapsed = time.time() - start
        print(f"[full run] epoch {epoch + 1}/{epochs}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} ({elapsed:.1f}s)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_losses, label="train loss")
    ax.plot(val_losses, label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Sequential model (GRU) training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curve.png")
    plt.close(fig)

    torch.save(model.state_dict(), OUTPUT_DIR / "sequential_model.pt")

    return model, train_losses, val_losses


if __name__ == "__main__":
    print("=== Overfit sanity check ===")
    overfit_sanity_check()

    print("\n=== Full training run ===")
    full_training_run()
