import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

try:
    from yamnet_data import build_dataset, load_embedding_cache
    from yamnet_model import YamnetClassifierHead
except ModuleNotFoundError:
    from src.yamnet_data import build_dataset, load_embedding_cache
    from src.yamnet_model import YamnetClassifierHead

OUTPUT_DIR = Path("runs/yamnet")
DEVICE = torch.device("cpu")


def frame_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    return (preds == targets).float().mean().item()


def overfit_sanity_check(embedding_cache, n_clips: int = 16, epochs: int = 150, lr: float = 1e-2):
    """Trains on a tiny fixed subset until it can near-perfectly memorize
    it -- confirms the pipeline (label alignment, loss, gradients) is
    correct before trusting a full run, same check as every other track."""
    train_ds = build_dataset("train", embedding_cache)
    subset = Subset(train_ds, list(range(n_clips)))
    loader = DataLoader(subset, batch_size=n_clips, shuffle=True)

    model = YamnetClassifierHead().to(DEVICE)
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
    embedding_cache,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_size: int = 128,
    num_workers: int = 0,
) -> Tuple[YamnetClassifierHead, List[float], List[float]]:
    train_ds = build_dataset("train", embedding_cache)
    val_ds = build_dataset("val", embedding_cache)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = YamnetClassifierHead(hidden_size=hidden_size).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()
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
    ax.set_title("YAMNet classifier head training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curve.png")
    plt.close(fig)

    torch.save(model.state_dict(), OUTPUT_DIR / "yamnet_head.pt")

    return model, train_losses, val_losses


if __name__ == "__main__":
    print("=== Loading embedding cache ===")
    cache = load_embedding_cache()

    print("\n=== Overfit sanity check ===")
    overfit_sanity_check(cache)

    print("\n=== Full training run ===")
    full_training_run(cache)
