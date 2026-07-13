from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

try:
    from eda import (
        CLASSES,
        PROCESSED_CSV,
        compute_features,
        load_training_frames,
        sample_frames_per_class,
    )
except ModuleNotFoundError:
    from src.eda import (
        CLASSES,
        PROCESSED_CSV,
        compute_features,
        load_training_frames,
        sample_frames_per_class,
    )

OUTPUT_DIR = Path("eda_outputs/step4_conv_filters")

SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])


def _gabor_kernel(size: int = 9, sigma: float = 2.0, wavelength: float = 4.0, theta: float = 0.0) -> torch.Tensor:
    half = size // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )
    x_theta = x * torch.cos(torch.tensor(theta)) + y * torch.sin(torch.tensor(theta))
    y_theta = -x * torch.sin(torch.tensor(theta)) + y * torch.cos(torch.tensor(theta))
    gaussian = torch.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * sigma ** 2))
    carrier = torch.cos(2 * torch.pi * x_theta / wavelength)
    kernel = gaussian * carrier
    return kernel - kernel.mean()


FILTERS = {
    "horizontal_edge": SOBEL_Y,
    "vertical_edge": SOBEL_X,
    "gabor_horizontal": _gabor_kernel(theta=0.0),
}


def apply_filter(mel_db: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    image = mel_db.unsqueeze(0).unsqueeze(0)
    k = kernel.unsqueeze(0).unsqueeze(0)
    padding = k.shape[-1] // 2
    filtered = F.conv2d(image, k, padding=padding)
    return filtered.squeeze(0).squeeze(0)


def plot_filtered(label: str, filepath: str, mel_db: torch.Tensor, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1 + len(FILTERS), 1, figsize=(10, 4 * (1 + len(FILTERS))))
    fig.suptitle(f"{label}: {Path(filepath).name}")

    axes[0].imshow(mel_db.numpy(), origin="lower", aspect="auto")
    axes[0].set_title("Original mel spectrogram (dB)")

    for ax, (name, kernel) in zip(axes[1:], FILTERS.items()):
        filtered = apply_filter(mel_db, kernel).numpy()
        # Zero-padded silence at the tail of short clips creates a sharp
        # discontinuity that dominates the raw min/max range and washes out
        # the real structure, so clip the display range to the 1st-99th
        # percentile instead of letting outliers set the color scale.
        vmin, vmax = np.percentile(filtered, [1, 99])
        ax.imshow(filtered, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"Filtered: {name}")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}_{Path(filepath).stem}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_conv_filter_eda(csv_path: Path = PROCESSED_CSV, out_dir: Path = OUTPUT_DIR) -> None:
    training_frames = load_training_frames(csv_path)
    sampled = sample_frames_per_class(training_frames)

    for label, filepaths in sampled.items():
        for filepath in filepaths:
            waveform, sample_rate = torchaudio.load(str(filepath))
            features = compute_features(waveform, sample_rate)
            out_path = plot_filtered(label, filepath, features["mel"], out_dir)
            print(f"saved {out_path}")


if __name__ == "__main__":
    run_conv_filter_eda()
