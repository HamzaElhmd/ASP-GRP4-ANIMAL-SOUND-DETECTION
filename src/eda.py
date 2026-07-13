from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio

try:
    from preprocessor import PREPROCESSED
except ModuleNotFoundError:
    from src.preprocessor import PREPROCESSED

PROCESSED_CSV = PREPROCESSED / "farmyard.csv"
EDA_OUTPUT_DIR = Path("eda_outputs")

CLASSES = ["cat", "cow", "dog", "rooster", "sheep", "background"]
FRAMES_PER_CLASS = 4
RANDOM_SEED = 42

N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 64
N_MFCC = 13


def load_training_frames(csv_path: Path = PROCESSED_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    training_frames = df[df["segment_type"] == "training_frame"].reset_index(drop=True)
    return training_frames


def sample_frames_per_class(
    df: pd.DataFrame,
    n_per_class: int = FRAMES_PER_CLASS,
    seed: int = RANDOM_SEED,
) -> Dict[str, List[str]]:
    sampled: Dict[str, List[str]] = {}
    for label in CLASSES:
        class_rows = df[df["label"] == label]
        n = min(n_per_class, len(class_rows))
        chosen = class_rows.sample(n=n, random_state=seed)
        sampled[label] = chosen["filepath"].tolist()
    return sampled


def _windowed_rms(waveform: torch.Tensor, win_length: int, hop_length: int) -> torch.Tensor:
    frames = waveform.unfold(dimension=1, size=win_length, step=hop_length)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=-1))
    return rms.squeeze(0)


def _windowed_zcr(waveform: torch.Tensor, win_length: int, hop_length: int) -> torch.Tensor:
    frames = waveform.unfold(dimension=1, size=win_length, step=hop_length)
    signs = torch.sign(frames)
    sign_changes = torch.abs(signs[..., 1:] - signs[..., :-1]) > 0
    zcr = sign_changes.float().mean(dim=-1)
    return zcr.squeeze(0)


def compute_features(waveform: torch.Tensor, sample_rate: int) -> Dict[str, torch.Tensor]:
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, power=2.0
    )
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
    )
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
            "n_mels": N_MELS,
        },
    )
    to_db = torchaudio.transforms.AmplitudeToDB()

    stft = to_db(spectrogram_transform(waveform))
    mel = to_db(mel_transform(waveform))
    mfcc = mfcc_transform(waveform)
    mfcc_delta = torchaudio.functional.compute_deltas(mfcc)

    rms = _windowed_rms(waveform, WIN_LENGTH, HOP_LENGTH)
    zcr = _windowed_zcr(waveform, WIN_LENGTH, HOP_LENGTH)

    return {
        "waveform": waveform.squeeze(0),
        "stft": stft.squeeze(0),
        "mel": mel.squeeze(0),
        "mfcc": mfcc.squeeze(0),
        "mfcc_delta": mfcc_delta.squeeze(0),
        "rms": rms,
        "zcr": zcr,
    }


def plot_class_features(label: str, filepath: str, features: Dict[str, torch.Tensor], sample_rate: int, out_dir: Path) -> Path:
    fig, axes = plt.subplots(6, 1, figsize=(10, 16))
    fig.suptitle(f"{label}: {Path(filepath).name}")

    axes[0].plot(features["waveform"].numpy())
    axes[0].set_title("Waveform")

    axes[1].imshow(features["stft"].numpy(), origin="lower", aspect="auto")
    axes[1].set_title("STFT spectrogram (dB)")

    axes[2].imshow(features["mel"].numpy(), origin="lower", aspect="auto")
    axes[2].set_title("Mel spectrogram (dB)")

    axes[3].imshow(features["mfcc"].numpy(), origin="lower", aspect="auto")
    axes[3].set_title("MFCC")

    axes[4].imshow(features["mfcc_delta"].numpy(), origin="lower", aspect="auto")
    axes[4].set_title("MFCC delta")

    ax_rms = axes[5]
    ax_rms.plot(features["rms"].numpy(), label="RMS energy")
    ax_zcr = ax_rms.twinx()
    ax_zcr.plot(features["zcr"].numpy(), color="orange", label="ZCR")
    ax_rms.set_title("RMS energy and zero-crossing rate")
    ax_rms.legend(loc="upper left")
    ax_zcr.legend(loc="upper right")

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}_{Path(filepath).stem}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_eda(csv_path: Path = PROCESSED_CSV, out_dir: Path = EDA_OUTPUT_DIR) -> None:
    training_frames = load_training_frames(csv_path)
    sampled = sample_frames_per_class(training_frames)

    for label, filepaths in sampled.items():
        for filepath in filepaths:
            waveform, sample_rate = torchaudio.load(filepath)
            features = compute_features(waveform, sample_rate)
            out_path = plot_class_features(label, filepath, features, sample_rate, out_dir)
            print(f"saved {out_path}")


if __name__ == "__main__":
    run_eda()
