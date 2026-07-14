from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

try:
    from eda import HOP_LENGTH, N_FFT, N_MELS, PROCESSED_CSV, WIN_LENGTH
    from split import SPLIT_CSV, apply_split, load_split
except ModuleNotFoundError:
    from src.eda import HOP_LENGTH, N_FFT, N_MELS, PROCESSED_CSV, WIN_LENGTH
    from src.split import SPLIT_CSV, apply_split, load_split

# Background has no dedicated output -- its target is all-zero across these 5,
# same as silence. Order is fixed since it defines the output vector's index.
ANIMAL_CLASSES = ["cat", "cow", "dog", "rooster", "sheep"]


def compute_feature_sequence_from_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Mel-spectrogram (dB) for an in-memory waveform, transposed to
    (time, n_mels) for RNN input. Shared by both the file-based path
    (compute_feature_sequence) and continuous/windowed inference
    (predict_continuous.py), so the exact same transform is used everywhere
    the model is called -- training, eval, and real-recording inference.
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
    )
    to_db = torchaudio.transforms.AmplitudeToDB()
    mel_db = to_db(mel_transform(waveform)).squeeze(0)  # (n_mels, time)
    return mel_db.transpose(0, 1)  # (time, n_mels)


def compute_feature_sequence(filepath: str, sample_rate: int) -> torch.Tensor:
    """Mel-spectrogram (dB), transposed to (time, n_mels) for RNN input.

    Same N_FFT/HOP_LENGTH/WIN_LENGTH/N_MELS as the EDA pass (src/eda.py) --
    input parameters carried forward from feature selection, not re-chosen.
    """
    waveform, sr = torchaudio.load(str(filepath))
    return compute_feature_sequence_from_waveform(waveform, sr)


def label_to_target(label: str) -> torch.Tensor:
    target = torch.zeros(len(ANIMAL_CLASSES))
    if label in ANIMAL_CLASSES:
        target[ANIMAL_CLASSES.index(label)] = 1.0
    return target


class SequentialFrameDataset(Dataset):
    def __init__(self, frames_df: pd.DataFrame):
        self.rows = frames_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[idx]
        features = compute_feature_sequence(row["filepath"], int(row["samplerate"]))
        target = label_to_target(row["label"]).unsqueeze(0).repeat(features.shape[0], 1)
        return features, target


def build_dataset(split_name: str, csv_path: Path = PROCESSED_CSV, split_csv: Path = SPLIT_CSV) -> SequentialFrameDataset:
    if split_name not in ("train", "val"):
        raise ValueError(f"split_name must be 'train' or 'val', got {split_name!r}")

    all_frames = pd.read_csv(csv_path)
    split_df = load_split(split_csv)
    frames_with_split = apply_split(all_frames, split_df)

    if frames_with_split["split"].isna().any():
        missing = frames_with_split[frames_with_split["split"].isna()]["source_file"].unique()
        raise RuntimeError(f"{len(missing)} source_file(s) in {csv_path} have no split assignment: {missing[:5]}")

    subset = frames_with_split[frames_with_split["split"] == split_name]
    return SequentialFrameDataset(subset)


if __name__ == "__main__":
    train_ds = build_dataset("train")
    val_ds = build_dataset("val")
    print(f"train: {len(train_ds)} frames, val: {len(val_ds)} frames")

    features, target = train_ds[0]
    print(f"sample feature shape: {tuple(features.shape)} (time, n_mels)")
    print(f"sample target shape: {tuple(target.shape)} (time, n_animal_classes)")
    print(f"animal classes (target column order): {ANIMAL_CLASSES}")

    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    batch_features, batch_targets = next(iter(loader))
    print(f"batch feature shape: {tuple(batch_features.shape)}")
    print(f"batch target shape: {tuple(batch_targets.shape)}")
