"""Runs the frozen YAMNet + classifier head over a real, continuous
recording -- not a pre-cut clip. Unlike the sequential track, no separate
windowing scan is needed: YAMNet already does its own internal ~0.96s
window / 0.48s hop framing for any input length, so this is simpler than
src/predict_continuous.py's equivalent.
"""

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio

try:
    from yamnet_features import TARGET_SR, get_yamnet_model
except ModuleNotFoundError:
    from src.yamnet_features import TARGET_SR, get_yamnet_model

HOP_SECONDS = 0.48  # YAMNet's native patch hop


def predict_continuous(wav_path: str, head_model: torch.nn.Module) -> Tuple[List[float], torch.Tensor]:
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != TARGET_SR:
        raise ValueError(f"expected {TARGET_SR}Hz, got {sr}Hz -- resample before calling this")
    if waveform.shape[0] != 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    model = get_yamnet_model()
    wav_np = waveform.squeeze(0).numpy().astype(np.float32)
    _, embeddings, _ = model(wav_np)
    embeddings = torch.from_numpy(embeddings.numpy())

    head_model.eval()
    with torch.no_grad():
        logits = head_model(embeddings.unsqueeze(0)).squeeze(0)
        probs = torch.sigmoid(logits)

    timestamps = [i * HOP_SECONDS for i in range(embeddings.shape[0])]
    return timestamps, probs


def load_confirmed_labels(chunk_name: str, candidate_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(candidate_csv)
    df = df[(df["chunk_file"] == chunk_name) & (df["animal_labels"].notna()) & (df["animal_labels"] != "")]
    return df


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    try:
        from evaluate_yamnet import load_trained_model
        from yamnet_data import ANIMAL_CLASSES
    except ModuleNotFoundError:
        from src.evaluate_yamnet import load_trained_model
        from src.yamnet_data import ANIMAL_CLASSES

    chunk_name = "4K_RELAXING_FARM_ANIMALS_VIDEO_RELAXING_FARM_ANIMAL_SOUNDS_part_01.wav"
    wav_path = Path("eda_outputs/multilabel_sources") / chunk_name
    candidate_csv = Path("eda_outputs/multilabel_sources/candidate_events.csv")

    head_model = load_trained_model()
    timestamps, probs = predict_continuous(str(wav_path), head_model)
    print(f"{len(timestamps)} frames scored, {timestamps[-1]:.1f}s covered")

    confirmed = load_confirmed_labels(chunk_name, candidate_csv)
    print(f"{len(confirmed)} manually-confirmed spans in this chunk")

    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    for c, cls in enumerate(ANIMAL_CLASSES):
        ax.plot(timestamps, probs[:, c].numpy(), label=cls, linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    for _, row in confirmed.iterrows():
        labels = [l for l in row["animal_labels"].split(",") if l in ANIMAL_CLASSES]
        if not labels:
            continue
        ax.axvspan(row["start_sec"], row["end_sec"], color="black", alpha=0.08)
        ax.text(row["start_sec"], 1.02, "+".join(labels), fontsize=7, va="bottom")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("seconds")
    ax.set_ylabel("predicted probability")
    ax.set_title(f"YAMNet + head: predictions vs. confirmed labels: {chunk_name}")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_dir = Path("runs/yamnet/continuous_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(chunk_name).stem}_predictions.png"
    fig.savefig(out_path)
    print(f"saved {out_path}")
