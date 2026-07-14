from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio

try:
    from eda import N_MELS
    from evaluate_sequential import CHECKPOINT, load_trained_model
    from preprocessor import TARGET_SR, inference_time_windowing
    from sequential_data import ANIMAL_CLASSES, compute_feature_sequence_from_waveform
    from train_sequential import DEVICE, OUTPUT_DIR
except ModuleNotFoundError:
    from src.eda import N_MELS
    from src.evaluate_sequential import CHECKPOINT, load_trained_model
    from src.preprocessor import TARGET_SR, inference_time_windowing
    from src.sequential_data import ANIMAL_CLASSES, compute_feature_sequence_from_waveform
    from src.train_sequential import DEVICE, OUTPUT_DIR

WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.25
CONTINUOUS_DIR = OUTPUT_DIR / "continuous_test"


def predict_continuous(wav_path: str, model: torch.nn.Module) -> Tuple[List[float], torch.Tensor]:
    """Runs the model over an arbitrary-length recording using Person A's
    shared inference-time scanner (1s window / 250ms hop), not the fixed
    2s training-clip format. Each window's per-timestep predictions are
    mean-pooled into a single score per window per class, so the output is
    one probability-per-class value every 250ms across the whole file --
    this is the "windowed scan" convention, not a single whole-file pass.
    """
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != TARGET_SR:
        raise ValueError(f"expected {TARGET_SR}Hz, got {sr}Hz -- resample before calling this")
    if waveform.shape[0] != 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    windows = inference_time_windowing(waveform, sr, WINDOW_SECONDS, HOP_SECONDS)  # (n_windows, window_samples)

    window_scores = []
    model.eval()
    with torch.no_grad():
        for window in windows:
            features = compute_feature_sequence_from_waveform(window.unsqueeze(0), sr)  # (time, n_mels)
            logits = model(features.unsqueeze(0).to(DEVICE))  # (1, time, 5)
            probs = torch.sigmoid(logits).squeeze(0)  # (time, 5)
            window_scores.append(probs.mean(dim=0))  # (5,) -- mean-pool over the window

    scores = torch.stack(window_scores)  # (n_windows, 5)
    timestamps = [i * HOP_SECONDS for i in range(len(windows))]
    return timestamps, scores


def load_confirmed_labels(chunk_name: str, candidate_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(candidate_csv)
    df = df[(df["chunk_file"] == chunk_name) & (df["animal_labels"].notna()) & (df["animal_labels"] != "")]
    return df


def plot_against_confirmed_labels(
    wav_path: str,
    timestamps: List[float],
    scores: torch.Tensor,
    confirmed: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    for c, cls in enumerate(ANIMAL_CLASSES):
        ax.plot(timestamps, scores[:, c].numpy(), label=cls, linewidth=1.2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)

    for _, row in confirmed.iterrows():
        labels = [l for l in row["animal_labels"].split(",") if l in ANIMAL_CLASSES]
        if not labels:
            continue
        ax.axvspan(row["start_sec"], row["end_sec"], color="black", alpha=0.08)
        ax.text(
            row["start_sec"], 1.02, "+".join(labels),
            fontsize=7, rotation=0, va="bottom",
        )

    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("seconds")
    ax.set_ylabel("predicted probability")
    ax.set_title(f"Model predictions vs. manually-confirmed labels: {Path(wav_path).name}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    chunk_name = "4K_RELAXING_FARM_ANIMALS_VIDEO_RELAXING_FARM_ANIMAL_SOUNDS_part_01.wav"
    wav_path = Path("eda_outputs/multilabel_sources") / chunk_name
    candidate_csv = Path("eda_outputs/multilabel_sources/candidate_events.csv")

    model = load_trained_model(CHECKPOINT)
    print(f"running continuous inference on {wav_path} ...")
    timestamps, scores = predict_continuous(str(wav_path), model)
    print(f"{len(timestamps)} windows scored, {timestamps[-1]:.1f}s covered")

    confirmed = load_confirmed_labels(chunk_name, candidate_csv)
    print(f"{len(confirmed)} manually-confirmed spans in this chunk")

    CONTINUOUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONTINUOUS_DIR / f"{Path(chunk_name).stem}_predictions.png"
    plot_against_confirmed_labels(str(wav_path), timestamps, scores, confirmed, out_path)
    print(f"saved {out_path}")
