from __future__ import annotations

import torchaudio
import torch
import pandas as pd
import csv
from pathlib import Path
from typing import List, Tuple, Dict

try:
    from audit import FULL_DATASET
except ModuleNotFoundError:
    from src.audit import FULL_DATASET


PREPROCESSED = Path('processed')
TARGET_SR = 16000
TRAIN_WINDOW_SECONDS = 2.0
INFERENCE_WINDOW_SECONDS = 1.0
INFERENCE_HOP_SECONDS = 0.25
DEFAULT_AUGMENTATIONS_PER_CLIP = 2
SILENCE_FLOOR = 1e-8
PROCESSED_CSV_COLUMNS = [
    "filepath",
    "n_samples",
    "channel",
    "samplerate",
    "duration",
    "label",
    "source_file",
    "segment_index",
    "segment_start_sample",
    "segment_end_sample",
    "segment_type",
    "augmentation_index",
    "augmentation_tags",
    "augmentation_group",
]


def preprocess_audio(sample: pd.Series, min_sr: int = TARGET_SR) -> Tuple[pd.Series, torch.Tensor]:
    try:
        channel = sample.loc["channel"]
        audio_file = str(sample.loc["filepath"])
        label = sample.loc["label"]
        audio, sr = torchaudio.load(Path(audio_file))

        # Down mixing to MONO
        if channel == 'stereo':
            audio_downmixed = torch.mean(audio, dim=0, keepdim=True)
            if audio_downmixed.shape[0] != 1:
                raise RuntimeError("Failed to down mix audio.")
        else:
            audio_downmixed = audio

        if sr != min_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=min_sr,
                lowpass_filter_width=10,
            )
            audio_resampled = resampler(audio_downmixed)
        else:
            audio_resampled = audio_downmixed

        # Peak normalization
        peak = torch.max(torch.abs(audio_resampled))
        if peak.item() == 0:
            audio_normalized = audio_resampled
        else:
            audio_normalized = torch.div(audio_resampled, peak)

        new_audio_file = Path(audio_file).name
        audio_dict = {
            "filepath": PREPROCESSED / Path(label) / Path(new_audio_file),
            "n_samples": audio_normalized.shape[1],
            "channel": 'mono',
            "samplerate": min_sr,
            "duration": audio_normalized.shape[1] / min_sr,
            "label": label
        }

        audio_series = pd.Series(audio_dict)
        return audio_series, audio_normalized
    except Exception as e:
        raise RuntimeError(f"Error: {e}")


def _load_mono_resampled_audio(sample: pd.Series, target_sr: int = TARGET_SR) -> torch.Tensor:
    channel = sample.loc["channel"]
    audio_file = str(sample.loc["filepath"])
    audio, sr = torchaudio.load(Path(audio_file))

    if channel == "stereo":
        audio = torch.mean(audio, dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=target_sr,
            lowpass_filter_width=10,
        )
        audio = resampler(audio)

    peak = torch.max(torch.abs(audio))
    if peak.item() > 0:
        audio = torch.div(audio, peak)

    return audio


def frame_training_audio(
    sample: pd.Series,
    frame_seconds: float = TRAIN_WINDOW_SECONDS,
    target_sr: int = TARGET_SR,
) -> List[Tuple[pd.Series, torch.Tensor]]:
    try:
        audio = _load_mono_resampled_audio(sample, target_sr=target_sr)
        label = sample.loc["label"]
        source_path = Path(str(sample.loc["filepath"]))
        frame_samples = int(frame_seconds * target_sr)
        total_samples = audio.shape[1]
        n_frames = max(1, (total_samples + frame_samples - 1) // frame_samples)

        framed_samples: List[Tuple[pd.Series, torch.Tensor]] = []
        for frame_idx in range(n_frames):
            start = frame_idx * frame_samples
            end = start + frame_samples
            frame = audio[:, start:end]
            if frame.shape[1] < frame_samples:
                pad_width = frame_samples - frame.shape[1]
                frame = torch.nn.functional.pad(frame, (0, pad_width))

            frame_dict = {
                "filepath": PREPROCESSED / Path(label) / f"{source_path.stem}_frame_{frame_idx:03d}.wav",
                "n_samples": frame.shape[1],
                "channel": "mono",
                "samplerate": target_sr,
                "duration": frame.shape[1] / target_sr,
                "label": label,
                "source_file": source_path.name,
                "segment_index": frame_idx,
                "segment_start_sample": start,
                "segment_end_sample": min(end, total_samples),
                "segment_type": "training_frame",
            }
            framed_samples.append((pd.Series(frame_dict), frame))

        return framed_samples
    except Exception as e:
        raise RuntimeError(f"Error: {e}")


def _frame_audio_tensor(
    audio: torch.Tensor,
    *,
    frame_seconds: float,
    target_sr: int,
    output_dir: Path,
    file_stem: str,
    label: str,
    segment_type: str,
    source_file: str,
    extra_metadata: Dict[str, object] | None = None,
) -> List[Tuple[pd.Series, torch.Tensor]]:
    frame_samples = int(frame_seconds * target_sr)
    total_samples = audio.shape[1]
    n_frames = max(1, (total_samples + frame_samples - 1) // frame_samples)
    framed_samples: List[Tuple[pd.Series, torch.Tensor]] = []
    for frame_idx in range(n_frames):
        start = frame_idx * frame_samples
        end = start + frame_samples
        frame = audio[:, start:end]
        if frame.shape[1] < frame_samples:
            pad_width = frame_samples - frame.shape[1]
            frame = torch.nn.functional.pad(frame, (0, pad_width))

        output_path = output_dir / f"{file_stem}_frame_{frame_idx:03d}.wav"
        frame_dict = {
            "filepath": output_path,
            "n_samples": frame.shape[1],
            "channel": "mono",
            "samplerate": target_sr,
            "duration": frame.shape[1] / target_sr,
            "label": label,
            "source_file": source_file,
            "segment_index": frame_idx,
            "segment_start_sample": start,
            "segment_end_sample": min(end, total_samples),
            "segment_type": segment_type,
            "augmentation_index": None,
            "augmentation_tags": None,
            "augmentation_group": None,
        }
        if extra_metadata:
            frame_dict.update(extra_metadata)
        framed_samples.append((pd.Series(frame_dict), frame))

    return framed_samples


def inference_time_windowing(
    audio: torch.Tensor,
    sample_rate: int,
    window_seconds: float = INFERENCE_WINDOW_SECONDS,
    hop_seconds: float = INFERENCE_HOP_SECONDS,
    window_function: str = "hann",
) -> torch.Tensor:
    try:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] != 1:
            raise ValueError("Inference windowing expects mono audio.")

        window_samples = max(1, int(window_seconds * sample_rate))
        hop_samples = max(1, int(hop_seconds * sample_rate))
        if audio.shape[1] <= window_samples:
            pad_width = window_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_width))

        if window_function == "hann":
            window = torch.hann_window(window_samples, periodic=False)
        else:
            raise ValueError(f"Unsupported window_function: {window_function}")

        frames = audio.unfold(dimension=1, size=window_samples, step=hop_samples)
        frames = frames.squeeze(0)
        return frames * window
    except Exception as e:
        raise RuntimeError(f"Error: {e}")


def augment_audio(
    sample: pd.Series,
    dataset_df: pd.DataFrame,
    target_sr: int = TARGET_SR,
    augmentations_per_clip: int = DEFAULT_AUGMENTATIONS_PER_CLIP,
) -> List[Tuple[pd.Series, torch.Tensor]]:
    try:
        base_audio = _load_mono_resampled_audio(sample, target_sr=target_sr)
        label = sample.loc["label"]
        source_path = Path(str(sample.loc["filepath"]))
        background_pool = dataset_df[dataset_df["label"] == "background"]

        if background_pool.empty:
            raise RuntimeError("No background clips available for augmentation.")

        augmented: List[Tuple[pd.Series, torch.Tensor]] = []
        for aug_idx in range(augmentations_per_clip):
            augmented_audio = base_audio.clone()
            tags: List[str] = []

            if label != "background":
                bg_row = background_pool.sample(n=1, replace=True).iloc[0]
                bg_audio = _load_mono_resampled_audio(bg_row, target_sr=target_sr)
                if bg_audio.shape[1] < augmented_audio.shape[1]:
                    pad_width = augmented_audio.shape[1] - bg_audio.shape[1]
                    bg_audio = torch.nn.functional.pad(bg_audio, (0, pad_width))
                if bg_audio.shape[1] > augmented_audio.shape[1]:
                    bg_audio = bg_audio[:, : augmented_audio.shape[1]]

                bg_gain = torch.empty(1).uniform_(0.15, 0.65).item()
                augmented_audio = augmented_audio + bg_audio * bg_gain
                tags.append(f"bgmix_{bg_row.name}")

            shift_ratio = torch.empty(1).uniform_(-0.2, 0.2).item()
            shift_samples = int(shift_ratio * augmented_audio.shape[1])
            if shift_samples != 0:
                augmented_audio = torch.roll(augmented_audio, shifts=shift_samples, dims=1)
                tags.append(f"shift_{shift_samples}")

            gain_db = torch.empty(1).uniform_(-6.0, 6.0).item()
            gain = 10 ** (gain_db / 20.0)
            augmented_audio = augmented_audio * gain
            tags.append(f"gain_{gain_db:.2f}db")

            if torch.rand(1).item() < 0.5:
                noise_level = torch.empty(1).uniform_(0.005, 0.03).item()
                noise = torch.randn_like(augmented_audio) * noise_level
                augmented_audio = augmented_audio + noise
                tags.append(f"noise_{noise_level:.4f}")

            peak = torch.max(torch.abs(augmented_audio))
            if peak.item() > 0:
                augmented_audio = augmented_audio / peak

            framed_augmented = _frame_audio_tensor(
                augmented_audio,
                frame_seconds=TRAIN_WINDOW_SECONDS,
                target_sr=target_sr,
                output_dir=PREPROCESSED / Path(label),
                file_stem=f"{source_path.stem}_aug_{aug_idx:03d}",
                label=label,
                segment_type="augmentation_frame",
                source_file=source_path.name,
                extra_metadata={
                    "augmentation_index": aug_idx,
                    "augmentation_tags": "|".join(tags) if tags else "none",
                    "augmentation_group": f"{source_path.stem}_aug_{aug_idx:03d}",
                },
            )
            augmented.extend(framed_augmented)

        return augmented
    except Exception as e:
        raise RuntimeError(f"Error: {e}")


def preprocess_dataset():
    try:
        dataset_df = pd.read_csv(FULL_DATASET)
        PREPROCESSED.mkdir(parents=True, exist_ok=True)
        source_balance = dataset_df["label"].value_counts().sort_index()
        print("Before augmentation balance:")
        print(source_balance.to_string())

        output_csv = PREPROCESSED / "farmyard.csv"
        if output_csv.exists():
            output_csv.unlink()

        with output_csv.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=PROCESSED_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()

            for _, row in dataset_df.iterrows():
                framed_samples = frame_training_audio(row, frame_seconds=TRAIN_WINDOW_SECONDS, target_sr=TARGET_SR)
                augmented_samples = augment_audio(
                    row,
                    dataset_df,
                    target_sr=TARGET_SR,
                    augmentations_per_clip=DEFAULT_AUGMENTATIONS_PER_CLIP,
                )

                for row_series, audio_processed in framed_samples + augmented_samples:
                    output_path = Path(row_series.loc["filepath"])
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    torchaudio.save(
                        str(output_path),
                        audio_processed,
                        sample_rate=int(row_series.loc["samplerate"]))

                    row_dict = row_series.to_dict()
                    row_dict["filepath"] = str(row_dict["filepath"])
                    writer.writerow(row_dict)

        print("After preprocessing balance:")
        processed_df = pd.read_csv(output_csv)
        print(processed_df["label"].value_counts().sort_index().to_string())
        return processed_df

    except Exception as e:
        raise RuntimeError(f"Error: {e}")


if __name__ == '__main__':
    preprocess_dataset()
