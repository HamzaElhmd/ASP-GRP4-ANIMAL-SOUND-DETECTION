import torchaudio
import torch
import pandas as pd
from pathlib import Path
from audit import FULL_DATASET
from typing import Tuple


PREPROCESSED = Path('processed')


def preprocess_audio(sample: pd.Series, min_sr: int = 16000) -> Tuple[pd.Series, torch.Tensor]:
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

        if sr > min_sr:
            # Down sample to min_sr
            resampler = torchaudio.transforms.Resample(
                sr,
                min_sr,
                lowpass_filter_width=10
            )
            audio_downsampled = resampler(audio_downmixed)
            peak = torch.max(audio_downsampled)
        else:
            audio_downsampled = audio_downmixed

        # Peak normalization
        peak = torch.max(audio_downsampled)
        audio_normalized = torch.div(audio_downsampled, peak)

        new_audio_file = audio_file.split("/")[2]
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


def preprocess_dataset():
    try:
        dataset_df = pd.read_csv(FULL_DATASET)
        dataset_df_len = len(dataset_df)
        new_dataset_df = pd.DataFrame(columns=dataset_df.columns)

        sr_col = torch.tensor(pd.to_numeric(dataset_df.loc["samplerate"]))
        min_sr = torch.min(sr_col)

        for i in range(dataset_df_len):
            row = dataset_df.iloc[i]
            row_series, audio_preprocessed = preprocess_audio(row, int(min_sr))
            new_dataset_df.iloc[i] = row_series

            torchaudio.save(
                row_series.loc["filepath"],
                audio_preprocessed,
                sample_rate=row_series.loc["samplerate"])

    except Exception as e:
        raise RuntimeError(f"Error: {e}")


if __name__ == '__main__':
    preprocess_dataset()
