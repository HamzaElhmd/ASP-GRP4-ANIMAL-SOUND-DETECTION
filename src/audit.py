import os
from pathlib import Path
from typing import Dict
import torchaudio
import torch
import pandas as pd


DATASET_PATH = Path('dataset')
FULL_DATASET = DATASET_PATH / Path('farmyard.csv')


def dataset_info() -> Dict[str, int]:
    try:
        dataset_dict = {}
        classes = os.listdir(DATASET_PATH)
        for cl in classes:
            path = DATASET_PATH / Path(cl)
            if path.is_dir():
                files = os.listdir(path)
                n_files = len(files)

                total_duration = 0
                for file in files:
                    file_path = DATASET_PATH / Path(cl) / Path(file)
                    pcm, sr = torchaudio.load(str(file_path))
                    duration = pcm.shape[1] / sr
                    total_duration += duration

                dataset_dict[str(path)] = {"num_samples": n_files, "total_duration": total_duration}
        return dataset_dict
    except Exception as e:
        raise RuntimeError(f"error: {e}")


def create_dataset_metadata():
    try:
        dataset = []
        classes = os.listdir(DATASET_PATH)

        for cl in classes:
            path = DATASET_PATH / Path(cl)
            if path.is_dir():
                files = os.listdir(path)

                for file in files:
                    file_path = DATASET_PATH / Path(cl) / Path(file)
                    pcm, sr = torchaudio.load(str(file_path))
                    duration = pcm.shape[1] / sr
                    channel = "mono" if pcm.shape[0] == 1 else "stereo"
                    n_samples = pcm.shape[1]

                    dataset.append({
                        "filepath": file_path,
                        "n_samples": n_samples,
                        "channel": channel,
                        "samplerate": sr,
                        "duration": duration,
                        "label": cl
                    })

        dataset_df = pd.DataFrame(dataset)
        dataset_df.to_csv(FULL_DATASET)
        return dataset_df
    except Exception as e:
        raise RuntimeError(f"error: {e}")

"""
def dataset_statistics(dt: pd.DataFrame) -> pd.DataFrame:
    pass
"""


if __name__ == '__main__':
    print(create_dataset_metadata())
