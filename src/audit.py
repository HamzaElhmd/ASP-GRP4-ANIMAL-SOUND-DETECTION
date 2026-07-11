import os
from pathlib import Path
from typing import Dict
import torchaudio
import torch
import pandas as pd


DATASET_PATH = Path('dataset')


def get_dataset() -> Dict[str, int]:
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

"""
def dataset_statistics(dt: pd.DataFrame) -> pd.DataFrame:
    pass


def standardize_format(audio: torch.tensor) -> torch.tensor:
    pass


def frame_audio(audio: torch.tensor) -> torch.tensor:
    pass
"""


if __name__ == '__main__':
    dataset_dict = get_dataset()
    print(dataset_dict)
