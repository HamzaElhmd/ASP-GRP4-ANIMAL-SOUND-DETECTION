from pathlib import Path
from typing import Dict

import pandas as pd

try:
    from preprocessor import PREPROCESSED
except ModuleNotFoundError:
    from src.preprocessor import PREPROCESSED

PROCESSED_CSV = PREPROCESSED / "farmyard.csv"
SPLIT_CSV = PREPROCESSED / "split.csv"

VAL_FRACTION = 0.2
RANDOM_SEED = 42


def build_split(
    csv_path: Path = PROCESSED_CSV,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Assigns train/val at the source-clip level, not the frame level.

    processed/farmyard.csv has multiple rows (training frames, augmented
    frames) per original raw clip (source_file). Splitting by row would let
    augmented copies of the same clip leak across train and val, so the
    split is decided once per (label, source_file) and then broadcast to
    every row derived from that clip.
    """
    df = pd.read_csv(csv_path)
    source_clips = df[["label", "source_file"]].drop_duplicates().reset_index(drop=True)

    assignments = []
    for label, group in source_clips.groupby("label"):
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val_files = set(shuffled.iloc[:n_val]["source_file"])
        for _, row in shuffled.iterrows():
            assignments.append({
                "label": label,
                "source_file": row["source_file"],
                "split": "val" if row["source_file"] in val_files else "train",
            })

    return pd.DataFrame(assignments)


def load_split(csv_path: Path = SPLIT_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def apply_split(frames_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    """Joins the source-clip split assignment onto a frame-level dataframe
    (e.g. the output of load_training_frames() in src/eda.py)."""
    return frames_df.merge(split_df[["source_file", "split"]], on="source_file", how="left")


def print_split_summary(frames_with_split: pd.DataFrame) -> None:
    summary = frames_with_split.groupby(["label", "split"]).size().unstack(fill_value=0)
    print(summary)
    print()
    print("row counts:", frames_with_split["split"].value_counts().to_dict())


if __name__ == "__main__":
    split_df = build_split()
    split_df.to_csv(SPLIT_CSV, index=False)
    print(f"saved {SPLIT_CSV} ({len(split_df)} source clips)")

    print("\nsource-clip counts per label/split:")
    print(split_df.groupby(["label", "split"]).size().unstack(fill_value=0))

    full_df = pd.read_csv(PROCESSED_CSV)
    full_with_split = apply_split(full_df, split_df)
    print("\nframe-row counts per label/split (after augmentation):")
    print_split_summary(full_with_split)
