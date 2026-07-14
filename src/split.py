from pathlib import Path

import pandas as pd

try:
    from preprocessor import PREPROCESSED
except ModuleNotFoundError:
    from src.preprocessor import PREPROCESSED

PROCESSED_CSV = PREPROCESSED / "farmyard.csv"
SPLIT_CSV = PREPROCESSED / "split.csv"

# Train/val/test, not train/val. A 2-way split meant the val set was being
# used for two different jobs -- tuning (threshold calibration) and final
# reporting -- which is a methodological problem: metrics measured on data
# you already used to make decisions are optimistic. Matching the
# hidden-markov-model branch's 70/15/15 ratios for consistency of practice
# across tracks (the actual clip assignments differ, this isn't the same
# split file, just the same split *design*).
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 42


def build_split(
    csv_path: Path = PROCESSED_CSV,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Assigns train/val/test at the source-clip level, not the frame level.

    processed/farmyard.csv has multiple rows (training frames, augmented
    frames) per original raw clip (source_file). Splitting by row would let
    augmented copies of the same clip leak across splits, so the split is
    decided once per (label, source_file) and then broadcast to every row
    derived from that clip.

    val is for tuning (threshold calibration, early stopping decisions).
    test is untouched until final reporting -- never used to pick a
    threshold, a checkpoint, or a hyperparameter.
    """
    df = pd.read_csv(csv_path)
    source_clips = df[["label", "source_file"]].drop_duplicates().reset_index(drop=True)

    assignments = []
    for label, group in source_clips.groupby("label"):
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_train = max(1, round(n * train_fraction))
        n_val = max(1, round(n * val_fraction))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train_files = set(shuffled.iloc[:n_train]["source_file"])
        val_files = set(shuffled.iloc[n_train : n_train + n_val]["source_file"])
        for _, row in shuffled.iterrows():
            if row["source_file"] in train_files:
                split = "train"
            elif row["source_file"] in val_files:
                split = "val"
            else:
                split = "test"
            assignments.append({"label": label, "source_file": row["source_file"], "split": split})

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
