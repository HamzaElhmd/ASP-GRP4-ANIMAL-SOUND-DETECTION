from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from split import SPLIT_CSV, apply_split, load_split
    from yamnet_features import extract_embeddings
except ModuleNotFoundError:
    from src.split import SPLIT_CSV, apply_split, load_split
    from src.yamnet_features import extract_embeddings

PROCESSED_CSV = Path("processed/farmyard.csv")
EMBEDDING_CACHE = Path("runs/yamnet/embedding_cache.pt")

# Background has no dedicated output -- same convention as the sequential
# track (src/sequential_data.py) -- its target is all-zero across these 5.
ANIMAL_CLASSES = ["cat", "cow", "dog", "rooster", "sheep"]


def label_to_target(label: str) -> torch.Tensor:
    target = torch.zeros(len(ANIMAL_CLASSES))
    if label in ANIMAL_CLASSES:
        target[ANIMAL_CLASSES.index(label)] = 1.0
    return target


def build_embedding_cache(
    csv_path: Path = PROCESSED_CSV,
    cache_path: Path = EMBEDDING_CACHE,
    segment_types=("training_frame",),
) -> Dict[str, torch.Tensor]:
    """Precomputes and caches YAMNet embeddings for every processed clip.
    Frozen features never change between epochs, so there's no reason to
    recompute them every time -- this is what makes frozen-feature training
    fast even without a GPU.

    Defaults to training_frame only, excluding augmentation_frame -- YAMNet
    was pretrained on millions of real-world clips, so it needs less
    augmentation-driven volume to get a working classifier head than
    training a model from scratch does (unlike the sequential track, which
    used the full augmented corpus). Also keeps the one-time embedding
    extraction pass tractable (~586ms/clip on this CPU-only setup).
    Pass segment_types=None to include augmented frames too.
    """
    df = pd.read_csv(csv_path)
    if segment_types is not None:
        df = df[df["segment_type"].isin(segment_types)].reset_index(drop=True)
    cache: Dict[str, torch.Tensor] = {}
    total = len(df)
    for i, row in enumerate(df.itertuples()):
        cache[str(row.filepath)] = extract_embeddings(row.filepath)
        if (i + 1) % 500 == 0 or i == total - 1:
            print(f"  {i + 1}/{total} embeddings cached")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def load_embedding_cache(cache_path: Path = EMBEDDING_CACHE) -> Dict[str, torch.Tensor]:
    return torch.load(cache_path)


class YamnetFrameDataset(Dataset):
    def __init__(self, frames_df: pd.DataFrame, embedding_cache: Dict[str, torch.Tensor]):
        self.rows = frames_df.reset_index(drop=True)
        self.cache = embedding_cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[idx]
        embeddings = self.cache[str(row["filepath"])]
        target = label_to_target(row["label"]).unsqueeze(0).repeat(embeddings.shape[0], 1)
        return embeddings, target


def build_dataset(
    split_name: str,
    embedding_cache: Dict[str, torch.Tensor],
    csv_path: Path = PROCESSED_CSV,
    split_csv: Path = SPLIT_CSV,
) -> YamnetFrameDataset:
    if split_name not in ("train", "val", "test"):
        raise ValueError(f"split_name must be 'train', 'val', or 'test', got {split_name!r}")

    all_frames = pd.read_csv(csv_path)
    split_df = load_split(split_csv)
    frames_with_split = apply_split(all_frames, split_df)

    if frames_with_split["split"].isna().any():
        missing = frames_with_split[frames_with_split["split"].isna()]["source_file"].unique()
        raise RuntimeError(f"{len(missing)} source_file(s) have no split assignment: {missing[:5]}")

    # Must match whatever build_embedding_cache() actually cached, or
    # lookups below raise KeyError -- default cache is training_frame only.
    subset = frames_with_split[
        (frames_with_split["split"] == split_name)
        & (frames_with_split["filepath"].astype(str).isin(embedding_cache.keys()))
    ]
    return YamnetFrameDataset(subset, embedding_cache)
