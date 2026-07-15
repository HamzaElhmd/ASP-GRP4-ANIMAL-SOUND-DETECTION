"""Frozen YAMNet feature extraction. Not fine-tuned -- see the reasoning in
runs/yamnet/SUMMARY.md: the dataset is too small to fine-tune the whole
network safely, there's no GPU here, and frozen features keep the
classifier head cheap enough to actually experiment with class-imbalance
fixes on, the same way the sequential track did.
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio

TARGET_SR = 16000
EMBEDDING_DIM = 1024

_yamnet_model = None


def get_yamnet_model():
    """Loads YAMNet once and reuses it -- loading from TF Hub is slow, and
    nothing about the model changes between calls since it's frozen."""
    global _yamnet_model
    if _yamnet_model is None:
        import tensorflow_hub as hub
        _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    return _yamnet_model


def extract_embeddings(wav_path: str) -> torch.Tensor:
    """Runs frozen YAMNet on a wav file (mono, 16kHz -- same contract as
    the rest of this project) and returns its native per-frame embeddings,
    shape (n_frames, 1024). YAMNet uses its own internal windowing
    (~0.96s window / 0.48s hop, not this project's usual hop) -- coarser
    temporal resolution than the sequential track, but well within the
    project's +/-500ms tolerance requirement.
    """
    model = get_yamnet_model()
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != TARGET_SR:
        raise ValueError(f"expected {TARGET_SR}Hz, got {sr}Hz -- resample before calling this")
    if waveform.shape[0] != 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    wav_np = waveform.squeeze(0).numpy().astype(np.float32)
    _, embeddings, _ = model(wav_np)
    return torch.from_numpy(embeddings.numpy())
