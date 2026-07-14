"""Shared testing webapp for the YAMNet (frozen) + classifier head model.

Run with: streamlit run yamnet_webapp.py

Plays one of the YouTube farm-audio chunks and reveals a timestamped
"animal detected" log as playback actually reaches each moment. Same idea
as the version built for the sequential GRU model, adapted for YAMNet's
own native windowing (~0.96s window / 0.48s hop) instead of a separate
scanner. Needs `streamlit` and `streamlit-advanced-audio` (see
requirements.txt) and the embedding cache + trained head from
runs/yamnet/ -- see runs/yamnet/SUMMARY.md for how those were built.
"""

from pathlib import Path

import streamlit as st
import torch
import torchaudio
from streamlit_advanced_audio import audix

from src.evaluate_yamnet import load_trained_model
from src.yamnet_data import ANIMAL_CLASSES
from src.yamnet_predict_continuous import HOP_SECONDS, load_confirmed_labels, predict_continuous

AUDIO_DIR = Path("eda_outputs/multilabel_sources")
CANDIDATE_CSV = AUDIO_DIR / "candidate_events.csv"
RMS_ACTIVITY_THRESHOLD = 0.02
YAMNET_WINDOW_SECONDS = 0.96

# From calibrate_thresholds() on the val set (see runs/yamnet/SUMMARY.md).
# Not yet validated against real audio the way the GRU's thresholds were --
# on the sequential track, val-calibration made cat *worse* on real audio,
# so treat these as a starting point to tune with the sensitivity slider
# below, not a final answer.
CALIBRATED_THRESHOLDS = {"cat": 0.40, "cow": 0.35, "dog": 0.30, "rooster": 0.20, "sheep": 0.35}

st.set_page_config(page_title="YAMNet -- personal test", layout="wide")
st.title("YAMNet (frozen) + classifier head -- personal test against real farm audio")
st.caption("Throwaway tool, not the team's Section 5 webapp -- just for eyeballing this model.")


@st.cache_resource
def get_model():
    return load_trained_model()


@st.cache_data
def get_predictions(wav_path: str):
    model = get_model()
    timestamps, probs = predict_continuous(wav_path, model)
    return timestamps, probs.numpy()


@st.cache_data
def get_activity_mask(wav_path: str, timestamps: tuple):
    """RMS energy per YAMNet frame, computed directly against YAMNet's own
    reported timestamps so it can't drift out of alignment with the model's
    frame count the way a separately-windowed scan could."""
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] != 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    window_samples = int(YAMNET_WINDOW_SECONDS * sr)

    activity = []
    for t in timestamps:
        start = int(t * sr)
        segment = waveform[:, start:start + window_samples]
        if segment.shape[1] == 0:
            activity.append(False)
            continue
        rms = torch.sqrt(torch.mean(segment ** 2)).item()
        activity.append(rms > RMS_ACTIVITY_THRESHOLD)
    return activity


def build_segments(timestamps, scores, activity_mask, sensitivity: float, min_duration: float):
    entries = []
    for t, row, active in zip(timestamps, scores, activity_mask):
        if not active:
            continue
        best_idx = row.argmax()
        best_class = ANIMAL_CLASSES[best_idx]
        class_threshold = CALIBRATED_THRESHOLDS[best_class] * sensitivity
        label = best_class if row[best_idx] >= class_threshold else "unknown"
        entries.append((t, label))

    if not entries:
        return []

    segments = []
    seg_start_t, seg_label = entries[0]
    prev_t = entries[0][0]
    for t, label in entries[1:]:
        gap = t - prev_t
        if label != seg_label or gap > HOP_SECONDS * 1.5:
            segments.append((seg_start_t, prev_t + HOP_SECONDS, seg_label))
            seg_start_t, seg_label = t, label
        prev_t = t
    segments.append((seg_start_t, prev_t + HOP_SECONDS, seg_label))

    return [s for s in segments if s[1] - s[0] >= min_duration]


chunk_files = sorted(AUDIO_DIR.glob("*_part_*.wav"))
chunk_names = [f.name for f in chunk_files]
selected = st.sidebar.selectbox("Audio chunk", chunk_names)
sensitivity = st.sidebar.slider(
    "Sensitivity (1.0 = val-calibrated thresholds; lower = more permissive)", 0.3, 2.0, 1.0, 0.05
)
st.sidebar.caption(f"Val-calibrated thresholds: {CALIBRATED_THRESHOLDS}")
min_duration = st.sidebar.slider("Minimum segment duration (s)", 0.0, 3.0, 0.0, 0.24)
show_confirmed = st.sidebar.checkbox("Also show my manually-confirmed labels (separate, for comparison)", value=False)

wav_path = AUDIO_DIR / selected
playback_state = audix(str(wav_path), key=f"audix_{selected}")

current_time = playback_state.get("currentTime", 0.0) if playback_state else 0.0
is_playing = bool(playback_state and playback_state.get("isPlaying"))

if "played_chunks" not in st.session_state:
    st.session_state.played_chunks = set()
if is_playing:
    st.session_state.played_chunks.add(selected)
has_played = selected in st.session_state.played_chunks

st.subheader("Detected animal log")

if not has_played:
    st.info("Press play and this will fill in as the recording plays -- nothing shown ahead of where you've listened to.")
else:
    timestamps, scores = get_predictions(str(wav_path))
    activity_mask = get_activity_mask(str(wav_path), tuple(timestamps))
    all_segments = build_segments(timestamps, scores, activity_mask, sensitivity, min_duration)

    revealed = [s for s in all_segments if s[0] <= current_time]

    if revealed:
        for start, end, label in revealed:
            st.write(f"**{start:6.2f}s**  →  {label}")
    else:
        st.write("Nothing detected yet at this point in playback.")

    if not is_playing:
        st.caption("Paused -- log stays as of the last playback position.")

if show_confirmed:
    confirmed = load_confirmed_labels(selected, CANDIDATE_CSV)
    st.subheader("Your manually-confirmed labels (from the multi-label EDA -- not model output)")
    if len(confirmed) > 0:
        st.dataframe(confirmed[["start_sec", "end_sec", "animal_labels"]].reset_index(drop=True))
    else:
        st.write("No manually-reviewed spans logged for this chunk.")
