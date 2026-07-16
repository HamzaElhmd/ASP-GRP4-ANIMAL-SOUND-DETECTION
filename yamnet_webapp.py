"""Streamlit webapp for YAMNet animal sound detection with file upload."""

import os
import json # NEW: Import json for the download button
from pathlib import Path

import streamlit as st
import torch
import torchaudio
from streamlit_advanced_audio import audix

from src.evaluate_yamnet import load_trained_model
from src.yamnet_data import ANIMAL_CLASSES
from src.yamnet_predict_continuous import HOP_SECONDS, predict_continuous

RMS_ACTIVITY_THRESHOLD = 0.02
YAMNET_WINDOW_SECONDS = 0.96

CALIBRATED_THRESHOLDS = {"cat": 0.40, "cow": 0.35, "dog": 0.30, "rooster": 0.20, "sheep": 0.35}

st.set_page_config(page_title="YAMNet Animal Detector", layout="wide")
st.title("Animal Sound Detection")

@st.cache_resource
def get_model():
    return load_trained_model()

@st.cache_data
def get_predictions(wav_path: str):
    model = get_model()
    waveform, sr = torchaudio.load(wav_path)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        torchaudio.save(wav_path, waveform, 16000)
        
    timestamps, probs = predict_continuous(wav_path, model)
    return timestamps, probs.numpy()

@st.cache_data
def get_activity_mask(wav_path: str, timestamps: tuple):
    waveform, sr = torchaudio.load(wav_path)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000
        
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

    # Return as a list of dicts mapped to the exact JSON schema requested
    results = []
    for s in segments:
        if s[1] - s[0] >= min_duration and s[2] != "unknown":
            results.append({
                "event_start": f"{s[0]:.3f}", 
                "event_end": f"{s[1]:.3f}", 
                "animal": s[2]
            })
    return results

# --- Sidebar UI ---
st.sidebar.header("Detection Settings")
sensitivity = st.sidebar.slider(
    "Sensitivity (1.0 = calibrated; lower = more permissive)", 0.3, 2.0, 1.0, 0.05
)
st.sidebar.caption(f"Calibrated thresholds: {CALIBRATED_THRESHOLDS}")
min_duration = st.sidebar.slider("Minimum segment duration (s)", 0.0, 3.0, 0.0, 0.24)

# --- Main UI ---
uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    temp_wav_path = "temp_upload.wav"
    with open(temp_wav_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    playback_state = audix(temp_wav_path, key="audix_player")
    current_time = playback_state.get("currentTime", 0.0) if playback_state else 0.0
    is_playing = bool(playback_state and playback_state.get("isPlaying"))

    st.divider()
    
    with st.spinner("Analyzing audio..."):
        timestamps, scores = get_predictions(temp_wav_path)
        activity_mask = get_activity_mask(temp_wav_path, tuple(timestamps))
        all_segments = build_segments(timestamps, scores, activity_mask, sensitivity, min_duration)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Live Playback Log")
        if not is_playing and current_time == 0.0:
            st.info("Press play to reveal detections in real-time.")
        
        # Check against float("event_start") since we formatted it as a string earlier
        revealed = [s for s in all_segments if float(s["event_start"]) <= current_time]
        if revealed:
            for seg in revealed:
                st.write(f"🔉 **{float(seg['event_start']):05.2f}s - {float(seg['event_end']):05.2f}s**  →  🐾 **{seg['animal']}**")
        else:
            st.write("Nothing detected yet at this point in playback.")
            
        if not is_playing and current_time > 0.0:
            st.caption("Paused -- log stays as of the last playback position.")

    with col2:
        st.subheader("All Detected Segments")
        if all_segments:
            # Format data for the dataframe
            formatted_segments = [{"Start (s)": s["event_start"], "End (s)": s["event_end"], "Animal": s["animal"].capitalize()} for s in all_segments]
            st.dataframe(formatted_segments, use_container_width=True)
            
            # NEW: Download Button
            json_string = json.dumps(all_segments, indent=2)
            st.download_button(
                label="📥 Download JSON Results",
                data=json_string,
                file_name=f"{Path(uploaded_file.name).stem}_inference.json",
                mime="application/json"
            )
        else:
            st.success("No animals detected in this audio file.")

else:
    st.info("👆 Please upload a .wav file to begin analysis.")
