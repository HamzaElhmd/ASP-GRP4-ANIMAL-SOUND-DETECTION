"""Streamlit webapp for YAMNet animal sound detection with file upload."""

import os
import json
from pathlib import Path

import streamlit as st
import torch
import torchaudio
from streamlit_advanced_audio import audix
import matplotlib.pyplot as plt
import numpy as np

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
    """Builds a list of events from model outputs, supporting overlapping events."""
    all_events = []
    for i, animal in enumerate(ANIMAL_CLASSES):
        class_threshold = CALIBRATED_THRESHOLDS[animal] * sensitivity
        
        in_event = False
        start_time = 0
        
        for t, score, active in zip(timestamps, scores[:, i], activity_mask):
            if not active:
                if in_event:
                    # End of event due to inactivity
                    if t - start_time >= min_duration:
                        all_events.append({
                            "event_start": f"{start_time:.3f}",
                            "event_end": f"{t:.3f}",
                            "animal": animal
                        })
                    in_event = False
                continue

            above_threshold = score >= class_threshold
            
            if above_threshold and not in_event:
                # Start of a new event
                in_event = True
                start_time = t
            elif not above_threshold and in_event:
                # End of an event
                if t - start_time >= min_duration:
                    all_events.append({
                        "event_start": f"{start_time:.3f}",
                        "event_end": f"{t:.3f}",
                        "animal": animal
                    })
                in_event = False
        
        # After loop, handle event that extends to the end
        if in_event:
            end_time = timestamps[-1]
            if end_time - start_time >= min_duration:
                all_events.append({
                    "event_start": f"{start_time:.3f}",
                    "event_end": f"{end_time:.3f}",
                    "animal": animal
                })

    # Sort events by start time for a chronological log
    all_events.sort(key=lambda x: float(x['event_start']))
    
    return all_events

def plot_waveform_with_segments(waveform_data, sr, segments):
    """Plots the waveform and overlays detected event segments."""
    fig, ax = plt.subplots(figsize=(16, 4))
    
    time_axis = np.arange(len(waveform_data)) / sr
    ax.plot(time_axis, waveform_data, color='gray', alpha=0.8, linewidth=0.7)

    class_colors = {
        "cat": "#1f77b4", "cow": "#ff7f0e", "dog": "#2ca02c", 
        "rooster": "#d62728", "sheep": "#9467bd"
    }
    
    for seg in segments:
        start_time = float(seg["event_start"])
        end_time = float(seg["event_end"])
        animal = seg["animal"]
        color = class_colors.get(animal, "k")
        
        ax.axvspan(start_time, end_time, color=color, alpha=0.3)
        
        y_max = ax.get_ylim()[1]
        ax.text(start_time + 0.05, y_max * 0.9, animal, fontsize=9, color=color, weight='bold')

    ax.set_title("Waveform with Detected Segments")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, time_axis[-1])
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    
    return fig

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
            formatted_segments = [{"Start (s)": s["event_start"], "End (s)": s["event_end"], "Animal": s["animal"].capitalize()} for s in all_segments]
            st.dataframe(formatted_segments, use_container_width=True)
            
            json_string = json.dumps(all_segments, indent=2)
            st.download_button(
                label="📥 Download JSON Results",
                data=json_string,
                file_name=f"{Path(uploaded_file.name).stem}_inference.json",
                mime="application/json"
            )

            waveform, sr = torchaudio.load(temp_wav_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            st.subheader("Waveform with Detections")
            fig = plot_waveform_with_segments(waveform.squeeze().numpy(), sr, all_segments)
            st.pyplot(fig)
            
            st.subheader("Play Detected Segments")
            with st.expander("Show/Hide Individual Segment Players"):
                for seg in all_segments:
                    animal = seg['animal']
                    start_s = float(seg['event_start'])
                    end_s = float(seg['event_end'])

                    st.markdown(f"**{animal.capitalize()}** (`{start_s:.2f}s` - `{end_s:.2f}s`)")

                    start_sample = int(start_s * sr)
                    end_sample = int(end_s * sr)
                    audio_segment = waveform[:, start_sample:end_sample]
                    
                    st.audio(audio_segment.numpy(), sample_rate=sr)
        else:
            st.success("No animals detected in this audio file.")

else:
    st.info("👆 Please upload a .wav file to begin analysis.")
