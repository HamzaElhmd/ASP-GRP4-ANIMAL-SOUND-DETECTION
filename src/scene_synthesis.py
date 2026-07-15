from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio

try:
    from preprocessor import TARGET_SR
    from postprocessing import Event, PostProcessingConfig, timeline_to_events
    from evaluation import event_based_metrics, segment_based_metrics, near_miss_breakdown
except ModuleNotFoundError:
    from src.preprocessor import TARGET_SR
    from src.postprocessing import Event, PostProcessingConfig, timeline_to_events
    from src.evaluation import event_based_metrics, segment_based_metrics, near_miss_breakdown

ANIMAL_LABELS = ["cat", "cow", "dog", "rooster", "sheep"]
PROCESSED_CSV = Path("processed") / "farmyard.csv"
VAL_FOLD = 5

# A model plugs into the harness through this signature:
#   predict_fn(waveform[1, samples], sample_rate) -> (probs[T, C], frame_times[T], energy_db[T])
PredictFn = Callable[[torch.Tensor, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def _fold_of(source_file: str) -> int:
    token = str(source_file).split("-")[0]
    return int(token) if token.isdigit() else -1


def load_holdout_frames(csv_path: Path = PROCESSED_CSV) -> pd.DataFrame:
    """The fold-5 hold-out clips used to build the evaluation scenes (no model
    ever trains on fold 5, so the event-level benchmark is leakage-free)."""
    df = pd.read_csv(csv_path)
    df = df[df["segment_type"] == "training_frame"].copy()
    df["fold"] = df["source_file"].map(_fold_of)
    return df[df["fold"] == VAL_FOLD].reset_index(drop=True)


def _load(filepath: str) -> np.ndarray:
    waveform, _ = torchaudio.load(filepath)
    audio = waveform.mean(dim=0).numpy().astype(np.float32)
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio


def _trim_active(clip: np.ndarray, top_db: float = 30.0,
                 frame: int = 400, hop: int = 160) -> np.ndarray:
    """Trim leading/trailing silence: keep the span from the first to the last
    frame whose energy is within `top_db` of the clip's peak frame. The processed
    2 s frames carry silent padding; without trimming, the ground-truth event
    would include that silence and a correct detection of the actual vocalisation
    would be scored 'mistimed' against the +/- 500 ms collar."""
    n = len(clip)
    if n < frame:
        return clip
    n_frames = 1 + (n - frame) // hop
    starts = np.arange(n_frames) * hop
    rms = np.sqrt(np.maximum(
        np.array([np.mean(clip[s:s + frame] ** 2) for s in starts]), 1e-12))
    db = 20.0 * np.log10(rms + 1e-9)
    active = np.flatnonzero(db >= db.max() - top_db)
    if active.size == 0:
        return clip
    start = int(starts[active[0]])
    end = min(n, int(starts[active[-1]]) + frame)
    trimmed = clip[start:end]
    return trimmed if trimmed.size > 0 else clip


def _merge_same_label(events: List[Event], gap: float = 0.5) -> List[Event]:
    """Two same-class clips that overlap or nearly touch are one continuous event
    by definition, so we merge them in the ground truth for a fair, definition-
    correct benchmark."""
    merged: List[Event] = []
    for event in sorted(events, key=lambda e: (e.label, e.onset)):
        if merged and merged[-1].label == event.label and event.onset - merged[-1].offset <= gap:
            merged[-1] = Event(merged[-1].onset, max(merged[-1].offset, event.offset), event.label)
        else:
            merged.append(Event(event.onset, event.offset, event.label))
    merged.sort(key=lambda e: e.onset)
    return merged


def synthesize_scene(
    frames: pd.DataFrame,
    duration_seconds: float = 60.0,
    n_events: int = 14,
    max_overlap: int = 3,
    seed: int = 0,
) -> Tuple[torch.Tensor, List[Event]]:
    """Build a continuous multi-event recording with strong (onset/offset) labels,
    a stand-in for the unseen 1-minute presentation file. Animal clips are placed
    at random onsets on a low background bed; overlap is allowed up to max_overlap
    (that is the polyphonic case the project is about). Returns the waveform and
    the ground-truth events."""
    rng = np.random.default_rng(seed)
    total = int(duration_seconds * TARGET_SR)
    mix = np.zeros(total, dtype=np.float32)

    background = frames[frames["label"] == "background"]
    if len(background):
        position = 0
        while position < total:
            clip = _load(background.iloc[int(rng.integers(0, len(background)))]["filepath"])
            end = min(position + len(clip), total)
            mix[position:end] += clip[:end - position]
            position = end
        peak = np.max(np.abs(mix))
        mix = 0.15 * (mix / peak) if peak > 0 else mix

    animals = frames[frames["label"].isin(ANIMAL_LABELS)]
    events: List[Event] = []
    intervals: List[Tuple[int, int]] = []
    attempts = 0
    while len(events) < n_events and attempts < n_events * 20:
        attempts += 1
        row = animals.iloc[int(rng.integers(0, len(animals)))]
        clip = _trim_active(_load(row["filepath"]))
        if len(clip) < TARGET_SR // 4:                 # nothing audible left
            continue
        clip = clip * float(rng.uniform(0.5, 0.95))
        if len(clip) >= total:
            continue
        start = int(rng.integers(0, total - len(clip)))
        end = start + len(clip)
        overlap = sum(1 for (s, e) in intervals if not (end <= s or start >= e))
        if overlap >= max_overlap:
            continue
        mix[start:end] += clip
        intervals.append((start, end))
        events.append(Event(start / TARGET_SR, end / TARGET_SR, row["label"]))

    mix += rng.normal(0, float(rng.uniform(0.001, 0.01)), total).astype(np.float32)
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = 0.97 * mix / peak
    return torch.tensor(mix).unsqueeze(0), _merge_same_label(events)


def evaluate_on_scenes(
    predict_fn: PredictFn,
    frames: Optional[pd.DataFrame] = None,
    n_scenes: int = 6,
    config: Optional[PostProcessingConfig] = None,
    duration_seconds: float = 60.0,
    seed_start: int = 0,
) -> Dict[str, object]:
    """The shared harness in action, model-agnostic. Synthesize evaluation scenes
    from held-out clips, then run ANY model's `predict_fn` through
    predict -> post-process -> event scoring, and average the event-based /
    segment-based / near-miss metrics across scenes. The Classical ML, CNN, and
    Sequential tracks are all scored identically by passing their own predict_fn."""
    if frames is None:
        frames = load_holdout_frames()
    config = config or PostProcessingConfig()

    event_f1, segment_f1 = [], []
    totals = {"correct": 0, "mistimed": 0, "confused": 0, "missed": 0, "spurious": 0}
    for i in range(n_scenes):
        waveform, reference = synthesize_scene(frames, duration_seconds=duration_seconds, seed=seed_start + i)
        probabilities, times, energy = predict_fn(waveform, TARGET_SR)
        predicted, _ = timeline_to_events(probabilities, times, config, energy_db=energy)
        event_f1.append(event_based_metrics(predicted, reference)["overall_macro"]["f1"])
        segment_f1.append(segment_based_metrics(predicted, reference, duration_seconds)["overall_macro"]["f1"])
        for key, value in near_miss_breakdown(predicted, reference).items():
            totals[key] += value

    return {
        "event_macro_f1": float(np.mean(event_f1)),
        "segment_macro_f1": float(np.mean(segment_f1)),
        "near_miss": totals,
        "n_scenes": n_scenes,
    }


def tune_thresholds(
    predict_fn: PredictFn,
    frames: Optional[pd.DataFrame] = None,
    n_scenes: int = 6,
    base_config: Optional[PostProcessingConfig] = None,
    grid: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    min_dur_grid: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.7),
    seed_start: int = 100,
) -> PostProcessingConfig:
    """Sweep the post-processing parameters on synthesized validation scenes so
    they are chosen on data, not hand-set (the Section 4 requirement that they be
    swept and justified). Two sweeps:

      1. per-class start threshold -- kept, per class, at the value that maximises
         that class's event-based F1 (each class's events depend only on its own
         threshold, so they are tuned independently);
      2. minimum event duration -- kept at the value that maximises macro
         event-F1, which drops short spurious false-positive events.

    Model inference is run once per scene and cached; only the cheap
    post-processing is re-run across the grids. Returns the tuned config.

    `seed_start` defaults to 100 so the tuning scenes are DISJOINT from the
    reporting scenes of `evaluate_on_scenes` (seed_start=0) -- tuning and
    scoring on the same scenes would overfit the thresholds and inflate the
    reported event-F1.
    """
    if frames is None:
        frames = load_holdout_frames()
    base_config = base_config or PostProcessingConfig()

    cached = []
    for i in range(n_scenes):
        waveform, reference = synthesize_scene(frames, seed=seed_start + i)
        probabilities, times, energy = predict_fn(waveform, TARGET_SR)
        cached.append((probabilities, times, energy, reference))

    # 1. per-class start thresholds
    tuned: Dict[str, float] = {}
    for label in ANIMAL_LABELS:
        best_threshold, best_f1 = base_config.threshold_on, -1.0
        for threshold in grid:
            config = copy.deepcopy(base_config)
            config.per_class_threshold_on = {label: threshold}
            scores = []
            for probabilities, times, energy, reference in cached:
                predicted, _ = timeline_to_events(probabilities, times, config, energy_db=energy)
                scores.append(event_based_metrics(predicted, reference)[label]["f1"])
            mean_f1 = float(np.mean(scores))
            if mean_f1 > best_f1:
                best_f1, best_threshold = mean_f1, threshold
        tuned[label] = best_threshold

    result = copy.deepcopy(base_config)
    result.per_class_threshold_on = tuned

    # 2. minimum event duration -- cut short spurious events
    best_min_dur, best_macro = result.min_duration_seconds, -1.0
    for min_dur in min_dur_grid:
        config = copy.deepcopy(result)
        config.min_duration_seconds = min_dur
        scores = []
        for probabilities, times, energy, reference in cached:
            predicted, _ = timeline_to_events(probabilities, times, config, energy_db=energy)
            scores.append(event_based_metrics(predicted, reference)["overall_macro"]["f1"])
        mean_macro = float(np.mean(scores))
        if mean_macro > best_macro:
            best_macro, best_min_dur = mean_macro, min_dur
    result.min_duration_seconds = best_min_dur
    return result
