from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

CLASSES = ["cat", "cow", "dog", "rooster", "sheep", "background"]
ANIMAL_CLASSES = CLASSES[:5]
BACKGROUND_INDEX = 5


@dataclass
class Event:
    onset: float
    offset: float
    label: str


@dataclass
class PostProcessingConfig:
    hop_seconds: float = 0.25
    median_frames: int = 3
    threshold_on: float = 0.5
    threshold_off: float = 0.3
    background_threshold: float = 0.5
    energy_floor_db: float = 35.0
    merge_gap_seconds: float = 0.3
    min_duration_seconds: float = 0.3
    per_class_threshold_on: Dict[str, float] = field(default_factory=dict)

    def thresholds_for(self, label: str) -> tuple:
        thr_on = self.per_class_threshold_on.get(label, self.threshold_on)
        # keep the on/off gap constant when a class overrides the on threshold
        thr_off = max(0.15, thr_on - (self.threshold_on - self.threshold_off))
        return thr_on, thr_off


def median_smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    padded = np.pad(x, pad, mode="edge")
    return np.array([np.median(padded[i:i + k]) for i in range(len(x))])


def _hysteresis_runs(prob: np.ndarray, thr_on: float, thr_off: float) -> List[tuple]:
    """Dual-threshold detection with onset/offset backtracking.

    A run is a contiguous span where prob stays above thr_off and contains at
    least one frame above thr_on. The high threshold confirms a real event
    (precision); the low threshold sets the boundaries, so the onset is
    backtracked to where the probability first started rising instead of where
    it finally crossed thr_on. This removes the late-onset bias of a windowed
    detector without admitting new false positives.
    """
    above_on = prob >= thr_on
    above_off = prob >= thr_off
    runs: List[tuple] = []
    i, n = 0, len(prob)
    while i < n:
        if above_off[i]:
            j = i
            while j < n and above_off[j]:
                j += 1
            if above_on[i:j].any():
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def _merge_and_filter(events: List[Event], merge_gap: float, min_duration: float) -> List[Event]:
    if not events:
        return []
    events = sorted(events, key=lambda e: (e.label, e.onset))
    merged: List[Event] = []
    for event in events:
        if merged and merged[-1].label == event.label and event.onset - merged[-1].offset <= merge_gap:
            merged[-1] = Event(merged[-1].onset, max(merged[-1].offset, event.offset), event.label)
        else:
            merged.append(Event(event.onset, event.offset, event.label))
    kept = [e for e in merged if (e.offset - e.onset) >= min_duration]
    kept.sort(key=lambda e: e.onset)
    return kept


def _energy_gate(energy_db: Optional[np.ndarray], config: PostProcessingConfig, n_frames: int) -> np.ndarray:
    """Frames more than energy_floor_db below the recording's own 95th percentile
    level are forced to silence, so nothing is detected in quiet passages. The
    threshold is relative to the clip, so it adapts to how loud the recording is."""
    if energy_db is None:
        return np.ones(n_frames)
    reference = np.percentile(energy_db, 95)
    return (energy_db >= reference - config.energy_floor_db).astype(float)


def timeline_to_events(
    probabilities: np.ndarray,
    frame_times: np.ndarray,
    config: Optional[PostProcessingConfig] = None,
    energy_db: Optional[np.ndarray] = None,
    emit_background: bool = False,
) -> tuple:
    """Convert a per-frame probability timeline into discrete events.

    probabilities : (T, C) sigmoid outputs at the frame centres in frame_times.
                    C is 6 (5 animals + explicit background head).
    Returns (events, smoothed_animal_probabilities).

    Pipeline, per class, in order:
      1. median smoothing        -> remove single-frame flicker
      2. energy gate             -> forbid detections in near-silence
      3. background gate         -> the learned background head blocks frames that
                                    are confidently non-animal (catches loud noise
                                    the energy gate would pass through)
      4. hysteresis thresholding -> a high threshold to start, a low one to sustain
      5. merge small gaps        -> one event with a brief dip stays one event
      6. minimum-duration filter -> drop blips too short to be a real vocalisation
    """
    config = config or PostProcessingConfig()
    n_frames, n_classes = probabilities.shape
    n_animals = len(ANIMAL_CLASSES)
    half_hop = config.hop_seconds / 2.0

    gate = _energy_gate(energy_db, config, n_frames)
    if n_classes > n_animals:
        background = median_smooth(probabilities[:, BACKGROUND_INDEX], config.median_frames)
        gate = gate * (background < config.background_threshold).astype(float)

    smoothed = np.zeros((n_frames, n_animals))
    events: List[Event] = []
    for c in range(n_animals):
        label = ANIMAL_CLASSES[c]
        prob = median_smooth(probabilities[:, c], config.median_frames) * gate
        smoothed[:, c] = prob
        thr_on, thr_off = config.thresholds_for(label)
        for start, end in _hysteresis_runs(prob, thr_on, thr_off):
            onset = max(0.0, frame_times[start] - half_hop)
            offset = frame_times[end] + half_hop
            events.append(Event(onset, offset, label))

    events = _merge_and_filter(events, config.merge_gap_seconds, config.min_duration_seconds)

    if emit_background and n_classes > n_animals:
        bg_prob = median_smooth(probabilities[:, BACKGROUND_INDEX], config.median_frames)
        thr_on, thr_off = config.thresholds_for("background")
        bg_events = [
            Event(max(0.0, frame_times[s] - half_hop), frame_times[e] + half_hop, "background")
            for s, e in _hysteresis_runs(bg_prob, thr_on, thr_off)
        ]
        events += _merge_and_filter(bg_events, config.merge_gap_seconds, config.min_duration_seconds)

    return events, smoothed


def events_to_records(events: List[Event]) -> List[dict]:
    """Serialise events to the reporting schema: event_start, event_end, animal."""
    records = []
    for event in sorted(events, key=lambda e: e.onset):
        records.append({
            "event_start": round(float(event.onset), 2),
            "event_end": round(float(event.offset), 2),
            "animal": event.label,
        })
    return records
