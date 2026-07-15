"""Calibrates per-class thresholds against the real farm recording plus the
manually-confirmed labels from the multi-label EDA, instead of the clean
isolated val-set clips used by calibrate_thresholds() in
src/evaluate_sequential.py.

Why this exists: thresholds calibrated on the val set optimize for the val
set's failure mode. For cat specifically, the val set's problem was
under-detection, so calibration pushed its threshold down -- but on real
continuous audio cat's problem is over-triggering on non-cat sounds, so a
lower threshold made that worse, not better. This calibrates against
something closer to the actual deployment scenario instead.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

try:
    from evaluate_sequential import CHECKPOINT, load_trained_model
    from predict_continuous import predict_continuous
    from sequential_data import ANIMAL_CLASSES
except ModuleNotFoundError:
    from src.evaluate_sequential import CHECKPOINT, load_trained_model
    from src.predict_continuous import predict_continuous
    from src.sequential_data import ANIMAL_CLASSES

AUDIO_DIR = Path("eda_outputs/multilabel_sources")
CANDIDATE_CSV = AUDIO_DIR / "candidate_events.csv"


def load_labeled_spans(candidate_csv: Path = CANDIDATE_CSV) -> List[dict]:
    with open(candidate_csv) as f:
        rows = list(csv.DictReader(f))
    return [
        r for r in rows
        if r["animal_labels"] and r["animal_labels"] not in ("unknown", "none")
    ]


def build_real_world_dataset(model, candidate_csv: Path = CANDIDATE_CSV) -> Tuple[np.ndarray, np.ndarray]:
    """For every manually-confirmed span, find the model's windows that
    fall inside it and build a target vector from the confirmed labels --
    1 for animals confirmed present, 0 for the other classes (since a
    human confirmed exactly what was and wasn't there at that moment)."""
    labeled_spans = load_labeled_spans(candidate_csv)

    chunk_names = sorted(set(r["chunk_file"] for r in labeled_spans))
    predictions_by_chunk = {}
    for chunk_name in chunk_names:
        wav_path = AUDIO_DIR / chunk_name
        timestamps, scores = predict_continuous(str(wav_path), model)
        predictions_by_chunk[chunk_name] = (timestamps, scores.numpy())

    all_scores, all_targets = [], []
    for span in labeled_spans:
        timestamps, scores = predictions_by_chunk[span["chunk_file"]]
        start, end = float(span["start_sec"]), float(span["end_sec"])
        confirmed = set(l for l in span["animal_labels"].split(",") if l in ANIMAL_CLASSES)
        if not confirmed:
            continue

        target = np.array([1.0 if cls in confirmed else 0.0 for cls in ANIMAL_CLASSES])

        for t, row_scores in zip(timestamps, scores):
            if start <= t <= end:
                all_scores.append(row_scores)
                all_targets.append(target)

    return np.stack(all_scores), np.stack(all_targets)


def calibrate_on_real_world(model, thresholds=None) -> Dict[str, float]:
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]

    scores, targets = build_real_world_dataset(model)
    print(f"calibration set: {len(scores)} windows drawn from {len(load_labeled_spans())} confirmed spans")

    best_thresholds = {}
    print(f"{'class':10s} {'best_thresh':>12s} {'f1_at_0.5':>10s} {'f1_real_calib':>14s}")

    for i, cls in enumerate(ANIMAL_CLASSES):
        best_f1, best_t = -1.0, 0.5
        f1_at_half = None
        for t in thresholds:
            preds = (scores[:, i] > t).astype(float)
            _, _, f1, _ = precision_recall_fscore_support(
                targets[:, i], preds, average="binary", zero_division=0
            )
            if abs(t - 0.5) < 1e-9:
                f1_at_half = f1
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[cls] = best_t
        print(f"{cls:10s} {best_t:12.2f} {f1_at_half:10.3f} {best_f1:14.3f}")

    return best_thresholds


if __name__ == "__main__":
    model = load_trained_model(CHECKPOINT)
    calibrate_on_real_world(model)
