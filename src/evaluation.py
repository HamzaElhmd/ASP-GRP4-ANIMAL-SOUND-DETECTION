from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from postprocessing import ANIMAL_CLASSES, Event
except ModuleNotFoundError:
    from src.postprocessing import ANIMAL_CLASSES, Event

COLLAR_SECONDS = 0.5
SEGMENT_SECONDS = 0.1


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _matches(predicted: Event, reference: Event, collar: float) -> bool:
    return (
        predicted.label == reference.label
        and abs(predicted.onset - reference.onset) <= collar
        and abs(predicted.offset - reference.offset) <= collar
    )


def event_based_metrics(
    predicted: List[Event],
    reference: List[Event],
    collar: float = COLLAR_SECONDS,
) -> Dict[str, Dict[str, float]]:
    """Event-based precision/recall/F1 with a +/- collar tolerance on both onset
    and offset (the grading rule: an event is correct within +/- 500 ms). A
    predicted event is a true positive if it matches an unused reference event of
    the same class; matching is greedy by onset distance so each reference is used
    at most once."""
    results: Dict[str, Dict[str, float]] = {}
    for label in ANIMAL_CLASSES:
        preds = sorted([e for e in predicted if e.label == label], key=lambda e: e.onset)
        refs = sorted([e for e in reference if e.label == label], key=lambda e: e.onset)
        used = [False] * len(refs)
        tp = 0
        for pred in preds:
            best, best_dist = -1, None
            for idx, ref in enumerate(refs):
                if used[idx] or not _matches(pred, ref, collar):
                    continue
                dist = abs(pred.onset - ref.onset)
                if best_dist is None or dist < best_dist:
                    best, best_dist = idx, dist
            if best >= 0:
                used[best] = True
                tp += 1
        fp = len(preds) - tp
        fn = len(refs) - tp
        results[label] = _prf(tp, fp, fn)

    tp = sum(results[c]["tp"] for c in ANIMAL_CLASSES)
    fp = sum(results[c]["fp"] for c in ANIMAL_CLASSES)
    fn = sum(results[c]["fn"] for c in ANIMAL_CLASSES)
    results["overall_micro"] = _prf(tp, fp, fn)
    results["overall_macro"] = {
        "precision": float(np.mean([results[c]["precision"] for c in ANIMAL_CLASSES])),
        "recall": float(np.mean([results[c]["recall"] for c in ANIMAL_CLASSES])),
        "f1": float(np.mean([results[c]["f1"] for c in ANIMAL_CLASSES])),
    }
    return results


def _segment_matrix(events: List[Event], duration: float, resolution: float) -> np.ndarray:
    n_segments = max(1, int(np.ceil(duration / resolution)))
    matrix = np.zeros((n_segments, len(ANIMAL_CLASSES)), dtype=bool)
    for event in events:
        if event.label not in ANIMAL_CLASSES:
            continue
        c = ANIMAL_CLASSES.index(event.label)
        start = int(np.floor(event.onset / resolution))
        end = int(np.ceil(event.offset / resolution))
        matrix[max(0, start):min(n_segments, end), c] = True
    return matrix


def segment_based_metrics(
    predicted: List[Event],
    reference: List[Event],
    duration: float,
    resolution: float = SEGMENT_SECONDS,
) -> Dict[str, Dict[str, float]]:
    """Frame/segment-level precision/recall/F1 per class: discretise the timeline
    into fixed segments and compare which classes are active in each. Complements
    the event-based score by rewarding partial temporal overlap."""
    pred_matrix = _segment_matrix(predicted, duration, resolution)
    ref_matrix = _segment_matrix(reference, duration, resolution)
    results: Dict[str, Dict[str, float]] = {}
    for c, label in enumerate(ANIMAL_CLASSES):
        p = pred_matrix[:, c]
        r = ref_matrix[:, c]
        tp = int(np.sum(p & r))
        fp = int(np.sum(p & ~r))
        fn = int(np.sum(~p & r))
        results[label] = _prf(tp, fp, fn)
    results["overall_macro"] = {
        "precision": float(np.mean([results[c]["precision"] for c in ANIMAL_CLASSES])),
        "recall": float(np.mean([results[c]["recall"] for c in ANIMAL_CLASSES])),
        "f1": float(np.mean([results[c]["f1"] for c in ANIMAL_CLASSES])),
    }
    return results


def near_miss_breakdown(
    predicted: List[Event],
    reference: List[Event],
    collar: float = COLLAR_SECONDS,
) -> Dict[str, int]:
    """Explain the errors, not just count them. Every reference event is one of:
      correct   - matched within the collar by a same-class prediction
      mistimed  - a same-class prediction overlaps it but misses the collar
      confused  - a different-class prediction overlaps it (wrong animal)
      missed    - nothing overlaps it at all
    Predictions that match no reference at all are counted as spurious.
    """
    def overlaps(a: Event, b: Event) -> bool:
        return not (a.offset <= b.onset or a.onset >= b.offset)

    counts = {"correct": 0, "mistimed": 0, "confused": 0, "missed": 0, "spurious": 0}
    matched_pred = [False] * len(predicted)

    for ref in reference:
        found = False
        for idx, pred in enumerate(predicted):
            if _matches(pred, ref, collar) and not matched_pred[idx]:
                matched_pred[idx] = True
                counts["correct"] += 1
                found = True
                break
        if found:
            continue
        if any(p.label == ref.label and overlaps(p, ref) for p in predicted):
            counts["mistimed"] += 1
        elif any(p.label != ref.label and overlaps(p, ref) for p in predicted):
            counts["confused"] += 1
        else:
            counts["missed"] += 1

    for idx, pred in enumerate(predicted):
        if not matched_pred[idx] and not any(overlaps(pred, r) for r in reference):
            counts["spurious"] += 1
    return counts


def format_report(
    predicted: List[Event],
    reference: List[Event],
    duration: float,
    collar: float = COLLAR_SECONDS,
) -> str:
    event = event_based_metrics(predicted, reference, collar)
    segment = segment_based_metrics(predicted, reference, duration)
    breakdown = near_miss_breakdown(predicted, reference, collar)

    lines = [f"event-based (+/- {collar:.1f}s collar)      P      R      F1"]
    for label in ANIMAL_CLASSES:
        m = event[label]
        lines.append(f"  {label:10} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}")
    macro = event["overall_macro"]
    lines.append(f"  {'macro':10} {macro['precision']:6.3f} {macro['recall']:6.3f} {macro['f1']:6.3f}")
    lines.append(f"segment-based macro F1: {segment['overall_macro']['f1']:.3f}")
    lines.append("near-miss breakdown: " + ", ".join(f"{k}={v}" for k, v in breakdown.items()))
    return "\n".join(lines)
