import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio

try:
    from preprocessor import inference_time_windowing
except ModuleNotFoundError:
    from src.preprocessor import inference_time_windowing

SOURCE_DIR = Path("eda_outputs/multilabel_sources")
OUTPUT_CSV = SOURCE_DIR / "candidate_events.csv"

WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.25
RMS_THRESHOLD = 0.02
MIN_GAP_SECONDS = 0.5


def find_active_spans(waveform: torch.Tensor, sample_rate: int) -> List[Tuple[float, float, float]]:
    frames = inference_time_windowing(waveform, sample_rate, WINDOW_SECONDS, HOP_SECONDS)
    rms_per_frame = torch.sqrt(torch.mean(frames ** 2, dim=-1))

    active = rms_per_frame > RMS_THRESHOLD
    spans: List[Tuple[float, float, float]] = []
    start_idx = None
    for i, is_active in enumerate(active.tolist()):
        if is_active and start_idx is None:
            start_idx = i
        elif not is_active and start_idx is not None:
            spans.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        spans.append((start_idx, len(active) - 1))

    merged: List[Tuple[float, float, float]] = []
    for start_idx, end_idx in spans:
        start_sec = start_idx * HOP_SECONDS
        end_sec = end_idx * HOP_SECONDS + WINDOW_SECONDS
        peak_rms = rms_per_frame[start_idx:end_idx + 1].max().item()
        if merged and start_sec - merged[-1][1] <= MIN_GAP_SECONDS:
            prev_start, _, prev_peak = merged[-1]
            merged[-1] = (prev_start, end_sec, max(prev_peak, peak_rms))
        else:
            merged.append((start_sec, end_sec, peak_rms))

    return merged


def scan_all_chunks() -> None:
    chunk_files = sorted(SOURCE_DIR.glob("*_part_*.wav"))
    if not chunk_files:
        raise RuntimeError(f"No chunk files found in {SOURCE_DIR}")

    all_rows = []
    for chunk_path in chunk_files:
        waveform, sample_rate = torchaudio.load(str(chunk_path))
        spans = find_active_spans(waveform, sample_rate)

        for start_sec, end_sec, peak_rms in spans:
            all_rows.append((chunk_path.name, start_sec, end_sec, peak_rms))

        print(f"{chunk_path.name}: {len(spans)} candidate active span(s)")

    all_rows.sort(key=lambda r: r[3], reverse=True)

    with OUTPUT_CSV.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["chunk_file", "start_sec", "end_sec", "duration_sec", "peak_rms", "animal_labels", "notes"])
        for chunk_name, start_sec, end_sec, peak_rms in all_rows:
            writer.writerow([
                chunk_name,
                round(start_sec, 2),
                round(end_sec, 2),
                round(end_sec - start_sec, 2),
                round(peak_rms, 4),
                "",
                "",
            ])

    print(f"\nWrote candidate spans to {OUTPUT_CSV}, sorted loudest first across all chunks.")
    print("Start at row 1 and work down; fill in animal_labels by ear.")


if __name__ == "__main__":
    scan_all_chunks()
