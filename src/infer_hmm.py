from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.hmm import infer_continuous_file, save_inference_json


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run binary HMM inference on a continuous wav file")
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/hmm"))
    parser.add_argument("--output", type=Path, default=Path("result_hmm.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--median-width", type=int, default=5)
    parser.add_argument("--gap-fill-frames", type=int, default=3)
    args = parser.parse_args(argv)

    events = infer_continuous_file(
        args.wav_path,
        model_dir=args.model_dir,
        threshold=args.threshold,
        median_width=args.median_width,
        gap_fill_frames=args.gap_fill_frames,
    )
    save_inference_json(events, args.output)


if __name__ == "__main__":
    main()
