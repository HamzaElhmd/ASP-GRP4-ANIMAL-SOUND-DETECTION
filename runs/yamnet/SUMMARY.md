## YAMNet (frozen) + classifier head -- pipeline, choices, and results

### Approach: frozen feature extraction, not fine-tuned

YAMNet (Google, pretrained on AudioSet, loaded via TF Hub) is never updated -- it runs in inference-only mode, producing a 1024-dim embedding every ~0.48s. A small trainable PyTorch head (`YamnetClassifierHead` in `src/yamnet_model.py`: Linear -> ReLU -> Dropout -> Linear) sits on top, trained per-timestep with independent sigmoid per class (not softmax), same multi-label design as every other track.

Chosen over full fine-tuning because: the dataset is small (908 raw clips) relative to YAMNet's size, so fine-tuning the whole network risks catastrophic forgetting -- overwriting the general audio knowledge that made using a pretrained model worthwhile in the first place. There's also no GPU in this environment; frozen extraction means YAMNet only ever runs forward once per clip (cached), while fine-tuning would mean a full backward pass through the whole network every epoch. Frozen features also keep the classifier head small and fast to retrain, so class-imbalance fixes (calibration, reweighting) stay cheap to experiment with -- same reasoning applied on the sequential track.

### New dependency

TensorFlow + TensorFlow Hub, needed because YAMNet's official pretrained weights are TF-native (unofficial PyTorch ports exist but carry correctness risk for a graded project). Added to `requirements.txt`. This is a second deep learning framework alongside PyTorch in this repo -- a real tradeoff, accepted because using the official Google weights mattered more than framework purity here.

### Scope: training_frame only, not the full augmented corpus

`build_embedding_cache()` in `src/yamnet_data.py` defaults to `segment_types=("training_frame",)`, excluding `augmentation_frame` -- 2,609 clips instead of the full 7,827. Two reasons: YAMNet was pretrained on millions of real-world clips, so the classifier head needs less augmentation-driven volume than training a model from scratch does (the sequential track's GRU used the full augmented corpus for exactly this reason -- it doesn't have that head start). Also, each embedding extraction call took ~586ms on this CPU-only setup -- the full corpus would take over an hour, training_frame-only takes about 25 minutes, a one-time cost since embeddings are cached to `runs/yamnet/embedding_cache.pt` (~44MB) and never recomputed.

Split: same `processed/split.csv` (70/15/15, source-clip level, no leakage) as the sequential track, brought over from `sequential-modeling` rather than re-deriving a different one -- this was a real, verified problem when the HMM branch and the sequential branch used different splits, so this track deliberately reuses the same one for comparability.

### Frame-level evaluation (test set, training_frame only)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cat | 0.859 | 0.682 | 0.760 | 500 |
| cow | 0.842 | 0.506 | 0.632 | 168 |
| dog | 0.865 | 0.583 | 0.697 | 384 |
| rooster | 0.891 | 0.340 | 0.492 | 144 |
| sheep | 0.798 | 0.540 | 0.644 | 124 |
| **macro avg** | **0.851** | **0.530** | **0.645** | |

After val-based per-class threshold calibration (`calibrate_thresholds()` in `src/evaluate_yamnet.py`, same approach as the sequential track):

| Class | Calibrated threshold | F1 |
|---|---:|---:|
| cat | 0.40 | 0.732 |
| cow | 0.35 | 0.709 |
| dog | 0.30 | 0.738 |
| rooster | 0.20 | 0.553 |
| sheep | 0.35 | 0.829 |

### Comparison to the sequential (GRU) track

Meaningfully stronger across the board, most notably on rooster -- the sequential track's weakest class by far (test F1 0.164 at flat 0.5, 0.454 calibrated) improves to 0.492 flat / 0.553 calibrated here. Precision is also much higher even before calibration (0.798-0.891 vs. the GRU's 0.569-0.792 range), meaning this approach is less prone to false-triggering out of the gate. Directly consistent with the reasoning for choosing frozen pretrained features: a small dataset benefits more from features learned on millions of real-world clips than from a small model learning its own features from ~900 raw clips.

### One weakness that persists across both architectures

A quick qualitative check (not yet a full plot pass, just spot-checking one clip per class) found the same cat/background confusion already flagged on the sequential track: on a background-labeled test clip, cat's probability hits 0.633 (would cross most reasonable thresholds) while every other class stays under 0.1. On its own clips, rooster (0.999) and cat (0.997) are both extremely confident and correct.

This is worth taking seriously rather than writing off as one model's quirk: the *same* confusion shows up in two architecturally very different models (a from-scratch GRU over raw mel-spectrograms, and a frozen pretrained embedding + small head). That's more consistent with a genuine acoustic overlap between some background clips and cat vocalizations in the underlying data than with either model's specific training process. Worth flagging directly in the presentation's "where does it fail, and why" section, backed by two independent pieces of evidence now, not one.

### Files

- `embedding_cache.pt` -- cached YAMNet embeddings, training_frame clips only (2,609 entries)
- `yamnet_head.pt` -- trained classifier head weights (15 epochs)
- `training_curve.png`, `overfit_check.png` -- training diagnostics
- `frame_level_metrics.txt` -- the table above, machine-readable
- `src/yamnet_features.py` -- frozen YAMNet wrapper
- `src/yamnet_data.py` -- dataset + embedding cache builder, reuses `src/split.py`
- `src/yamnet_model.py` -- the small classifier head
- `src/train_yamnet.py`, `src/evaluate_yamnet.py` -- training and evaluation, same structure as the sequential track's equivalents

### Not yet done

- Qualitative overlay plots (the sequential track has these; this track only has a quick numeric spot-check so far)
- Training on the full augmented corpus, if the training_frame-only result needs more data to improve further
- PANNs comparison and the combined/ensemble experiment -- someone else's part of this branch, not started here
- A corpus-wide real-audio scan across all 11 chunks (the sequential track has this; only one chunk checked here so far)

### Real-audio test (added after the above)

`src/yamnet_predict_continuous.py` -- simpler than the sequential track's equivalent, since YAMNet does its own internal windowing for any input length, no separate scanner needed. Tested on the same chunk and the same 95-155s window used for the GRU comparison, so the two are directly comparable.

**Rooster is now correctly detected on real audio, not just clean clips.** Across all 6 confirmed rooster spans in the 95-155s window, rooster's probability rises clearly (0.6-0.9+) during nearly every one of them -- the exact opposite of the GRU, which stayed flat and low on every one of these same spans. The confirmed dog+rooster overlap (~143-148s) shows both dog and rooster elevated together, correctly capturing the overlap rather than one class dominating.

This is a meaningfully stronger result than the clean test-set numbers alone suggested, and it directly contradicts what happened on the sequential track, where clean-clip metrics didn't reveal the real-audio rooster failure until this exact kind of test was run. Good sign that this result is real, not an artifact of the test set.

Cat/background over-triggering was not re-checked across the full recording here (only the single background clip spot-check earlier) -- worth doing before treating that finding as fully resolved or fully persistent on this architecture.
