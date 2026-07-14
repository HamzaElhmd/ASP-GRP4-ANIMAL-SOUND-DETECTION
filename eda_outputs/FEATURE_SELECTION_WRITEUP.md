## Feature Selection Writeup

Section 2 deliverable (Owner: B). Synthesizes the feature-visualization pass (`src/eda.py`), the multi-label co-occurrence check (`eda_outputs/multilabel_sources/`), the convolutional-filter pass (`eda_outputs/conv_filters/`), and the clustering sanity check (`eda_outputs/clustering/`) into a recommendation for which audio representation each model track should use.

### Recommendation

| Track | Owner | Representation | Reasoning source |
|---|---|---|---|
| Classical ML | A | MFCC / RMS / ZCR, **per short frame** (not pooled per clip) | Clustering result |
| CNN | B (this track) | Mel-spectrogram images (2D, time preserved) | Feature-viz + conv-filter results |
| Sequential (RNN/LSTM/GRU) | C | Same frame-level features as A, or a CNN-derived embedding sequence | Clustering result (time structure matters) |
| Raw waveform / WaveNet-style dilated conv | -- | **Not chosen** for the CNN track | See "Why not raw waveform" below |

### Why mel-spectrogram + 2D CNN (this track's choice)

Three pieces of evidence point the same direction:

1. **The feature-visualization pass** (`eda_outputs/*.png`) showed each of the 5 animal classes has a visually distinct, fairly consistent shape in the mel spectrogram across multiple samples: rooster's flat stable harmonic bands, dog's isolated broadband bursts, cat's wavering harmonic bursts, cow's smeared low-frequency energy, sheep's wavering-but-noisier bleat. These are 2D time-frequency shapes, not scalar properties.
2. **The convolutional-filter pass** confirmed that even 3 small *fixed, untrained* filters running directly on the mel-spectrogram image already sharpen class-relevant structure -- a horizontal-edge filter isolates rooster's bands cleanly, a vertical-edge filter isolates dog's bursts. A trained 2D CNN does the same thing but learns the filters instead of us hand-picking them, and can stack many of them -- this is a small-scale preview of exactly the mechanism a 2D CNN would exploit.
3. **The clustering sanity check** showed that collapsing a clip into a single pooled feature vector (MFCC mean/std over the full 2 seconds) destroys class separability (ARI ~0.04, near-random). The information that actually distinguishes classes lives in *how energy is distributed over time and frequency together*, not in a time-averaged summary. A 2D CNN over the mel-spectrogram preserves exactly that time-frequency structure, which the failed clustering experiment shows is necessary, not optional.

### Why not raw waveform (WaveNet-style dilated 1D conv)

The spec allows either mel-spectrogram -> 2D conv or raw waveform -> 1D dilated conv for the CNN track, and asks for a justification of whichever is skipped.

Raw waveform is not chosen here because:

- The mel-spectrogram already does the work of decomposing the signal into frequency bands via a fixed, well-understood transform (STFT + mel filterbank). Steps 2 and 4 show that decomposition is already enough to make class structure visible to a human eye and to simple fixed filters. A waveform-based model would have to *learn* an equivalent frequency decomposition from scratch, which needs more data, depth, and training time than a mel-spectrogram model that starts from a representation already known to carry the signal.
- The dataset is a small seed set (908 raw clips) expanded via augmentation to ~7,827 processed frames -- workable for a moderately sized 2D CNN over compact mel-spectrogram images, but a tighter budget for a deep dilated waveform stack, which typically needs a large receptive field (and therefore more layers/parameters) to reach the same effective frequency resolution a mel filterbank gives for free.
- Person C's RNN track already covers the "let the model use temporal context beyond a fixed window" angle the spec highlights as WaveNet's main advantage over a plain CNN. Between the three tracks, mel-spectrogram CNN (local time-frequency pattern detection, evidenced directly by Step 4) and RNN (temporal context across frames) together cover more ground than mel-spectrogram CNN and waveform CNN would, since the latter two are both fundamentally convolutional and would likely learn overlapping things.

### Caveat for Person A (classical ML track)

The clustering result is a specific, evidence-backed warning worth passing on directly: **do not summarize each clip into one pooled feature vector** (e.g. mean MFCC over the whole 2-second frame) -- that is exactly what was tested here and it collapsed class separability to near-random. Compute MFCC/RMS/ZCR at a finer frame granularity (e.g. 25ms window / 10ms hop, matching typical classical-ML audio pipelines) and feed those per-frame vectors as separate rows/timesteps, not one vector per clip.

### Caveats that apply regardless of track

- **Background is not a clean negative class.** The convolutional-filter pass found background clips that produce filter responses resembling both a harmonic band (steady drone/hum) and a transient burst (rhythmic insect/ambient noise). Any track needs enough background training examples covering this variety, not just quiet/silent background clips.
- **Sheep has high within-class amplitude variance** -- one of two sampled clips was much quieter than the other. Worth accounting for in any energy-based thresholding, and worth checking during training that sheep recall doesn't quietly suffer relative to other classes.
- **Multi-label overlap is real but rare** -- 2.7% of identified spans in the one external recording reviewed so far (`eda_outputs/multilabel_sources/SUMMARY.md`) showed two animals active at once. This is the evidence behind the sigmoid-per-class (not softmax) output design shared by all three tracks, but it's a lower bound from a single source, not a precise population rate.
- **Sourced audio license gap is still open** (both the original dataset and the YouTube multi-label source) -- unresolved from earlier, flagged again here since this writeup is a natural point to close it out before the final report.

### Files referenced

- `eda_outputs/*.png` -- feature-visualization pass (Step 2 equivalent)
- `eda_outputs/multilabel_sources/SUMMARY.md` -- co-occurrence evidence
- `eda_outputs/conv_filters/SUMMARY.md` -- convolutional filter findings
- `eda_outputs/clustering/SUMMARY.md` -- clustering sanity check, full metrics and contingency tables
