# CNN Model (Track B)

## Scope

Section 3 CNN deliverable, implemented in `src/cnn_model.py`. A 2D convolutional
network over the mel-spectrogram with a multi-label sigmoid head. It reads the
mel representation from the feature-extraction track (Section 2) and its output
is scored by the shared event harness (Section 4).

## Output format (shared across all tracks)

The model produces **frame-level, multi-label predictions**: an **independent
sigmoid per class per frame**, not a softmax, so overlapping animals are not
structurally excluded. At inference `predict_timeline()` slides the 1 s window
(0.25 s hop, Hann) over a recording and returns a `(T, 6)` probability timeline
plus frame times and per-frame energy — exactly the shape the Section 4 harness
consumes, so this model is scored by the same code as the HMM and Sequential
tracks.

## Architecture choice: mel-spectrogram -> 2D CNN

The spec allows either (a) mel-spectrogram -> 2D conv or (b) raw waveform -> 1D
dilated conv (WaveNet-style), and asks for a justification of the one chosen and a
note on the one skipped. We chose **(a)**, and the EDA (Section 2) gives three
pieces of evidence that point the same way:

1. **Distinct 2D shapes per class.** The feature-visualisation pass showed each
   animal has a consistent time-frequency signature in the mel-spectrogram
   (rooster's flat harmonic bands, dog's broadband bursts, cat's wavering
   harmonics, cow's smeared low-frequency energy, sheep's noisier bleat). These
   are 2D patterns, which is exactly what stacked 2D convolutions detect.
2. **Fixed filters already sharpen class structure.** The convolutional-filter
   pass found that even 3 small *untrained* kernels on the mel-spectrogram already
   isolate class-relevant structure (a horizontal-edge filter cleans up rooster's
   bands, a vertical-edge filter isolates dog's bursts). A trained 2D CNN learns
   and stacks many such filters — a direct scaled-up version of that mechanism.
3. **Time-frequency structure is necessary.** The clustering sanity check showed
   that collapsing a clip to a pooled feature vector destroys class separability
   (ARI ~0.04). The discriminative information lives in *how energy is distributed
   over time and frequency together*, which a 2D CNN over the mel-spectrogram
   preserves.

### Why not raw waveform (WaveNet-style 1D dilated conv)

- The mel-spectrogram already performs the frequency decomposition (STFT + mel
  filterbank) that the EDA shows is enough to make class structure visible to
  simple fixed filters. A raw-waveform model would have to *learn* an equivalent
  decomposition from scratch, which needs more data, depth, and training time than
  our small dataset (~2,600 original frames) comfortably supports.
- A dilated waveform stack needs a large receptive field (many layers) to reach
  the frequency resolution a mel filterbank gives for free.
- The three tracks together already cover more ground with **mel-CNN** (local
  time-frequency patterns) + **Sequential/RNN** (temporal context across frames)
  than mel-CNN + waveform-CNN would, since the latter two are both convolutional
  and would learn overlapping things.

## Network

Four 2D-conv blocks (32 -> 64 -> 128 -> 256 channels, each BatchNorm + ReLU, first
three with 2x2 max-pooling), then **adaptive average pooling**, dropout, and a
linear multi-label head (~0.4 M parameters). Adaptive pooling keeps the head
independent of the input length, so it trains on 1 s windows and runs on the 1 s
inference window unchanged.

## Training -- synthesised, scene-matched windows

The key training decision: the model is **not** trained on the isolated 2 s clips
directly. The task is event detection in **continuous, background-heavy,
overlapping** audio, so training on clean isolated clips creates a distribution
mismatch (the model classifies clips well but fails on continuous scenes). Instead
each training window is **synthesised on the fly** to look like the inference scenes
(`synth_window` / `SynthWindowDataset`):

1. clips from **folds 1-4** are loaded and **trimmed to their active region**
   (`trim_active`), removing the silent padding of the fixed 2 s frames -- so a
   silent tail is never labelled as the animal;
2. **0, 1 or 2** animals are chosen (P = 0.30 / 0.45 / 0.25), each cropped to 1 s,
   gain-varied, and **mixed onto a background bed at a random SNR**;
3. a **gaussian noise floor** is added, then **SpecAugment** masks random
   frequency/time bands.

This makes the training distribution match the evaluation distribution and gives
three things at once: **overlaps** (polyphony), **background-only windows** (P=0.30,
so the model learns to say "nothing"), and effectively unlimited augmentation
(fresh windows every epoch). Features use a **fixed** mel normalisation (a constant
dB offset), not per-example standardisation, so silence stays low instead of being
stretched into spurious activations.

Trained to convergence with **early stopping** on the fold-5 validation macro-F1
(validation windows are synthesised from fold-5-sourced clips) and a learning-rate
scheduler that halves the rate on a plateau. The course **overfit sanity check**
(`overfit_check()`) confirms the model can memorise a tiny set of synthesised
windows before a full run.

## Reporting

- **Per-class precision / recall / F1 at the frame level** on the fold-5 hold-out
  (`frame_level_report()`) — the cheap diagnostic before event formation.
- **Training curves** (loss and val macro-F1 vs epoch), saved by the notebook.
- **Predictions visualised** on the mel-spectrogram (`plot_predictions()`) — the
  per-class probability curves lined up under the spectrogram, for qualitative
  inspection.
- Event-based P/R/F1 (+/- 500 ms) via the Section 4 harness, for the final
  three-model comparison table.

## Files

- `src/cnn_model.py` — model, training, `predict_timeline`, `frame_level_report`, `plot_predictions`
- `train_cnn_colab.ipynb` — end-to-end train + evaluate notebook
