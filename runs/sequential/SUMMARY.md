## Sequential model (GRU) -- architecture, evaluation, and qualitative findings

### Status: v1, not a final result

This checkpoint (`sequential_model.pt`) was trained for 6 epochs on CPU, and the loss curve (`training_curve.png`) was still declining at epoch 6 -- it had not converged. Treat every number in this document as a first working baseline, not a tuned final result. Better numbers are expected from further training; if you're using this checkpoint to build or compare against, don't treat today's metrics as the ceiling of what this architecture can do.

### Input contract -- what you must feed the model

Before calling `SequentialEventDetector`, audio must be: mono, 16kHz sample rate (matching `TARGET_SR` in `src/preprocessor.py`), converted to a mel-spectrogram with `n_fft=512, hop_length=160, win_length=400, n_mels=64` (the constants `N_FFT/HOP_LENGTH/WIN_LENGTH/N_MELS` in `src/eda.py`), in dB scale (`torchaudio.transforms.AmplitudeToDB`), and shaped `(batch, time, 64)` -- time-major, not the `(n_mels, time)` shape `torchaudio` returns by default, so transpose after computing the spectrogram. `compute_feature_sequence_from_waveform()` in `src/sequential_data.py` does all of this for you; call that rather than reimplementing it, so there's exactly one place this logic lives.

Output is raw logits, shape `(batch, time, 5)`, one per animal in this fixed order: `cat, cow, dog, rooster, sheep` (`ANIMAL_CLASSES` in `src/sequential_data.py`). Apply `torch.sigmoid()` yourself to get probabilities -- the model doesn't do this internally, since training uses `BCEWithLogitsLoss` directly on the logits for numerical stability.

### Continuous-audio test (not a pre-cut clip)

Every other test in this document runs the model on fixed 2-second training-format clips. This one runs it on a real, continuous, 3.5-minute unedited recording it has never seen the like of during training, using the shared scanner (`inference_time_windowing()` in `src/preprocessor.py`, 1s window / 250ms hop) -- see `src/predict_continuous.py`.

Tested against 15 spans in that recording that were manually confirmed by ear during the multi-label EDA (`eda_outputs/multilabel_sources/candidate_events.csv`), including the one confirmed dog+rooster overlap in the whole corpus.

Results, from `runs/sequential/continuous_test/zoomed_95_155.png`:

- **The dog+rooster overlap (143.75-148.5s) produces the strongest rooster response in the entire clip** -- the model correctly spikes on the one case that matters most for this project's multi-label premise.
- **Isolated rooster crows are mostly missed.** Of the 6 confirmed rooster spans in the 95-155s window, only the overlap one produced a clear spike; the other 5 stayed flat and low. This is a real generalization gap between clean training clips and real field audio, not just the frame-level labeling artifact described below -- worth taking seriously as a genuine weakness, not explaining away.
- **Dog and rooster get confused at least once** -- a strong dog spike (~0.95) appears during a confirmed rooster-only span (~120-125s).
- **Cat sits persistently elevated** across the whole window regardless of what's actually happening, consistent with the background/cat false-positive pattern found during EDA and again in the frame-level qualitative check below -- this isn't a one-off, it shows up in every real-audio test so far.

This is the first time this model has been run on anything other than a pre-cut 2-second clip, and it surfaced real weaknesses the frame-level val metrics didn't show on their own. Worth rerunning this same test once the model has had more training, to see which of these issues are undertraining and which are structural.

### Architecture and why

`SequentialEventDetector` (`src/sequential_model.py`): a single-layer GRU (hidden size 64) over a mel-spectrogram time sequence, followed by a per-timestep linear layer producing 5 raw logits (cat, cow, dog, rooster, sheep), trained with `BCEWithLogitsLoss` -- independent sigmoid per class per timestep, not softmax, so overlapping animals aren't structurally excluded, same as every track in this project.

**GRU/LSTM over a vanilla RNN**: a vanilla RNN's gradient signal decays multiplicatively at every timestep it's backpropagated through, so over a ~200-timestep sequence (2s at this hop length) it effectively can't learn dependencies that span more than a few steps -- the vanishing gradient problem covered in class. GRU (and LSTM) use gated connections that let gradients flow through many timesteps largely unchanged, which is the entire reason to pick either over a vanilla RNN for a sequence this long. GRU was chosen over LSTM specifically for this project mainly for practicality: fewer parameters, faster to train, and the dataset here is small enough that LSTM's extra gate (and extra capacity) is unlikely to be necessary.

**Why this track can target the "gap in the middle of a continuous event" problem the spec calls out**: a frame-independent classifier (Person A's track) makes each prediction from that frame's features alone, so a brief dip in energy mid-bark can flip the prediction to "not dog" for an isolated frame even though the event is clearly still ongoing to a human listener. A recurrent model's prediction at each timestep is a function of everything the network has seen in the sequence so far (and, implicitly, the pattern it's learned about what typically follows), so it has the structural capacity to smooth over a brief dip rather than treating it as a hard boundary. The dog qualitative example below shows this already starting to happen even at this early training stage.

### Frame-level evaluation (val set, 1,566 clips)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cat | 0.775 | 0.480 | 0.593 | 107,334 |
| cow | 0.664 | 0.424 | 0.518 | 33,165 |
| dog | 0.770 | 0.660 | 0.711 | 74,169 |
| rooster | 0.786 | 0.154 | 0.257 | 28,944 |
| sheep | 0.634 | 0.393 | 0.485 | 24,120 |
| **macro avg** | **0.726** | **0.422** | **0.513** | |

This is from the 6-epoch run in step 5 -- a working baseline, not a tuned final model. Precision is consistently higher than recall across every class, which the qualitative check below explains rather than just states as a number.

### Qualitative findings (predictions overlaid on spectrogram)

Three clips checked by eye, not just by the aggregate numbers:

- **Rooster** (`rooster_2-71162-A-1_frame_000.png`): the crow only starts around timestep ~118 of 201 in this particular clip -- the first half is quiet lead-in. The model's rooster probability correctly stays low (~0.25-0.3) during the quiet portion and rises sharply to ~0.7-0.85 exactly when the crow starts, staying elevated for its duration. **This is the model behaving correctly** -- but every timestep in this clip is labeled "rooster=1" (the whole-clip label broadcast to all timesteps, per the label scheme in `src/sequential_data.py`), so the frame-level evaluation counts the correctly-quiet lead-in as missed positives. This is very likely the dominant cause of rooster's low recall (0.154) -- not the model failing to recognize rooster, but the label scheme penalizing it for correctly recognizing silence. Worth flagging as a known limitation of frame-level scoring against whole-clip labels, not a model defect.
- **Dog** (`dog_2-117271-A-0_frame_000.png`): dog probability rises sharply at the very first bark burst (~timestep 8) and tracks each subsequent burst closely, including dipping and recovering between bursts rather than staying flat -- a small, real example of exactly the "smooth over the gap between bursts of the same event" behavior the sequential track is meant to provide. Consistent with dog having the best F1 (0.711) of the five classes.
- **Background** (`background_5-237315-A-31_frame_000.png`): mostly correct (all probabilities low), but cat probability is noisy and repeatedly crosses the 0.5 threshold on this clip -- a real false-positive tendency, not a fluke of one plot. This lines up directly with the EDA finding from the convolutional-filter pass that background noise can mimic tonal/harmonic structure and fool simple filters -- the same confound is visible here in an actual trained model, not just the earlier diagnostic filters.

### Interpretation

The low recall numbers are, at least in part, a measurement artifact rather than a model failure -- the rooster example makes a concrete case for this rather than asserting it. The real, model-level issue worth carrying forward is the **cat/background confusion**, which was already predicted by the EDA conv-filter findings before any model existed. Both are worth raising directly in the "where does your system fail, and why" section of the presentation, backed by an actual example plot rather than a guess.

### Files

- `training_curve.png`, `overfit_check.png` -- training diagnostics from step 4/5
- `sequential_model.pt` -- trained weights (6 epochs)
- `frame_level_metrics.txt` -- the table above, machine-readable
- `qualitative/*.png` -- 6 prediction-overlay plots, one per class including background
- `src/sequential_model.py`, `src/train_sequential.py`, `src/evaluate_sequential.py` -- reusable; rerun `train_sequential.py` with more epochs to push past this baseline
