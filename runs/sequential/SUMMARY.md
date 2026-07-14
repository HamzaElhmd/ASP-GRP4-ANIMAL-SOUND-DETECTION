## Sequential model (GRU) -- architecture, evaluation, and qualitative findings

### Status: v1, not a final result

This checkpoint (`sequential_model.pt`) was trained for 6 epochs on CPU, and the loss curve (`training_curve.png`) was still declining at epoch 6 -- it had not converged. Treat every number in this document as a first working baseline, not a tuned final result. Better numbers are expected from further training; if you're using this checkpoint to build or compare against, don't treat today's metrics as the ceiling of what this architecture can do.

### Input contract -- what you must feed the model

Before calling `SequentialEventDetector`, audio must be: mono, 16kHz sample rate (matching `TARGET_SR` in `src/preprocessor.py`), converted to a mel-spectrogram with `n_fft=512, hop_length=160, win_length=400, n_mels=64` (the constants `N_FFT/HOP_LENGTH/WIN_LENGTH/N_MELS` in `src/eda.py`), in dB scale (`torchaudio.transforms.AmplitudeToDB`), and shaped `(batch, time, 64)` -- time-major, not the `(n_mels, time)` shape `torchaudio` returns by default, so transpose after computing the spectrogram. `compute_feature_sequence_from_waveform()` in `src/sequential_data.py` does all of this for you; call that rather than reimplementing it, so there's exactly one place this logic lives.

Output is raw logits, shape `(batch, time, 5)`, one per animal in this fixed order: `cat, cow, dog, rooster, sheep` (`ANIMAL_CLASSES` in `src/sequential_data.py`). Apply `torch.sigmoid()` yourself to get probabilities -- the model doesn't do this internally, since training uses `BCEWithLogitsLoss` directly on the logits for numerical stability.

### Continuous-audio test (not a pre-cut clip)

Every other test in this document runs the model on fixed 2-second training-format clips. This one runs it on a real, continuous, 3.5-minute unedited recording it has never seen the like of during training, using the shared scanner (`inference_time_windowing()` in `src/preprocessor.py`, 1s window / 250ms hop) -- see `src/predict_continuous.py`.

Tested against 15 spans in that recording that were manually confirmed by ear during the multi-label EDA (`eda_outputs/multilabel_sources/candidate_events.csv`), including the one confirmed dog+rooster overlap in the whole corpus.

Results, from `runs/sequential/continuous_test/zoomed_95_155.png` (regenerated after the retrain and the max-pooling fix below -- these numbers supersede the first version of this test):

- **Rooster stays weak even on its own confirmed spans, including the dog+rooster overlap (143.75-148.5s).** Cat and dog both dominate throughout the 95-155s window; rooster's probability never clearly separates from its baseline noise, even during the 6 confirmed rooster-labeled spans in this range. This is a real, consistent weakness -- it shows up the same way in the frame-level test-set numbers (rooster recall 0.092) and in the qualitative rooster plot below, not just here.
- **Cat and dog are both over-active across the whole window**, including stretches with no confirmed animal at all -- the same false-positive pattern found repeatedly since the EDA conv-filter findings, now visible at the level of an actual continuous real-world scan.
- This pattern held across **three** independently-trained checkpoints now (the original, the retrain after the split fix, and a second retrain after a stress-test rerun) -- not a fluke of one run. Rooster under-detection and cat/dog over-triggering look like a structural issue with this architecture/training recipe (6 epochs, no imbalance correction applied by default), not noise from one unlucky initialization.

Worth rerunning this same test once a longer or imbalance-corrected training run exists, to see whether more training or the calibration approaches below close this gap.

### Architecture and why

`SequentialEventDetector` (`src/sequential_model.py`): a single-layer GRU (hidden size 64) over a mel-spectrogram time sequence, followed by a per-timestep linear layer producing 5 raw logits (cat, cow, dog, rooster, sheep), trained with `BCEWithLogitsLoss` -- independent sigmoid per class per timestep, not softmax, so overlapping animals aren't structurally excluded, same as every track in this project.

**GRU/LSTM over a vanilla RNN**: a vanilla RNN's gradient signal decays multiplicatively at every timestep it's backpropagated through, so over a ~200-timestep sequence (2s at this hop length) it effectively can't learn dependencies that span more than a few steps -- the vanishing gradient problem covered in class. GRU (and LSTM) use gated connections that let gradients flow through many timesteps largely unchanged, which is the entire reason to pick either over a vanilla RNN for a sequence this long. GRU was chosen over LSTM specifically for this project mainly for practicality: fewer parameters, faster to train, and the dataset here is small enough that LSTM's extra gate (and extra capacity) is unlikely to be necessary.

**Why this track can target the "gap in the middle of a continuous event" problem the spec calls out**: a frame-independent classifier (Person A's track) makes each prediction from that frame's features alone, so a brief dip in energy mid-bark can flip the prediction to "not dog" for an isolated frame even though the event is clearly still ongoing to a human listener. A recurrent model's prediction at each timestep is a function of everything the network has seen in the sequence so far (and, implicitly, the pattern it's learned about what typically follows), so it has the structural capacity to smooth over a brief dip rather than treating it as a hard boundary. The dog qualitative example below shows this already starting to happen even at this early training stage.

### Split methodology

`processed/split.csv` is train/val/test (70/15/15), not train/val -- source clips are assigned once per (label, source_file) so every frame/augmented-frame derived from a clip stays in the same split (`src/split.py`). This was originally a 2-way split; changed to 3-way after comparing notes with the `hidden-markov-model` branch, which correctly keeps val (for tuning decisions like threshold calibration) separate from test (untouched until final reporting). A 2-way split that uses the same data for both tuning and the final reported number is optimistic -- the numbers below are now from the test split specifically, which nothing here has been tuned against.

### Frame-level evaluation (test set, 1,152 clips)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cat | 0.694 | 0.561 | 0.621 | 75,375 |
| cow | 0.695 | 0.335 | 0.452 | 25,326 |
| dog | 0.783 | 0.593 | 0.675 | 57,888 |
| rooster | 0.792 | 0.092 | 0.164 | 21,708 |
| sheep | 0.569 | 0.300 | 0.393 | 18,693 |
| **macro avg** | **0.706** | **0.376** | **0.461** | |

**This required two retrains, not just a re-measurement.** After building the split, a check found that 138 of the 138 clips now in test had previously been in the *train* set used for the original checkpoint -- meaning the first "test set" evaluation was actually measuring the model partly on data it had already seen, which is not a valid held-out result. Retrained fresh on the corrected train partition before reporting anything. Separately, a stress-test smoke-testing the new `use_pos_weight` option accidentally overwrote that checkpoint (both save to the same default path) -- retrained a second time to restore a clean, non-reweighted baseline, since the smoke test itself wasn't meant to produce the checkpoint being reported here. Training isn't seeded, so each retrain has a slightly different random init -- the numbers above are from the actual final checkpoint on disk, not an earlier run. All three checkpoints trained so far (original, first retrain, second retrain) show the same qualitative pattern (rooster weak, cat/dog over-triggering), so this looks like a property of the recipe, not one unlucky run.

### Qualitative findings (predictions overlaid on spectrogram)

Regenerated from the test split after the retrain -- these are different specific clips than earlier versions of this doc referenced, since the split fix moved which clips are in test. Three checked by eye, not just the aggregate numbers:

- **Rooster** (`rooster_5-200334-B-1_frame_000.png`): despite the true label being rooster, rooster's own predicted probability stays low through nearly the whole clip -- cat and dog are both higher for stretches of it instead. This matches the frame-level numbers directly (rooster recall is only 0.092 on the clean test set) -- rooster is the model's weakest class, and this plot shows why: it isn't just scoring low, it's actively being out-competed by other classes on rooster's own clip.
- **Dog** (`dog_1-59513-A-0_frame_000.png`): dog probability rises sharply at the first bark burst and stays elevated (0.6-0.95) across the whole clip, dipping and recovering between each of the three visible bursts in the spectrogram rather than flattening out -- the same "smooth over the gap between bursts" behavior found before, still holding after the retrain. Consistent with dog remaining the strongest class (F1 0.675, best of the five).
- **Background** (`background_3-163607-A-13_frame_000.png`): cat and dog both show noisy, repeated activity on a clip that should have all five near zero -- the same false-positive pattern flagged since the EDA conv-filter findings, still present after retraining. This is a model-level issue, not something the split fix or retrain addresses.

### Interpretation

Rooster's poor recall is a genuine model weakness, confirmed two independent ways (frame-level test metrics and the continuous real-audio scan) rather than a labeling or scoring artifact. The **cat/dog over-triggering and rooster under-detection** are the two real, model-level issues worth carrying forward -- both were flagged before any model existed (the EDA conv-filter pass predicted background could be confused for cat; the class-imbalance section below explains why rooster specifically is weak). Worth raising directly in the "where does your system fail, and why" section of the presentation, backed by actual example plots rather than a guess.

### Class imbalance: what was tried

Train split is imbalanced (cat has the most source clips, rooster and sheep the fewest -- see `processed/split.csv`), and it's the most likely reason rooster is the weakest class. Three things tried:

**Per-class threshold calibration on val** (`calibrate_thresholds()` in `src/evaluate_sequential.py`) -- no retraining, just picks each class's best decision threshold instead of a flat 0.5. Real improvement on every class (e.g. rooster's F1 goes from 0.227 to 0.454 at its own calibrated threshold), but calibrating on the clean val set doesn't always transfer to real audio -- see the next point.

**Per-class threshold calibration on the real farm recording** (`src/calibrate_real_world.py`), using the manually-confirmed labels from the multi-label EDA instead of val clips. This matters because a threshold that helps on clean isolated clips can hurt on real audio: cat needs a much stricter threshold on real audio (0.85) than val calibration would suggest, and even at its own best real-world threshold cat's F1 only reaches 0.070 (vs. 0.028 at flat 0.5) -- real-world cat detection is weak regardless of threshold, the problem is the model's confidence, not just where the cutoff sits. Rooster, by contrast, improves sharply on real audio with a much looser threshold (0.20 vs 0.5 gives F1 0.673) -- consistent with the frame-level finding that rooster is under-confident, not over-confident. Calibrating directly against real audio instead of val is the more trustworthy approach for anything meant to run on unseen recordings.

**Loss reweighting and balanced sampling** (`compute_class_pos_weight()`, `build_balanced_sampler()` in `src/train_sequential.py`, both optional flags on `full_training_run()`) -- upweights rare classes in the loss, or oversamples them per batch. Tried in isolation and combined; neither clearly beat the plain baseline within 6 epochs, and combining both at full strength overcorrected (recall shot up, precision collapsed). Kept as available options since the idea is sound, but not turned on by default -- they need more training time than tested here to prove out, and per-class threshold calibration was the more reliable win for the time spent.

### Files

- `training_curve.png`, `overfit_check.png` -- training diagnostics
- `sequential_model.pt` -- trained weights (6 epochs, current split)
- `frame_level_metrics.txt` -- the table above, machine-readable
- `qualitative/*.png` -- 6 prediction-overlay plots, one per class including background, from the test split
- `continuous_test/zoomed_95_155.png` -- the real-audio scan referenced above
- `src/sequential_model.py`, `src/train_sequential.py`, `src/evaluate_sequential.py`, `src/calibrate_real_world.py`, `src/split.py` -- reusable; rerun `train_sequential.py` with more epochs, or with `use_pos_weight`/`use_balanced_sampler`, to push past this baseline
