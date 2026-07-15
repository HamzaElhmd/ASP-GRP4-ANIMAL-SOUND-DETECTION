# Post-Processing & Event Formation

## Scope

Section 4 deliverable, implemented in `src/postprocessing.py`,
`src/scene_synthesis.py`, and `src/evaluation.py`.

This section owns two things:

1. turning any model's per-frame probabilities into discrete **events**, and
2. the **shared evaluation harness** that scores every model track (Classical ML /
   CNN / Sequential) the same way, so the comparison is fair.

It consumes the preprocessing (Section 1) and the mel-spectrogram representation
(Section 2), and it is **model-agnostic**: it does not contain any model. Each
model track plugs in through a single predict function

    predict_fn(waveform, sample_rate) -> (probabilities[T, C], frame_times[T], energy_db[T])

so the RandomForest, CNN, and Sequential models are all scored by the same code.

## Frame -> event conversion

At inference a model slides the 1 s window (0.25 s hop, Hann) from
`inference_time_windowing()` and produces a per-class probability timeline.
`timeline_to_events()` in `src/postprocessing.py` converts it to events, in order:

1. **Median smoothing** (`median_frames = 3`) — removes single-frame flicker
   without shifting boundaries.
2. **Energy gate** (`energy_floor_db = 35`) — frames more than 35 dB below the
   recording's own 95th-percentile level are forced to silence. The threshold is
   relative to the clip, so it adapts to how loud the recording is and detects
   nothing in quiet passages.
3. **Background gate** (`background_threshold = 0.5`) — frames where the learned
   background head is confident are blocked from hosting an animal event. Unlike
   the energy gate this catches **loud** non-animal sound (machinery, voices),
   which is where a pure energy threshold fails.
4. **Hysteresis thresholding** (`threshold_on = 0.5`, `threshold_off = 0.3`) — an
   event starts at the high threshold and is sustained down to the low one. Two
   thresholds stop one wavering event from being chopped into many, and the onset
   is **backtracked** to where the probability first rose, correcting the
   late-onset bias of a windowed detector.
5. **Gap-merge** (`merge_gap_seconds = 0.3`) — same-class events less than 300 ms
   apart become one, so a brief mid-vocalisation dip does not split the event.
6. **Minimum-duration filter** (`min_duration_seconds = 0.3`) — events shorter
   than 300 ms are dropped as spurious.

Every value is a named field of `PostProcessingConfig`, and the post-processing is
**swept on the validation scenes** by `tune_thresholds()` rather than hand-set — so
the choices are justified by data, as the spec requires: the **per-class start
thresholds** are each kept at the value that maximises that class's event-based F1,
and the **minimum event duration** is kept at the value that maximises macro
event-F1 (which drops short, spurious false-positive events). The tuned config is
then used for the reported event scores and the model comparison.

### Background / silence handling

Frames that are gated out (energy or background) emit no event — this is where
most of a real recording lands. Animal events are reported through the five animal
heads; `background` is only emitted on request (`emit_background=True`), since the
JSON schema is for animal events (`event_start`, `event_end`, `animal`).

## Evaluation (the metric that matches the grading rule)

`src/evaluation.py` implements a DCASE-style event-based metric:

- **Event-based F1 with a +/- 500 ms collar** on both onset and offset — a
  predicted event is a true positive only if it matches an unused reference event
  of the same class within the collar. This is exactly the grading tolerance.
- **Segment-based (frame-level) F1** at 100 ms resolution — rewards partial
  temporal overlap and is a cheaper diagnostic.
- **Near-miss breakdown** — every reference event is labelled *correct*,
  *mistimed* (right animal, boundary off), *confused* (wrong animal), or *missed*,
  and unmatched predictions are *spurious*. This turns a single F1 into an
  actionable failure analysis.

### Measuring events without a labelled test recording

The processed clips are isolated, so there is no continuous audio with onset/offset
labels to score against. `src/scene_synthesis.py` builds continuous ~1-minute
scenes from **fold-5 hold-out** clips with known event times (same-class overlaps
merged, since by definition they are one event, and overlap up to a cap is allowed
because that is the polyphonic case the project is about). No model ever trains on
fold 5, so the event-level benchmark is leakage-free.

## How the model tracks plug in

`evaluate_on_scenes(predict_fn, ...)` takes any model's predict function and runs
it through predict -> post-process -> event scoring, averaging the metrics across
scenes. Because the three tracks share this one function signature, the
three-model comparison table (Classical ML vs CNN vs Sequential, event-based
P/R/F1 with the +/- 500 ms collar plus frame-based P/R/F1) is produced by the same
code for all of them — which is what makes it apples-to-apples.

## Files

- `src/postprocessing.py` — `timeline_to_events`, gates, hysteresis, `events_to_records`
- `src/scene_synthesis.py` — labelled evaluation scenes + model-agnostic `evaluate_on_scenes`
- `src/evaluation.py` — event-based (+/- 500 ms) and segment-based metrics, near-miss breakdown
