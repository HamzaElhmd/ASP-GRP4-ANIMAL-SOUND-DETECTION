## Multi-label / co-occurrence EDA

### Source

A ~37 minute farm ambience recording, sourced from YouTube (see `sources.csv` for URL and download timestamp), resampled to mono/16kHz to match the project standard, split into 11 chunks of ~3.5 minutes each.

### Method

1. `src/multilabel_eda.py` scanned all 11 chunks with a 1s window / 250ms hop (reusing `inference_time_windowing()` from `src/preprocessor.py`) and flagged 210 candidate "something is happening" spans by RMS energy, merging adjacent active windows.
2. The 150 loudest candidate spans (of 210) were manually reviewed by ear and labeled with the animal(s) heard, logged in `results.txt` and merged into `candidate_events.csv`.
3. 60 lower-priority spans were left unreviewed -- the reviewed sample already gives a stable enough picture, see Results.

### Results

Of 150 reviewed spans:

| Category | Count |
|---|---:|
| Unidentifiable (wind, ambience, unclear) | 38 |
| Single animal | 109 |
| Two animals overlapping | 3 |

Animal frequency (spans each animal was heard in):

| Animal | Spans |
|---|---:|
| rooster | 55 |
| sheep | 31 |
| cow | 20 |
| dog | 5 |
| cat | 4 |

Overlap rate among identified spans: **3 / 112 = 2.7%**

Confirmed overlapping pairs:

| Pair | Count | Example |
|---|---:|---|
| dog + rooster | 2 | `..._part_01.wav` 143.75s-148.5s; `..._part_07.wav` 60.0s-64.75s |
| cow + dog | 1 | `..._part_05.wav` 117.25s-133.25s |

### Interpretation

Even in a single ~37-minute recording, multiple animals were confirmed vocalizing simultaneously (~2.7% of identifiable moments) -- rare, but real, and enough to demonstrate that a single-label-per-frame model would be structurally wrong for this task. This is the evidence behind the project's independent-sigmoid-per-class design (Section 3 of the spec) instead of softmax.

A ~2.7% overlap rate should be read as a lower bound, not a precise population estimate: it comes from one source video, only the loudest 150/210 candidate spans were reviewed, and the RMS-based candidate detector only flags spans above a fixed energy threshold, so quieter simultaneous events could be missed.

### Caveats

- **Single source.** All evidence comes from one YouTube video. Overlap rate and animal mix (rooster/sheep/cow-heavy, dog/cat rare) reflects this one recording's content, not a general farm distribution. Treat 2.7% as directional, not a hard population estimate.
- **Content nature confirmed genuine.** Despite the "relaxing farm animal sounds" title, manual listening confirmed this is a genuine field recording, not a produced/mixed ambience track. Safe to present as unedited field audio.
- **License/source not yet logged.** Per the same gap flagged for Person A's dataset, the `license_notes` column in `sources.csv` is still blank. Fill in before citing this recording in the final report.
- **Background co-occurring with a single animal was not counted as multi-label overlap** -- only two or more of the five target animal classes active at once counts, consistent with how the multi-label design question is framed in the spec.
- **38 "unidentifiable" spans** are excluded from the overlap rate denominator since they weren't confirmed as animal sounds at all.

### Files

- `candidate_events.csv` -- all 210 candidate spans, `animal_labels` filled for rows 1-150
- `results.txt` -- raw manual listening log (order matches CSV rows 1-150)
- `LISTENING_GUIDE.md` -- copy-paste playback commands used to generate `results.txt`
- `sources.csv` -- source URL, chunk length, download timestamp (license_notes still needs filling in)
- `*_part_*.wav` -- the 11 audio chunks (gitignored, not committed -- see `.gitignore`)
