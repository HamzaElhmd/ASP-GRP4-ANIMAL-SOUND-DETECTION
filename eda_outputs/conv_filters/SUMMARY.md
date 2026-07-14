## Convolutional filters on spectrograms

### Method

Three small fixed (non-learned) kernels applied to the mel spectrograms from the feature-visualization pass (`src/eda.py`), via a single `conv2d` pass over each spectrogram image:

- **`horizontal_edge`** (Sobel-y): responds to sharp changes across the frequency axis. Expected to highlight the top/bottom boundaries of sustained harmonic bands.
- **`vertical_edge`** (Sobel-x): responds to sharp changes across the time axis. Expected to highlight onsets/transients -- sudden bursts of energy.
- **`gabor_horizontal`**: a horizontally-oriented Gabor kernel (Gaussian-windowed cosine), meant to resonate with periodic horizontal texture, i.e. sustained tonal content.

Data used: the same 24 sampled mel spectrograms (4 per class x 6 classes) from `src/eda.py`'s `load_training_frames` / `sample_frames_per_class`, same seed, so the comparison is apples-to-apples.

**Fix applied along the way:** the display was initially auto-scaled to each filtered image's raw min/max, which let a single artifact dominate the color range -- clips shorter than the fixed 2s window are zero-padded at the tail (`frame_training_audio` in `src/preprocessor.py`), and the sharp silence boundary produced a large spike that washed out all the real structure elsewhere in the image. Fixed by clipping the display range to the 1st-99th percentile of each filtered image instead of raw min/max. Worth remembering if this pattern shows up again elsewhere: padding artifacts can dominate a naive min/max color scale.

### Observations

- **Rooster**: `horizontal_edge` sharpens the crow into 4-5 clean, flat, stable horizontal bands (harmonic stack) between mel bins ~20-50, clearly separated from background texture. This is the most visually distinctive result of the four classes checked.
- **Cat**: `horizontal_edge` also reveals harmonic structure, but it's a *wavering/curved* contour rather than flat lines -- tracks the meow's pitch glide. This is a useful distinguishing signal from rooster: both classes show harmonic banding under this filter, but rooster's bands are flat/stable while cat's are modulated.
- **Dog**: neither edge filter shows clean harmonic banding. `vertical_edge` instead highlights a broadband transient texture aligned with the two bark bursts visible in the original spectrogram -- consistent with a bark being a short noisy/broadband event rather than a tonal one.
- **Cow**: `horizontal_edge` shows some low-frequency banding (mel bins ~20-40) but less sharply defined than rooster's -- a mooing call has some harmonic content but it's noisier/broader than a crow.
- **Sheep**: `horizontal_edge` shows a wavering, modulated harmonic contour similar in character to the cat (bleat has a pitch glide), but noisier and centered lower (mel bins ~25-45). One of the two sheep clips checked was mostly quiet/low-energy in this particular frame -- amplitude varies a lot within the class, which is itself worth noting for framing/threshold decisions later.
- **Background**: this is the important caveat. Both background clips checked show real structure under these filters -- one has a persistent low-frequency drone (steady horizontal band around mel bin ~10-15, likely wind/machinery hum), the other shows strong regular periodic striping under `vertical_edge` and `gabor_horizontal` (likely crickets or another rhythmic ambient sound). **Background is not filter-flat/featureless** -- it can mimic both the tonal signature (drone -> looks like a weak harmonic band) and the transient signature (periodic chirping -> looks like repeated onsets) that we're using to distinguish animal classes. This is a real confound, not just a theoretical one.
- `gabor_horizontal` produced a similar qualitative signal to `horizontal_edge` across all classes checked (unsurprising, both are tuned to horizontal/tonal structure) -- didn't add a clearly distinct signal beyond the Sobel filter in this sample, though it may behave differently at other wavelength/sigma settings.

### Interpretation

This supports treating **harmonic-band presence/stability** as a discriminative feature between vocal/tonal classes (rooster, cat, sheep, and to a lesser extent cow) versus **broadband transient energy** for percussive classes (dog bark). That intuition is useful evidence for Section 3 model choice: it's consistent with mel-spectrogram input being informative for a 2D CNN, since these are exactly the kind of local time-frequency patterns 2D convolutional kernels are suited to pick up.

The background finding is the more important takeaway, though: simple fixed filters alone can't cleanly separate "animal" from "not animal," since some background noise produces filter responses that superficially resemble both a harmonic band and a transient burst. That's evidence *for* needing a learned model rather than a handcrafted-filter rule, and it's a legitimate point to raise in the "where might the system fail" part of the presentation -- background clips with tonal or rhythmic structure (engine hum, insects, wind through structures) are the likely confusion cases.

All 6 classes now checked (1-2 representative clips each, out of 4 sampled per class). This is still a spot-check, not exhaustive -- worth keeping in mind that only ~1/4 of the 24 saved plots were closely eyeballed; the rest are available if a specific class needs deeper review later.

### Files

- `*.png` -- one 4-panel plot per sampled clip (original + 3 filtered views), 24 total
- `src/conv_filters_eda.py` -- the filter implementation, reusable if you want to try different kernel parameters (size/sigma/wavelength for the Gabor kernel are exposed as arguments)
