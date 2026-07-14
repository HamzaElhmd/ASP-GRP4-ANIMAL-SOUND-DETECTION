## Unsupervised clustering sanity check

### Method

- **Data**: 200 frames sampled per class (1,200 total) from the full `processed/farmyard.csv` manifest, including both `training_frame` and `augmentation_frame` rows -- unlike the feature-visualization pass, augmented frames are included here for statistical volume, especially for rooster (80 raw -> 720 processed) and sheep (83 raw -> 666 processed).
- **Features**: each 2-second frame reduced to a 26-dim vector -- mean and standard deviation of each of the 13 MFCC coefficients across the frame's time axis. This is a common lightweight "clip summary" feature, not the full time-varying MFCC image.
- **Algorithms**: K-Means and Gaussian Mixture Model, both with 6 components (matching the 6 known classes), after standardizing features.
- **Metrics**: Adjusted Rand Index (ARI, 0=random, 1=perfect match to true labels), Normalized Mutual Info (NMI), and purity (fraction of each cluster's most common true label).

### Results

| Algorithm | ARI | NMI | Purity |
|---|---:|---:|---:|
| K-Means | 0.040 | 0.074 | 0.299 |
| GMM | 0.041 | 0.068 | 0.292 |

For reference, a single "put everyone in one cluster" baseline would score 0.167 purity (1/6 classes) here, so both algorithms did better than that floor, but nowhere close to clean separation (which would be near 1.0 on all three metrics).

The K-Means contingency table (true label vs. assigned cluster) confirms this quantitatively -- every true class is spread across most of the 6 clusters instead of concentrating in one:

```
cluster       0   1   2   3   4   5
true_label
background   96  34   7  17  40   6
cat         100  21   2  51  21   5
cow          34  68   7  32  57   2
dog          41  41   5  72  34   7
rooster      45  17  26  31  45  36
sheep        65  47   0  39  49   0
```

The PCA scatter plot (`pca_scatter.png`) shows the same thing visually: colored by true label, the 6 classes form one overlapping blob with no visible separation. Colored by K-Means cluster, the algorithm did find *some* structure, but it splits along an axis that doesn't correspond to animal identity.

### Interpretation

**Clusters do not separate cleanly by animal class using simple mean/std-pooled MFCC features.** This is a legitimate, useful negative result, not a bug -- the spec anticipates this outcome ("if clusters don't separate cleanly, that's useful evidence for your writeup regardless of which model track you pick").

Most likely causes:

1. **Time-averaging destroys the temporal pattern that actually distinguishes classes.** The conv-filter pass showed that what separates a rooster from a dog is a *temporal/structural* pattern -- flat sustained bands vs. sharp isolated bursts -- not the average spectral content over 2 seconds. Averaging over time collapses exactly the information that mattered.
2. **Most of a 2-second frame is quiet even for animal-labeled clips** (established during the feature-visualization pass, where RMS often shows short bursts inside mostly-quiet frames). Averaging in a lot of near-silent time dilutes the vocalization's signature with background noise, pulling different classes toward a similar quiet-frame baseline.
3. **26 dimensions summarizing 2 seconds is coarse.** MFCC mean/std per clip is a common lightweight feature for tasks like speaker identification over short segments, but it isn't built to capture the burst-vs-sustained distinction this project depends on.

### Implication for model choice

This is direct evidence *against* relying on simple time-averaged features or distance-based methods for this task, and *for* the project's emphasis on time-aware representations:

- Supports using the full mel-spectrogram (2D, time-preserved) as CNN input rather than a pooled summary vector, consistent with the conv-filter findings.
- Supports Person C's sequential/RNN track, which explicitly keeps frame-order information a pooled feature throws away.
- For Person A's classical ML track, this suggests frame-level (not whole-clip-averaged) MFCC/RMS/ZCR features will likely be necessary -- a single pooled vector per clip is probably too coarse, per this result.

### Files

- `pca_scatter.png` -- 2D PCA projection, true labels vs. K-Means clusters side by side
- `kmeans_contingency.csv`, `gmm_contingency.csv` -- full true-label x cluster tables
- `metrics.txt` -- ARI/NMI/purity for both algorithms
- `src/clustering_eda.py` -- reusable; sample size, feature choice (mean/std vs. full time series), and number of clusters are all easy to adjust if you want to re-run with different assumptions
