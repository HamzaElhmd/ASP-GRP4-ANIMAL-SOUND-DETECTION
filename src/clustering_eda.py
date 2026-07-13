from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from eda import CLASSES, N_FFT, N_MELS, N_MFCC, HOP_LENGTH, WIN_LENGTH, PROCESSED_CSV
except ModuleNotFoundError:
    from src.eda import CLASSES, N_FFT, N_MELS, N_MFCC, HOP_LENGTH, WIN_LENGTH, PROCESSED_CSV

OUTPUT_DIR = Path("eda_outputs/clustering")
SAMPLES_PER_CLASS = 200
RANDOM_SEED = 42


def load_all_frames(csv_path: Path = PROCESSED_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def sample_for_clustering(df: pd.DataFrame, n_per_class: int = SAMPLES_PER_CLASS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    parts = []
    for label in CLASSES:
        class_rows = df[df["label"] == label]
        n = min(n_per_class, len(class_rows))
        parts.append(class_rows.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def mfcc_summary_vector(filepath: str) -> np.ndarray:
    waveform, sample_rate = torchaudio.load(str(filepath))
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
            "n_mels": N_MELS,
        },
    )
    mfcc = mfcc_transform(waveform).squeeze(0)  # (n_mfcc, time)
    mean = mfcc.mean(dim=1)
    std = mfcc.std(dim=1)
    return torch.cat([mean, std]).numpy()


def build_feature_matrix(sampled_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for _, row in sampled_df.iterrows():
        vec = mfcc_summary_vector(row["filepath"])
        features.append(vec)
        labels.append(row["label"])
    return np.stack(features), np.array(labels)


def purity_score(true_labels: np.ndarray, cluster_ids: np.ndarray) -> float:
    total = 0
    for cluster_id in np.unique(cluster_ids):
        mask = cluster_ids == cluster_id
        if mask.sum() == 0:
            continue
        values, counts = np.unique(true_labels[mask], return_counts=True)
        total += counts.max()
    return total / len(true_labels)


def contingency_table(true_labels: np.ndarray, cluster_ids: np.ndarray) -> pd.DataFrame:
    return pd.crosstab(pd.Series(true_labels, name="true_label"), pd.Series(cluster_ids, name="cluster"))


def run_clustering() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = load_all_frames()
    sampled = sample_for_clustering(all_frames)
    print(f"sampled {len(sampled)} frames across {len(CLASSES)} classes")

    X, y = build_feature_matrix(sampled)
    print("feature matrix shape:", X.shape)

    X_scaled = StandardScaler().fit_transform(X)

    n_clusters = len(CLASSES)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10).fit(X_scaled)
    gmm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_SEED).fit(X_scaled)
    gmm_labels = gmm.predict(X_scaled)

    kmeans_ari = adjusted_rand_score(y, kmeans.labels_)
    kmeans_nmi = normalized_mutual_info_score(y, kmeans.labels_)
    kmeans_purity = purity_score(y, kmeans.labels_)

    gmm_ari = adjusted_rand_score(y, gmm_labels)
    gmm_nmi = normalized_mutual_info_score(y, gmm_labels)
    gmm_purity = purity_score(y, gmm_labels)

    print("\nK-Means: ARI={:.3f} NMI={:.3f} purity={:.3f}".format(kmeans_ari, kmeans_nmi, kmeans_purity))
    print("GMM:     ARI={:.3f} NMI={:.3f} purity={:.3f}".format(gmm_ari, gmm_nmi, gmm_purity))

    kmeans_table = contingency_table(y, kmeans.labels_)
    gmm_table = contingency_table(y, gmm_labels)
    kmeans_table.to_csv(OUTPUT_DIR / "kmeans_contingency.csv")
    gmm_table.to_csv(OUTPUT_DIR / "gmm_contingency.csv")
    print("\nK-Means contingency table (rows=true label, cols=cluster id):")
    print(kmeans_table)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    print(f"\nPCA explained variance ratio (2 components): {pca.explained_variance_ratio_.sum():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for label in CLASSES:
        mask = y == label
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.5, s=12)
    axes[0].set_title("PCA projection, colored by true label")
    axes[0].legend()

    scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap="tab10", alpha=0.5, s=12)
    axes[1].set_title("PCA projection, colored by K-Means cluster")
    fig.colorbar(scatter, ax=axes[1])

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_scatter.png")
    plt.close(fig)
    print(f"\nsaved {OUTPUT_DIR / 'pca_scatter.png'}")

    with open(OUTPUT_DIR / "metrics.txt", "w") as f:
        f.write(f"samples: {len(sampled)}, feature dim: {X.shape[1]}\n")
        f.write(f"K-Means: ARI={kmeans_ari:.4f} NMI={kmeans_nmi:.4f} purity={kmeans_purity:.4f}\n")
        f.write(f"GMM:     ARI={gmm_ari:.4f} NMI={gmm_nmi:.4f} purity={gmm_purity:.4f}\n")
        f.write(f"PCA 2-component explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}\n")


if __name__ == "__main__":
    run_clustering()
