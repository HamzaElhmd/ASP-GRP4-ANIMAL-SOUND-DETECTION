from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None


TARGET_SR = 16000
FEATURE_HOP_SECONDS = 0.01
INFERENCE_WINDOW_SECONDS = 1.0
INFERENCE_HOP_SECONDS = 0.25
CLASSES = ["dog", "cat", "sheep", "cow", "rooster", "background"]
ACTIVE_STATE = 1
INACTIVE_STATE = 0
EPS = 1e-8


def _load_audio(path: Path, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    if torchaudio is not None:
        audio_t, sr = torchaudio.load(str(path))
        audio = audio_t.mean(dim=0).cpu().numpy().astype(np.float32)
    else:
        sr, audio = wavfile.read(str(path))
        audio = np.asarray(audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if np.issubdtype(audio.dtype, np.integer):
            max_val = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    if sr != target_sr and audio.size:
        gcd = math.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return audio, sr


def _frame_signal(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if audio.size == 0:
        return np.zeros((1, frame_length), dtype=np.float32)
    if audio.shape[0] < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.shape[0]))
    n_frames = 1 + max(0, (audio.shape[0] - frame_length) // hop_length)
    frames = []
    for idx in range(n_frames):
        start = idx * hop_length
        end = start + frame_length
        frame = audio[start:end]
        if frame.shape[0] < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.shape[0]))
        frames.append(frame)
    return np.stack(frames, axis=0)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 26, fmin: int = 0, fmax: Optional[int] = None) -> np.ndarray:
    fmax = fmax or sr // 2
    mels = np.linspace(_hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        left = max(left, 0)
        right = min(right, n_fft // 2)
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


def _delta(features: np.ndarray, width: int = 2) -> np.ndarray:
    if features.shape[0] == 1:
        return np.zeros_like(features)
    denom = 2 * sum(i * i for i in range(1, width + 1))
    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    out = np.zeros_like(features)
    for t in range(features.shape[0]):
        acc = np.zeros(features.shape[1], dtype=np.float32)
        for i in range(1, width + 1):
            acc += i * (padded[t + width + i] - padded[t + width - i])
        out[t] = acc / denom
    return out


def extract_mfcc_features(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 26,
) -> np.ndarray:
    frames = _frame_signal(audio, win_length, hop_length)
    window = np.hanning(win_length).astype(np.float32)
    frames = frames * window[None, :]
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    fb = _mel_filterbank(sr, n_fft, n_mels=n_mels)
    mel = np.dot(spec, fb.T)
    mel = np.log(np.maximum(mel, EPS))
    mfcc = dct(mel, type=2, norm="ortho", axis=1)[:, :n_mfcc]
    delta = _delta(mfcc)
    delta2 = _delta(delta)
    return np.concatenate([mfcc, delta, delta2], axis=1).astype(np.float32)


def load_manifest(processed_root: Path = Path("processed")) -> pd.DataFrame:
    csv_path = processed_root / "farmyard.csv"
    if csv_path.exists():
        manifest = pd.read_csv(csv_path)
        if "filepath" in manifest.columns:
            manifest["filepath"] = manifest["filepath"].astype(str)
        if "source_file" not in manifest.columns:
            manifest["source_file"] = manifest["filepath"].map(lambda p: Path(p).stem)
        return manifest

    rows = []
    for label_dir in processed_root.iterdir():
        if not label_dir.is_dir():
            continue
        for wav in label_dir.glob("*.wav"):
            rows.append(
                {
                    "filepath": str(wav),
                    "label": label_dir.name,
                    "source_file": wav.stem.split("_frame_")[0],
                }
            )
    if not rows:
        raise FileNotFoundError("No processed clips found.")
    return pd.DataFrame(rows)


def recording_level_split(
    manifest: pd.DataFrame,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 13,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    test_idx = []
    for label, grp in manifest.groupby("label"):
        recs = grp["source_file"].fillna(grp["filepath"]).astype(str).unique().tolist()
        rng.shuffle(recs)
        n = len(recs)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train_recs = set(recs[:n_train])
        val_recs = set(recs[n_train : n_train + n_val])
        test_recs = set(recs[n_train + n_val :])
        train_idx.extend(grp[grp["source_file"].isin(train_recs)].index.tolist())
        val_idx.extend(grp[grp["source_file"].isin(val_recs)].index.tolist())
        test_idx.extend(grp[grp["source_file"].isin(test_recs)].index.tolist())
    return manifest.loc[train_idx].copy(), manifest.loc[val_idx].copy(), manifest.loc[test_idx].copy()


@dataclass
class BinaryHMMConfig:
    n_states: int = 2
    n_components: int = 2
    n_iter: int = 25
    tol: float = 1e-3
    covariance_floor: float = 1e-3
    random_state: int = 13


class BinaryGMMHMM:
    def __init__(self, config: BinaryHMMConfig):
        self.config = config
        self.n_states = config.n_states
        self.n_components = config.n_components
        self.random_state = np.random.default_rng(config.random_state)
        self.startprob_ = np.array([0.95, 0.05], dtype=np.float64)
        self.transmat_ = np.array([[0.97, 0.03], [0.08, 0.92]], dtype=np.float64)
        self.weights_ = None
        self.means_ = None
        self.covars_ = None

    def _log_gaussian(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        cov = np.maximum(cov, self.config.covariance_floor)
        diff = X - mean
        return -0.5 * (
            np.sum(np.log(2.0 * np.pi * cov))
            + np.sum((diff * diff) / cov, axis=1)
        )

    def _log_mix_emission(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_states), dtype=np.float64)
        for s in range(self.n_states):
            comp = []
            for k in range(self.n_components):
                lp = self._log_gaussian(X, self.means_[s, k], self.covars_[s, k]) + np.log(self.weights_[s, k] + EPS)
                comp.append(lp)
            comp = np.vstack(comp)
            max_lp = np.max(comp, axis=0)
            log_prob[:, s] = max_lp + np.log(np.sum(np.exp(comp - max_lp), axis=0) + EPS)
        return log_prob

    def _forward_backward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        log_emit = self._log_mix_emission(X)
        log_start = np.log(self.startprob_ + EPS)
        log_trans = np.log(self.transmat_ + EPS)
        T = X.shape[0]
        alpha = np.zeros((T, self.n_states), dtype=np.float64)
        scale = np.zeros(T, dtype=np.float64)
        alpha[0] = log_start + log_emit[0]
        scale[0] = np.logaddexp.reduce(alpha[0])
        alpha[0] -= scale[0]
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.logaddexp.reduce(alpha[t - 1] + log_trans[:, j]) + log_emit[t, j]
            scale[t] = np.logaddexp.reduce(alpha[t])
            alpha[t] -= scale[t]
        beta = np.zeros((T, self.n_states), dtype=np.float64)
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.logaddexp.reduce(log_trans[i] + log_emit[t + 1] + beta[t + 1]) - scale[t + 1]
        gamma = np.exp(alpha + beta)
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), EPS)
        xi = np.zeros((T - 1, self.n_states, self.n_states), dtype=np.float64)
        for t in range(T - 1):
            m = (
                alpha[t][:, None]
                + log_trans
                + log_emit[t + 1][None, :]
                + beta[t + 1][None, :]
            )
            m -= np.logaddexp.reduce(m.ravel())
            xi[t] = np.exp(m)
        return gamma, xi, float(np.sum(scale))

    def _init_params(self, X: np.ndarray) -> None:
        n_features = X.shape[1]
        self.weights_ = np.full((self.n_states, self.n_components), 1.0 / self.n_components, dtype=np.float64)
        self.means_ = np.zeros((self.n_states, self.n_components, n_features), dtype=np.float64)
        self.covars_ = np.zeros((self.n_states, self.n_components, n_features), dtype=np.float64)
        overall_mean = X.mean(axis=0)
        overall_var = X.var(axis=0) + self.config.covariance_floor
        for s in range(self.n_states):
            for k in range(self.n_components):
                jitter = self.random_state.normal(scale=0.1, size=n_features)
                self.means_[s, k] = overall_mean + jitter
                self.covars_[s, k] = overall_var.copy()

    def fit(self, sequences: Sequence[np.ndarray]) -> "BinaryGMMHMM":
        X = np.concatenate(sequences, axis=0)
        self._init_params(X)
        last_ll = -np.inf
        for _ in range(self.config.n_iter):
            start_acc = np.zeros(self.n_states, dtype=np.float64)
            trans_acc = np.zeros((self.n_states, self.n_states), dtype=np.float64)
            gamma_acc = np.zeros(self.n_states, dtype=np.float64)
            comp_gamma = np.zeros((self.n_states, self.n_components), dtype=np.float64)
            mean_num = np.zeros_like(self.means_, dtype=np.float64)
            cov_num = np.zeros_like(self.covars_, dtype=np.float64)
            ll = 0.0
            for seq in sequences:
                gamma, xi, seq_ll = self._forward_backward(seq)
                ll += seq_ll
                start_acc += gamma[0]
                trans_acc += xi.sum(axis=0)
                gamma_acc += gamma.sum(axis=0)
                for s in range(self.n_states):
                    for k in range(self.n_components):
                        resp = gamma[:, s]
                        comp_gamma[s, k] += resp.sum() / self.n_components
                        mean_num[s, k] += (resp[:, None] * seq).sum(axis=0) / self.n_components
                        cov_num[s, k] += (resp[:, None] * (seq ** 2)).sum(axis=0) / self.n_components
            self.startprob_ = start_acc / np.maximum(start_acc.sum(), EPS)
            self.transmat_ = trans_acc / np.maximum(trans_acc.sum(axis=1, keepdims=True), EPS)
            for s in range(self.n_states):
                for k in range(self.n_components):
                    denom = max(comp_gamma[s, k], EPS)
                    self.weights_[s, k] = denom / max(gamma_acc[s], EPS)
                    mean = mean_num[s, k] / denom
                    var = cov_num[s, k] / denom - mean**2
                    self.means_[s, k] = mean
                    self.covars_[s, k] = np.maximum(var, self.config.covariance_floor)
            if abs(ll - last_ll) < self.config.tol:
                break
            last_ll = ll
        return self

    def score(self, X: np.ndarray) -> float:
        _, _, ll = self._forward_backward(X)
        return ll

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        log_emit = self._log_mix_emission(X)
        log_start = np.log(self.startprob_ + EPS)
        log_trans = np.log(self.transmat_ + EPS)
        T = X.shape[0]
        delta = np.zeros((T, self.n_states), dtype=np.float64)
        psi = np.zeros((T, self.n_states), dtype=np.int32)
        delta[0] = log_start + log_emit[0]
        for t in range(1, T):
            for j in range(self.n_states):
                vals = delta[t - 1] + log_trans[:, j]
                psi[t, j] = int(np.argmax(vals))
                delta[t, j] = np.max(vals) + log_emit[t, j]
        states = np.zeros(T, dtype=np.int32)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states


def _group_by_recording(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    groups = {}
    for key, grp in df.groupby(["label", "source_file"], dropna=False):
        groups[str(key)] = grp.copy()
    return groups


def _load_sequences(df: pd.DataFrame) -> List[np.ndarray]:
    sequences = []
    for _, row in df.iterrows():
        audio, sr = _load_audio(Path(row["filepath"]), TARGET_SR)
        features = extract_mfcc_features(audio, sr=sr)
        sequences.append(features)
    return sequences


def _fit_scaler(sequences: Sequence[np.ndarray]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.concatenate(sequences, axis=0))
    return scaler


def _transform_sequences(sequences: Sequence[np.ndarray], scaler: StandardScaler) -> List[np.ndarray]:
    return [scaler.transform(seq).astype(np.float32) for seq in sequences]


def _binary_event_segments(states: np.ndarray, frame_hop_seconds: float) -> List[Tuple[float, float]]:
    segments = []
    active = False
    start = 0.0
    for idx, state in enumerate(states):
        if state == ACTIVE_STATE and not active:
            active = True
            start = idx * frame_hop_seconds
        elif state != ACTIVE_STATE and active:
            end = idx * frame_hop_seconds
            segments.append((start, end))
            active = False
    if active:
        segments.append((start, len(states) * frame_hop_seconds))
    return segments


def _collapse_events(pred: pd.DataFrame) -> List[Dict[str, str]]:
    events = []
    for _, row in pred.iterrows():
        events.append(
            {
                "event_start": f"{float(row['event_start']):.3f}",
                "event_end": f"{float(row['event_end']):.3f}",
                "animal": str(row["animal"]),
            }
        )
    return events


def temporal_postprocess(probabilities: np.ndarray, threshold: float = 0.5, median_width: int = 5, gap_fill: int = 3) -> np.ndarray:
    active = (probabilities >= threshold).astype(np.int32)
    if median_width > 1:
        pad = median_width // 2
        padded = np.pad(active, (pad, pad), mode="edge")
        smooth = np.zeros_like(active)
        for i in range(active.shape[0]):
            smooth[i] = int(np.median(padded[i : i + median_width]))
        active = smooth
    if gap_fill > 0:
        i = 0
        while i < len(active):
            if active[i] == 0:
                j = i
                while j < len(active) and active[j] == 0:
                    j += 1
                if i > 0 and j < len(active) and (j - i) <= gap_fill:
                    active[i:j] = 1
                i = j
            else:
                i += 1
    return active


def event_based_scores(
    truth: Sequence[Tuple[float, float]],
    pred: Sequence[Tuple[float, float]],
    collar: float = 0.5,
) -> Dict[str, float]:
    matched_pred = set()
    tp = 0
    for gt_start, gt_end in truth:
        for idx, (pr_start, pr_end) in enumerate(pred):
            if idx in matched_pred:
                continue
            if abs(gt_start - pr_start) <= collar and abs(gt_end - pr_end) <= collar:
                tp += 1
                matched_pred.add(idx)
                break
    fp = len(pred) - tp
    fn = len(truth) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    return {"precision": precision, "recall": recall, "f1": f1}


def frame_based_scores(truth: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def train_hmm_suite(
    processed_root: Path = Path("processed"),
    output_dir: Path = Path("artifacts/hmm"),
    config: Optional[BinaryHMMConfig] = None,
) -> Dict[str, object]:
    config = config or BinaryHMMConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(processed_root)
    train_df, val_df, test_df = recording_level_split(manifest)

    train_sequences_by_label: Dict[str, List[np.ndarray]] = {}
    val_sequences_by_label: Dict[str, List[np.ndarray]] = {}
    test_sequences_by_label: Dict[str, List[np.ndarray]] = {}
    for label in CLASSES:
        train_sequences_by_label[label] = _load_sequences(train_df[train_df["label"] == label])
        val_sequences_by_label[label] = _load_sequences(val_df[val_df["label"] == label])
        test_sequences_by_label[label] = _load_sequences(test_df[test_df["label"] == label])

    scaler = _fit_scaler([seq for seqs in train_sequences_by_label.values() for seq in seqs])
    joblib.dump(scaler, output_dir / "feature_scaler.joblib")

    metrics = {"splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)}, "classes": {}}
    for label in CLASSES:
        train_seq = _transform_sequences(train_sequences_by_label[label], scaler)
        val_seq = _transform_sequences(val_sequences_by_label[label], scaler)
        test_seq = _transform_sequences(test_sequences_by_label[label], scaler)

        model = BinaryGMMHMM(config).fit(train_seq)
        joblib.dump(model, output_dir / f"{label}_hmm.joblib")

        val_scores = [model.score(seq) for seq in val_seq] if val_seq else []
        test_scores = [model.score(seq) for seq in test_seq] if test_seq else []
        metrics["classes"][label] = {
            "validation_log_likelihood_mean": float(np.mean(val_scores)) if val_scores else None,
            "test_log_likelihood_mean": float(np.mean(test_scores)) if test_scores else None,
        }

    metrics_path = output_dir / "training_diagnostics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def infer_continuous_file(
    wav_path: Path,
    model_dir: Path = Path("artifacts/hmm"),
    threshold: float = 0.5,
    median_width: int = 5,
    gap_fill_ms: int = 300,
    hop_seconds: float = FEATURE_HOP_SECONDS,
) -> List[Dict[str, str]]:
    scaler: StandardScaler = joblib.load(model_dir / "feature_scaler.joblib")
    models = {label: joblib.load(model_dir / f"{label}_hmm.joblib") for label in CLASSES}
    audio, sr = _load_audio(wav_path, TARGET_SR)
    window_samples = int(INFERENCE_WINDOW_SECONDS * sr)
    hop_samples = int(INFERENCE_HOP_SECONDS * sr)
    if audio.shape[0] < window_samples:
        audio = np.pad(audio, (0, window_samples - audio.shape[0]))
    n_windows = 1 + max(0, (audio.shape[0] - window_samples) // hop_samples)
    total_frames = int(math.ceil(audio.shape[0] / sr / hop_seconds))
    probs = {label: np.zeros(total_frames, dtype=np.float32) for label in CLASSES}
    window = np.hanning(window_samples).astype(np.float32)
    for w_idx in range(n_windows):
        start_sample = w_idx * hop_samples
        end_sample = start_sample + window_samples
        clip = audio[start_sample:end_sample]
        if clip.shape[0] < window_samples:
            clip = np.pad(clip, (0, window_samples - clip.shape[0]))
        clip = clip * window
        features = scaler.transform(extract_mfcc_features(clip, sr)).astype(np.float32)
        frame_offset = int(round(start_sample / sr / hop_seconds))
        for label, model in models.items():
            gamma, _, _ = model._forward_backward(features)
            active = gamma[:, ACTIVE_STATE]
            end_frame = min(frame_offset + len(active), total_frames)
            if end_frame > frame_offset:
                probs[label][frame_offset:end_frame] = np.maximum(
                    probs[label][frame_offset:end_frame],
                    active[: end_frame - frame_offset],
                )
    outputs = []
    for label in CLASSES:
        active = temporal_postprocess(
            probs[label],
            threshold=threshold,
            median_width=median_width,
            gap_fill=max(1, int(round((gap_fill_ms / 1000.0) / hop_seconds))),
        )
        segments = _binary_event_segments(active, hop_seconds)
        for start, end in segments:
            if end > start:
                outputs.append({"event_start": f"{start:.3f}", "event_end": f"{end:.3f}", "animal": label})
    outputs.sort(key=lambda x: (float(x["event_start"]), x["animal"]))
    return outputs


def evaluate_suite(
    model_dir: Path = Path("artifacts/hmm"),
    processed_root: Path = Path("processed"),
    output_json: Path = Path("artifacts/hmm/final_diagnostics.json"),
    collar_seconds: float = 0.5,
) -> Dict[str, object]:
    manifest = load_manifest(processed_root)
    _, _, test_df = recording_level_split(manifest)
    scaler: StandardScaler = joblib.load(model_dir / "feature_scaler.joblib")
    models = {label: joblib.load(model_dir / f"{label}_hmm.joblib") for label in CLASSES}

    diagnostics = {"frame_based": {}, "event_based": {}, "per_class": {}}
    for label in CLASSES:
        rows = test_df[test_df["label"] == label]
        truths = []
        preds = []
        frame_truth = []
        frame_pred = []
        for _, row in rows.iterrows():
            audio, sr = _load_audio(Path(row["filepath"]), TARGET_SR)
            feats = scaler.transform(extract_mfcc_features(audio, sr)).astype(np.float32)
            gamma, _, _ = models[label]._forward_backward(feats)
            active = temporal_postprocess(gamma[:, ACTIVE_STATE])
            pred_segments = _binary_event_segments(active, hop_seconds=FEATURE_HOP_SECONDS)
            gt_duration = len(active) * FEATURE_HOP_SECONDS
            truths.append((0.0, gt_duration))
            preds.extend(pred_segments)
            frame_truth.extend([1] * len(active))
            frame_pred.extend(active.tolist())
        diagnostics["frame_based"][label] = frame_based_scores(np.array(frame_truth), np.array(frame_pred))
        diagnostics["event_based"][label] = event_based_scores(truths, preds, collar=collar_seconds)
        diagnostics["per_class"][label] = {
            "num_test_files": int(len(rows)),
            "num_predicted_events": int(len(preds)),
            "num_truth_events": int(len(truths)),
        }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(diagnostics, indent=2))
    return diagnostics


def save_inference_json(events: List[Dict[str, str]], output_path: Path = Path("result_hmm.json")) -> None:
    output_path.write_text(json.dumps(events, indent=2))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binary HMM animal sound pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--processed-root", type=Path, default=Path("processed"))
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/hmm"))

    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--processed-root", type=Path, default=Path("processed"))
    eval_p.add_argument("--model-dir", type=Path, default=Path("artifacts/hmm"))
    eval_p.add_argument("--output-json", type=Path, default=Path("artifacts/hmm/final_diagnostics.json"))

    infer = sub.add_parser("infer")
    infer.add_argument("wav_path", type=Path)
    infer.add_argument("--model-dir", type=Path, default=Path("artifacts/hmm"))
    infer.add_argument("--output", type=Path, default=Path("result_hmm.json"))
    infer.add_argument("--threshold", type=float, default=0.5)
    infer.add_argument("--median-width", type=int, default=5)
    infer.add_argument("--gap-fill-frames", type=int, default=3)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.cmd == "train":
        train_hmm_suite(args.processed_root, args.output_dir)
    elif args.cmd == "evaluate":
        evaluate_suite(args.model_dir, args.processed_root, args.output_json)
    elif args.cmd == "infer":
        events = infer_continuous_file(
            args.wav_path,
            args.model_dir,
            threshold=args.threshold,
            median_width=args.median_width,
            gap_fill_frames=args.gap_fill_frames,
        )
        save_inference_json(events, args.output)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
