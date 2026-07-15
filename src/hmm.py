from __future__ import annotations

import argparse
import json
import math
import warnings
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None

try:
    from preprocessor import (
        INFERENCE_HOP_SECONDS as PREP_INFERENCE_HOP_SECONDS,
        INFERENCE_WINDOW_SECONDS as PREP_INFERENCE_WINDOW_SECONDS,
        TARGET_SR as PREP_TARGET_SR,
        load_standardized_audio,
        inference_time_windowing,
    )
except ModuleNotFoundError:  # pragma: no cover
    from src.preprocessor import (  # type: ignore
        INFERENCE_HOP_SECONDS as PREP_INFERENCE_HOP_SECONDS,
        INFERENCE_WINDOW_SECONDS as PREP_INFERENCE_WINDOW_SECONDS,
        TARGET_SR as PREP_TARGET_SR,
        load_standardized_audio,
        inference_time_windowing,
    )


TARGET_SR = PREP_TARGET_SR
FEATURE_HOP_SECONDS = 0.01
INFERENCE_WINDOW_SECONDS = PREP_INFERENCE_WINDOW_SECONDS
INFERENCE_HOP_SECONDS = PREP_INFERENCE_HOP_SECONDS
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["dog", "cat", "sheep", "cow", "rooster", "background"]
ACTIVE_STATE = 1
INACTIVE_STATE = 0
EPS = 1e-8
_MFCC_TRANSFORM_CACHE: Dict[Tuple[int, int, int, int, int], torch.nn.Module] = {}


def _load_audio(path: Path, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    if torchaudio is None:
        raise RuntimeError("torchaudio is required for the HMM pipeline")
    audio = load_standardized_audio(path, target_sr=target_sr).squeeze(0).cpu().numpy().astype(np.float32)
    return audio, target_sr


def _load_audio_tensor(path: Path, target_sr: int = TARGET_SR) -> Tuple[torch.Tensor, int]:
    if torchaudio is None:
        raise RuntimeError("torchaudio is required for the HMM pipeline")
    audio = load_standardized_audio(path, target_sr=target_sr)
    return audio, target_sr


def _delta_torch(features: torch.Tensor, width: int = 2) -> torch.Tensor:
    if features.shape[0] == 1:
        return torch.zeros_like(features)
    denom = 2 * sum(i * i for i in range(1, width + 1))
    padded = torch.cat([features[:1].repeat(width, 1), features, features[-1:].repeat(width, 1)], dim=0)
    out = torch.zeros_like(features)
    for t in range(features.shape[0]):
        acc = torch.zeros(features.shape[1], device=features.device, dtype=features.dtype)
        for i in range(1, width + 1):
            acc = acc + i * (padded[t + width + i] - padded[t + width - i])
        out[t] = acc / denom
    return out


def extract_mfcc_features_torch(
    audio: torch.Tensor,
    sr: int = TARGET_SR,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 26,
    device: torch.device = DEFAULT_DEVICE,
) -> torch.Tensor:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio.to(device=device, dtype=torch.float32)
    if audio.shape[0] != 1:
        audio = audio.mean(dim=0, keepdim=True)
    if audio.shape[1] < win_length:
        audio = torch.nn.functional.pad(audio, (0, win_length - audio.shape[1]))
    cache_key = (sr, n_mfcc, n_fft, hop_length, win_length, n_mels)
    mfcc_transform = _MFCC_TRANSFORM_CACHE.get(cache_key)
    if mfcc_transform is None or getattr(mfcc_transform, "_device", None) != device:
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "n_mels": n_mels,
                "center": False,
                "power": 2.0,
                "norm": "slaney",
                "mel_scale": "htk",
            },
        ).to(device)
        mfcc_transform._device = device  # type: ignore[attr-defined]
        _MFCC_TRANSFORM_CACHE[cache_key] = mfcc_transform
    mfcc = mfcc_transform(audio).squeeze(0).transpose(0, 1)
    delta = _delta_torch(mfcc)
    delta2 = _delta_torch(delta)
    return torch.cat([mfcc, delta, delta2], dim=1).contiguous()


def _mfcc_frame_count(num_samples: int, win_length: int = 400, hop_length: int = 160) -> int:
    if num_samples <= win_length:
        return 1
    return 1 + max(0, (num_samples - win_length) // hop_length)


def _extract_mfcc_batch(
    audio_batch: List[torch.Tensor],
    sr: int = TARGET_SR,
    device: torch.device = DEFAULT_DEVICE,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 26,
) -> List[np.ndarray]:
    if not audio_batch:
        return []
    batch = []
    lengths = []
    for audio in audio_batch:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] != 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.to(device=device, dtype=torch.float32)
        lengths.append(audio.shape[1])
        batch.append(audio)
    max_len = max(lengths)
    padded = torch.cat(
        [torch.nn.functional.pad(audio, (0, max_len - audio.shape[1])) for audio in batch],
        dim=0,
    )
    cache_key = (sr, n_mfcc, n_fft, hop_length, win_length, n_mels)
    mfcc_transform = _MFCC_TRANSFORM_CACHE.get(cache_key)
    if mfcc_transform is None or getattr(mfcc_transform, "_device", None) != device:
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "n_mels": n_mels,
                "center": False,
                "power": 2.0,
                "norm": "slaney",
                "mel_scale": "htk",
            },
        ).to(device)
        mfcc_transform._device = device  # type: ignore[attr-defined]
        _MFCC_TRANSFORM_CACHE[cache_key] = mfcc_transform
    mfcc = mfcc_transform(padded).transpose(1, 2)  # [B, T, F]
    out = []
    for i, length in enumerate(lengths):
        n_frames = _mfcc_frame_count(length, win_length=win_length, hop_length=hop_length)
        feat = mfcc[i, :n_frames]
        delta = _delta_torch(feat)
        delta2 = _delta_torch(delta)
        out.append(torch.cat([feat, delta, delta2], dim=1).cpu().numpy().astype(np.float32))
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
    features = extract_mfcc_features_torch(
        torch.from_numpy(audio.astype(np.float32)),
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        device=torch.device("cpu"),
    )
    return features.cpu().numpy().astype(np.float32)


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
    n_iter: int = 8
    tol: float = 1e-3
    covariance_floor: float = 1e-3
    random_state: int = 13
    device: str = "cpu"
    verbose: bool = False
    max_train_sequences_per_class: Optional[int] = None


class BinaryGMMHMM:
    def __init__(self, config: BinaryHMMConfig):
        self.config = config
        self.n_states = config.n_states
        self.n_components = config.n_components
        if config.device == "auto":
            self.device = DEFAULT_DEVICE
        else:
            self.device = torch.device(config.device)
        self.random_state = np.random.default_rng(config.random_state)
        self.startprob_ = torch.tensor([0.95, 0.05], dtype=torch.float32, device=self.device)
        self.transmat_ = torch.tensor([[0.97, 0.03], [0.08, 0.92]], dtype=torch.float32, device=self.device)
        self.weights_ = None
        self.means_ = None
        self.covars_ = None

    def _log_gaussian(self, X: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        cov = torch.clamp(cov, min=self.config.covariance_floor)
        diff = X - mean
        return -0.5 * (
            torch.log(2.0 * torch.pi * cov).sum(dim=-1)
            + (diff * diff / cov).sum(dim=-1)
        )

    def _log_mix_emission(self, X: torch.Tensor) -> torch.Tensor:
        log_prob = []
        for s in range(self.n_states):
            comps = []
            for k in range(self.n_components):
                lp = self._log_gaussian(X, self.means_[s, k], self.covars_[s, k]) + torch.log(self.weights_[s, k] + EPS)
                comps.append(lp)
            comp = torch.stack(comps, dim=0)
            log_prob.append(torch.logsumexp(comp, dim=0))
        return torch.stack(log_prob, dim=1)

    def _forward_backward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        log_emit = self._log_mix_emission(X)
        log_start = torch.log(self.startprob_ + EPS)
        log_trans = torch.log(self.transmat_ + EPS)
        T = X.shape[0]
        alpha = torch.empty((T, self.n_states), device=self.device, dtype=torch.float32)
        scale = torch.empty(T, device=self.device, dtype=torch.float32)
        alpha[0] = log_start + log_emit[0]
        scale[0] = torch.logsumexp(alpha[0], dim=0)
        alpha[0] -= scale[0]
        for t in range(1, T):
            alpha[t] = torch.logsumexp(alpha[t - 1].unsqueeze(1) + log_trans, dim=0) + log_emit[t]
            scale[t] = torch.logsumexp(alpha[t], dim=0)
            alpha[t] -= scale[t]
        beta = torch.zeros((T, self.n_states), device=self.device, dtype=torch.float32)
        for t in range(T - 2, -1, -1):
            beta[t] = torch.logsumexp(log_trans + log_emit[t + 1].unsqueeze(0) + beta[t + 1].unsqueeze(0), dim=1) - scale[t + 1]
        gamma = torch.exp(alpha + beta)
        gamma = gamma / torch.clamp(gamma.sum(dim=1, keepdim=True), min=EPS)
        xi = torch.empty((T - 1, self.n_states, self.n_states), device=self.device, dtype=torch.float32)
        for t in range(T - 1):
            m = alpha[t].unsqueeze(1) + log_trans + log_emit[t + 1].unsqueeze(0) + beta[t + 1].unsqueeze(0)
            m = m - torch.logsumexp(m.reshape(-1), dim=0)
            xi[t] = torch.exp(m)
        return gamma, xi, float(scale.sum().item())

    def _init_params(self, X: torch.Tensor) -> None:
        n_features = X.shape[1]
        self.weights_ = torch.full((self.n_states, self.n_components), 1.0 / self.n_components, dtype=torch.float32, device=self.device)
        self.means_ = torch.zeros((self.n_states, self.n_components, n_features), dtype=torch.float32, device=self.device)
        self.covars_ = torch.zeros((self.n_states, self.n_components, n_features), dtype=torch.float32, device=self.device)
        overall_mean = X.mean(dim=0)
        overall_var = X.var(dim=0, unbiased=False) + self.config.covariance_floor
        for s in range(self.n_states):
            for k in range(self.n_components):
                jitter = torch.tensor(self.random_state.normal(scale=0.1, size=n_features), dtype=torch.float32, device=self.device)
                self.means_[s, k] = overall_mean + jitter
                self.covars_[s, k] = overall_var.clone()

    def fit(self, sequences: Sequence[np.ndarray]) -> "BinaryGMMHMM":
        torch_sequences = [torch.as_tensor(seq, dtype=torch.float32, device=self.device) for seq in sequences if len(seq)]
        if not torch_sequences:
            raise ValueError("No training sequences provided.")
        if len(torch_sequences) > 1:
            order = torch.randperm(len(torch_sequences), generator=torch.Generator().manual_seed(self.config.random_state))
            torch_sequences = [torch_sequences[i] for i in order.tolist()]
        X = torch.cat(torch_sequences, dim=0)
        self._init_params(X)
        last_ll = -float("inf")
        for iter_idx in range(self.config.n_iter):
            if self.config.verbose:
                print(f"[hmm] iter {iter_idx + 1}/{self.config.n_iter} on {len(torch_sequences)} seqs", flush=True)
            start_acc = torch.zeros(self.n_states, dtype=torch.float32, device=self.device)
            trans_acc = torch.zeros((self.n_states, self.n_states), dtype=torch.float32, device=self.device)
            gamma_acc = torch.zeros(self.n_states, dtype=torch.float32, device=self.device)
            comp_gamma = torch.zeros((self.n_states, self.n_components), dtype=torch.float32, device=self.device)
            mean_num = torch.zeros_like(self.means_)
            cov_num = torch.zeros_like(self.covars_)
            ll = 0.0
            for seq in torch_sequences:
                gamma, xi, seq_ll = self._forward_backward(seq)
                ll += seq_ll
                start_acc += gamma[0]
                trans_acc += xi.sum(dim=0)
                gamma_acc += gamma.sum(dim=0)
                for s in range(self.n_states):
                    resp = gamma[:, s]
                    for k in range(self.n_components):
                        comp_gamma[s, k] += resp.sum() / self.n_components
                        mean_num[s, k] += (resp.unsqueeze(1) * seq).sum(dim=0) / self.n_components
                        cov_num[s, k] += (resp.unsqueeze(1) * (seq ** 2)).sum(dim=0) / self.n_components
            self.startprob_ = start_acc / torch.clamp(start_acc.sum(), min=EPS)
            self.transmat_ = trans_acc / torch.clamp(trans_acc.sum(dim=1, keepdim=True), min=EPS)
            for s in range(self.n_states):
                for k in range(self.n_components):
                    denom = torch.clamp(comp_gamma[s, k], min=EPS)
                    self.weights_[s, k] = denom / torch.clamp(gamma_acc[s], min=EPS)
                    mean = mean_num[s, k] / denom
                    var = cov_num[s, k] / denom - mean ** 2
                    self.means_[s, k] = mean
                    self.covars_[s, k] = torch.clamp(var, min=self.config.covariance_floor)
            if abs(ll - last_ll) < self.config.tol:
                break
            last_ll = ll
        return self

    def score(self, X: np.ndarray) -> float:
        seq = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        _, _, ll = self._forward_backward(seq)
        return ll

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        seq = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        log_emit = self._log_mix_emission(seq)
        log_start = torch.log(self.startprob_ + EPS)
        log_trans = torch.log(self.transmat_ + EPS)
        T = seq.shape[0]
        delta = torch.empty((T, self.n_states), device=self.device, dtype=torch.float32)
        psi = torch.zeros((T, self.n_states), dtype=torch.int64, device=self.device)
        delta[0] = log_start + log_emit[0]
        for t in range(1, T):
            vals = delta[t - 1].unsqueeze(1) + log_trans
            psi[t] = torch.argmax(vals, dim=0)
            delta[t] = torch.max(vals, dim=0).values + log_emit[t]
        states = torch.zeros(T, dtype=torch.int64, device=self.device)
        states[-1] = torch.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states.cpu().numpy()


def _group_by_recording(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    groups = {}
    for key, grp in df.groupby(["label", "source_file"], dropna=False):
        groups[str(key)] = grp.copy()
    return groups


def _load_sequences(df: pd.DataFrame, device: torch.device = torch.device("cpu")) -> List[np.ndarray]:
    rows = [row for _, row in df.iterrows()]
    if not rows:
        return []

    if device.type == "cuda":
        batch_size = 64  # Increased batch size for more efficient GPU utilization

        def _load_audio_chunk(chunk: List[pd.Series]) -> List[torch.Tensor]:
            # Run the parallel execution without indexing the delayed object
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(_load_audio_tensor)(Path(row["filepath"]), TARGET_SR) for row in chunk
            )

            # Extract the first item (the audio tensor) from each returned tuple
            return [audio for audio, sr in results]

        sequences: List[np.ndarray] = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            audio_batch = _load_audio_chunk(chunk)
            sequences.extend(_extract_mfcc_batch(audio_batch, sr=TARGET_SR, device=device))
        return sequences

    # CPU path remains the same
    def _extract_row(row: pd.Series) -> np.ndarray:
        audio, sr = _load_audio_tensor(Path(row["filepath"]), TARGET_SR)
        return extract_mfcc_features_torch(audio, sr=sr, device=device).cpu().numpy().astype(np.float32)

    return Parallel(n_jobs=-1, prefer="threads")(delayed(_extract_row)(row) for row in rows)


def _fit_scaler(sequences: Sequence[np.ndarray]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.concatenate(sequences, axis=0))
    return scaler


def _transform_sequences(sequences: Sequence[np.ndarray], scaler: StandardScaler) -> List[np.ndarray]:
    return [scaler.transform(seq).astype(np.float32) for seq in sequences]


def _train_one_label(
    label: str,
    train_sequences: Sequence[np.ndarray],
    val_sequences: Sequence[np.ndarray],
    test_sequences: Sequence[np.ndarray],
    scaler: StandardScaler,
    config: BinaryHMMConfig,
    output_dir: Path,
) -> Tuple[str, Dict[str, Optional[float]]]:
    if config.verbose:
        print(f"[train:{label}] scaling sequences", flush=True)
    t0 = time.perf_counter()
    if config.max_train_sequences_per_class is not None and len(train_sequences) > config.max_train_sequences_per_class:
        rng = np.random.default_rng(config.random_state + abs(hash(label)) % 10000)
        chosen = rng.choice(len(train_sequences), size=config.max_train_sequences_per_class, replace=False)
        chosen = sorted(chosen.tolist())
        train_sequences = [train_sequences[i] for i in chosen]
        if config.verbose:
            print(f"[train:{label}] capped train seqs to {len(train_sequences)}", flush=True)
    train_seq = _transform_sequences(train_sequences, scaler)
    val_seq = _transform_sequences(val_sequences, scaler)
    test_seq = _transform_sequences(test_sequences, scaler)

    if config.verbose:
        print(f"[train:{label}] fitting model on {len(train_seq)} sequences", flush=True)
    model = BinaryGMMHMM(config).fit(train_seq)
    joblib.dump(model, output_dir / f"{label}_hmm.joblib")

    val_scores = [model.score(seq) for seq in val_seq] if val_seq else []
    test_scores = [model.score(seq) for seq in test_seq] if test_seq else []
    if config.verbose:
        elapsed = time.perf_counter() - t0
        print(f"[train:{label}] done in {elapsed:.1f}s", flush=True)
    return label, {
        "validation_log_likelihood_mean": float(np.mean(val_scores)) if val_scores else None,
        "test_log_likelihood_mean": float(np.mean(test_scores)) if test_scores else None,
    }


def _score_one_label(
    label: str,
    rows: pd.DataFrame,
    scaler: StandardScaler,
    model: BinaryGMMHMM,
    collar_seconds: float,
) -> Tuple[str, Dict[str, object]]:
    truths = []
    preds = []
    frame_truth = []
    frame_pred = []
    for _, row in rows.iterrows():
        audio_t, sr = _load_audio_tensor(Path(row["filepath"]), TARGET_SR)
        feats = extract_mfcc_features_torch(audio_t, sr=sr, device=DEFAULT_DEVICE).cpu().numpy().astype(np.float32)
        feats = scaler.transform(feats).astype(np.float32)
        feats_t = torch.as_tensor(feats, dtype=torch.float32, device=DEFAULT_DEVICE)
        gamma, _, _ = model._forward_backward(feats_t)
        active = temporal_postprocess(gamma[:, ACTIVE_STATE].cpu().numpy())
        pred_segments = _binary_event_segments(active, FEATURE_HOP_SECONDS)
        gt_duration = len(active) * FEATURE_HOP_SECONDS
        truths.append((0.0, gt_duration))
        preds.extend(pred_segments)
        frame_truth.extend([1] * len(active))
        frame_pred.extend(active.tolist())
    return label, {
        "frame_based": frame_based_scores(np.array(frame_truth), np.array(frame_pred)),
        "event_based": event_based_scores(truths, preds, collar=collar_seconds),
        "num_test_files": int(len(rows)),
        "num_predicted_events": int(len(preds)),
        "num_truth_events": int(len(truths)),
    }


def _score_model_window(
    label: str,
    model: BinaryGMMHMM,
    features: torch.Tensor,
) -> Tuple[str, np.ndarray]:
    # Calculate the log-emission probabilities
    log_emit = model._log_mix_emission(features)
    log_start = torch.log(model.startprob_ + EPS)
    log_trans = torch.log(model.transmat_ + EPS)
    
    T = features.shape[0]
    scale = torch.empty(T, device=features.device, dtype=torch.float32)
    
    # Forward pass to get frame-wise marginal log-likelihoods (scale)
    alpha = log_start + log_emit[0]
    scale[0] = torch.logsumexp(alpha, dim=0)
    alpha = alpha - scale[0]
    
    for t in range(1, T):
        alpha = torch.logsumexp(alpha.unsqueeze(1) + log_trans, dim=0) + log_emit[t]
        scale[t] = torch.logsumexp(alpha, dim=0)
        alpha = alpha - scale[t]
        
    # Return the actual log-likelihood per frame, not the internal gamma state
    return label, scale.cpu().numpy()


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
    total_t0 = time.perf_counter()

    if config.verbose:
        print(f"[train] using config: {config}", flush=True)

    # Timers
    t_manifest, t_split, t_features, t_scaler, t_model_fit = 0.0, 0.0, 0.0, 0.0, 0.0

    # Load manifest
    t0 = time.perf_counter()
    manifest = load_manifest(processed_root)
    t_manifest = time.perf_counter() - t0

    # Split data
    t0 = time.perf_counter()
    train_df, val_df, test_df = recording_level_split(manifest)
    t_split = time.perf_counter() - t0

    train_device = torch.device(config.device if config and config.device != "auto" else "cpu")
    if config.verbose:
        print(f"[train] split sizes train={len(train_df)} val={len(val_df)} test={len(test_df)}", flush=True)
        print(f"[train] feature device={train_device}", flush=True)

    # Extract features
    t0 = time.perf_counter()
    train_sequences_by_label: Dict[str, List[np.ndarray]] = {}
    val_sequences_by_label: Dict[str, List[np.ndarray]] = {}
    test_sequences_by_label: Dict[str, List[np.ndarray]] = {}
    for label in CLASSES:
        if config.verbose:
            print(f"[train] extracting features for {label}", flush=True)
        train_sequences_by_label[label] = _load_sequences(train_df[train_df["label"] == label], device=train_device)
        val_sequences_by_label[label] = _load_sequences(val_df[val_df["label"] == label], device=train_device)
        test_sequences_by_label[label] = _load_sequences(test_df[test_df["label"] == label], device=train_device)
    t_features = time.perf_counter() - t0

    # Fit scaler
    t0 = time.perf_counter()
    if config.verbose:
        print("[train] fitting scaler", flush=True)
    scaler = _fit_scaler([seq for seqs in train_sequences_by_label.values() for seq in seqs])
    joblib.dump(scaler, output_dir / "feature_scaler.joblib")
    t_scaler = time.perf_counter() - t0

    # Fit models
    t0 = time.perf_counter()
    metrics = {"splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)}, "classes": {}}
    if config.verbose:
        print("[train] fitting class models", flush=True)
    trained = Parallel(n_jobs=-1, prefer="threads")(

        delayed(_train_one_label)(
            label,
            train_sequences_by_label[label],
            val_sequences_by_label[label],
            test_sequences_by_label[label],
            scaler,
            config,
            output_dir,
        )
        for label in CLASSES
    )
    metrics["classes"] = {label: values for label, values in trained}
    t_model_fit = time.perf_counter() - t0

    metrics_path = output_dir / "training_diagnostics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    
    total_elapsed = time.perf_counter() - total_t0
    if config.verbose:
        print(f"[train] total elapsed {total_elapsed:.1f}s", flush=True)
        print("\n--- Timing Summary ---")
        print(f"  Manifest loading: {t_manifest:.2f}s")
        print(f"  Data splitting:   {t_split:.2f}s")
        print(f"  Feature extraction: {t_features:.2f}s")
        print(f"  Scaler fitting:     {t_scaler:.2f}s")
        print(f"  Model fitting:      {t_model_fit:.2f}s")
        print(f"  Total:              {total_elapsed:.2f}s")
        print("--------------------\n")
    return metrics


def infer_continuous_file(
    wav_path: Path,
    model_dir: Path = Path("artifacts/hmm"),
    threshold: float = 0.5,
    median_width: int = 15,
    gap_fill_ms: int = 300,
    min_duration_ms: int = 200,
    hop_seconds: float = FEATURE_HOP_SECONDS,
    verbose: bool = False,
) -> List[Dict[str, str]]:
    total_t0 = time.perf_counter()
    if verbose:
        print(f"[infer] processing {wav_path.name}", flush=True)

    # Timers
    t_load_models, t_load_audio, t_windowing, t_scoring, t_postprocess = 0.0, 0.0, 0.0, 0.0, 0.0

    # Load models
    t0 = time.perf_counter()
    scaler: StandardScaler = joblib.load(model_dir / "feature_scaler.joblib")
    models = {label: joblib.load(model_dir / f"{label}_hmm.joblib") for label in CLASSES}
    t_load_models = time.perf_counter() - t0

    # Load audio
    t0 = time.perf_counter()
    audio_t, sr = _load_audio_tensor(wav_path, TARGET_SR)
    t_load_audio = time.perf_counter() - t0

    # Windowing
    t0 = time.perf_counter()
    if torchaudio is None:
        raise RuntimeError("torchaudio is required for inference windowing")
    frames = inference_time_windowing(
        audio_t,
        sr,
        window_seconds=INFERENCE_WINDOW_SECONDS,
        hop_seconds=INFERENCE_HOP_SECONDS,
        window_function="hann",
    )
    t_windowing = time.perf_counter() - t0

    # Scoring
    t0 = time.perf_counter()
    n_windows = frames.shape[0]
    total_frames = int(math.ceil(audio_t.shape[1] / sr / hop_seconds))
    probs = {label: np.zeros(total_frames, dtype=np.float32) for label in CLASSES}
    for w_idx in range(n_windows):
        clip = frames[w_idx]
        features = extract_mfcc_features_torch(clip.unsqueeze(0), sr=sr, device=DEFAULT_DEVICE).cpu().numpy().astype(np.float32)
        features = scaler.transform(features).astype(np.float32)
        features_t = torch.as_tensor(features, dtype=torch.float32, device=DEFAULT_DEVICE)
        frame_offset = int(round((w_idx * INFERENCE_HOP_SECONDS) / hop_seconds))
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_score_model_window)(label, model, features_t)
            for label, model in models.items()
        )

        # Compute Softmax probabilities across all models
        ll_dict = dict(results)
        ll_stack = np.stack([ll_dict[lbl] for lbl in CLASSES], axis=0) # Shape: (6, T)

        ll_max = np.max(ll_stack, axis=0)
        exp_ll = np.exp(ll_stack - ll_max)
        softmax_probs = exp_ll / np.sum(exp_ll, axis=0)

        for i, label in enumerate(CLASSES):
            active_prob = softmax_probs[i]
            end_frame = min(frame_offset + len(active_prob), total_frames)
            if end_frame > frame_offset:
                probs[label][frame_offset:end_frame] = np.maximum(
                    probs[label][frame_offset:end_frame],
                    active_prob[: end_frame - frame_offset],
                )

                t_scoring = time.perf_counter() - t0

    # --- NEW: Enforce Mutually Exclusive Classes ---
    # Stack probabilities to find the dominant class per frame
    prob_stack = np.stack([probs[lbl] for lbl in CLASSES], axis=0)
    dominant_indices = np.argmax(prob_stack, axis=0)

    for i, label in enumerate(CLASSES):
        # Zero out the probability if it's not the highest scoring class
        probs[label] = np.where(dominant_indices == i, probs[label], 0.0)
    # -----------------------------------------------

    # Post-processing
    t0 = time.perf_counter()
    outputs = []
    min_dur_sec = min_duration_ms / 1000.0

    for label in CLASSES:
        if label == "background":
            continue

        active = temporal_postprocess(
            probs[label],
            threshold=threshold,
            median_width=median_width,
            gap_fill=max(1, int(round((gap_fill_ms / 1000.0) / hop_seconds))),
        )
        segments = _binary_event_segments(active, hop_seconds)

        for start, end in segments:
            # Filter out the garbage micro-events
            if (end - start) >= min_dur_sec:
                outputs.append({"event_start": f"{start:.3f}", "event_end": f"{end:.3f}", "animal": label})

    outputs.sort(key=lambda x: (float(x["event_start"]), x["animal"]))
    t_postprocess = time.perf_counter() - t0


    total_elapsed = time.perf_counter() - total_t0
    if verbose:
        print(f"[infer] total elapsed {total_elapsed:.1f}s", flush=True)
        print("\n--- Timing Summary ---")
        print(f"  Model loading:   {t_load_models:.2f}s")
        print(f"  Audio loading:   {t_load_audio:.2f}s")
        print(f"  Windowing:       {t_windowing:.2f}s")
        print(f"  Scoring:         {t_scoring:.2f}s")
        print(f"  Post-processing: {t_postprocess:.2f}s")
        print(f"  Total:           {total_elapsed:.2f}s")
        print("--------------------\n")

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

    scored = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_score_one_label)(label, test_df[test_df["label"] == label], scaler, models[label], collar_seconds)
        for label in CLASSES
    )
    diagnostics = {"frame_based": {}, "event_based": {}, "per_class": {}}
    for label, values in scored:
        diagnostics["frame_based"][label] = values["frame_based"]
        diagnostics["event_based"][label] = values["event_based"]
        diagnostics["per_class"][label] = {
            "num_test_files": values["num_test_files"],
            "num_predicted_events": values["num_predicted_events"],
            "num_truth_events": values["num_truth_events"],
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
    train.add_argument("--device", type=str, default="auto", help="Device to use for training (cpu, cuda, auto)")
    train.add_argument("--n-iter", type=int, default=8, help="Number of training iterations")
    train.add_argument("--n-components", type=int, default=2, help="Number of GMM components")
    train.add_argument("--max-train-sequences", type=int, default=None, help="Cap the number of training sequences per class")

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
    infer.add_argument("--gap-fill-ms", type=int, default=300)
    infer.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.cmd == "train":
        config = BinaryHMMConfig(
            n_iter=args.n_iter,
            n_components=args.n_components,
            device=args.device,
            verbose=True,
            max_train_sequences_per_class=args.max_train_sequences,
        )
        train_hmm_suite(args.processed_root, args.output_dir, config=config)
    elif args.cmd == "evaluate":
        evaluate_suite(args.model_dir, args.processed_root, args.output_json)
    elif args.cmd == "infer":
        events = infer_continuous_file(
            args.wav_path,
            args.model_dir,
            threshold=args.threshold,
            median_width=args.median_width,
            gap_fill_ms=args.gap_fill_ms,
            verbose=args.verbose,
        )
        save_inference_json(events, args.output)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
