from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from preprocessor import TARGET_SR, INFERENCE_WINDOW_SECONDS, INFERENCE_HOP_SECONDS
except ModuleNotFoundError:
    from src.preprocessor import TARGET_SR, INFERENCE_WINDOW_SECONDS, INFERENCE_HOP_SECONDS

CLASSES = ["cat", "cow", "dog", "rooster", "sheep", "background"]
ANIMAL_CLASSES = CLASSES[:5]
BACKGROUND_INDEX = 5

PROCESSED_CSV = Path("processed") / "farmyard.csv"
VAL_FOLD = 5

# Mel front-end matches the EDA (src/eda.py).
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 64

# The model is trained on 1 s windows that are SYNTHESISED to look like the
# continuous inference scenes: 0/1/2 animal clips mixed onto a background bed at a
# random SNR, plus a noise floor. This makes the training distribution match the
# evaluation distribution (events in continuous, background-heavy, overlapping
# audio) instead of training on isolated clips -- the fix that recovers event-level
# performance.
WINDOW_SAMPLES = int(1.0 * TARGET_SR)
P_NUM_ACTIVE = (0.30, 0.45, 0.25)        # P(0), P(1), P(2) animals in a window
SNR_DB_RANGE = (3.0, 18.0)
NOISE_FLOOR_DB_RANGE = (-45.0, -30.0)
WINDOWS_PER_EPOCH = 4000
VAL_WINDOWS = 1500

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_mels=N_MELS
)
_to_db = torchaudio.transforms.AmplitudeToDB()


def waveform_to_mel(waveform: torch.Tensor) -> torch.Tensor:
    """(1, samples) -> (1, n_mels, time) log-mel with a FIXED normalisation (a
    constant dB offset/scale), not per-example standardisation -- so silence stays
    low instead of being stretched up into spurious activations."""
    mel = _to_db(_mel(waveform))
    return (mel + 40.0) / 40.0


def spec_augment(mel: torch.Tensor, rng: np.random.Generator,
                 n_freq_masks: int = 2, n_time_masks: int = 2,
                 max_freq: int = 8, max_time: int = 16) -> torch.Tensor:
    """SpecAugment: zero out a few random frequency bands and time spans (training
    only), a cheap regulariser on top of the synthesis augmentation."""
    mel = mel.clone()
    n_mels, n_time = mel.shape[-2], mel.shape[-1]
    for _ in range(n_freq_masks):
        f = int(rng.integers(0, max_freq + 1))
        if f:
            f0 = int(rng.integers(0, max(1, n_mels - f)))
            mel[..., f0:f0 + f, :] = 0.0
    for _ in range(n_time_masks):
        t = int(rng.integers(0, max_time + 1))
        if t:
            t0 = int(rng.integers(0, max(1, n_time - t)))
            mel[..., :, t0:t0 + t] = 0.0
    return mel


# --------------------------------------------------------------------------- #
# audio helpers for on-the-fly synthesis
# --------------------------------------------------------------------------- #
def _peak_normalize(waveform: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    m = waveform.abs().max()
    return waveform * peak / m if m > 0 else waveform


def trim_active(waveform: torch.Tensor, top_db: float = 30.0,
                frame: int = WIN_LENGTH, hop: int = HOP_LENGTH) -> torch.Tensor:
    """Trim leading/trailing silence: keep the span from the first to the last
    frame whose energy is within `top_db` of the clip's peak frame. Removes the
    silent padding of Hamza's fixed 2 s frames so synthesis places real sound, not
    silence labelled as the animal."""
    n = waveform.shape[-1]
    if n < frame:
        return waveform
    frames = waveform.unfold(-1, frame, hop)              # (1, n_frames, frame)
    rms = frames.pow(2).mean(-1).clamp_min(1e-12).sqrt().squeeze(0)
    db = 20.0 * torch.log10(rms + 1e-9)
    active = (db >= (db.max() - top_db)).nonzero().flatten()
    if active.numel() == 0:
        return waveform
    start = int(active[0]) * hop
    end = min(n, int(active[-1]) * hop + frame)
    trimmed = waveform[..., start:end]
    return trimmed if trimmed.shape[-1] > 0 else waveform


def _random_crop(waveform: torch.Tensor, n: int, rng: np.random.Generator) -> torch.Tensor:
    length = waveform.shape[-1]
    if length <= n:
        return torch.nn.functional.pad(waveform, (0, n - length))
    start = int(rng.integers(0, length - n + 1))
    return waveform[..., start:start + n]


def _mix_at_snr(clip: torch.Tensor, background: torch.Tensor, snr_db: float) -> torch.Tensor:
    p_clip = float(clip.pow(2).mean())
    p_bg = float(background.pow(2).mean())
    if p_clip > 0 and p_bg > 0:
        scale = math.sqrt((10 ** (snr_db / 10.0)) * p_bg / p_clip)
        return clip * scale + background
    return clip + background


def _add_noise(waveform: torch.Tensor, level_db: float, rng: np.random.Generator) -> torch.Tensor:
    level = 10 ** (level_db / 20.0)
    noise = torch.from_numpy(rng.normal(0.0, level, waveform.shape[-1]).astype(np.float32)).unsqueeze(0)
    return waveform + noise


def _fold_of(source_file: str) -> int:
    token = str(source_file).split("-")[0]
    return int(token) if token.isdigit() else -1


class ClipBank:
    """Loads the training clips grouped by class (and background), each trimmed to
    its active region, ready to be synthesised into windows."""

    def __init__(self, frames: pd.DataFrame, loader: Optional[Callable] = None):
        self.by_class: Dict[str, List[torch.Tensor]] = {c: [] for c in ANIMAL_CLASSES}
        self.background: List[torch.Tensor] = []
        load = loader or self._load
        for _, row in frames.iterrows():
            waveform = trim_active(load(row["filepath"]))
            if waveform.shape[-1] < TARGET_SR // 4:       # keep clips >= 0.25 s
                continue
            label = row["label"]
            if label == "background":
                self.background.append(waveform)
            elif label in ANIMAL_CLASSES:
                self.by_class[label].append(waveform)

    @staticmethod
    def _load(filepath: str) -> torch.Tensor:
        waveform, _ = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        peak = waveform.abs().max()
        return waveform / peak if peak > 0 else waveform

    def sample_clip(self, cls: str, rng: np.random.Generator) -> torch.Tensor:
        pool = self.by_class[cls]
        return pool[int(rng.integers(0, len(pool)))]

    def sample_background(self, n: int, rng: np.random.Generator) -> torch.Tensor:
        if not self.background:
            return torch.zeros(1, n)
        return _random_crop(self.background[int(rng.integers(0, len(self.background)))], n, rng)

    def counts(self) -> Dict[str, int]:
        d = {c: len(v) for c, v in self.by_class.items()}
        d["background"] = len(self.background)
        return d


def synth_window(bank: ClipBank, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build one 1 s training window and its multi-hot label: 0/1/2 animals mixed
    onto a background bed at a random SNR, with a noise floor -- the same
    distribution the model meets at inference on continuous scenes."""
    n = WINDOW_SAMPLES
    label = torch.zeros(len(CLASSES))
    available = [i for i, c in enumerate(ANIMAL_CLASSES) if bank.by_class[c]]
    k = min(int(rng.choice([0, 1, 2], p=P_NUM_ACTIVE)), len(available))
    chosen = list(rng.choice(available, size=k, replace=False)) if k > 0 else []
    if k == 0:
        label[BACKGROUND_INDEX] = 1.0

    bg = bank.sample_background(n, rng)
    mix = 0.3 * _peak_normalize(bg, 0.7) if bg.abs().max() > 0 else torch.zeros(1, n)

    for ci in chosen:
        clip = bank.sample_clip(ANIMAL_CLASSES[ci], rng)
        clip = _random_crop(clip, n, rng) * float(rng.uniform(0.6, 1.0))
        clip = _peak_normalize(clip, 0.9)
        snr = float(rng.uniform(*SNR_DB_RANGE))
        mix = _mix_at_snr(clip, mix, snr) if mix.abs().max() > 0 else clip
        label[ci] = 1.0

    mix = _add_noise(mix, float(rng.uniform(*NOISE_FLOOR_DB_RANGE)), rng)
    mix = _peak_normalize(mix, 0.95)
    return mix, label


class SynthWindowDataset(Dataset):
    """Synthesises windows on the fly. `fixed=True` (validation) makes each index
    reproducible; `fixed=False` (training) draws fresh windows every epoch."""

    def __init__(self, bank: ClipBank, n_windows: int, augment: bool = False,
                 fixed: bool = False, seed: int = 0):
        self.bank = bank
        self.n_windows = n_windows
        self.augment = augment
        self.fixed = fixed
        self.seed = seed

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed * 1_000_003 + index) if self.fixed else np.random.default_rng()
        waveform, label = synth_window(self.bank, rng)
        mel = waveform_to_mel(waveform)
        if self.augment:
            mel = spec_augment(mel, rng)
        return mel, label


class AnimalCNN(nn.Module):
    """A 4-block 2D CNN over the mel-spectrogram with a multi-label sigmoid head
    (~0.5 M params). Adaptive average pooling keeps the head independent of the
    input length, so it trains on 1 s windows and runs on the 1 s inference window
    unchanged."""

    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(256, n_classes)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.features(mel)
        x = self.pool(x).flatten(1)
        return self.head(self.dropout(x))


def _collate(batch):
    mels, labels = zip(*batch)
    width = max(m.shape[-1] for m in mels)
    padded = [torch.nn.functional.pad(m, (0, width - m.shape[-1])) for m in mels]
    return torch.stack(padded), torch.stack(labels)


def load_frame_splits(csv_path: Path = PROCESSED_CSV) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Original frames only, split by fold: folds 1-4 are the clip source for
    training-window synthesis; fold 5 is the held-out source for validation."""
    df = pd.read_csv(csv_path)
    df = df[df["segment_type"] == "training_frame"].copy()
    df["fold"] = df["source_file"].map(_fold_of)
    return df[df["fold"] != VAL_FOLD], df[df["fold"] == VAL_FOLD]


def train_model(
    csv_path: Path = PROCESSED_CSV,
    max_epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 10,
    windows_per_epoch: int = WINDOWS_PER_EPOCH,
    seed: int = 0,
    num_workers: int = 2,
    checkpoint_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train the CNN on synthesised scene-matched windows, to convergence with
    early stopping, an LR scheduler, and validation on fold-5-sourced windows."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_df, val_df = load_frame_splits(csv_path)

    train_bank = ClipBank(train_df)
    val_bank = ClipBank(val_df)
    if verbose:
        print("[train] clip bank:", train_bank.counts())

    train_ds = SynthWindowDataset(train_bank, windows_per_epoch, augment=True, fixed=False)
    val_ds = SynthWindowDataset(val_bank, VAL_WINDOWS, augment=False, fixed=True, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=_collate)

    model = AnimalCNN().to(device)
    if resume_from and Path(resume_from).is_file():
        model.load_state_dict(torch.load(resume_from, map_location=device))
        if verbose:
            print(f"[train] warm start from {resume_from}")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1, best_state, bad_epochs, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for mel, label in train_loader:
            mel, label = mel.to(device), label.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(mel), label)
            loss.backward()
            optimizer.step()
            running += loss.item() * mel.size(0)
            seen += mel.size(0)
        train_loss = running / max(seen, 1)

        val_loss, val_f1 = _validate(model, val_loader, loss_fn, device)
        scheduler.step(val_f1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1, best_state, best_epoch, bad_epochs = val_f1, copy.deepcopy(model.state_dict()), epoch, 0
            if checkpoint_path:                       # persist best-so-far, survives an interrupt
                torch.save(best_state, checkpoint_path)
        else:
            bad_epochs += 1
        if verbose:
            print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  val_f1 {val_f1:.3f}")
        if bad_epochs >= patience:
            if verbose:
                print(f"early stop at epoch {epoch} (best {best_f1:.3f} @ epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model.cpu(), "val_f1": best_f1, "history": history, "best_epoch": best_epoch}


def overfit_check(n_windows: int = 32, epochs: int = 80, device: Optional[str] = None) -> float:
    """Sanity check: the model must memorise a tiny fixed set of synthesised
    windows. Returns final train macro-F1 (should approach 1.0)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    train_df, _ = load_frame_splits()
    bank = ClipBank(train_df)
    ds = SynthWindowDataset(bank, n_windows, augment=False, fixed=True, seed=0)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=_collate)

    model = AnimalCNN().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for mel, label in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(mel.to(device)), label.to(device))
            loss.backward()
            optimizer.step()
    _, f1 = _validate(model, loader, loss_fn, device)
    return f1


@torch.no_grad()
def _validate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: str) -> Tuple[float, float]:
    model.eval()
    n_animals = len(ANIMAL_CLASSES)
    tp = np.zeros(n_animals); fp = np.zeros(n_animals); fn = np.zeros(n_animals)
    running, seen = 0.0, 0
    for mel, label in loader:
        logits = model(mel.to(device))
        running += loss_fn(logits, label.to(device)).item() * mel.size(0)
        seen += mel.size(0)
        pred = torch.sigmoid(logits).cpu().numpy()[:, :n_animals] >= 0.5
        true = label.numpy()[:, :n_animals] >= 0.5
        tp += np.sum(pred & true, axis=0)
        fp += np.sum(pred & ~true, axis=0)
        fn += np.sum(~pred & true, axis=0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    return running / max(seen, 1), float(np.mean(f1))


def _slide_windows(
    waveform: torch.Tensor,
    sample_rate: int,
    window_seconds: float = INFERENCE_WINDOW_SECONDS,
    hop_seconds: float = INFERENCE_HOP_SECONDS,
) -> torch.Tensor:
    """Slide the shared 1 s / 0.25 s-hop inference window WITHOUT an amplitude
    taper. The Hann window belongs inside the STFT (the mel front-end already
    applies one per 25 ms analysis frame); tapering the whole 1 s chunk fed to
    the CNN would fade out sound near the window edges, which the training
    windows (synth_window) never have -- a train/inference mismatch that
    suppresses onsets. Same timing grid as preprocessor.inference_time_windowing."""
    window_samples = max(1, int(window_seconds * sample_rate))
    hop_samples = max(1, int(hop_seconds * sample_rate))
    if waveform.shape[1] <= window_samples:
        waveform = torch.nn.functional.pad(waveform, (0, window_samples - waveform.shape[1]))
    return waveform.unfold(dimension=1, size=window_samples, step=hop_samples).squeeze(0)


@torch.no_grad()
def predict_timeline(
    model: nn.Module,
    waveform: torch.Tensor,
    sample_rate: int,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide the 1 s inference window over a recording -> (probs[T, C],
    frame_times[T], energy_db[T]) for the Section 4 harness."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if sample_rate != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sample_rate, TARGET_SR)(waveform)
        sample_rate = TARGET_SR
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    windows = _slide_windows(waveform, sample_rate)
    probs, times, energy = [], [], []
    for i, window in enumerate(windows):
        mel = waveform_to_mel(window.unsqueeze(0)).unsqueeze(0).to(device)
        probs.append(torch.sigmoid(model(mel)).cpu().numpy()[0])
        times.append(i * INFERENCE_HOP_SECONDS + INFERENCE_WINDOW_SECONDS / 2.0)
        rms = float(torch.sqrt(torch.mean(window ** 2)) + 1e-9)
        energy.append(20 * np.log10(rms))
    return np.array(probs), np.array(times), np.array(energy)


def predict_fn_for(model: nn.Module) -> Callable:
    """Wrap a trained model as the predict_fn the Section 4 harness consumes."""
    return lambda waveform, sample_rate: predict_timeline(model, waveform, sample_rate)


@torch.no_grad()
def frame_level_report(model: nn.Module, csv_path: Path = PROCESSED_CSV,
                       device: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Per-class precision/recall/F1 at the frame level on the fold-5 hold-out
    frames (the cheap diagnostic the spec asks for before event formation)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, val_df = load_frame_splits(csv_path)
    model = model.to(device).eval()
    probs, trues = [], []
    for _, row in val_df.iterrows():
        waveform = ClipBank._load(row["filepath"])
        mel = waveform_to_mel(waveform).unsqueeze(0).to(device)
        probs.append(torch.sigmoid(model(mel)).cpu().numpy()[0])
        target = np.zeros(len(CLASSES)); target[CLASSES.index(row["label"])] = 1.0
        trues.append(target)
    probs = np.array(probs); trues = np.array(trues)

    report: Dict[str, Dict[str, float]] = {}
    f1s = []
    for i, label in enumerate(CLASSES):
        pred = probs[:, i] >= 0.5
        true = trues[:, i] >= 0.5
        tp = int(np.sum(pred & true)); fp = int(np.sum(pred & ~true)); fn = int(np.sum(~pred & true))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[label] = {"precision": precision, "recall": recall, "f1": f1, "support": int(true.sum())}
        if label in ANIMAL_CLASSES:
            f1s.append(f1)
    report["macro_animals"] = {"f1": float(np.mean(f1s))}
    return report


def plot_predictions(model: nn.Module, waveform: torch.Tensor, sample_rate: int,
                     out_path: str = "cnn_predictions.png"):
    """Mel-spectrogram of a recording with the CNN's per-class probability curves
    lined up underneath, for qualitative inspection."""
    import matplotlib.pyplot as plt
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    probs, times, _ = predict_timeline(model, waveform, sample_rate)
    mel = waveform_to_mel(waveform).squeeze(0).numpy()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7))
    ax_top.imshow(mel, origin="lower", aspect="auto"); ax_top.set_title("mel spectrogram")
    for i, label in enumerate(ANIMAL_CLASSES):
        ax_bot.plot(times, probs[:, i], label=label)
    ax_bot.set_title("CNN per-class probability over time")
    ax_bot.set_xlabel("time (s)"); ax_bot.set_ylim(0, 1); ax_bot.legend(loc="upper right", ncol=5)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str) -> nn.Module:
    model = AnimalCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
