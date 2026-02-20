"""
Pipeline (terminal) equivalente a `DPPM.ipynb`, sin generar imágenes.

Qué hace (secuencial):
- Carga BCI Competition III - Dataset V (Subject 1): raw01+raw02+raw03 (labeled)
- Preprocesa (paper-like): annotate muscle + ICA (Fp1/Fp2, fold-safe) + bandpass 8–30Hz + z-score por canal
- Segmenta epochs de 1s (512) con stride 0.5s (256), sin cruzar cambios de etiqueta ni límites de sesión
- Split 70/30 estratificado (paper)
- Entrena DDPM por canal target (8 canales)
- Evalúa generación (MSE y Pearson) y muestra Tabla 1 (computed vs reported)
- Construye TEST(hybrid) reemplazando esos 8 canales por sintéticos (DDPM)
- Evalúa CSP+LDA (logs + matrices de confusión como arrays, sin plots)
- Entrena ScoreNet/SGM por canal y repite evaluación + híbrido + CSP+LDA (sin plots)

Uso:
  python pipeline.py

Opcional (para pruebas rápidas):
  python pipeline.py --ddpm-epochs 10 --sgm-epochs 10 --skip-sgm
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


def _log_section(title: str) -> None:
    bar = "=" * 80
    print("\n" + bar)
    print(title)
    print(bar + "\n")


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Improve run-to-run stability.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------------------
# DDPM base (copiado del notebook, con logging reducido por tqdm)
# --------------------------------------------------------------------------------------


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 64, time_emb_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels + time_emb_dim, n_feat, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_feat),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_feat, n_feat * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_feat * 2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(n_feat * 2, n_feat, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_feat),
            nn.ReLU(),
        )
        self.conv4 = nn.ConvTranspose1d(n_feat, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t).unsqueeze(-1).repeat(1, 1, x.size(-1))
        x = torch.cat([x, t_emb], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


def train_diffusion_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    dataloader: DataLoader,
    timesteps: int,
    betas: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    epochs: int = 100,
    *,
    device: torch.device,
    print_every: int = 0,
    show_epoch_progress: bool = True,
    epoch_desc: str = "Train",
    show_batch_progress: bool = False,
) -> None:
    """Train DDPM noise-prediction model (misma lógica; logs via tqdm)."""

    model.train()

    epoch_iter = range(epochs)
    if show_epoch_progress:
        epoch_iter = tqdm(epoch_iter, total=epochs, desc=epoch_desc, leave=True, mininterval=0.5)

    for epoch in epoch_iter:
        epoch_loss = 0.0

        batch_iter = enumerate(dataloader)
        if show_batch_progress:
            batch_iter = tqdm(batch_iter, total=len(dataloader), leave=False, mininterval=2.0)

        for _, (batch_inputs, batch_targets) in batch_iter:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_size = batch_inputs.size(0)

            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(batch_targets)

            sqrt_acp_t = sqrt_alphas_cumprod[t].view(batch_size, 1, 1)
            sqrt_om_acp_t = sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1)
            x_t = sqrt_acp_t * batch_targets + sqrt_om_acp_t * noise

            model_input = torch.cat([batch_inputs, x_t], dim=1)
            optimizer.zero_grad()
            noise_pred = model(model_input, t)
            loss = nn.MSELoss()(noise_pred, noise)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(dataloader))
        if show_epoch_progress and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")

        is_first = epoch == 0
        is_last = epoch == epochs - 1
        should_print = is_first or is_last or (print_every > 0 and (epoch + 1) % int(print_every) == 0)
        if should_print:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step()


def sample_from_model(
    model: nn.Module,
    conditioning_inputs: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    betas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    """DDPM reverse process (tal como en el notebook)."""
    model.eval()

    batch_size = shape[0]
    seq_length = shape[2]
    x = torch.randn((batch_size, 1, seq_length), device=device)

    with torch.no_grad():
        for t in reversed(range(int(len(betas)))):
            t_batch = torch.full((batch_size,), int(t), device=device, dtype=torch.long)

            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)

            model_input = torch.cat([conditioning_inputs, x], dim=1)
            noise_pred = model(model_input, t_batch)

            x = (1.0 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred)

    return x


# --------------------------------------------------------------------------------------
# Dataset helpers (epochs) + metrics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentSpec:
    target: str
    inputs: tuple[str, str]
    mse_reported: float
    pearson_reported: float


EXPERIMENTS: list[ExperimentSpec] = [
    ExperimentSpec("AF3", ("Fp1", "F3"), 5.230499, 0.791356),
    ExperimentSpec("AF4", ("Fp2", "F4"), 4.655864, 0.839933),
    ExperimentSpec("F7", ("FC5", "F3"), 4.170372, 0.854676),
    ExperimentSpec("F8", ("T8", "F4"), 5.457867, 0.840970),
    ExperimentSpec("Fp1", ("AF3", "F3"), 5.040723, 0.788673),
    ExperimentSpec("Fp2", ("AF4", "F4"), 11.081029, 0.683048),
    ExperimentSpec("T7", ("C3", "CP1"), 20.746539, 0.465758),
    ExperimentSpec("T8", ("C4", "CP2"), 19.659182, 0.482980),
]


CHANNEL_NAMES = [
    "Fp1",
    "AF3",
    "F7",
    "F3",
    "FC1",
    "FC5",
    "T7",
    "C3",
    "CP1",
    "CP5",
    "P7",
    "P3",
    "Pz",
    "PO3",
    "O1",
    "Oz",
    "O2",
    "PO4",
    "P4",
    "P8",
    "CP6",
    "CP2",
    "C4",
    "T8",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "AF4",
    "Fp2",
    "Fz",
    "Cz",
]

CHANNEL_TO_IDX = {name: i for i, name in enumerate(CHANNEL_NAMES)}


class EEGEpochs(Dataset):
    """Dataset de epochs (n_epochs, n_channels, n_times)."""

    def __init__(self, epochs: np.ndarray, *, input_idxs: tuple[int, ...], target_idx: int):
        if epochs.ndim != 3:
            raise ValueError(f"Expected epochs (n_epochs, n_channels, n_times), got {epochs.shape}")
        self.epochs = epochs.astype(np.float32, copy=False)
        self.input_idxs = tuple(int(i) for i in input_idxs)
        self.target_idx = int(target_idx)

    def __len__(self) -> int:
        return int(self.epochs.shape[0])

    def __getitem__(self, idx: int):
        ep = self.epochs[idx]  # (C, T)
        conditioning = ep[list(self.input_idxs), :]  # (n_cond, T) (can be 0 for non-conditional)
        target = ep[self.target_idx : self.target_idx + 1, :]  # (1, T)
        return torch.from_numpy(conditioning), torch.from_numpy(target)


def pearson_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False).reshape(-1)
    b = b.astype(np.float64, copy=False).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def build_epochs_from_labels(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epoch_samples: int,
    stride_samples: int,
    allowed_labels: set[int],
    max_epochs_per_class: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Ventanas de 1s con salida cada 0.5s, sin cruzar cambios de etiqueta."""
    n_times, _ = x.shape
    if y.shape[0] != n_times:
        raise ValueError("x and y length mismatch")
    if epoch_samples <= 0:
        raise ValueError("epoch_samples must be > 0")
    if stride_samples <= 0:
        raise ValueError("stride_samples must be > 0")

    epochs_by_class: dict[int, list[np.ndarray]] = {k: [] for k in allowed_labels}

    start = 0
    while start < n_times:
        label = int(y[start])
        end = start + 1
        while end < n_times and int(y[end]) == label:
            end += 1

        if label in allowed_labels:
            s = start
            while s + epoch_samples <= end:
                e = s + epoch_samples
                ep = x[s:e].T  # (C, T)
                epochs_by_class[label].append(ep.astype(np.float32, copy=False))
                s += stride_samples

        start = end

    epochs_list: list[np.ndarray] = []
    labels_list: list[int] = []
    for k in sorted(allowed_labels):
        eps = epochs_by_class[k]
        if max_epochs_per_class is not None:
            eps = eps[: int(max_epochs_per_class)]
        epochs_list.extend(eps)
        labels_list.extend([k] * len(eps))

    if not epochs_list:
        raise ValueError("No epochs were created. Check labels and epoch/stride.")

    epochs = np.stack(epochs_list, axis=0)  # (N, C, T)
    labels = np.asarray(labels_list, dtype=np.int64)
    return epochs, labels


def build_epochs_from_labels_with_groups(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epoch_samples: int,
    stride_samples: int,
    allowed_labels: set[int],
    max_epochs_per_class: Optional[int],
    group_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Same windowing as `build_epochs_from_labels`, but also returns `groups` so that all
    overlapping windows from the same contiguous-label segment stay together in CV.

    - group_id increments per contiguous label segment (within a session)
    - all windows produced from that segment share the same group_id
    """
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)
    n_times, _ = x.shape
    if y.shape[0] != n_times:
        raise ValueError("x and y length mismatch")
    if epoch_samples <= 0:
        raise ValueError("epoch_samples must be > 0")
    if stride_samples <= 0:
        raise ValueError("stride_samples must be > 0")

    epochs_list: list[np.ndarray] = []
    labels_list: list[int] = []
    groups_list: list[int] = []

    per_class_count: dict[int, int] = {int(k): 0 for k in allowed_labels}
    gid = int(group_offset)

    start = 0
    while start < n_times:
        label = int(y[start])
        end = start + 1
        while end < n_times and int(y[end]) == label:
            end += 1

        if label in allowed_labels:
            # If limiting per class, stop generating once the class reaches the cap.
            if max_epochs_per_class is not None and int(per_class_count[int(label)]) >= int(max_epochs_per_class):
                pass
            else:
                s = start
                n_added = 0
                while s + int(epoch_samples) <= end:
                    if max_epochs_per_class is not None and int(per_class_count[int(label)]) >= int(max_epochs_per_class):
                        break
                    e = s + int(epoch_samples)
                    ep = x[s:e].T  # (C, T)
                    epochs_list.append(ep.astype(np.float32, copy=False))
                    labels_list.append(int(label))
                    groups_list.append(int(gid))
                    per_class_count[int(label)] = int(per_class_count[int(label)]) + 1
                    n_added += 1
                    s += int(stride_samples)
                if n_added > 0:
                    gid += 1

        start = end

    if not epochs_list:
        raise ValueError("No epochs were created. Check labels and epoch/stride.")

    epochs = np.stack(epochs_list, axis=0)
    labels = np.asarray(labels_list, dtype=np.int64)
    groups = np.asarray(groups_list, dtype=np.int64)
    return epochs, labels, groups, int(gid)


def iter_stratified_group_kfold_indices(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Stratified K-fold split at the *group* level.
    Prevents leakage when samples are overlapping windows.
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y, dtype=int).reshape(-1)
    groups = np.asarray(groups, dtype=int).reshape(-1)
    if y.shape[0] != groups.shape[0]:
        raise ValueError(f"y/groups mismatch: {y.shape} vs {groups.shape}")

    uniq_groups = np.unique(groups)
    # Each group corresponds to one contiguous-label segment, so label is constant within group.
    group_to_label: dict[int, int] = {}
    for g in uniq_groups.tolist():
        idx = int(np.flatnonzero(groups == int(g))[0])
        group_to_label[int(g)] = int(y[idx])

    group_labels = np.asarray([group_to_label[int(g)] for g in uniq_groups.tolist()], dtype=int)
    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for g_tr_idx, g_va_idx in skf.split(uniq_groups, group_labels):
        g_tr = set(int(uniq_groups[int(i)]) for i in np.asarray(g_tr_idx, dtype=int).tolist())
        g_va = set(int(uniq_groups[int(i)]) for i in np.asarray(g_va_idx, dtype=int).tolist())
        tr_idx = np.flatnonzero(np.isin(groups, list(g_tr))).astype(int)
        va_idx = np.flatnonzero(np.isin(groups, list(g_va))).astype(int)
        splits.append((tr_idx, va_idx))
    return splits


def load_window_labels_from_asc(path: str) -> np.ndarray:
    """
    Load per-window labels from a .asc file (one number per line, scientific notation).
    Returns int labels (shape: (n_windows,)).
    """
    ys: list[int] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                ys.append(int(round(float(s))))
            except Exception:
                continue
    if not ys:
        raise ValueError(f"No labels found in asc: {path}")
    return np.asarray(ys, dtype=np.int64)


def build_epochs_from_window_labels(
    x: np.ndarray,
    y_windows: np.ndarray,
    *,
    epoch_samples: int,
    stride_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build epochs from continuous signal using labels provided *per window*.

    Matches BCI-III test labels in `labels8_subject*_raw.asc`:
    - window length = epoch_samples (1s = 512)
    - `y_windows[i]` labels the window starting at `i * (label_stride)`

    NOTE:
    - The official `.asc` files are commonly provided at **0.5s step** (label per window starting every 256 samples),
      even though each window is still 1s long.
    - If you ever choose a coarser stride (e.g., 1.0s), we can align by subsampling labels.
    """
    x = np.asarray(x, dtype=np.float32)
    y_windows = np.asarray(y_windows, dtype=np.int64).reshape(-1)
    n_times, n_ch = x.shape
    if epoch_samples <= 0 or stride_samples <= 0:
        raise ValueError("epoch_samples/stride_samples must be > 0")
    n_expected = int((n_times - int(epoch_samples)) // int(stride_samples) + 1)
    if n_expected <= 0:
        raise ValueError(f"Signal too short for epoching: n_times={n_times} epoch_samples={epoch_samples}")
    n_labels = int(y_windows.shape[0])
    if n_labels != n_expected:
        # Common BCI-III case: labels are provided at 0.5s step (stride=256), but we extract non-overlapping 1s windows.
        # Then: n_labels ~= 2*n_expected - 1  (because 0, 0.5, 1.0, ..., last)
        if int(stride_samples) == int(epoch_samples) and abs(int(n_labels) - int(2 * n_expected - 1)) <= 1:
            y_sub = y_windows[::2]
            n_labels_sub = int(y_sub.shape[0])
            min_n = min(n_labels_sub, n_expected)
            print(
                f"[WARN] Window-label mismatch: labels={n_labels} expected={n_expected} (stride={stride_samples}). "
                f"Assuming `.asc` labels are at 0.5s step; using every 2nd label -> labels={n_labels_sub}. "
                f"Aligning to min_n={min_n}."
            )
            y_windows = y_sub[:min_n]
            n_expected = min_n
        else:
            # Some provided .asc files are off by 1 line; align safely without inventing labels.
            min_n = min(n_labels, n_expected)
            if abs(n_labels - n_expected) <= 1 and min_n > 0:
                print(
                    f"[WARN] Window-label mismatch by 1: labels={n_labels} expected={n_expected}. "
                    f"Aligning to min_n={min_n}."
                )
                y_windows = y_windows[:min_n]
                n_expected = min_n
            else:
                raise ValueError(
                    f"Window-label mismatch: labels={n_labels} expected={n_expected} "
                    f"(n_times={n_times}, epoch_samples={epoch_samples}, stride={stride_samples})"
                )

    epochs = np.zeros((n_expected, n_ch, int(epoch_samples)), dtype=np.float32)
    for i in range(n_expected):
        s = int(i * int(stride_samples))
        e = int(s + int(epoch_samples))
        epochs[i] = x[s:e].T
    return epochs, y_windows.astype(np.int64, copy=False)


def stratified_split(labels: np.ndarray, *, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx_parts = []
    test_idx_parts = []
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        rng.shuffle(idx)
        n_train = int(round(train_ratio * len(idx)))
        n_train = max(1, min(len(idx) - 1, n_train))
        train_idx_parts.append(idx[:n_train])
        test_idx_parts.append(idx[n_train:])
    return np.concatenate(train_idx_parts), np.concatenate(test_idx_parts)


def stratified_group_subsample_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    train_fraction: float,
    seed: int,
) -> np.ndarray:
    """
    Select a stratified subset of groups and return sample indices.
    Useful to keep held-out training size closer to CV fold-train size.
    """
    labels = np.asarray(labels, dtype=int).reshape(-1)
    groups = np.asarray(groups, dtype=int).reshape(-1)
    if labels.shape[0] != groups.shape[0]:
        raise ValueError(f"labels/groups mismatch: {labels.shape} vs {groups.shape}")
    frac = float(train_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"train_fraction must be in (0,1], got {train_fraction}")
    if frac >= 0.999999:
        return np.arange(labels.shape[0], dtype=int)

    uniq_groups = np.unique(groups)
    group_to_label: dict[int, int] = {}
    for g in uniq_groups.tolist():
        idx = int(np.flatnonzero(groups == int(g))[0])
        group_to_label[int(g)] = int(labels[idx])

    rng = np.random.default_rng(int(seed))
    keep_groups: list[int] = []
    class_vals = np.unique(labels)
    for cls in class_vals.tolist():
        cls_groups = np.asarray([int(g) for g in uniq_groups.tolist() if group_to_label[int(g)] == int(cls)], dtype=int)
        if cls_groups.size == 0:
            continue
        n_keep = int(round(frac * int(cls_groups.size)))
        n_keep = max(1, min(int(cls_groups.size), int(n_keep)))
        pick = rng.choice(cls_groups, size=int(n_keep), replace=False)
        keep_groups.extend(int(v) for v in np.asarray(pick, dtype=int).tolist())

    keep_set = set(keep_groups)
    idx_keep = np.flatnonzero(np.isin(groups, list(keep_set))).astype(int)
    if idx_keep.size == 0:
        raise ValueError("stratified_group_subsample_indices produced empty selection.")
    return idx_keep


def balanced_train_indices(
    labels: np.ndarray,
    *,
    seed: int,
    target_ratio: float = 1.0,
) -> np.ndarray:
    """
    Undersample TRAIN to balance classes.
    target_ratio=1.0 keeps min-class count for all classes (strict balance).
    """
    y = np.asarray(labels, dtype=int).reshape(-1)
    if y.size == 0:
        return np.zeros((0,), dtype=int)
    classes = np.unique(y)
    counts = np.asarray([(y == int(c)).sum() for c in classes.tolist()], dtype=int)
    min_count = int(np.min(counts))
    ratio = float(target_ratio)
    ratio = max(0.1, min(1.0, ratio))
    target_n = max(1, int(round(ratio * float(min_count))))

    rng = np.random.default_rng(int(seed))
    idx_parts: list[np.ndarray] = []
    for cls in classes.tolist():
        idx = np.flatnonzero(y == int(cls)).astype(int)
        if idx.size <= target_n:
            idx_parts.append(idx)
            continue
        pick = rng.choice(idx, size=int(target_n), replace=False)
        idx_parts.append(np.asarray(pick, dtype=int))
    out = np.concatenate(idx_parts).astype(int) if idx_parts else np.zeros((0,), dtype=int)
    rng.shuffle(out)
    return out


def _class_balance(labels: np.ndarray) -> dict[int, int]:
    return {int(k): int((labels == k).sum()) for k in np.unique(labels)}


# --------------------------------------------------------------------------------------
# Hybrid builder (DDPM/SGM)
# --------------------------------------------------------------------------------------


def build_hybrid_epochs_ddpm(
    epochs_real: np.ndarray,
    *,
    specs: list[ExperimentSpec],
    trained_models: dict[str, nn.Module],
    trained_input_idxs: dict[str, tuple[int, ...]],
    gen_batch_size: int,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    betas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Reemplaza canales target por reconstrucción DDPM (sin per-epoch z-score)."""
    # IMPORTANT:
    # Use the ORIGINAL real epochs for conditioning inputs, even if some input channels are also targets.
    # Otherwise, synthesis becomes cascaded (later targets condition on previously synthesized channels),
    # which can inflate TEST vs CV gaps and make hybrid results unstable.
    epochs_h = epochs_real.copy()
    n = epochs_h.shape[0]
    t_len = epochs_h.shape[2]

    for spec in specs:
        target = spec.target
        if target not in trained_models:
            print(f"[WARN] No model found for {target}; skipping.")
            continue

        model = trained_models[target]
        input_idxs = trained_input_idxs[target]
        target_idx = CHANNEL_TO_IDX[target]

        pbar = tqdm(total=n, desc=f"DDPM Synthesize {target}", leave=False)
        for s in range(0, n, gen_batch_size):
            e = min(n, s + gen_batch_size)
            # Condition ONLY on real inputs (no cascade).
            batch_real = epochs_real[s:e]  # (B, C, T)
            cond = np.asarray(batch_real[:, list(input_idxs), :], dtype=np.float32)  # (B, n_cond, T)
            cond_t = torch.from_numpy(cond).to(device)

            with torch.no_grad():
                syn = sample_from_model(
                    model,
                    cond_t,
                    alphas,
                    alphas_cumprod,
                    betas,
                    sqrt_one_minus_alphas_cumprod,
                    shape=(cond_t.size(0), 1, t_len),
                    device=device,
                )

            syn_np = syn.detach().cpu().numpy()[:, 0]  # (B, T)
            syn_np = np.asarray(syn_np, dtype=np.float32)
            epochs_h[s:e, target_idx, :] = syn_np
            pbar.update(e - s)
        pbar.close()

    return epochs_h


def build_partial_hybrid_epochs_ddpm(
    epochs_real: np.ndarray,
    *,
    specs: list[ExperimentSpec],
    trained_models: dict[str, nn.Module],
    trained_input_idxs: dict[str, tuple[int, ...]],
    gen_batch_size: int,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    betas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    device: torch.device,
    replace_ratio: float,
    seed: int,
) -> np.ndarray:
    """
    PARTIAL-HYBRID: replace only a subset of epochs in each target channel.
    The remaining epochs stay real. This is evaluation of channel replacement (not augmentation).
    """
    # Same rule as full-hybrid: always condition on ORIGINAL real channels.
    epochs_h = epochs_real.copy()
    n = int(epochs_h.shape[0])
    if n <= 0:
        return epochs_h
    replace_ratio = float(replace_ratio)
    replace_ratio = max(0.0, min(1.0, replace_ratio))
    k = int(round(replace_ratio * n))
    if k <= 0:
        return epochs_h

    rng = np.random.default_rng(int(seed))
    replace_idx = np.sort(rng.choice(np.arange(n, dtype=int), size=k, replace=False))
    t_len = int(epochs_h.shape[2])
    for spec in specs:
        target = spec.target
        if target not in trained_models:
            print(f"[WARN] No model found for {target}; skipping.")
            continue
        model = trained_models[target]
        input_idxs = trained_input_idxs[target]
        target_idx = CHANNEL_TO_IDX[target]

        pbar = tqdm(total=int(replace_idx.size), desc=f"DDPM Partial Synthesize {target}", leave=False)
        # batch over the selected indices
        for s0 in range(0, int(replace_idx.size), int(gen_batch_size)):
            idx_batch = replace_idx[s0 : s0 + int(gen_batch_size)]
            batch_real = epochs_real[idx_batch]  # (B, C, T)
            cond = np.asarray(batch_real[:, list(input_idxs), :], dtype=np.float32)  # (B, n_cond, T)
            cond_t = torch.from_numpy(cond).to(device)
            with torch.no_grad():
                syn = sample_from_model(
                    model,
                    cond_t,
                    alphas,
                    alphas_cumprod,
                    betas,
                    sqrt_one_minus_alphas_cumprod,
                    shape=(cond_t.size(0), 1, t_len),
                    device=device,
                )
            syn_np = syn.detach().cpu().numpy()[:, 0]
            syn_np = np.asarray(syn_np, dtype=np.float32)
            epochs_h[idx_batch, target_idx, :] = syn_np
            pbar.update(int(idx_batch.size))
        pbar.close()

    return epochs_h


# --------------------------------------------------------------------------------------
# Classifiers: CSP+LDA + EEGNet
# --------------------------------------------------------------------------------------


def _label_classes(*ys: np.ndarray) -> np.ndarray:
    classes = np.unique(np.concatenate([np.asarray(y, dtype=int).reshape(-1) for y in ys]))
    classes = np.asarray(sorted(int(c) for c in classes), dtype=int)
    return classes


def _encode_labels_with_classes(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    label_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}
    return np.asarray([label_to_idx[int(v)] for v in np.asarray(y, dtype=int).reshape(-1)], dtype=int)


def run_csp_lda(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_real: np.ndarray,
    X_test_hybrid: np.ndarray,
    y_test: np.ndarray,
    csp_n_components: int = 4,
    csp_primary_reg: Optional[str] = "ledoit_wolf",
    csp_fallback_reg: Optional[str] = "oas",
    csp_primary_cov_est: str = "concat",
    csp_fallback_cov_est: str = "concat",
    csp_primary_norm_trace: bool = False,
    csp_fallback_norm_trace: bool = False,
) -> dict[str, dict[str, float]]:
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, cohen_kappa_score
    from sklearn.model_selection import StratifiedKFold

    # Stable CSP setup for hybrid-shifted data.
    # `norm_trace=False` avoids over-normalizing covariance scales, which can destabilize CSP.
    CSP_CFG_PRIMARY = dict(
        n_components=int(csp_n_components),
        reg=csp_primary_reg,
        log=True,
        norm_trace=bool(csp_primary_norm_trace),
        cov_est=str(csp_primary_cov_est),
    )
    CSP_CFG_FALLBACK = dict(
        n_components=int(csp_n_components),
        reg=csp_fallback_reg,
        log=True,
        norm_trace=bool(csp_fallback_norm_trace),
        cov_est=str(csp_fallback_cov_est),
    )
    LDA_CFG = dict(solver="lsqr", shrinkage="auto")

    classes = _label_classes(y_train, y_test)
    y_train_enc = _encode_labels_with_classes(y_train, classes)
    y_test_enc = _encode_labels_with_classes(y_test, classes)

    print("classes (original labels):", classes.tolist())
    print("TRAIN:", X_train.shape, "TEST(real):", X_test_real.shape, "TEST(hybrid):", X_test_hybrid.shape)

    def _fit(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        csp = CSP(**cfg)
        F_tr = csp.fit_transform(X_train, y_train_enc)
        F_r = csp.transform(X_test_real)
        F_h = csp.transform(X_test_hybrid)
        return F_tr, F_r, F_h

    try:
        F_train, F_test_real, F_test_hybrid = _fit(CSP_CFG_PRIMARY)
    except Exception as e:
        print(f"[WARN] CSP primary config failed ({e}); retrying fallback CSP config.")
        F_train, F_test_real, F_test_hybrid = _fit(CSP_CFG_FALLBACK)
    if not np.isfinite(F_train).all() or not np.isfinite(F_test_real).all() or not np.isfinite(F_test_hybrid).all():
        try:
            print("[WARN] Non-finite CSP features detected; retrying fallback CSP config.")
            F_train, F_test_real, F_test_hybrid = _fit(CSP_CFG_FALLBACK)
        except Exception:
            print("[WARN] Fallback failed; applying nan_to_num on CSP features.")
            F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
            F_test_real = np.nan_to_num(F_test_real, nan=0.0, posinf=0.0, neginf=0.0)
            F_test_hybrid = np.nan_to_num(F_test_hybrid, nan=0.0, posinf=0.0, neginf=0.0)
    # Degenerate CSP projections (very low variance/rank) can collapse LDA to one class.
    if (np.std(F_train, axis=0) < 1e-8).any() or np.linalg.matrix_rank(F_train) < min(F_train.shape):
        try:
            print("[WARN] Degenerate CSP train features; retrying fallback CSP config.")
            F_train, F_test_real, F_test_hybrid = _fit(CSP_CFG_FALLBACK)
        except Exception:
            pass

    lda = LinearDiscriminantAnalysis(**LDA_CFG)
    decision_threshold: Optional[float] = None
    if int(np.unique(y_train_enc).size) == 2 and int(F_train.shape[0]) >= 20:
        # More stable threshold calibration using out-of-fold train probabilities.
        n_splits = min(5, int(np.min(np.bincount(y_train_enc))))
        if n_splits >= 2:
            skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=42)
            p_oof = np.zeros((int(F_train.shape[0]),), dtype=np.float64)
            for tr_i, va_i in skf.split(F_train, y_train_enc):
                lda_cal = LinearDiscriminantAnalysis(**LDA_CFG)
                lda_cal.fit(F_train[tr_i], y_train_enc[tr_i])
                p_oof[np.asarray(va_i, dtype=int)] = lda_cal.predict_proba(F_train[va_i])[:, 1]
            thr_grid = np.linspace(0.1, 0.9, 81, dtype=np.float64)
            best_thr = 0.5
            best_bal = -np.inf
            for thr in thr_grid.tolist():
                pred_oof = (p_oof >= float(thr)).astype(int)
                bal = float(balanced_accuracy_score(y_train_enc, pred_oof))
                if bal > best_bal:
                    best_bal = bal
                    best_thr = float(thr)
            decision_threshold = float(best_thr)
            print(f"[CSP] calibrated binary threshold={decision_threshold:.3f} (oof_bal_acc={best_bal:.3f})")
    lda.fit(F_train, y_train_enc)
    p_real = lda.predict_proba(F_test_real)[:, 1] if int(np.unique(y_train_enc).size) == 2 else None
    p_hyb = lda.predict_proba(F_test_hybrid)[:, 1] if int(np.unique(y_train_enc).size) == 2 else None
    if decision_threshold is None or p_real is None or p_hyb is None:
        pred_real = lda.predict(F_test_real)
        pred_hybrid = lda.predict(F_test_hybrid)
    else:
        pred_real = (p_real >= float(decision_threshold)).astype(int)
        pred_hybrid = (p_hyb >= float(decision_threshold)).astype(int)
        # Guardrail: avoid degenerate one-class predictions at test time.
        if int(np.unique(pred_real).size) < 2 or int(np.unique(pred_hybrid).size) < 2:
            print("[CSP] degenerate one-class prediction detected; applying threshold guard.")
            thr_guard = 0.5
            pred_real = (p_real >= float(thr_guard)).astype(int)
            pred_hybrid = (p_hyb >= float(thr_guard)).astype(int)
            if int(np.unique(pred_real).size) < 2 or int(np.unique(pred_hybrid).size) < 2:
                prior_pos = float(np.mean(y_train_enc == 1))
                prior_pos = min(0.95, max(0.05, prior_pos))
                thr_real = float(np.quantile(p_real, 1.0 - prior_pos))
                thr_hyb = float(np.quantile(p_hyb, 1.0 - prior_pos))
                pred_real = (p_real >= thr_real).astype(int)
                pred_hybrid = (p_hyb >= thr_hyb).astype(int)
                print(
                    f"[CSP] prior-rate threshold guard applied (prior_pos={prior_pos:.3f}, "
                    f"thr_real={thr_real:.3f}, thr_hyb={thr_hyb:.3f})."
                )

    acc_real = float(accuracy_score(y_test_enc, pred_real))
    bal_real = float(balanced_accuracy_score(y_test_enc, pred_real))
    kappa_real = float(cohen_kappa_score(y_test_enc, pred_real))
    acc_h = float(accuracy_score(y_test_enc, pred_hybrid))
    bal_h = float(balanced_accuracy_score(y_test_enc, pred_hybrid))
    kappa_h = float(cohen_kappa_score(y_test_enc, pred_hybrid))

    print("\n=== CSP+LDA Results ===")
    print(f"TEST(real):   acc={acc_real:.3f} | bal_acc={bal_real:.3f} | kappa={kappa_real:.3f}")
    print(f"TEST(hybrid): acc={acc_h:.3f} | bal_acc={bal_h:.3f} | kappa={kappa_h:.3f}")

    cm_real = confusion_matrix(y_test_enc, pred_real, normalize="true")
    cm_h = confusion_matrix(y_test_enc, pred_hybrid, normalize="true")
    print("\nConfusionMatrix (normalize=true) TEST(real):\n", np.array2string(cm_real, precision=3, floatmode="fixed"))
    print("\nConfusionMatrix (normalize=true) TEST(hybrid):\n", np.array2string(cm_h, precision=3, floatmode="fixed"))

    return {
        "no_cv": {
            "acc_real": acc_real,
            "acc_hybrid": acc_h,
            "bal_acc_real": bal_real,
            "bal_acc_hybrid": bal_h,
            "kappa_real": kappa_real,
            "kappa_hybrid": kappa_h,
        },
        "cv": {},
    }


def run_fbcsp_lda(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_real: np.ndarray,
    X_test_hybrid: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
    fbcsp_n_components: int = 4,
    fbcsp_primary_reg: Optional[str] = "ledoit_wolf",
    fbcsp_fallback_reg: Optional[str] = "oas",
    fbcsp_primary_cov_est: str = "concat",
    fbcsp_fallback_cov_est: str = "concat",
    fbcsp_primary_norm_trace: bool = False,
    fbcsp_fallback_norm_trace: bool = False,
    fbcsp_k_best: int = 4,
) -> dict[str, dict[str, float]]:
    """
    FBCSP + LDA (adaptado del notebook EEGNet_2a.ipynb), sin plots.
    Mantiene bandas 8–30 Hz y selección MIBIF (top-k + pares).
    """

    from mne.decoding import CSP
    from mne.filter import filter_data
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict, cross_val_score

    seed = 42

    # MNE `filter_data` puede ser estricto con el dtype en algunas versiones.
    # Forzamos float64 (real) para el filtrado y luego CSP/LDA trabajan normal.
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test_real = np.asarray(X_test_real, dtype=np.float64)
    X_test_hybrid = np.asarray(X_test_hybrid, dtype=np.float64)

    # Bandas (≈4 Hz) restringidas a 8–30 Hz
    bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30)]

    classes = _label_classes(y_train, y_test)
    y_train_enc = _encode_labels_with_classes(y_train, classes)
    y_test_enc = _encode_labels_with_classes(y_test, classes)

    # Keep symmetric pairs; n_components is configurable (2/4/6 recommended).
    m_pairs = max(1, int(fbcsp_n_components) // 2)
    per_band = int(2 * m_pairs)  # features por banda

    # Stable FBCSP config, now configurable from CLI.
    CSP_CFG_PRIMARY = dict(
        n_components=per_band,
        reg=fbcsp_primary_reg,
        log=True,
        norm_trace=bool(fbcsp_primary_norm_trace),
        cov_est=str(fbcsp_primary_cov_est),
    )
    CSP_CFG_FALLBACK = dict(
        n_components=per_band,
        reg=fbcsp_fallback_reg,
        log=True,
        norm_trace=bool(fbcsp_fallback_norm_trace),
        cov_est=str(fbcsp_fallback_cov_est),
    )
    LDA_CFG = dict(solver="lsqr", shrinkage="auto")

    # ---- Fit CSP per band on TRAIN once, transform both TEST sets with same CSPs
    csp_per_band: list[CSP] = []
    F_train_list: list[np.ndarray] = []
    F_test_real_list: list[np.ndarray] = []
    F_test_hybrid_list: list[np.ndarray] = []

    for (l_f, h_f) in bands:
        Xtr_b = filter_data(X_train, sfreq, l_f, h_f, method="iir", verbose=False)
        Xte_r_b = filter_data(X_test_real, sfreq, l_f, h_f, method="iir", verbose=False)
        Xte_h_b = filter_data(X_test_hybrid, sfreq, l_f, h_f, method="iir", verbose=False)

        try:
            csp_b = CSP(**CSP_CFG_PRIMARY)
            Ftr_b = csp_b.fit_transform(Xtr_b, y_train_enc)
            Fr_b = csp_b.transform(Xte_r_b)
            Fh_b = csp_b.transform(Xte_h_b)
        except Exception as e:
            print(
                f"[WARN][FBCSP] band {l_f}-{h_f} primary failed ({e}); "
                "retrying with OAS fallback."
            )
            csp_b = CSP(**CSP_CFG_FALLBACK)
            Ftr_b = csp_b.fit_transform(Xtr_b, y_train_enc)
            Fr_b = csp_b.transform(Xte_r_b)
            Fh_b = csp_b.transform(Xte_h_b)
        if (not np.isfinite(Ftr_b).all()) or (not np.isfinite(Fr_b).all()) or (not np.isfinite(Fh_b).all()):
            print(f"[WARN][FBCSP] non-finite features in band {l_f}-{h_f}; applying nan_to_num.")
            Ftr_b = np.nan_to_num(Ftr_b, nan=0.0, posinf=0.0, neginf=0.0)
            Fr_b = np.nan_to_num(Fr_b, nan=0.0, posinf=0.0, neginf=0.0)
            Fh_b = np.nan_to_num(Fh_b, nan=0.0, posinf=0.0, neginf=0.0)
        F_train_list.append(Ftr_b)
        F_test_real_list.append(Fr_b)
        F_test_hybrid_list.append(Fh_b)
        csp_per_band.append(csp_b)

    F_train_full = np.concatenate(F_train_list, axis=1)
    F_test_real_full = np.concatenate(F_test_real_list, axis=1)
    F_test_hybrid_full = np.concatenate(F_test_hybrid_list, axis=1)

    # MIBIF selection (top-k + pares) usando SOLO TRAIN features
    # Best-known config: k_best=8 + include symmetric pairs
    k_best = int(fbcsp_k_best)
    mi = mutual_info_classif(F_train_full, y_train_enc, random_state=seed)
    order = np.argsort(mi)[::-1]
    topk = order[: min(int(k_best), int(order.size))].tolist()

    selected = set(int(i) for i in topk)
    for idx in topk:
        idx = int(idx)
        band_idx = idx // per_band
        j = idx % per_band
        pair_j = int(per_band - 1 - j)
        selected.add(int(band_idx * per_band + pair_j))

    selected_idx = sorted(selected)
    F_train = F_train_full[:, selected_idx]
    F_test_real = F_test_real_full[:, selected_idx]
    F_test_hybrid = F_test_hybrid_full[:, selected_idx]

    lda = LinearDiscriminantAnalysis(**LDA_CFG)
    decision_threshold: Optional[float] = None
    if int(np.unique(y_train_enc).size) == 2 and int(F_train.shape[0]) >= 20:
        n_splits = min(5, int(np.min(np.bincount(y_train_enc))))
        if n_splits >= 2:
            skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=42)
            p_oof = np.zeros((int(F_train.shape[0]),), dtype=np.float64)
            for tr_i, va_i in skf.split(F_train, y_train_enc):
                lda_cal = LinearDiscriminantAnalysis(**LDA_CFG)
                lda_cal.fit(F_train[tr_i], y_train_enc[tr_i])
                p_oof[np.asarray(va_i, dtype=int)] = lda_cal.predict_proba(F_train[va_i])[:, 1]
            thr_grid = np.linspace(0.1, 0.9, 81, dtype=np.float64)
            best_thr = 0.5
            best_bal = -np.inf
            for thr in thr_grid.tolist():
                pred_oof = (p_oof >= float(thr)).astype(int)
                bal = float(balanced_accuracy_score(y_train_enc, pred_oof))
                if bal > best_bal:
                    best_bal = bal
                    best_thr = float(thr)
            decision_threshold = float(best_thr)
            print(f"[FBCSP] calibrated binary threshold={decision_threshold:.3f} (oof_bal_acc={best_bal:.3f})")

    lda.fit(F_train, y_train_enc)
    p_real = lda.predict_proba(F_test_real)[:, 1] if int(np.unique(y_train_enc).size) == 2 else None
    p_hyb = lda.predict_proba(F_test_hybrid)[:, 1] if int(np.unique(y_train_enc).size) == 2 else None
    if decision_threshold is None or p_real is None or p_hyb is None:
        pred_real = lda.predict(F_test_real)
        pred_h = lda.predict(F_test_hybrid)
    else:
        pred_real = (p_real >= float(decision_threshold)).astype(int)
        pred_h = (p_hyb >= float(decision_threshold)).astype(int)
        if int(np.unique(pred_real).size) < 2 or int(np.unique(pred_h).size) < 2:
            print("[FBCSP] degenerate one-class prediction detected; applying threshold guard.")
            thr_guard = 0.5
            pred_real = (p_real >= float(thr_guard)).astype(int)
            pred_h = (p_hyb >= float(thr_guard)).astype(int)
            if int(np.unique(pred_real).size) < 2 or int(np.unique(pred_h).size) < 2:
                prior_pos = float(np.mean(y_train_enc == 1))
                prior_pos = min(0.95, max(0.05, prior_pos))
                thr_real = float(np.quantile(p_real, 1.0 - prior_pos))
                thr_hyb = float(np.quantile(p_hyb, 1.0 - prior_pos))
                pred_real = (p_real >= thr_real).astype(int)
                pred_h = (p_hyb >= thr_hyb).astype(int)
                print(
                    f"[FBCSP] prior-rate threshold guard applied (prior_pos={prior_pos:.3f}, "
                    f"thr_real={thr_real:.3f}, thr_hyb={thr_hyb:.3f})."
                )

    acc_real = float(accuracy_score(y_test_enc, pred_real))
    bal_real = float(balanced_accuracy_score(y_test_enc, pred_real))
    kappa_real = float(cohen_kappa_score(y_test_enc, pred_real))

    acc_h = float(accuracy_score(y_test_enc, pred_h))
    bal_h = float(balanced_accuracy_score(y_test_enc, pred_h))
    kappa_h = float(cohen_kappa_score(y_test_enc, pred_h))

    print("\n=== FBCSP+LDA Results ===")
    print(f"TEST(real):   acc={acc_real:.3f} | bal_acc={bal_real:.3f} | kappa={kappa_real:.3f}")
    print(f"TEST(hybrid): acc={acc_h:.3f} | bal_acc={bal_h:.3f} | kappa={kappa_h:.3f}")

    cm_real = confusion_matrix(y_test_enc, pred_real, normalize="true")
    cm_h = confusion_matrix(y_test_enc, pred_h, normalize="true")
    print("\nConfusionMatrix (normalize=true) TEST(real):\n", np.array2string(cm_real, precision=3, floatmode="fixed"))
    print("\nConfusionMatrix (normalize=true) TEST(hybrid):\n", np.array2string(cm_h, precision=3, floatmode="fixed"))

    return {
        "no_cv": {
            "acc_real": acc_real,
            "acc_hybrid": acc_h,
            "bal_acc_real": bal_real,
            "bal_acc_hybrid": bal_h,
            "kappa_real": kappa_real,
            "kappa_hybrid": kappa_h,
        },
        "cv": {},
    }


class EEGNetModel(nn.Module):
    """
    EEGNet (estructura de EEGNet_2a.ipynb), adaptado a (B, 1, C, T).
    Mantiene forward_features para embeddings.
    """

    def __init__(
        self,
        chans: int,
        classes: int,
        time_points: int,
        *,
        temp_kernel: int = 32,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        pk1: int = 4,
        pk2: int = 8,
        dropout_rate: float = 0.3,
        max_norm1: float = 1.0,
        max_norm2: float = 0.25,
    ):
        super().__init__()

        self.max_norm1 = float(max_norm1)
        self.max_norm2 = float(max_norm2)

        depthwise_ch = int(d * f1)
        linear_size = int((time_points // (pk1 * pk2)) * f2)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding="same", bias=False),
            nn.BatchNorm2d(f1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(f1, depthwise_ch, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(depthwise_ch),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(depthwise_ch, depthwise_ch, (1, 16), groups=depthwise_ch, padding="same", bias=False),
            nn.Conv2d(depthwise_ch, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)

        self.apply_constraints()

    def _apply_max_norm(self, layer: nn.Module, max_norm: float) -> None:
        with torch.no_grad():
            for name, param in layer.named_parameters():
                if "weight" in name:
                    param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=float(max_norm))

    def apply_constraints(self) -> None:
        self._apply_max_norm(self.block2[0], self.max_norm1)
        self._apply_max_norm(self.fc, self.max_norm2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.fc(feats)


def _stratified_train_val_indices(y: np.ndarray, *, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    n = int(y.shape[0])
    if n <= 1:
        return np.arange(n, dtype=int), np.zeros((0,), dtype=int)
    ratio = float(val_ratio)
    ratio = min(0.5, max(0.05, ratio))
    rng = np.random.default_rng(int(seed))
    tr_parts: list[np.ndarray] = []
    va_parts: list[np.ndarray] = []
    for cls in np.unique(y).tolist():
        idx = np.flatnonzero(y == int(cls)).astype(int)
        rng.shuffle(idx)
        if idx.size <= 1:
            tr_parts.append(idx)
            continue
        n_val = int(round(ratio * int(idx.size)))
        n_val = max(1, min(int(idx.size) - 1, n_val))
        va_parts.append(idx[:n_val])
        tr_parts.append(idx[n_val:])
    tr_idx = np.concatenate(tr_parts) if tr_parts else np.arange(n, dtype=int)
    va_idx = np.concatenate(va_parts) if va_parts else np.zeros((0,), dtype=int)
    if va_idx.size == 0:
        return np.arange(n, dtype=int), np.zeros((0,), dtype=int)
    return np.asarray(tr_idx, dtype=int), np.asarray(va_idx, dtype=int)


def _compute_class_weights_from_labels(y: np.ndarray, n_classes: int) -> torch.Tensor:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    counts = np.bincount(y, minlength=int(n_classes)).astype(np.float64)
    counts[counts <= 0.0] = 1.0
    # Inverse-frequency weights normalized to mean 1 for stable loss scale.
    w = (float(y.size) / (float(n_classes) * counts)).astype(np.float32)
    w = w / max(1e-6, float(np.mean(w)))
    return torch.tensor(w, dtype=torch.float32)


def run_eegnet(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_real: np.ndarray,
    X_test_hybrid: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    eegnet_lr: float = 1e-4,
    eegnet_dropout: float = 0.3,
    eegnet_patience: int = 80,
) -> dict[str, dict[str, float]]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score

    classes = _label_classes(y_train, y_test)
    y_train_enc = _encode_labels_with_classes(y_train, classes)
    y_test_enc = _encode_labels_with_classes(y_test, classes)
    n_classes = int(np.max(np.concatenate([y_train_enc, y_test_enc])) + 1)

    # Normalización robusta (por canal) usando stats del TRAIN real.
    # Esto reduce el domain shift que puede hacer colapsar EEGNet en híbrido.
    X_train_f = np.asarray(X_train, dtype=np.float32)
    X_test_real_f = np.asarray(X_test_real, dtype=np.float32)
    X_test_hybrid_f = np.asarray(X_test_hybrid, dtype=np.float32)

    mu_ch = X_train_f.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    sd_ch = X_train_f.std(axis=(0, 2), keepdims=True) + 1e-6

    X_train_f = (X_train_f - mu_ch) / sd_ch
    X_test_real_f = (X_test_real_f - mu_ch) / sd_ch
    X_test_hybrid_f = (X_test_hybrid_f - mu_ch) / sd_ch

    # Normalización por-epoch (por canal, sobre el tiempo).
    # Ayuda cuando el híbrido introduce drift de ganancia/offset por ventana.
    def _per_epoch_z(x: np.ndarray) -> np.ndarray:
        mu_t = x.mean(axis=2, keepdims=True)
        sd_t = x.std(axis=2, keepdims=True) + 1e-6
        return (x - mu_t) / sd_t

    X_train_f = _per_epoch_z(X_train_f)
    X_test_real_f = _per_epoch_z(X_test_real_f)
    X_test_hybrid_f = _per_epoch_z(X_test_hybrid_f)

    print("[EEGNet] applied normalization: train-channel z + per-epoch z")

    # EEGNet espera (B, 1, C, T)
    Xtr = X_train_f[:, None, :, :]
    Xte_r = X_test_real_f[:, None, :, :]
    Xte_h = X_test_hybrid_f[:, None, :, :]
    ytr = np.asarray(y_train_enc, dtype=np.int64)
    yte = np.asarray(y_test_enc, dtype=np.int64)

    model = EEGNetModel(
        chans=int(X_train.shape[1]),
        classes=n_classes,
        time_points=int(X_train.shape[2]),
        temp_kernel=32,
        dropout_rate=float(eegnet_dropout),
    ).to(device)
    # Requested config: Adam + configurable LR
    opt = torch.optim.Adam(model.parameters(), lr=float(eegnet_lr), weight_decay=1e-4)

    tr_idx, va_idx = _stratified_train_val_indices(ytr, val_ratio=0.2, seed=42)
    Xtr_tr = Xtr[tr_idx]
    ytr_tr = ytr[tr_idx]
    Xtr_va = Xtr[va_idx] if va_idx.size > 0 else None
    ytr_va = ytr[va_idx] if va_idx.size > 0 else None

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr_tr), torch.from_numpy(ytr_tr)),
        batch_size=int(batch_size),
        shuffle=True,
    )
    class_w = _compute_class_weights_from_labels(ytr_tr, n_classes=int(n_classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    print(f"[EEGNet] class weights: {class_w.detach().cpu().numpy().round(4).tolist()}")
    val_loader = None
    if Xtr_va is not None and ytr_va is not None and int(ytr_va.shape[0]) > 0:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr_va), torch.from_numpy(ytr_va)),
            batch_size=int(batch_size),
            shuffle=False,
        )

    pbar = tqdm(range(int(epochs)), desc="EEGNet train", leave=True, mininterval=0.5)
    best_val_loss = float("inf")
    best_state: Optional[dict[str, torch.Tensor]] = None
    patience = int(eegnet_patience)
    min_delta = 1e-4
    wait = 0
    min_epochs_before_stop = 120
    for ep in pbar:
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            model.apply_constraints()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        if val_loader is None:
            pbar.set_postfix(loss=f"{train_loss:.4f}")
            continue
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_losses.append(float(criterion(logits, yb).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < (best_val_loss - float(min_delta)):
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        pbar.set_postfix(loss=f"{train_loss:.4f}", val=f"{val_loss:.4f}", wait=str(wait))
        if (int(ep) + 1) >= int(min_epochs_before_stop) and wait >= int(patience):
            print(f"[EEGNet] early stopping at epoch {int(ep)+1}, best_val_loss={best_val_loss:.4f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    def _eval(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        model.eval()
        y_pred_parts = []
        with torch.no_grad():
            for i in range(0, x.shape[0], int(batch_size)):
                xb = torch.from_numpy(x[i : i + int(batch_size)]).to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                y_pred_parts.append(torch.argmax(probs, dim=1).detach().cpu().numpy())
        y_pred = np.concatenate(y_pred_parts, axis=0)
        acc = float(accuracy_score(y, y_pred))
        bal_acc = float(balanced_accuracy_score(y, y_pred))
        kappa = float(cohen_kappa_score(y, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y, y_pred, normalize="true")
        return {
            "acc": acc,
            "bal_acc": bal_acc,
            "kappa": kappa,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "cm": cm,
        }

    m_real = _eval(Xte_r, yte)
    m_h = _eval(Xte_h, yte)

    print("\n=== EEGNet Results (macro) ===")
    print("TEST(real):  ", {k: v for k, v in m_real.items() if k != "cm"})
    print("TEST(hybrid):", {k: v for k, v in m_h.items() if k != "cm"})
    print("\nConfusionMatrix (normalize=true) TEST(real):\n", np.array2string(m_real["cm"], precision=3, floatmode="fixed"))
    print("\nConfusionMatrix (normalize=true) TEST(hybrid):\n", np.array2string(m_h["cm"], precision=3, floatmode="fixed"))

    return {
        "no_cv": {
            "acc_real": float(m_real["acc"]),
            "acc_hybrid": float(m_h["acc"]),
            "bal_acc_real": float(m_real["bal_acc"]),
            "bal_acc_hybrid": float(m_h["bal_acc"]),
            "kappa_real": float(m_real["kappa"]),
            "kappa_hybrid": float(m_h["kappa"]),
        },
        "cv": {},
    }


def _train_eegnet_once(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    patience: int = 80,
) -> dict[str, object]:
    """
    Train EEGNet ONCE and return the fitted model + label classes + normalization stats.
    This is used to make the ablation study fair (same weights across REAL/PARTIAL/FULL).
    """
    classes = _label_classes(y_train, y_train)
    y_train_enc = _encode_labels_with_classes(y_train, classes)
    n_classes = int(np.max(y_train_enc) + 1)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    mu_ch = X_train_f.mean(axis=(0, 2), keepdims=True)
    sd_ch = X_train_f.std(axis=(0, 2), keepdims=True) + 1e-6

    def _per_epoch_z(x: np.ndarray) -> np.ndarray:
        mu_t = x.mean(axis=2, keepdims=True)
        sd_t = x.std(axis=2, keepdims=True) + 1e-6
        return (x - mu_t) / sd_t

    X_train_f = (X_train_f - mu_ch) / sd_ch
    X_train_f = _per_epoch_z(X_train_f)

    Xtr = X_train_f[:, None, :, :]
    ytr = np.asarray(y_train_enc, dtype=np.int64)

    model = EEGNetModel(
        chans=int(X_train.shape[1]),
        classes=n_classes,
        time_points=int(X_train.shape[2]),
        temp_kernel=32,
        dropout_rate=float(dropout),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    tr_idx, va_idx = _stratified_train_val_indices(ytr, val_ratio=0.2, seed=42)
    Xtr_tr = Xtr[tr_idx]
    ytr_tr = ytr[tr_idx]
    Xtr_va = Xtr[va_idx] if va_idx.size > 0 else None
    ytr_va = ytr[va_idx] if va_idx.size > 0 else None

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr_tr), torch.from_numpy(ytr_tr)),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    class_w = _compute_class_weights_from_labels(ytr_tr, n_classes=int(n_classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    print(f"[EEGNet single] class weights: {class_w.detach().cpu().numpy().round(4).tolist()}")
    val_loader = None
    if Xtr_va is not None and ytr_va is not None and int(ytr_va.shape[0]) > 0:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr_va), torch.from_numpy(ytr_va)),
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
        )

    pbar = tqdm(range(int(epochs)), desc="EEGNet train (single)", leave=True, mininterval=0.5)
    best_val_loss = float("inf")
    best_state: Optional[dict[str, torch.Tensor]] = None
    patience = int(patience)
    min_delta = 1e-4
    wait = 0
    min_epochs_before_stop = 120
    for ep in pbar:
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            model.apply_constraints()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        if val_loader is None:
            if losses:
                pbar.set_postfix(loss=f"{train_loss:.4f}")
            continue
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_losses.append(float(criterion(logits, yb).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < (best_val_loss - float(min_delta)):
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        pbar.set_postfix(loss=f"{train_loss:.4f}", val=f"{val_loss:.4f}", wait=str(wait))
        if (int(ep) + 1) >= int(min_epochs_before_stop) and wait >= int(patience):
            print(f"[EEGNet single] early stopping at epoch {int(ep)+1}, best_val_loss={best_val_loss:.4f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    return {"model": model.eval(), "classes": classes, "mu_ch": mu_ch, "sd_ch": sd_ch}


def _eval_eegnet_fixed(
    *,
    model: EEGNetModel,
    classes: np.ndarray,
    mu_ch: np.ndarray,
    sd_ch: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    """Evaluate a fixed EEGNet model (no retraining) using the same normalization style as `run_eegnet`."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

    y_test_enc = _encode_labels_with_classes(y_test, classes)

    Xf = np.asarray(X_test, dtype=np.float32)
    Xf = (Xf - np.asarray(mu_ch, dtype=np.float32)) / (np.asarray(sd_ch, dtype=np.float32) + 1e-6)

    mu_t = Xf.mean(axis=2, keepdims=True)
    sd_t = Xf.std(axis=2, keepdims=True) + 1e-6
    Xf = (Xf - mu_t) / sd_t

    Xte = Xf[:, None, :, :]
    yte = np.asarray(y_test_enc, dtype=np.int64)

    model.eval()
    y_pred_parts = []
    with torch.no_grad():
        for i in range(0, Xte.shape[0], int(batch_size)):
            xb = torch.from_numpy(Xte[i : i + int(batch_size)]).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            y_pred_parts.append(torch.argmax(probs, dim=1).detach().cpu().numpy())
    y_pred = np.concatenate(y_pred_parts, axis=0) if y_pred_parts else np.zeros((0,), dtype=int)

    acc = float(accuracy_score(yte, y_pred))
    kappa = float(cohen_kappa_score(yte, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(yte, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(yte, y_pred, normalize="true")
    return {"acc": acc, "kappa": kappa, "precision": float(prec), "recall": float(rec), "f1": float(f1), "cm": cm}



# --------------------------------------------------------------------------------------
# Score-based model (SDE) + EMA + PC sampler
# --------------------------------------------------------------------------------------

class VPSDE:
    """VP SDE (Song et al.) con beta lineal."""

    def __init__(self, *, beta_min: float, beta_max: float, N: int, T: float = 1.0):
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.N = int(N)
        self.T = float(T)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _int_beta(self, t: torch.Tensor) -> torch.Tensor:
        # ∫0^t beta(s) ds for linear beta(s) = beta_min + s*(beta_max-beta_min)
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t**2)

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        int_beta = self._int_beta(t).view(-1, 1, 1)
        mean = x0 * torch.exp(-0.5 * int_beta)
        std = torch.sqrt(1.0 - torch.exp(-int_beta))
        return mean, std

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta_t = self.beta(t).view(-1, 1, 1)
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def prior_sampling(self, shape: tuple[int, int, int], *, device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device)


def ve_marginal_prob_std(t: torch.Tensor, *, sigma: float) -> torch.Tensor:
    """Std of p_{0t}(x(t)|x(0)) for VE-SDE tutorial (Song et al.)."""
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.to(dtype=torch.float32)
    s = float(sigma)
    return torch.sqrt((s ** (2.0 * t) - 1.0) / (2.0 * float(np.log(s))))


def ve_diffusion_coeff(t: torch.Tensor, *, sigma: float) -> torch.Tensor:
    """Diffusion coefficient g(t) for VE-SDE tutorial (Song et al.)."""
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.to(dtype=torch.float32)
    return float(sigma) ** t


class EMA:
    """Exponential Moving Average of parameters (para sampling/metrics)."""

    def __init__(self, beta: float):
        self.beta = float(beta)

    @torch.no_grad()
    def init(self, model: nn.Module) -> dict[str, torch.Tensor]:
        # Incluye buffers (p.ej. BatchNorm.num_batches_tracked es Long).
        # Para EMA solo aplicamos promedio a tensores float; el resto se copia directo.
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, ema_state: dict[str, torch.Tensor], model: nn.Module) -> None:
        msd = model.state_dict()
        b = self.beta
        for k in ema_state.keys():
            cur = msd[k].detach()
            buf = ema_state[k]

            # Solo aplicar EMA a floating point. Buffers Long/Bool se copian.
            if torch.is_floating_point(buf):
                # asegurar dtype compatible
                ema_state[k].mul_(b).add_(cur.to(dtype=buf.dtype), alpha=(1.0 - b))
            else:
                # p.ej. num_batches_tracked (Long)
                if buf.shape == cur.shape and buf.dtype == cur.dtype:
                    buf.copy_(cur)
                else:
                    ema_state[k] = cur.clone()

    @torch.no_grad()
    def copy_to(self, ema_state: dict[str, torch.Tensor], model: nn.Module) -> None:
        model.load_state_dict(ema_state, strict=True)


def _clip_score(score: torch.Tensor, *, max_norm: float) -> torch.Tensor:
    """Clip per-sample L2 norm of score (stability at high noise)."""
    max_norm = float(max_norm)
    if max_norm <= 0:
        return score
    b = int(score.shape[0])
    norms = torch.norm(score.reshape(b, -1), dim=1)  # (B,)
    scales = (max_norm / (norms + 1e-12)).clamp(max=1.0).view(b, 1, 1)
    return score * scales


def train_score_model_vp(
    *,
    model: nn.Module,
    sde: VPSDE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int,
    grad_clip: float,
    ema: EMA,
    ema_state: dict[str, torch.Tensor],
    eval_every: int,
    desc: str,
    t_embed_scale: float | None,
    likelihood_weighting: bool,
    score_clip: float,
) -> None:
    """Entrenamiento por steps (paper-style: k steps)."""
    model.train()
    data_iter = iter(dataloader)

    pbar = tqdm(range(int(steps)), desc=desc, leave=True, mininterval=0.5)
    running = []

    for step in pbar:
        try:
            batch_inputs, batch_targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_inputs, batch_targets = next(data_iter)

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_size = int(batch_targets.size(0))

        # t ~ Uniform(eps, T]
        t = torch.rand(batch_size, device=device) * (sde.T - 1e-5) + 1e-5
        # Embedding time passed to the network:
        # - For DDPM-style UNet: scale t to ~[0..N] to make sinusoidal embedding informative.
        # - For ScoreNet-style UNet (Gaussian Fourier): use continuous t directly.
        t_net = t if t_embed_scale is None else (t * float(t_embed_scale))
        z = torch.randn_like(batch_targets)

        mean, std = sde.marginal_prob(batch_targets, t)
        x_t = mean + std * z

        model_input = torch.cat([batch_inputs, x_t], dim=1)
        optimizer.zero_grad()

        # Score matching target: score ≈ -(x_t - mean)/std^2 = -z/std
        score = model(model_input, t_net)
        score = _clip_score(score, max_norm=float(score_clip))
        per_ex = torch.mean((score * std + z) ** 2, dim=(1, 2))  # (B,)
        if bool(likelihood_weighting):
            # g(t)^2 for VP-SDE = beta(t)
            beta_t = sde.beta(t).detach()
            loss = torch.mean(per_ex * beta_t)
        else:
            loss = torch.mean(per_ex)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        ema.update(ema_state, model)

        loss_val = float(loss.item())
        running.append(loss_val)
        if len(running) > 200:
            running = running[-200:]
        pbar.set_postfix(loss=f"{float(np.mean(running)):.4f}")

        if eval_every > 0 and (step + 1) % int(eval_every) == 0:
            print(f"[{desc}] step {step+1}/{steps} | loss(avg200)={float(np.mean(running)):.4f}")


def train_score_model_ve(
    *,
    model: nn.Module,
    sigma: float,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int,
    grad_clip: float,
    ema: EMA,
    ema_state: dict[str, torch.Tensor],
    eval_every: int,
    desc: str,
    likelihood_weighting: bool,
    score_clip: float,
    t_embed_scale: float | None,
) -> None:
    """VE-SDE tutorial DSM training (Song et al.), step-based."""
    model.train()
    data_iter = iter(dataloader)

    pbar = tqdm(range(int(steps)), desc=desc, leave=True, mininterval=0.5)
    running: list[float] = []

    for step in pbar:
        try:
            batch_inputs, batch_targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_inputs, batch_targets = next(data_iter)

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_size = int(batch_targets.size(0))

        # t ~ Uniform(eps, 1]
        t = torch.rand(batch_size, device=device) * (1.0 - 1e-5) + 1e-5
        z = torch.randn_like(batch_targets)

        std = ve_marginal_prob_std(t, sigma=float(sigma)).view(batch_size, 1, 1)
        x_t = batch_targets + std * z

        model_input = torch.cat([batch_inputs, x_t], dim=1)
        optimizer.zero_grad()
        t_net = t if t_embed_scale is None else (t * float(t_embed_scale))
        score = model(model_input, t_net)
        score = _clip_score(score, max_norm=float(score_clip))

        # Tutorial loss: sum over dims, then mean over batch
        per_ex = torch.sum((score * std + z) ** 2, dim=(1, 2))
        if bool(likelihood_weighting):
            g = ve_diffusion_coeff(t, sigma=float(sigma)).detach()
            loss = torch.mean(per_ex * (g**2))
        else:
            loss = torch.mean(per_ex)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        ema.update(ema_state, model)

        loss_val = float(loss.item())
        running.append(loss_val)
        if len(running) > 200:
            running = running[-200:]
        pbar.set_postfix(loss=f"{float(np.mean(running)):.4f}")

        if eval_every > 0 and (step + 1) % int(eval_every) == 0:
            print(f"[{desc}] step {step+1}/{steps} | loss(avg200)={float(np.mean(running)):.4f}")


@torch.no_grad()
def pc_sampler_vp(
    *,
    model: nn.Module,
    sde: VPSDE,
    conditioning_inputs: torch.Tensor,
    shape: tuple[int, int, int],
    device: torch.device,
    sampling_N: int,
    eps: float,
    snr: float,
    n_steps_each: int,
    noise_removal: bool,
    t_embed_scale: float | None,
    init_corrector_steps: int,
    score_clip: float,
) -> torch.Tensor:
    """Predictor-Corrector sampler (Euler-Maruyama + Langevin)."""
    model.eval()

    batch_size = int(shape[0])
    x = sde.prior_sampling(shape, device=device)
    conditioning_inputs = conditioning_inputs.to(device)

    sampling_N = int(max(2, sampling_N))
    eps = float(eps)
    if not (0.0 < eps < float(sde.T)):
        raise ValueError(f"sgm_sampling_eps must satisfy 0 < eps < T. Got eps={eps} T={sde.T}")

    t_seq = torch.linspace(sde.T, eps, sampling_N, device=device)
    dt = float(t_seq[1] - t_seq[0])  # negative

    # Optional early correction at t=T1 (Pedrotti et al.)
    if int(init_corrector_steps) > 0:
        t0 = torch.full((batch_size,), float(t_seq[0].item()), device=device)
        t0_net = t0 if t_embed_scale is None else (t0 * float(t_embed_scale))
        noise_norm0 = math.sqrt(x[0].numel())
        for _ in range(int(init_corrector_steps)):
            model_in = torch.cat([conditioning_inputs, x], dim=1)
            score0 = model(model_in, t0_net)
            score0 = _clip_score(score0, max_norm=float(score_clip))
            grad_norm0 = torch.norm(score0.reshape(batch_size, -1), dim=1).mean()
            step_size0 = 2.0 * (float(snr) * float(noise_norm0) / (float(grad_norm0) + 1e-12)) ** 2
            step_size0 = float(min(step_size0, 0.01))
            x = x + step_size0 * score0 + math.sqrt(2.0 * step_size0) * torch.randn_like(x)

    last_idx = int(t_seq.numel()) - 1
    for idx, t in enumerate(t_seq):
        t_b = torch.full((batch_size,), float(t.item()), device=device)
        t_net = t_b if t_embed_scale is None else (t_b * float(t_embed_scale))

        # Corrector (Langevin)
        for _ in range(int(n_steps_each)):
            model_in = torch.cat([conditioning_inputs, x], dim=1)
            score = model(model_in, t_net)
            score = _clip_score(score, max_norm=float(score_clip))

            noise = torch.randn_like(x)
            grad_norm = torch.norm(score.reshape(batch_size, -1), dim=1).mean()
            noise_norm = math.sqrt(x[0].numel())
            step_size = 2.0 * (float(snr) * float(noise_norm) / (float(grad_norm) + 1e-12)) ** 2
            step_size = float(min(step_size, 0.01))

            x = x + step_size * score + math.sqrt(2.0 * step_size) * noise

        # Predictor (Euler-Maruyama reverse)
        model_in = torch.cat([conditioning_inputs, x], dim=1)
        score = model(model_in, t_net)
        score = _clip_score(score, max_norm=float(score_clip))
        drift, diffusion = sde.sde(x, t_b)
        rev_drift = drift - (diffusion**2) * score

        noise = torch.randn_like(x)
        x_mean = x + rev_drift * dt
        x = x_mean + diffusion * math.sqrt(-dt) * noise

        if noise_removal:
            if idx == last_idx:
                x = x_mean

    return x


@torch.no_grad()
def pc_sampler_ve(
    *,
    model: nn.Module,
    sigma: float,
    conditioning_inputs: torch.Tensor,
    shape: tuple[int, int, int],
    device: torch.device,
    sampling_N: int,
    eps: float,
    snr: float,
    n_steps_each: int,
    init_corrector_steps: int,
    score_clip: float,
    t_embed_scale: float | None,
) -> torch.Tensor:
    """PC sampler (VE-SDE tutorial): Langevin corrector + Euler–Maruyama predictor."""
    model.eval()
    batch_size = int(shape[0])
    seq_length = int(shape[2])

    conditioning_inputs = conditioning_inputs.to(device)

    sampling_N = int(max(2, sampling_N))
    eps = float(eps)
    if not (0.0 < eps < 1.0):
        raise ValueError(f"sgm_sampling_eps must satisfy 0 < eps < 1. Got eps={eps}")

    t_init = torch.ones(batch_size, device=device)
    init_std = ve_marginal_prob_std(t_init, sigma=float(sigma)).view(batch_size, 1, 1)
    x = torch.randn((batch_size, 1, seq_length), device=device) * init_std

    time_steps = torch.linspace(1.0, eps, sampling_N, device=device)
    step_size = float(time_steps[0] - time_steps[1])  # positive

    noise_norm = math.sqrt(x[0].numel())  # matches tutorial

    # Optional early correction at t=T1 (Pedrotti et al.)
    if int(init_corrector_steps) > 0:
        t0 = torch.ones(batch_size, device=device) * float(time_steps[0].item())
        for _ in range(int(init_corrector_steps)):
            model_in = torch.cat([conditioning_inputs, x], dim=1)
            t0_net = t0 if t_embed_scale is None else (t0 * float(t_embed_scale))
            grad0 = model(model_in, t0_net)
            grad0 = _clip_score(grad0, max_norm=float(score_clip))
            grad_norm0 = torch.norm(grad0.reshape(batch_size, -1), dim=-1).mean()
            langevin_step_size0 = 2.0 * (float(snr) * float(noise_norm) / (float(grad_norm0) + 1e-12)) ** 2
            x = x + langevin_step_size0 * grad0 + math.sqrt(2.0 * langevin_step_size0) * torch.randn_like(x)

    for t in time_steps:
        t_b = torch.full((batch_size,), float(t.item()), device=device)
        t_net = t_b if t_embed_scale is None else (t_b * float(t_embed_scale))

        # Corrector: Langevin MCMC
        for _ in range(int(n_steps_each)):
            model_in = torch.cat([conditioning_inputs, x], dim=1)
            grad = model(model_in, t_net)
            grad = _clip_score(grad, max_norm=float(score_clip))
            grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=-1).mean()
            langevin_step_size = 2.0 * (float(snr) * float(noise_norm) / (float(grad_norm) + 1e-12)) ** 2
            x = x + langevin_step_size * grad + math.sqrt(2.0 * langevin_step_size) * torch.randn_like(x)

        # Predictor: Euler–Maruyama
        g = ve_diffusion_coeff(t_b, sigma=float(sigma)).view(batch_size, 1, 1)
        model_in = torch.cat([conditioning_inputs, x], dim=1)
        score = model(model_in, t_net)
        score = _clip_score(score, max_norm=float(score_clip))
        x_mean = x + (g**2) * score * step_size
        x = x_mean + math.sqrt(step_size) * g * torch.randn_like(x)

    # last step without noise
    return x_mean


@torch.no_grad()
def ode_sampler_vp(
    *,
    model: nn.Module,
    sde: VPSDE,
    conditioning_inputs: torch.Tensor,
    shape: tuple[int, int, int],
    device: torch.device,
    sampling_N: int,
    eps: float,
    t_embed_scale: float | None,
    method: str,
    init_corrector_steps: int,
    snr: float,
    score_clip: float,
) -> torch.Tensor:
    """Probability flow ODE sampler for VP-SDE (Song et al.)."""
    model.eval()
    batch_size = int(shape[0])
    x = sde.prior_sampling(shape, device=device)
    conditioning_inputs = conditioning_inputs.to(device)

    t_seq = torch.linspace(sde.T, float(eps), int(max(2, sampling_N)), device=device)

    # Optional early correction at t=T1
    if int(init_corrector_steps) > 0:
        t0 = torch.full((batch_size,), float(t_seq[0].item()), device=device)
        t0_net = t0 if t_embed_scale is None else (t0 * float(t_embed_scale))
        noise_norm0 = math.sqrt(x[0].numel())
        for _ in range(int(init_corrector_steps)):
            model_in = torch.cat([conditioning_inputs, x], dim=1)
            score0 = _clip_score(model(model_in, t0_net), max_norm=float(score_clip))
            grad_norm0 = torch.norm(score0.reshape(batch_size, -1), dim=1).mean()
            step_size0 = 2.0 * (float(snr) * float(noise_norm0) / (float(grad_norm0) + 1e-12)) ** 2
            step_size0 = float(min(step_size0, 0.01))
            x = x + step_size0 * score0 + math.sqrt(2.0 * step_size0) * torch.randn_like(x)

    method = str(method).lower().strip()
    if method not in {"euler", "rk4"}:
        raise ValueError(f"ode method must be euler|rk4. Got {method}")

    def f(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        t_net = t_in if t_embed_scale is None else (t_in * float(t_embed_scale))
        model_in = torch.cat([conditioning_inputs, x_in], dim=1)
        score = _clip_score(model(model_in, t_net), max_norm=float(score_clip))
        drift, diffusion = sde.sde(x_in, t_in)
        return drift - 0.5 * (diffusion**2) * score

    for i in range(int(t_seq.numel()) - 1):
        t0 = torch.full((batch_size,), float(t_seq[i].item()), device=device)
        t1 = torch.full((batch_size,), float(t_seq[i + 1].item()), device=device)
        dt = float(t1[0].item() - t0[0].item())  # negative
        if method == "euler":
            x = x + f(x, t0) * dt
        else:
            k1 = f(x, t0)
            k2 = f(x + 0.5 * dt * k1, (t0 + t1) * 0.5)
            k3 = f(x + 0.5 * dt * k2, (t0 + t1) * 0.5)
            k4 = f(x + dt * k3, t1)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


@torch.no_grad()
def ode_sampler_ve(
    *,
    model: nn.Module,
    sigma: float,
    conditioning_inputs: torch.Tensor,
    shape: tuple[int, int, int],
    device: torch.device,
    sampling_N: int,
    eps: float,
    method: str,
    init_corrector_steps: int,
    snr: float,
    score_clip: float,
    t_embed_scale: float | None,
) -> torch.Tensor:
    """Probability flow ODE sampler for VE-SDE (Song et al. tutorial)."""
    model.eval()
    batch_size = int(shape[0])
    seq_length = int(shape[2])
    conditioning_inputs = conditioning_inputs.to(device)

    t_init = torch.ones(batch_size, device=device)
    init_std = ve_marginal_prob_std(t_init, sigma=float(sigma)).view(batch_size, 1, 1)
    x = torch.randn((batch_size, 1, seq_length), device=device) * init_std

    t_seq = torch.linspace(1.0, float(eps), int(max(2, sampling_N)), device=device)

    noise_norm0 = math.sqrt(x[0].numel())
    if int(init_corrector_steps) > 0:
        t0 = torch.ones(batch_size, device=device) * float(t_seq[0].item())
        for _ in range(int(init_corrector_steps)):
            model_in = torch.cat([conditioning_inputs, x], dim=1)
            t0_net = t0 if t_embed_scale is None else (t0 * float(t_embed_scale))
            grad0 = _clip_score(model(model_in, t0_net), max_norm=float(score_clip))
            grad_norm0 = torch.norm(grad0.reshape(batch_size, -1), dim=-1).mean()
            langevin_step_size0 = 2.0 * (float(snr) * float(noise_norm0) / (float(grad_norm0) + 1e-12)) ** 2
            x = x + langevin_step_size0 * grad0 + math.sqrt(2.0 * langevin_step_size0) * torch.randn_like(x)

    method = str(method).lower().strip()
    if method not in {"euler", "rk4"}:
        raise ValueError(f"ode method must be euler|rk4. Got {method}")

    def f(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        t_net = t_in if t_embed_scale is None else (t_in * float(t_embed_scale))
        model_in = torch.cat([conditioning_inputs, x_in], dim=1)
        score = _clip_score(model(model_in, t_net), max_norm=float(score_clip))
        g = ve_diffusion_coeff(t_in, sigma=float(sigma)).view(batch_size, 1, 1)
        return -0.5 * (g**2) * score

    for i in range(int(t_seq.numel()) - 1):
        t0 = torch.full((batch_size,), float(t_seq[i].item()), device=device)
        t1 = torch.full((batch_size,), float(t_seq[i + 1].item()), device=device)
        dt = float(t1[0].item() - t0[0].item())  # negative
        if method == "euler":
            x = x + f(x, t0) * dt
        else:
            tm = (t0 + t1) * 0.5
            k1 = f(x, t0)
            k2 = f(x + 0.5 * dt * k1, tm)
            k3 = f(x + 0.5 * dt * k2, tm)
            k4 = f(x + dt * k3, t1)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


# --------------------------------------------------------------------------------------
# Additional metrics (MICCAI-style reporting helpers)
# --------------------------------------------------------------------------------------

def bandpower_similarity(real_signal: np.ndarray, synth_signal: np.ndarray, sfreq: float) -> dict[str, float]:
    """
    Compute normalized bandpower (8–30 Hz) similarity using Welch PSD.

    Steps:
    - Welch PSD for real and synthetic 1D signals
    - Bandpower in 8–30 Hz
    - Normalize by total power (0..Nyquist)

    Returns a dict with:
    - real_rel_bp: relative bandpower of real (8–30 / total)
    - synth_rel_bp: relative bandpower of synth (8–30 / total)
    - abs_diff: absolute difference between relative bandpowers
    - similarity: 1 - abs_diff / max(real_rel_bp, synth_rel_bp, eps)
    """
    from scipy.signal import welch

    x_r = np.asarray(real_signal, dtype=np.float64).reshape(-1)
    x_s = np.asarray(synth_signal, dtype=np.float64).reshape(-1)
    sfreq = float(sfreq)
    if sfreq <= 0:
        raise ValueError("sfreq must be > 0")
    if x_r.size < 8 or x_s.size < 8:
        raise ValueError("Signals too short for Welch PSD.")

    # Replace non-finite values defensively
    if not np.isfinite(x_r).all():
        x_r = np.nan_to_num(x_r, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(x_s).all():
        x_s = np.nan_to_num(x_s, nan=0.0, posinf=0.0, neginf=0.0)

    nperseg = int(min(256, x_r.size, x_s.size))
    noverlap = int(max(0, nperseg // 2))

    f_r, pxx_r = welch(x_r, fs=sfreq, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    f_s, pxx_s = welch(x_s, fs=sfreq, nperseg=nperseg, noverlap=noverlap, detrend="constant")

    # Ensure identical frequency grids (Welch should match with same params)
    if f_r.shape != f_s.shape or np.max(np.abs(f_r - f_s)) > 1e-9:
        # Fallback: interpolate synth to real freqs
        pxx_s = np.interp(f_r, f_s, pxx_s)
        f = f_r
    else:
        f = f_r

    df = float(f[1] - f[0]) if f.size > 1 else float(sfreq / 2.0)
    nyq = 0.5 * sfreq

    def _bandpower(pxx: np.ndarray, fgrid: np.ndarray, fmin: float, fmax: float) -> float:
        m = (fgrid >= float(fmin)) & (fgrid <= float(fmax))
        if not np.any(m):
            return 0.0
        return float(np.sum(np.asarray(pxx)[m]) * df)

    # Total power over [0, Nyquist]
    total_r = _bandpower(pxx_r, f, 0.0, nyq)
    total_s = _bandpower(pxx_s, f, 0.0, nyq)
    bp_r = _bandpower(pxx_r, f, 8.0, 30.0)
    bp_s = _bandpower(pxx_s, f, 8.0, 30.0)

    eps = 1e-12
    real_rel = float(bp_r / (total_r + eps))
    synth_rel = float(bp_s / (total_s + eps))
    abs_diff = float(abs(real_rel - synth_rel))
    denom = float(max(real_rel, synth_rel, eps))
    sim = float(max(0.0, 1.0 - abs_diff / denom))
    return {"real_rel_bp": real_rel, "synth_rel_bp": synth_rel, "abs_diff": abs_diff, "similarity": sim}


# --------------------------------------------------------------------------------------
# Ablation helpers (spatial/conditioning analyses)
# --------------------------------------------------------------------------------------

def psd_band_cosine_similarity(
    real_signal: np.ndarray,
    synth_signal: np.ndarray,
    sfreq: float,
    *,
    fmin: float = 8.0,
    fmax: float = 30.0,
) -> float:
    """
    Cosine similarity between Welch PSD *shapes* within a frequency band.

    This matches the requested "Bandpower similarity (PSD 8–30 Hz using cosine similarity)".
    """
    from scipy.signal import welch

    x_r = np.asarray(real_signal, dtype=np.float64).reshape(-1)
    x_s = np.asarray(synth_signal, dtype=np.float64).reshape(-1)
    sfreq = float(sfreq)
    if sfreq <= 0:
        raise ValueError("sfreq must be > 0")
    if x_r.size < 8 or x_s.size < 8:
        return float("nan")
    if not np.isfinite(x_r).all():
        x_r = np.nan_to_num(x_r, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(x_s).all():
        x_s = np.nan_to_num(x_s, nan=0.0, posinf=0.0, neginf=0.0)

    nperseg = int(min(256, x_r.size, x_s.size))
    noverlap = int(max(0, nperseg // 2))
    f_r, pxx_r = welch(x_r, fs=sfreq, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    f_s, pxx_s = welch(x_s, fs=sfreq, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    if f_r.shape != f_s.shape or np.max(np.abs(f_r - f_s)) > 1e-9:
        pxx_s = np.interp(f_r, f_s, pxx_s)
        f = f_r
    else:
        f = f_r

    m = (f >= float(fmin)) & (f <= float(fmax))
    if not np.any(m):
        return float("nan")
    v_r = np.asarray(pxx_r, dtype=np.float64)[m]
    v_s = np.asarray(pxx_s, dtype=np.float64)[m]
    nr = float(np.linalg.norm(v_r))
    ns = float(np.linalg.norm(v_s))
    denom = nr * ns
    if denom <= 0:
        return 0.0
    return float(np.dot(v_r, v_s) / denom)


def _standard_1020_positions(ch_names: list[str]) -> dict[str, np.ndarray]:
    """3D positions from MNE standard_1020 montage (subset for provided names)."""
    import mne

    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        pos = montage.get_positions().get("ch_pos", {})
    except Exception:
        pos = {}
    out: dict[str, np.ndarray] = {}
    for name in ch_names:
        if name in pos:
            out[name] = np.asarray(pos[name], dtype=np.float64)
    return out


def _pick_far_channels(
    target: str,
    *,
    k: int,
    candidates: list[str],
    positions: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    k = int(k)
    if k <= 0:
        return tuple()
    pool = [c for c in candidates if c != target]
    if target not in positions:
        rng.shuffle(pool)
        return tuple(pool[:k])
    tpos = positions[target]
    scored: list[tuple[float, str]] = []
    for c in pool:
        if c not in positions:
            continue
        d = float(np.linalg.norm(positions[c] - tpos))
        scored.append((d, c))
    if len(scored) < k:
        rng.shuffle(pool)
        return tuple(pool[:k])
    scored.sort(key=lambda x: x[0], reverse=True)
    return tuple([c for _, c in scored[:k]])


def _pick_random_channels(
    target: str,
    *,
    k: int,
    candidates: list[str],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    k = int(k)
    if k <= 0:
        return tuple()
    pool = [c for c in candidates if c != target]
    rng.shuffle(pool)
    return tuple(pool[:k])


def _pick_nearest_ring_channels(
    target: str,
    *,
    base: tuple[str, ...],
    k_total: int,
    candidates: list[str],
    positions: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """
    Conditioning set of size k_total:
    - start from `base` (Table 1 neighbors)
    - add nearest channels (spatial ring) until k_total
    """
    k_total = int(k_total)
    if k_total <= 0:
        return tuple()
    if k_total <= len(base):
        return tuple(base[:k_total])

    chosen = [c for c in base if c != target]
    chosen_set = set(chosen)
    pool = [c for c in candidates if c != target and c not in chosen_set]
    if target not in positions:
        rng.shuffle(pool)
        chosen.extend(pool[: max(0, k_total - len(chosen))])
        return tuple(chosen[:k_total])

    tpos = positions[target]
    scored: list[tuple[float, str]] = []
    for c in pool:
        if c not in positions:
            continue
        d = float(np.linalg.norm(positions[c] - tpos))
        scored.append((d, c))
    scored.sort(key=lambda x: x[0])  # nearest first
    for _, c in scored:
        if len(chosen) >= k_total:
            break
        chosen.append(c)
    if len(chosen) < k_total:
        rng.shuffle(pool)
        chosen.extend(pool[: max(0, k_total - len(chosen))])
    return tuple(chosen[:k_total])


def _reconstruction_metrics(
    X_real: np.ndarray,
    X_hybrid: np.ndarray,
    *,
    targets: list[str],
    sfreq: float,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Per-target reconstruction metrics + global mean (MSE, Pearson, PSD cosine sim 8–30)."""
    rows: list[dict[str, object]] = []
    mse_all: list[float] = []
    pear_all: list[float] = []
    psdcos_all: list[float] = []

    for tname in targets:
        tidx = CHANNEL_TO_IDX[tname]
        xr = np.asarray(X_real[:, tidx, :], dtype=np.float64)
        xs = np.asarray(X_hybrid[:, tidx, :], dtype=np.float64)
        if xr.shape != xs.shape:
            raise ValueError(f"Shape mismatch for target {tname}: {xr.shape} vs {xs.shape}")

        mse_list: list[float] = []
        pear_list: list[float] = []
        psdcos_list: list[float] = []
        for i in range(int(xr.shape[0])):
            r = xr[i]
            s = xs[i]
            mse_list.append(float(np.mean((s - r) ** 2)))
            pear_list.append(float(pearson_corr_1d(r.astype(np.float32, copy=False), s.astype(np.float32, copy=False))))
            psdcos_list.append(float(psd_band_cosine_similarity(r, s, float(sfreq), fmin=8.0, fmax=30.0)))

        mse_m = float(np.mean(mse_list)) if mse_list else float("nan")
        pear_m = float(np.mean(pear_list)) if pear_list else float("nan")
        psdcos_m = float(np.mean(psdcos_list)) if psdcos_list else float("nan")
        rows.append(
            {
                "Target Channel": tname,
                "MSE": mse_m,
                "Pearson": pear_m,
                "BandpowerSim (PSD cos 8-30)": psdcos_m,
                "n_eval": int(xr.shape[0]),
            }
        )
        mse_all.append(mse_m)
        pear_all.append(pear_m)
        psdcos_all.append(psdcos_m)

    def _mean_finite(x: list[float]) -> float:
        a = np.asarray(x, dtype=np.float64)
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if a.size else float("nan")

    summary = {
        "MSE_mean": _mean_finite(mse_all),
        "Pearson_mean": _mean_finite(pear_all),
        "BandpowerSim_mean": _mean_finite(psdcos_all),
    }
    return rows, summary


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="BCI Competition III - Dataset V held-out subject ID (1,2,3).",
    )
    parser.add_argument(
        "--loso",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable LOSO protocol: train on all listed subjects except --subject, test on held-out --subject raw04.",
    )
    parser.add_argument(
        "--loso-subjects",
        type=str,
        default="1,2,3",
        help="Comma-separated subject list for LOSO pool (e.g., '1,2,3').",
    )
    parser.add_argument(
        "--use-official-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use official test session raw04 + labels8 .asc (instead of internal 70/30 split).",
    )
    parser.add_argument(
        "--lr-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Secondary option: use ONLY left/right classes (exclude verbal/word class). Default=False keeps 3 classes.",
    )
    parser.add_argument(
        "--balance-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Undersample TRAIN to balance class counts before CV/final training.",
    )
    parser.add_argument(
        "--balance-ratio",
        type=float,
        default=1.0,
        help="Balance ratio in (0,1]. 1.0 means strict class balance to min-class count.",
    )
    parser.add_argument("--test-mat", type=str, default="", help="Override test .mat path (default uses dataset/BCI_3/test_subjectN_raw04.mat)")
    parser.add_argument("--test-asc", type=str, default="", help="Override test labels .asc path (default uses dataset/BCI_3/labels8_subjectN_raw.asc)")

    parser.add_argument("--ddpm-epochs", type=int, default=250)
    parser.add_argument(
        "--ddpm-epochs-cv",
        type=int,
        default=250,
        help="DDPM epochs used inside CV folds (0 => use --ddpm-epochs). Increase to reduce CV-vs-TEST hybrid gap.",
    )
    parser.add_argument(
        "--final-train-fraction",
        type=float,
        default=1.0,
        help="Fraction of TRAIN used in full-train held-out stage (0,1]. Default=1.0 uses all TRAIN.",
    )
    # CSP/FBCSP tuning flags
    parser.add_argument("--csp-n-components", type=int, choices=[2, 4, 6], default=4)
    parser.add_argument("--csp-primary-reg", type=str, choices=["none", "ledoit_wolf", "oas"], default="ledoit_wolf")
    parser.add_argument("--csp-fallback-reg", type=str, choices=["none", "ledoit_wolf", "oas"], default="oas")
    parser.add_argument("--csp-primary-cov-est", type=str, choices=["epoch", "concat"], default="concat")
    parser.add_argument("--csp-fallback-cov-est", type=str, choices=["epoch", "concat"], default="concat")
    parser.add_argument("--csp-primary-norm-trace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--csp-fallback-norm-trace", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--fbcsp-n-components", type=int, choices=[2, 4, 6], default=4)
    parser.add_argument("--fbcsp-primary-reg", type=str, choices=["none", "ledoit_wolf", "oas"], default="ledoit_wolf")
    parser.add_argument("--fbcsp-fallback-reg", type=str, choices=["none", "ledoit_wolf", "oas"], default="oas")
    parser.add_argument("--fbcsp-primary-cov-est", type=str, choices=["epoch", "concat"], default="concat")
    parser.add_argument("--fbcsp-fallback-cov-est", type=str, choices=["epoch", "concat"], default="concat")
    parser.add_argument("--fbcsp-primary-norm-trace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fbcsp-fallback-norm-trace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fbcsp-k-best", type=int, choices=[4, 8, 12], default=4)
    # Score-based (VP-SDE) training (paper-style steps)
    parser.add_argument("--sgm-train-steps", type=int, default=30_000)
    parser.add_argument("--sgm-eval-every", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-eval-epochs", type=int, default=0, help="0 => eval todos; si >0 limita n")
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=5, help="StratifiedKFold folds on TRAIN only.")
    parser.add_argument("--cv-seed", type=int, default=42, help="Random seed for StratifiedKFold shuffle.")
    parser.add_argument(
        "--partial-hybrid-ratio",
        type=float,
        default=0.5,
        help="Ablation: fraction of TEST epochs to replace per target channel (0..1).",
    )
    # SGM sampling (PC sampler)
    parser.add_argument("--sgm-sampling-n", type=int, default=2000)
    parser.add_argument("--sgm-sampling-eps", type=float, default=1e-5)
    parser.add_argument("--sgm-snr", type=float, default=0.10)
    parser.add_argument("--sgm-n-steps-each", type=int, default=2)
    parser.add_argument(
        "--sgm-sampler",
        type=str,
        default="pc",
        choices=["pc", "ode"],
        help="Sampler para SGM: pc (predictor-corrector) u ode (probability flow ODE).",
    )
    parser.add_argument(
        "--sgm-ode-method",
        type=str,
        default="rk4",
        choices=["euler", "rk4"],
        help="Método ODE si --sgm-sampler=ode.",
    )
    parser.add_argument("--sgm-init-corrector-steps", type=int, default=0, help="Langevin steps extra en t=T1 (Pedrotti).")
    parser.add_argument("--sgm-score-clip", type=float, default=0.0, help="Clip L2 del score (0 desactiva).")
    parser.add_argument("--sgm-ve-sigma", type=float, default=25.0, help="Sigma_max para VE-SDE (tutorial Song).")
    parser.add_argument("--sgm-vp-beta-max", type=float, default=20.0, help="beta_max para VP-SDE.")
    parser.add_argument("--sgm-ckpt-dir", type=str, default="checkpoints/sgm", help="Directorio para guardar/cargar checkpoints SGM.")
    parser.add_argument("--sgm-cache", action=argparse.BooleanOptionalAction, default=True, help="Auto-cargar checkpoints si existen.")
    parser.add_argument("--sgm-force-train", action="store_true", help="Ignora checkpoints y re-entrena.")
    parser.add_argument("--sgm-likelihood-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sgm-noise-removal", action="store_true", help="Use x_mean (noiseless) each step")
    parser.add_argument(
        "--sgm-backbone",
        type=str,
        default="scorenet-unet",
        choices=["scorenet-unet", "ddpm-unet"],
        help="Backbone para score model. scorenet-unet requiere `SGDM.py` en el mismo directorio.",
    )
    # Requested defaults for EEGNet
    parser.add_argument("--cls-epochs", type=int, default=200, help="Epochs para EEGNet")
    parser.add_argument("--cls-batch-size", type=int, default=128, help="Batch size para EEGNet")
    parser.add_argument("--eegnet-lr", type=float, default=1e-3, help="Learning rate for EEGNet (Adam).")
    parser.add_argument("--eegnet-dropout", type=float, default=0.5, help="Dropout rate inside EEGNet.")
    parser.add_argument("--eegnet-patience", type=int, default=80, help="Early-stopping patience (val loss) for EEGNet.")

    parser.add_argument(
        "--only-eegnet",
        action="store_true",
        help="Run ONLY EEGNet (segment-aware CV + held-out TEST). Skips DDPM/CSP/FBCSP/SGM for debugging.",
    )
    parser.add_argument(
        "--only-eegnet-no-cv",
        action="store_true",
        help="With --only-eegnet: skip CV and only run full-TRAIN -> held-out TEST evaluation.",
    )
    parser.add_argument(
        "--only-eegnet-cv-epochs",
        type=int,
        default=100,
        help="With --only-eegnet: epochs per CV fold (default 100 for speed).",
    )
    parser.add_argument(
        "--only-csp-fbcsp",
        action="store_true",
        help="Run ONLY CSP+LDA and FBCSP+LDA (segment-aware CV + held-out TEST). Skips DDPM/EEGNet/SGM.",
    )
    parser.add_argument(
        "--only-csp-fbcsp-no-cv",
        action="store_true",
        help="With --only-csp-fbcsp: skip CV and only run full-TRAIN -> held-out TEST evaluation.",
    )

    # NOTE: per-epoch z-score post-processing for synthetic channels was removed because it can
    # break covariance structure and collapse CSP/FBCSP on hybrid evaluation.
    parser.add_argument(
        "--hybrid-clip",
        type=float,
        default=0.0,
        help="(Deprecated) Hybrid clipping is disabled and ignored.",
    )

    parser.add_argument("--skip-ddpm", action="store_true")
    parser.add_argument("--skip-sgm", action="store_true")
    args = parser.parse_args()

    _set_seeds(args.seed)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _parse_reg(v: str) -> Optional[str]:
        vv = str(v).strip().lower()
        return None if vv == "none" else vv

    def _reseed_for_eegnet(offset: int) -> None:
        # Seed behavior kept as at script start (no per-stage reseeding).
        _ = int(offset)

    csp_cfg_cli = {
        "csp_n_components": int(args.csp_n_components),
        "csp_primary_reg": _parse_reg(args.csp_primary_reg),
        "csp_fallback_reg": _parse_reg(args.csp_fallback_reg),
        "csp_primary_cov_est": str(args.csp_primary_cov_est),
        "csp_fallback_cov_est": str(args.csp_fallback_cov_est),
        "csp_primary_norm_trace": bool(args.csp_primary_norm_trace),
        "csp_fallback_norm_trace": bool(args.csp_fallback_norm_trace),
    }
    fbcsp_cfg_cli = {
        "fbcsp_n_components": int(args.fbcsp_n_components),
        "fbcsp_primary_reg": _parse_reg(args.fbcsp_primary_reg),
        "fbcsp_fallback_reg": _parse_reg(args.fbcsp_fallback_reg),
        "fbcsp_primary_cov_est": str(args.fbcsp_primary_cov_est),
        "fbcsp_fallback_cov_est": str(args.fbcsp_fallback_cov_est),
        "fbcsp_primary_norm_trace": bool(args.fbcsp_primary_norm_trace),
        "fbcsp_fallback_norm_trace": bool(args.fbcsp_fallback_norm_trace),
        "fbcsp_k_best": int(args.fbcsp_k_best),
    }

    subject_id = int(args.subject)
    loso_subjects = [int(s.strip()) for s in str(args.loso_subjects).split(",") if str(s).strip()]
    if len(loso_subjects) == 0:
        raise ValueError("--loso-subjects is empty.")
    if any(s not in (1, 2, 3) for s in loso_subjects):
        raise ValueError(f"--loso-subjects must contain IDs in [1,2,3], got: {loso_subjects}")
    loso_subjects = sorted(set(loso_subjects))

    # ============================== LOSO subject loop starts ==============================
    # In LOSO mode, training data is built by iterating all subjects in --loso-subjects
    # except the held-out subject (--subject). The rest of the pipeline remains unchanged.
    # ======================================================================================
    if bool(args.loso):
        if not bool(args.use_official_test):
            raise ValueError("LOSO requires --use-official-test (no random split).")
        if subject_id not in loso_subjects:
            raise ValueError(f"Held-out subject {subject_id} is not in --loso-subjects={loso_subjects}")
        train_subject_ids = [s for s in loso_subjects if s != subject_id]
        if len(train_subject_ids) == 0:
            raise ValueError("LOSO needs at least one training subject different from --subject.")
        _log_section(
            f"Load + preprocess + epoching (LOSO held-out S{subject_id}: "
            f"train subjects={train_subject_ids} [raw01-03], test=raw04 + labels8 .asc)"
        )
        TRAIN_MAT_PATHS: list[str] = []
        for sid in train_subject_ids:
            TRAIN_MAT_PATHS.extend(
                [
                    f"dataset/BCI_3/train_subject{sid}_raw01.mat",
                    f"dataset/BCI_3/train_subject{sid}_raw02.mat",
                    f"dataset/BCI_3/train_subject{sid}_raw03.mat",
                ]
            )
    else:
        if bool(args.use_official_test):
            _log_section(
                f"Load + preprocess (paper-like) + epoching "
                f"(Subject {subject_id}: raw01+raw02+raw03 train, raw04 test + labels8 .asc)"
            )
        else:
            _log_section(f"Load + preprocess (paper-like) + epoching (Subject {subject_id}: raw01+raw02+raw03, 70/30)")
        TRAIN_MAT_PATHS = [
            f"dataset/BCI_3/train_subject{subject_id}_raw01.mat",
            f"dataset/BCI_3/train_subject{subject_id}_raw02.mat",
            f"dataset/BCI_3/train_subject{subject_id}_raw03.mat",
        ]

    def _load_xy_fs(mat_path: str) -> tuple[np.ndarray, np.ndarray, float]:
        mat = scipy.io.loadmat(mat_path)
        x = mat["X"].astype(np.float64)  # (n_times, 32)
        y = mat.get("Y")
        if y is None:
            raise ValueError(f"{mat_path} does not contain Y labels.")
        y = y.reshape(-1).astype(np.int64)
        try:
            fs = float(np.array(mat["nfo"]["fs"][0, 0]).reshape(-1)[0])
        except Exception:
            fs = 512.0
        return x, y, fs

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    fs_parts: list[float] = []
    len_parts: list[int] = []

    for p in TRAIN_MAT_PATHS:
        x_i, y_i, fs_i = _load_xy_fs(p)
        x_parts.append(x_i)
        y_parts.append(y_i)
        fs_parts.append(fs_i)
        len_parts.append(int(x_i.shape[0]))

    sfreq = float(fs_parts[0])
    if any(abs(fs - sfreq) > 1e-6 for fs in fs_parts[1:]):
        raise ValueError(f"Inconsistent sampling rates across sessions: {fs_parts}")

    x_raw = np.concatenate(x_parts, axis=0)
    y_raw = np.concatenate(y_parts, axis=0)

    print("device:", device)
    print("raw X:", x_raw.shape, "raw Y:", y_raw.shape, "sfreq:", sfreq)

    # ---- preprocessing (paper-like): annotate_muscle + fold-safe ICA(Fp1/Fp2) + 8-30Hz + z-score
    import mne
    from mne.preprocessing import ICA

    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(x_raw.T, info, verbose=False)

    try:
        annotations, _ = mne.preprocessing.annotate_muscle_zscore(
            raw,
            ch_type="eeg",
            threshold=4.0,
            min_length_good=0.1,
            filter_freq=(20, 140),
            verbose=False,
        )
        raw.set_annotations(annotations)
    except Exception as e:
        print("annotate_muscle_zscore skipped:", e)

    # IMPORTANT (rigor / CV hygiene):
    # Do NOT fit ICA here on the full concatenated TRAIN, because later we run CV folds *within* TRAIN.
    # Fitting ICA globally would leak validation-fold statistics into the preprocessing.
    #
    # Instead:
    # - build epochs from the raw signal first (no ICA yet)
    # - during CV: fit ICA on fold-train epochs, apply to fold-train + fold-val
    # - for held-out TEST: fit ICA on full TRAIN epochs, apply to TRAIN + TEST
    x_unfilt = raw.get_data().T.astype(np.float32)  # (n_times, 32)
    # Importante (anti-leakage): NO z-score aquí con todo el dataset.
    # Hacemos el z-score usando SOLO stats del TRAIN (por fold en CV, y full-TRAIN para held-out TEST).
    x_for_epoching = x_unfilt

    # ---- epoching
    # Dataset V requirement (BCI Comp III): output every 0.5s using the last 1s of data.
    EPOCH_SAMPLES = 512  # 1.0s at 512 Hz
    STRIDE_SAMPLES = EPOCH_SAMPLES // 2  # 0.5s step (overlap) at 512 Hz

    # Dataset V labels: 2=left, 3=right, 7=word (verbal).
    # Default: use all classes present. Optional: --lr-only keeps only left/right (2,3).
    if bool(args.lr_only):
        allowed_labels = {2, 3}
    else:
        allowed_labels = set(int(v) for v in np.unique(y_raw).tolist())
    max_epochs_per_class = None

    epochs_parts: list[np.ndarray] = []
    labels_parts_out: list[np.ndarray] = []
    groups_parts_out: list[np.ndarray] = []
    group_cursor = 0
    offset = 0
    for n_sess in len_parts:
        x_sess = x_for_epoching[offset : offset + n_sess]
        y_sess = y_raw[offset : offset + n_sess]
        ep_sess, lab_sess, grp_sess, group_cursor = build_epochs_from_labels_with_groups(
            x_sess,
            y_sess,
            epoch_samples=EPOCH_SAMPLES,
            stride_samples=STRIDE_SAMPLES,
            allowed_labels=allowed_labels,
            max_epochs_per_class=max_epochs_per_class,
            group_offset=int(group_cursor),
        )
        epochs_parts.append(ep_sess)
        labels_parts_out.append(lab_sess)
        groups_parts_out.append(grp_sess)
        offset += n_sess

    epochs_train_raw = np.concatenate(epochs_parts, axis=0)
    labels_train = np.concatenate(labels_parts_out, axis=0)
    groups_train = np.concatenate(groups_parts_out, axis=0)

    # ============================== Continue original pipeline =============================
    # From here onward, the pipeline is unchanged. It consumes epochs_train_raw/labels_train
    # built above (single-subject or LOSO multi-subject TRAIN), and evaluates on held-out TEST.
    # ======================================================================================
    if bool(args.use_official_test):
        test_mat_path = str(args.test_mat).strip() or f"dataset/BCI_3/test_subject{subject_id}_raw04.mat"
        test_asc_path = str(args.test_asc).strip() or f"dataset/BCI_3/labels8_subject{subject_id}_raw.asc"
        mat_test = scipy.io.loadmat(test_mat_path)
        x_test_raw = mat_test["X"].astype(np.float64)  # (n_times, 32)

        # Keep TEST unfiltered here; fold-safe/full-train ICA+8-30 is applied later.
        raw_test = mne.io.RawArray(x_test_raw.T, info, verbose=False)
        x_test_unfilt = raw_test.get_data().T.astype(np.float32)

        y_test_windows = load_window_labels_from_asc(test_asc_path)
        epochs_test_raw, labels_test = build_epochs_from_window_labels(
            x_test_unfilt,
            y_test_windows,
            epoch_samples=EPOCH_SAMPLES,
            stride_samples=STRIDE_SAMPLES,
        )
        # If user requested binary left/right only, drop windows with other labels (e.g., verbal=7).
        if bool(args.lr_only):
            keep = np.isin(labels_test.astype(int), np.array([2, 3], dtype=int))
            epochs_test_raw = epochs_test_raw[keep]
            labels_test = labels_test[keep]
        print("allowed_labels (train):", sorted(int(v) for v in allowed_labels))
        print("official TEST paths:", test_mat_path, "|", test_asc_path)
    else:
        train_ratio = 0.7
        print("allowed_labels:", sorted(int(v) for v in allowed_labels))
        train_idx, test_idx = stratified_split(labels_train, train_ratio=train_ratio, seed=args.seed)
        epochs_test_raw = epochs_train_raw[test_idx]
        labels_test = labels_train[test_idx]
        epochs_train_raw = epochs_train_raw[train_idx]
        labels_train = labels_train[train_idx]
        groups_train = groups_train[train_idx]

    if bool(args.balance_train):
        idx_bal = balanced_train_indices(labels_train, seed=int(args.seed), target_ratio=float(args.balance_ratio))
        if int(idx_bal.size) > 0:
            epochs_train_raw = np.asarray(epochs_train_raw[idx_bal], dtype=np.float32)
            labels_train = np.asarray(labels_train[idx_bal], dtype=np.int64)
            groups_train = np.asarray(groups_train[idx_bal], dtype=np.int64)
            print(
                f"[balance] enabled: ratio={float(args.balance_ratio):.3f} | "
                f"kept {int(idx_bal.size)} train epochs | class balance train={_class_balance(labels_train)}"
            )

    # (IMPORTANT) We will z-score:
    # - inside each CV fold (to avoid leakage)
    # - once on full TRAIN for final held-out TEST evaluation
    print("epochs_train_raw:", epochs_train_raw.shape, "epochs_test_raw:", epochs_test_raw.shape)
    print("class balance train:", _class_balance(labels_train))
    print("class balance test :", _class_balance(labels_test))

    def _ica_fit_apply_epochs(Xtr_raw: np.ndarray, Xother_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit ICA on TRAIN epochs only; apply to TRAIN + other (val/test).
        ICA is estimated on a temporary 1 Hz high-pass copy, then applied to
        unfiltered epochs. After ICA, we keep the pipeline 8-30 Hz filtering.

        This keeps CV fold-safe (no access to validation samples during ICA fit).
        """
        Xtr_raw = np.asarray(Xtr_raw, dtype=np.float64)
        Xother_raw = np.asarray(Xother_raw, dtype=np.float64)

        info_ep = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=float(sfreq), ch_types="eeg")
        ep_tr = mne.EpochsArray(Xtr_raw, info_ep, verbose=False)
        ep_other = mne.EpochsArray(Xother_raw, info_ep, verbose=False)
        ep_tr_hp = ep_tr.copy().filter(l_freq=1.0, h_freq=None, method="iir", verbose=False)

        ica_f = ICA(n_components=20, random_state=97, max_iter=800)
        ica_f.fit(ep_tr_hp, verbose=False)
        try:
            eog_idx, _ = ica_f.find_bads_eog(ep_tr_hp, ch_name=["Fp1", "Fp2"], verbose=False)
            ica_f.exclude = eog_idx
        except Exception:
            pass

        ep_tr_c = ica_f.apply(ep_tr.copy(), verbose=False)
        ep_other_c = ica_f.apply(ep_other.copy(), verbose=False)
        ep_tr_c = ep_tr_c.filter(l_freq=8.0, h_freq=30.0, method="iir", verbose=False)
        ep_other_c = ep_other_c.filter(l_freq=8.0, h_freq=30.0, method="iir", verbose=False)
        return (
            ep_tr_c.get_data(copy=True).astype(np.float32, copy=False),
            ep_other_c.get_data(copy=True).astype(np.float32, copy=False),
        )

    epochs_train: Optional[np.ndarray] = None
    epochs_test: Optional[np.ndarray] = None

    def _prepare_full_train_test_preproc() -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare held-out TRAIN/TEST views once:
        - fit ICA on full TRAIN only, apply to TRAIN+TEST
        - fit z-score on TRAIN only, apply to TRAIN+TEST
        """
        nonlocal epochs_train, epochs_test
        if epochs_train is not None and epochs_test is not None:
            return epochs_train, epochs_test
        epochs_train_raw_ica, epochs_test_raw_ica = _ica_fit_apply_epochs(epochs_train_raw, epochs_test_raw)
        train_mu_full = epochs_train_raw_ica.mean(axis=(0, 2), keepdims=True)
        train_sd_full = epochs_train_raw_ica.std(axis=(0, 2), keepdims=True) + 1e-6
        epochs_train = ((epochs_train_raw_ica - train_mu_full) / train_sd_full).astype(np.float32, copy=False)
        epochs_test = ((epochs_test_raw_ica - train_mu_full) / train_sd_full).astype(np.float32, copy=False)
        print("[preproc] prepared (ICA fit on full TRAIN + z-score full-TRAIN stats) for held-out TEST; CV uses fold-specific fit")
        return epochs_train, epochs_test

    # ----------------------------------------------------------------------------------
    # EEGNet-only debug mode (no DDPM, no CSP/FBCSP, no SGM)
    # ----------------------------------------------------------------------------------
    if bool(args.only_eegnet):
        _log_section(f"EEGNet ONLY - SUBJECT {subject_id} (segment-aware CV + held-out TEST)")

        X_train_full_raw = epochs_train_raw
        y_train_full = labels_train
        y_test = labels_test

        splits = iter_stratified_group_kfold_indices(
            y_train_full, groups_train, n_splits=int(args.cv_folds), seed=int(args.cv_seed)
        )

        def _zscore_fit_apply(Xtr_raw: np.ndarray, Xother_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mu = Xtr_raw.mean(axis=(0, 2), keepdims=True)
            sd = Xtr_raw.std(axis=(0, 2), keepdims=True) + 1e-6
            return ((Xtr_raw - mu) / sd).astype(np.float32, copy=False), ((Xother_raw - mu) / sd).astype(np.float32, copy=False)

        cv_acc_real: list[float] = []
        cv_kappa_real: list[float] = []
        if not bool(args.only_eegnet_no_cv):
            epochs_cv = int(args.only_eegnet_cv_epochs)
            for fold_i, (tr_idx, va_idx) in enumerate(splits, start=1):
                tr_idx = np.asarray(tr_idx, dtype=int)
                va_idx = np.asarray(va_idx, dtype=int)
                X_tr_raw = np.asarray(X_train_full_raw[tr_idx], dtype=np.float32)
                y_tr = np.asarray(y_train_full[tr_idx], dtype=int)
                X_va_raw = np.asarray(X_train_full_raw[va_idx], dtype=np.float32)
                y_va = np.asarray(y_train_full[va_idx], dtype=int)

                _log_section(f"[EEGNet-only CV fold {fold_i}/{int(args.cv_folds)}] train->val (real only)")
                # Fold-safe preprocessing: ICA fit on fold-train, applied to train+val; then z-score fit on fold-train
                X_tr_ica, X_va_ica = _ica_fit_apply_epochs(X_tr_raw, X_va_raw)
                X_tr, X_va = _zscore_fit_apply(X_tr_ica, X_va_ica)
                _reseed_for_eegnet(10_000 + int(fold_i))

                out = run_eegnet(
                    X_train=X_tr,
                    y_train=y_tr,
                    X_test_real=X_va,
                    X_test_hybrid=X_va,  # placeholder (hybrid not evaluated in this mode)
                    y_test=y_va,
                    device=device,
                    epochs=int(epochs_cv),
                    batch_size=int(args.cls_batch_size),
                    eegnet_lr=float(args.eegnet_lr),
                    eegnet_dropout=float(args.eegnet_dropout),
                    eegnet_patience=int(args.eegnet_patience),
                )
                no_cv = out.get("no_cv", {})
                cv_acc_real.append(float(no_cv.get("acc_real", float("nan"))))
                cv_kappa_real.append(float(no_cv.get("kappa_real", float("nan"))))

        def _mean_std(vals: list[float]) -> tuple[float, float]:
            a = np.asarray(vals, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan"), float("nan")
            return float(np.mean(a)), float(np.std(a))

        cv_acc_m, cv_acc_s = _mean_std(cv_acc_real)

        _log_section("EEGNet-only: train on FULL TRAIN, evaluate held-out TEST (real only)")
        epochs_train, epochs_test = _prepare_full_train_test_preproc()
        X_test_real = epochs_test
        # Keep held-out EEGNet path identical to DDPM final summary for strict comparability.
        _reseed_for_eegnet(13_000)
        eegnet_bundle = _train_eegnet_once(
            X_train=epochs_train,
            y_train=labels_train,
            device=device,
            epochs=int(args.cls_epochs),
            batch_size=int(args.cls_batch_size),
            lr=float(args.eegnet_lr),
            dropout=float(args.eegnet_dropout),
            patience=int(args.eegnet_patience),
        )
        m_real = _eval_eegnet_fixed(
            model=eegnet_bundle["model"],
            classes=eegnet_bundle["classes"],
            mu_ch=eegnet_bundle["mu_ch"],
            sd_ch=eegnet_bundle["sd_ch"],
            X_test=X_test_real,
            y_test=y_test,
            device=device,
            batch_size=int(args.cls_batch_size),
        )
        acc_real = float(m_real.get("acc", float("nan")))
        kappa_real = float(m_real.get("kappa", float("nan")))

        print("\n" + "=" * 80)
        print(f"EEGNet ONLY - SUBJECT {subject_id} — Final summary (segment-CV + held-out TEST)")
        print("=" * 80)
        print(
            {
                "Model": "EEGNet",
                "Acc Original": acc_real,
                "Kappa Original": kappa_real,
                "CV Acc Original (mean/std)": f"{cv_acc_m:.3f} ({cv_acc_s:.3f})" if cv_acc_real else "skipped",
                "Note": "Hybrid/DDPM skipped in --only-eegnet mode",
            }
        )
        _log_section("Done")
        return

    # ----------------------------------------------------------------------------------
    # CSP/FBCSP-only debug mode (no DDPM, no EEGNet, no SGM)
    # ----------------------------------------------------------------------------------
    if bool(args.only_csp_fbcsp):
        _log_section(f"CSP+FBCSP ONLY - SUBJECT {subject_id} (segment-aware CV + held-out TEST)")

        X_train_full_raw = epochs_train_raw
        y_train_full = labels_train
        y_test = labels_test

        splits = iter_stratified_group_kfold_indices(
            y_train_full, groups_train, n_splits=int(args.cv_folds), seed=int(args.cv_seed)
        )

        def _zscore_fit_apply(Xtr_raw: np.ndarray, Xother_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mu = Xtr_raw.mean(axis=(0, 2), keepdims=True)
            sd = Xtr_raw.std(axis=(0, 2), keepdims=True) + 1e-6
            return ((Xtr_raw - mu) / sd).astype(np.float32, copy=False), ((Xother_raw - mu) / sd).astype(np.float32, copy=False)

        cv_acc: dict[str, list[float]] = {"CSP+LDA": [], "FBCSP+LDA": []}
        cv_kappa: dict[str, list[float]] = {"CSP+LDA": [], "FBCSP+LDA": []}
        if not bool(args.only_csp_fbcsp_no_cv):
            for fold_i, (tr_idx, va_idx) in enumerate(splits, start=1):
                tr_idx = np.asarray(tr_idx, dtype=int)
                va_idx = np.asarray(va_idx, dtype=int)
                X_tr_raw = np.asarray(X_train_full_raw[tr_idx], dtype=np.float32)
                y_tr = np.asarray(y_train_full[tr_idx], dtype=int)
                X_va_raw = np.asarray(X_train_full_raw[va_idx], dtype=np.float32)
                y_va = np.asarray(y_train_full[va_idx], dtype=int)

                _log_section(f"[CSP/FBCSP-only CV fold {fold_i}/{int(args.cv_folds)}] train->val (real only)")
                X_tr_ica, X_va_ica = _ica_fit_apply_epochs(X_tr_raw, X_va_raw)
                X_tr, X_va = _zscore_fit_apply(X_tr_ica, X_va_ica)

                out_csp = run_csp_lda(
                    X_train=X_tr,
                    y_train=y_tr,
                    X_test_real=X_va,
                    X_test_hybrid=X_va,  # placeholder (hybrid not evaluated in this mode)
                    y_test=y_va,
                    **csp_cfg_cli,
                )
                out_f = run_fbcsp_lda(
                    X_train=X_tr,
                    y_train=y_tr,
                    X_test_real=X_va,
                    X_test_hybrid=X_va,  # placeholder (hybrid not evaluated in this mode)
                    y_test=y_va,
                    sfreq=float(sfreq),
                    **fbcsp_cfg_cli,
                )
                for name, out in [("CSP+LDA", out_csp), ("FBCSP+LDA", out_f)]:
                    no_cv = out.get("no_cv", {})
                    cv_acc[name].append(float(no_cv.get("acc_real", float("nan"))))
                    cv_kappa[name].append(float(no_cv.get("kappa_real", float("nan"))))

        def _mean_std(vals: list[float]) -> tuple[float, float]:
            a = np.asarray(vals, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan"), float("nan")
            return float(np.mean(a)), float(np.std(a))

        _log_section("CSP/FBCSP-only: train on FULL TRAIN, evaluate held-out TEST (real only)")
        epochs_train, epochs_test = _prepare_full_train_test_preproc()
        X_test_real = epochs_test
        out_csp_test = run_csp_lda(
            X_train=epochs_train,
            y_train=labels_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_real,
            y_test=y_test,
            **csp_cfg_cli,
        )
        out_f_test = run_fbcsp_lda(
            X_train=epochs_train,
            y_train=labels_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_real,
            y_test=y_test,
            sfreq=float(sfreq),
            **fbcsp_cfg_cli,
        )

        cv_csp_m, cv_csp_s = _mean_std(cv_acc["CSP+LDA"])
        cv_fbcsp_m, cv_fbcsp_s = _mean_std(cv_acc["FBCSP+LDA"])
        csp_no_cv = out_csp_test.get("no_cv", {})
        fbcsp_no_cv = out_f_test.get("no_cv", {})

        print("\n" + "=" * 80)
        print(f"CSP/FBCSP ONLY - SUBJECT {subject_id} — Final summary (segment-CV + held-out TEST)")
        print("=" * 80)
        print(
            {
                "Model": "CSP+LDA",
                "Acc Original": float(csp_no_cv.get("acc_real", float("nan"))),
                "Kappa Original": float(csp_no_cv.get("kappa_real", float("nan"))),
                "CV Acc Original (mean/std)": f"{cv_csp_m:.3f} ({cv_csp_s:.3f})" if cv_acc["CSP+LDA"] else "skipped",
                "Note": "Hybrid/DDPM skipped in --only-csp-fbcsp mode",
            }
        )
        print(
            {
                "Model": "FBCSP+LDA",
                "Acc Original": float(fbcsp_no_cv.get("acc_real", float("nan"))),
                "Kappa Original": float(fbcsp_no_cv.get("kappa_real", float("nan"))),
                "CV Acc Original (mean/std)": f"{cv_fbcsp_m:.3f} ({cv_fbcsp_s:.3f})" if cv_acc["FBCSP+LDA"] else "skipped",
                "Note": "Hybrid/DDPM skipped in --only-csp-fbcsp mode",
            }
        )
        _log_section("Done")
        return

    # ---- diffusion constants (DDPM) for compatibility with pipeline
    TIMESTEPS = 1000
    betas = linear_beta_schedule(TIMESTEPS).to(device)
    alphas = (1.0 - betas).to(device)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    n_eval_epochs = None if int(args.n_eval_epochs) <= 0 else int(args.n_eval_epochs)

    def _accuracy_table_rows(name_to_metrics: dict[str, dict[str, dict[str, float]]]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for clf_name, m in name_to_metrics.items():
            no_cv = m.get("no_cv", {})
            cv = m.get("cv", {})
            rows.append(
                {
                    "Classifier": clf_name,
                    "Acc Original": float(no_cv.get("acc_real", float("nan"))),
                    "Acc Hybrid": float(no_cv.get("acc_hybrid", float("nan"))),
                    "Bal Acc Original": float(no_cv.get("bal_acc_real", float("nan"))),
                    "Bal Acc Hybrid": float(no_cv.get("bal_acc_hybrid", float("nan"))),
                    "CV Acc Original (mean)": float(cv.get("acc_real", float("nan"))),
                    "CV Acc Original (std)": float(cv.get("acc_real_std", float("nan"))),
                    "CV Acc Hybrid (mean)": float(cv.get("acc_hybrid", float("nan"))),
                    "CV Acc Hybrid (std)": float(cv.get("acc_hybrid_std", float("nan"))),
                }
            )
        return rows

    def _print_accuracy_table(title: str, name_to_metrics: dict[str, dict[str, dict[str, float]]]) -> None:
        _log_section(title)
        rows = _accuracy_table_rows(name_to_metrics)
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            print(df.to_string(index=False))
        except Exception:
            for r in rows:
                print(r)

    # ----------------------------------------------------------------------------------
    # DDPM pipeline
    # ----------------------------------------------------------------------------------
    trained_models: dict[str, nn.Module] = {}
    trained_input_idxs: dict[str, tuple[int, int]] = {}
    ddpm_rows: list[dict[str, object]] = []
    ddpm_acc_summary: dict[str, dict[str, dict[str, float]]] = {}
    sgm_acc_summary: dict[str, dict[str, dict[str, float]]] = {}

    if not args.skip_ddpm:
        X_train_full_raw = epochs_train_raw
        y_train_full = labels_train
        X_test_real_raw = epochs_test_raw
        y_test = labels_test

        # -------------------------------
        # 1) TRAIN-only CV (k=5) protocol
        # -------------------------------
        _log_section(
            f"DDPM - SUBJECT {subject_id} — TRAIN-only StratifiedGroupKFold (segment-aware) CV (k={int(args.cv_folds)})"
        )

        splits = iter_stratified_group_kfold_indices(
            y_train_full, groups_train, n_splits=int(args.cv_folds), seed=int(args.cv_seed)
        )

        cv_acc: dict[str, list[float]] = {k: [] for k in ["CSP+LDA", "FBCSP+LDA", "EEGNet"]}
        cv_acc_h: dict[str, list[float]] = {k: [] for k in ["CSP+LDA", "FBCSP+LDA", "EEGNet"]}
        cv_kappa: dict[str, list[float]] = {k: [] for k in ["CSP+LDA", "FBCSP+LDA", "EEGNet"]}
        cv_kappa_h: dict[str, list[float]] = {k: [] for k in ["CSP+LDA", "FBCSP+LDA", "EEGNet"]}

        def _zscore_fit_apply(Xtr_raw: np.ndarray, Xother_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mu = Xtr_raw.mean(axis=(0, 2), keepdims=True)
            sd = Xtr_raw.std(axis=(0, 2), keepdims=True) + 1e-6
            return ((Xtr_raw - mu) / sd).astype(np.float32, copy=False), ((Xother_raw - mu) / sd).astype(np.float32, copy=False)

        for fold_i, (tr_idx, va_idx) in enumerate(splits, start=1):
            tr_idx = np.asarray(tr_idx, dtype=int)
            va_idx = np.asarray(va_idx, dtype=int)
            X_tr_raw = np.asarray(X_train_full_raw[tr_idx], dtype=np.float32)
            y_tr = np.asarray(y_train_full[tr_idx], dtype=int)
            X_va_real_raw = np.asarray(X_train_full_raw[va_idx], dtype=np.float32)
            y_va = np.asarray(y_train_full[va_idx], dtype=int)

            _log_section(f"[CV fold {fold_i}/{int(args.cv_folds)}] Train DDPM on train_real, eval on val_real/val_hybrid")

            # Fold-safe preprocessing: ICA fit on fold-train, applied to train+val; then z-score fit on fold-train
            X_tr_ica, X_va_real_ica = _ica_fit_apply_epochs(X_tr_raw, X_va_real_raw)
            X_tr, X_va_real = _zscore_fit_apply(X_tr_ica, X_va_real_ica)

            # Train DDPM models per target on train_real only
            fold_models: dict[str, nn.Module] = {}
            fold_input_idxs: dict[str, tuple[int, int]] = {}
            ddpm_epochs_cv = int(args.ddpm_epochs_cv) if int(args.ddpm_epochs_cv) > 0 else int(args.ddpm_epochs)
            for spec in EXPERIMENTS:
                target_idx = CHANNEL_TO_IDX[spec.target]
                input_idxs = (CHANNEL_TO_IDX[spec.inputs[0]], CHANNEL_TO_IDX[spec.inputs[1]])
                train_ds = EEGEpochs(X_tr, input_idxs=input_idxs, target_idx=target_idx)
                train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)

                model = UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
                train_diffusion_model(
                    model,
                    optimizer,
                    scheduler,
                    train_loader,
                    TIMESTEPS,
                    betas,
                    sqrt_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod,
                    epochs=int(ddpm_epochs_cv),
                    device=device,
                    epoch_desc=f"DDPM Train {spec.target} (fold {fold_i})",
                )
                fold_models[spec.target] = model.eval()
                fold_input_idxs[spec.target] = input_idxs

            X_va_hybrid = build_hybrid_epochs_ddpm(
                X_va_real,
                specs=EXPERIMENTS,
                trained_models=fold_models,
                trained_input_idxs=fold_input_idxs,
                gen_batch_size=int(args.gen_batch_size),
                alphas=alphas,
                alphas_cumprod=alphas_cumprod,
                betas=betas,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                device=device,
            )

            # Classifiers: train on train_real, evaluate on val_real/val_hybrid
            out_csp = run_csp_lda(
                X_train=X_tr,
                y_train=y_tr,
                X_test_real=X_va_real,
                X_test_hybrid=X_va_hybrid,
                y_test=y_va,
                **csp_cfg_cli,
            )
            out_f = run_fbcsp_lda(
                X_train=X_tr,
                y_train=y_tr,
                X_test_real=X_va_real,
                X_test_hybrid=X_va_hybrid,
                y_test=y_va,
                sfreq=float(sfreq),
                **fbcsp_cfg_cli,
            )

            _reseed_for_eegnet(12_000 + int(fold_i))
            out_eeg = run_eegnet(
                X_train=X_tr,
                y_train=y_tr,
                X_test_real=X_va_real,
                X_test_hybrid=X_va_hybrid,
                y_test=y_va,
                device=device,
                epochs=int(args.cls_epochs),
                batch_size=int(args.cls_batch_size),
                eegnet_lr=float(args.eegnet_lr),
                eegnet_dropout=float(args.eegnet_dropout),
                eegnet_patience=int(args.eegnet_patience),
            )

            for name, out in [("CSP+LDA", out_csp), ("FBCSP+LDA", out_f), ("EEGNet", out_eeg)]:
                no_cv = out.get("no_cv", {})
                cv_acc[name].append(float(no_cv.get("acc_real", float("nan"))))
                cv_acc_h[name].append(float(no_cv.get("acc_hybrid", float("nan"))))
                cv_kappa[name].append(float(no_cv.get("kappa_real", float("nan"))))
                cv_kappa_h[name].append(float(no_cv.get("kappa_hybrid", float("nan"))))

        def _mean_std(vals: list[float]) -> tuple[float, float]:
            a = np.asarray(vals, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan"), float("nan")
            return float(np.mean(a)), float(np.std(a))

        cv_summary: dict[str, dict[str, float]] = {}
        for name in ["CSP+LDA", "FBCSP+LDA", "EEGNet"]:
            m1, s1 = _mean_std(cv_acc[name])
            m2, s2 = _mean_std(cv_acc_h[name])
            cv_summary[name] = {"cv_acc_real_mean": m1, "cv_acc_real_std": s1, "cv_acc_h_mean": m2, "cv_acc_h_std": s2}

        # -------------------------------
        # 2) Train on full TRAIN, eval on held-out TEST (raw04 + labels8 asc)
        # -------------------------------
        _log_section("DDPM: train per target channel on FULL TRAIN + Table 1 (MSE/Pearson) on TEST")

        # Prepare held-out train/test preprocessing only now (after CV).
        X_train_full, X_test_real = _prepare_full_train_test_preproc()
        if float(args.final_train_fraction) < 0.999999:
            idx_sub = stratified_group_subsample_indices(
                y_train_full,
                groups_train,
                train_fraction=float(args.final_train_fraction),
                seed=int(args.seed),
            )
            X_train_full = np.asarray(X_train_full[idx_sub], dtype=np.float32)
            y_train_full = np.asarray(y_train_full[idx_sub], dtype=int)
            print(
                f"[stability] final-train-fraction={float(args.final_train_fraction):.3f} "
                f"=> held-out training uses {int(idx_sub.size)}/{int(labels_train.size)} epochs"
            )

        trained_models = {}
        trained_input_idxs = {}
        ddpm_rows = []
        for spec in EXPERIMENTS:
            target_idx = CHANNEL_TO_IDX[spec.target]
            input_idxs = (CHANNEL_TO_IDX[spec.inputs[0]], CHANNEL_TO_IDX[spec.inputs[1]])

            train_ds = EEGEpochs(X_train_full, input_idxs=input_idxs, target_idx=target_idx)
            test_ds = EEGEpochs(X_test_real, input_idxs=input_idxs, target_idx=target_idx)
            train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
            test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False)

            model = UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

            print("\n------------------------------------")
            print(f"Training target={spec.target} inputs={spec.inputs}")
            print("train epochs:", len(train_ds), "test epochs:", len(test_ds))

            train_diffusion_model(
                model,
                optimizer,
                scheduler,
                train_loader,
                TIMESTEPS,
                betas,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                epochs=int(args.ddpm_epochs),
                device=device,
                epoch_desc=f"DDPM Train {spec.target}",
            )

            trained_models[spec.target] = model.eval()
            trained_input_idxs[spec.target] = input_idxs

            mse_list = []
            corr_list = []
            eval_total = len(test_ds) if n_eval_epochs is None else min(int(n_eval_epochs), len(test_ds))
            eval_pbar = tqdm(total=eval_total, desc=f"DDPM Eval {spec.target}", leave=True)
            eval_seen = 0
            for conditioning, target in test_loader:
                if eval_seen >= eval_total:
                    break
                conditioning = conditioning.to(device)
                target = target.to(device)
                remaining = eval_total - eval_seen
                if conditioning.size(0) > remaining:
                    conditioning = conditioning[:remaining]
                    target = target[:remaining]
                syn = sample_from_model(
                    model,
                    conditioning,
                    alphas,
                    alphas_cumprod,
                    betas,
                    sqrt_one_minus_alphas_cumprod,
                    shape=target.shape,
                    device=device,
                )
                target_np = target.detach().cpu().numpy()[:, 0]
                syn_np = syn.detach().cpu().numpy()[:, 0]
                for i in range(target_np.shape[0]):
                    real_1d = target_np[i]
                    syn_1d = syn_np[i]
                    mse_list.append(float(np.mean((syn_1d - real_1d) ** 2)))
                    corr_list.append(pearson_corr_1d(real_1d, syn_1d))
                eval_seen += int(target_np.shape[0])
                eval_pbar.update(int(target_np.shape[0]))
            eval_pbar.close()

            mse = float(np.mean(mse_list)) if mse_list else float("nan")
            corr = float(np.mean(corr_list)) if corr_list else float("nan")
            print(f"DDPM Computed  MSE={mse:.6f} Pearson={corr:.6f} (n={eval_seen})")
            print(f"DDPM Reported  MSE={spec.mse_reported:.6f} Pearson={spec.pearson_reported:.6f}")
            ddpm_rows.append(
                {
                    "Target Channel": spec.target,
                    "Input Channels": f"{spec.inputs[0]}, {spec.inputs[1]}",
                    "MSE (computed)": mse,
                    "Pearson (computed)": corr,
                    "MSE (reported)": spec.mse_reported,
                    "Pearson (reported)": spec.pearson_reported,
                }
            )

        try:
            import pandas as pd

            df = pd.DataFrame(ddpm_rows)
            print("\nDDPM Table 1 (computed vs reported)")
            print(df.to_string(index=False))
        except Exception:
            print("\nDDPM Table 1 (computed vs reported)")
            for r in ddpm_rows:
                print(r)

        _log_section("DDPM: build TEST(hybrid) + evaluate held-out TEST once")
        X_test_hybrid = build_hybrid_epochs_ddpm(
            X_test_real,
            specs=EXPERIMENTS,
            trained_models=trained_models,
            trained_input_idxs=trained_input_idxs,
            gen_batch_size=int(args.gen_batch_size),
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            betas=betas,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            device=device,
        )

        ddpm_acc_summary["CSP+LDA"] = run_csp_lda(
            X_train=X_train_full,
            y_train=y_train_full,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid,
            y_test=y_test,
            **csp_cfg_cli,
        )
        ddpm_acc_summary["FBCSP+LDA"] = run_fbcsp_lda(
            X_train=X_train_full,
            y_train=y_train_full,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid,
            y_test=y_test,
            sfreq=float(sfreq),
            **fbcsp_cfg_cli,
        )
        # EEGNet: train ONCE, then evaluate real/hybrid (for fair ablation reuse)
        _reseed_for_eegnet(13_000)
        eegnet_bundle = _train_eegnet_once(
            X_train=X_train_full,
            y_train=y_train_full,
            device=device,
            epochs=int(args.cls_epochs),
            batch_size=int(args.cls_batch_size),
            lr=float(args.eegnet_lr),
            dropout=float(args.eegnet_dropout),
            patience=int(args.eegnet_patience),
        )
        eeg_model = eegnet_bundle["model"]
        eeg_classes = eegnet_bundle["classes"]
        eeg_mu = eegnet_bundle["mu_ch"]
        eeg_sd = eegnet_bundle["sd_ch"]

        m_real = _eval_eegnet_fixed(
            model=eeg_model,
            classes=eeg_classes,
            mu_ch=eeg_mu,
            sd_ch=eeg_sd,
            X_test=X_test_real,
            y_test=y_test,
            device=device,
            batch_size=int(args.cls_batch_size),
        )
        m_h = _eval_eegnet_fixed(
            model=eeg_model,
            classes=eeg_classes,
            mu_ch=eeg_mu,
            sd_ch=eeg_sd,
            X_test=X_test_hybrid,
            y_test=y_test,
            device=device,
            batch_size=int(args.cls_batch_size),
        )
        print("\n=== EEGNet Results (fixed model) ===")
        print("TEST(real):  ", {k: v for k, v in m_real.items() if k != "cm"})
        print("TEST(hybrid):", {k: v for k, v in m_h.items() if k != "cm"})
        ddpm_acc_summary["EEGNet"] = {
            "no_cv": {
                "acc_real": float(m_real["acc"]),
                "acc_hybrid": float(m_h["acc"]),
                "kappa_real": float(m_real["kappa"]),
                "kappa_hybrid": float(m_h["kappa"]),
            },
            "cv": {},
        }

        # Final table: held-out TEST metrics + TRAIN-only CV mean/std
        print("\n" + "=" * 80)
        print(f"DDPM - SUBJECT {subject_id} — Final summary (TRAIN-CV + held-out TEST)")
        print("=" * 80)
        rows = []
        for name, out in ddpm_acc_summary.items():
            no_cv = out.get("no_cv", {})
            cv = cv_summary.get(name, {})
            rows.append(
                {
                    "Model": name,
                    "Acc Original": float(no_cv.get("acc_real", float("nan"))),
                    "Acc Hybrid": float(no_cv.get("acc_hybrid", float("nan"))),
                    "Kappa Original": float(no_cv.get("kappa_real", float("nan"))),
                    "Kappa Hybrid": float(no_cv.get("kappa_hybrid", float("nan"))),
                    "CV Acc Original (mean/std)": f"{cv.get('cv_acc_real_mean', float('nan')):.3f} ({cv.get('cv_acc_real_std', float('nan')):.3f})",
                    "CV Acc Hybrid (mean/std)": f"{cv.get('cv_acc_h_mean', float('nan')):.3f} ({cv.get('cv_acc_h_std', float('nan')):.3f})",
                }
            )
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            print(df.to_string(index=False))
        except Exception:
            for r in rows:
                print(r)

        # -------------------------------
        # 3) Ablation studies (spatial + conditioning)  [replaces old 0/50/100% block]
        # -------------------------------
        _log_section(f"=== ABLATION STUDIES (spatial + conditioning) - DDPM SUBJECT {subject_id} ===")

        ablation_targets = [spec.target for spec in EXPERIMENTS]
        candidates = list(CHANNEL_NAMES)
        positions_1020 = _standard_1020_positions(list(CHANNEL_NAMES))
        rng_ab = np.random.default_rng(int(args.seed))

        def _print_df(title: str, rows: list[dict[str, object]]) -> None:
            _log_section(title)
            try:
                import pandas as pd

                df = pd.DataFrame(rows)
                print(df.to_string(index=False))
            except Exception:
                for r in rows:
                    print(r)

        def _eval_three_models(X_h: np.ndarray) -> dict[str, dict[str, float]]:
            out_csp = run_csp_lda(
                X_train=X_train_full,
                y_train=y_train_full,
                X_test_real=X_test_real,
                X_test_hybrid=X_h,
                y_test=y_test,
                **csp_cfg_cli,
            )
            out_f = run_fbcsp_lda(
                X_train=X_train_full,
                y_train=y_train_full,
                X_test_real=X_test_real,
                X_test_hybrid=X_h,
                y_test=y_test,
                sfreq=float(sfreq),
                **fbcsp_cfg_cli,
            )

            # FAIR: reuse the SAME trained EEGNet weights across ablation scenarios
            m_hh = _eval_eegnet_fixed(
                model=eeg_model,
                classes=eeg_classes,
                mu_ch=eeg_mu,
                sd_ch=eeg_sd,
                X_test=X_h,
                y_test=y_test,
                device=device,
                batch_size=int(args.cls_batch_size),
            )
            out_eeg = {
                "no_cv": {
                    "acc_real": float(m_real["acc"]),
                    "kappa_real": float(m_real["kappa"]),
                    "acc_hybrid": float(m_hh["acc"]),
                    "kappa_hybrid": float(m_hh["kappa"]),
                },
                "cv": {},
            }

            outs = {"CSP+LDA": out_csp, "FBCSP+LDA": out_f, "EEGNet": out_eeg}
            flat: dict[str, dict[str, float]] = {}
            for name, out in outs.items():
                no_cv = out.get("no_cv", {})
                flat[name] = {
                    "acc_hybrid": float(no_cv.get("acc_hybrid", float("nan"))),
                    "kappa_hybrid": float(no_cv.get("kappa_hybrid", float("nan"))),
                }
            return flat

        # (A) Spatial Neighbor Ablation: NEAR vs FAR vs RANDOM (same DDPM weights, different conditioning channels)
        _log_section("Ablation 1/3 — Spatial Neighbor Ablation (NEAR vs FAR vs RANDOM)")
        near_input_map_idxs: dict[str, tuple[int, ...]] = {
            spec.target: (CHANNEL_TO_IDX[spec.inputs[0]], CHANNEL_TO_IDX[spec.inputs[1]]) for spec in EXPERIMENTS
        }
        far_input_map_idxs: dict[str, tuple[int, ...]] = {}
        rnd_input_map_idxs: dict[str, tuple[int, ...]] = {}
        for spec in EXPERIMENTS:
            k = int(len(near_input_map_idxs[spec.target]))
            far_names = _pick_far_channels(spec.target, k=k, candidates=candidates, positions=positions_1020, rng=rng_ab)
            rnd_names = _pick_random_channels(spec.target, k=k, candidates=candidates, rng=rng_ab)
            far_input_map_idxs[spec.target] = tuple(CHANNEL_TO_IDX[n] for n in far_names)
            rnd_input_map_idxs[spec.target] = tuple(CHANNEL_TO_IDX[n] for n in rnd_names)

        # NEAR hybrid already computed above as X_test_hybrid
        X_test_far = build_hybrid_epochs_ddpm(
            X_test_real,
            specs=EXPERIMENTS,
            trained_models=trained_models,
            trained_input_idxs=far_input_map_idxs,
            gen_batch_size=int(args.gen_batch_size),
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            betas=betas,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            device=device,
        )
        X_test_rnd = build_hybrid_epochs_ddpm(
            X_test_real,
            specs=EXPERIMENTS,
            trained_models=trained_models,
            trained_input_idxs=rnd_input_map_idxs,
            gen_batch_size=int(args.gen_batch_size),
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            betas=betas,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            device=device,
        )
        spatial_scenarios = {"NEAR": X_test_hybrid, "FAR": X_test_far, "RANDOM": X_test_rnd}
        spatial_summary_rows: list[dict[str, object]] = []
        for scen_name, X_h in spatial_scenarios.items():
            clf = _eval_three_models(X_h)
            rec_rows, rec_sum = _reconstruction_metrics(X_test_real, X_h, targets=ablation_targets, sfreq=float(sfreq))
            _print_df(f"Spatial Neighbor Ablation — {scen_name} reconstruction metrics", rec_rows)
            spatial_summary_rows.append(
                {
                    "Scenario": scen_name,
                    "CSP Acc": clf["CSP+LDA"]["acc_hybrid"],
                    "CSP Kappa": clf["CSP+LDA"]["kappa_hybrid"],
                    "FBCSP Acc": clf["FBCSP+LDA"]["acc_hybrid"],
                    "FBCSP Kappa": clf["FBCSP+LDA"]["kappa_hybrid"],
                    "EEGNet Acc": clf["EEGNet"]["acc_hybrid"],
                    "EEGNet Kappa": clf["EEGNet"]["kappa_hybrid"],
                    "Pearson (mean)": float(rec_sum["Pearson_mean"]),
                    "BandpowerSim (mean)": float(rec_sum["BandpowerSim_mean"]),
                }
            )
        _print_df("Spatial Neighbor Ablation — Summary", spatial_summary_rows)

        # (B) Condition Set Size Ablation: COND_2 vs COND_4 vs COND_8 (retrain DDPM per target; only conditioning set changes)
        _log_section("Ablation 2/3 — Condition Set Size Ablation (COND_2 vs COND_4 vs COND_8)")
        size_configs: list[tuple[str, int]] = [("COND_2", 2), ("COND_4", 4), ("COND_8", 8)]
        size_results_summary: list[dict[str, object]] = []

        # COND_2: reuse already-trained models and already-built hybrid
        X_by_size: dict[str, np.ndarray] = {"COND_2": X_test_hybrid}
        models_by_size: dict[str, dict[str, nn.Module]] = {"COND_2": trained_models}
        input_by_size: dict[str, dict[str, tuple[int, ...]]] = {"COND_2": near_input_map_idxs}

        for cfg_name, k_total in size_configs:
            if cfg_name == "COND_2":
                continue
            _log_section(f"Training DDPMs for {cfg_name} (conditioning size={k_total})")
            cfg_models: dict[str, nn.Module] = {}
            cfg_inputs: dict[str, tuple[int, ...]] = {}
            for spec in EXPERIMENTS:
                base = (spec.inputs[0], spec.inputs[1])
                cond_names = _pick_nearest_ring_channels(
                    spec.target,
                    base=base,
                    k_total=int(k_total),
                    candidates=candidates,
                    positions=positions_1020,
                    rng=rng_ab,
                )
                input_idxs = tuple(CHANNEL_TO_IDX[n] for n in cond_names)
                target_idx = CHANNEL_TO_IDX[spec.target]

                train_ds = EEGEpochs(X_train_full, input_idxs=input_idxs, target_idx=target_idx)
                train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)

                model = UNet(in_channels=int(len(input_idxs) + 1), out_channels=1, n_feat=64, time_emb_dim=256).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
                train_diffusion_model(
                    model,
                    optimizer,
                    scheduler,
                    train_loader,
                    TIMESTEPS,
                    betas,
                    sqrt_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod,
                    epochs=int(args.ddpm_epochs),
                    device=device,
                    epoch_desc=f"DDPM Train {spec.target} ({cfg_name})",
                )
                cfg_models[spec.target] = model.eval()
                cfg_inputs[spec.target] = input_idxs

            X_h = build_hybrid_epochs_ddpm(
                X_test_real,
                specs=EXPERIMENTS,
                trained_models=cfg_models,
                trained_input_idxs=cfg_inputs,
                gen_batch_size=int(args.gen_batch_size),
                alphas=alphas,
                alphas_cumprod=alphas_cumprod,
                betas=betas,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                device=device,
            )
            X_by_size[cfg_name] = X_h
            models_by_size[cfg_name] = cfg_models
            input_by_size[cfg_name] = cfg_inputs

        for cfg_name, _ in size_configs:
            X_h = X_by_size[cfg_name]
            clf = _eval_three_models(X_h)
            rec_rows, rec_sum = _reconstruction_metrics(X_test_real, X_h, targets=ablation_targets, sfreq=float(sfreq))
            _print_df(f"Condition Set Size Ablation — {cfg_name} reconstruction metrics", rec_rows)
            size_results_summary.append(
                {
                    "Config": cfg_name,
                    "CSP Acc": clf["CSP+LDA"]["acc_hybrid"],
                    "CSP Kappa": clf["CSP+LDA"]["kappa_hybrid"],
                    "FBCSP Acc": clf["FBCSP+LDA"]["acc_hybrid"],
                    "FBCSP Kappa": clf["FBCSP+LDA"]["kappa_hybrid"],
                    "EEGNet Acc": clf["EEGNet"]["acc_hybrid"],
                    "EEGNet Kappa": clf["EEGNet"]["kappa_hybrid"],
                    "Pearson (mean)": float(rec_sum["Pearson_mean"]),
                    "BandpowerSim (mean)": float(rec_sum["BandpowerSim_mean"]),
                }
            )
        _print_df("Condition Set Size Ablation — Summary", size_results_summary)

        # (C) Conditional vs Non-Conditional DDPM Ablation
        _log_section("Ablation 3/3 — Conditional vs Non-Conditional DDPM")

        # Conditional: reuse existing models + hybrid
        rec_rows_cond, rec_sum_cond = _reconstruction_metrics(X_test_real, X_test_hybrid, targets=ablation_targets, sfreq=float(sfreq))
        _print_df("Conditional DDPM — reconstruction metrics", rec_rows_cond)
        clf_cond = _eval_three_models(X_test_hybrid)

        # NonConditional: train DDPM per target with empty conditioning inputs
        _log_section("Training Non-Conditional DDPMs (no spatial conditioning)")
        nc_models: dict[str, nn.Module] = {}
        nc_inputs: dict[str, tuple[int, ...]] = {}
        for spec in EXPERIMENTS:
            target_idx = CHANNEL_TO_IDX[spec.target]
            input_idxs: tuple[int, ...] = tuple()  # no conditioning channels
            train_ds = EEGEpochs(X_train_full, input_idxs=input_idxs, target_idx=target_idx)
            train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)

            model = UNet(in_channels=1, out_channels=1, n_feat=64, time_emb_dim=256).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            train_diffusion_model(
                model,
                optimizer,
                scheduler,
                train_loader,
                TIMESTEPS,
                betas,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                epochs=int(args.ddpm_epochs),
                device=device,
                epoch_desc=f"DDPM Train {spec.target} (NonConditional)",
            )
            nc_models[spec.target] = model.eval()
            nc_inputs[spec.target] = input_idxs

        X_test_nc = build_hybrid_epochs_ddpm(
            X_test_real,
            specs=EXPERIMENTS,
            trained_models=nc_models,
            trained_input_idxs=nc_inputs,
            gen_batch_size=int(args.gen_batch_size),
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            betas=betas,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            device=device,
        )
        rec_rows_nc, rec_sum_nc = _reconstruction_metrics(X_test_real, X_test_nc, targets=ablation_targets, sfreq=float(sfreq))
        _print_df("Non-Conditional DDPM — reconstruction metrics", rec_rows_nc)
        clf_nc = _eval_three_models(X_test_nc)

        cond_vs_nc_rows = [
            {
                "Mode": "Conditional",
                "CSP Acc": clf_cond["CSP+LDA"]["acc_hybrid"],
                "CSP Kappa": clf_cond["CSP+LDA"]["kappa_hybrid"],
                "FBCSP Acc": clf_cond["FBCSP+LDA"]["acc_hybrid"],
                "FBCSP Kappa": clf_cond["FBCSP+LDA"]["kappa_hybrid"],
                "EEGNet Acc": clf_cond["EEGNet"]["acc_hybrid"],
                "EEGNet Kappa": clf_cond["EEGNet"]["kappa_hybrid"],
                "Pearson (mean)": float(rec_sum_cond["Pearson_mean"]),
                "BandpowerSim (mean)": float(rec_sum_cond["BandpowerSim_mean"]),
            },
            {
                "Mode": "NonConditional",
                "CSP Acc": clf_nc["CSP+LDA"]["acc_hybrid"],
                "CSP Kappa": clf_nc["CSP+LDA"]["kappa_hybrid"],
                "FBCSP Acc": clf_nc["FBCSP+LDA"]["acc_hybrid"],
                "FBCSP Kappa": clf_nc["FBCSP+LDA"]["kappa_hybrid"],
                "EEGNet Acc": clf_nc["EEGNet"]["acc_hybrid"],
                "EEGNet Kappa": clf_nc["EEGNet"]["kappa_hybrid"],
                "Pearson (mean)": float(rec_sum_nc["Pearson_mean"]),
                "BandpowerSim (mean)": float(rec_sum_nc["BandpowerSim_mean"]),
            },
        ]
        _print_df("Conditional vs Non-Conditional DDPM — Summary", cond_vs_nc_rows)

    # ----------------------------------------------------------------------------------
    # SGM/ScoreNet pipeline
    # ----------------------------------------------------------------------------------
    if not args.skip_sgm:
        _log_section("SGM/ScoreNet: train per target channel + Table (MSE/Pearson) + hybrid + CSP+LDA")
        epochs_train, epochs_test = _prepare_full_train_test_preproc()

        # VP-SDE + ScoreNet (U-Net 1D) + EMA + Predictor-Corrector sampler (paper-like)
        # Compatibilidad "código anterior":
        # - scorenet-unet -> VE-SDE (tutorial Song)
        # - ddpm-unet -> VP-SDE
        sde_train = VPSDE(beta_min=0.1, beta_max=float(args.sgm_vp_beta_max), N=1000, T=1.0)
        sde_sampling = VPSDE(beta_min=0.1, beta_max=float(args.sgm_vp_beta_max), N=1000, T=1.0)
        ve_sigma = float(args.sgm_ve_sigma)

        ema = EMA(beta=0.9999)

        print(
            "[SGM cfg] train_steps="
            f"{int(args.sgm_train_steps)} | eval_every={int(args.sgm_eval_every)} | "
            "optimizer=Adam | lr=1e-4 | betas=(0.99,0.999) | grad_clip=1.0 | ema_beta=0.9999"
        )
        print(f"[SGM backbone] {args.sgm_backbone}")
        print(
            "[SGM sampling cfg] "
            f"sampling_N={int(args.sgm_sampling_n)} | eps={float(args.sgm_sampling_eps):g} | "
            f"snr={float(args.sgm_snr):g} | n_steps_each={int(args.sgm_n_steps_each)} | "
            f"sampler={args.sgm_sampler} | ode_method={args.sgm_ode_method} | "
            f"init_corrector_steps={int(args.sgm_init_corrector_steps)} | "
            f"noise_removal={bool(args.sgm_noise_removal)} | "
            f"lw={bool(args.sgm_likelihood_weighting)} | score_clip={float(args.sgm_score_clip):g} | "
            f"ve_sigma={float(ve_sigma):g} | vp_beta_max={float(args.sgm_vp_beta_max):g}"
        )

        # Load ScoreNet-style UNet from SGDM.py (Gaussian Fourier + GroupNorm)
        # (Requiere que `SGDM.py` esté en el mismo directorio que `pipeline.py`.)
        def _load_sgdm_unet():
            import importlib.util

            sgdm_path = Path(__file__).resolve().parent / "SGDM.py"
            if not sgdm_path.exists():
                raise FileNotFoundError(
                    "No se encontró `SGDM.py` junto a `pipeline.py`.\n"
                    f"Ruta esperada: {sgdm_path}\n"
                    "Solución: copia/descarga `SGDM.py` en el mismo directorio o usa `--sgm-backbone ddpm-unet`."
                )
            spec = importlib.util.spec_from_file_location("SGDM", str(sgdm_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"No se pudo importar SGDM.py desde {sgdm_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        sgdm_mod = None
        if args.sgm_backbone == "scorenet-unet":
            sgdm_mod = _load_sgdm_unet()

        ckpt_dir = Path(str(args.sgm_ckpt_dir)).expanduser().resolve()
        if bool(args.sgm_cache):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            print(f"[SGM ckpt] cache=on | dir={ckpt_dir}")
        else:
            print("[SGM ckpt] cache=off")

        # Single bundle file per subject (requested): BCI3_subject{N}.pth
        bundle_path = ckpt_dir / f"BCI3_subject{subject_id}.pth"
        bundle: dict[str, object] = {"models": {}, "meta": {}}
        if bool(args.sgm_cache) and (not bool(args.sgm_force_train)) and bundle_path.exists():
            try:
                bundle = torch.load(bundle_path, map_location=device)
                print(f"[SGM ckpt] bundle loaded: {bundle_path.name}")
            except Exception as e:
                print(f"[SGM ckpt] bundle load failed ({bundle_path.name}): {e}. Will start a new bundle.")
                bundle = {"models": {}, "meta": {}}

        def _bundle_save() -> None:
            if not bool(args.sgm_cache):
                return
            try:
                tmp_path = bundle_path.with_suffix(".tmp")
                torch.save(bundle, tmp_path)
                tmp_path.replace(bundle_path)
            except Exception as e:
                print(f"[SGM ckpt] bundle save failed ({bundle_path.name}): {e}")

        models_by_cfg = bundle.get("models")
        if not isinstance(models_by_cfg, dict):
            models_by_cfg = {}
            bundle["models"] = models_by_cfg

        # include minimal metadata for traceability
        meta = bundle.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            bundle["meta"] = meta
        meta.setdefault("subject_id", int(subject_id))
        meta.setdefault("channel_names", list(CHANNEL_NAMES))

        trained_models_sgm: dict[str, nn.Module] = {}
        trained_input_idxs_sgm: dict[str, tuple[int, int]] = {}
        sgm_rows: list[dict[str, object]] = []

        for spec in EXPERIMENTS:
            target_idx = CHANNEL_TO_IDX[spec.target]
            input_idxs = (CHANNEL_TO_IDX[spec.inputs[0]], CHANNEL_TO_IDX[spec.inputs[1]])

            train_ds = EEGEpochs(epochs_train, input_idxs=input_idxs, target_idx=target_idx)
            test_ds = EEGEpochs(epochs_test, input_idxs=input_idxs, target_idx=target_idx)
            train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
            test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False)

            sde_name = "ve" if args.sgm_backbone == "scorenet-unet" else "vp"
            cfg_key = (
                f"{sde_name}|{args.sgm_backbone}|steps{int(args.sgm_train_steps)}|"
                f"ve_sigma{float(ve_sigma):g}|vp_beta_max{float(args.sgm_vp_beta_max):g}|"
                f"lw{int(bool(args.sgm_likelihood_weighting))}|clip{float(args.sgm_score_clip):g}"
            )
            cfg_bucket = models_by_cfg.get(cfg_key)
            if not isinstance(cfg_bucket, dict):
                cfg_bucket = {}
                models_by_cfg[cfg_key] = cfg_bucket

            if args.sgm_backbone == "scorenet-unet":
                assert sgdm_mod is not None
                model = sgdm_mod.UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256, sigma=25.0).to(device)
                t_embed_scale = None  # continuous t (Gaussian Fourier)
            else:
                model = UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256).to(device)
                t_embed_scale = float(sde_train.N)  # DDPM-style time embedding
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.99, 0.999), eps=1e-8, weight_decay=0.0)
            ema_state = ema.init(model)

            print("\n------------------------------------")
            print(f"SGM target={spec.target} inputs={spec.inputs}")
            print("train epochs:", len(train_ds), "test epochs:", len(test_ds), "| train_steps:", int(args.sgm_train_steps))

            loaded_from_ckpt = False
            if bool(args.sgm_cache) and (not bool(args.sgm_force_train)) and spec.target in cfg_bucket:
                try:
                    model.load_state_dict(cfg_bucket[spec.target], strict=True)
                    loaded_from_ckpt = True
                    print(f"[SGM ckpt] loaded from bundle: {bundle_path.name} | cfg={cfg_key} | target={spec.target}")
                except Exception as e:
                    print(f"[SGM ckpt] load failed (bundle target={spec.target}): {e}. Will train from scratch.")

            if not loaded_from_ckpt:
                if args.sgm_backbone == "scorenet-unet":
                    train_score_model_ve(
                        model=model,
                        sigma=float(ve_sigma),
                        dataloader=train_loader,
                        optimizer=optimizer,
                        device=device,
                        steps=int(args.sgm_train_steps),
                        grad_clip=1.0,
                        ema=ema,
                        ema_state=ema_state,
                        eval_every=int(args.sgm_eval_every),
                        desc=f"SGM Train {spec.target}",
                        likelihood_weighting=bool(args.sgm_likelihood_weighting),
                        score_clip=float(args.sgm_score_clip),
                        t_embed_scale=t_embed_scale,
                    )
                else:
                    train_score_model_vp(
                        model=model,
                        sde=sde_train,
                        dataloader=train_loader,
                        optimizer=optimizer,
                        device=device,
                        steps=int(args.sgm_train_steps),
                        grad_clip=1.0,
                        ema=ema,
                        ema_state=ema_state,
                        eval_every=int(args.sgm_eval_every),
                        desc=f"SGM Train {spec.target}",
                        t_embed_scale=t_embed_scale,
                        likelihood_weighting=bool(args.sgm_likelihood_weighting),
                        score_clip=float(args.sgm_score_clip),
                    )

            # Guardamos el EMA model para sampling/metrics
            if args.sgm_backbone == "scorenet-unet":
                assert sgdm_mod is not None
                ema_model = sgdm_mod.UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256, sigma=25.0).to(device)
            else:
                ema_model = UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256).to(device)
            if loaded_from_ckpt:
                ema_model.load_state_dict(model.state_dict(), strict=True)
            else:
                ema.copy_to(ema_state, ema_model)

            if bool(args.sgm_cache) and (not loaded_from_ckpt):
                cfg_bucket[spec.target] = ema_model.state_dict()
                meta.setdefault("train_steps", int(args.sgm_train_steps))
                meta.setdefault("backbone", str(args.sgm_backbone))
                meta.setdefault("sde_default_by_backbone", {"scorenet-unet": "ve", "ddpm-unet": "vp"})
                _bundle_save()
                print(f"[SGM ckpt] saved to bundle: {bundle_path.name} | cfg={cfg_key} | target={spec.target}")
            trained_models_sgm[spec.target] = ema_model.eval()
            trained_input_idxs_sgm[spec.target] = input_idxs

            mse_list = []
            corr_list = []

            eval_total = len(test_ds) if n_eval_epochs is None else min(int(n_eval_epochs), len(test_ds))
            eval_pbar = tqdm(total=eval_total, desc=f"SGM Eval {spec.target}", leave=True)
            eval_seen = 0

            for conditioning, target in test_loader:
                if eval_seen >= eval_total:
                    break

                conditioning = conditioning.to(device)
                target = target.to(device)

                remaining = eval_total - eval_seen
                if conditioning.size(0) > remaining:
                    conditioning = conditioning[:remaining]
                    target = target[:remaining]

                if args.sgm_backbone == "scorenet-unet":
                    if args.sgm_sampler == "ode":
                        syn = ode_sampler_ve(
                            model=trained_models_sgm[spec.target],
                            sigma=float(ve_sigma),
                            conditioning_inputs=conditioning,
                            shape=tuple(int(v) for v in target.shape),
                            device=device,
                            sampling_N=int(args.sgm_sampling_n),
                            eps=float(args.sgm_sampling_eps),
                            method=str(args.sgm_ode_method),
                            init_corrector_steps=int(args.sgm_init_corrector_steps),
                            snr=float(args.sgm_snr),
                            score_clip=float(args.sgm_score_clip),
                            t_embed_scale=t_embed_scale,
                        )
                    else:
                        syn = pc_sampler_ve(
                            model=trained_models_sgm[spec.target],
                            sigma=float(ve_sigma),
                            conditioning_inputs=conditioning,
                            shape=tuple(int(v) for v in target.shape),
                            device=device,
                            sampling_N=int(args.sgm_sampling_n),
                            eps=float(args.sgm_sampling_eps),
                            snr=float(args.sgm_snr),
                            n_steps_each=int(args.sgm_n_steps_each),
                            init_corrector_steps=int(args.sgm_init_corrector_steps),
                            score_clip=float(args.sgm_score_clip),
                            t_embed_scale=t_embed_scale,
                        )
                else:
                    if args.sgm_sampler == "ode":
                        syn = ode_sampler_vp(
                            model=trained_models_sgm[spec.target],
                            sde=sde_sampling,
                            conditioning_inputs=conditioning,
                            shape=tuple(int(v) for v in target.shape),
                            device=device,
                            sampling_N=int(args.sgm_sampling_n),
                            eps=float(args.sgm_sampling_eps),
                            t_embed_scale=t_embed_scale,
                            method=str(args.sgm_ode_method),
                            init_corrector_steps=int(args.sgm_init_corrector_steps),
                            snr=float(args.sgm_snr),
                            score_clip=float(args.sgm_score_clip),
                        )
                    else:
                        syn = pc_sampler_vp(
                            model=trained_models_sgm[spec.target],
                            sde=sde_sampling,
                            conditioning_inputs=conditioning,
                            shape=tuple(int(v) for v in target.shape),
                            device=device,
                            sampling_N=int(args.sgm_sampling_n),
                            eps=float(args.sgm_sampling_eps),
                            snr=float(args.sgm_snr),
                            n_steps_each=int(args.sgm_n_steps_each),
                            noise_removal=bool(args.sgm_noise_removal),
                            t_embed_scale=t_embed_scale,
                            init_corrector_steps=int(args.sgm_init_corrector_steps),
                            score_clip=float(args.sgm_score_clip),
                        )

                target_np = target.detach().cpu().numpy()[:, 0]
                syn_np = syn.detach().cpu().numpy()[:, 0]

                for i in range(target_np.shape[0]):
                    real_1d = target_np[i]
                    syn_1d = syn_np[i]
                    mse_list.append(float(np.mean((syn_1d - real_1d) ** 2)))
                    corr_list.append(pearson_corr_1d(real_1d, syn_1d))

                eval_seen += int(target_np.shape[0])
                eval_pbar.update(int(target_np.shape[0]))

            eval_pbar.close()

            mse = float(np.mean(mse_list)) if mse_list else float("nan")
            corr = float(np.mean(corr_list)) if corr_list else float("nan")
            print(f"SGM Computed  MSE={mse:.6f} Pearson={corr:.6f} (n={eval_seen})")

            sgm_rows.append(
                {
                    "Target Channel": spec.target,
                    "Input Channels": f"{spec.inputs[0]}, {spec.inputs[1]}",
                    "MSE (computed)": mse,
                    "Pearson (computed)": corr,
                }
            )

        try:
            import pandas as pd

            df_sgm = pd.DataFrame(sgm_rows)
            print("\nSGM Table (computed)")
            print(df_sgm.to_string(index=False))
        except Exception:
            print("\nSGM Table (computed)")
            for r in sgm_rows:
                print(r)

        # ---- hybrid test set (SGM)
        def build_hybrid_epochs_sgm(epochs_real: np.ndarray) -> np.ndarray:
            epochs_h = epochs_real.copy()
            n = epochs_h.shape[0]
            t_len = epochs_h.shape[2]

            for spec in EXPERIMENTS:
                target = spec.target
                if target not in trained_models_sgm:
                    continue

                model_s = trained_models_sgm[target]
                input_idxs = trained_input_idxs_sgm[target]
                target_idx = CHANNEL_TO_IDX[target]

                pbar = tqdm(total=n, desc=f"SGM Synthesize {target}", leave=False)
                for s in range(0, n, int(args.gen_batch_size)):
                    e = min(n, s + int(args.gen_batch_size))
                    # Condition on REAL inputs to avoid cascaded synthesis.
                    batch_real = epochs_real[s:e]
                    cond = np.stack(
                        [batch_real[:, input_idxs[0], :], batch_real[:, input_idxs[1], :]],
                        axis=1,
                    )
                    cond_t = torch.from_numpy(cond).to(device)
                    if args.sgm_backbone == "scorenet-unet":
                        if args.sgm_sampler == "ode":
                            syn = ode_sampler_ve(
                                model=model_s,
                                sigma=float(ve_sigma),
                                conditioning_inputs=cond_t,
                                shape=(cond_t.size(0), 1, t_len),
                                device=device,
                                sampling_N=int(args.sgm_sampling_n),
                                eps=float(args.sgm_sampling_eps),
                                method=str(args.sgm_ode_method),
                                init_corrector_steps=int(args.sgm_init_corrector_steps),
                                snr=float(args.sgm_snr),
                                score_clip=float(args.sgm_score_clip),
                                t_embed_scale=t_embed_scale,
                            )
                        else:
                            syn = pc_sampler_ve(
                                model=model_s,
                                sigma=float(ve_sigma),
                                conditioning_inputs=cond_t,
                                shape=(cond_t.size(0), 1, t_len),
                                device=device,
                                sampling_N=int(args.sgm_sampling_n),
                                eps=float(args.sgm_sampling_eps),
                                snr=float(args.sgm_snr),
                                n_steps_each=int(args.sgm_n_steps_each),
                                init_corrector_steps=int(args.sgm_init_corrector_steps),
                                score_clip=float(args.sgm_score_clip),
                                t_embed_scale=t_embed_scale,
                            )
                    else:
                        if args.sgm_sampler == "ode":
                            syn = ode_sampler_vp(
                                model=model_s,
                                sde=sde_sampling,
                                conditioning_inputs=cond_t,
                                shape=(cond_t.size(0), 1, t_len),
                                device=device,
                                sampling_N=int(args.sgm_sampling_n),
                                eps=float(args.sgm_sampling_eps),
                                t_embed_scale=t_embed_scale,
                                method=str(args.sgm_ode_method),
                                init_corrector_steps=int(args.sgm_init_corrector_steps),
                                snr=float(args.sgm_snr),
                                score_clip=float(args.sgm_score_clip),
                            )
                        else:
                            syn = pc_sampler_vp(
                                model=model_s,
                                sde=sde_sampling,
                                conditioning_inputs=cond_t,
                                shape=(cond_t.size(0), 1, t_len),
                                device=device,
                                sampling_N=int(args.sgm_sampling_n),
                                eps=float(args.sgm_sampling_eps),
                                snr=float(args.sgm_snr),
                                n_steps_each=int(args.sgm_n_steps_each),
                                noise_removal=bool(args.sgm_noise_removal),
                                t_embed_scale=t_embed_scale,
                                init_corrector_steps=int(args.sgm_init_corrector_steps),
                                score_clip=float(args.sgm_score_clip),
                            )
                    epochs_h[s:e, target_idx, :] = syn.detach().cpu().numpy()[:, 0]
                    pbar.update(e - s)
                pbar.close()

            return epochs_h

        X_train = epochs_train
        y_train = labels_train
        X_test_real = epochs_test
        y_test = labels_test
        X_test_hybrid_sgm = build_hybrid_epochs_sgm(X_test_real)

        _log_section("SGM: hybrid TEST + CSP+LDA + FBCSP+LDA + EEGNet")
        sgm_acc_summary["CSP+LDA"] = run_csp_lda(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid_sgm,
            y_test=y_test,
            **csp_cfg_cli,
        )

        _log_section("SGM: FBCSP+LDA (real vs hybrid) + CV")
        sgm_acc_summary["FBCSP+LDA"] = run_fbcsp_lda(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid_sgm,
            y_test=y_test,
            sfreq=float(sfreq),
            **fbcsp_cfg_cli,
        )

        _log_section("SGM: EEGNet (real vs hybrid)")
        print(
            f"[EEGNet cfg] epochs={int(args.cls_epochs)} | batch_size={int(args.cls_batch_size)} "
            f"| optimizer=Adam | lr={float(args.eegnet_lr):g} | dropout={float(args.eegnet_dropout):g} "
            f"| patience={int(args.eegnet_patience)}"
        )
        _reseed_for_eegnet(14_000)
        sgm_acc_summary["EEGNet"] = run_eegnet(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid_sgm,
            y_test=y_test,
            device=device,
            epochs=int(args.cls_epochs),
            batch_size=int(args.cls_batch_size),
            eegnet_lr=float(args.eegnet_lr),
            eegnet_dropout=float(args.eegnet_dropout),
            eegnet_patience=int(args.eegnet_patience),
        )

        _print_accuracy_table(
            "SGM/ScoreNet — Accuracy summary (no-CV + CV within-test) — Original vs Hybrid",
            sgm_acc_summary,
        )

    _log_section("Done")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.1f}s")

