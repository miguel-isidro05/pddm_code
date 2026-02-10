"""
Pipeline (terminal) equivalente a `DPPM.ipynb`, sin generar imágenes.

Qué hace (secuencial):
- Carga BCI Competition III - Dataset V (Subject 1): raw01+raw02+raw03 (labeled)
- Preprocesa (paper-like): bandpass 8–30Hz + annotate muscle + ICA (Fp1/Fp2) + z-score por canal
- Segmenta epochs de 1s (512) con stride 0.5s (256), sin cruzar cambios de etiqueta ni límites de sesión
- Split 70/30 estratificado (paper)
- Entrena DDPM por canal target (8 canales)
- Evalúa generación (MSE y Pearson) y muestra Tabla 1 (computed vs reported)
- Construye TEST(hybrid) reemplazando esos 8 canales por sintéticos (DDPM)
- Evalúa CSP+LDA (logs + matrices de confusión como arrays, sin plots)
- Entrena/evalúa U-Net clasificador (paper hyperparams: Adam 1e-3, batch 32, epochs 10)
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

    def __init__(self, epochs: np.ndarray, *, input_idxs: tuple[int, int], target_idx: int):
        if epochs.ndim != 3:
            raise ValueError(f"Expected epochs (n_epochs, n_channels, n_times), got {epochs.shape}")
        self.epochs = epochs.astype(np.float32, copy=False)
        self.input_idxs = input_idxs
        self.target_idx = int(target_idx)

    def __len__(self) -> int:
        return int(self.epochs.shape[0])

    def __getitem__(self, idx: int):
        ep = self.epochs[idx]  # (C, T)
        conditioning = ep[list(self.input_idxs), :]  # (2, T)
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
    trained_input_idxs: dict[str, tuple[int, int]],
    gen_batch_size: int,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    betas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Reemplaza canales target por reconstrucción DDPM (z-score)."""
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
            batch = epochs_h[s:e]  # (B, C, T)

            cond = np.stack([batch[:, input_idxs[0], :], batch[:, input_idxs[1], :]], axis=1)  # (B, 2, T)
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
            epochs_h[s:e, target_idx, :] = syn_np
            pbar.update(e - s)
        pbar.close()

    return epochs_h


# --------------------------------------------------------------------------------------
# Classifiers: CSP+LDA + U-Net (paper hyperparams)
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
) -> dict[str, dict[str, float]]:
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, cohen_kappa_score

    CSP_CFG = dict(n_components=4, reg=None, log=True, norm_trace=True, cov_est="epoch")
    LDA_CFG = dict(solver="lsqr", shrinkage="auto")

    classes = _label_classes(y_train, y_test)
    y_train_enc = _encode_labels_with_classes(y_train, classes)
    y_test_enc = _encode_labels_with_classes(y_test, classes)

    print("classes (original labels):", classes.tolist())
    print("TRAIN:", X_train.shape, "TEST(real):", X_test_real.shape, "TEST(hybrid):", X_test_hybrid.shape)

    csp = CSP(**CSP_CFG)
    F_train = csp.fit_transform(X_train, y_train_enc)
    F_test_real = csp.transform(X_test_real)
    F_test_hybrid = csp.transform(X_test_hybrid)

    lda = LinearDiscriminantAnalysis(**LDA_CFG)
    lda.fit(F_train, y_train_enc)

    pred_real = lda.predict(F_test_real)
    pred_hybrid = lda.predict(F_test_hybrid)

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

    # -----------------------
    # Cross-validation (5-fold) dentro del TEST (como en EEGNet_2a.ipynb)
    # -----------------------
    from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
    from sklearn.metrics import cohen_kappa_score

    def _cv_on_features(feats: np.ndarray, y_enc: np.ndarray, *, title: str) -> dict[str, float]:
        groups = np.arange(int(y_enc.shape[0]), dtype=int)
        cv = GroupKFold(n_splits=5)
        lda_cv = LinearDiscriminantAnalysis(**LDA_CFG)

        scores_cv = cross_val_score(lda_cv, feats, y_enc, cv=cv, groups=groups, n_jobs=1)
        y_pred_cv = cross_val_predict(lda_cv, feats, y_enc, cv=cv, groups=groups, method="predict", n_jobs=1)

        cm_cv = confusion_matrix(y_enc, y_pred_cv, normalize="true")
        acc_cv = float(np.mean(y_pred_cv == y_enc))
        kappa_cv = float(cohen_kappa_score(y_enc, y_pred_cv))

        print("\n" + "=" * 80)
        print(f"CSP+LDA — {title} CV (5-fold, within-test)")
        print("=" * 80)
        print(f"CV accuracy      : {acc_cv:.3f}")
        print(f"CV kappa         : {kappa_cv:.3f}")
        print(f"CV mean ± std    : {scores_cv.mean():.3f} ± {scores_cv.std():.3f}")
        print(f"[{title} TEST-CV] confusion_matrix (normalize='true'):\n{np.round(cm_cv, 3)}")

        return {
            "cv_acc": float(scores_cv.mean()),
            "cv_acc_std": float(scores_cv.std()),
            "cv_acc_pred": acc_cv,
            "cv_kappa": kappa_cv,
        }

    cv_real = _cv_on_features(F_test_real, y_test_enc, title="TEST(real)")
    cv_hybrid = _cv_on_features(F_test_hybrid, y_test_enc, title="TEST(hybrid)")

    return {
        "no_cv": {
            "acc_real": acc_real,
            "acc_hybrid": acc_h,
            "kappa_real": kappa_real,
            "kappa_hybrid": kappa_h,
        },
        "cv": {
            "acc_real": float(cv_real["cv_acc"]),
            "acc_real_std": float(cv_real["cv_acc_std"]),
            "acc_hybrid": float(cv_hybrid["cv_acc"]),
            "acc_hybrid_std": float(cv_hybrid["cv_acc_std"]),
        },
    }


def run_fbcsp_lda(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_real: np.ndarray,
    X_test_hybrid: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
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
    from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score

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

    # 32 canales => usamos m_pairs=2 (como en tu snippet)
    m_pairs = 2 if int(X_train.shape[1]) > 3 else 1
    per_band = int(2 * m_pairs)  # features por banda

    CSP_CFG = dict(n_components=per_band, reg=None, log=True, norm_trace=True, cov_est="epoch")
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

        csp_b = CSP(**CSP_CFG)
        F_train_list.append(csp_b.fit_transform(Xtr_b, y_train_enc))
        F_test_real_list.append(csp_b.transform(Xte_r_b))
        F_test_hybrid_list.append(csp_b.transform(Xte_h_b))
        csp_per_band.append(csp_b)

    F_train_full = np.concatenate(F_train_list, axis=1)
    F_test_real_full = np.concatenate(F_test_real_list, axis=1)
    F_test_hybrid_full = np.concatenate(F_test_hybrid_list, axis=1)

    # MIBIF selection (top-k + pares) usando SOLO TRAIN features
    k_best = 4
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
    lda.fit(F_train, y_train_enc)

    pred_real = lda.predict(F_test_real)
    pred_h = lda.predict(F_test_hybrid)

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

    # CV dentro del TEST (sobre features FBCSP)
    def _cv_on_features(feats: np.ndarray, y_enc: np.ndarray, *, title: str) -> dict[str, float]:
        groups = np.arange(int(y_enc.shape[0]), dtype=int)
        cv = GroupKFold(n_splits=5)
        lda_cv = LinearDiscriminantAnalysis(**LDA_CFG)

        scores_cv = cross_val_score(lda_cv, feats, y_enc, cv=cv, groups=groups, n_jobs=1)
        y_pred_cv = cross_val_predict(lda_cv, feats, y_enc, cv=cv, groups=groups, method="predict", n_jobs=1)

        cm_cv = confusion_matrix(y_enc, y_pred_cv, normalize="true")
        acc_cv = float(np.mean(y_pred_cv == y_enc))
        kappa_cv = float(cohen_kappa_score(y_enc, y_pred_cv))

        print("\n" + "=" * 80)
        print(f"FBCSP+LDA — {title} CV (5-fold, within-test)")
        print("=" * 80)
        print(f"CV accuracy      : {acc_cv:.3f}")
        print(f"CV kappa         : {kappa_cv:.3f}")
        print(f"CV mean ± std    : {scores_cv.mean():.3f} ± {scores_cv.std():.3f}")
        print(f"[{title} TEST-CV] confusion_matrix (normalize='true'):\n{np.round(cm_cv, 3)}")

        return {
            "cv_acc": float(scores_cv.mean()),
            "cv_acc_std": float(scores_cv.std()),
            "cv_acc_pred": acc_cv,
            "cv_kappa": kappa_cv,
        }

    cv_real = _cv_on_features(F_test_real, y_test_enc, title="TEST(real)")
    cv_h = _cv_on_features(F_test_hybrid, y_test_enc, title="TEST(hybrid)")

    return {
        "no_cv": {
            "acc_real": acc_real,
            "acc_hybrid": acc_h,
            "kappa_real": kappa_real,
            "kappa_hybrid": kappa_h,
        },
        "cv": {
            "acc_real": float(cv_real["cv_acc"]),
            "acc_real_std": float(cv_real["cv_acc_std"]),
            "acc_hybrid": float(cv_h["cv_acc"]),
            "acc_hybrid_std": float(cv_h["cv_acc_std"]),
        },
    }


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet1DClassifier(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock1D(in_ch, base_ch)
        self.down1 = nn.MaxPool1d(2)

        self.enc2 = ConvBlock1D(base_ch, base_ch * 2)
        self.down2 = nn.MaxPool1d(2)

        self.enc3 = ConvBlock1D(base_ch * 2, base_ch * 4)
        self.down3 = nn.MaxPool1d(2)

        self.bottleneck = ConvBlock1D(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock1D(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock1D(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(base_ch * 2, base_ch)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(base_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        feats = self.head(d1)
        return self.classifier(feats)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Embeddings antes de la capa final (para CV tipo EEGNet_2a.ipynb)."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def run_unet_classifier(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_real: np.ndarray,
    X_test_hybrid: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.metrics import cohen_kappa_score

    # Override (pedido): entrenar más fuerte por terminal
    # - User pidió 64 batch y 500 epochs (y misma configuración para EEGNet y U-Net)
    PAPER_OPTIMIZER = "Adam"
    PAPER_LEARNING_RATE = 1e-4
    PAPER_BATCH_SIZE = int(getattr(run_unet_classifier, "_batch_size", 64))
    PAPER_EPOCHS = int(getattr(run_unet_classifier, "_epochs", 500))

    classes = _label_classes(y_train, y_test)
    y_train_enc = _encode_labels_with_classes(y_train, classes)
    y_test_enc = _encode_labels_with_classes(y_test, classes)
    n_classes = int(np.max(np.concatenate([y_train_enc, y_test_enc])) + 1)

    Xtr = np.asarray(X_train, dtype=np.float32)
    Xte_r = np.asarray(X_test_real, dtype=np.float32)
    Xte_h = np.asarray(X_test_hybrid, dtype=np.float32)
    ytr = np.asarray(y_train_enc, dtype=np.int64)
    yte = np.asarray(y_test_enc, dtype=np.int64)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=PAPER_BATCH_SIZE, shuffle=True)

    model = UNet1DClassifier(in_ch=int(Xtr.shape[1]), n_classes=n_classes, base_ch=32).to(device)
    if PAPER_OPTIMIZER != "Adam":
        raise ValueError(f"Paper optimizer esperado 'Adam', got: {PAPER_OPTIMIZER}")
    opt = torch.optim.Adam(model.parameters(), lr=PAPER_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(PAPER_EPOCHS), desc="U-Net classifier train", leave=True)
    for _ in pbar:
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
            losses.append(float(loss.item()))
        pbar.set_postfix(loss=f"{float(np.mean(losses)):.4f}")

    def _eval(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        model.eval()
        y_pred_parts = []
        with torch.no_grad():
            for i in range(0, x.shape[0], PAPER_BATCH_SIZE):
                xb = torch.from_numpy(x[i : i + PAPER_BATCH_SIZE]).to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                y_pred_parts.append(torch.argmax(probs, dim=1).detach().cpu().numpy())
        y_pred = np.concatenate(y_pred_parts, axis=0)
        acc = float(accuracy_score(y, y_pred))
        kappa = float(cohen_kappa_score(y, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
        return {"acc": acc, "kappa": kappa, "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    metrics_real = _eval(Xte_r, yte)
    metrics_hybrid = _eval(Xte_h, yte)

    print("\n=== U-Net Classifier Results (macro) ===")
    print("TEST(real):  ", metrics_real)
    print("TEST(hybrid):", metrics_hybrid)

    # -----------------------
    # Cross-validation (5-fold) dentro del TEST sobre embeddings (como EEGNet_2a.ipynb)
    # -----------------------
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score

    def _embeddings(x: np.ndarray) -> np.ndarray:
        model.eval()
        feat_parts = []
        with torch.no_grad():
            for i in range(0, x.shape[0], PAPER_BATCH_SIZE):
                xb = torch.from_numpy(x[i : i + PAPER_BATCH_SIZE]).to(device)
                feats = model.forward_features(xb).detach().cpu().numpy()
                feat_parts.append(np.asarray(feats, dtype=float).reshape(feats.shape[0], -1))
        return np.vstack(feat_parts)

    def _cv_embeddings(X_feat: np.ndarray, y_enc: np.ndarray, *, title: str) -> dict[str, float]:
        groups = np.arange(int(y_enc.shape[0]), dtype=int)
        cv = GroupKFold(n_splits=5)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        scores_cv = cross_val_score(clf, X_feat, y_enc, cv=cv, groups=groups, n_jobs=1)
        y_pred_cv = cross_val_predict(clf, X_feat, y_enc, cv=cv, groups=groups, method="predict", n_jobs=1)

        cm_cv = confusion_matrix(y_enc, y_pred_cv, normalize="true")
        acc_cv = float(np.mean(y_pred_cv == y_enc))
        kappa_cv = float(cohen_kappa_score(y_enc, y_pred_cv))

        print("\n" + "=" * 80)
        print(f"U-Net embeddings — {title} CV (5-fold, within-test)")
        print("=" * 80)
        print(f"CV accuracy      : {acc_cv:.3f}")
        print(f"CV kappa         : {kappa_cv:.3f}")
        print(f"CV mean ± std    : {scores_cv.mean():.3f} ± {scores_cv.std():.3f}")
        print(f"[{title} TEST-CV] confusion_matrix (normalize='true'):\n{np.round(cm_cv, 3)}")

        return {
            "cv_acc": float(scores_cv.mean()),
            "cv_acc_std": float(scores_cv.std()),
            "cv_acc_pred": acc_cv,
            "cv_kappa": kappa_cv,
        }

        return {
            "cv_acc": float(scores_cv.mean()),
            "cv_acc_std": float(scores_cv.std()),
            "cv_acc_pred": acc_cv,
            "cv_kappa": kappa_cv,
        }

    X_feat_real = _embeddings(Xte_r)
    X_feat_h = _embeddings(Xte_h)
    cv_real = _cv_embeddings(X_feat_real, yte, title="TEST(real)")
    cv_h = _cv_embeddings(X_feat_h, yte, title="TEST(hybrid)")

    return {
        "no_cv": {
            "acc_real": float(metrics_real["acc"]),
            "acc_hybrid": float(metrics_hybrid["acc"]),
            "kappa_real": float(metrics_real["kappa"]),
            "kappa_hybrid": float(metrics_hybrid["kappa"]),
        },
        "cv": {
            "acc_real": float(cv_real["cv_acc"]),
            "acc_real_std": float(cv_real["cv_acc_std"]),
            "acc_hybrid": float(cv_h["cv_acc"]),
            "acc_hybrid_std": float(cv_h["cv_acc_std"]),
        },
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
        temp_kernel: int = 64,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        pk1: int = 4,
        pk2: int = 8,
        dropout_rate: float = 0.5,
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
) -> dict[str, dict[str, float]]:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
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

    model = EEGNetModel(chans=int(X_train.shape[1]), classes=n_classes, time_points=int(X_train.shape[2])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=int(batch_size), shuffle=True)

    pbar = tqdm(range(int(epochs)), desc="EEGNet train", leave=True, mininterval=0.5)
    for _ in pbar:
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
        pbar.set_postfix(loss=f"{float(np.mean(losses)):.4f}")

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
        kappa = float(cohen_kappa_score(y, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y, y_pred, normalize="true")
        return {"acc": acc, "kappa": kappa, "precision": float(prec), "recall": float(rec), "f1": float(f1), "cm": cm}

    m_real = _eval(Xte_r, yte)
    m_h = _eval(Xte_h, yte)

    print("\n=== EEGNet Results (macro) ===")
    print("TEST(real):  ", {k: v for k, v in m_real.items() if k != "cm"})
    print("TEST(hybrid):", {k: v for k, v in m_h.items() if k != "cm"})
    print("\nConfusionMatrix (normalize=true) TEST(real):\n", np.array2string(m_real["cm"], precision=3, floatmode="fixed"))
    print("\nConfusionMatrix (normalize=true) TEST(hybrid):\n", np.array2string(m_h["cm"], precision=3, floatmode="fixed"))

    # CV (5-fold) dentro del TEST sobre embeddings
    def _embeddings(x: np.ndarray) -> np.ndarray:
        model.eval()
        feat_parts = []
        with torch.no_grad():
            for i in range(0, x.shape[0], int(batch_size)):
                xb = torch.from_numpy(x[i : i + int(batch_size)]).to(device)
                feats = model.forward_features(xb).detach().cpu().numpy()
                feat_parts.append(np.asarray(feats, dtype=float).reshape(feats.shape[0], -1))
        return np.vstack(feat_parts)

    def _cv_embeddings(X_feat: np.ndarray, y_enc: np.ndarray, *, title: str) -> dict[str, float]:
        groups = np.arange(int(y_enc.shape[0]), dtype=int)
        cv = GroupKFold(n_splits=5)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        scores_cv = cross_val_score(clf, X_feat, y_enc, cv=cv, groups=groups, n_jobs=1)
        y_pred_cv = cross_val_predict(clf, X_feat, y_enc, cv=cv, groups=groups, method="predict", n_jobs=1)
        cm_cv = confusion_matrix(y_enc, y_pred_cv, normalize="true")
        acc_cv = float(np.mean(y_pred_cv == y_enc))
        kappa_cv = float(cohen_kappa_score(y_enc, y_pred_cv))

        print("\n" + "=" * 80)
        print(f"EEGNet embeddings — {title} CV (5-fold, within-test)")
        print("=" * 80)
        print(f"CV accuracy      : {acc_cv:.3f}")
        print(f"CV kappa         : {kappa_cv:.3f}")
        print(f"CV mean ± std    : {scores_cv.mean():.3f} ± {scores_cv.std():.3f}")
        print(f"[{title} TEST-CV] confusion_matrix (normalize='true'):\n{np.round(cm_cv, 3)}")

        return {
            "cv_acc": float(scores_cv.mean()),
            "cv_acc_std": float(scores_cv.std()),
            "cv_acc_pred": acc_cv,
            "cv_kappa": kappa_cv,
        }

    X_feat_real = _embeddings(Xte_r)
    X_feat_h = _embeddings(Xte_h)
    cv_real = _cv_embeddings(X_feat_real, yte, title="TEST(real)")
    cv_h = _cv_embeddings(X_feat_h, yte, title="TEST(hybrid)")

    return {
        "no_cv": {"acc_real": float(m_real["acc"]), "acc_hybrid": float(m_h["acc"])},
        "cv": {
            "acc_real": float(cv_real["cv_acc"]),
            "acc_real_std": float(cv_real["cv_acc_std"]),
            "acc_hybrid": float(cv_h["cv_acc"]),
            "acc_hybrid_std": float(cv_h["cv_acc_std"]),
        },
    }



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
        help="BCI Competition III - Dataset V subject ID (1,2,3).",
    )

    parser.add_argument("--ddpm-epochs", type=int, default=200)
    # Score-based (VP-SDE) training (paper-style steps)
    parser.add_argument("--sgm-train-steps", type=int, default=30_000)
    parser.add_argument("--sgm-eval-every", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-eval-epochs", type=int, default=0, help="0 => eval todos; si >0 limita n")
    parser.add_argument("--gen-batch-size", type=int, default=8)
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
    parser.add_argument("--cls-epochs", type=int, default=500, help="Epochs para U-Net classifier y EEGNet")
    parser.add_argument("--cls-batch-size", type=int, default=64, help="Batch size para U-Net classifier y EEGNet")

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

    subject_id = int(args.subject)
    _log_section(f"Load + preprocess (paper-like) + epoching (Subject {subject_id}: raw01+raw02+raw03, 70/30)")

    # ---- load all 3 training sessions (Subject N)
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

    # ---- preprocessing (paper-like): filter + annotate_muscle + ICA(Fp1/Fp2) + z-score
    import mne
    from mne.preprocessing import ICA

    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(x_raw.T, info, verbose=False)

    raw = raw.copy().filter(l_freq=8.0, h_freq=30.0, method="iir", verbose=False)

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

    ica = ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw, verbose=False)

    try:
        eog_idx, _ = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"], verbose=False)
        ica.exclude = eog_idx
        print("ICA exclude components (EOG):", list(map(int, eog_idx)))
    except Exception as e:
        print("ICA EOG detection skipped:", e)

    raw = ica.apply(raw.copy(), verbose=False)

    x_clean = raw.get_data().T.astype(np.float32)  # (n_times, 32)
    # Importante (anti-leakage): NO z-score aquí con todo el dataset.
    # Hacemos el z-score usando SOLO statistics del TRAIN después del split 70/30.
    x_for_epoching = x_clean

    # ---- epoching
    EPOCH_SAMPLES = 512
    STRIDE_SAMPLES = EPOCH_SAMPLES // 2  # 0.5s at 512 Hz

    # Usar todas las clases presentes en este dataset (2=left, 3=right, 7=word)
    allowed_labels = set(int(v) for v in np.unique(y_raw).tolist())
    max_epochs_per_class = None
    train_ratio = 0.7

    epochs_parts: list[np.ndarray] = []
    labels_parts_out: list[np.ndarray] = []
    offset = 0
    for n_sess in len_parts:
        x_sess = x_for_epoching[offset : offset + n_sess]
        y_sess = y_raw[offset : offset + n_sess]
        ep_sess, lab_sess = build_epochs_from_labels(
            x_sess,
            y_sess,
            epoch_samples=EPOCH_SAMPLES,
            stride_samples=STRIDE_SAMPLES,
            allowed_labels=allowed_labels,
            max_epochs_per_class=max_epochs_per_class,
        )
        epochs_parts.append(ep_sess)
        labels_parts_out.append(lab_sess)
        offset += n_sess

    epochs_all = np.concatenate(epochs_parts, axis=0)
    labels_all = np.concatenate(labels_parts_out, axis=0)

    print("allowed_labels:", sorted(int(v) for v in allowed_labels))
    train_idx, test_idx = stratified_split(labels_all, train_ratio=train_ratio, seed=args.seed)
    epochs_train = epochs_all[train_idx]
    labels_train = labels_all[train_idx]
    epochs_test = epochs_all[test_idx]
    labels_test = labels_all[test_idx]

    # Z-score per channel using TRAIN statistics only (paper-like + no leakage)
    train_mu = epochs_train.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    train_sd = epochs_train.std(axis=(0, 2), keepdims=True) + 1e-6
    epochs_train = ((epochs_train - train_mu) / train_sd).astype(np.float32, copy=False)
    epochs_test = ((epochs_test - train_mu) / train_sd).astype(np.float32, copy=False)
    print("[z-score] applied per-channel using TRAIN stats only")

    print("epochs_train:", epochs_train.shape, "epochs_test:", epochs_test.shape)
    print("class balance train:", _class_balance(labels_train))
    print("class balance test :", _class_balance(labels_test))

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
        _log_section("DDPM: train per target channel + Table 1 (MSE/Pearson)")

        for spec in EXPERIMENTS:
            target_idx = CHANNEL_TO_IDX[spec.target]
            input_idxs = (CHANNEL_TO_IDX[spec.inputs[0]], CHANNEL_TO_IDX[spec.inputs[1]])

            train_ds = EEGEpochs(epochs_train, input_idxs=input_idxs, target_idx=target_idx)
            test_ds = EEGEpochs(epochs_test, input_idxs=input_idxs, target_idx=target_idx)
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

            # ---- eval (z-score)
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

        _log_section("DDPM: hybrid TEST + CSP+LDA + U-Net classifier")

        X_train = epochs_train
        y_train = labels_train
        X_test_real = epochs_test
        y_test = labels_test

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
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid,
            y_test=y_test,
        )

        _log_section("FBCSP+LDA (real vs hybrid) + CV")
        ddpm_acc_summary["FBCSP+LDA"] = run_fbcsp_lda(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid,
            y_test=y_test,
            sfreq=float(sfreq),
        )

        _log_section("U-Net classifier (real vs hybrid) + embeddings-CV")
        # pasar hiperparámetros solicitados vía atributos (mantener firma simple)
        run_unet_classifier._epochs = int(args.cls_epochs)  # type: ignore[attr-defined]
        run_unet_classifier._batch_size = int(args.cls_batch_size)  # type: ignore[attr-defined]
        print(f"[U-Net cfg] epochs={int(args.cls_epochs)} | batch_size={int(args.cls_batch_size)} | optimizer=Adam | lr=1e-4")
        ddpm_acc_summary["U-Net"] = run_unet_classifier(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid,
            y_test=y_test,
            device=device,
        )

        _log_section("EEGNet (real vs hybrid) + embeddings-CV")
        print(f"[EEGNet cfg] epochs={int(args.cls_epochs)} | batch_size={int(args.cls_batch_size)} | optimizer=Adam | lr=1e-4")
        ddpm_acc_summary["EEGNet"] = run_eegnet(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid,
            y_test=y_test,
            device=device,
            epochs=int(args.cls_epochs),
            batch_size=int(args.cls_batch_size),
        )

        _print_accuracy_table(
            "DDPM — Accuracy summary (no-CV + CV within-test) — Original vs Hybrid",
            ddpm_acc_summary,
        )

    # ----------------------------------------------------------------------------------
    # SGM/ScoreNet pipeline
    # ----------------------------------------------------------------------------------
    if not args.skip_sgm:
        _log_section("SGM/ScoreNet: train per target channel + Table (MSE/Pearson) + hybrid + CSP+LDA")

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
                    batch = epochs_h[s:e]
                    cond = np.stack([batch[:, input_idxs[0], :], batch[:, input_idxs[1], :]], axis=1)
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

        _log_section("SGM: hybrid TEST + CSP+LDA + FBCSP+LDA + U-Net + EEGNet")
        sgm_acc_summary["CSP+LDA"] = run_csp_lda(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid_sgm,
            y_test=y_test,
        )

        _log_section("SGM: FBCSP+LDA (real vs hybrid) + CV")
        sgm_acc_summary["FBCSP+LDA"] = run_fbcsp_lda(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid_sgm,
            y_test=y_test,
            sfreq=float(sfreq),
        )

        _log_section("SGM: U-Net classifier (real vs hybrid) + embeddings-CV")
        run_unet_classifier._epochs = int(args.cls_epochs)  # type: ignore[attr-defined]
        run_unet_classifier._batch_size = int(args.cls_batch_size)  # type: ignore[attr-defined]
        print(f"[U-Net cfg] epochs={int(args.cls_epochs)} | batch_size={int(args.cls_batch_size)} | optimizer=Adam | lr=1e-4")
        sgm_acc_summary["U-Net"] = run_unet_classifier(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid_sgm,
            y_test=y_test,
            device=device,
        )

        _log_section("SGM: EEGNet (real vs hybrid) + embeddings-CV")
        print(f"[EEGNet cfg] epochs={int(args.cls_epochs)} | batch_size={int(args.cls_batch_size)} | optimizer=Adam | lr=1e-4")
        sgm_acc_summary["EEGNet"] = run_eegnet(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid_sgm,
            y_test=y_test,
            device=device,
            epochs=int(args.cls_epochs),
            batch_size=int(args.cls_batch_size),
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

