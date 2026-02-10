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
import time
from dataclasses import dataclass
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
) -> None:
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

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
    acc_h = float(accuracy_score(y_test_enc, pred_hybrid))
    bal_h = float(balanced_accuracy_score(y_test_enc, pred_hybrid))

    print("\n=== CSP+LDA Results ===")
    print(f"TEST(real):   acc={acc_real:.3f} | bal_acc={bal_real:.3f}")
    print(f"TEST(hybrid): acc={acc_h:.3f} | bal_acc={bal_h:.3f}")

    cm_real = confusion_matrix(y_test_enc, pred_real, normalize="true")
    cm_h = confusion_matrix(y_test_enc, pred_hybrid, normalize="true")
    print("\nConfusionMatrix (normalize=true) TEST(real):\n", np.array2string(cm_real, precision=3, floatmode="fixed"))
    print("\nConfusionMatrix (normalize=true) TEST(hybrid):\n", np.array2string(cm_h, precision=3, floatmode="fixed"))


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
            nn.Linear(base_ch, n_classes),
        )

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

        return self.head(d1)


def run_unet_classifier(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_real: np.ndarray,
    X_test_hybrid: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> None:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    PAPER_OPTIMIZER = "Adam"
    PAPER_LEARNING_RATE = 1e-3
    PAPER_BATCH_SIZE = 32
    PAPER_EPOCHS = 10

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
        prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
        return {"acc": acc, "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    metrics_real = _eval(Xte_r, yte)
    metrics_hybrid = _eval(Xte_h, yte)

    print("\n=== U-Net Classifier Results (macro) ===")
    print("TEST(real):  ", metrics_real)
    print("TEST(hybrid):", metrics_hybrid)


# --------------------------------------------------------------------------------------
# SGM/ScoreNet: import + wrappers
# --------------------------------------------------------------------------------------


def sgm_train_with_progress(
    *,
    sgm_mod,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    dataloader: DataLoader,
    timesteps: int,
    epochs: int,
    device: torch.device,
    epoch_desc: str,
    print_every: int = 0,
) -> None:
    """Wrapper de entrenamiento SGM con tqdm (evita 200 prints)."""
    model.train()

    epoch_iter = tqdm(range(epochs), total=epochs, desc=epoch_desc, leave=True, mininterval=0.5)
    for epoch in epoch_iter:
        epoch_loss = 0.0
        for _, (batch_inputs, batch_targets) in enumerate(dataloader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_size = batch_inputs.size(0)

            t_idx = torch.randint(0, timesteps, (batch_size,), device=device).long()
            t = (t_idx.float() + 1.0) / float(timesteps)

            z = torch.randn_like(batch_targets)
            sigma = float(getattr(model, "sigma", 25.0))
            std = sgm_mod.marginal_prob_std(t, sigma).view(batch_size, 1, 1)
            x_t = batch_targets + z * std

            model_input = torch.cat([batch_inputs, x_t], dim=1)

            optimizer.zero_grad()
            score = model(model_input, t)
            loss = torch.mean((score * std + z) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(dataloader))
        epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")

        is_first = epoch == 0
        is_last = epoch == epochs - 1
        should_print = is_first or is_last or (print_every > 0 and (epoch + 1) % int(print_every) == 0)
        if should_print:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--ddpm-epochs", type=int, default=200)
    parser.add_argument("--sgm-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-eval-epochs", type=int, default=0, help="0 => eval todos; si >0 limita n")
    parser.add_argument("--gen-batch-size", type=int, default=8)

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

    _log_section("Load + preprocess (paper-like) + epoching (raw01+raw02+raw03, 70/30)")

    # ---- load all 3 training sessions (Subject 1)
    TRAIN_MAT_PATHS = [
        "dataset/BCI_3/train_subject1_raw01.mat",
        "dataset/BCI_3/train_subject1_raw02.mat",
        "dataset/BCI_3/train_subject1_raw03.mat",
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

    mu = x_clean.mean(axis=0, keepdims=True)
    sigma = x_clean.std(axis=0, keepdims=True) + 1e-6
    x_z = (x_clean - mu) / sigma

    # ---- epoching
    EPOCH_SAMPLES = 512
    STRIDE_SAMPLES = EPOCH_SAMPLES // 2  # 0.5s at 512 Hz

    allowed_labels = set(int(v) for v in np.unique(y_raw).tolist())
    max_epochs_per_class = None
    train_ratio = 0.7

    epochs_parts: list[np.ndarray] = []
    labels_parts_out: list[np.ndarray] = []
    offset = 0
    for n_sess in len_parts:
        x_sess = x_z[offset : offset + n_sess]
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

    train_idx, test_idx = stratified_split(labels_all, train_ratio=train_ratio, seed=args.seed)
    epochs_train = epochs_all[train_idx]
    labels_train = labels_all[train_idx]
    epochs_test = epochs_all[test_idx]
    labels_test = labels_all[test_idx]

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

    # ----------------------------------------------------------------------------------
    # DDPM pipeline
    # ----------------------------------------------------------------------------------
    trained_models: dict[str, nn.Module] = {}
    trained_input_idxs: dict[str, tuple[int, int]] = {}
    ddpm_rows: list[dict[str, object]] = []

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
                    # MSE_centered (por epoch): quita offset (media) antes de comparar
                    real_c = real_1d - float(np.mean(real_1d))
                    syn_c = syn_1d - float(np.mean(syn_1d))
                    mse_list.append(float(np.mean((syn_c - real_c) ** 2)))
                    corr_list.append(pearson_corr_1d(real_1d, syn_1d))

                eval_seen += int(target_np.shape[0])
                eval_pbar.update(int(target_np.shape[0]))

            eval_pbar.close()

            mse_centered = float(np.mean(mse_list)) if mse_list else float("nan")
            corr = float(np.mean(corr_list)) if corr_list else float("nan")

            print(f"DDPM Computed  MSE_centered={mse_centered:.6f} Pearson={corr:.6f} (n={eval_seen})")
            print(f"DDPM Reported  MSE={spec.mse_reported:.6f} Pearson={spec.pearson_reported:.6f}")

            ddpm_rows.append(
                {
                    "Target Channel": spec.target,
                    "Input Channels": f"{spec.inputs[0]}, {spec.inputs[1]}",
                    "MSE_centered (computed)": mse_centered,
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

        run_csp_lda(X_train=X_train, y_train=y_train, X_test_real=X_test_real, X_test_hybrid=X_test_hybrid, y_test=y_test)
        run_unet_classifier(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid,
            y_test=y_test,
            device=device,
        )

    # ----------------------------------------------------------------------------------
    # SGM/ScoreNet pipeline
    # ----------------------------------------------------------------------------------
    if not args.skip_sgm:
        _log_section("SGM/ScoreNet: train per target channel + Table (MSE/Pearson) + hybrid + CSP+LDA")

        import SGDM as sgm_mod

        # ensure SGDM uses the same device selection
        sgm_mod.device = device

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

            model = sgm_mod.UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256, sigma=25.0).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

            print("\n------------------------------------")
            print(f"SGM target={spec.target} inputs={spec.inputs}")
            print("train epochs:", len(train_ds), "test epochs:", len(test_ds))

            sgm_train_with_progress(
                sgm_mod=sgm_mod,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=train_loader,
                timesteps=TIMESTEPS,
                epochs=int(args.sgm_epochs),
                device=device,
                epoch_desc=f"SGM Train {spec.target}",
            )

            trained_models_sgm[spec.target] = model.eval()
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

                with torch.no_grad():
                    syn = sgm_mod.sample_from_model(
                        model,
                        conditioning,
                        alphas,
                        alphas_cumprod,
                        betas,
                        sqrt_one_minus_alphas_cumprod,
                        shape=target.shape,
                        sigma=float(getattr(model, "sigma", 25.0)),
                    )

                target_np = target.detach().cpu().numpy()[:, 0]
                syn_np = syn.detach().cpu().numpy()[:, 0]

                for i in range(target_np.shape[0]):
                    real_1d = target_np[i]
                    syn_1d = syn_np[i]
                    # MSE_centered (por epoch): quita offset (media) antes de comparar
                    real_c = real_1d - float(np.mean(real_1d))
                    syn_c = syn_1d - float(np.mean(syn_1d))
                    mse_list.append(float(np.mean((syn_c - real_c) ** 2)))
                    corr_list.append(pearson_corr_1d(real_1d, syn_1d))

                eval_seen += int(target_np.shape[0])
                eval_pbar.update(int(target_np.shape[0]))

            eval_pbar.close()

            mse_centered = float(np.mean(mse_list)) if mse_list else float("nan")
            corr = float(np.mean(corr_list)) if corr_list else float("nan")
            print(f"SGM Computed  MSE_centered={mse_centered:.6f} Pearson={corr:.6f} (n={eval_seen})")

            sgm_rows.append(
                {
                    "Target Channel": spec.target,
                    "Input Channels": f"{spec.inputs[0]}, {spec.inputs[1]}",
                    "MSE_centered (computed)": mse_centered,
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
                    with torch.no_grad():
                        syn = sgm_mod.sample_from_model(
                            model_s,
                            cond_t,
                            alphas,
                            alphas_cumprod,
                            betas,
                            sqrt_one_minus_alphas_cumprod,
                            shape=(cond_t.size(0), 1, t_len),
                            sigma=float(getattr(model_s, "sigma", 25.0)),
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

        run_csp_lda(X_train=X_train, y_train=y_train, X_test_real=X_test_real, X_test_hybrid=X_test_hybrid_sgm, y_test=y_test)

    _log_section("Done")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.1f}s")

