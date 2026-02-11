"""
Pipeline2: BCI Competition IV 2a (GDF) — DDPM + SGM (score-based) channel reconstruction.

Goal: mirror the structure of `pipeline.py` (BCI-III) but using BCICIV 2a:
- Load AxxT.gdf (train) + AxxE.gdf (eval)
- Preprocess "paper-like" similar to BCI3 pipeline:
  bandpass 8–30 Hz + annotate_muscle_zscore + ICA ocular correction + z-score using TRAIN stats only
- Epoching differs: 0–4s post-event (trial-based), per SBGM_EEG.ipynb
- Train DDPM per target channel and evaluate MSE/Pearson
- Train SGM per target channel and evaluate MSE/Pearson
- Build TEST(hybrid) replacing chosen target channels with synthetic ones
- Run CSP+LDA, FBCSP+LDA, U-Net classifier, EEGNet (reusing implementations from `pipeline.py`)
- Save checkpoints to a single file: `checkpoints/BCI4_subject{Axx}.pth`
  and auto-resume: if a target model exists, skip training and synthesize directly.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader

import mne
from mne.preprocessing import ICA

# Reuse models/samplers/classifiers from BCI3 pipeline
import pipeline as p


@dataclass(frozen=True)
class ExperimentSpec2A:
    target: str
    inputs: tuple[str, str]

BCI2A_EEG_22 = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]

BCI2A_EOG_3 = ["EOG-left", "EOG-central", "EOG-right"]

BCI2A_MOTOR8 = ["FC1", "FC2", "C3", "Cz", "C4", "CP1", "CP2", "Pz"]

def _prepare_classifier_epochs_2a(
    X: np.ndarray,
    *,
    ch_names: list[str],
    sfreq: float,
    channel_set: str,
    tmin: float,
    tmax: float,
    resample_sfreq: float,
) -> tuple[np.ndarray, list[str], float]:
    """
    Prepare epochs for classifiers (CSP/FBCSP/EEGNet/U-Net) following typical 2a practice:
    - select a channel subset (motor8 or all22)
    - crop a MI-relevant time window (e.g., 0.5–2.5s)
    - optionally resample to 128 Hz
    """
    Xp = np.asarray(X, dtype=np.float32)
    if not np.isfinite(Xp).all():
        print("[WARN] Non-finite values in classifier input; applying nan_to_num.")
        Xp = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)

    # channel selection
    if str(channel_set) == "motor8":
        want = [c for c in BCI2A_MOTOR8 if c in ch_names]
        if len(want) != len(BCI2A_MOTOR8):
            raise ValueError(f"Missing motor8 channels. want={BCI2A_MOTOR8} have={ch_names}")
        idx = [int(ch_names.index(c)) for c in want]
        Xp = Xp[:, idx, :]
        ch_out = want
    else:
        ch_out = list(ch_names)

    # crop time window (relative to cue at t=0)
    tmin = float(tmin)
    tmax = float(tmax)
    if tmax <= tmin:
        raise ValueError("clf_tmax must be > clf_tmin")
    # If user asks for the full 0..T window, keep the original length (avoid off-by-one).
    n_times = int(Xp.shape[2])
    dur_s = float(max(0, n_times - 1)) / float(sfreq)  # MNE epochs often include both endpoints
    if tmin <= 0.0 and tmax >= dur_s - 1e-6:
        pass
    else:
        start = int(round(tmin * float(sfreq)))
        end = int(round(tmax * float(sfreq)))
        start = max(0, min(start, n_times - 1))
        end = max(start + 1, min(end, n_times))
        Xp = Xp[:, :, start:end]

    # resample (optional)
    resample_sfreq = float(resample_sfreq)
    sfreq_out = float(sfreq)
    if resample_sfreq > 0 and abs(resample_sfreq - float(sfreq)) > 1e-6:
        from mne.filter import resample as mne_resample

        up = int(round(resample_sfreq))
        down = int(round(float(sfreq)))
        Xp = mne_resample(Xp, up=up, down=down, axis=2, verbose=False).astype(np.float32)
        sfreq_out = float(resample_sfreq)

    return Xp, ch_out, sfreq_out

def run_csp_lda_robust(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_real: np.ndarray,
    X_test_hybrid: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
) -> dict[str, dict[str, float]]:
    """CSP+LDA matching EEGNet_2a.ipynb (bandpass 8–30) with safe fallbacks."""
    from mne.decoding import CSP
    from mne.filter import filter_data
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, cohen_kappa_score

    # Try to match EEGNet_2a.ipynb first (reg=None, cov_est="epoch"), then fall back if unstable.
    CSP_CFG_PRIMARY = dict(n_components=4, reg=None, log=True, norm_trace=True, cov_est="epoch")
    CSP_CFG_FALLBACK = dict(n_components=4, reg="ledoit_wolf", log=True, norm_trace=True, cov_est="concat")
    LDA_CFG = dict(solver="lsqr", shrinkage="auto")

    classes = p._label_classes(y_train, y_test)
    y_train_enc = p._encode_labels_with_classes(y_train, classes)
    y_test_enc = p._encode_labels_with_classes(y_test, classes)

    print("classes (original labels):", classes.tolist())

    # Match notebook: same preproc (bandpass 8–30) for TRAIN/TEST before CSP
    X_train_bp = filter_data(np.asarray(X_train, np.float64), float(sfreq), 8.0, 30.0, method="iir", verbose=False)
    X_test_real_bp = filter_data(
        np.asarray(X_test_real, np.float64), float(sfreq), 8.0, 30.0, method="iir", verbose=False
    )
    X_test_hybrid_bp = filter_data(
        np.asarray(X_test_hybrid, np.float64), float(sfreq), 8.0, 30.0, method="iir", verbose=False
    )

    def _fit_transform_with_cfg(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        csp = CSP(**cfg)
        F_tr = csp.fit_transform(X_train_bp, y_train_enc)
        F_r = csp.transform(X_test_real_bp)
        F_h = csp.transform(X_test_hybrid_bp)
        return F_tr, F_r, F_h

    try:
        F_train, F_test_real, F_test_hybrid = _fit_transform_with_cfg(CSP_CFG_PRIMARY)
    except Exception as e:
        print(f"[WARN] CSP primary config failed ({e}); using fallback regularization.")
        F_train, F_test_real, F_test_hybrid = _fit_transform_with_cfg(CSP_CFG_FALLBACK)

    if not np.isfinite(F_train).all() or not np.isfinite(F_test_real).all() or not np.isfinite(F_test_hybrid).all():
        # If primary produced non-finites, retry with fallback.
        try:
            F_train, F_test_real, F_test_hybrid = _fit_transform_with_cfg(CSP_CFG_FALLBACK)
        except Exception:
            print("[WARN] Non-finite CSP features; applying nan_to_num.")
            F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
            F_test_real = np.nan_to_num(F_test_real, nan=0.0, posinf=0.0, neginf=0.0)
            F_test_hybrid = np.nan_to_num(F_test_hybrid, nan=0.0, posinf=0.0, neginf=0.0)

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

    print("\n=== CSP+LDA Results ===")
    print(f"TEST(real):   acc={acc_real:.3f} | bal_acc={bal_real:.3f} | kappa={kappa_real:.3f}")
    print(f"TEST(hybrid): acc={acc_h:.3f} | bal_acc={bal_h:.3f} | kappa={kappa_h:.3f}")

    cm_real = confusion_matrix(y_test_enc, pred_real, normalize="true")
    cm_h = confusion_matrix(y_test_enc, pred_h, normalize="true")
    print("\nConfusionMatrix (normalize=true) TEST(real):\n", np.array2string(cm_real, precision=3, floatmode="fixed"))
    print("\nConfusionMatrix (normalize=true) TEST(hybrid):\n", np.array2string(cm_h, precision=3, floatmode="fixed"))

    # CV within-test (features)
    def _cv_on_features(feats: np.ndarray, y_enc: np.ndarray, *, title: str) -> dict[str, float]:
        from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score

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

        return {"cv_acc": float(scores_cv.mean()), "cv_acc_std": float(scores_cv.std())}

    cv_real = _cv_on_features(F_test_real, y_test_enc, title="TEST(real)")
    cv_h = _cv_on_features(F_test_hybrid, y_test_enc, title="TEST(hybrid)")

    return {
        "no_cv": {"acc_real": acc_real, "acc_hybrid": acc_h, "kappa_real": kappa_real, "kappa_hybrid": kappa_h},
        "cv": {
            "acc_real": float(cv_real["cv_acc"]),
            "acc_real_std": float(cv_real["cv_acc_std"]),
            "acc_hybrid": float(cv_h["cv_acc"]),
            "acc_hybrid_std": float(cv_h["cv_acc_std"]),
        },
    }


def _crop_last_if_odd_time(X: np.ndarray) -> np.ndarray:
    """U-Net/EEGNet in pipeline.py are sensitive to odd time lengths (skip-connection cat mismatch)."""
    X = np.asarray(X)
    if X.ndim != 3:
        return X
    t = int(X.shape[2])
    if t % 2 == 1:
        return X[:, :, : t - 1]
    return X


def run_fbcsp_lda_safe(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_real: np.ndarray,
    X_test_hybrid: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
) -> dict[str, dict[str, float]]:
    """
    Try p.run_fbcsp_lda (matching EEGNet_2a.ipynb style: reg=None).
    If it fails due to NaNs/Infs, retry with CSP regularization.
    """
    try:
        return p.run_fbcsp_lda(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_test_real,
            X_test_hybrid=X_test_hybrid,
            y_test=y_test,
            sfreq=sfreq,
        )
    except Exception as e:
        print(f"[WARN] FBCSP+LDA primary failed ({e}). Retrying with CSP regularization.")

    from mne.decoding import CSP
    from mne.filter import filter_data
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
    from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score

    seed = 42

    X_train = np.asarray(X_train, dtype=np.float64)
    X_test_real = np.asarray(X_test_real, dtype=np.float64)
    X_test_hybrid = np.asarray(X_test_hybrid, dtype=np.float64)

    bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30)]

    classes = p._label_classes(y_train, y_test)
    y_train_enc = p._encode_labels_with_classes(y_train, classes)
    y_test_enc = p._encode_labels_with_classes(y_test, classes)

    m_pairs = 2 if int(X_train.shape[1]) > 3 else 1
    per_band = int(2 * m_pairs)

    CSP_CFG = dict(n_components=per_band, reg="ledoit_wolf", log=True, norm_trace=True, cov_est="concat")
    LDA_CFG = dict(solver="lsqr", shrinkage="auto")

    F_train_list: list[np.ndarray] = []
    F_test_real_list: list[np.ndarray] = []
    F_test_hybrid_list: list[np.ndarray] = []

    for (l_f, h_f) in bands:
        Xtr_b = filter_data(X_train, float(sfreq), l_f, h_f, method="iir", verbose=False)
        Xte_r_b = filter_data(X_test_real, float(sfreq), l_f, h_f, method="iir", verbose=False)
        Xte_h_b = filter_data(X_test_hybrid, float(sfreq), l_f, h_f, method="iir", verbose=False)

        csp_b = CSP(**CSP_CFG)
        F_train_list.append(csp_b.fit_transform(Xtr_b, y_train_enc))
        F_test_real_list.append(csp_b.transform(Xte_r_b))
        F_test_hybrid_list.append(csp_b.transform(Xte_h_b))

    F_train_full = np.concatenate(F_train_list, axis=1)
    F_test_real_full = np.concatenate(F_test_real_list, axis=1)
    F_test_hybrid_full = np.concatenate(F_test_hybrid_list, axis=1)

    if not np.isfinite(F_train_full).all() or not np.isfinite(F_test_real_full).all() or not np.isfinite(F_test_hybrid_full).all():
        print("[WARN] Non-finite FBCSP features after fallback; applying nan_to_num.")
        F_train_full = np.nan_to_num(F_train_full, nan=0.0, posinf=0.0, neginf=0.0)
        F_test_real_full = np.nan_to_num(F_test_real_full, nan=0.0, posinf=0.0, neginf=0.0)
        F_test_hybrid_full = np.nan_to_num(F_test_hybrid_full, nan=0.0, posinf=0.0, neginf=0.0)

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
        return {"cv_acc": float(scores_cv.mean()), "cv_acc_std": float(scores_cv.std())}

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

def build_hybrid_epochs_ddpm_2a(
    epochs_real: np.ndarray,
    *,
    specs: list[ExperimentSpec2A],
    trained_models: dict[str, torch.nn.Module],
    trained_input_idxs: dict[str, tuple[int, int]],
    ch_to_idx: dict[str, int],
    gen_batch_size: int,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    betas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Reemplaza canales target por reconstrucción DDPM (z-score), usando el mapping de canales de 2a."""
    epochs_h = epochs_real.copy()
    n = int(epochs_h.shape[0])
    t_len = int(epochs_h.shape[2])

    for spec in specs:
        target = spec.target
        if target not in trained_models:
            print(f"[WARN] No model found for {target}; skipping.")
            continue
        if target not in ch_to_idx:
            raise KeyError(f"Target channel not found in ch_to_idx: {target}")

        model = trained_models[target]
        input_idxs = trained_input_idxs[target]
        target_idx = int(ch_to_idx[target])

        pbar = p.tqdm(total=n, desc=f"DDPM Synthesize {target}", leave=False, mininterval=0.5)
        for s in range(0, n, int(gen_batch_size)):
            e = min(n, s + int(gen_batch_size))
            batch = epochs_h[s:e]  # (B, C, T)

            cond = np.stack([batch[:, input_idxs[0], :], batch[:, input_idxs[1], :]], axis=1).astype(np.float32)  # (B,2,T)
            cond_t = torch.from_numpy(cond).to(device)

            with torch.no_grad():
                syn = p.sample_from_model(
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


def _pick_train_class_keys_2a(event_id: dict[str, int]) -> dict[str, str]:
    # BCI-IV 2a typical: 4 classes (769–772)
    if all(k in event_id for k in ("769", "770", "771", "772")):
        return {
            "left_hand": "769",
            "right_hand": "770",
            "feet": "771",
            "tongue": "772",
        }

    # fallback: names like class 1..4
    for prefix in ("class ", "Class "):
        keys = [f"{prefix}{i}" for i in (1, 2, 3, 4)]
        if all(k in event_id for k in keys):
            return {
                "left_hand": keys[0],
                "right_hand": keys[1],
                "feet": keys[2],
                "tongue": keys[3],
            }

    raise ValueError(
        "No encontré eventos de 4 clases (769–772 / class 1–4) en TRAIN. "
        f"Keys disponibles: {sorted(event_id.keys())}"
    )


def _pick_eval_cue_key_2a(event_id: dict[str, int]) -> str:
    # In 2a, cue often 783 (or 768 depending on parsing)
    for k in ("783", "768"):
        if k in event_id:
            return k
    # fallback
    for k in ("781",):
        if k in event_id:
            print("[WARN] No encontré 783/768 en EVAL; usando 781 como cue event.")
            return k
    raise ValueError(
        "No encontré el evento cue (783/768) en EVAL. "
        f"Keys disponibles: {sorted(event_id.keys())}"
    )


def _infer_eog_channels(raw: mne.io.BaseRaw) -> list[str]:
    # Prefer explicit common names from BCI-IV exports, but also support generic "EOG" substring.
    candidates = [
        *BCI2A_EOG_3,
        "EOG-left",
        "EOG-central",
        "EOG-right",
        "EOG:ch01",
        "EOG:ch02",
        "EOG:ch03",
        "EOG 1",
        "EOG 2",
        "EOG 3",
    ]
    eog = [c for c in candidates if c in raw.ch_names]
    if eog:
        return eog
    return [ch for ch in raw.ch_names if "EOG" in ch.upper()]


def _drop_eog_if_present(raw: mne.io.BaseRaw) -> None:
    eog = _infer_eog_channels(raw)
    if eog:
        raw.drop_channels(eog)

def _maybe_assign_standard_2a_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Some GDF exports come with generic/duplicated channel names (e.g., many channels called 'EEG').
    For BCI-IV 2a we want the canonical 22 EEG (+3 EOG) names so that:
    - montage positions resolve (standard_1020)
    - auto target/input selection works
    - EOG channels can be detected/dropped reliably
    """
    raw = raw.copy()
    ch_names = list(raw.ch_names)
    n_ch = len(ch_names)

    canonical_25 = BCI2A_EEG_22 + BCI2A_EOG_3
    canonical_22 = BCI2A_EEG_22

    def _looks_canonical(names: list[str]) -> bool:
        known = set(canonical_25)
        return sum(1 for c in names if c in known) >= 10

    # If already mostly canonical, do nothing
    if _looks_canonical(ch_names):
        return raw

    # If channel count matches expected 2a layouts, rename by index
    if n_ch == 25:
        mapping = {ch_names[i]: canonical_25[i] for i in range(25)}
        raw.rename_channels(mapping)
        raw.set_channel_types({c: "eog" for c in BCI2A_EOG_3 if c in raw.ch_names})
        return raw

    if n_ch == 22:
        mapping = {ch_names[i]: canonical_22[i] for i in range(22)}
        raw.rename_channels(mapping)
        return raw

    # Otherwise: keep as-is; MNE already ensured uniqueness (may append running numbers)
    print(f"[WARN] Unexpected n_channels={n_ch}; keeping original channel names.")
    return raw


def _preprocess_like_bci3(
    raw: mne.io.BaseRaw,
    *,
    export_fif_path: Path | None = None,
    export_edf_path: Path | None = None,
    notch_freq: float = 50.0,
) -> mne.io.BaseRaw:
    """
    Preprocess similar to BCI3 pipeline, but also considers SBGM_EEG.ipynb eval-cleaning:
    - (optional export) drop EOG + bandpass 8-30 + notch(50) + save FIF (+ try EDF)
    - then: bandpass 8–30 + notch(50) + annotate_muscle_zscore + ICA EOG correction
    - set standard montage
    - drop EOG channels (return only EEG like BCI3 arrays)
    """
    raw = raw.copy()

    # Optional: export a "cleaned" file like SBGM_EEG.ipynb (eval cleaning)
    if export_fif_path is not None:
        raw_export = raw.copy()
        _drop_eog_if_present(raw_export)
        raw_export.filter(l_freq=8.0, h_freq=30.0, method="iir", verbose=False)
        try:
            raw_export.notch_filter(freqs=float(notch_freq), verbose=False)
        except Exception as e:
            print("[WARN] notch_filter skipped:", repr(e))
        export_fif_path.parent.mkdir(parents=True, exist_ok=True)
        raw_export.save(str(export_fif_path), overwrite=True, verbose=False)
        print(f"[CLEAN] saved FIF: {export_fif_path}")
        if export_edf_path is not None:
            try:
                from mne.export import export_raw

                export_edf_path.parent.mkdir(parents=True, exist_ok=True)
                export_raw(str(export_edf_path), raw_export, fmt="edf", overwrite=True)
                print(f"[CLEAN] saved EDF: {export_edf_path}")
            except Exception as e:
                print("[WARN] could not export EDF (install pyedflib). Error:", repr(e))

    # bandpass 8-30Hz (IIR)
    raw.filter(l_freq=8.0, h_freq=30.0, method="iir", verbose=False)
    # notch (line noise) like SBGM_EEG.ipynb
    try:
        raw.notch_filter(freqs=float(notch_freq), verbose=False)
    except Exception as e:
        print("[WARN] notch_filter skipped:", repr(e))

    # muscle annotations (same parameters as BCI3 pipeline)
    try:
        # IMPORTANT: preserve original event annotations (e.g., 769–772) from the GDF.
        # We append muscle annotations instead of overwriting `raw.annotations`.
        orig_annotations = raw.annotations.copy()

        sfreq = float(raw.info.get("sfreq", 0.0) or 0.0)
        nyq = 0.5 * sfreq if sfreq > 0 else 0.0
        # BCI-IV 2a is typically 250 Hz => Nyquist 125; keep h_freq safely below Nyquist.
        h_freq = 140.0
        if nyq > 0:
            h_freq = min(h_freq, max(20.0, nyq * 0.98))
        if h_freq <= 20.0:
            raise ValueError(f"sfreq too low for muscle band (sfreq={sfreq}, nyq={nyq})")

        annotations, _ = mne.preprocessing.annotate_muscle_zscore(
            raw,
            ch_type="eeg",
            threshold=4.0,
            min_length_good=0.1,
            filter_freq=(20, float(h_freq)),
            verbose=False,
        )
        try:
            raw.set_annotations(orig_annotations + annotations)
        except Exception:
            # Fallback if concat fails for any reason
            raw.set_annotations(orig_annotations)
    except Exception as e:
        print("annotate_muscle_zscore skipped:", e)

    # ICA
    ica = ICA(n_components=20, random_state=97, max_iter=800)
    # Fit only on EEG picks to avoid EOG channels dominating
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude=[])
    ica.fit(raw, picks=picks_eeg, verbose=False)

    eog_chs = _infer_eog_channels(raw)
    if eog_chs:
        try:
            eog_idx, _ = ica.find_bads_eog(raw, ch_name=eog_chs, verbose=False)
            ica.exclude = eog_idx
            print("ICA exclude components (EOG):", list(map(int, eog_idx)))
        except Exception as e:
            print("ICA EOG detection skipped:", e)
    else:
        print("[WARN] No EOG channels found; ICA will not exclude ocular comps automatically.")

    raw = ica.apply(raw.copy(), verbose=False)

    # standard montage for consistency (ignore missing)
    try:
        raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
    except Exception:
        pass

    # Drop EOG channels after correction (to keep only EEG like BCI3 arrays)
    if eog_chs:
        raw = raw.copy().drop_channels(eog_chs)

    # Keep EEG only (2a = 22 EEG). This also avoids any stray non-EEG channels.
    try:
        picks_keep = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, misc=False, exclude=[])
        raw = raw.copy().pick(picks_keep)
    except Exception:
        pass

    return raw


def _load_eval_labels_mat(*, labels_dir: Path, subject_id: str) -> np.ndarray:
    mat_path = labels_dir / f"{subject_id}E.mat"
    if not mat_path.exists():
        raise FileNotFoundError(
            f"No encontré {mat_path}. Coloca `{subject_id}E.mat` dentro de `{labels_dir}/` "
            f"(ej: `true_labels/`) o ejecuta con `--labels-dir true_labels`."
        )
    mat = scipy.io.loadmat(str(mat_path))
    y_raw = np.asarray(mat["classlabel"]).squeeze().astype(int)
    if not set(np.unique(y_raw)).issubset({1, 2, 3, 4}):
        raise ValueError(f"Valores inesperados en classlabel: {np.unique(y_raw)}")
    return (y_raw - 1).astype(int)  # 0..3


def _default_experiments_2a() -> list[ExperimentSpec2A]:
    """
    Default 8 target channels (2a montage) with two "adjacent" inputs.
    This mimics the BCI3 setup: reconstruct a subset from nearby channels.
    """
    return [
        # IMPORTANT: do NOT synthesize C3/C4/Cz (motor-critical). Use them as inputs only.
        ExperimentSpec2A("FC3", ("FC1", "C3")),
        ExperimentSpec2A("FC4", ("FC2", "C4")),
        ExperimentSpec2A("CP3", ("CP1", "C3")),
        ExperimentSpec2A("CP4", ("CP2", "C4")),
        ExperimentSpec2A("P1", ("CP1", "Pz")),
        ExperimentSpec2A("P2", ("CP2", "Pz")),
        ExperimentSpec2A("Pz", ("P1", "P2")),
        ExperimentSpec2A("POz", ("Pz", "CPz")),
    ]


def _auto_experiments_2a(
    *,
    ch_names: list[str],
    epochs: mne.Epochs,
    n_targets: int,
) -> list[ExperimentSpec2A]:
    """
    Auto-choose "low importance / far from motor cortex" targets, and pick two nearest motor-relevant
    channels as inputs (using standard_1020 positions).
    """
    n_targets = int(n_targets)
    if n_targets <= 0:
        raise ValueError("n_targets must be > 0")

    # Motor-relevant pool (2a typical)
    motor_pool = [
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
    ]
    motor_pool = [c for c in motor_pool if c in ch_names]
    # Never synthesize these (explicit requirement)
    forbidden_targets = {"C3", "C4", "Cz"}

    # Get channel positions
    pos = {}
    try:
        montage = epochs.get_montage()
        if montage is not None:
            pos = montage.get_positions().get("ch_pos", {}) or {}
    except Exception:
        pos = {}

    def _has_pos(c: str) -> bool:
        return c in pos and np.all(np.isfinite(np.asarray(pos[c], dtype=float)))

    ch_with_pos = [c for c in ch_names if _has_pos(c)]
    motor_with_pos = [c for c in motor_pool if _has_pos(c)]
    if len(motor_with_pos) < 2 or len(ch_with_pos) < 4:
        # Fallback: keep the hardcoded mapping
        print("[WARN] Could not infer channel positions reliably. Falling back to default experiments.")
        return _default_experiments_2a()

    # Motor center = mean of (C3,Cz,C4) if available else mean of motor pool
    motor_center_keys = [c for c in ("C3", "Cz", "C4") if c in motor_with_pos]
    if not motor_center_keys:
        motor_center_keys = motor_with_pos
    motor_center = np.mean([np.asarray(pos[c], dtype=float) for c in motor_center_keys], axis=0)

    # Candidates = channels not in motor pool
    candidates = [c for c in ch_with_pos if c not in set(motor_with_pos) and c not in forbidden_targets]
    if not candidates:
        print("[WARN] No non-motor candidates found. Falling back to default experiments.")
        return _default_experiments_2a()

    # Sort by distance from motor center desc => "farthest"
    cand_sorted = sorted(
        candidates,
        key=lambda c: float(np.linalg.norm(np.asarray(pos[c], dtype=float) - motor_center)),
        reverse=True,
    )
    targets = cand_sorted[: min(n_targets, len(cand_sorted))]

    specs: list[ExperimentSpec2A] = []
    for tgt in targets:
        tgt_pos = np.asarray(pos[tgt], dtype=float)
        # Inputs: 2 nearest channels among motor pool
        neighbors = sorted(
            motor_with_pos,
            key=lambda c: float(np.linalg.norm(np.asarray(pos[c], dtype=float) - tgt_pos)),
        )
        if len(neighbors) < 2:
            # fallback: nearest among all channels excluding target
            neighbors = sorted(
                [c for c in ch_with_pos if c != tgt],
                key=lambda c: float(np.linalg.norm(np.asarray(pos[c], dtype=float) - tgt_pos)),
            )
        specs.append(ExperimentSpec2A(tgt, (neighbors[0], neighbors[1])))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--subject-id", type=str, default="A01", help="BCI-IV 2a subject id, e.g. A01..A09")
    parser.add_argument("--data-dir", type=str, default="BCICIV_2a_gdf", help="Folder containing AxxT.gdf and AxxE.gdf")
    parser.add_argument("--labels-dir", type=str, default="true_labels", help="Folder containing AxxE.mat (classlabel)")
    parser.add_argument("--cleaned-dir", type=str, default="cleaned_data/second_session", help="Where to write cleaned FIF/EDF for EVAL.")
    parser.add_argument("--export-cleaned-eval", action="store_true", help="Export cleaned Evaluation_{Axx}.fif and .edf (if possible).")
    parser.add_argument(
        "--auto-specs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-select far/low-importance target channels and choose 2 motor-relevant inputs.",
    )
    parser.add_argument("--n-targets", type=int, default=8, help="Number of channels to synthesize (only when --auto-specs).")

    parser.add_argument("--ddpm-epochs", type=int, default=200)

    parser.add_argument("--sgm-train-steps", type=int, default=20_000)
    parser.add_argument("--sgm-eval-every", type=int, default=5_000)
    parser.add_argument("--sgm-backbone", type=str, default="scorenet-unet", choices=["scorenet-unet", "ddpm-unet"])
    parser.add_argument("--sgm-sampler", type=str, default="pc", choices=["pc", "ode"])
    parser.add_argument("--sgm-ode-method", type=str, default="rk4", choices=["euler", "rk4"])
    parser.add_argument("--sgm-sampling-n", type=int, default=1000)
    parser.add_argument("--sgm-sampling-eps", type=float, default=1e-4)
    parser.add_argument("--sgm-snr", type=float, default=0.10)
    parser.add_argument("--sgm-n-steps-each", type=int, default=2)
    parser.add_argument("--sgm-init-corrector-steps", type=int, default=0)
    parser.add_argument("--sgm-score-clip", type=float, default=0.0)
    parser.add_argument("--sgm-ve-sigma", type=float, default=25.0)
    parser.add_argument("--sgm-vp-beta-max", type=float, default=20.0)
    parser.add_argument(
        "--sgm-likelihood-weighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Likelihood weighting g(t)^2 for SGM loss.",
    )
    parser.add_argument("--sgm-noise-removal", action="store_true")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--n-eval-epochs", type=int, default=0, help="0 => eval todos; si >0 limita n")

    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Folder to save BCI4_subjectXX.pth")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-train", action="store_true")

    parser.add_argument("--cls-epochs", type=int, default=500)
    parser.add_argument("--cls-batch-size", type=int, default=64)
    parser.add_argument(
        "--clf-channels",
        type=str,
        default="all22",
        choices=["motor8", "all22"],
        help="Channel set used for classifiers (NOT for synthesis/MSE). motor8 matches EEGNet_2a.ipynb.",
    )
    # Defaults: keep full 0..4s at native 250 Hz (as requested).
    parser.add_argument("--clf-tmin", type=float, default=0.0, help="Classifier time window start (s) post-cue.")
    parser.add_argument("--clf-tmax", type=float, default=4.0, help="Classifier time window end (s) post-cue.")
    parser.add_argument("--clf-resample", type=float, default=0.0, help="Classifier resample Hz. 0 disables.")

    parser.add_argument("--skip-ddpm", action="store_true")
    parser.add_argument("--skip-sgm", action="store_true")
    args = parser.parse_args()

    p._set_seeds(int(args.seed))

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subject_id = str(args.subject_id).strip()
    data_dir = Path(str(args.data_dir)).expanduser().resolve()
    labels_dir = Path(str(args.labels_dir)).expanduser().resolve()

    train_path = data_dir / f"{subject_id}T.gdf"
    eval_path = data_dir / f"{subject_id}E.gdf"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval file: {eval_path}")

    ckpt_dir = Path(str(args.ckpt_dir)).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"BCI4_subject{subject_id}.pth"

    # Load existing checkpoint bundle if present
    bundle: dict[str, object] = {"ddpm": {}, "sgm": {}, "meta": {}}
    if bool(args.cache) and (not bool(args.force_train)) and ckpt_path.exists():
        try:
            bundle = torch.load(ckpt_path, map_location="cpu")
            print(f"[CKPT] loaded: {ckpt_path}")
        except Exception as e:
            print(f"[CKPT] load failed: {e}. Starting from scratch.")
            bundle = {"ddpm": {}, "sgm": {}, "meta": {}}

    def _save_bundle() -> None:
        if not bool(args.cache):
            return
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(bundle, tmp)
        tmp.replace(ckpt_path)
        print(f"[CKPT] saved: {ckpt_path}")

    p._log_section(f"BCI-IV 2a load + preprocess (Subject {subject_id})")

    raw_train = mne.io.read_raw_gdf(str(train_path), preload=True, verbose=False)
    raw_eval = mne.io.read_raw_gdf(str(eval_path), preload=True, verbose=False)

    # Ensure we have canonical 2a channel names (helps montage + auto target selection)
    raw_train = _maybe_assign_standard_2a_channel_names(raw_train)
    raw_eval = _maybe_assign_standard_2a_channel_names(raw_eval)

    raw_train = _preprocess_like_bci3(raw_train)
    cleaned_dir = Path(str(args.cleaned_dir)).expanduser().resolve()
    eval_fif = cleaned_dir / f"Evaluation_{subject_id}.fif"
    eval_edf = cleaned_dir / f"Evaluation_{subject_id}.edf"
    raw_eval = _preprocess_like_bci3(
        raw_eval,
        export_fif_path=eval_fif if bool(args.export_cleaned_eval) else None,
        export_edf_path=eval_edf if bool(args.export_cleaned_eval) else None,
        notch_freq=50.0,
    )

    # Sanity: 2a should end up with 22 EEG channels (after dropping 3 EOG)
    print(f"[SANITY] TRAIN channels after preprocessing: n={len(raw_train.ch_names)} | {raw_train.ch_names}")
    print(f"[SANITY] EVAL  channels after preprocessing: n={len(raw_eval.ch_names)} | {raw_eval.ch_names}")

    events_train, event_id_train = mne.events_from_annotations(raw_train, verbose=False)
    events_eval, event_id_eval = mne.events_from_annotations(raw_eval, verbose=False)

    class_keys = _pick_train_class_keys_2a(event_id_train)
    mi_event_id_train = {name: int(event_id_train[k]) for name, k in class_keys.items()}
    # Load eval labels early so we can choose the correct eval event key (783 vs 768) by count.
    y_eval_full = _load_eval_labels_mat(labels_dir=labels_dir, subject_id=subject_id)

    # Choose eval cue key whose occurrence count matches labels length when possible.
    candidate_keys = [k for k in ("783", "768", "781") if k in event_id_eval]
    counts = {}
    for k in candidate_keys:
        code = int(event_id_eval[k])
        counts[k] = int((events_eval[:, 2].astype(int) == code).sum())
    eval_cue_key = _pick_eval_cue_key_2a(event_id_eval)
    if counts:
        # prefer exact match with labels length
        exact = [k for k, c in counts.items() if int(c) == int(len(y_eval_full))]
        if exact:
            # prefer 783 if both match; else first exact
            eval_cue_key = "783" if "783" in exact else exact[0]
        print(f"[EVAL events] candidate counts: {counts} | labels={len(y_eval_full)} | chosen={eval_cue_key}")
    eval_cue_code = int(event_id_eval[eval_cue_key])

    print("TRAIN class keys:", class_keys)
    print("EVAL  cue key   :", eval_cue_key)
    print("mi_event_id_train:", mi_event_id_train)

    # Epoching (0..4s) like SBGM_EEG.ipynb
    epochs_train = mne.Epochs(
        raw_train,
        events_train,
        event_id=mi_event_id_train,
        tmin=0.0,
        tmax=4.0,
        reject=None,
        baseline=None,
        preload=True,
        verbose=False,
    )
    # Precompute the cue-event indices in the original `events_eval` array for robust alignment.
    cue_event_indices = np.where(events_eval[:, 2].astype(int) == int(eval_cue_code))[0].astype(int)
    epochs_eval = mne.Epochs(
        raw_eval,
        events_eval,
        event_id={"cue": int(eval_cue_code)},
        tmin=0.0,
        tmax=4.0,
        reject=None,
        baseline=None,
        preload=True,
        verbose=False,
    )

    X_train = epochs_train.get_data(copy=True).astype(np.float32)  # (n_trials, n_ch, n_times)
    y_train_codes = epochs_train.events[:, -1].astype(int)
    code_to_class = {int(mi_event_id_train[name]): i for i, name in enumerate(class_keys.keys())}
    y_train = np.array([code_to_class[int(c)] for c in y_train_codes], dtype=int)

    X_eval = epochs_eval.get_data(copy=True).astype(np.float32)
    # Align eval labels with the epochs that survived annotation-based dropping.
    # If MNE drops epochs due to BAD annotations, we must drop the corresponding labels too.
    y_eval: np.ndarray
    if len(y_eval_full) == len(cue_event_indices) and hasattr(epochs_eval, "selection"):
        sel = np.asarray(getattr(epochs_eval, "selection"), dtype=int)
        # `sel` is usually indices into the original `events_eval`; map them into cue-order positions.
        try:
            idx_to_pos = {int(idx): i for i, idx in enumerate(cue_event_indices.tolist())}
            keep_pos = np.asarray([idx_to_pos[int(idx)] for idx in sel.tolist()], dtype=int)
            y_eval = y_eval_full[keep_pos]
        except Exception:
            # Fallback: some MNE versions store `selection` as indices within the cue-only list.
            if len(sel) == len(epochs_eval) and int(sel.max(initial=0)) < len(y_eval_full):
                y_eval = y_eval_full[sel]
            else:
                min_n = min(len(X_eval), len(y_eval_full))
                print(
                    f"[WARN] EVAL label alignment fallback: epochs={len(X_eval)} labels={len(y_eval_full)}. "
                    f"Using min_n={min_n}"
                )
                X_eval = X_eval[:min_n]
                y_eval = y_eval_full[:min_n]
    else:
        # Fallback when label file doesn't match number of cue events (or selection not available)
        min_n = min(len(X_eval), len(y_eval_full))
        if len(X_eval) != len(y_eval_full):
            print(
                f"[WARN] EVAL mismatch: epochs={len(X_eval)} labels={len(y_eval_full)}. "
                f"Aligning min_n={min_n}"
            )
        X_eval = X_eval[:min_n]
        y_eval = y_eval_full[:min_n]

    # z-score using TRAIN stats only (anti-leakage), same style as BCI3 pipeline
    mu_ch = X_train.mean(axis=(0, 2), keepdims=True)
    sd_ch = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train = (X_train - mu_ch) / sd_ch
    X_eval = (X_eval - mu_ch) / sd_ch

    ch_names = [c.strip() for c in epochs_train.ch_names]
    ch_to_idx = {c: i for i, c in enumerate(ch_names)}

    # Dataset summary (similar to pipeline.py)
    sfreq = float(epochs_train.info["sfreq"])
    uniq, cnt = np.unique(y_train, return_counts=True)
    class_balance = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
    print(f"sfreq: {sfreq}")
    print(f"TRAIN: {tuple(int(v) for v in X_train.shape)} | EVAL: {tuple(int(v) for v in X_eval.shape)}")
    print(f"class balance train (0..3): {class_balance}")
    print(f"n_channels EEG: {len(ch_names)} | channels: {ch_names}")

    if bool(args.auto_specs):
        specs = _auto_experiments_2a(ch_names=ch_names, epochs=epochs_train, n_targets=int(args.n_targets))
    else:
        specs = _default_experiments_2a()

    # Hard constraint: never synthesize motor-critical channels
    forbidden_targets = {"C3", "C4", "Cz"}
    bad = [s.target for s in specs if s.target in forbidden_targets]
    if bad:
        raise ValueError(f"Invalid target(s) selected for synthesis (forbidden): {bad}. Specs={specs}")

    # Print which targets/inputs were selected (like BCI3 traceability)
    p._log_section("BCI-IV 2a — Selected target/input channels for synthesis")
    try:
        import pandas as pd

        df_specs = pd.DataFrame(
            [{"Target Channel": s.target, "Input Channels": f"{s.inputs[0]}, {s.inputs[1]}"} for s in specs]
        )
        print(df_specs.to_string(index=False))
    except Exception:
        for s in specs:
            print(f"- Target={s.target} | Inputs={s.inputs}")

    for s in specs:
        if s.target not in ch_to_idx:
            raise ValueError(f"Target channel not found in data: {s.target}. Available: {ch_names}")
        for inp in s.inputs:
            if inp not in ch_to_idx:
                raise ValueError(f"Input channel not found in data: {inp}. Available: {ch_names}")

    # Precompute DDPM schedule
    TIMESTEPS = 1000
    betas = p.linear_beta_schedule(TIMESTEPS).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    ddpm_models: dict[str, torch.nn.Module] = {}
    ddpm_input_idxs: dict[str, tuple[int, int]] = {}
    ddpm_rows: list[dict[str, object]] = []

    sgm_models: dict[str, torch.nn.Module] = {}
    sgm_input_idxs: dict[str, tuple[int, int]] = {}
    sgm_rows: list[dict[str, object]] = []

    # SGM SDE params (compat with pipeline.py logic)
    sde_train = p.VPSDE(beta_min=0.1, beta_max=float(args.sgm_vp_beta_max), N=1000, T=1.0)
    sde_sampling = p.VPSDE(beta_min=0.1, beta_max=float(args.sgm_vp_beta_max), N=1000, T=1.0)
    ema = p.EMA(beta=0.9999)

    # Load ScoreNet backbone if needed
    sgdm_mod = None
    if args.sgm_backbone == "scorenet-unet":
        import importlib.util

        sgdm_path = Path(__file__).resolve().parent / "SGDM.py"
        if not sgdm_path.exists():
            raise FileNotFoundError(f"Missing SGDM.py next to pipeline2.py: {sgdm_path}")
        spec_mod = importlib.util.spec_from_file_location("SGDM", str(sgdm_path))
        if spec_mod is None or spec_mod.loader is None:
            raise ImportError(f"Could not import SGDM.py from {sgdm_path}")
        sgdm_mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(sgdm_mod)

    # Helper: load saved state_dicts from bundle
    saved_ddpm = bundle.get("ddpm")
    if not isinstance(saved_ddpm, dict):
        saved_ddpm = {}
        bundle["ddpm"] = saved_ddpm

    saved_sgm = bundle.get("sgm")
    if not isinstance(saved_sgm, dict):
        saved_sgm = {}
        bundle["sgm"] = saved_sgm

    meta = bundle.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        bundle["meta"] = meta
    meta.update(
        {
            "dataset": "BCICIV_2a",
            "subject_id": subject_id,
            "channels": ch_names,
            "specs": [{"target": s.target, "inputs": list(s.inputs)} for s in specs],
        }
    )

    # -------------------------
    # DDPM per target
    # -------------------------
    if not args.skip_ddpm:
        p._log_section("BCI-IV 2a — DDPM: train per target + Table (MSE/Pearson) + hybrid + classifiers")

        for s in specs:
            target_idx = int(ch_to_idx[s.target])
            input_idxs = (int(ch_to_idx[s.inputs[0]]), int(ch_to_idx[s.inputs[1]]))
            ddpm_input_idxs[s.target] = input_idxs
            print(
                "-" * 36
                + f"\nDDPM target={s.target} inputs={s.inputs} "
                + f"| train epochs: {len(X_train)} eval epochs: {len(X_eval)} | ddpm_epochs: {int(args.ddpm_epochs)}"
            )

            # Create/train model or load from bundle
            model = p.UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256).to(device)
            key = f"{s.target}|steps{int(args.ddpm_epochs)}|inputs{input_idxs}"
            if bool(args.cache) and (not bool(args.force_train)) and key in saved_ddpm:
                model.load_state_dict(saved_ddpm[key], strict=True)
                print(f"[DDPM ckpt] loaded: target={s.target}")
            else:
                train_ds = p.EEGEpochs(X_train, input_idxs=input_idxs, target_idx=target_idx)
                train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
                opt = torch.optim.Adam(model.parameters(), lr=1e-4)
                sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
                p.train_diffusion_model(
                    model,
                    opt,
                    sched,
                    train_loader,
                    TIMESTEPS,
                    betas,
                    sqrt_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod,
                    epochs=int(args.ddpm_epochs),
                    device=device,
                    print_every=0,
                    show_epoch_progress=True,
                    epoch_desc=f"DDPM Train {s.target}",
                )
                if bool(args.cache):
                    saved_ddpm[key] = model.state_dict()
                    _save_bundle()

            ddpm_models[s.target] = model.eval()

            # Eval
            test_ds = p.EEGEpochs(X_eval, input_idxs=input_idxs, target_idx=target_idx)
            test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
            mse_list = []
            corr_list = []
            eval_total = len(test_ds) if int(args.n_eval_epochs) == 0 else min(int(args.n_eval_epochs), len(test_ds))
            seen = 0
            pbar = p.tqdm(total=eval_total, desc=f"DDPM Eval {s.target}", leave=True, mininterval=0.5)
            for cond, tgt in test_loader:
                if seen >= eval_total:
                    break
                remaining = eval_total - seen
                if cond.size(0) > remaining:
                    cond = cond[:remaining]
                    tgt = tgt[:remaining]
                cond = cond.to(device)
                tgt = tgt.to(device)
                syn = p.sample_from_model(
                    ddpm_models[s.target],
                    cond,
                    alphas,
                    alphas_cumprod,
                    betas,
                    sqrt_one_minus_alphas_cumprod,
                    shape=tuple(int(v) for v in tgt.shape),
                    device=device,
                )
                tgt_np = tgt.detach().cpu().numpy()[:, 0]
                syn_np = syn.detach().cpu().numpy()[:, 0]
                for i in range(tgt_np.shape[0]):
                    real_1d = tgt_np[i]
                    syn_1d = syn_np[i]
                    mse_list.append(float(np.mean((syn_1d - real_1d) ** 2)))
                    corr_list.append(p.pearson_corr_1d(real_1d, syn_1d))
                seen += int(tgt_np.shape[0])
                pbar.update(int(tgt_np.shape[0]))
            pbar.close()
            ddpm_rows.append(
                {
                    "Target Channel": s.target,
                    "Input Channels": f"{s.inputs[0]}, {s.inputs[1]}",
                    "MSE (computed)": float(np.mean(mse_list)) if mse_list else float("nan"),
                    "Pearson (computed)": float(np.mean(corr_list)) if corr_list else float("nan"),
                }
            )
            if mse_list and corr_list:
                print(
                    f"DDPM Computed MSE={float(np.mean(mse_list)):.6f} "
                    f"Pearson={float(np.mean(corr_list)):.6f} (n={len(mse_list)})"
                )

        try:
            import pandas as pd

            df = pd.DataFrame(ddpm_rows)
            print("\nDDPM Table (computed)")
            print(df.to_string(index=False))
        except Exception:
            print("\nDDPM Table (computed)")
            for r in ddpm_rows:
                print(r)

        # Hybrid (DDPM)
        X_test_hybrid_ddpm = build_hybrid_epochs_ddpm_2a(
            X_eval,
            specs=specs,
            trained_models=ddpm_models,
            trained_input_idxs=ddpm_input_idxs,
            ch_to_idx=ch_to_idx,
            gen_batch_size=int(args.gen_batch_size),
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            betas=betas,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            device=device,
        )

        # Classifiers (real vs hybrid)
        p._log_section(
            f"BCI-IV 2a — Classifier preprocessing (channels={args.clf_channels}, "
            f"window={float(args.clf_tmin)}–{float(args.clf_tmax)}s, resample={float(args.clf_resample)}Hz)"
        )
        X_train_clf, ch_names_clf, sfreq_clf = _prepare_classifier_epochs_2a(
            X_train,
            ch_names=ch_names,
            sfreq=float(epochs_train.info["sfreq"]),
            channel_set=str(args.clf_channels),
            tmin=float(args.clf_tmin),
            tmax=float(args.clf_tmax),
            resample_sfreq=float(args.clf_resample),
        )
        X_eval_clf, _, _ = _prepare_classifier_epochs_2a(
            X_eval,
            ch_names=ch_names,
            sfreq=float(epochs_train.info["sfreq"]),
            channel_set=str(args.clf_channels),
            tmin=float(args.clf_tmin),
            tmax=float(args.clf_tmax),
            resample_sfreq=float(args.clf_resample),
        )
        X_hybrid_clf, _, _ = _prepare_classifier_epochs_2a(
            X_test_hybrid_ddpm,
            ch_names=ch_names,
            sfreq=float(epochs_train.info["sfreq"]),
            channel_set=str(args.clf_channels),
            tmin=float(args.clf_tmin),
            tmax=float(args.clf_tmax),
            resample_sfreq=float(args.clf_resample),
        )
        print(f"[CLF] X_train: {X_train_clf.shape} | X_test: {X_eval_clf.shape} | channels: {ch_names_clf} | sfreq: {sfreq_clf}")

        ddpm_acc = {}
        ddpm_acc["CSP+LDA"] = run_csp_lda_robust(
            X_train=X_train_clf,
            y_train=y_train,
            X_test_real=X_eval_clf,
            X_test_hybrid=X_hybrid_clf,
            y_test=y_eval,
            sfreq=float(sfreq_clf),
        )
        ddpm_acc["FBCSP+LDA"] = run_fbcsp_lda_safe(
            X_train=X_train_clf,
            y_train=y_train,
            X_test_real=X_eval_clf,
            X_test_hybrid=X_hybrid_clf,
            y_test=y_eval,
            sfreq=float(sfreq_clf),
        )
        p.run_unet_classifier._epochs = int(args.cls_epochs)  # type: ignore[attr-defined]
        p.run_unet_classifier._batch_size = int(args.cls_batch_size)  # type: ignore[attr-defined]
        X_train_nn = _crop_last_if_odd_time(X_train_clf)
        X_eval_nn = _crop_last_if_odd_time(X_eval_clf)
        X_hybrid_nn = _crop_last_if_odd_time(X_hybrid_clf)
        ddpm_acc["U-Net"] = p.run_unet_classifier(
            X_train=X_train_nn,
            y_train=y_train,
            X_test_real=X_eval_nn,
            X_test_hybrid=X_hybrid_nn,
            y_test=y_eval,
            device=device,
        )
        ddpm_acc["EEGNet"] = p.run_eegnet(
            X_train=X_train_nn,
            y_train=y_train,
            X_test_real=X_eval_nn,
            X_test_hybrid=X_hybrid_nn,
            y_test=y_eval,
            device=device,
            epochs=int(args.cls_epochs),
            batch_size=int(args.cls_batch_size),
        )
        p._print_accuracy_table("BCI-IV 2a — DDPM accuracy summary (real vs hybrid)", ddpm_acc)

    # -------------------------
    # SGM per target
    # -------------------------
    if not args.skip_sgm:
        p._log_section("BCI-IV 2a — SGM/ScoreNet: train per target + Table (MSE/Pearson) + hybrid + classifiers")

        for s in specs:
            target_idx = int(ch_to_idx[s.target])
            input_idxs = (int(ch_to_idx[s.inputs[0]]), int(ch_to_idx[s.inputs[1]]))
            sgm_input_idxs[s.target] = input_idxs
            print(
                "-" * 36
                + f"\nSGM target={s.target} inputs={s.inputs} "
                + f"| train epochs: {len(X_train)} eval epochs: {len(X_eval)} | train_steps: {int(args.sgm_train_steps)}"
            )

            # Create model
            if args.sgm_backbone == "scorenet-unet":
                assert sgdm_mod is not None
                model = sgdm_mod.UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256, sigma=float(args.sgm_ve_sigma)).to(device)
                t_embed_scale = None
                sde_name = "ve"
            else:
                model = p.UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256).to(device)
                t_embed_scale = float(sde_train.N)
                sde_name = "vp"

            key = (
                f"{s.target}|{sde_name}|{args.sgm_backbone}|steps{int(args.sgm_train_steps)}|"
                f"ve_sigma{float(args.sgm_ve_sigma):g}|vp_beta_max{float(args.sgm_vp_beta_max):g}|"
                f"lw{int(bool(args.sgm_likelihood_weighting))}|clip{float(args.sgm_score_clip):g}"
            )

            if bool(args.cache) and (not bool(args.force_train)) and key in saved_sgm:
                model.load_state_dict(saved_sgm[key], strict=True)
                print(f"[SGM ckpt] loaded: target={s.target}")
            else:
                train_ds = p.EEGEpochs(X_train, input_idxs=input_idxs, target_idx=target_idx)
                train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
                opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.99, 0.999), eps=1e-8)
                ema_state = ema.init(model)
                if args.sgm_backbone == "scorenet-unet":
                    p.train_score_model_ve(
                        model=model,
                        sigma=float(args.sgm_ve_sigma),
                        dataloader=train_loader,
                        optimizer=opt,
                        device=device,
                        steps=int(args.sgm_train_steps),
                        grad_clip=1.0,
                        ema=ema,
                        ema_state=ema_state,
                        eval_every=int(args.sgm_eval_every),
                        desc=f"SGM Train {s.target}",
                        likelihood_weighting=bool(args.sgm_likelihood_weighting),
                        score_clip=float(args.sgm_score_clip),
                        t_embed_scale=t_embed_scale,
                    )
                else:
                    p.train_score_model_vp(
                        model=model,
                        sde=sde_train,
                        dataloader=train_loader,
                        optimizer=opt,
                        device=device,
                        steps=int(args.sgm_train_steps),
                        grad_clip=1.0,
                        ema=ema,
                        ema_state=ema_state,
                        eval_every=int(args.sgm_eval_every),
                        desc=f"SGM Train {s.target}",
                        t_embed_scale=t_embed_scale,
                        likelihood_weighting=bool(args.sgm_likelihood_weighting),
                        score_clip=float(args.sgm_score_clip),
                    )

                # build EMA model for sampling
                if args.sgm_backbone == "scorenet-unet":
                    ema_model = sgdm_mod.UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256, sigma=float(args.sgm_ve_sigma)).to(device)
                else:
                    ema_model = p.UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256).to(device)
                ema.copy_to(ema_state, ema_model)
                model = ema_model

                if bool(args.cache):
                    saved_sgm[key] = model.state_dict()
                    _save_bundle()

            sgm_models[s.target] = model.eval()

            # Eval
            test_ds = p.EEGEpochs(X_eval, input_idxs=input_idxs, target_idx=target_idx)
            test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False)

            mse_list = []
            corr_list = []
            eval_total = len(test_ds) if int(args.n_eval_epochs) == 0 else min(int(args.n_eval_epochs), len(test_ds))
            seen = 0
            pbar = p.tqdm(total=eval_total, desc=f"SGM Eval {s.target}", leave=True, mininterval=0.5)
            for cond, tgt in test_loader:
                if seen >= eval_total:
                    break
                remaining = eval_total - seen
                if cond.size(0) > remaining:
                    cond = cond[:remaining]
                    tgt = tgt[:remaining]
                cond = cond.to(device)
                tgt = tgt.to(device)

                if sde_name == "ve":
                    if args.sgm_sampler == "ode":
                        syn = p.ode_sampler_ve(
                            model=sgm_models[s.target],
                            sigma=float(args.sgm_ve_sigma),
                            conditioning_inputs=cond,
                            shape=tuple(int(v) for v in tgt.shape),
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
                        syn = p.pc_sampler_ve(
                            model=sgm_models[s.target],
                            sigma=float(args.sgm_ve_sigma),
                            conditioning_inputs=cond,
                            shape=tuple(int(v) for v in tgt.shape),
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
                        syn = p.ode_sampler_vp(
                            model=sgm_models[s.target],
                            sde=sde_sampling,
                            conditioning_inputs=cond,
                            shape=tuple(int(v) for v in tgt.shape),
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
                        syn = p.pc_sampler_vp(
                            model=sgm_models[s.target],
                            sde=sde_sampling,
                            conditioning_inputs=cond,
                            shape=tuple(int(v) for v in tgt.shape),
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

                tgt_np = tgt.detach().cpu().numpy()[:, 0]
                syn_np = syn.detach().cpu().numpy()[:, 0]
                for i in range(tgt_np.shape[0]):
                    real_1d = tgt_np[i]
                    syn_1d = syn_np[i]
                    mse_list.append(float(np.mean((syn_1d - real_1d) ** 2)))
                    corr_list.append(p.pearson_corr_1d(real_1d, syn_1d))
                seen += int(tgt_np.shape[0])
                pbar.update(int(tgt_np.shape[0]))
            pbar.close()

            sgm_rows.append(
                {
                    "Target Channel": s.target,
                    "Input Channels": f"{s.inputs[0]}, {s.inputs[1]}",
                    "MSE (computed)": float(np.mean(mse_list)) if mse_list else float("nan"),
                    "Pearson (computed)": float(np.mean(corr_list)) if corr_list else float("nan"),
                }
            )
            if mse_list and corr_list:
                print(
                    f"SGM Computed MSE={float(np.mean(mse_list)):.6f} "
                    f"Pearson={float(np.mean(corr_list)):.6f} (n={len(mse_list)})"
                )

        try:
            import pandas as pd

            df = pd.DataFrame(sgm_rows)
            print("\nSGM Table (computed)")
            print(df.to_string(index=False))
        except Exception:
            print("\nSGM Table (computed)")
            for r in sgm_rows:
                print(r)

        # Hybrid (SGM)
        X_hybrid = X_eval.copy()
        n = X_hybrid.shape[0]
        t_len = X_hybrid.shape[2]
        for s in specs:
            if s.target not in sgm_models:
                continue
            model_s = sgm_models[s.target]
            input_idxs = sgm_input_idxs[s.target]
            target_idx = int(ch_to_idx[s.target])
            pbar = p.tqdm(total=n, desc=f"SGM Synthesize {s.target}", leave=False, mininterval=0.5)
            for start in range(0, n, int(args.gen_batch_size)):
                end = min(n, start + int(args.gen_batch_size))
                batch = X_hybrid[start:end]
                cond = np.stack([batch[:, input_idxs[0], :], batch[:, input_idxs[1], :]], axis=1)
                cond_t = torch.from_numpy(cond).to(device)
                with torch.no_grad():
                    if args.sgm_backbone == "scorenet-unet":
                        syn = p.pc_sampler_ve(
                            model=model_s,
                            sigma=float(args.sgm_ve_sigma),
                            conditioning_inputs=cond_t,
                            shape=(cond_t.size(0), 1, t_len),
                            device=device,
                            sampling_N=int(args.sgm_sampling_n),
                            eps=float(args.sgm_sampling_eps),
                            snr=float(args.sgm_snr),
                            n_steps_each=int(args.sgm_n_steps_each),
                            init_corrector_steps=int(args.sgm_init_corrector_steps),
                            score_clip=float(args.sgm_score_clip),
                            t_embed_scale=None,
                        )
                    else:
                        syn = p.pc_sampler_vp(
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
                            t_embed_scale=float(sde_sampling.N),
                            init_corrector_steps=int(args.sgm_init_corrector_steps),
                            score_clip=float(args.sgm_score_clip),
                        )
                X_hybrid[start:end, target_idx, :] = syn.detach().cpu().numpy()[:, 0]
                pbar.update(end - start)
            pbar.close()

        # Classifiers
        p._log_section(
            f"BCI-IV 2a — Classifier preprocessing (channels={args.clf_channels}, "
            f"window={float(args.clf_tmin)}–{float(args.clf_tmax)}s, resample={float(args.clf_resample)}Hz)"
        )
        X_train_clf, ch_names_clf, sfreq_clf = _prepare_classifier_epochs_2a(
            X_train,
            ch_names=ch_names,
            sfreq=float(epochs_train.info["sfreq"]),
            channel_set=str(args.clf_channels),
            tmin=float(args.clf_tmin),
            tmax=float(args.clf_tmax),
            resample_sfreq=float(args.clf_resample),
        )
        X_eval_clf, _, _ = _prepare_classifier_epochs_2a(
            X_eval,
            ch_names=ch_names,
            sfreq=float(epochs_train.info["sfreq"]),
            channel_set=str(args.clf_channels),
            tmin=float(args.clf_tmin),
            tmax=float(args.clf_tmax),
            resample_sfreq=float(args.clf_resample),
        )
        X_hybrid_clf, _, _ = _prepare_classifier_epochs_2a(
            X_hybrid,
            ch_names=ch_names,
            sfreq=float(epochs_train.info["sfreq"]),
            channel_set=str(args.clf_channels),
            tmin=float(args.clf_tmin),
            tmax=float(args.clf_tmax),
            resample_sfreq=float(args.clf_resample),
        )
        print(f"[CLF] X_train: {X_train_clf.shape} | X_test: {X_eval_clf.shape} | channels: {ch_names_clf} | sfreq: {sfreq_clf}")

        sgm_acc = {}
        sgm_acc["CSP+LDA"] = run_csp_lda_robust(
            X_train=X_train_clf,
            y_train=y_train,
            X_test_real=X_eval_clf,
            X_test_hybrid=X_hybrid_clf,
            y_test=y_eval,
            sfreq=float(sfreq_clf),
        )
        sgm_acc["FBCSP+LDA"] = run_fbcsp_lda_safe(
            X_train=X_train_clf,
            y_train=y_train,
            X_test_real=X_eval_clf,
            X_test_hybrid=X_hybrid_clf,
            y_test=y_eval,
            sfreq=float(sfreq_clf),
        )
        p.run_unet_classifier._epochs = int(args.cls_epochs)  # type: ignore[attr-defined]
        p.run_unet_classifier._batch_size = int(args.cls_batch_size)  # type: ignore[attr-defined]
        X_train_nn = _crop_last_if_odd_time(X_train_clf)
        X_eval_nn = _crop_last_if_odd_time(X_eval_clf)
        X_hybrid_nn = _crop_last_if_odd_time(X_hybrid_clf)
        sgm_acc["U-Net"] = p.run_unet_classifier(
            X_train=X_train_nn,
            y_train=y_train,
            X_test_real=X_eval_nn,
            X_test_hybrid=X_hybrid_nn,
            y_test=y_eval,
            device=device,
        )
        sgm_acc["EEGNet"] = p.run_eegnet(
            X_train=X_train_nn,
            y_train=y_train,
            X_test_real=X_eval_nn,
            X_test_hybrid=X_hybrid_nn,
            y_test=y_eval,
            device=device,
            epochs=int(args.cls_epochs),
            batch_size=int(args.cls_batch_size),
        )
        p._print_accuracy_table("BCI-IV 2a — SGM accuracy summary (real vs hybrid)", sgm_acc)

    # Final verification
    p._log_section("Verification")
    _save_bundle()
    if ckpt_path.exists():
        ckpt2 = torch.load(ckpt_path, map_location="cpu")
        ddpm_n = len(ckpt2.get("ddpm", {})) if isinstance(ckpt2.get("ddpm", {}), dict) else -1
        sgm_n = len(ckpt2.get("sgm", {})) if isinstance(ckpt2.get("sgm", {}), dict) else -1
        print(f"[VERIFY] ckpt exists: {ckpt_path}")
        print(f"[VERIFY] ddpm entries: {ddpm_n} | sgm entries: {sgm_n}")
    else:
        print("[VERIFY] ckpt was not saved.")

    print("\nDone.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.1f}s")

