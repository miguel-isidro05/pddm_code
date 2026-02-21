"""
Pipeline2: BCI Competition IV 2a (GDF) — DDPM + SGM (score-based) channel reconstruction.

Goal: mirror the structure of `pipeline.py` (BCI-III) but using BCICIV 2a:
- Load AxxT.gdf (train) + AxxE.gdf (eval)
- Preprocess "paper-like" similar to BCI3 pipeline:
  bandpass 8–30 Hz + annotate_muscle_zscore (optional) + z-score using TRAIN stats only
- Epoching differs: 0–4s post-event (trial-based), per SBGM_EEG.ipynb
- Train DDPM per target channel and evaluate MSE/Pearson
- Train SGM per target channel and evaluate MSE/Pearson
- Build TEST(hybrid) replacing chosen target channels with synthetic ones
- Run CSP+LDA, FBCSP+LDA, EEGNet (reusing implementations from `pipeline.py`)
- Save checkpoints to a single file: `checkpoints/BCI4_subject{Axx}.pth`
  and auto-resume: if a target model exists, skip training and synthesize directly.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset

import mne

# Reuse models/samplers/classifiers from BCI3 pipeline
import pipeline as p


def _per_epoch_zscore_2d(x: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """
    Per-epoch (trial-wise) z-score for a single channel batch.
    Input: (B, T) -> Output: (B, T)
    """
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < float(eps), 1.0, sd)
    return (x - mu) / sd


def balanced_train_indices(labels: np.ndarray, *, seed: int, target_ratio: float = 1.0) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        return np.zeros((0,), dtype=int)
    ratio = float(target_ratio)
    ratio = max(1e-6, min(1.0, ratio))
    rng = np.random.default_rng(int(seed))
    cls_vals = np.unique(labels).astype(int)
    cls_to_idx: dict[int, np.ndarray] = {}
    min_count = np.inf
    for c in cls_vals.tolist():
        idx = np.flatnonzero(labels == int(c)).astype(int)
        cls_to_idx[int(c)] = idx
        min_count = min(min_count, float(idx.size))
    if not np.isfinite(min_count) or int(min_count) <= 0:
        return np.arange(labels.size, dtype=int)
    target_n = max(1, int(np.floor(float(min_count) * ratio)))
    picks: list[np.ndarray] = []
    for c in cls_vals.tolist():
        idx = cls_to_idx[int(c)].copy()
        rng.shuffle(idx)
        keep = min(int(idx.size), int(target_n))
        picks.append(np.sort(idx[:keep]))
    out = np.sort(np.concatenate(picks, axis=0)).astype(int)
    return out


def stratified_subsample_indices(labels: np.ndarray, *, train_fraction: float, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = int(labels.size)
    if n == 0:
        return np.zeros((0,), dtype=int)
    frac = float(train_fraction)
    if frac >= 0.999999:
        return np.arange(n, dtype=int)
    frac = min(0.999999, max(0.01, frac))
    rng = np.random.default_rng(int(seed))
    picks: list[np.ndarray] = []
    for cls in np.unique(labels).tolist():
        idx = np.flatnonzero(labels == int(cls)).astype(int)
        if idx.size <= 1:
            picks.append(idx)
            continue
        rng.shuffle(idx)
        k = int(round(float(idx.size) * frac))
        k = max(1, min(int(idx.size), k))
        picks.append(np.sort(idx[:k]))
    out = np.sort(np.concatenate(picks, axis=0)).astype(int)
    if out.size == 0:
        raise ValueError("stratified_subsample_indices produced empty selection.")
    return out


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
                "Kappa Original": float(no_cv.get("kappa_real", float("nan"))),
                "Kappa Hybrid": float(no_cv.get("kappa_hybrid", float("nan"))),
                "CV Acc Original (mean)": float(cv.get("acc_real", float("nan"))),
                "CV Acc Original (std)": float(cv.get("acc_real_std", float("nan"))),
                "CV Acc Hybrid (mean)": float(cv.get("acc_hybrid", float("nan"))),
                "CV Acc Hybrid (std)": float(cv.get("acc_hybrid_std", float("nan"))),
            }
        )
    return rows


def _print_accuracy_table(title: str, name_to_metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    p._log_section(title)
    rows = _accuracy_table_rows(name_to_metrics)
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    except Exception:
        for r in rows:
            print(r)


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
    Prepare epochs for classifiers (CSP/FBCSP/EEGNet) following typical 2a practice:
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
    csp_cfg: dict[str, object] | None = None,
) -> dict[str, dict[str, float]]:
    """CSP+LDA with the same robust logic used in pipeline.py."""
    cfg = dict(csp_cfg or {})
    return p.run_csp_lda(
        X_train=X_train,
        y_train=y_train,
        X_test_real=X_test_real,
        X_test_hybrid=X_test_hybrid,
        y_test=y_test,
        **cfg,
    )


def _crop_last_if_odd_time(X: np.ndarray) -> np.ndarray:
    """EEGNet in pipeline.py can be sensitive to odd time lengths."""
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
    fbcsp_cfg: dict[str, object] | None = None,
) -> dict[str, dict[str, float]]:
    """FBCSP+LDA with the same robust logic used in pipeline.py."""
    cfg = dict(fbcsp_cfg or {})
    return p.run_fbcsp_lda(
        X_train=X_train,
        y_train=y_train,
        X_test_real=X_test_real,
        X_test_hybrid=X_test_hybrid,
        y_test=y_test,
        sfreq=sfreq,
        **cfg,
    )

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
    """Reemplaza canales target por reconstrucción DDPM (sin post-procesado per-epoch z-score)."""
    # IMPORTANT: always condition on ORIGINAL real epochs (no cascaded synthesis).
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
            batch_real = epochs_real[s:e]  # (B, C, T)
            cond = np.stack(
                [batch_real[:, input_idxs[0], :], batch_real[:, input_idxs[1], :]],
                axis=1,
            ).astype(np.float32)  # (B,2,T)
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
    muscle_annotate: bool = False,
) -> mne.io.BaseRaw:
    """
    Preprocess similar to BCI3 pipeline, but leaving ICA+8-30 for epoch stage:
    - (optional export) drop EOG + bandpass 8-30 + notch(50) + save FIF (+ try EDF)
    - then: notch(50) + annotate_muscle_zscore
    - set standard montage
    - drop EOG channels (return only EEG arrays)
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

    # notch (line noise) like SBGM_EEG.ipynb
    try:
        raw.notch_filter(freqs=float(notch_freq), verbose=False)
    except Exception as e:
        print("[WARN] notch_filter skipped:", repr(e))

    # muscle annotations are OPTIONAL for 2a.
    # In practice, they can cause subject-dependent epoch dropping (reject_by_annotation=True default),
    # which makes CSP/FBCSP collapse for some subjects. Keep it off by default to match EEGNet_2a.ipynb.
    if bool(muscle_annotate):
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
                raw.set_annotations(orig_annotations)
        except Exception as e:
            print("annotate_muscle_zscore skipped:", e)

    # standard montage for consistency (ignore missing)
    try:
        raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
    except Exception:
        pass

    # Drop EOG channels; ICA artifact component detection will use frontal EEG if available.
    eog_chs = _infer_eog_channels(raw)
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
    Fixed target/input configuration (user-defined).
    """
    return [
        # IMPORTANT: do NOT synthesize C3/C4/Cz (motor-critical). Use them as inputs only.
        ExperimentSpec2A("C5", ("FC3", "C3")),
        ExperimentSpec2A("FC3", ("C1", "FC1")),
        ExperimentSpec2A("C6", ("FC4", "C4")),
        ExperimentSpec2A("P1", ("Pz", "CP1")),
        ExperimentSpec2A("P2", ("Pz", "CP2")),
    ]


def _auto_experiments_2a(
    *,
    ch_names: list[str],
    epochs: mne.Epochs,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_targets: int,
    target_selection: str,
    show_mi: bool,
) -> list[ExperimentSpec2A]:
    """
    Auto-choose targets that should be "low importance" for MI classification.

    Two modes:
    - target_selection="low-mi": compute per-channel mutual information (MI) with labels using a simple
      per-epoch feature (log-variance across time). Pick the lowest-MI channels outside motor pool.
    - target_selection="farthest": pick channels farthest from motor cortex (geometry-based fallback).

    In both modes, inputs are 2 nearest channels among motor-relevant pool (using standard_1020 positions).
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

    target_selection = str(target_selection).strip().lower()
    if target_selection not in {"low-mi", "farthest"}:
        raise ValueError("target_selection must be one of: low-mi, farthest")

    targets: list[str]
    if target_selection == "low-mi":
        # Data-driven: per-channel MI with labels using log-variance feature.
        try:
            from sklearn.feature_selection import mutual_info_classif

            Xtr = np.asarray(X_train, dtype=np.float32)
            ytr = np.asarray(y_train, dtype=int).reshape(-1)
            if Xtr.ndim != 3 or Xtr.shape[0] != ytr.shape[0]:
                raise ValueError(f"Bad shapes for MI selection: X_train={Xtr.shape} y_train={ytr.shape}")

            ch_to_idx_local = {c: int(i) for i, c in enumerate(ch_names)}
            cand_idx = [ch_to_idx_local[c] for c in candidates if c in ch_to_idx_local]
            cand_names = [c for c in candidates if c in ch_to_idx_local]
            if len(cand_idx) < 2:
                raise ValueError("Too few candidate channels for MI selection.")

            # Feature: log-variance across time for each epoch/channel (n_epochs, n_candidates)
            var = np.var(Xtr[:, cand_idx, :], axis=2).astype(np.float32)
            feats = np.log(var + 1e-6)
            mi = mutual_info_classif(feats, ytr, random_state=42)
            mi = np.asarray(mi, dtype=np.float64)
            order = np.argsort(mi)  # ascending => lowest MI first

            # Build a ranked list with optional tie-breaker (prefer farther-from-motor for similar MI)
            cand_ranked = [cand_names[int(i)] for i in order.tolist()]
            cand_ranked = sorted(
                cand_ranked,
                key=lambda c: (
                    float(mi[cand_names.index(c)]),
                    -float(np.linalg.norm(np.asarray(pos[c], dtype=float) - motor_center)),
                ),
            )
            targets = cand_ranked[: min(n_targets, len(cand_ranked))]

            # Print MI ranking for transparency (MICCAI-friendly)
            if bool(show_mi):
                rows = []
                for c in cand_ranked:
                    mi_c = float(mi[cand_names.index(c)])
                    dist_c = float(np.linalg.norm(np.asarray(pos[c], dtype=float) - motor_center))
                    rows.append(
                        {
                            "Channel": c,
                            "MI(logvar, y)": mi_c,
                            "DistFromMotor": dist_c,
                            "SelectedTarget": bool(c in set(targets)),
                        }
                    )
                print("\n" + "-" * 80)
                print("Auto target selection (low-mi): candidate ranking (ascending MI)")
                print("-" * 80)
                try:
                    import pandas as pd

                    df = pd.DataFrame(rows)
                    # Show a compact view: top 15 least-informative
                    print(df.head(15).to_string(index=False))
                except Exception:
                    for r in rows[:15]:
                        print(r)
        except Exception as e:
            print(f"[WARN] low-mi target selection failed ({e}); falling back to farthest-from-motor.")
            cand_sorted = sorted(
                candidates,
                key=lambda c: float(np.linalg.norm(np.asarray(pos[c], dtype=float) - motor_center)),
                reverse=True,
            )
            targets = cand_sorted[: min(n_targets, len(cand_sorted))]
    else:
        # Geometry-based: farthest from motor center
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
        "--reject-by-annotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, MNE Epochs drops trials overlapping BAD annotations. Default=False to match EEGNet_2a.ipynb.",
    )
    parser.add_argument(
        "--muscle-annotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, run annotate_muscle_zscore (can cause subject-dependent epoch dropping). Default=False.",
    )
    parser.add_argument(
        "--debug-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print extra diagnostics for debugging CV/std/hybrid collapse. Does not change results.",
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
    parser.add_argument(
        # NOTE: keep target channels fixed (no auto-selection) for reproducibility.
        # (Auto-selection removed per user request.)
        "--auto-specs",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--n-targets", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--target-selection", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--show-target-mi", action="store_false", help=argparse.SUPPRESS)

    parser.add_argument("--ddpm-epochs", type=int, default=250)
    parser.add_argument(
        "--ddpm-epochs-cv",
        type=int,
        default=250,
        help="DDPM epochs used inside TRAIN-only CV folds (0 => use --ddpm-epochs).",
    )
    parser.add_argument(
        "--final-train-fraction",
        type=float,
        default=1.0,
        help="Fraction of TRAIN used in held-out stage (0,1]. Default=1.0 uses all TRAIN.",
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
    parser.add_argument(
        "--run-train-cv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, run TRAIN-only CV (fold-safe z-score) before held-out EVAL evaluation.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="TRAIN-only CV folds (StratifiedKFold).")
    parser.add_argument("--cv-seed", type=int, default=42, help="Random seed for TRAIN-only CV split.")
    parser.add_argument(
        "--cv-deep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, also train EEGNet inside each CV fold (slow). Default=False (CSP/FBCSP only).",
    )

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

    parser.add_argument("--cls-epochs", type=int, default=200)
    parser.add_argument("--cls-batch-size", type=int, default=128)

    # EEGNet-only (notebook baseline) debugging
    parser.add_argument(
        "--only-eegnet",
        action="store_true",
        help="Run ONLY EEGNet baseline (no DDPM/SGM/CSP/FBCSP). Mirrors EEGNet_2a.ipynb preprocessing.",
    )
    parser.add_argument(
        "--only-csp-fbcsp",
        action="store_true",
        help="Run ONLY CSP+LDA and FBCSP+LDA (TRAIN-CV + held-out EVAL). Skips DDPM/EEGNet/SGM.",
    )
    parser.add_argument(
        "--only-csp-fbcsp-no-cv",
        action="store_true",
        help="With --only-csp-fbcsp: skip CV and only run full-TRAIN -> held-out EVAL.",
    )
    parser.add_argument("--eegnet-temp-kernel", type=int, default=32)
    parser.add_argument("--eegnet-f1", type=int, default=16)
    parser.add_argument("--eegnet-d", type=int, default=2)
    parser.add_argument("--eegnet-f2", type=int, default=32)
    parser.add_argument("--eegnet-pk1", type=int, default=8)
    parser.add_argument("--eegnet-pk2", type=int, default=16)
    parser.add_argument("--eegnet-dropout", type=float, default=0.5)
    parser.add_argument("--eegnet-lr", type=float, default=1e-3)
    parser.add_argument("--eegnet-patience", type=int, default=80)
    parser.add_argument("--eegnet-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--eegnet-use-plateau",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use ReduceLROnPlateau on val_loss. Default=False to keep fixed learning rate.",
    )
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

    # Bump this string whenever you change preprocessing that affects DDPM inputs.
    # This prevents accidentally reusing stale checkpoints trained on a different data distribution.
    PREPROC_ID = "2a|notch50|foldsafe-ica-1hzfit-then-bp8-30|drop_eog|zscore_train|t0-4s|stable-cls-cspfbcsp|v6"

    debug_logs = bool(args.debug_logs)
    if debug_logs:
        p._log_section("DEBUG — Run configuration")
        try:
            print("argv:", " ".join(sys.argv))
        except Exception:
            pass
        print("PREPROC_ID:", PREPROC_ID)
        print(
            "cfg:",
            {
                "seed": int(args.seed),
                "device": str(args.device),
                "ddpm_epochs": int(args.ddpm_epochs),
                "ddpm_epochs_cv": int(args.ddpm_epochs_cv),
                "cv_folds": int(args.cv_folds),
                "cv_seed": int(args.cv_seed),
                "batch_size": int(args.batch_size),
                "gen_batch_size": int(args.gen_batch_size),
                "reject_by_annotation": bool(args.reject_by_annotation),
                "muscle_annotate": bool(args.muscle_annotate),
                "clf_channels": str(args.clf_channels),
                "clf_tmin": float(args.clf_tmin),
                "clf_tmax": float(args.clf_tmax),
                "clf_resample": float(args.clf_resample),
            },
        )

    p._set_seeds(int(args.seed))

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _parse_reg(v: str) -> str | None:
        vv = str(v).strip().lower()
        return None if vv == "none" else vv

    def _reseed_for_eegnet(offset: int) -> None:
        # Keep seed behavior consistent with script start (no per-stage reseeding).
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

    # Fold-safe ICA is applied later at epoch level (train-only fit in held-out/CV contexts).
    raw_train = _preprocess_like_bci3(
        raw_train,
        muscle_annotate=bool(args.muscle_annotate),
    )
    cleaned_dir = Path(str(args.cleaned_dir)).expanduser().resolve()
    eval_fif = cleaned_dir / f"Evaluation_{subject_id}.fif"
    eval_edf = cleaned_dir / f"Evaluation_{subject_id}.edf"
    raw_eval = _preprocess_like_bci3(
        raw_eval,
        export_fif_path=eval_fif if bool(args.export_cleaned_eval) else None,
        export_edf_path=eval_edf if bool(args.export_cleaned_eval) else None,
        notch_freq=50.0,
        muscle_annotate=bool(args.muscle_annotate),
    )

    # Sanity: expect 22 EEG channels after preprocessing (EOG dropped).
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
        reject_by_annotation=bool(args.reject_by_annotation),
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
        reject_by_annotation=bool(args.reject_by_annotation),
        baseline=None,
        preload=True,
        verbose=False,
    )

    try:
        n_drop_tr = int(sum(len(x) > 0 for x in getattr(epochs_train, "drop_log", [])))
        n_drop_ev = int(sum(len(x) > 0 for x in getattr(epochs_eval, "drop_log", [])))
        print(
            f"[EPOCHS] dropped TRAIN={n_drop_tr} | dropped EVAL={n_drop_ev} | "
            f"reject_by_annotation={bool(args.reject_by_annotation)} | muscle_annotate={bool(args.muscle_annotate)}"
        )
    except Exception:
        pass

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

    # PAPER-STANDARD: do NOT modify the held-out EVAL set to "balance" labels.
    # Keep EVAL intact and report Balanced Accuracy / per-class metrics when needed.

    # ------------------------------------------------------------------------------
    # Fold-safe preprocessing philosophy (same as pipeline.py):
    # - ICA -> fit ONLY on TRAIN (or fold-train) using temporary 1 Hz high-pass copy
    # - then apply ICA to unfiltered epochs and bandpass 8–30
    # - z-score -> fit ONLY on TRAIN (or fold-train) and apply to val/test.
    # ------------------------------------------------------------------------------
    info_all = epochs_train.info.copy()
    # Keep raw (pre-ICA/pre-z) epochs for TRAIN-only CV
    X_train_raw_all = np.asarray(X_train, dtype=np.float32)
    X_eval_raw_all = np.asarray(X_eval, dtype=np.float32)

    # Keep EEG channels only (EOG already dropped at raw stage)
    picks_eeg_only = mne.pick_types(info_all, eeg=True, eog=False, stim=False, exclude=[])
    if int(len(picks_eeg_only)) <= 0:
        raise ValueError("No EEG channels found after preprocessing.")
    X_train = X_train_raw_all[:, picks_eeg_only, :]
    X_eval = X_eval_raw_all[:, picks_eeg_only, :]
    ch_names = [str(info_all["ch_names"][int(i)]).strip() for i in picks_eeg_only]
    ch_to_idx = {c: i for i, c in enumerate(ch_names)}

    if bool(args.balance_train):
        idx_bal = balanced_train_indices(y_train, seed=int(args.seed), target_ratio=float(args.balance_ratio))
        if int(idx_bal.size) > 0:
            X_train = np.asarray(X_train[idx_bal], dtype=np.float32)
            y_train = np.asarray(y_train[idx_bal], dtype=np.int64)
            print(
                f"[balance] enabled: ratio={float(args.balance_ratio):.3f} | "
                f"kept {int(idx_bal.size)} train epochs"
            )

    # Keep CV views on full TRAIN (after optional balancing, before final-train-fraction).
    X_train_cv_raw_all = np.asarray(X_train, dtype=np.float32)
    y_train_cv_all = np.asarray(y_train, dtype=np.int64)

    def _ica_fit_apply_epochs(Xtr_raw: np.ndarray, Xother_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit ICA on TRAIN epochs only using a temporary 1 Hz high-pass copy.
        Apply ICA to unfiltered TRAIN/other epochs, then bandpass 8–30.
        """
        from mne.preprocessing import ICA

        Xtr_raw = np.asarray(Xtr_raw, dtype=np.float64)
        Xother_raw = np.asarray(Xother_raw, dtype=np.float64)
        info_ep = mne.create_info(ch_names=ch_names, sfreq=float(info_all["sfreq"]), ch_types="eeg")
        ep_tr = mne.EpochsArray(Xtr_raw, info_ep, verbose=False)
        ep_other = mne.EpochsArray(Xother_raw, info_ep, verbose=False)
        ep_tr_hp = ep_tr.copy().filter(l_freq=1.0, h_freq=None, method="iir", verbose=False)

        ica_f = ICA(n_components=min(20, int(ep_tr_hp.info["nchan"])), random_state=97, max_iter=800)
        ica_f.fit(ep_tr_hp, verbose=False)
        try:
            frontal_candidates = [c for c in ["Fp1", "Fp2", "Fpz", "Fz", "FC1", "FC2"] if c in ch_names]
            if len(frontal_candidates) > 0:
                eog_idx, _ = ica_f.find_bads_eog(ep_tr_hp, ch_name=frontal_candidates, verbose=False)
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

    # Held-out preprocessing: fit ICA on full TRAIN only, apply to TRAIN+EVAL, then z-score on TRAIN stats.
    X_train_ica, X_eval_ica = _ica_fit_apply_epochs(X_train, X_eval)
    mu_ch = X_train_ica.mean(axis=(0, 2), keepdims=True)
    sd_ch = X_train_ica.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train = ((X_train_ica - mu_ch) / sd_ch).astype(np.float32, copy=False)
    X_eval = ((X_eval_ica - mu_ch) / sd_ch).astype(np.float32, copy=False)

    # Final held-out train size control (do not affect CV folds).
    if float(args.final_train_fraction) < 0.999999:
        idx_sub = stratified_subsample_indices(
            y_train,
            train_fraction=float(args.final_train_fraction),
            seed=int(args.seed),
        )
        X_train = np.asarray(X_train[idx_sub], dtype=np.float32)
        y_train = np.asarray(y_train[idx_sub], dtype=np.int64)
        print(
            f"[stability] final-train-fraction={float(args.final_train_fraction):.3f} "
            f"=> held-out training uses {int(idx_sub.size)} epochs"
        )

    # Dataset summary (similar to pipeline.py)
    sfreq = float(epochs_train.info["sfreq"])
    uniq, cnt = np.unique(y_train, return_counts=True)
    class_balance = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
    uniq_e, cnt_e = np.unique(y_eval, return_counts=True)
    class_balance_eval = {int(k): int(v) for k, v in zip(uniq_e.tolist(), cnt_e.tolist())}
    print(f"sfreq: {sfreq}")
    print(f"TRAIN: {tuple(int(v) for v in X_train.shape)} | EVAL: {tuple(int(v) for v in X_eval.shape)}")
    print(f"class balance train (0..3): {class_balance}")
    print(f"class balance eval  (0..3): {class_balance_eval}")
    print(f"n_channels EEG: {len(ch_names)} | channels: {ch_names}")

    if debug_logs:
        p._log_section("DEBUG — Dataset balance / shapes (post-alignment, pre-CV)")
        print("TRAIN balance:", class_balance)
        print("EVAL  balance:", class_balance_eval)

    # ----------------------------------------------------------------------------------
    # EEGNet-only baseline (mirror EEGNet_2a.ipynb data prep) — no DDPM/SGM/hybrid
    # ----------------------------------------------------------------------------------
    if bool(args.only_eegnet):
        from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
        from sklearn.model_selection import StratifiedShuffleSplit

        p._log_section(f"BCI-IV 2a — EEGNet ONLY (notebook baseline) — Subject {subject_id}")

        # 1) classifier window/channel prep (keep defaults = all22, 0..4s, no resample)
        Xtr_p, ch_used, sfreq_used = _prepare_classifier_epochs_2a(
            X_train,
            ch_names=ch_names,
            sfreq=float(sfreq),
            channel_set=str(args.clf_channels),
            tmin=float(args.clf_tmin),
            tmax=float(args.clf_tmax),
            resample_sfreq=float(args.clf_resample),
        )
        Xev_p, _, _ = _prepare_classifier_epochs_2a(
            X_eval,
            ch_names=ch_names,
            sfreq=float(sfreq),
            channel_set=str(args.clf_channels),
            tmin=float(args.clf_tmin),
            tmax=float(args.clf_tmax),
            resample_sfreq=float(args.clf_resample),
        )

        # 2) Data already passed fold-safe ICA then 8-30. Apply channel-wise z-score using TRAIN stats only.
        Xtr_bp = np.asarray(Xtr_p, dtype=np.float64)
        Xev_bp = np.asarray(Xev_p, dtype=np.float64)
        mu = Xtr_bp.mean(axis=(0, 2), keepdims=True)
        sd = Xtr_bp.std(axis=(0, 2), keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        Xtr_norm = (Xtr_bp - mu) / sd
        Xev_norm = (Xev_bp - mu) / sd

        # 3) Trialwise z-score (per trial, per channel over time)
        def _trialwise_zscore(x: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
            mu_t = x.mean(axis=2, keepdims=True)
            sd_t = x.std(axis=2, keepdims=True)
            sd_t = np.where(sd_t < eps, 1.0, sd_t)
            return (x - mu_t) / sd_t

        Xtr_norm = _trialwise_zscore(np.asarray(Xtr_norm, dtype=np.float32))
        Xev_norm = _trialwise_zscore(np.asarray(Xev_norm, dtype=np.float32))

        # tensors: (B, 1, C, T)
        Xtr_t = torch.as_tensor(Xtr_norm, dtype=torch.float32).unsqueeze(1).to(device)
        Xev_t = torch.as_tensor(Xev_norm, dtype=torch.float32).unsqueeze(1).to(device)
        ytr_t = torch.as_tensor(np.asarray(y_train, dtype=int), dtype=torch.long).to(device)
        yev_t = torch.as_tensor(np.asarray(y_eval, dtype=int), dtype=torch.long).to(device)

        n_classes = int(np.max(np.concatenate([y_train, y_eval])) + 1)
        if n_classes != 4:
            raise ValueError(f"Expected 4 classes for 2a, got n_classes={n_classes}")

        chans = int(Xtr_t.shape[2])
        time_points = int(Xtr_t.shape[3])
        print(f"[EEGNet-only] dims -> chans={chans} | time_points={time_points} | n_classes={n_classes}")
        print(
            f"[EEGNet-only] cfg: temp_kernel={int(args.eegnet_temp_kernel)} f1={int(args.eegnet_f1)} "
            f"d={int(args.eegnet_d)} f2={int(args.eegnet_f2)} pk1={int(args.eegnet_pk1)} pk2={int(args.eegnet_pk2)} "
            f"dropout={float(args.eegnet_dropout):g} | lr={float(args.eegnet_lr):g} wd={float(args.eegnet_weight_decay):g} "
            f"| epochs={int(args.cls_epochs)} batch={int(args.cls_batch_size)} plateau={bool(args.eegnet_use_plateau)}"
        )
        print(f"[EEGNet-only] channels used ({len(ch_used)}): {ch_used}")

        _reseed_for_eegnet(11_000)
        model = p.EEGNetModel(
            chans=chans,
            classes=n_classes,
            time_points=time_points,
            temp_kernel=int(args.eegnet_temp_kernel),
            f1=int(args.eegnet_f1),
            d=int(args.eegnet_d),
            f2=int(args.eegnet_f2),
            pk1=int(args.eegnet_pk1),
            pk2=int(args.eegnet_pk2),
            dropout_rate=float(args.eegnet_dropout),
        ).to(device)

        # Train/val split like notebook (StratifiedShuffleSplit 80/20)
        y_np = ytr_t.detach().cpu().numpy().astype(int).reshape(-1)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(args.seed))
        tr_idx, va_idx = next(sss.split(np.zeros(len(y_np)), y_np))
        tr_idx_t = torch.as_tensor(tr_idx, dtype=torch.long, device=device)
        va_idx_t = torch.as_tensor(va_idx, dtype=torch.long, device=device)

        X_train_split = Xtr_t.index_select(0, tr_idx_t)
        y_train_split = ytr_t.index_select(0, tr_idx_t)
        X_val_split = Xtr_t.index_select(0, va_idx_t)
        y_val_split = ytr_t.index_select(0, va_idx_t)

        opt = torch.optim.Adam(model.parameters(), lr=float(args.eegnet_lr), weight_decay=float(args.eegnet_weight_decay))
        y_train_split_np = y_train_split.detach().cpu().numpy().astype(int)
        class_w = p._compute_class_weights_from_labels(y_train_split_np, n_classes=int(n_classes)).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_w)
        print(f"[EEGNet-only] class weights: {class_w.detach().cpu().numpy().round(4).tolist()}")
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20, min_lr=1e-5)
            if bool(args.eegnet_use_plateau)
            else None
        )

        train_loader = DataLoader(
            TensorDataset(X_train_split, y_train_split),
            batch_size=int(args.cls_batch_size),
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_split, y_val_split),
            batch_size=int(args.cls_batch_size),
            shuffle=False,
            drop_last=False,
        )

        best_state = None
        best_val_acc = -1.0
        best_val_loss = float("inf")
        wait = 0
        patience = int(args.eegnet_patience)
        min_delta = 1e-4
        min_epochs_before_stop = 120

        pbar = p.tqdm(range(int(args.cls_epochs)), desc="EEGNet-only train", leave=True, mininterval=0.5)
        for ep in pbar:
            model.train()
            loss_sum = 0.0
            n_seen = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                try:
                    model.apply_constraints()
                except Exception:
                    pass
                loss_sum += float(loss.item()) * int(yb.size(0))
                n_seen += int(yb.size(0))
            train_loss = float(loss_sum / max(1, n_seen))

            model.eval()
            val_loss_sum = 0.0
            n_val = 0
            y_true = []
            y_pred = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss_sum += float(loss.item()) * int(yb.size(0))
                    n_val += int(yb.size(0))
                    y_true.extend(yb.detach().cpu().numpy().astype(int).tolist())
                    y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist())
            val_loss = float(val_loss_sum / max(1, n_val))
            val_acc = float(accuracy_score(np.asarray(y_true), np.asarray(y_pred)))

            if scheduler is not None:
                scheduler.step(val_loss)
            lr_now = float(opt.param_groups[0]["lr"])
            if val_loss < (best_val_loss - float(min_delta)):
                best_val_loss = float(val_loss)
                wait = 0
            else:
                wait += 1

            pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.3f}",
                wait=str(wait),
                lr=f"{lr_now:.2e}",
            )

            if val_acc > best_val_acc + 1e-6:
                best_val_acc = float(val_acc)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if (int(ep) + 1) >= int(min_epochs_before_stop) and wait >= int(patience):
                print(f"[EEGNet-only] early stopping at epoch {int(ep)+1}, best_val_loss={best_val_loss:.4f}")
                break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Final evaluation on official EVAL
        model.eval()
        with torch.no_grad():
            logits = model(Xev_t)
            y_pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
        y_true = yev_t.detach().cpu().numpy().astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        kappa = float(cohen_kappa_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, normalize="true")

        print("\n=== EEGNet ONLY (notebook baseline) ===")
        print({"acc_eval": acc, "kappa_eval": kappa, "best_val_acc": float(best_val_acc), "best_val_loss": float(best_val_loss)})
        print("\nConfusionMatrix (normalize=true) EVAL:\n", np.array2string(cm, precision=3, floatmode="fixed"))
        return

    # ----------------------------------------------------------------------------------
    # CSP/FBCSP-only mode (no DDPM, no EEGNet, no SGM)
    # ----------------------------------------------------------------------------------
    if bool(args.only_csp_fbcsp):
        from sklearn.model_selection import StratifiedKFold

        p._log_section(f"BCI-IV 2a — CSP/FBCSP ONLY (TRAIN-CV + held-out EVAL) — Subject {subject_id}")

        def _zscore_fit_apply(Xtr_in: np.ndarray, Xother_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mu = Xtr_in.mean(axis=(0, 2), keepdims=True)
            sd = Xtr_in.std(axis=(0, 2), keepdims=True) + 1e-6
            return ((Xtr_in - mu) / sd).astype(np.float32, copy=False), ((Xother_in - mu) / sd).astype(np.float32, copy=False)

        cv_acc_real: dict[str, list[float]] = {"CSP+LDA": [], "FBCSP+LDA": []}
        cv_kappa_real: dict[str, list[float]] = {"CSP+LDA": [], "FBCSP+LDA": []}
        if not bool(args.only_csp_fbcsp_no_cv):
            X_train_cv_raw = np.asarray(X_train_cv_raw_all, dtype=np.float32)
            y_train_cv = np.asarray(y_train_cv_all, dtype=int)
            skf = StratifiedKFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.cv_seed))
            for fold_i, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_train_cv)), y_train_cv), start=1):
                tr_idx = np.asarray(tr_idx, dtype=int)
                va_idx = np.asarray(va_idx, dtype=int)
                Xtr_raw = X_train_cv_raw[tr_idx]
                Xva_raw = X_train_cv_raw[va_idx]
                ytr = y_train_cv[tr_idx]
                yva = y_train_cv[va_idx]

                p._log_section(f"[CSP/FBCSP-only CV fold {fold_i}/{int(args.cv_folds)}] train->val (real only)")
                Xtr_ica, Xva_ica = _ica_fit_apply_epochs(Xtr_raw, Xva_raw)
                Xtr_z, Xva_z = _zscore_fit_apply(Xtr_ica, Xva_ica)

                Xtr_clf, _, sfreq_clf = _prepare_classifier_epochs_2a(
                    Xtr_z,
                    ch_names=ch_names,
                    sfreq=float(info_all["sfreq"]),
                    channel_set=str(args.clf_channels),
                    tmin=float(args.clf_tmin),
                    tmax=float(args.clf_tmax),
                    resample_sfreq=float(args.clf_resample),
                )
                Xva_clf, _, _ = _prepare_classifier_epochs_2a(
                    Xva_z,
                    ch_names=ch_names,
                    sfreq=float(info_all["sfreq"]),
                    channel_set=str(args.clf_channels),
                    tmin=float(args.clf_tmin),
                    tmax=float(args.clf_tmax),
                    resample_sfreq=float(args.clf_resample),
                )

                out_csp = run_csp_lda_robust(
                    X_train=Xtr_clf,
                    y_train=ytr,
                    X_test_real=Xva_clf,
                    X_test_hybrid=Xva_clf,  # placeholder (hybrid not evaluated in this mode)
                    y_test=yva,
                    sfreq=float(sfreq_clf),
                    csp_cfg=csp_cfg_cli,
                )
                out_f = run_fbcsp_lda_safe(
                    X_train=Xtr_clf,
                    y_train=ytr,
                    X_test_real=Xva_clf,
                    X_test_hybrid=Xva_clf,  # placeholder (hybrid not evaluated in this mode)
                    y_test=yva,
                    sfreq=float(sfreq_clf),
                    fbcsp_cfg=fbcsp_cfg_cli,
                )
                for name, out in [("CSP+LDA", out_csp), ("FBCSP+LDA", out_f)]:
                    no_cv = out.get("no_cv", {})
                    cv_acc_real[name].append(float(no_cv.get("acc_real", float("nan"))))
                    cv_kappa_real[name].append(float(no_cv.get("kappa_real", float("nan"))))

        def _mean_std(vals: list[float]) -> tuple[float, float]:
            a = np.asarray(vals, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan"), float("nan")
            return float(np.mean(a)), float(np.std(a))

        p._log_section("CSP/FBCSP-only: train on FULL TRAIN, evaluate held-out EVAL (real only)")
        X_train_clf, _, sfreq_clf = _prepare_classifier_epochs_2a(
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
        out_csp_eval = run_csp_lda_robust(
            X_train=X_train_clf,
            y_train=y_train,
            X_test_real=X_eval_clf,
            X_test_hybrid=X_eval_clf,
            y_test=y_eval,
            sfreq=float(sfreq_clf),
            csp_cfg=csp_cfg_cli,
        )
        out_f_eval = run_fbcsp_lda_safe(
            X_train=X_train_clf,
            y_train=y_train,
            X_test_real=X_eval_clf,
            X_test_hybrid=X_eval_clf,
            y_test=y_eval,
            sfreq=float(sfreq_clf),
            fbcsp_cfg=fbcsp_cfg_cli,
        )

        cv_csp_m, cv_csp_s = _mean_std(cv_acc_real["CSP+LDA"])
        cv_fbcsp_m, cv_fbcsp_s = _mean_std(cv_acc_real["FBCSP+LDA"])
        no_cv_csp = out_csp_eval.get("no_cv", {})
        no_cv_f = out_f_eval.get("no_cv", {})
        print("\n" + "=" * 80)
        print(f"CSP/FBCSP ONLY - SUBJECT {subject_id} — Final summary (TRAIN-CV + held-out EVAL)")
        print("=" * 80)
        print(
            {
                "Model": "CSP+LDA",
                "Acc Original": float(no_cv_csp.get("acc_real", float("nan"))),
                "Kappa Original": float(no_cv_csp.get("kappa_real", float("nan"))),
                "CV Acc Original (mean/std)": f"{cv_csp_m:.3f} ({cv_csp_s:.3f})" if cv_acc_real["CSP+LDA"] else "skipped",
                "Note": "Hybrid/DDPM/SGM/EEGNet skipped in --only-csp-fbcsp mode",
            }
        )
        print(
            {
                "Model": "FBCSP+LDA",
                "Acc Original": float(no_cv_f.get("acc_real", float("nan"))),
                "Kappa Original": float(no_cv_f.get("kappa_real", float("nan"))),
                "CV Acc Original (mean/std)": f"{cv_fbcsp_m:.3f} ({cv_fbcsp_s:.3f})" if cv_acc_real["FBCSP+LDA"] else "skipped",
                "Note": "Hybrid/DDPM/SGM/EEGNet skipped in --only-csp-fbcsp mode",
            }
        )
        return

    # Fixed channel specs only (no auto-selection)
    # Choose less-informative targets by geometry: farthest from motor center (10-20 montage).
    # Conditioning inputs remain 2 nearest channels from the motor-relevant pool.
    specs = _auto_experiments_2a(
        ch_names=ch_names,
        epochs=epochs_train,
        X_train=X_train,
        y_train=y_train,
        n_targets=5,
        target_selection="farthest",
        show_mi=False,
    )

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

    # Invalidate cached models when preprocessing changed
    prev_preproc = str(meta.get("preproc_id", "") or "")
    if prev_preproc and prev_preproc != PREPROC_ID:
        print(f"[CKPT] preproc_id mismatch. prev={prev_preproc!r} now={PREPROC_ID!r}. Ignoring cached models.")
        bundle["ddpm"] = {}
        bundle["sgm"] = {}
    meta["preproc_id"] = PREPROC_ID
    meta.update(
        {
            "dataset": "BCICIV_2a",
            "subject_id": subject_id,
            "channels": ch_names,
            "specs": [{"target": s.target, "inputs": list(s.inputs)} for s in specs],
        }
    )

    # ----------------------------------------------------------------------------------
    # TRAIN-only CV (rigorous, fold-safe z-score) — DDPM pipeline
    # ----------------------------------------------------------------------------------
    cv_summary_ddpm: dict[str, dict[str, float]] = {}
    if bool(args.run_train_cv) and (not bool(args.skip_ddpm)):
        from sklearn.model_selection import StratifiedKFold

        p._log_section(
            f"BCI-IV 2a — DDPM TRAIN-only CV (fold-safe z-score) (k={int(args.cv_folds)}) — Subject {subject_id}"
        )

        # We run CV on raw (pre-ICA/pre-zscore) TRAIN epochs,
        # then within each fold:
        # 1) fit ICA on fold-train using temporary 1 Hz HP copy, apply to train+val
        # 2) fit z-score on fold-train, apply to train+val
        X_train_cv_raw_all = np.asarray(X_train_cv_raw_all, dtype=np.float32)
        y_train_cv = np.asarray(y_train_cv_all, dtype=int)

        skf = StratifiedKFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.cv_seed))

        cv_acc_real: dict[str, list[float]] = {"CSP+LDA": [], "FBCSP+LDA": []}
        cv_acc_h: dict[str, list[float]] = {"CSP+LDA": [], "FBCSP+LDA": []}
        cv_kappa_real: dict[str, list[float]] = {"CSP+LDA": [], "FBCSP+LDA": []}
        cv_kappa_h: dict[str, list[float]] = {"CSP+LDA": [], "FBCSP+LDA": []}

        if bool(args.cv_deep):
            cv_acc_real.update({"EEGNet": []})
            cv_acc_h.update({"EEGNet": []})
            cv_kappa_real.update({"EEGNet": []})
            cv_kappa_h.update({"EEGNet": []})

        def _mean_std(vals: list[float]) -> tuple[float, float]:
            a = np.asarray(vals, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan"), float("nan")
            return float(np.mean(a)), float(np.std(a))

        def _zscore_fit_apply(Xtr: np.ndarray, Xother: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mu = Xtr.mean(axis=(0, 2), keepdims=True)
            sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
            return ((Xtr - mu) / sd).astype(np.float32, copy=False), ((Xother - mu) / sd).astype(np.float32, copy=False)

        ddpm_epochs_cv = int(args.ddpm_epochs_cv) if int(args.ddpm_epochs_cv) > 0 else int(args.ddpm_epochs)

        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_train_cv)), y_train_cv), start=1):
            tr_idx = np.asarray(tr_idx, dtype=int)
            va_idx = np.asarray(va_idx, dtype=int)
            Xtr_all = X_train_cv_raw_all[tr_idx]
            Xva_all = X_train_cv_raw_all[va_idx]
            ytr = y_train_cv[tr_idx]
            yva = y_train_cv[va_idx]

            p._log_section(f"[CV fold {fold_i}/{int(args.cv_folds)}] fit DDPM on train, eval on val (real vs hybrid)")
            if debug_logs:
                ub_tr, cb_tr = np.unique(ytr, return_counts=True)
                ub_va, cb_va = np.unique(yva, return_counts=True)
                bal_tr = {int(k): int(v) for k, v in zip(ub_tr.tolist(), cb_tr.tolist())}
                bal_va = {int(k): int(v) for k, v in zip(ub_va.tolist(), cb_va.tolist())}
                print(f"[DEBUG fold {fold_i}] n_train={len(ytr)} balance_train={bal_tr}")
                print(f"[DEBUG fold {fold_i}] n_val  ={len(yva)} balance_val  ={bal_va}")

            # ICA fold-safe + z-score fold-safe (fit on train only)
            Xtr_ica, Xva_ica = _ica_fit_apply_epochs(Xtr_all, Xva_all)
            Xtr_z, Xva_z = _zscore_fit_apply(Xtr_ica, Xva_ica)

            # Train DDPM per target on fold-train only (no caching)
            fold_models: dict[str, torch.nn.Module] = {}
            fold_input_idxs: dict[str, tuple[int, int]] = {}
            for s in specs:
                target_idx = int(ch_to_idx[s.target])
                input_idxs = (int(ch_to_idx[s.inputs[0]]), int(ch_to_idx[s.inputs[1]]))
                fold_input_idxs[s.target] = input_idxs

                model = p.UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256).to(device)
                train_ds = p.EEGEpochs(Xtr_z, input_idxs=input_idxs, target_idx=target_idx)
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
                    epochs=int(ddpm_epochs_cv),
                    device=device,
                    print_every=0,
                    show_epoch_progress=True,
                    epoch_desc=f"DDPM Train {s.target} (fold {fold_i})",
                )
                fold_models[s.target] = model.eval()

            # Build val hybrid (DDPM)
            Xva_hybrid = build_hybrid_epochs_ddpm_2a(
                Xva_z,
                specs=specs,
                trained_models=fold_models,
                trained_input_idxs=fold_input_idxs,
                ch_to_idx=ch_to_idx,
                gen_batch_size=int(args.gen_batch_size),
                alphas=alphas,
                alphas_cumprod=alphas_cumprod,
                betas=betas,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                device=device,
            )
            if debug_logs:
                print(f"[DEBUG fold {fold_i}] reconstruction quality on VAL (per target):")
                for s in specs:
                    ti = int(ch_to_idx[s.target])
                    real = np.asarray(Xva_z[:, ti, :], dtype=np.float64)
                    syn = np.asarray(Xva_hybrid[:, ti, :], dtype=np.float64)
                    mse = float(np.mean((real - syn) ** 2))
                    real_c = real - real.mean(axis=1, keepdims=True)
                    syn_c = syn - syn.mean(axis=1, keepdims=True)
                    num = np.sum(real_c * syn_c, axis=1)
                    den = np.sqrt(np.sum(real_c**2, axis=1) * np.sum(syn_c**2, axis=1)) + 1e-12
                    corr = num / den
                    pear = float(np.nanmean(corr))
                    print(f"  - {s.target}: MSE={mse:.4f} | Pearson={pear:.3f}")

            # Classifier preprocessing (match held-out evaluation)
            Xtr_clf, _, sfreq_clf = _prepare_classifier_epochs_2a(
                Xtr_z,
                ch_names=ch_names,
                sfreq=float(info_all["sfreq"]),
                channel_set=str(args.clf_channels),
                tmin=float(args.clf_tmin),
                tmax=float(args.clf_tmax),
                resample_sfreq=float(args.clf_resample),
            )
            Xva_clf, _, _ = _prepare_classifier_epochs_2a(
                Xva_z,
                ch_names=ch_names,
                sfreq=float(info_all["sfreq"]),
                channel_set=str(args.clf_channels),
                tmin=float(args.clf_tmin),
                tmax=float(args.clf_tmax),
                resample_sfreq=float(args.clf_resample),
            )
            Xva_h_clf, _, _ = _prepare_classifier_epochs_2a(
                Xva_hybrid,
                ch_names=ch_names,
                sfreq=float(info_all["sfreq"]),
                channel_set=str(args.clf_channels),
                tmin=float(args.clf_tmin),
                tmax=float(args.clf_tmax),
                resample_sfreq=float(args.clf_resample),
            )

            out_csp = run_csp_lda_robust(
                X_train=Xtr_clf,
                y_train=ytr,
                X_test_real=Xva_clf,
                X_test_hybrid=Xva_h_clf,
                y_test=yva,
                sfreq=float(sfreq_clf),
                csp_cfg=csp_cfg_cli,
            )
            out_f = run_fbcsp_lda_safe(
                X_train=Xtr_clf,
                y_train=ytr,
                X_test_real=Xva_clf,
                X_test_hybrid=Xva_h_clf,
                y_test=yva,
                sfreq=float(sfreq_clf),
                fbcsp_cfg=fbcsp_cfg_cli,
            )

            for name, out in [("CSP+LDA", out_csp), ("FBCSP+LDA", out_f)]:
                no_cv = out.get("no_cv", {})
                cv_acc_real[name].append(float(no_cv.get("acc_real", float("nan"))))
                cv_acc_h[name].append(float(no_cv.get("acc_hybrid", float("nan"))))
                cv_kappa_real[name].append(float(no_cv.get("kappa_real", float("nan"))))
                cv_kappa_h[name].append(float(no_cv.get("kappa_hybrid", float("nan"))))

            if bool(args.cv_deep):
                Xtr_nn = _crop_last_if_odd_time(Xtr_clf)
                Xva_nn = _crop_last_if_odd_time(Xva_clf)
                Xva_h_nn = _crop_last_if_odd_time(Xva_h_clf)
                _reseed_for_eegnet(12_000 + int(fold_i))
                out_e = p.run_eegnet(
                    X_train=Xtr_nn,
                    y_train=ytr,
                    X_test_real=Xva_nn,
                    X_test_hybrid=Xva_h_nn,
                    y_test=yva,
                    device=device,
                    epochs=int(args.cls_epochs),
                    batch_size=int(args.cls_batch_size),
                    eegnet_lr=float(args.eegnet_lr),
                    eegnet_dropout=float(args.eegnet_dropout),
                    eegnet_patience=int(args.eegnet_patience),
                )
                no_cv = out_e.get("no_cv", {})
                cv_acc_real["EEGNet"].append(float(no_cv.get("acc_real", float("nan"))))
                cv_acc_h["EEGNet"].append(float(no_cv.get("acc_hybrid", float("nan"))))
                cv_kappa_real["EEGNet"].append(float(no_cv.get("kappa_real", float("nan"))))
                cv_kappa_h["EEGNet"].append(float(no_cv.get("kappa_hybrid", float("nan"))))

        # summarize
        for name in cv_acc_real.keys():
            m1, s1 = _mean_std(cv_acc_real[name])
            m2, s2 = _mean_std(cv_acc_h[name])
            cv_summary_ddpm[name] = {
                "cv_acc_real_mean": m1,
                "cv_acc_real_std": s1,
                "cv_acc_h_mean": m2,
                "cv_acc_h_std": s2,
            }

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
            key = f"{PREPROC_ID}|{s.target}|steps{int(args.ddpm_epochs)}|inputs{input_idxs}"
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
            csp_cfg=csp_cfg_cli,
        )
        ddpm_acc["FBCSP+LDA"] = run_fbcsp_lda_safe(
            X_train=X_train_clf,
            y_train=y_train,
            X_test_real=X_eval_clf,
            X_test_hybrid=X_hybrid_clf,
            y_test=y_eval,
            sfreq=float(sfreq_clf),
            fbcsp_cfg=fbcsp_cfg_cli,
        )
        X_train_nn = _crop_last_if_odd_time(X_train_clf)
        X_eval_nn = _crop_last_if_odd_time(X_eval_clf)
        X_hybrid_nn = _crop_last_if_odd_time(X_hybrid_clf)
        _reseed_for_eegnet(13_000)
        ddpm_acc["EEGNet"] = p.run_eegnet(
            X_train=X_train_nn,
            y_train=y_train,
            X_test_real=X_eval_nn,
            X_test_hybrid=X_hybrid_nn,
            y_test=y_eval,
            device=device,
            epochs=int(args.cls_epochs),
            batch_size=int(args.cls_batch_size),
            eegnet_lr=float(args.eegnet_lr),
            eegnet_dropout=float(args.eegnet_dropout),
            eegnet_patience=int(args.eegnet_patience),
        )
        _print_accuracy_table("BCI-IV 2a — DDPM accuracy summary (real vs hybrid)", ddpm_acc)

        # Final summary: TRAIN-only CV (if enabled) + held-out EVAL
        print("\n" + "=" * 80)
        print(f"DDPM - SUBJECT {subject_id} — Final summary (TRAIN-CV + held-out EVAL)")
        print("=" * 80)
        rows = []
        for name, out in ddpm_acc.items():
            no_cv = out.get("no_cv", {})
            cv = cv_summary_ddpm.get(name, {})
            rows.append(
                {
                    "Model": name,
                    "Acc Original": float(no_cv.get("acc_real", float("nan"))),
                    "Acc Hybrid": float(no_cv.get("acc_hybrid", float("nan"))),
                    "Kappa Original": float(no_cv.get("kappa_real", float("nan"))),
                    "Kappa Hybrid": float(no_cv.get("kappa_hybrid", float("nan"))),
                    "CV Acc Original (mean/std)": (
                        f"{cv.get('cv_acc_real_mean', float('nan')):.3f} ({cv.get('cv_acc_real_std', float('nan')):.3f})"
                        if cv
                        else "skipped"
                    ),
                    "CV Acc Hybrid (mean/std)": (
                        f"{cv.get('cv_acc_h_mean', float('nan')):.3f} ({cv.get('cv_acc_h_std', float('nan')):.3f})"
                        if cv
                        else "skipped"
                    ),
                }
            )
        try:
            import pandas as pd

            df_final = pd.DataFrame(rows)
            print(df_final.to_string(index=False))
        except Exception:
            for r in rows:
                print(r)

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
                # Condition on ORIGINAL EVAL (real) inputs to avoid cascaded synthesis.
                batch_real = X_eval[start:end]
                cond = np.stack([batch_real[:, input_idxs[0], :], batch_real[:, input_idxs[1], :]], axis=1)
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
                syn_np = syn.detach().cpu().numpy()[:, 0]
                X_hybrid[start:end, target_idx, :] = syn_np.astype(np.float32, copy=False)
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
            csp_cfg=csp_cfg_cli,
        )
        sgm_acc["FBCSP+LDA"] = run_fbcsp_lda_safe(
            X_train=X_train_clf,
            y_train=y_train,
            X_test_real=X_eval_clf,
            X_test_hybrid=X_hybrid_clf,
            y_test=y_eval,
            sfreq=float(sfreq_clf),
            fbcsp_cfg=fbcsp_cfg_cli,
        )
        X_train_nn = _crop_last_if_odd_time(X_train_clf)
        X_eval_nn = _crop_last_if_odd_time(X_eval_clf)
        X_hybrid_nn = _crop_last_if_odd_time(X_hybrid_clf)
        _reseed_for_eegnet(14_000)
        sgm_acc["EEGNet"] = p.run_eegnet(
            X_train=X_train_nn,
            y_train=y_train,
            X_test_real=X_eval_nn,
            X_test_hybrid=X_hybrid_nn,
            y_test=y_eval,
            device=device,
            epochs=int(args.cls_epochs),
            batch_size=int(args.cls_batch_size),
            eegnet_lr=float(args.eegnet_lr),
            eegnet_dropout=float(args.eegnet_dropout),
            eegnet_patience=int(args.eegnet_patience),
        )
        _print_accuracy_table("BCI-IV 2a — SGM accuracy summary (real vs hybrid)", sgm_acc)

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

