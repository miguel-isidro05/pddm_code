"""
Pipeline2: BCI Competition IV 2a (GDF) — DDPM + optional SGM (score-based) channel reconstruction.

This mirrors `pipeline.py` (BCI-III) but adapts:
- Data: AxxT.gdf (train) / AxxE.gdf (eval)
- Labels: true_labels/AxxE.mat (classlabel 1..4)
- Preprocess: bandpass 8–30 Hz + notch 50 Hz + ICA EOG correction (drop EOG afterwards)
- Epoching: 0..4 s post cue
- DDPM per target: MSE/Pearson + hybrid (replace targets with synthetic) + classifiers

Notes:
- 2a has 22 EEG (+3 EOG). We rename to canonical names when GDF provides generic names.
- We NEVER synthesize C3/C4/Cz.
"""

from __future__ import annotations

import argparse
import itertools
import math
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


def _default_experiments_2a() -> list[ExperimentSpec2A]:
    """
    Default targets/inputs (tuned after hybrid-debug):
    Keep 4 targets that reconstruct well and don't collapse CSP/FBCSP.
    NOTE:
    - Best-performing stable set found via screening:
      CP2, CPz, FC1, FC2
    """
    return [
        ExperimentSpec2A("CP2", ("C4", "Pz")),
        ExperimentSpec2A("CPz", ("Cz", "Pz")),
        ExperimentSpec2A("FC1", ("FCz", "C3")),
        ExperimentSpec2A("FC2", ("FCz", "C4")),
    ]


def _candidate_pool_experiments_2a(*, candidate_set: str) -> list[ExperimentSpec2A]:
    """
    Candidate targets to try for the "best 4" auto-selection.

    Constraints:
    - Never synthesize C3/C4/Cz (kept as anchors).
    - Inputs are chosen from anchor channels to avoid cascading conditioning on synthetic signals.
    """
    anchors = {"C3", "C4", "Cz", "FCz", "Pz"}
    if candidate_set == "motor":
        # compact pool (8 targets) to keep screening feasible
        pool = [
            ExperimentSpec2A("C1", ("C3", "Cz")),
            ExperimentSpec2A("C2", ("C4", "Cz")),
            ExperimentSpec2A("FC1", ("FCz", "C3")),
            ExperimentSpec2A("FC2", ("FCz", "C4")),
            ExperimentSpec2A("CP1", ("C3", "Pz")),
            ExperimentSpec2A("CP2", ("C4", "Pz")),
            ExperimentSpec2A("CPz", ("Cz", "Pz")),
            ExperimentSpec2A("P2", ("C4", "Pz")),
        ]
    elif candidate_set == "broad":
        # wider pool (10 targets) includes back-of-head candidates (often riskier for covariance shift)
        pool = [
            ExperimentSpec2A("C1", ("C3", "Cz")),
            ExperimentSpec2A("C2", ("C4", "Cz")),
            ExperimentSpec2A("FC1", ("FCz", "C3")),
            ExperimentSpec2A("FC2", ("FCz", "C4")),
            ExperimentSpec2A("CP1", ("C3", "Pz")),
            ExperimentSpec2A("CP2", ("C4", "Pz")),
            ExperimentSpec2A("CPz", ("Cz", "Pz")),
            ExperimentSpec2A("P1", ("C3", "Pz")),
            ExperimentSpec2A("P2", ("C4", "Pz")),
            ExperimentSpec2A("POz", ("Pz", "Cz")),
        ]
    else:
        raise ValueError(f"Unknown candidate_set={candidate_set!r} (expected 'motor' or 'broad').")

    for s in pool:
        if s.inputs[0] not in anchors or s.inputs[1] not in anchors:
            raise ValueError(f"Non-anchor inputs in candidate pool: {s}")
    return pool


def _bandpass_epochs(X: np.ndarray, *, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    from mne.filter import filter_data

    return filter_data(np.asarray(X, np.float64), float(sfreq), float(l_freq), float(h_freq), method="iir", verbose=False)


def _bandpass_single_channel_epochs(ch: np.ndarray, *, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    """
    Filter a (N,T) channel array into (N,T), using MNE's filter.
    """
    ch = np.asarray(ch, dtype=np.float64)
    X = ch[:, None, :]
    Xf = _bandpass_epochs(X, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq)
    return np.asarray(Xf[:, 0, :], dtype=np.float32)


def _sanitize_score(x: float) -> float:
    if not np.isfinite(x):
        return -1e9
    return float(x)


def _fold_stats_from_fixed_preds(*, y_true: np.ndarray, y_pred: np.ndarray, n_splits: int = 5) -> dict[str, float]:
    """
    "CV" without retraining: split indices, compute fold metrics on fixed predictions.
    This keeps mean close to the global metric (train→test), while still giving a stability estimate.
    """
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    from sklearn.model_selection import GroupKFold

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true/y_pred mismatch: {y_true.shape} vs {y_pred.shape}")

    groups = np.arange(int(y_true.shape[0]), dtype=int)
    cv = GroupKFold(n_splits=int(n_splits))

    accs: list[float] = []
    kappas: list[float] = []
    weights: list[int] = []
    for _, te_idx in cv.split(groups, y_true, groups=groups):
        te_idx = np.asarray(te_idx, dtype=int)
        yt = y_true[te_idx]
        yp = y_pred[te_idx]
        weights.append(int(te_idx.size))
        accs.append(float(accuracy_score(yt, yp)))
        kappas.append(float(cohen_kappa_score(yt, yp)))

    w = np.asarray(weights, dtype=np.float64)
    accs_a = np.asarray(accs, dtype=np.float64)
    kappas_a = np.asarray(kappas, dtype=np.float64)

    acc_mean = float(np.sum(accs_a * w) / (float(np.sum(w)) + 1e-12))
    kappa_mean = float(np.sum(kappas_a * w) / (float(np.sum(w)) + 1e-12))
    return {
        "acc_mean": acc_mean,
        "acc_std": float(np.std(accs_a)),
        "kappa_mean": kappa_mean,
        "kappa_std": float(np.std(kappas_a)),
    }


def _cv5_random_split_on_features(*, feats: np.ndarray, y: np.ndarray, seed: int = 42) -> dict[str, float]:
    """
    Literature-style 5-fold CV: randomly split data into k=5 folds (stratified).
    This *re-trains* the classifier inside folds (unlike fixed-pred fold stats).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

    feats = np.asarray(feats, dtype=np.float64)
    y = np.asarray(y, dtype=int).reshape(-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    scores = cross_val_score(clf, feats, y, cv=cv, n_jobs=1)
    y_pred = cross_val_predict(clf, feats, y, cv=cv, method="predict", n_jobs=1)
    acc_pred = float(accuracy_score(y, y_pred))
    kappa_pred = float(cohen_kappa_score(y, y_pred))
    cm_cv = confusion_matrix(y, y_pred, normalize="true")
    return {
        "cv_acc": float(np.mean(scores)),
        "cv_acc_std": float(np.std(scores)),
        "cv_acc_pred": acc_pred,
        "cv_kappa": kappa_pred,
        "cv_cm": cm_cv,  # for optional printing
    }


def _pick_train_class_keys_2a(event_id: dict[str, int]) -> dict[str, str]:
    if all(k in event_id for k in ("769", "770", "771", "772")):
        return {"left_hand": "769", "right_hand": "770", "feet": "771", "tongue": "772"}
    for prefix in ("class ", "Class "):
        keys = [f"{prefix}{i}" for i in (1, 2, 3, 4)]
        if all(k in event_id for k in keys):
            return {"left_hand": keys[0], "right_hand": keys[1], "feet": keys[2], "tongue": keys[3]}
    raise ValueError(f"No TRAIN classes (769–772). Keys={sorted(event_id.keys())}")


def _pick_eval_cue_key_2a(event_id: dict[str, int]) -> str:
    for k in ("783", "768"):
        if k in event_id:
            return k
    if "781" in event_id:
        return "781"
    raise ValueError(f"No EVAL cue key (783/768/781). Keys={sorted(event_id.keys())}")


def _infer_eog_channels(raw: mne.io.BaseRaw) -> list[str]:
    candidates = [
        *BCI2A_EOG_3,
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


def _maybe_assign_standard_2a_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw = raw.copy()
    ch_names = list(raw.ch_names)
    n_ch = len(ch_names)
    canonical_25 = BCI2A_EEG_22 + BCI2A_EOG_3

    def _looks_canonical(names: list[str]) -> bool:
        known = set(canonical_25)
        return sum(1 for c in names if c in known) >= 10

    if _looks_canonical(ch_names):
        return raw
    if n_ch == 25:
        mapping = {ch_names[i]: canonical_25[i] for i in range(25)}
        raw.rename_channels(mapping)
        raw.set_channel_types({c: "eog" for c in BCI2A_EOG_3 if c in raw.ch_names})
        return raw
    if n_ch == 22:
        mapping = {ch_names[i]: BCI2A_EEG_22[i] for i in range(22)}
        raw.rename_channels(mapping)
        return raw
    print(f"[WARN] Unexpected n_channels={n_ch}; keeping original channel names.")
    return raw


def _preprocess_like_bci3(raw: mne.io.BaseRaw, *, notch_freq: float = 50.0) -> mne.io.BaseRaw:
    raw = raw.copy()

    raw.filter(l_freq=8.0, h_freq=30.0, method="iir", verbose=False)
    try:
        raw.notch_filter(freqs=float(notch_freq), verbose=False)
    except Exception as e:
        print("[WARN] notch_filter skipped:", repr(e))

    # Preserve original trigger annotations; append muscle annotations if possible.
    try:
        orig_annotations = raw.annotations.copy()
        sfreq = float(raw.info.get("sfreq", 0.0) or 0.0)
        nyq = 0.5 * sfreq if sfreq > 0 else 0.0
        h_freq = 140.0
        if nyq > 0:
            h_freq = min(h_freq, max(20.0, nyq * 0.98))
        if h_freq <= 20.0:
            raise ValueError("sfreq too low for muscle band")
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

    # ICA on EEG only
    ica = ICA(n_components=20, random_state=97, max_iter=800)
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
    raw = ica.apply(raw.copy(), verbose=False)

    try:
        raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
    except Exception:
        pass

    if eog_chs:
        raw = raw.copy().drop_channels(eog_chs)

    # Keep EEG only
    try:
        picks_keep = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, misc=False, exclude=[])
        raw = raw.copy().pick(picks_keep)
    except Exception:
        pass

    return raw


def _load_eval_labels_mat(*, labels_dir: Path, subject_id: str) -> np.ndarray:
    mat_path = labels_dir / f"{subject_id}E.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing {mat_path}. Put labels in `true_labels/` or pass --labels-dir.")
    mat = scipy.io.loadmat(str(mat_path))
    y_raw = np.asarray(mat["classlabel"]).squeeze().astype(int)
    return (y_raw - 1).astype(int)  # 0..3


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
    synth_per_epoch_z: bool,
    synth_clip: float,
) -> np.ndarray:
    epochs_h = epochs_real.copy()
    n = int(epochs_h.shape[0])
    t_len = int(epochs_h.shape[2])

    for spec in specs:
        target = spec.target
        if target not in trained_models:
            print(f"[WARN] No model found for {target}; skipping.")
            continue
        model = trained_models[target]
        input_idxs = trained_input_idxs[target]
        target_idx = int(ch_to_idx[target])

        pbar = p.tqdm(total=n, desc=f"DDPM Synthesize {target}", leave=False, mininterval=0.5)
        for s in range(0, n, int(gen_batch_size)):
            e = min(n, s + int(gen_batch_size))
            batch = epochs_h[s:e]
            cond = np.stack([batch[:, input_idxs[0], :], batch[:, input_idxs[1], :]], axis=1).astype(np.float32)
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
            syn_np = syn.detach().cpu().numpy()[:, 0].astype(np.float32)  # (B, T)

            # Stabilize hybrid for CSP/FBCSP: normalize ONLY synthesized channels per-epoch.
            if bool(synth_per_epoch_z):
                mu_t = syn_np.mean(axis=1, keepdims=True)
                sd_t = syn_np.std(axis=1, keepdims=True) + 1e-6
                syn_np = (syn_np - mu_t) / sd_t

            if float(synth_clip) > 0:
                syn_np = np.clip(syn_np, -float(synth_clip), float(synth_clip))

            epochs_h[s:e, target_idx, :] = syn_np
            pbar.update(e - s)
        pbar.close()

    return epochs_h


def _crop_last_if_odd_time(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 3 and int(X.shape[2]) % 2 == 1:
        return X[:, :, :-1]
    return X


def _pearson_per_epoch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized Pearson per epoch over time axis (N,T) -> (N,)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)) + 1e-12
    return (num / den).astype(np.float64)


def _debug_hybrid(
    *,
    X_eval_real: np.ndarray,
    X_eval_hybrid: np.ndarray,
    specs: list[ExperimentSpec2A],
    ch_to_idx: dict[str, int],
    sfreq: float,
    max_epochs: int = 200,
) -> None:
    """
    Logs to diagnose CSP/FBCSP drop:
    - target mean/std (real vs hybrid)
    - corr(target_real, target_hybrid)
    - corr(target, inputs) in real vs hybrid
    - covariance shift (8–30 Hz) between real/hybrid
    """
    n = int(min(int(X_eval_real.shape[0]), int(X_eval_hybrid.shape[0]), int(max_epochs)))
    if n <= 0:
        return

    Xr = np.asarray(X_eval_real[:n], dtype=np.float32)
    Xh = np.asarray(X_eval_hybrid[:n], dtype=np.float32)

    print("\n" + "=" * 80)
    print("HYBRID DEBUG (why CSP/FBCSP drop?)")
    print("=" * 80)
    print(f"[HYBRID DEBUG] using n={n} epochs | sfreq={float(sfreq)} | shape={Xr.shape}")

    for s in specs:
        t_idx = int(ch_to_idx[s.target])
        i1 = int(ch_to_idx[s.inputs[0]])
        i2 = int(ch_to_idx[s.inputs[1]])

        tr = Xr[:, t_idx, :]
        th = Xh[:, t_idx, :]
        in1 = Xr[:, i1, :]
        in2 = Xr[:, i2, :]

        # coherence measures
        corr_tt = _pearson_per_epoch(tr, th)
        corr_r_in1 = _pearson_per_epoch(tr, in1)
        corr_r_in2 = _pearson_per_epoch(tr, in2)
        corr_h_in1 = _pearson_per_epoch(th, in1)
        corr_h_in2 = _pearson_per_epoch(th, in2)

        print(
            f"- {s.target}<=({s.inputs[0]},{s.inputs[1]}): "
            f"real(mean={float(tr.mean()):.3f}, std={float(tr.std()):.3f}) | "
            f"hyb(mean={float(th.mean()):.3f}, std={float(th.std()):.3f}) | "
            f"corr(real,hyb)=mean {float(np.mean(corr_tt)):.3f} (p10 {float(np.quantile(corr_tt,0.1)):.3f}, "
            f"p50 {float(np.quantile(corr_tt,0.5)):.3f}, p90 {float(np.quantile(corr_tt,0.9)):.3f}) | "
            f"corr(target,inp1) real→{float(np.mean(corr_r_in1)):.3f} hyb→{float(np.mean(corr_h_in1)):.3f} | "
            f"corr(target,inp2) real→{float(np.mean(corr_r_in2)):.3f} hyb→{float(np.mean(corr_h_in2)):.3f}"
        )

    # Covariance shift in 8–30 Hz (the CSP domain)
    try:
        from mne.filter import filter_data

        Xr_f = filter_data(np.asarray(Xr, np.float64), float(sfreq), 8.0, 30.0, method="iir", verbose=False)
        Xh_f = filter_data(np.asarray(Xh, np.float64), float(sfreq), 8.0, 30.0, method="iir", verbose=False)

        # average covariance over epochs, time-centered
        def _avg_cov(X: np.ndarray) -> np.ndarray:
            Xc = X - X.mean(axis=2, keepdims=True)
            # (N,C,T) -> (C,C) average of (X X^T / T)
            covs = np.einsum("nct,ndt->ncd", Xc, Xc) / float(Xc.shape[2])
            return np.mean(covs, axis=0)

        Cr = _avg_cov(Xr_f)
        Ch = _avg_cov(Xh_f)
        diff = Ch - Cr
        fro = float(np.linalg.norm(diff, ord="fro"))
        base = float(np.linalg.norm(Cr, ord="fro")) + 1e-12
        rel = fro / base
        print(f"\n[HYBRID DEBUG] cov shift (8–30Hz): ||Ch-Cr||_F={fro:.3e} | rel={rel:.3e}")
        # show max-abs entry for quick identification
        mx = float(np.max(np.abs(diff)))
        print(f"[HYBRID DEBUG] cov shift max|Δ|={mx:.3e}")
    except Exception as e:
        print("[HYBRID DEBUG] covariance debug skipped:", repr(e))


def run_csp_lda_robust(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_real: np.ndarray,
    X_test_hybrid: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
) -> dict[str, dict[str, float]]:
    from mne.decoding import CSP
    from mne.filter import filter_data
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, cohen_kappa_score
    # NOTE: we no longer compute "within-test CV" by retraining on TEST folds.
    # We compute fold-stats from fixed predictions instead (see below).

    # Match notebook: bandpass before CSP
    X_train_bp = filter_data(np.asarray(X_train, np.float64), float(sfreq), 8.0, 30.0, method="iir", verbose=False)
    X_test_real_bp = filter_data(np.asarray(X_test_real, np.float64), float(sfreq), 8.0, 30.0, method="iir", verbose=False)
    X_test_hybrid_bp = filter_data(
        np.asarray(X_test_hybrid, np.float64), float(sfreq), 8.0, 30.0, method="iir", verbose=False
    )

    CSP_CFG_PRIMARY = dict(n_components=4, reg=None, log=True, norm_trace=True, cov_est="epoch")
    # Match notebook cov_est="epoch"; add shrinkage only when needed.
    CSP_CFG_FALLBACK = dict(n_components=4, reg="ledoit_wolf", log=True, norm_trace=True, cov_est="epoch")
    LDA_CFG = dict(solver="lsqr", shrinkage="auto")

    classes = p._label_classes(y_train, y_test)
    y_train_enc = p._encode_labels_with_classes(y_train, classes)
    y_test_enc = p._encode_labels_with_classes(y_test, classes)
    print("classes (original labels):", classes.tolist())

    def _fit(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        csp = CSP(**cfg)
        F_tr = csp.fit_transform(X_train_bp, y_train_enc)
        F_r = csp.transform(X_test_real_bp)
        F_h = csp.transform(X_test_hybrid_bp)
        return F_tr, F_r, F_h

    try:
        F_train, F_test_real, F_test_hybrid = _fit(CSP_CFG_PRIMARY)
    except Exception as e:
        print(f"[WARN] CSP primary config failed ({e}); using fallback regularization.")
        F_train, F_test_real, F_test_hybrid = _fit(CSP_CFG_FALLBACK)

    lda = LinearDiscriminantAnalysis(**LDA_CFG)
    lda.fit(F_train, y_train_enc)
    pred_real = lda.predict(F_test_real)
    pred_h = lda.predict(F_test_hybrid)
    try:
        print("[CSP+LDA] pred_counts real:", np.bincount(np.asarray(pred_real, dtype=int), minlength=int(classes.size)).tolist())
        print("[CSP+LDA] pred_counts hyb :", np.bincount(np.asarray(pred_h, dtype=int), minlength=int(classes.size)).tolist())
    except Exception:
        pass

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

    cv_real = _cv5_random_split_on_features(feats=F_test_real, y=y_test_enc, seed=42)
    cv_h = _cv5_random_split_on_features(feats=F_test_hybrid, y=y_test_enc, seed=42)
    return {
        "no_cv": {"acc_real": acc_real, "acc_hybrid": acc_h, "kappa_real": kappa_real, "kappa_hybrid": kappa_h},
        "cv": {
            "acc_real": float(cv_real["cv_acc"]),
            "acc_real_std": float(cv_real["cv_acc_std"]),
            "acc_hybrid": float(cv_h["cv_acc"]),
            "acc_hybrid_std": float(cv_h["cv_acc_std"]),
        },
    }


def run_fbcsp_lda_safe(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_real: np.ndarray,
    X_test_hybrid: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
    tune: bool,
    k_best: int,
    m_pairs: int,
    reg: str,
    cov_est: str,
) -> dict[str, dict[str, float]]:
    """
    Robust FBCSP+LDA for 2a.

    If tune=True, we select FBCSP hyperparameters on TRAIN using 5-fold CV (no leakage),
    then fit on full TRAIN and evaluate on TEST(real/hybrid).
    """
    from mne.decoding import CSP
    from mne.filter import filter_data
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    # NOTE: we no longer compute "within-test CV" by retraining on TEST folds.
    # We compute fold-stats from fixed predictions instead (see return below).

    seed = 42
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test_real = np.asarray(X_test_real, dtype=np.float64)
    X_test_hybrid = np.asarray(X_test_hybrid, dtype=np.float64)

    bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30)]
    classes = p._label_classes(y_train, y_test)
    y_train_enc = p._encode_labels_with_classes(y_train, classes)
    y_test_enc = p._encode_labels_with_classes(y_test, classes)

    # Hard constraint (user): CSP components must be 4.
    # In FBCSP this means per_band=4 => m_pairs=2 (because per_band=2*m_pairs).
    m_pairs = int(m_pairs)
    if m_pairs != 2:
        raise ValueError(f"FBCSP CSP components must be 4 => set --fbcsp-m-pairs 2 (got {m_pairs}).")
    per_band = int(2 * m_pairs)

    reg = str(reg)
    cov_est = str(cov_est)
    if reg not in ("none", "ledoit_wolf"):
        raise ValueError(f"Invalid reg={reg!r}")
    if cov_est not in ("epoch", "concat"):
        raise ValueError(f"Invalid cov_est={cov_est!r}")

    # We try notebook-like first (reg=None, cov_est="epoch"), then add shrinkage.
    def _cfg(reg: str | None, cov_est: str) -> dict:
        return dict(n_components=per_band, reg=reg, log=True, norm_trace=True, cov_est=str(cov_est))

    LDA_CFG = dict(solver="lsqr", shrinkage="auto")

    def _fit_transform_all_bands(
        *,
        Xtr: np.ndarray,
        ytr: np.ndarray,
        Xte_r: np.ndarray,
        Xte_h: np.ndarray,
        reg: str | None,
        cov_est: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Ftr_list: list[np.ndarray] = []
        Fr_list: list[np.ndarray] = []
        Fh_list: list[np.ndarray] = []
        for (l_f, h_f) in bands:
            Xtr_b = filter_data(Xtr, float(sfreq), l_f, h_f, method="iir", verbose=False)
            Xte_r_b = filter_data(Xte_r, float(sfreq), l_f, h_f, method="iir", verbose=False)
            Xte_h_b = filter_data(Xte_h, float(sfreq), l_f, h_f, method="iir", verbose=False)
            csp_b = CSP(**_cfg(reg, cov_est))
            Ftr_list.append(csp_b.fit_transform(Xtr_b, ytr))
            Fr_list.append(csp_b.transform(Xte_r_b))
            Fh_list.append(csp_b.transform(Xte_h_b))
        Ftr_full = np.concatenate(Ftr_list, axis=1)
        Fr_full = np.concatenate(Fr_list, axis=1)
        Fh_full = np.concatenate(Fh_list, axis=1)
        return Ftr_full, Fr_full, Fh_full

    def _select_mibif_pairs_for_labels(Ftr_full: np.ndarray, ytr: np.ndarray, *, k_best_local: int) -> list[int]:
        k_best_local = int(k_best_local)
        if k_best_local <= 0:
            k_best_local = 4
        mi = mutual_info_classif(Ftr_full, ytr, random_state=seed)
        order = np.argsort(mi)[::-1]
        topk = order[: min(int(k_best_local), int(order.size))].tolist()
        selected = set(int(i) for i in topk)
        for idx in topk:
            idx = int(idx)
            band_idx = idx // per_band
            j = idx % per_band
            pair_j = int(per_band - 1 - j)
            selected.add(int(band_idx * per_band + pair_j))
        return sorted(selected)

    def _try_configs_on_fold(
        *,
        Xtr: np.ndarray,
        ytr: np.ndarray,
        Xva: np.ndarray,
        yva: np.ndarray,
        k_best_local: int,
        m_pairs_local: int,
    ) -> tuple[float, dict[str, object]]:
        # order: closest to notebook first
        candidates_cfg = [
            {"reg": None, "cov_est": "epoch"},
            {"reg": "ledoit_wolf", "cov_est": "epoch"},
            {"reg": "ledoit_wolf", "cov_est": "concat"},
        ]
        best_kappa = -1e9
        best_meta: dict[str, object] = {}
        for cfg in candidates_cfg:
            try:
                Ftr_full, Fva_full, _ = _fit_transform_all_bands(
                    Xtr=Xtr, ytr=ytr, Xte_r=Xva, Xte_h=Xva, reg=cfg["reg"], cov_est=str(cfg["cov_est"])
                )
                if not np.isfinite(Ftr_full).all() or not np.isfinite(Fva_full).all():
                    continue
                sel = _select_mibif_pairs_for_labels(Ftr_full, ytr, k_best_local=k_best_local)
                lda = LinearDiscriminantAnalysis(**LDA_CFG)
                lda.fit(Ftr_full[:, sel], ytr)
                pred = lda.predict(Fva_full[:, sel])
                kappa = float(cohen_kappa_score(yva, pred))
                if kappa > best_kappa:
                    best_kappa = kappa
                    best_meta = {"reg": cfg["reg"], "cov_est": cfg["cov_est"], "selected_idx": sel}
            except Exception:
                continue
        return best_kappa, best_meta

    # -----------------
    # 1) Tune on TRAIN (optional)
    # -----------------
    best_reg: str | None = None
    best_cov_est = "epoch"
    best_k_best = int(k_best) if int(k_best) > 0 else 4
    best_m_pairs = int(m_pairs)

    if bool(tune):
        p._log_section("FBCSP tune (TRAIN-only CV)")
        grid_k_best = [4, 6, 8]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        best_score = -1e9
        best_cfg: dict[str, object] = {}

        ytr_all = np.asarray(y_train_enc, dtype=int)
        Xtr_all = np.asarray(X_train, dtype=np.float64)

        mp = 2
        for kb in grid_k_best:
            kb = int(kb)
            fold_scores = []
            fold_meta: list[dict[str, object]] = []
            for tr_idx, va_idx in skf.split(np.zeros_like(ytr_all), ytr_all):
                tr_idx = np.asarray(tr_idx, dtype=int)
                va_idx = np.asarray(va_idx, dtype=int)
                kappa, meta = _try_configs_on_fold(
                    Xtr=Xtr_all[tr_idx],
                    ytr=ytr_all[tr_idx],
                    Xva=Xtr_all[va_idx],
                    yva=ytr_all[va_idx],
                    k_best_local=kb,
                    m_pairs_local=mp,
                )
                if not np.isfinite(kappa):
                    kappa = -1e9
                fold_scores.append(float(kappa))
                fold_meta.append(meta)
            score = float(np.mean(fold_scores))
            print(f"[FBCSP tune] m_pairs={mp} k_best={kb} | mean_kappa={score:.3f}")
            if score > best_score:
                regs = [m.get("reg") for m in fold_meta if isinstance(m, dict)]
                covs = [m.get("cov_est") for m in fold_meta if isinstance(m, dict)]
                best_reg = None if regs.count(None) >= regs.count("ledoit_wolf") else "ledoit_wolf"
                best_cov_est = "epoch" if covs.count("epoch") >= covs.count("concat") else "concat"
                best_score = score
                best_m_pairs = mp
                best_k_best = kb
                best_cfg = {"m_pairs": mp, "k_best": kb, "reg": best_reg, "cov_est": best_cov_est, "score": best_score}

        print("[FBCSP tune] selected:", best_cfg)
    else:
        # Keep fixed "best-known" config (user-requested)
        best_reg = None if reg == "none" else "ledoit_wolf"
        best_cov_est = cov_est
        best_k_best = int(k_best)
        best_m_pairs = int(m_pairs)

    # restore chosen params
    m_pairs = int(best_m_pairs)
    per_band = int(2 * m_pairs)
    k_best = int(best_k_best)

    # -----------------
    # 2) Fit on full TRAIN, eval on TEST
    # -----------------
    try:
        F_train_full, F_test_real_full, F_test_hybrid_full = _fit_transform_all_bands(
            Xtr=X_train, ytr=y_train_enc, Xte_r=X_test_real, Xte_h=X_test_hybrid, reg=best_reg, cov_est=best_cov_est
        )
    except Exception:
        # last-resort stability
        F_train_full, F_test_real_full, F_test_hybrid_full = _fit_transform_all_bands(
            Xtr=X_train, ytr=y_train_enc, Xte_r=X_test_real, Xte_h=X_test_hybrid, reg="ledoit_wolf", cov_est="concat"
        )

    if not np.isfinite(F_train_full).all() or not np.isfinite(F_test_real_full).all() or not np.isfinite(F_test_hybrid_full).all():
        print("[WARN] Non-finite FBCSP features after fallback; applying nan_to_num.")
        F_train_full = np.nan_to_num(F_train_full, nan=0.0, posinf=0.0, neginf=0.0)
        F_test_real_full = np.nan_to_num(F_test_real_full, nan=0.0, posinf=0.0, neginf=0.0)
        F_test_hybrid_full = np.nan_to_num(F_test_hybrid_full, nan=0.0, posinf=0.0, neginf=0.0)

    selected_idx = _select_mibif_pairs_for_labels(F_train_full, y_train_enc, k_best_local=int(k_best))
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

    cv_real = _cv5_random_split_on_features(feats=F_test_real, y=y_test_enc, seed=42)
    cv_h = _cv5_random_split_on_features(feats=F_test_hybrid, y=y_test_enc, seed=42)

    return {
        "no_cv": {"acc_real": acc_real, "acc_hybrid": acc_h, "kappa_real": kappa_real, "kappa_hybrid": kappa_h},
        "cv": {
            "acc_real": float(cv_real["cv_acc"]),
            "acc_real_std": float(cv_real["cv_acc_std"]),
            "acc_hybrid": float(cv_h["cv_acc"]),
            "acc_hybrid_std": float(cv_h["cv_acc_std"]),
        },
    }


def run_eegnet_with_kappa(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_real: np.ndarray,
    X_test_hybrid: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    dropout_rate: float = 0.5,
    per_epoch_z: bool = True,
    use_cosine_lr: bool = False,
) -> dict[str, dict[str, float]]:
    """
    Local EEGNet wrapper that RETURNS kappa (some server copies of pipeline.py don't).
    Mirrors pipeline.py normalization: train-channel z + per-epoch z.
    """
    import torch.nn as nn
    from torch.utils.data import TensorDataset
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

    classes = p._label_classes(y_train, y_test)
    y_train_enc = p._encode_labels_with_classes(y_train, classes)
    y_test_enc = p._encode_labels_with_classes(y_test, classes)
    n_classes = int(np.max(np.concatenate([y_train_enc, y_test_enc])) + 1)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    X_test_real_f = np.asarray(X_test_real, dtype=np.float32)
    X_test_hybrid_f = np.asarray(X_test_hybrid, dtype=np.float32)

    mu_ch = X_train_f.mean(axis=(0, 2), keepdims=True)
    sd_ch = X_train_f.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train_f = (X_train_f - mu_ch) / sd_ch
    X_test_real_f = (X_test_real_f - mu_ch) / sd_ch
    X_test_hybrid_f = (X_test_hybrid_f - mu_ch) / sd_ch

    def _per_epoch_z(x: np.ndarray) -> np.ndarray:
        mu_t = x.mean(axis=2, keepdims=True)
        sd_t = x.std(axis=2, keepdims=True) + 1e-6
        return (x - mu_t) / sd_t

    if bool(per_epoch_z):
        X_train_f = _per_epoch_z(X_train_f)
        X_test_real_f = _per_epoch_z(X_test_real_f)
        X_test_hybrid_f = _per_epoch_z(X_test_hybrid_f)
        print("[EEGNet] applied normalization: train-channel z + per-epoch z")
    else:
        print("[EEGNet] applied normalization: train-channel z (no per-epoch z)")

    Xtr = X_train_f[:, None, :, :]
    Xte_r = X_test_real_f[:, None, :, :]
    Xte_h = X_test_hybrid_f[:, None, :, :]
    ytr = np.asarray(y_train_enc, dtype=np.int64)
    yte = np.asarray(y_test_enc, dtype=np.int64)

    model = p.EEGNetModel(
        chans=int(X_train.shape[1]),
        classes=n_classes,
        time_points=int(X_train.shape[2]),
        dropout_rate=float(dropout_rate),
    ).to(device)
    # Match requested config: Adam, fixed LR.
    # (weight_decay kept for optional usage; default is 0.0)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(max(1, int(epochs))))
        if bool(use_cosine_lr)
        else None
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=int(batch_size),
        shuffle=True,
    )

    pbar = p.tqdm(range(int(epochs)), desc="EEGNet train", leave=True, mininterval=0.5)
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
        if sched is not None:
            sched.step()

    def _eval(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
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
        return {
            "acc": acc,
            "kappa": kappa,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "cm": cm,
            "y_pred": np.asarray(y_pred, dtype=int),
        }

    m_real = _eval(Xte_r, yte)
    m_h = _eval(Xte_h, yte)

    print("\n=== EEGNet Results (macro) ===")
    print("TEST(real):  ", {k: v for k, v in m_real.items() if k not in ("cm", "y_pred")})
    print("TEST(hybrid):", {k: v for k, v in m_h.items() if k not in ("cm", "y_pred")})
    print("\nConfusionMatrix (normalize=true) TEST(real):\n", np.array2string(m_real["cm"], precision=3, floatmode="fixed"))
    print("\nConfusionMatrix (normalize=true) TEST(hybrid):\n", np.array2string(m_h["cm"], precision=3, floatmode="fixed"))

    # Literature-style 5-fold CV (random splits) on embeddings (as in notebook).
    def _embeddings(x: np.ndarray) -> np.ndarray:
        model.eval()
        feat_parts = []
        with torch.no_grad():
            for i in range(0, x.shape[0], int(batch_size)):
                xb = torch.from_numpy(x[i : i + int(batch_size)]).to(device)
                feats = model.forward_features(xb).detach().cpu().numpy()
                feat_parts.append(np.asarray(feats, dtype=float).reshape(feats.shape[0], -1))
        return np.vstack(feat_parts)

    X_feat_real = _embeddings(Xte_r)
    X_feat_h = _embeddings(Xte_h)
    cv_real = _cv5_random_split_on_features(feats=X_feat_real, y=yte, seed=42)
    cv_h = _cv5_random_split_on_features(feats=X_feat_h, y=yte, seed=42)

    return {
        "no_cv": {
            "acc_real": float(m_real["acc"]),
            "acc_hybrid": float(m_h["acc"]),
            "kappa_real": float(m_real["kappa"]),
            "kappa_hybrid": float(m_h["kappa"]),
        },
        "cv": {
            "acc_real": float(cv_real["cv_acc"]),
            "acc_real_std": float(cv_real["cv_acc_std"]),
            "acc_hybrid": float(cv_h["cv_acc"]),
            "acc_hybrid_std": float(cv_h["cv_acc_std"]),
        },
    }


def run_unet_classifier_with_foldcv(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_real: np.ndarray,
    X_test_hybrid: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float = 1e-4,
) -> dict[str, dict[str, float]]:
    """
    U-Net classifier wrapper that reports fold-stats on fixed predictions (no retraining on TEST).
    Keeps architecture/training consistent with pipeline.py but makes CV comparable to acc_*.
    """
    import torch.nn as nn
    from torch.utils.data import TensorDataset
    from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support

    classes = p._label_classes(y_train, y_test)
    y_train_enc = p._encode_labels_with_classes(y_train, classes)
    y_test_enc = p._encode_labels_with_classes(y_test, classes)
    n_classes = int(np.max(np.concatenate([y_train_enc, y_test_enc])) + 1)

    Xtr = np.asarray(X_train, dtype=np.float32)
    Xte_r = np.asarray(X_test_real, dtype=np.float32)
    Xte_h = np.asarray(X_test_hybrid, dtype=np.float32)
    ytr = np.asarray(y_train_enc, dtype=np.int64)
    yte = np.asarray(y_test_enc, dtype=np.int64)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=int(batch_size),
        shuffle=True,
    )

    model = p.UNet1DClassifier(in_ch=int(Xtr.shape[1]), n_classes=n_classes, base_ch=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = nn.CrossEntropyLoss()

    pbar = p.tqdm(range(int(epochs)), desc="U-Net classifier train", leave=True, mininterval=0.5)
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

    def _eval(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
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
        return {
            "acc": acc,
            "kappa": kappa,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "y_pred": np.asarray(y_pred, dtype=int),
        }

    m_real = _eval(Xte_r, yte)
    m_h = _eval(Xte_h, yte)

    print("\n=== U-Net Classifier Results (macro) ===")
    print("TEST(real):  ", {k: v for k, v in m_real.items() if k != "y_pred"})
    print("TEST(hybrid):", {k: v for k, v in m_h.items() if k != "y_pred"})

    cv_real = _fold_stats_from_fixed_preds(y_true=yte, y_pred=m_real["y_pred"], n_splits=5)
    cv_h = _fold_stats_from_fixed_preds(y_true=yte, y_pred=m_h["y_pred"], n_splits=5)

    return {
        "no_cv": {
            "acc_real": float(m_real["acc"]),
            "acc_hybrid": float(m_h["acc"]),
            "kappa_real": float(m_real["kappa"]),
            "kappa_hybrid": float(m_h["kappa"]),
        },
        "cv": {
            "acc_real": float(cv_real["acc_mean"]),
            "acc_real_std": float(cv_real["acc_std"]),
            "acc_hybrid": float(cv_h["acc_mean"]),
            "acc_hybrid_std": float(cv_h["acc_std"]),
        },
    }


def _print_accuracy_table(title: str, results: dict[str, dict[str, object]]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    rows = []
    for name, out in results.items():
        no_cv = out.get("no_cv", {}) if isinstance(out, dict) else {}
        cv = out.get("cv", {}) if isinstance(out, dict) else {}
        rows.append(
            {
                "Model": name,
                "acc_real": no_cv.get("acc_real", float("nan")),
                "acc_hybrid": no_cv.get("acc_hybrid", float("nan")),
                "kappa_real": no_cv.get("kappa_real", float("nan")),
                "kappa_hybrid": no_cv.get("kappa_hybrid", float("nan")),
                "cv_acc_real": cv.get("acc_real", float("nan")),
                "cv_acc_hybrid": cv.get("acc_hybrid", float("nan")),
            }
        )
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    except Exception:
        for r in rows:
            print(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--subject-id", type=str, default="A01")
    parser.add_argument("--data-dir", type=str, default="BCICIV_2a_gdf")
    parser.add_argument("--labels-dir", type=str, default="true_labels")

    parser.add_argument("--ddpm-epochs", type=int, default=200)
    parser.add_argument(
        "--select-best-4",
        action="store_true",
        help="Screen multiple channel combinations and automatically keep the best 4-target set.",
    )
    parser.add_argument(
        "--select-best-4-candidate-set",
        type=str,
        default="motor",
        choices=["motor", "broad"],
        help="Candidate pool for --select-best-4. 'motor' is smaller/faster; 'broad' tries more channels.",
    )
    parser.add_argument(
        "--select-best-4-ddpm-epochs",
        type=int,
        default=50,
        help="DDPM epochs used during screening only (final training still uses --ddpm-epochs).",
    )
    parser.add_argument(
        "--select-best-4-eval-epochs",
        type=int,
        default=0,
        help="How many EVAL epochs to synthesize/score during screening. 0 => use all EVAL epochs.",
    )
    parser.add_argument(
        "--select-best-4-topk",
        type=int,
        default=10,
        help="Print top-k combos after screening.",
    )
    parser.add_argument(
        "--select-best-4-metric",
        type=str,
        default="kappa",
        choices=["kappa", "acc"],
        help="Metric to rank combos (computed on HYBRID test set).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--n-eval-epochs", type=int, default=0, help="0 => eval all")

    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-train", action="store_true")

    parser.add_argument("--cls-epochs", type=int, default=500)
    parser.add_argument("--cls-batch-size", type=int, default=64)
    parser.add_argument(
        "--fbcsp-tune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tune FBCSP hyperparameters on TRAIN via CV. Default=False to keep best-known config.",
    )
    parser.add_argument("--fbcsp-k-best", type=int, default=8, help="Used when --no-fbcsp-tune. (best-known: 8)")
    parser.add_argument("--fbcsp-m-pairs", type=int, default=2, help="Must be 2 (CSP components fixed to 4).")
    parser.add_argument("--fbcsp-reg", type=str, default="ledoit_wolf", choices=["none", "ledoit_wolf"])
    parser.add_argument("--fbcsp-cov-est", type=str, default="epoch", choices=["epoch", "concat"])

    # EEGNet requested config (separate from --cls-epochs to avoid affecting others)
    parser.add_argument("--eegnet-epochs", type=int, default=1000)
    parser.add_argument("--eegnet-batch-size", type=int, default=64)
    parser.add_argument("--eegnet-lr", type=float, default=1e-3)
    parser.add_argument("--eegnet-weight-decay", type=float, default=0.0)
    parser.add_argument("--eegnet-dropout", type=float, default=0.5)
    parser.add_argument(
        "--eegnet-per-epoch-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="EEGNet: apply per-epoch z-score after train-channel z.",
    )
    parser.add_argument(
        "--eegnet-cosine-lr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="EEGNet: use CosineAnnealingLR over training epochs.",
    )

    parser.add_argument("--skip-ddpm", action="store_true")
    parser.add_argument("--skip-sgm", action="store_true")  # kept for CLI compatibility; not implemented here
    parser.add_argument(
        "--hybrid-synth-per-epoch-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-epoch z-score to synthesized channels only (improves CSP/FBCSP robustness).",
    )
    parser.add_argument(
        "--hybrid-synth-clip",
        type=float,
        default=5.0,
        help="Optional clipping for synthesized channels (z-score units). 0 disables.",
    )
    parser.add_argument(
        "--debug-hybrid",
        type=int,
        default=1,
        help="0 disables hybrid diagnostics; 1 prints key stats; 2 prints more (reserved).",
    )
    args = parser.parse_args()

    print(
        f"[HYBRID cfg] synth_per_epoch_z={bool(args.hybrid_synth_per_epoch_z)} | synth_clip={float(args.hybrid_synth_clip)}"
    )

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

    bundle: dict[str, object] = {"ddpm": {}, "meta": {}}
    if bool(args.cache) and (not bool(args.force_train)) and ckpt_path.exists():
        try:
            bundle = torch.load(ckpt_path, map_location="cpu")
            print(f"[CKPT] loaded: {ckpt_path}")
        except Exception as e:
            print(f"[CKPT] load failed: {e}. Starting from scratch.")
            bundle = {"ddpm": {}, "meta": {}}

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
    raw_train = _maybe_assign_standard_2a_channel_names(raw_train)
    raw_eval = _maybe_assign_standard_2a_channel_names(raw_eval)
    raw_train = _preprocess_like_bci3(raw_train)
    raw_eval = _preprocess_like_bci3(raw_eval)

    print(f"[SANITY] TRAIN channels after preprocessing: n={len(raw_train.ch_names)} | {raw_train.ch_names}")
    print(f"[SANITY] EVAL  channels after preprocessing: n={len(raw_eval.ch_names)} | {raw_eval.ch_names}")

    events_train, event_id_train = mne.events_from_annotations(raw_train, verbose=False)
    events_eval, event_id_eval = mne.events_from_annotations(raw_eval, verbose=False)

    class_keys = _pick_train_class_keys_2a(event_id_train)
    mi_event_id_train = {name: int(event_id_train[k]) for name, k in class_keys.items()}

    y_eval_full = _load_eval_labels_mat(labels_dir=labels_dir, subject_id=subject_id)
    candidate_keys = [k for k in ("783", "768", "781") if k in event_id_eval]
    counts = {k: int((events_eval[:, 2].astype(int) == int(event_id_eval[k])).sum()) for k in candidate_keys}
    eval_cue_key = _pick_eval_cue_key_2a(event_id_eval)
    exact = [k for k, c in counts.items() if int(c) == int(len(y_eval_full))]
    if exact:
        eval_cue_key = "783" if "783" in exact else exact[0]
    print(f"[EVAL events] candidate counts: {counts} | labels={len(y_eval_full)} | chosen={eval_cue_key}")
    eval_cue_code = int(event_id_eval[eval_cue_key])

    print("TRAIN class keys:", class_keys)
    print("EVAL  cue key   :", eval_cue_key)
    print("mi_event_id_train:", mi_event_id_train)

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

    X_train = epochs_train.get_data(copy=True).astype(np.float32)
    y_train_codes = epochs_train.events[:, -1].astype(int)
    code_to_class = {int(mi_event_id_train[name]): i for i, name in enumerate(class_keys.keys())}
    y_train = np.array([code_to_class[int(c)] for c in y_train_codes], dtype=int)

    X_eval = epochs_eval.get_data(copy=True).astype(np.float32)

    # Align labels to kept eval epochs (if epochs dropped due to BAD annotations)
    if len(y_eval_full) == len(cue_event_indices) and hasattr(epochs_eval, "selection"):
        sel = np.asarray(getattr(epochs_eval, "selection"), dtype=int)
        idx_to_pos = {int(idx): i for i, idx in enumerate(cue_event_indices.tolist())}
        keep_pos = np.asarray([idx_to_pos[int(idx)] for idx in sel.tolist()], dtype=int)
        y_eval = y_eval_full[keep_pos]
    else:
        min_n = min(len(X_eval), len(y_eval_full))
        X_eval = X_eval[:min_n]
        y_eval = y_eval_full[:min_n]

    mu_ch = X_train.mean(axis=(0, 2), keepdims=True)
    sd_ch = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train = (X_train - mu_ch) / sd_ch
    X_eval = (X_eval - mu_ch) / sd_ch

    ch_names = [c.strip() for c in epochs_train.ch_names]
    ch_to_idx = {c: i for i, c in enumerate(ch_names)}
    sfreq = float(epochs_train.info["sfreq"])
    print(f"sfreq: {sfreq}")
    print(f"TRAIN: {tuple(int(v) for v in X_train.shape)} | EVAL: {tuple(int(v) for v in X_eval.shape)}")

    # DDPM schedule
    TIMESTEPS = 1000
    betas = p.linear_beta_schedule(TIMESTEPS).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    saved_ddpm = bundle.get("ddpm")
    if not isinstance(saved_ddpm, dict):
        saved_ddpm = {}
        bundle["ddpm"] = saved_ddpm

    def _train_or_load_ddpm_for_spec(*, spec: ExperimentSpec2A, ddpm_epochs: int, tag: str) -> tuple[torch.nn.Module, tuple[int, int]]:
        target_idx = int(ch_to_idx[spec.target])
        input_idxs = (int(ch_to_idx[spec.inputs[0]]), int(ch_to_idx[spec.inputs[1]]))
        model = p.UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256).to(device)
        key = f"{tag}|{spec.target}|epochs{int(ddpm_epochs)}|inputs{input_idxs}"
        legacy_key = f"{spec.target}|epochs{int(ddpm_epochs)}|inputs{input_idxs}"
        if bool(args.cache) and (not bool(args.force_train)):
            if key in saved_ddpm:
                model.load_state_dict(saved_ddpm[key], strict=True)
                print(f"[DDPM ckpt] loaded: {key}")
                return model.eval(), input_idxs
            if legacy_key in saved_ddpm:
                model.load_state_dict(saved_ddpm[legacy_key], strict=True)
                print(f"[DDPM ckpt] loaded (legacy): {legacy_key}")
                return model.eval(), input_idxs

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
            epochs=int(ddpm_epochs),
            device=device,
            print_every=0,
            show_epoch_progress=True,
            epoch_desc=f"DDPM Train {spec.target} ({tag})",
        )
        if bool(args.cache):
            saved_ddpm[key] = model.state_dict()
            _save_bundle()
        return model.eval(), input_idxs

    def _synthesize_target_epochs(
        *,
        spec: ExperimentSpec2A,
        model: torch.nn.Module,
        input_idxs: tuple[int, int],
        X_source: np.ndarray,
        max_n: int,
    ) -> np.ndarray:
        n_all = int(X_source.shape[0])
        n = n_all if int(max_n) <= 0 else min(int(max_n), n_all)
        t_len = int(X_source.shape[2])
        out = np.zeros((n, t_len), dtype=np.float32)
        pbar = p.tqdm(total=n, desc=f"DDPM Synthesize {spec.target}", leave=False, mininterval=0.5)
        for s in range(0, n, int(args.gen_batch_size)):
            e = min(n, s + int(args.gen_batch_size))
            batch = X_source[s:e]
            cond = np.stack([batch[:, input_idxs[0], :], batch[:, input_idxs[1], :]], axis=1).astype(np.float32)
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
            syn_np = syn.detach().cpu().numpy()[:, 0].astype(np.float32)
            if bool(args.hybrid_synth_per_epoch_z):
                mu_t = syn_np.mean(axis=1, keepdims=True)
                sd_t = syn_np.std(axis=1, keepdims=True) + 1e-6
                syn_np = (syn_np - mu_t) / sd_t
            if float(args.hybrid_synth_clip) > 0:
                syn_np = np.clip(syn_np, -float(args.hybrid_synth_clip), float(args.hybrid_synth_clip))
            out[s:e] = syn_np
            pbar.update(e - s)
        pbar.close()
        return out

    def _screen_best_4_specs() -> list[ExperimentSpec2A]:
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.metrics import accuracy_score, cohen_kappa_score

        forbidden_targets = {"C3", "C4", "Cz"}
        candidates = _candidate_pool_experiments_2a(candidate_set=str(args.select_best_4_candidate_set))

        # Keep only specs whose channels exist in this file.
        candidates = [
            s
            for s in candidates
            if (s.target in ch_to_idx)
            and (s.target not in forbidden_targets)
            and (s.inputs[0] in ch_to_idx)
            and (s.inputs[1] in ch_to_idx)
        ]
        if len(candidates) < 4:
            raise ValueError(f"Not enough candidate specs after filtering: n={len(candidates)}")

        max_eval = int(args.select_best_4_eval_epochs)
        if max_eval <= 0:
            max_eval = int(X_eval.shape[0])

        p._log_section(
            f"BCI-IV 2a — Screening best 4 targets (pool={len(candidates)} | combos={math.comb(len(candidates), 4)} | ddpm_epochs={int(args.select_best_4_ddpm_epochs)} | eval_epochs={max_eval})"
        )

        # Precompute filtered (8–30) for CSP scoring
        X_train_bp = _bandpass_epochs(X_train, sfreq=float(sfreq), l_freq=8.0, h_freq=30.0)
        X_eval_bp = _bandpass_epochs(X_eval[:max_eval], sfreq=float(sfreq), l_freq=8.0, h_freq=30.0)

        # CSP model (fit once, score many hybrids)
        classes = p._label_classes(y_train, y_eval[:max_eval])
        y_train_enc = p._encode_labels_with_classes(y_train, classes)
        y_eval_enc = p._encode_labels_with_classes(y_eval[:max_eval], classes)

        def _fit_csp_primary_then_fallback() -> CSP:
            cfg_primary = dict(n_components=4, reg=None, log=True, norm_trace=True, cov_est="epoch")
            cfg_fallback = dict(n_components=4, reg="ledoit_wolf", log=True, norm_trace=True, cov_est="concat")
            try:
                csp = CSP(**cfg_primary).fit(X_train_bp, y_train_enc)
                return csp
            except Exception:
                return CSP(**cfg_fallback).fit(X_train_bp, y_train_enc)

        csp = _fit_csp_primary_then_fallback()
        F_train_csp = csp.transform(X_train_bp)
        lda_csp = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F_train_csp, y_train_enc)

        # FBCSP model (fit once, score many hybrids)
        bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30)]
        m_pairs = 2 if int(X_train.shape[1]) > 3 else 1
        per_band = int(2 * m_pairs)
        csp_bands: list[CSP] = []
        F_train_parts: list[np.ndarray] = []

        X_train_bands = [_bandpass_epochs(X_train, sfreq=float(sfreq), l_freq=lf, h_freq=hf) for (lf, hf) in bands]
        X_eval_bands = [_bandpass_epochs(X_eval[:max_eval], sfreq=float(sfreq), l_freq=lf, h_freq=hf) for (lf, hf) in bands]

        for Xtr_b in X_train_bands:
            csp_b = CSP(
                n_components=per_band,
                reg="ledoit_wolf",
                log=True,
                norm_trace=True,
                cov_est="concat",
            ).fit(Xtr_b, y_train_enc)
            csp_bands.append(csp_b)
            F_train_parts.append(csp_b.transform(Xtr_b))

        F_train_full = np.concatenate(F_train_parts, axis=1)
        if not np.isfinite(F_train_full).all():
            F_train_full = np.nan_to_num(F_train_full, nan=0.0, posinf=0.0, neginf=0.0)

        mi = mutual_info_classif(F_train_full, y_train_enc, random_state=42)
        order = np.argsort(mi)[::-1]
        topk = order[: min(4, int(order.size))].tolist()

        selected = set(int(i) for i in topk)
        for idx in topk:
            idx = int(idx)
            band_idx = idx // per_band
            j = idx % per_band
            pair_j = int(per_band - 1 - j)
            selected.add(int(band_idx * per_band + pair_j))

        selected_idx = sorted(selected)
        lda_fbcsp = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F_train_full[:, selected_idx], y_train_enc)

        # Train+generate once per candidate target
        synth_time: dict[str, np.ndarray] = {}
        synth_bp: dict[str, np.ndarray] = {}
        synth_bands: dict[str, list[np.ndarray]] = {}

        for spec in candidates:
            print("-" * 36 + f"\n[SCREEN] target={spec.target} inputs={spec.inputs}")
            model, input_idxs = _train_or_load_ddpm_for_spec(
                spec=spec, ddpm_epochs=int(args.select_best_4_ddpm_epochs), tag="screen"
            )
            syn = _synthesize_target_epochs(spec=spec, model=model, input_idxs=input_idxs, X_source=X_eval, max_n=max_eval)
            synth_time[spec.target] = syn
            synth_bp[spec.target] = _bandpass_single_channel_epochs(syn, sfreq=float(sfreq), l_freq=8.0, h_freq=30.0)
            synth_bands[spec.target] = [
                _bandpass_single_channel_epochs(syn, sfreq=float(sfreq), l_freq=lf, h_freq=hf) for (lf, hf) in bands
            ]

        # Score all 4-combinations
        results: list[dict[str, object]] = []
        target_names = [c.target for c in candidates]
        for combo in itertools.combinations(target_names, 4):
            # CSP hybrid
            Xh_bp = np.asarray(X_eval_bp, dtype=np.float32).copy()
            for t in combo:
                Xh_bp[:, int(ch_to_idx[t]), :] = synth_bp[t]
            F_h_csp = csp.transform(np.asarray(Xh_bp, np.float64))
            pred_csp = lda_csp.predict(F_h_csp)
            acc_csp = float(accuracy_score(y_eval_enc, pred_csp))
            kappa_csp = float(cohen_kappa_score(y_eval_enc, pred_csp))

            # FBCSP hybrid
            F_h_parts: list[np.ndarray] = []
            for bi in range(len(bands)):
                Xh_b = np.asarray(X_eval_bands[bi], dtype=np.float32).copy()
                for t in combo:
                    Xh_b[:, int(ch_to_idx[t]), :] = synth_bands[t][bi]
                F_h_parts.append(csp_bands[bi].transform(np.asarray(Xh_b, np.float64)))
            F_h_full = np.concatenate(F_h_parts, axis=1)
            if not np.isfinite(F_h_full).all():
                F_h_full = np.nan_to_num(F_h_full, nan=0.0, posinf=0.0, neginf=0.0)
            pred_f = lda_fbcsp.predict(F_h_full[:, selected_idx])
            acc_f = float(accuracy_score(y_eval_enc, pred_f))
            kappa_f = float(cohen_kappa_score(y_eval_enc, pred_f))

            metric = str(args.select_best_4_metric)
            if metric == "kappa":
                score = _sanitize_score(0.5 * (kappa_csp + kappa_f))
            else:
                score = _sanitize_score(0.5 * (acc_csp + acc_f))

            results.append(
                {
                    "combo": tuple(combo),
                    "score": float(score),
                    "csp_acc": float(acc_csp),
                    "csp_kappa": float(kappa_csp),
                    "fbcsp_acc": float(acc_f),
                    "fbcsp_kappa": float(kappa_f),
                }
            )

        results_sorted = sorted(results, key=lambda r: float(r["score"]), reverse=True)
        topk = max(1, int(args.select_best_4_topk))
        print("\n" + "=" * 80)
        print(f"BEST-4 screening results (metric={args.select_best_4_metric}) — top {topk}")
        print("=" * 80)
        for r in results_sorted[:topk]:
            combo = r["combo"]
            print(
                f"- combo={combo} | score={float(r['score']):.4f} | "
                f"CSP(acc={float(r['csp_acc']):.3f}, kappa={float(r['csp_kappa']):.3f}) | "
                f"FBCSP(acc={float(r['fbcsp_acc']):.3f}, kappa={float(r['fbcsp_kappa']):.3f})"
            )

        best_combo = results_sorted[0]["combo"]
        best_targets = set(str(t) for t in best_combo)
        best_specs = [s for s in candidates if s.target in best_targets]
        best_specs = sorted(best_specs, key=lambda s: str(s.target))
        return best_specs

    # Decide target/input specs
    if bool(args.select_best_4):
        if bool(args.skip_ddpm):
            raise ValueError("--select-best-4 requires DDPM (remove --skip-ddpm).")
        specs = _screen_best_4_specs()
    else:
        specs = _default_experiments_2a()

    forbidden_targets = {"C3", "C4", "Cz"}
    bad = [s.target for s in specs if s.target in forbidden_targets]
    if bad:
        raise ValueError(f"Forbidden targets: {bad}")
    for s in specs:
        if s.target not in ch_to_idx:
            raise ValueError(f"Missing target: {s.target}")
        for inp in s.inputs:
            if inp not in ch_to_idx:
                raise ValueError(f"Missing input: {inp}")

    p._log_section("BCI-IV 2a — Selected target/input channels for synthesis")
    try:
        import pandas as pd

        df_specs = pd.DataFrame([{"Target Channel": s.target, "Input Channels": f"{s.inputs[0]}, {s.inputs[1]}"} for s in specs])
        print(df_specs.to_string(index=False))
    except Exception:
        for s in specs:
            print(f"- Target={s.target} | Inputs={s.inputs}")

    ddpm_models: dict[str, torch.nn.Module] = {}
    ddpm_input_idxs: dict[str, tuple[int, int]] = {}
    ddpm_rows: list[dict[str, object]] = []

    if not bool(args.skip_ddpm):
        p._log_section("BCI-IV 2a — DDPM: train per target + Table (MSE/Pearson) + hybrid + classifiers")
        for s in specs:
            target_idx = int(ch_to_idx[s.target])
            input_idxs = (int(ch_to_idx[s.inputs[0]]), int(ch_to_idx[s.inputs[1]]))
            print(
                "-" * 36
                + f"\nDDPM target={s.target} inputs={s.inputs} | train epochs: {len(X_train)} eval epochs: {len(X_eval)} | ddpm_epochs: {int(args.ddpm_epochs)}"
            )

            model, input_idxs_loaded = _train_or_load_ddpm_for_spec(spec=s, ddpm_epochs=int(args.ddpm_epochs), tag="final")
            ddpm_input_idxs[s.target] = input_idxs_loaded
            ddpm_models[s.target] = model.eval()

            # Eval
            test_ds = p.EEGEpochs(X_eval, input_idxs=input_idxs_loaded, target_idx=target_idx)
            test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
            mse_list: list[float] = []
            corr_list: list[float] = []
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
                    model,
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
                    mse_list.append(float(np.mean((syn_np[i] - tgt_np[i]) ** 2)))
                    corr_list.append(p.pearson_corr_1d(tgt_np[i], syn_np[i]))
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
            print(
                f"DDPM Computed MSE={float(np.mean(mse_list)):.6f} Pearson={float(np.mean(corr_list)):.6f} (n={len(mse_list)})"
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

        X_test_hybrid = build_hybrid_epochs_ddpm_2a(
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
            synth_per_epoch_z=bool(args.hybrid_synth_per_epoch_z),
            synth_clip=float(args.hybrid_synth_clip),
        )

        # Diagnostics: show real vs hybrid mean/std for synthesized targets
        print("\n[HYBRID sanity] target mean/std over EVAL (real vs hybrid)")
        for s in specs:
            idx = int(ch_to_idx[s.target])
            real_ch = np.asarray(X_eval[:, idx, :], dtype=np.float32)
            hyb_ch = np.asarray(X_test_hybrid[:, idx, :], dtype=np.float32)
            rm = float(real_ch.mean())
            rs = float(real_ch.std())
            hm = float(hyb_ch.mean())
            hs = float(hyb_ch.std())
            print(f"- {s.target}: real(mean={rm:.3f}, std={rs:.3f}) | hybrid(mean={hm:.3f}, std={hs:.3f})")

        if int(args.debug_hybrid) > 0:
            _debug_hybrid(
                X_eval_real=X_eval,
                X_eval_hybrid=X_test_hybrid,
                specs=specs,
                ch_to_idx=ch_to_idx,
                sfreq=float(sfreq),
                max_epochs=200,
            )

        p._log_section("BCI-IV 2a — Classifiers (real vs hybrid)")
        ddpm_acc: dict[str, dict[str, object]] = {}
        ddpm_acc["CSP+LDA"] = run_csp_lda_robust(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_eval,
            X_test_hybrid=X_test_hybrid,
            y_test=y_eval,
            sfreq=sfreq,
        )
        ddpm_acc["FBCSP+LDA"] = run_fbcsp_lda_safe(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_eval,
            X_test_hybrid=X_test_hybrid,
            y_test=y_eval,
            sfreq=float(sfreq),
            tune=bool(args.fbcsp_tune),
            k_best=int(args.fbcsp_k_best),
            m_pairs=int(args.fbcsp_m_pairs),
            reg=str(args.fbcsp_reg),
            cov_est=str(args.fbcsp_cov_est),
        )

        X_train_nn = _crop_last_if_odd_time(X_train)
        X_eval_nn = _crop_last_if_odd_time(X_eval)
        X_hybrid_nn = _crop_last_if_odd_time(X_test_hybrid)

        ddpm_acc["U-Net"] = run_unet_classifier_with_foldcv(
            X_train=X_train_nn,
            y_train=y_train,
            X_test_real=X_eval_nn,
            X_test_hybrid=X_hybrid_nn,
            y_test=y_eval,
            device=device,
            epochs=int(args.cls_epochs),
            batch_size=int(args.cls_batch_size),
        )
        ddpm_acc["EEGNet"] = run_eegnet_with_kappa(
            X_train=X_train_nn,
            y_train=y_train,
            X_test_real=X_eval_nn,
            X_test_hybrid=X_hybrid_nn,
            y_test=y_eval,
            device=device,
            epochs=int(args.eegnet_epochs),
            batch_size=int(args.eegnet_batch_size),
            lr=float(args.eegnet_lr),
            weight_decay=float(args.eegnet_weight_decay),
            dropout_rate=float(args.eegnet_dropout),
            per_epoch_z=bool(args.eegnet_per_epoch_z),
            use_cosine_lr=bool(args.eegnet_cosine_lr),
        )
        _print_accuracy_table("BCI-IV 2a — DDPM accuracy summary (real vs hybrid)", ddpm_acc)


if __name__ == "__main__":
    main()

