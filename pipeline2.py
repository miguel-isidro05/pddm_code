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
    Default targets/inputs requested by user:
      C1, C2, Fz, POz, P1, P2
    """
    return [
        ExperimentSpec2A("C1", ("C3", "Cz")),
        ExperimentSpec2A("C2", ("C4", "Cz")),
        ExperimentSpec2A("Fz", ("FCz", "FC1")),
        ExperimentSpec2A("POz", ("Pz", "CPz")),
        ExperimentSpec2A("P1", ("CP1", "Pz")),
        ExperimentSpec2A("P2", ("CP2", "Pz")),
    ]


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
    from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score

    # Match notebook: bandpass before CSP
    X_train_bp = filter_data(np.asarray(X_train, np.float64), float(sfreq), 8.0, 30.0, method="iir", verbose=False)
    X_test_real_bp = filter_data(np.asarray(X_test_real, np.float64), float(sfreq), 8.0, 30.0, method="iir", verbose=False)
    X_test_hybrid_bp = filter_data(
        np.asarray(X_test_hybrid, np.float64), float(sfreq), 8.0, 30.0, method="iir", verbose=False
    )

    CSP_CFG_PRIMARY = dict(n_components=4, reg=None, log=True, norm_trace=True, cov_est="epoch")
    CSP_CFG_FALLBACK = dict(n_components=4, reg="ledoit_wolf", log=True, norm_trace=True, cov_est="concat")
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--n-eval-epochs", type=int, default=0, help="0 => eval all")

    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-train", action="store_true")

    parser.add_argument("--cls-epochs", type=int, default=500)
    parser.add_argument("--cls-batch-size", type=int, default=64)

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

    # DDPM schedule
    TIMESTEPS = 1000
    betas = p.linear_beta_schedule(TIMESTEPS).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    ddpm_models: dict[str, torch.nn.Module] = {}
    ddpm_input_idxs: dict[str, tuple[int, int]] = {}
    ddpm_rows: list[dict[str, object]] = []

    saved_ddpm = bundle.get("ddpm")
    if not isinstance(saved_ddpm, dict):
        saved_ddpm = {}
        bundle["ddpm"] = saved_ddpm

    if not bool(args.skip_ddpm):
        p._log_section("BCI-IV 2a — DDPM: train per target + Table (MSE/Pearson) + hybrid + classifiers")
        for s in specs:
            target_idx = int(ch_to_idx[s.target])
            input_idxs = (int(ch_to_idx[s.inputs[0]]), int(ch_to_idx[s.inputs[1]]))
            ddpm_input_idxs[s.target] = input_idxs
            print(
                "-" * 36
                + f"\nDDPM target={s.target} inputs={s.inputs} | train epochs: {len(X_train)} eval epochs: {len(X_eval)} | ddpm_epochs: {int(args.ddpm_epochs)}"
            )

            model = p.UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256).to(device)
            key = f"{s.target}|epochs{int(args.ddpm_epochs)}|inputs{input_idxs}"
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
        ddpm_acc["FBCSP+LDA"] = p.run_fbcsp_lda(
            X_train=X_train,
            y_train=y_train,
            X_test_real=X_eval,
            X_test_hybrid=X_test_hybrid,
            y_test=y_eval,
            sfreq=float(sfreq),
        )

        X_train_nn = _crop_last_if_odd_time(X_train)
        X_eval_nn = _crop_last_if_odd_time(X_eval)
        X_hybrid_nn = _crop_last_if_odd_time(X_test_hybrid)

        p.run_unet_classifier._epochs = int(args.cls_epochs)  # type: ignore[attr-defined]
        p.run_unet_classifier._batch_size = int(args.cls_batch_size)  # type: ignore[attr-defined]
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
        _print_accuracy_table("BCI-IV 2a — DDPM accuracy summary (real vs hybrid)", ddpm_acc)


if __name__ == "__main__":
    main()

