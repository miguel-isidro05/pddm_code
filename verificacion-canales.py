"""
Verificación de canales (BCI-IV 2a): DDPM reconstruction quality for a FIXED set of targets.

Qué hace:
- Carga AxxT.gdf (TRAIN) y AxxE.gdf (EVAL)
- Preprocesa similar a pipeline2.py (bandpass 8–30 + notch 50; ICA fit on TRAIN epochs, apply to EVAL)
- Epoching 0–4s post-cue
- Ajusta z-score con stats de TRAIN (anti-leakage)
- Entrena/evalúa DDPM SOLO para estos targets/inputs (misma config que pipeline2.py):
  - C5  <- (FC3, C3)
  - FC3 <- (C1, FC1)
  - FC4 <- (C2, FC2)
  - C6  <- (FC4, C4)
  - P1  <- (Pz, CP1)
  - P2  <- (Pz, CP2)

Salida:
- Tabla con inputs, MSE, Pearson
- (Opcional) guarda CSV
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader

import mne
from mne.preprocessing import ICA

import pipeline as p  # reuse UNet, EEGEpochs, training + sampling helpers
import pipeline2 as p2  # reuse 2a-specific preprocessing helpers


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_eval_labels_mat(*, labels_dir: Path, subject_id: str) -> np.ndarray:
    mat_path = labels_dir / f"{subject_id}E.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing labels file: {mat_path}")
    mat = scipy.io.loadmat(str(mat_path))
    y_raw = np.asarray(mat["classlabel"]).squeeze().astype(int)
    if not set(np.unique(y_raw)).issubset({1, 2, 3, 4}):
        raise ValueError(f"Unexpected classlabel values: {np.unique(y_raw)}")
    return (y_raw - 1).astype(int)


def _align_eval_labels(
    *,
    epochs_eval: mne.Epochs,
    events_eval: np.ndarray,
    eval_cue_code: int,
    y_eval_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align eval labels to kept epochs when annotation-based dropping happens."""
    X_eval = epochs_eval.get_data(copy=True).astype(np.float32)

    cue_event_indices = np.where(events_eval[:, 2].astype(int) == int(eval_cue_code))[0].astype(int)
    if len(y_eval_full) == len(cue_event_indices) and hasattr(epochs_eval, "selection"):
        sel = np.asarray(getattr(epochs_eval, "selection"), dtype=int)
        try:
            idx_to_pos = {int(idx): i for i, idx in enumerate(cue_event_indices.tolist())}
            keep_pos = np.asarray([idx_to_pos[int(idx)] for idx in sel.tolist()], dtype=int)
            y_eval = y_eval_full[keep_pos]
            return X_eval, np.asarray(y_eval, dtype=int)
        except Exception:
            if len(sel) == len(epochs_eval) and int(sel.max(initial=0)) < len(y_eval_full):
                y_eval = y_eval_full[sel]
                return X_eval, np.asarray(y_eval, dtype=int)

    min_n = min(len(X_eval), len(y_eval_full))
    if len(X_eval) != len(y_eval_full):
        print(f"[WARN] EVAL mismatch: epochs={len(X_eval)} labels={len(y_eval_full)}. Using min_n={min_n}")
    return X_eval[:min_n], np.asarray(y_eval_full[:min_n], dtype=int)


FIXED_SPECS: list[tuple[str, tuple[str, str]]] = [
    ("C5", ("FC3", "C3")),
    ("FC3", ("C1", "FC1")),
    ("FC4", ("C2", "FC2")),
    ("C6", ("FC4", "C4")),
    ("P1", ("Pz", "CP1")),
    ("P2", ("Pz", "CP2")),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--subject-id", type=str, default="A01")
    parser.add_argument("--data-dir", type=str, default="BCICIV_2a_gdf")
    parser.add_argument("--labels-dir", type=str, default="true_labels")
    parser.add_argument("--reject-by-annotation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--muscle-annotate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ddpm-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--n-eval-epochs", type=int, default=0, help="0 => evaluate all; if >0, limit n")
    parser.add_argument("--exclude", type=str, default="C3,C4,Cz", help="Comma-separated channels to exclude as targets.")
    parser.add_argument("--sort-by", type=str, default="pearson", choices=["pearson", "mse"])
    parser.add_argument("--csv-out", type=str, default="", help="Optional path to save results CSV.")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Where to store verify checkpoints.")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    _set_seeds(int(args.seed))

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

    exclude_targets = {c.strip() for c in str(args.exclude).split(",") if c.strip()}
    exclude_targets |= {"C3", "C4", "Cz"}

    print(f"[CFG] subject={subject_id} | ddpm_epochs={int(args.ddpm_epochs)} | fixed_specs={len(FIXED_SPECS)}")

    raw_train = mne.io.read_raw_gdf(str(train_path), preload=True, verbose=False)
    raw_eval = mne.io.read_raw_gdf(str(eval_path), preload=True, verbose=False)
    raw_train = p2._maybe_assign_standard_2a_channel_names(raw_train)
    raw_eval = p2._maybe_assign_standard_2a_channel_names(raw_eval)

    # Filter/notch only; keep EOG and defer ICA to epoch-level (fit on TRAIN only).
    raw_train = p2._preprocess_like_bci3(
        raw_train, muscle_annotate=bool(args.muscle_annotate), run_ica=False, keep_eog=True
    )
    raw_eval = p2._preprocess_like_bci3(
        raw_eval, muscle_annotate=bool(args.muscle_annotate), run_ica=False, keep_eog=True
    )

    events_train, event_id_train = mne.events_from_annotations(raw_train, verbose=False)
    events_eval, event_id_eval = mne.events_from_annotations(raw_eval, verbose=False)

    class_keys = p2._pick_train_class_keys_2a(event_id_train)
    mi_event_id_train = {name: int(event_id_train[k]) for name, k in class_keys.items()}
    y_eval_full = _load_eval_labels_mat(labels_dir=labels_dir, subject_id=subject_id)

    # Choose eval cue key matching labels length if possible
    candidate_keys = [k for k in ("783", "768", "781") if k in event_id_eval]
    counts = {k: int((events_eval[:, 2].astype(int) == int(event_id_eval[k])).sum()) for k in candidate_keys}
    eval_cue_key = p2._pick_eval_cue_key_2a(event_id_eval)
    exact = [k for k, c in counts.items() if int(c) == int(len(y_eval_full))]
    if exact:
        eval_cue_key = "783" if "783" in exact else exact[0]
    eval_cue_code = int(event_id_eval[eval_cue_key])
    print(f"[EVAL events] candidate counts: {counts} | labels={len(y_eval_full)} | chosen={eval_cue_key}")

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

    X_train_all = epochs_train.get_data(copy=True).astype(np.float32)
    y_train_codes = epochs_train.events[:, -1].astype(int)
    code_to_class = {int(mi_event_id_train[name]): i for i, name in enumerate(class_keys.keys())}
    y_train = np.array([code_to_class[int(c)] for c in y_train_codes], dtype=int)

    X_eval_all, y_eval = _align_eval_labels(
        epochs_eval=epochs_eval,
        events_eval=events_eval,
        eval_cue_code=int(eval_cue_code),
        y_eval_full=y_eval_full,
    )

    info_all = epochs_train.info.copy()
    sfreq = float(info_all["sfreq"])
    eog_chs = p2._infer_eog_channels(raw_train)
    eog_present = [c for c in eog_chs if c in info_all["ch_names"]]
    picks_eeg = mne.pick_types(info_all, eeg=True, eog=False, stim=False, exclude=[])
    if len(picks_eeg) <= 0:
        raise ValueError("No EEG channels found.")
    eeg_ch_names = [str(info_all["ch_names"][int(i)]) for i in picks_eeg]

    # Epoch-level ICA fit on TRAIN only; apply to TRAIN + EVAL
    ep_tr = mne.EpochsArray(np.asarray(X_train_all, dtype=np.float64), info_all, verbose=False)
    ep_ev = mne.EpochsArray(np.asarray(X_eval_all, dtype=np.float64), info_all, verbose=False)
    n_comp = min(20, len(picks_eeg))
    ica = ICA(n_components=int(n_comp), random_state=97, max_iter=800)
    ica.fit(ep_tr, picks=picks_eeg, verbose=False)
    if eog_present:
        try:
            eog_idx, _ = ica.find_bads_eog(ep_tr, ch_name=eog_present, verbose=False)
            ica.exclude = eog_idx
            print("ICA exclude components (EOG):", list(map(int, eog_idx)))
        except Exception as e:
            print("ICA EOG detection skipped:", e)
    X_train_ica = ica.apply(ep_tr.copy(), verbose=False).get_data(copy=True).astype(np.float32, copy=False)
    X_eval_ica = ica.apply(ep_ev.copy(), verbose=False).get_data(copy=True).astype(np.float32, copy=False)

    # Drop EOG -> EEG-only
    X_train_eeg = X_train_ica[:, picks_eeg, :]
    X_eval_eeg = X_eval_ica[:, picks_eeg, :]

    # z-score using TRAIN stats only
    mu = X_train_eeg.mean(axis=(0, 2), keepdims=True)
    sd = X_train_eeg.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train_z = (X_train_eeg - mu) / sd
    X_eval_z = (X_eval_eeg - mu) / sd

    # DDPM schedule
    TIMESTEPS = 1000
    betas = p.linear_beta_schedule(TIMESTEPS).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Checkpoint bundle (optional)
    ckpt_dir = Path(str(args.ckpt_dir)).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"BCI4_verify_channels_{subject_id}.pth"
    bundle: dict[str, object] = {"ddpm": {}, "meta": {}}
    if bool(args.cache) and (not bool(args.force_train)) and ckpt_path.exists():
        try:
            bundle = torch.load(ckpt_path, map_location="cpu")
            print(f"[CKPT] loaded: {ckpt_path}")
        except Exception as e:
            print(f"[CKPT] load failed: {e}. Starting from scratch.")
            bundle = {"ddpm": {}, "meta": {}}
    saved = bundle.get("ddpm")
    if not isinstance(saved, dict):
        saved = {}
        bundle["ddpm"] = saved

    def _save() -> None:
        if not bool(args.cache):
            return
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(bundle, tmp)
        tmp.replace(ckpt_path)
        print(f"[CKPT] saved: {ckpt_path}")

    # Iterate fixed targets
    results: list[dict[str, object]] = []
    n_eval = int(X_eval_z.shape[0])
    eval_limit = n_eval if int(args.n_eval_epochs) <= 0 else min(n_eval, int(args.n_eval_epochs))

    for target, (inp1, inp2) in FIXED_SPECS:
        if target in exclude_targets:
            raise ValueError(f"Fixed spec uses excluded target: {target}")
        if target not in eeg_ch_names:
            raise ValueError(f"Target channel not found in EEG channels: {target}. Have={eeg_ch_names}")
        if inp1 not in eeg_ch_names or inp2 not in eeg_ch_names:
            raise ValueError(f"Input channel(s) not found for target={target}: inputs=({inp1},{inp2})")
        target_idx = eeg_ch_names.index(target)
        input_idxs = (eeg_ch_names.index(inp1), eeg_ch_names.index(inp2))

        key = f"{target}|inputs{input_idxs}|epochs{int(args.ddpm_epochs)}"
        model = p.UNet(in_channels=3, out_channels=1, n_feat=64, time_emb_dim=256).to(device)
        if bool(args.cache) and (not bool(args.force_train)) and key in saved:
            model.load_state_dict(saved[key], strict=True)
            print(f"[DDPM ckpt] loaded: {target} inputs=({inp1},{inp2})")
        else:
            print("-" * 40)
            print(f"Train DDPM target={target} inputs=({inp1},{inp2})")
            train_ds = p.EEGEpochs(X_train_z, input_idxs=input_idxs, target_idx=target_idx)
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
                epoch_desc=f"DDPM Train {target}",
            )
            if bool(args.cache):
                saved[key] = model.state_dict()
                _save()

        model.eval()

        # Eval on EVAL set
        test_ds = p.EEGEpochs(X_eval_z, input_idxs=input_idxs, target_idx=target_idx)
        test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
        mse_list: list[float] = []
        corr_list: list[float] = []
        seen = 0
        for cond, tgt in test_loader:
            if seen >= eval_limit:
                break
            remaining = eval_limit - seen
            if cond.size(0) > remaining:
                cond = cond[:remaining]
                tgt = tgt[:remaining]
            cond = cond.to(device)
            tgt = tgt.to(device)
            with torch.no_grad():
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
                real_1d = tgt_np[i]
                syn_1d = syn_np[i]
                mse_list.append(float(np.mean((syn_1d - real_1d) ** 2)))
                corr_list.append(float(p.pearson_corr_1d(real_1d, syn_1d)))
            seen += int(tgt_np.shape[0])

        mse = float(np.mean(mse_list)) if mse_list else float("nan")
        pear = float(np.mean(corr_list)) if corr_list else float("nan")
        results.append(
            {
                "Target Channel": target,
                "Input Channels": f"{inp1}, {inp2}",
                "MSE (computed)": mse,
                "Pearson (computed)": pear,
                "n_eval": int(seen),
            }
        )
        print(f"[EVAL] {target:>4s} <- ({inp1},{inp2}) | MSE={mse:.6f} | Pearson={pear:.6f} | n={seen}")

    # Sort + print
    if str(args.sort_by).lower() == "mse":
        results_sorted = sorted(results, key=lambda r: float(r.get("MSE (computed)", math.inf)))
    else:
        results_sorted = sorted(results, key=lambda r: float(r.get("Pearson (computed)", -math.inf)), reverse=True)

    print("\n" + "=" * 80)
    print(f"BCI-IV 2a — DDPM channel verification — Subject {subject_id}")
    print("=" * 80)
    try:
        import pandas as pd

        df = pd.DataFrame(results_sorted)
        print(df.to_string(index=False))
        if str(args.csv_out).strip():
            out_path = Path(str(args.csv_out)).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"[CSV] saved: {out_path}")
    except Exception:
        for r in results_sorted:
            print(r)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.1f}s")

