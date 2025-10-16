# -----------------------------
# sleep_hsmm.py
# -----------------------------
# Hybrid per-epoch discriminative classifier + HSMM sequence smoothing for 3-state mouse sleep staging
# TRAIN and INFER flows.
#
# Public API:
#   model = train_sleep_hsmm(sessions, labels_by_session, ...)
#   bout_df, epoch_df = infer_sleep_hsmm(model, sessions, ...)
#
# Data schema
# -----------
# 'sessions' is a list of dicts; each dict MUST have:
#   {
#     "session_id": str,
#     "eeg": {"signal": np.ndarray, "timestamps": np.ndarray, "fs": float},
#     "loadcell": {"signal": np.ndarray, "timestamps": np.ndarray, "fs": float},
#     "pupil": {"diameter": np.ndarray, "motion": np.ndarray, "eyelid": np.ndarray,
#               "whisking": np.ndarray, "timestamps": np.ndarray, "fs": float}
#   }
#
# 'labels_by_session' is a dict: session_id -> pd.DataFrame with columns ['start_s','end_s','label']
# Labels should be among {'NREM','REM','Wake'} (case-insensitive accepted); others will be ignored.
#
# Returns
# -------
#  - train_sleep_hsmm: a model dict that can be saved with save_model(path) and later loaded with load_model(path)
#  - infer_sleep_hsmm: bout_df (per-bout), epoch_df (per-epoch with probabilities)
#
# Design choices are annotated with 'DECISION:' comments.
#
from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy, iqr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

STATE_NAMES = ["NREM", "REM", "Wake"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_NAMES)}

# -----------------------------
# Utility helpers
# -----------------------------


def _validate_and_standardize_labels(df: pd.DataFrame) -> pd.DataFrame:
    assert {"start_s", "end_s", "label"}.issubset(
        df.columns
    ), "labels df needs start_s,end_s,label"
    df = df.copy().sort_values(["start_s", "end_s"]).reset_index(drop=True)
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    mapping = {
        "NREM": "NREM",
        "NR": "NREM",
        "SWS": "NREM",
        "REM": "REM",
        "R": "REM",
        "WAKE": "Wake",
        "W": "Wake",
    }
    df["label"] = df["label"].map(lambda x: mapping.get(x, x))
    df = df[df["label"].isin(STATE_NAMES)]
    return df


def _epoch_grid_from_sessions(
    sessions: list[dict], epoch_len_s: float, use_loadcell: bool = True
) -> dict[str, np.ndarray]:
    """Create epoch edges per session using the common time range across required modalities."""
    edges_by_session = {}
    for sess in sessions:
        sid = sess["session_id"]
        # Always require EEG and pupil; include loadcell only if requested and present
        required_keys = ["eeg", "pupil"] + (
            ["loadcell"] if (use_loadcell and ("loadcell" in sess)) else []
        )
        starts, ends = [], []
        for key in required_keys:
            if key not in sess:
                raise ValueError(
                    f"Session {sid}: missing required modality '{key}' for epoching"
                )
            ts = sess[key]["timestamps"]
            if len(ts) == 0:
                raise ValueError(f"Session {sid}:{key} has empty timestamps")
            starts.append(float(ts[0]))
            ends.append(float(ts[-1]))
        start_s = max(starts)
        end_s = min(ends)
        if end_s <= start_s + epoch_len_s:
            raise ValueError(
                f"Session {sid}: insufficient overlap across modalities to epoch"
            )
        n_epochs = int(np.floor((end_s - start_s) / epoch_len_s))
        edges_by_session[sid] = start_s + np.arange(n_epochs + 1) * epoch_len_s
    return edges_by_session


def _segment_indices(ts: np.ndarray, start: float, end: float) -> slice:
    i0 = np.searchsorted(ts, start, side="left")
    i1 = np.searchsorted(ts, end, side="left")
    return slice(i0, i1)


def _spectral_entropy(psd: np.ndarray) -> float:
    p = psd / (psd.sum() + 1e-12)
    return float(entropy(p))


def _bandpower_from_welch(
    x: np.ndarray,
    fs: float,
    bands: list[tuple[float, float]],
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if nperseg is None:
        # DECISION: for short epochs use up to 2s worth of samples but cap for speed
        nperseg = min(int(fs * 2), 512)
        nperseg = max(nperseg, 64)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    band_powers = []
    for f0, f1 in bands:
        m = (f >= f0) & (f < f1)
        band_powers.append(float(np.trapz(Pxx[m], f[m]) if np.any(m) else 0.0))
    return np.array(band_powers, dtype=float), Pxx


def _zero_crossings(x: np.ndarray) -> int:
    return int(np.sum(np.diff(np.signbit(x)) != 0))


def _robust_scale_per_session(X: pd.DataFrame, session_ids: pd.Series) -> pd.DataFrame:
    Xs = X.copy()
    for _sid, idx in session_ids.groupby(session_ids).groups.items():
        med = X.loc[idx].median(axis=0)
        sp = X.loc[idx].apply(iqr, axis=0).replace(0, 1.0)
        Xs.loc[idx] = (X.loc[idx] - med) / sp
    return Xs


# -----------------------------
# Feature extraction (incl. whisking)
# -----------------------------


def _extract_epoch_features_for_session(
    sess: dict, edges: np.ndarray, epoch_len_s: float, use_loadcell: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sid = sess["session_id"]
    eeg = sess["eeg"]
    has_lc = use_loadcell and ("loadcell" in sess)
    if has_lc:
        lc = sess["loadcell"]
        fs_lc = float(lc["fs"])
        ts_lc = lc["timestamps"]
        x_lc = lc["signal"]

    pu = sess["pupil"]
    fs_eeg = float(eeg["fs"])
    ts_eeg = eeg["timestamps"]
    x_eeg = eeg["signal"]
    fs_pu = float(pu["fs"])
    ts_pu = pu["timestamps"]
    x_pd = pu["diameter"]
    x_pm = pu["motion"]
    x_pe = pu["eyelid"]
    x_wh = pu["whisking"]

    EEG_BANDS = [
        (0.5, 4.0),
        (6.0, 10.0),
        (10.0, 15.0),
        (15.0, 30.0),
        (30.0, 55.0),
        (55.0, 90.0),
    ]
    EEG_NAMES = [
        "eeg_delta",
        "eeg_theta",
        "eeg_sigma",
        "eeg_beta",
        "eeg_lgamma",
        "eeg_hgamma",
    ]

    rows, cov_rows = [], []
    for ei in range(len(edges) - 1):
        t0, t1 = float(edges[ei]), float(edges[ei + 1])
        # slices
        sl_eeg = _segment_indices(ts_eeg, t0, t1)
        sl_pu = _segment_indices(ts_pu, t0, t1)

        xe = x_eeg[sl_eeg]
        xd = x_pd[sl_pu]
        xm = x_pm[sl_pu]
        xeid = x_pe[sl_pu]
        xw = x_wh[sl_pu]

        eeg_cov = len(xe) >= 0.5 * fs_eeg * epoch_len_s
        pu_cov = len(xd) >= 0.5 * fs_pu * epoch_len_s

        feats = {"session_id": sid, "epoch_idx": ei, "start_s": t0, "end_s": t1}

        # EEG features
        if eeg_cov and len(xe) > 4:
            bp, Pxx = _bandpower_from_welch(xe, fs_eeg, EEG_BANDS)
            for name, val in zip(EEG_NAMES, bp, strict=False):
                feats[name] = math.log(val + 1e-12)
            delta = bp[0] + 1e-9
            theta = bp[1] + 1e-9
            sigma = bp[2] + 1e-9
            beta = bp[3] + 1e-9
            feats["eeg_theta_delta"] = math.log(theta / delta)
            feats["eeg_theta_sigma_delta"] = math.log((theta + sigma) / delta)
            feats["eeg_beta_delta"] = math.log(beta / delta)
            feats["eeg_spec_entropy"] = _spectral_entropy(Pxx)
        else:
            for name in EEG_NAMES + [
                "eeg_theta_delta",
                "eeg_theta_sigma_delta",
                "eeg_beta_delta",
                "eeg_spec_entropy",
            ]:
                feats[name] = np.nan

        # Load-cell features

        if has_lc:
            sl_lc = _segment_indices(ts_lc, t0, t1)
            xl = x_lc[sl_lc]
            lc_cov = len(xl) >= 0.5 * fs_lc * epoch_len_s
            if lc_cov and len(xl) > 2:
                xl_center = xl - np.median(xl)
                feats["lc_rms"] = float(np.sqrt(np.mean(xl_center**2)))
                mad = np.median(np.abs(xl_center - np.median(xl_center))) + 1e-9
                feats["lc_frac_active"] = float(np.mean(np.abs(xl_center) > 3.5 * mad))
                feats["lc_zero_cross"] = _zero_crossings(xl_center) / (
                    len(xl_center) + 1e-9
                )
                bp_resp, _ = _bandpower_from_welch(xl_center, fs_lc, [(1.0, 5.0)])
                feats["lc_resp_power"] = math.log(bp_resp[0] + 1e-12)
            else:
                for k in ["lc_rms", "lc_frac_active", "lc_zero_cross", "lc_resp_power"]:
                    feats[k] = np.nan
        else:
            lc_cov = False
            # Ensure columns exist but will be dropped later if desired
            for k in ["lc_rms", "lc_frac_active", "lc_zero_cross", "lc_resp_power"]:
                feats[k] = np.nan

        # Pupil & eyelid features
        if pu_cov and len(xd) > 2:
            dt = 1.0 / fs_pu
            vel = np.diff(xd) / dt if len(xd) > 1 else np.array([0.0])
            feats["pu_diam_mean"] = float(np.mean(xd))
            feats["pu_diam_std"] = float(np.std(xd))
            feats["pu_diam_iqr"] = float(iqr(xd)) if len(xd) > 4 else 0.0
            feats["pu_vel_absmean"] = float(np.mean(np.abs(vel)))
            feats["pu_vel_std"] = float(np.std(vel))
            feats["pu_motion_mean"] = float(np.mean(xm)) if len(xm) > 0 else np.nan
            feats["pu_motion_std"] = float(np.std(xm)) if len(xm) > 0 else np.nan
            feats["pu_eyelid_mean"] = float(np.mean(xeid)) if len(xeid) > 0 else np.nan
            feats["pu_eyelid_std"] = float(np.std(xeid)) if len(xeid) > 0 else np.nan
            # Whisking features (absolute frame-diff-like signal)
            feats["whisk_mean"] = float(np.mean(xw)) if len(xw) > 0 else np.nan
            feats["whisk_std"] = float(np.std(xw)) if len(xw) > 0 else np.nan
            feats["whisk_iqr"] = float(iqr(xw)) if len(xw) > 4 else 0.0
            # Fraction above robust threshold
            wmad = np.median(np.abs(xw - np.median(xw))) + 1e-9
            feats["whisk_active_frac"] = (
                float(np.mean(xw > (np.median(xw) + 3.0 * wmad)))
                if len(xw) > 0
                else np.nan
            )
        else:
            for k in [
                "pu_diam_mean",
                "pu_diam_std",
                "pu_diam_iqr",
                "pu_vel_absmean",
                "pu_vel_std",
                "pu_motion_mean",
                "pu_motion_std",
                "pu_eyelid_mean",
                "pu_eyelid_std",
                "whisk_mean",
                "whisk_std",
                "whisk_iqr",
                "whisk_active_frac",
            ]:
                feats[k] = np.nan

        rows.append(feats)
        cov_rows.append(
            {
                "session_id": sid,
                "epoch_idx": ei,
                "cov_eeg": int(eeg_cov),
                "cov_loadcell": int(lc_cov),
                "cov_pupil": int(pu_cov),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(cov_rows)


def _add_context_features(
    df: pd.DataFrame, session_col: str = "session_id", window_epochs: int = 15
) -> pd.DataFrame:
    key_feats = [
        "eeg_delta",
        "eeg_theta_delta",
        "eeg_theta_sigma_delta",
        "lc_rms",
        "lc_resp_power",
        "pu_diam_mean",
        "pu_vel_absmean",
        "pu_eyelid_mean",
        "whisk_mean",
        "whisk_active_frac",
    ]
    df2 = df.copy().sort_values([session_col, "epoch_idx"])
    for k in key_feats:
        if k not in df2.columns:
            continue
        roll = (
            df2.groupby(session_col)[k]
            .apply(
                lambda x: x.rolling(
                    window=window_epochs,
                    min_periods=max(3, window_epochs // 3),
                    center=True,
                ).mean()
            )
            .reset_index(level=0, drop=True)
        )
        df2[f"{k}_rollmean"] = roll
    return df2


# -----------------------------
# Labels: intervals -> epoch labels
# -----------------------------


def _intervals_to_epoch_labels(
    edges: np.ndarray, labels_df: pd.DataFrame | None, min_overlap_frac: float = 0.0
) -> np.ndarray:
    """
    Map label intervals to fixed epochs via maximum overlap, requiring a minimum overlap
    fraction of the epoch (relative to the epoch length). Epochs that do not meet the
    minimum overlap are left unlabeled (-1).

    Parameters
    ----------
    edges : array-like
        Epoch edges of length n+1 (seconds). Epoch e spans [edges[e], edges[e+1]).
    labels_df : DataFrame or None
        Must have columns ['start_s','end_s','label']. May be sparse/non-contiguous.
    min_overlap_frac : float
        Minimum required fraction of the epoch (0..1) that must be covered by a label
        interval for that label to be assigned. Example: with 2-s epochs, 0.5 means ≥1.0 s.

    Returns
    -------
    y : np.ndarray of shape (n,)
        Per-epoch labels encoded as {NREM:0, REM:1, Wake:2}; unlabeled epochs are -1.
    """
    n = len(edges) - 1
    y = np.full(n, -1, dtype=int)
    if labels_df is None or labels_df.empty:
        return y

    L = _validate_and_standardize_labels(labels_df)
    starts = L["start_s"].to_numpy()
    ends = L["end_s"].to_numpy()
    labs = L["label"].map(lambda s: STATE_TO_IDX[s]).to_numpy()

    epoch_len = float(edges[1] - edges[0]) if n > 0 else 0.0
    min_ov = float(min_overlap_frac) * epoch_len

    i = 0
    for e in range(n):
        a, b = float(edges[e]), float(edges[e + 1])
        # Skip label intervals that end before this epoch
        while i < len(L) and ends[i] <= a:
            i += 1
        # Among intervals that intersect [a, b), keep the one with the largest overlap
        j = i
        best_lab, best_ov = -1, 0.0
        while j < len(L) and starts[j] < b:
            ov = max(0.0, min(b, ends[j]) - max(a, starts[j]))  # overlap (seconds)
            if ov >= min_ov and ov > best_ov:
                best_ov = ov
                best_lab = int(labs[j])
            j += 1
        y[e] = best_lab if best_lab != -1 else -1
    return y


# -----------------------------
# HSMM (explicit durations) Viterbi
# -----------------------------


@dataclass
class HSMMParams:
    mean_dur_s: dict[str, float]
    max_dur_s: float
    lognorm_sigma: float
    change_penalty: float
    stay_bonus: float


def _discrete_trunc_lognorm_logpmf(
    d_range: np.ndarray, mean_epochs: float, sigma_log: float, trunc_max: int
) -> np.ndarray:
    mu = math.log(max(mean_epochs, 1e-6)) - 0.5 * (sigma_log**2)
    xs = d_range.astype(float)
    pdf = np.exp(-((np.log(xs) - mu) ** 2) / (2 * sigma_log**2)) / (
        xs * sigma_log * math.sqrt(2 * math.pi)
    )
    pdf[(xs < 1) | (xs > trunc_max)] = 0.0
    pmf = pdf / (pdf.sum() + 1e-12)
    return np.log(pmf + 1e-300)


def _hsmm_viterbi_log(
    log_emissions: np.ndarray,
    init_logp: np.ndarray,
    logA: np.ndarray,
    dur_logpmf: list[np.ndarray],
    D_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, S = log_emissions.shape
    csum = np.zeros((T + 1, S), dtype=float)
    csum[1:] = np.cumsum(log_emissions, axis=0)
    dp = np.full((T, S), -np.inf, dtype=float)
    bp_state = np.full((T, S), -1, dtype=int)
    bp_dur = np.full((T, S), -1, dtype=int)

    for t in range(T):
        d_max_here = min(D_max, t + 1)
        for s in range(S):
            best_score = -np.inf
            best_prev = -1
            best_d = -1
            dur_lp = dur_logpmf[s]
            for d in range(1, d_max_here + 1):
                seg_ll = csum[t + 1, s] - csum[t + 1 - d, s]
                lp_d = dur_lp[d - 1] if d - 1 < len(dur_lp) else -np.inf
                if t - d < 0:
                    score = init_logp[s] + seg_ll + lp_d
                else:
                    prev_scores = dp[t - d, :] + logA[:, s]
                    q = int(np.argmax(prev_scores))
                    score = prev_scores[q] + seg_ll + lp_d
                if score > best_score:
                    best_score = score
                    best_prev = q if t - d >= 0 else -1
                    best_d = d
            dp[t, s] = best_score
            bp_state[t, s] = best_prev
            bp_dur[t, s] = best_d

    t = T - 1
    s = int(np.argmax(dp[t, :]))
    states, durs, starts = [], [], []
    while t >= 0 and s >= 0:
        d = int(bp_dur[t, s])
        start = t - d + 1
        states.append(s)
        durs.append(d)
        starts.append(start)
        s = int(bp_state[t, s])
        t = start - 1
        if t < 0:
            break
    return (
        np.array(states[::-1], int),
        np.array(durs[::-1], int),
        np.array(starts[::-1], int),
    )


# -----------------------------
# Core pipeline helpers
# -----------------------------


def _build_features_for_sessions(
    sessions: list[dict], epoch_len_s: float, use_loadcell: bool = True
):
    edges_by_session = _epoch_grid_from_sessions(
        sessions, epoch_len_s, use_loadcell=use_loadcell
    )
    feat_frames, cov_frames = [], []
    for sess in sessions:
        sid = sess["session_id"]
        feats, cov = _extract_epoch_features_for_session(
            sess, edges_by_session[sid], epoch_len_s, use_loadcell=use_loadcell
        )
        feat_frames.append(feats)
        cov_frames.append(cov)
    feats_all = pd.concat(feat_frames, ignore_index=True)
    feats_all = _add_context_features(
        feats_all, window_epochs=int(round(30.0 / epoch_len_s))
    )
    cov_all = pd.concat(cov_frames, ignore_index=True)

    if not use_loadcell:
        # Drop all load-cell derived columns and the coverage flag
        lc_cols = [c for c in feats_all.columns if c.startswith("lc_")]
        feats_all = feats_all.drop(columns=lc_cols, errors="ignore")
        cov_all = cov_all.drop(columns=["cov_loadcell"], errors="ignore")
    return feats_all, cov_all, edges_by_session


def _prepare_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    meta_cols = ["session_id", "epoch_idx", "start_s", "end_s"]
    y_col = "y"
    num_cols = [c for c in df.columns if c not in meta_cols + [y_col]]
    # Impute per-session medians first
    X_impute = df[num_cols].copy()
    for _sid, idx in df.groupby("session_id").groups.items():
        med = X_impute.loc[idx].median(axis=0)
        X_impute.loc[idx] = X_impute.loc[idx].fillna(med)
    X_scaled = _robust_scale_per_session(X_impute, df["session_id"])
    return X_scaled, num_cols


# -----------------------------
# Public API: TRAIN
# -----------------------------


def train_sleep_hsmm(
    sessions: list[dict],
    labels_by_session: dict[str, pd.DataFrame],
    epoch_len_s: float = 2.0,
    hsmm_params: dict | None = None,
    classifier_params: dict | None = None,
    smoothness: float = 0.5,
    random_state: int = 42,
    min_overlap_frac: float = 0.0,
    use_loadcell: bool = True,  # ← NEW
):
    """
    Train the per-epoch classifier and assemble HSMM config. Returns a model dict.
    """
    # Defaults
    hsmm_cfg = {
        "mean_dur_s": {"NREM": 30.0, "REM": 100.0, "Wake": 45.0},
        "max_dur_s": 300.0,
        "lognorm_sigma": 0.75,
        "change_penalty": -0.8,
        "stay_bonus": 0.2,
    }
    if hsmm_params:
        for k, v in hsmm_params.items():
            if k == "mean_dur_s":
                hsmm_cfg["mean_dur_s"].update(v)
            else:
                hsmm_cfg[k] = v
    # Smoothness scaling
    scale = 0.6 + 1.2 * float(np.clip(smoothness, 0, 1))
    hsmm_cfg["mean_dur_s"] = {
        k: float(v) * scale for k, v in hsmm_cfg["mean_dur_s"].items()
    }
    hsmm_cfg["change_penalty"] = float(hsmm_cfg["change_penalty"]) * (0.5 + smoothness)
    hsmm_cfg["stay_bonus"] = float(hsmm_cfg["stay_bonus"]) * (0.5 + smoothness)

    clf_cfg = {
        "gbdt": {
            "max_depth": None,
            "max_iter": 300,
            "learning_rate": 0.05,
            "l2_regularization": 0.0,
        },
        "calibration": {"method": "isotonic", "cv": 3},
        "test_size": 0.15,
    }
    if classifier_params:
        for k, v in classifier_params.items():
            if k in ["gbdt", "calibration"]:
                clf_cfg[k].update(v)
            else:
                clf_cfg[k] = v

    # Features
    feats_all, cov_all, edges_by_session = _build_features_for_sessions(
        sessions, epoch_len_s, use_loadcell=use_loadcell
    )

    # Labels -> epochs
    y_list = []
    for sess in sessions:
        sid = sess["session_id"]
        y = _intervals_to_epoch_labels(
            edges_by_session[sid],
            labels_by_session.get(sid, None),
            min_overlap_frac=min_overlap_frac,
        )
        y_list.append(
            pd.DataFrame({"session_id": sid, "epoch_idx": np.arange(len(y)), "y": y})
        )
    y_all = pd.concat(y_list, ignore_index=True)

    # Merge
    df = feats_all.merge(y_all, on=["session_id", "epoch_idx"], how="left").merge(
        cov_all, on=["session_id", "epoch_idx"], how="left"
    )
    if use_loadcell:
        for k in ["cov_eeg", "cov_loadcell", "cov_pupil"]:
            df[k] = df[k].fillna(0).astype(int)
    else:
        for k in ["cov_eeg", "cov_pupil"]:
            df[k] = df[k].fillna(0).astype(int)

    # Prepare matrices
    X_scaled, feature_cols = _prepare_matrix(df)
    y = df["y"].to_numpy()
    labeled_mask = y >= 0
    if not np.any(labeled_mask):
        raise ValueError("No labeled epochs found for training.")
    X_lab = X_scaled.loc[labeled_mask].to_numpy(dtype=float)
    y_lab = y[labeled_mask]

    # Ensure all 3 classes present
    present = np.unique(y_lab)
    need = np.array([0, 1, 2], dtype=int)
    if set(present.tolist()) != set(need.tolist()):
        raise ValueError(
            f"Training labels must include all three classes NREM(0), REM(1), Wake(2). Present: {present}"
        )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_lab,
        y_lab,
        test_size=clf_cfg["test_size"],
        random_state=random_state,
        stratify=y_lab,
    )

    # Compute sample weights to balance classes
    class_counts = np.bincount(y_tr, minlength=3).astype(float)
    inv = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    sw = inv[y_tr] * (len(y_tr) / np.sum(inv[y_tr]))

    # Base classifier
    base = HistGradientBoostingClassifier(**clf_cfg["gbdt"], random_state=random_state)
    base.fit(X_tr, y_tr, sample_weight=sw)

    # Calibrator
    calibrator = CalibratedClassifierCV(base, **clf_cfg["calibration"])
    calibrator.fit(X_te, y_te)  # no sample_weight here for simplicity/compatibility

    # Build model dict
    model = {
        "epoch_len_s": float(epoch_len_s),
        "feature_cols": feature_cols,
        "hsmm_cfg": hsmm_cfg,
        "calibrated_clf": calibrator,
        "state_names": STATE_NAMES,
        "use_loadcell": bool(use_loadcell),
    }
    return model


def save_model(model: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# Public API: INFER
# -----------------------------


def infer_sleep_hsmm(
    model: dict,
    sessions: list[dict],
    output_csv_path: str | None = None,
    return_epoch_level: bool = True,
    use_loadcell: bool | None = None,
):
    if use_loadcell is None:
        use_loadcell = bool(model.get("use_loadcell", True))
    epoch_len_s = float(model["epoch_len_s"])
    hsmm_cfg = model["hsmm_cfg"]
    clf = model["calibrated_clf"]
    feature_cols = model["feature_cols"]

    # Features on new data
    feats_all, cov_all, edges_by_session = _build_features_for_sessions(
        sessions, epoch_len_s, use_loadcell=use_loadcell
    )
    df = feats_all.merge(cov_all, on=["session_id", "epoch_idx"], how="left")
    if use_loadcell:
        for k in ["cov_eeg", "cov_loadcell", "cov_pupil"]:
            df[k] = df[k].fillna(0).astype(int)
    else:
        for k in ["cov_eeg", "cov_pupil"]:
            df[k] = df[k].fillna(0).astype(int)

    # Prepare matrices with same recipe (impute per session medians, robust scale per session)
    X_scaled, feat_cols_made = _prepare_matrix(df)
    # Ensure column order matches training feature_cols
    missing_cols = [c for c in feature_cols if c not in X_scaled.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected feature columns in inference: {missing_cols}"
        )
    X_mat = X_scaled[feature_cols].to_numpy(dtype=float)

    # Predict probabilities
    P = clf.predict_proba(X_mat)
    # Align column order to [0,1,2]
    if list(clf.classes_) != [0, 1, 2]:
        # reorder
        order = [list(clf.classes_).index(c) for c in [0, 1, 2]]
        P = P[:, order]

    # HSMM components
    S = 3
    logA = np.full((S, S), math.log(1e-6), dtype=float)
    for s in range(S):
        for sp in range(S):
            if s == sp:
                logA[s, sp] = math.log(0.90) + hsmm_cfg["stay_bonus"]
            else:
                logA[s, sp] = math.log(0.05) + hsmm_cfg["change_penalty"]
    # normalize rows
    row_logsum = np.logaddexp.reduce(logA, axis=1)
    logA = logA - row_logsum[:, None]
    init_logp = np.full(S, -math.log(S))

    D_max = int(max(1, math.floor(hsmm_cfg["max_dur_s"] / epoch_len_s)))
    d_range = np.arange(1, D_max + 1, dtype=int)
    dur_logpmf = []
    for s, name in enumerate(STATE_NAMES):
        mean_epochs = float(hsmm_cfg["mean_dur_s"][name]) / epoch_len_s
        dur_logpmf.append(
            _discrete_trunc_lognorm_logpmf(
                d_range, mean_epochs, float(hsmm_cfg["lognorm_sigma"]), D_max
            )
        )

    # Decode per session
    epoch_rows, bout_rows = [], []
    eps = 1e-12
    # Iterate through sessions in given order
    for sess in sessions:
        sid = sess["session_id"]
        m = (df["session_id"] == sid).to_numpy()
        P_s = P[m]
        T = P_s.shape[0]
        logE = np.log(np.clip(P_s, eps, 1.0))
        states, durs, starts = _hsmm_viterbi_log(
            logE, init_logp, logA, dur_logpmf, D_max
        )

        sess_df = df.loc[
            m, ["session_id", "epoch_idx", "start_s", "end_s"]
        ].reset_index(drop=True)
        dec = np.repeat(states, durs)[:T]
        if len(dec) < T:
            dec = np.concatenate([dec, np.full(T - len(dec), dec[-1])])
        yhat = dec.astype(int)
        ep = pd.DataFrame(
            {
                "session_id": sess_df["session_id"],
                "epoch_idx": sess_df["epoch_idx"],
                "start_s": sess_df["start_s"],
                "end_s": sess_df["end_s"],
                "pred_state_idx": yhat,
                "pred_label": [STATE_NAMES[i] for i in yhat],
                "P_NREM": P_s[:, 0],
                "P_REM": P_s[:, 1],
                "P_Wake": P_s[:, 2],
            }
        )
        epoch_rows.append(ep)

        # collapse to bouts
        if T > 0:
            cur_state = int(yhat[0])
            cur_start = float(sess_df.loc[0, "start_s"])
            ps = [P_s[0]]
            for t in range(1, T):
                if int(yhat[t]) != cur_state:
                    cur_end = float(sess_df.loc[t, "start_s"])
                    Ps = np.vstack(ps)
                    bout_rows.append(
                        {
                            "session_id": sid,
                            "start_s": cur_start,
                            "end_s": cur_end,
                            "label": STATE_NAMES[cur_state],
                            "n_epochs": len(ps),
                            "mean_P_NREM": float(Ps[:, 0].mean()),
                            "mean_P_REM": float(Ps[:, 1].mean()),
                            "mean_P_Wake": float(Ps[:, 2].mean()),
                        }
                    )
                    cur_state = int(yhat[t])
                    cur_start = float(sess_df.loc[t, "start_s"])
                    ps = [P_s[t]]
                else:
                    ps.append(P_s[t])
            # last bout
            cur_end = float(sess_df.loc[T - 1, "end_s"])
            Ps = np.vstack(ps)
            bout_rows.append(
                {
                    "session_id": sid,
                    "start_s": cur_start,
                    "end_s": cur_end,
                    "label": STATE_NAMES[cur_state],
                    "n_epochs": len(ps),
                    "mean_P_NREM": float(Ps[:, 0].mean()),
                    "mean_P_REM": float(Ps[:, 1].mean()),
                    "mean_P_Wake": float(Ps[:, 2].mean()),
                }
            )

    epoch_df = (
        pd.concat(epoch_rows, ignore_index=True) if epoch_rows else pd.DataFrame()
    )
    bout_df = pd.DataFrame(bout_rows) if bout_rows else pd.DataFrame()

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        bout_df.to_csv(output_csv_path, index=False)

    if return_epoch_level:
        return bout_df, epoch_df
    return bout_df, None
