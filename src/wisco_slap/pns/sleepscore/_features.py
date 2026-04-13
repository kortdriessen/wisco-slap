"""Feature extraction for sleep scoring: multi-resolution EEG, pupil, whisking, and context."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.signal import welch
from scipy.stats import entropy, iqr

from ._config import (
    CONTEXT_KEY_FEATURES,
    CONTEXT_WINDOWS,
    EEG_BANDS_1S,
    EEG_BANDS_LONG,
    ScoreConfig,
)
from ._utils import segment_indices


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _bandpower_from_welch(
    x: np.ndarray,
    fs: float,
    bands: list[tuple[str, float, float]],
    nperseg: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Compute band powers from Welch PSD.

    Returns
    -------
    powers : dict mapping band name to log band power.
    freqs : frequency axis.
    Pxx : power spectral density.
    """
    if nperseg is None:
        nperseg = min(int(fs * 2), 512)
        nperseg = max(nperseg, 64)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    powers: dict[str, float] = {}
    for name, f0, f1 in bands:
        m = (f >= f0) & (f < f1)
        bp = float(np.trapz(Pxx[m], f[m])) if np.any(m) else 0.0
        powers[name] = math.log(bp + 1e-12)
    return powers, f, Pxx


def _spectral_entropy(Pxx: np.ndarray) -> float:
    """Shannon entropy of the normalized PSD."""
    p = Pxx / (Pxx.sum() + 1e-12)
    return float(entropy(p))


def _spectral_edge_frequency(f: np.ndarray, Pxx: np.ndarray, percentile: float = 0.95) -> float:
    """Frequency below which `percentile` of total power lies."""
    cumpower = np.cumsum(Pxx)
    total = cumpower[-1] if len(cumpower) > 0 else 0.0
    if total <= 0:
        return 0.0
    idx = np.searchsorted(cumpower, percentile * total)
    idx = min(idx, len(f) - 1)
    return float(f[idx])


def _line_length(x: np.ndarray) -> float:
    """Sum of absolute sample-to-sample differences, normalized by length."""
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(x)))) / len(x)


# ---------------------------------------------------------------------------
# Per-epoch feature extraction
# ---------------------------------------------------------------------------


def _extract_eeg_features_for_epoch(
    eeg_signal: np.ndarray,
    eeg_timestamps: np.ndarray,
    fs_eeg: float,
    t0: float,
    t1: float,
    feature_windows_s: list[float],
    epoch_len_s: float,
) -> dict[str, float]:
    """Extract EEG features at multiple resolutions for a single epoch centered at (t0+t1)/2."""
    feats: dict[str, float] = {}
    center = (t0 + t1) / 2.0

    for win_s in feature_windows_s:
        half = win_s / 2.0
        w_start = center - half
        w_end = center + half
        sl = segment_indices(eeg_timestamps, w_start, w_end)
        xe = eeg_signal[sl]
        tag = f"{win_s:.0f}s"

        min_samples = max(int(0.5 * fs_eeg * win_s), 8)
        if len(xe) < min_samples:
            # Insufficient coverage — fill NaN
            bands = EEG_BANDS_1S if win_s <= 1.0 else EEG_BANDS_LONG
            for name, _, _ in bands:
                feats[f"eeg_{name}_{tag}"] = float("nan")
            if win_s <= 1.0:
                for ratio_name in [
                    "theta_delta", "theta_sigma", "beta_delta",
                    "ugamma_delta", "spec_entropy", "spec_edge_95",
                    "line_length",
                ]:
                    feats[f"eeg_{ratio_name}_{tag}"] = float("nan")
            continue

        bands = EEG_BANDS_1S if win_s <= 1.0 else EEG_BANDS_LONG
        powers, f_axis, Pxx = _bandpower_from_welch(xe, fs_eeg, bands)

        for name in powers:
            feats[f"eeg_{name}_{tag}"] = powers[name]

        # Ratios and derived features only for the 1s base window
        if win_s <= 1.0:
            delta = math.exp(powers.get("delta", -27)) + 1e-9
            theta = math.exp(powers.get("theta", -27)) + 1e-9
            sigma = math.exp(powers.get("sigma", -27)) + 1e-9
            beta = math.exp(powers.get("beta", -27)) + 1e-9
            ugamma = math.exp(powers.get("ugamma", -27)) + 1e-9

            feats[f"eeg_theta_delta_{tag}"] = math.log(theta / delta)
            feats[f"eeg_theta_sigma_{tag}"] = math.log(theta / sigma)
            feats[f"eeg_beta_delta_{tag}"] = math.log(beta / delta)
            feats[f"eeg_ugamma_delta_{tag}"] = math.log(ugamma / delta)
            feats[f"eeg_spec_entropy_{tag}"] = _spectral_entropy(Pxx)
            feats[f"eeg_spec_edge_95_{tag}"] = _spectral_edge_frequency(f_axis, Pxx, 0.95)
            feats[f"eeg_line_length_{tag}"] = _line_length(xe)

    return feats


def _extract_pupil_features_for_epoch(
    pupil: dict,
    t0: float,
    t1: float,
    fs_pu: float,
    epoch_len_s: float,
) -> dict[str, float]:
    """Extract pupil/eyelid features for one epoch."""
    feats: dict[str, float] = {}
    ts = pupil["timestamps"]
    sl = segment_indices(ts, t0, t1)

    xd = pupil["diameter"][sl]
    xm = pupil["motion"][sl]
    xe = pupil["eyelid"][sl]
    xen = pupil["eyelid_norm"][sl]

    min_samples = max(int(0.5 * fs_pu * epoch_len_s), 2)
    if len(xd) < min_samples:
        for k in [
            "pu_diam_mean", "pu_diam_std", "pu_diam_iqr",
            "pu_vel_absmean", "pu_vel_std",
            "pu_motion_mean", "pu_motion_std",
            "pu_eyelid_mean", "pu_eyelid_std",
            "pu_eyelid_norm_mean",
            "pu_constriction_rate",
            "pu_eye_closure_frac",
        ]:
            feats[k] = float("nan")
        return feats

    dt = 1.0 / fs_pu
    vel = np.diff(xd) / dt if len(xd) > 1 else np.array([0.0])

    feats["pu_diam_mean"] = float(np.nanmean(xd))
    feats["pu_diam_std"] = float(np.nanstd(xd))
    feats["pu_diam_iqr"] = float(iqr(xd[~np.isnan(xd)])) if np.sum(~np.isnan(xd)) > 4 else 0.0
    feats["pu_vel_absmean"] = float(np.nanmean(np.abs(vel)))
    feats["pu_vel_std"] = float(np.nanstd(vel))
    feats["pu_motion_mean"] = float(np.nanmean(xm))
    feats["pu_motion_std"] = float(np.nanstd(xm))
    feats["pu_eyelid_mean"] = float(np.nanmean(xe))
    feats["pu_eyelid_std"] = float(np.nanstd(xe))
    feats["pu_eyelid_norm_mean"] = float(np.nanmean(xen))

    # Constriction rate: fraction of velocity samples that are negative (pupil shrinking)
    feats["pu_constriction_rate"] = float(np.mean(vel < 0)) if len(vel) > 0 else 0.5

    # Eye closure fraction: fraction of epoch where normalized eyelid is below threshold
    # Threshold of 0.3 is empirical — means eyelid is ~30% of maximum openness
    valid_en = xen[~np.isnan(xen)]
    if len(valid_en) > 0:
        feats["pu_eye_closure_frac"] = float(np.mean(valid_en < 0.3))
    else:
        feats["pu_eye_closure_frac"] = float("nan")

    return feats


def _extract_whisking_features_for_epoch(
    pupil: dict,
    t0: float,
    t1: float,
    fs_pu: float,
    epoch_len_s: float,
) -> dict[str, float]:
    """Extract whisking features for one epoch."""
    feats: dict[str, float] = {}
    ts = pupil["timestamps"]
    sl = segment_indices(ts, t0, t1)
    xw = pupil["whisking"][sl]

    min_samples = max(int(0.5 * fs_pu * epoch_len_s), 2)
    if len(xw) < min_samples:
        for k in ["whisk_mean", "whisk_std", "whisk_iqr", "whisk_active_frac", "whisk_burst_count"]:
            feats[k] = float("nan")
        return feats

    feats["whisk_mean"] = float(np.nanmean(xw))
    feats["whisk_std"] = float(np.nanstd(xw))
    feats["whisk_iqr"] = float(iqr(xw[~np.isnan(xw)])) if np.sum(~np.isnan(xw)) > 4 else 0.0

    # Active fraction: above 3 * MAD threshold
    wmad = float(np.median(np.abs(xw - np.median(xw)))) + 1e-9
    threshold = np.median(xw) + 3.0 * wmad
    feats["whisk_active_frac"] = float(np.mean(xw > threshold))

    # Burst count: number of upward threshold crossings
    above = xw > threshold
    crossings = np.diff(above.astype(int))
    feats["whisk_burst_count"] = float(np.sum(crossings == 1))

    return feats


# ---------------------------------------------------------------------------
# Session-level feature extraction
# ---------------------------------------------------------------------------


def extract_features_for_session(
    sess: dict,
    edges: np.ndarray,
    config: ScoreConfig,
) -> pl.DataFrame:
    """Extract all features for every epoch in a session.

    Parameters
    ----------
    sess : session dict with ``session_id``, ``eeg``, ``pupil`` keys.
    edges : epoch edge array of length n_epochs + 1.
    config : scoring configuration.

    Returns
    -------
    polars DataFrame with one row per epoch and all feature columns.
    """
    sid = sess["session_id"]
    eeg = sess["eeg"]
    pupil = sess["pupil"]
    fs_eeg = float(eeg["fs"])
    fs_pu = float(pupil["fs"])
    epoch_len_s = config.epoch_len_s

    rows: list[dict] = []
    for ei in range(len(edges) - 1):
        t0, t1 = float(edges[ei]), float(edges[ei + 1])
        row: dict = {
            "session_id": sid,
            "epoch_idx": ei,
            "start_s": t0,
            "end_s": t1,
        }

        # EEG features (multi-resolution)
        eeg_feats = _extract_eeg_features_for_epoch(
            eeg["signal"], eeg["timestamps"], fs_eeg,
            t0, t1, config.feature_windows_s, epoch_len_s,
        )
        row.update(eeg_feats)

        # Pupil features
        pu_feats = _extract_pupil_features_for_epoch(pupil, t0, t1, fs_pu, epoch_len_s)
        row.update(pu_feats)

        # Whisking features
        wh_feats = _extract_whisking_features_for_epoch(pupil, t0, t1, fs_pu, epoch_len_s)
        row.update(wh_feats)

        rows.append(row)

    df = pl.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Context features (rolling statistics across epochs)
# ---------------------------------------------------------------------------


def add_context_features(
    df: pl.DataFrame,
    session_col: str = "session_id",
    key_features: list[str] | None = None,
    context_windows: list[int] | None = None,
) -> pl.DataFrame:
    """Add multi-scale rolling context features and rate-of-change features.

    Parameters
    ----------
    df : feature DataFrame (one row per epoch, sorted by session and epoch_idx).
    session_col : session identifier column.
    key_features : feature names to compute context for; defaults to CONTEXT_KEY_FEATURES.
    context_windows : rolling window sizes in epochs; defaults to CONTEXT_WINDOWS.

    Returns
    -------
    DataFrame with additional context columns appended.
    """
    if key_features is None:
        key_features = CONTEXT_KEY_FEATURES
    if context_windows is None:
        context_windows = CONTEXT_WINDOWS

    # Only compute context for features that actually exist in the dataframe
    key_features = [f for f in key_features if f in df.columns]

    df = df.sort(session_col, "epoch_idx")

    for feat in key_features:
        for win in context_windows:
            min_periods = max(3, win // 3)
            # Rolling mean
            col_name = f"{feat}_ctx{win}_mean"
            df = df.with_columns(
                pl.col(feat)
                .rolling_mean(window_size=win, min_periods=min_periods, center=True)
                .over(session_col)
                .alias(col_name)
            )
            # Rolling std for short window only
            if win == context_windows[0]:
                col_name_std = f"{feat}_ctx{win}_std"
                df = df.with_columns(
                    pl.col(feat)
                    .rolling_std(window_size=win, min_periods=min_periods, center=True)
                    .over(session_col)
                    .alias(col_name_std)
                )

        # Rate of change (backward difference over 5 and 15 epochs)
        for roc_win in [5, 15]:
            col_name = f"{feat}_roc{roc_win}"
            df = df.with_columns(
                (pl.col(feat) - pl.col(feat).shift(roc_win))
                .over(session_col)
                .alias(col_name)
            )

    return df


# ---------------------------------------------------------------------------
# Full pipeline: features for multiple sessions
# ---------------------------------------------------------------------------


def build_features(
    sessions: list[dict],
    config: ScoreConfig,
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Extract features for all sessions.

    Parameters
    ----------
    sessions : list of session dicts.
    config : scoring configuration.

    Returns
    -------
    feats_df : polars DataFrame with all features (one row per epoch across all sessions).
    edges_by_session : dict mapping session_id to epoch edge arrays.
    """
    from ._data import build_epoch_grid

    edges_by_session = build_epoch_grid(sessions, config.epoch_len_s)

    feat_frames = []
    for sess in sessions:
        sid = sess["session_id"]
        feats = extract_features_for_session(sess, edges_by_session[sid], config)
        feat_frames.append(feats)

    feats_df = pl.concat(feat_frames)
    feats_df = add_context_features(feats_df)
    return feats_df, edges_by_session
