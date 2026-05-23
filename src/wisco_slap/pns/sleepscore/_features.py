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

PUPIL_FEATURE_NAMES: list[str] = [
    "pu_diam_mean",
    "pu_diam_std",
    "pu_diam_iqr",
    "pu_vel_absmean",
    "pu_vel_std",
    "pu_motion_mean",
    "pu_motion_std",
    "pu_eyelid_mean",
    "pu_eyelid_std",
    "pu_eyelid_norm_mean",
    "pu_constriction_rate",
    "pu_eye_closure_frac",
]

WHISK_FEATURE_NAMES: list[str] = [
    "whisk_mean",
    "whisk_std",
    "whisk_iqr",
    "whisk_active_frac",
    "whisk_burst_count",
]


def _missing_feature_map(names: list[str]) -> dict[str, None]:
    """Return a feature dict with null values."""
    return {name: None for name in names}


def _safe_iqr(x: np.ndarray) -> float | None:
    """Return the IQR of finite values, or ``None`` if not enough samples exist."""
    valid = x[np.isfinite(x)]
    if valid.size < 5:
        return 0.0 if valid.size > 0 else None
    return float(iqr(valid))


def _safe_quantile(x: np.ndarray, q: float, *, default: float = 0.0) -> float:
    """Return a finite quantile or *default* if unavailable."""
    valid = x[np.isfinite(x)]
    if valid.size == 0:
        return default
    return float(np.quantile(valid, q))


def _fraction(mask: np.ndarray) -> float:
    """Return the mean of a boolean mask, defaulting to 0 for empty arrays."""
    return float(mask.mean()) if mask.size > 0 else 0.0


def _coerce_pupil_array(
    pupil: dict,
    key: str,
    *,
    length: int,
    default: float,
) -> np.ndarray:
    """Return a float array of the expected length for one pupil-field."""
    values = pupil.get(key, None)
    if values is None:
        return np.full(length, default, dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] != length:
        raise ValueError(
            f"Pupil field '{key}' has length {arr.shape[0]}, expected {length}."
        )
    return arr


def _build_video_quality_masks(
    pupil: dict,
    config: ScoreConfig,
) -> dict[str, np.ndarray]:
    """Compute per-frame validity masks for pupil, eyelid, whisking, and camera-off."""
    ts = np.asarray(pupil["timestamps"], dtype=float)
    n_frames = len(ts)
    diameter = _coerce_pupil_array(pupil, "diameter", length=n_frames, default=np.nan)
    motion = _coerce_pupil_array(pupil, "motion", length=n_frames, default=np.nan)
    eyelid = _coerce_pupil_array(pupil, "eyelid", length=n_frames, default=np.nan)
    eyelid_norm = _coerce_pupil_array(pupil, "eyelid_norm", length=n_frames, default=np.nan)
    whisking = _coerce_pupil_array(pupil, "whisking", length=n_frames, default=np.nan)
    pup_likelihood = _coerce_pupil_array(
        pupil, "pup_likelihood", length=n_frames, default=1.0
    )
    lid_likelihood = _coerce_pupil_array(
        pupil, "lid_likelihood", length=n_frames, default=1.0
    )

    vcfg = config.video_quality
    whisk_floor = _safe_quantile(
        whisking,
        vcfg.camera_off_whisk_quantile,
        default=0.0,
    )

    pup_lh_clean = np.nan_to_num(pup_likelihood, nan=0.0, posinf=1.0, neginf=0.0)
    lid_lh_clean = np.nan_to_num(lid_likelihood, nan=0.0, posinf=1.0, neginf=0.0)
    whisk_clean = np.nan_to_num(whisking, nan=-np.inf, neginf=-np.inf, posinf=np.inf)

    camera_off_frame = (
        (pup_lh_clean <= vcfg.camera_off_likelihood_max)
        & (lid_lh_clean <= vcfg.camera_off_likelihood_max)
        & (whisk_clean <= whisk_floor)
    )
    pupil_valid = (
        (pup_lh_clean >= vcfg.pupil_valid_likelihood_min)
        & np.isfinite(diameter)
    )
    pupil_motion_valid = pupil_valid & np.isfinite(motion)
    if pupil_motion_valid.size > 0:
        pupil_motion_valid[0] = False
        pupil_motion_valid[1:] = pupil_motion_valid[1:] & pupil_valid[:-1]

    eyelid_valid = (
        (lid_lh_clean >= vcfg.eyelid_valid_likelihood_min)
        & np.isfinite(eyelid)
        & np.isfinite(eyelid_norm)
    )
    whisk_valid = (~camera_off_frame) & np.isfinite(whisking)

    return {
        "pupil_valid": pupil_valid,
        "pupil_motion_valid": pupil_motion_valid,
        "eyelid_valid": eyelid_valid,
        "whisk_valid": whisk_valid,
        "camera_off_frame": camera_off_frame,
    }


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
        nperseg = min(int(fs * 2), 512, len(x))
        nperseg = max(nperseg, min(len(x), 64))
    nperseg = min(nperseg, len(x))
    noverlap = min(nperseg // 2, max(nperseg - 1, 0))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    powers: dict[str, float] = {}
    for name, f0, f1 in bands:
        m = (f >= f0) & (f < f1)
        bp = float(np.trapezoid(Pxx[m], f[m])) if np.any(m) else 0.0
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
) -> dict[str, float | None]:
    """Extract EEG features at multiple resolutions for a single epoch centered at (t0+t1)/2."""
    feats: dict[str, float | None] = {}
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
            # Insufficient coverage — fill with nulls so downstream context/scaling
            # preserves missingness rather than propagating NaN.
            bands = EEG_BANDS_1S if win_s <= 1.0 else EEG_BANDS_LONG
            for name, _, _ in bands:
                feats[f"eeg_{name}_{tag}"] = None
            if win_s <= 1.0:
                for ratio_name in [
                    "theta_delta", "theta_sigma", "beta_delta",
                    "ugamma_delta", "spec_entropy", "spec_edge_95",
                    "line_length",
                ]:
                    feats[f"eeg_{ratio_name}_{tag}"] = None
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
    quality_masks: dict[str, np.ndarray],
    sl: slice,
    ts: np.ndarray,
    fs_pu: float,
    epoch_len_s: float,
) -> dict[str, float | None]:
    """Extract pupil/eyelid features for one epoch."""
    xd = pupil["diameter"][sl]
    xm = pupil["motion"][sl]
    xe = pupil["eyelid"][sl]
    xen = pupil["eyelid_norm"][sl]
    xt = ts[sl]

    pupil_valid = quality_masks["pupil_valid"][sl]
    pupil_motion_valid = quality_masks["pupil_motion_valid"][sl]
    eyelid_valid = quality_masks["eyelid_valid"][sl]

    feats: dict[str, float | None] = {}
    feats.update(_missing_feature_map(PUPIL_FEATURE_NAMES))

    min_samples = max(int(0.5 * fs_pu * epoch_len_s), 2)
    xd_valid = xd[pupil_valid]
    xt_valid = xt[pupil_valid]
    xm_valid = xm[pupil_motion_valid]
    xe_valid = xe[eyelid_valid]
    xen_valid = xen[eyelid_valid]

    if xd_valid.size >= min_samples:
        feats["pu_diam_mean"] = float(np.mean(xd_valid))
        feats["pu_diam_std"] = float(np.std(xd_valid))
        feats["pu_diam_iqr"] = _safe_iqr(xd_valid)

        if xd_valid.size > 1:
            dt = np.diff(xt_valid)
            dx = np.diff(xd_valid)
            valid_dt = dt > 0
            vel = dx[valid_dt] / dt[valid_dt]
        else:
            vel = np.array([0.0], dtype=float)

        if vel.size > 0:
            feats["pu_vel_absmean"] = float(np.mean(np.abs(vel)))
            feats["pu_vel_std"] = float(np.std(vel))
            feats["pu_constriction_rate"] = float(np.mean(vel < 0))

    if xm_valid.size >= min_samples:
        feats["pu_motion_mean"] = float(np.mean(xm_valid))
        feats["pu_motion_std"] = float(np.std(xm_valid))

    if xe_valid.size >= min_samples:
        feats["pu_eyelid_mean"] = float(np.mean(xe_valid))
        feats["pu_eyelid_std"] = float(np.std(xe_valid))
        feats["pu_eyelid_norm_mean"] = float(np.mean(xen_valid))
        feats["pu_eye_closure_frac"] = float(np.mean(xen_valid < 0.3))

    return feats


def _extract_whisking_features_for_epoch(
    pupil: dict,
    quality_masks: dict[str, np.ndarray],
    sl: slice,
    fs_pu: float,
    epoch_len_s: float,
) -> dict[str, float | None]:
    """Extract whisking features for one epoch."""
    xw = pupil["whisking"][sl]
    whisk_valid = quality_masks["whisk_valid"][sl]
    xw_valid = xw[whisk_valid]

    feats: dict[str, float | None] = {}
    feats.update(_missing_feature_map(WHISK_FEATURE_NAMES))

    min_samples = max(int(0.5 * fs_pu * epoch_len_s), 2)
    if xw_valid.size < min_samples:
        return feats

    feats["whisk_mean"] = float(np.mean(xw_valid))
    feats["whisk_std"] = float(np.std(xw_valid))
    feats["whisk_iqr"] = _safe_iqr(xw_valid)

    # Active fraction: above 3 * MAD threshold
    wmed = float(np.median(xw_valid))
    wmad = float(np.median(np.abs(xw_valid - wmed))) + 1e-9
    threshold = wmed + 3.0 * wmad
    feats["whisk_active_frac"] = float(np.mean(xw_valid > threshold))

    # Burst count: number of upward threshold crossings
    above = xw_valid > threshold
    crossings = np.diff(above.astype(int))
    feats["whisk_burst_count"] = float(np.sum(crossings == 1))

    return feats


def _extract_video_quality_features_for_epoch(
    quality_masks: dict[str, np.ndarray],
    sl: slice,
    config: ScoreConfig,
) -> dict[str, float]:
    """Summarize video validity and camera-off fractions for one epoch."""
    camera_off = quality_masks["camera_off_frame"][sl]
    camera_off_frac = _fraction(camera_off)
    return {
        "pu_valid_frac": _fraction(quality_masks["pupil_valid"][sl]),
        "eyelid_valid_frac": _fraction(quality_masks["eyelid_valid"][sl]),
        "whisk_valid_frac": _fraction(quality_masks["whisk_valid"][sl]),
        "camera_off_frac": camera_off_frac,
        "camera_off_epoch": float(
            camera_off_frac >= config.video_quality.camera_off_min_frame_frac
        ),
    }


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
    ts_pu = np.asarray(pupil["timestamps"], dtype=float)
    quality_masks = _build_video_quality_masks(pupil, config)

    rows: list[dict] = []
    for ei in range(len(edges) - 1):
        t0, t1 = float(edges[ei]), float(edges[ei + 1])
        sl = segment_indices(ts_pu, t0, t1)
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
        pu_feats = _extract_pupil_features_for_epoch(
            pupil, quality_masks, sl, ts_pu, fs_pu, epoch_len_s
        )
        row.update(pu_feats)

        # Whisking features
        wh_feats = _extract_whisking_features_for_epoch(
            pupil, quality_masks, sl, fs_pu, epoch_len_s
        )
        row.update(wh_feats)
        row.update(_extract_video_quality_features_for_epoch(quality_masks, sl, config))

        rows.append(row)

    df = pl.DataFrame(rows, infer_schema_length=None)
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
            min_samples = max(3, win // 3)
            # Rolling mean
            col_name = f"{feat}_ctx{win}_mean"
            df = df.with_columns(
                pl.when(pl.col(feat).is_not_null())
                .then(
                    pl.col(feat)
                    .rolling_mean(window_size=win, min_samples=min_samples, center=True)
                    .over(session_col)
                )
                .otherwise(None)
                .alias(col_name)
            )
            # Rolling std for short window only
            if win == context_windows[0]:
                col_name_std = f"{feat}_ctx{win}_std"
                df = df.with_columns(
                    pl.when(pl.col(feat).is_not_null())
                    .then(
                        pl.col(feat)
                        .rolling_std(window_size=win, min_samples=min_samples, center=True)
                        .over(session_col)
                    )
                    .otherwise(None)
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
