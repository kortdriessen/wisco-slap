"""Lagged pairwise correlation.

Zero-lag Pearson r conflates two different things: how much two synapses
co-vary, and how tightly time-locked their co-variation is. If state
sharpens timing without changing magnitude, zero-lag r will rise even
though "the same amount" is happening — and conversely, a state that
shifts the relative phase of two synapses can drop zero-lag r without
reducing co-variation.

This module computes per-pair, per-bout cross-correlations within a
window of ±``max_lag_s`` and reports the peak |r| and the corresponding
lag. Aggregation across bouts uses the same Fisher-z weighted /
pooled-sums modes available in :mod:`wisco_slap.scope.corr.aggregate`.

Note: this is a coarse FFT-based zero-padded cross-correlation per pair,
which is fast but treats NaN samples as zeros after centering. We
restrict slicing to ``all_segments`` bouts (NaN-free by construction) at
the table-builder level so this is exact within each segment. Aggregation
across segments is then via Fisher-z weighting.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import polars as pl
import xarray as xr

from ... import get as _get
from ...util.validity.mask import validity_mask
from .aggregate import _r_to_z
from .bouts import all_segments_in_state, fixed_valid_bouts, state_hypno_bouts
from .core import _to_combined, _prep_combined
from .state_compare import (
    _classify_pair,
    _coord_to_float_array,
    _coord_to_int_array,
    _coord_to_str_array,
    _flatten_upper,
    _load_dn,
)

_BOUT_BUILDERS = {
    "fixed_valid": fixed_valid_bouts,
    "all_segments": all_segments_in_state,
    "hypno_bouts": state_hypno_bouts,
}
_DEFAULT_FIXED_BOUT_KWARGS = {
    "epoch_length": 10.0,
    "min_bout_length": 10.0,
    "max_nan_span": 2.0,
    "mode": "span",
}
_DEFAULT_ALL_SEGMENT_KWARGS = {"min_bout_length": 4.0}
_DEFAULT_HYPNO_BOUT_KWARGS = {"min_bout_length": 0.0}


def _peak_lagged_r_per_bout(
    sl: xr.DataArray,
    *,
    dt: float,
    max_lag_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute peak |r| and corresponding lag for every pair in one bout slice.

    Uses an FFT-based per-pair cross-correlation on z-scored, NaN→0
    traces. Pairs where either synapse has all-NaN over the bout get NaN.

    Returns
    -------
    peak_r : np.ndarray
        ``(n_syn, n_syn)`` peak Pearson correlation in the lag window
        ``[-max_lag_samples, +max_lag_samples]``.
    peak_lag_s : np.ndarray
        ``(n_syn, n_syn)`` lag at which peak_r occurs (in seconds; positive
        lag = j leads i, i.e. shift j back to align with i).
    n_samples : np.ndarray
        ``(n_syn, n_syn)`` per-pair time-base length (zero-lag joint count).
    """
    da = _prep_combined(sl)
    X = np.asarray(da.values, dtype=np.float64)  # (n_syn, n_time)
    n_syn, T = X.shape
    if T == 0 or n_syn < 2:
        return (
            np.full((n_syn, n_syn), np.nan),
            np.full((n_syn, n_syn), np.nan),
            np.zeros((n_syn, n_syn), dtype=np.int64),
        )
    # Per-synapse: drop NaN tail by setting NaN→0 after centering, but keep
    # variance estimates on non-NaN samples only.
    valid = ~np.isnan(X)
    n_valid_per_syn = valid.sum(axis=1)
    # Center each row over its non-NaN entries; replace NaN with 0.
    with np.errstate(invalid="ignore"):
        mu = np.where(
            n_valid_per_syn > 0,
            np.nansum(X, axis=1) / np.maximum(n_valid_per_syn, 1),
            0.0,
        )
    Xc = np.where(valid, X - mu[:, None], 0.0)
    # Per-row std over non-NaN samples (population, ddof=0 for stability).
    var = np.where(
        n_valid_per_syn > 1,
        np.nansum(Xc**2, axis=1) / np.maximum(n_valid_per_syn, 1),
        0.0,
    )
    sd = np.sqrt(var)
    # Per-pair joint sample count (intersection of validities).
    M = valid.astype(np.float64)
    N_pair = (M @ M.T).astype(np.int64)

    # FFT cross-correlation across all pairs at once.
    n_fft = 1
    while n_fft < 2 * T:
        n_fft *= 2
    F = np.fft.rfft(Xc, n=n_fft, axis=1)
    # cov[i,j,lag] = sum_t Xc[i,t] * Xc[j, t + lag]
    # = IFFT( F[i] * conj(F[j]) ); we'll center the result around lag=0.
    # Compute pair-by-pair via matrix product over freq dim.
    # F: (n_syn, n_freq). cov_full has shape (n_syn, n_syn, n_fft).
    # Build it via outer product per frequency, then IFFT.
    # For typical n_syn ~50–100 this is O(n_syn^2 * n_fft) and ok in memory.
    cov_freq = F[:, None, :] * np.conj(F[None, :, :])  # (n_syn, n_syn, n_freq)
    cov_full = np.fft.irfft(cov_freq, n=n_fft, axis=2)  # (n_syn, n_syn, n_fft)
    # cov_full[..., 0] is lag=0; cov_full[..., k] is lag=+k for k <= T-1; the
    # negative lags live at the end (k = n_fft - L for L > 0 means lag = -L).
    # Keep lags ∈ [-max_lag_samples, +max_lag_samples].
    L = max(int(max_lag_samples), 0)
    if L >= T:
        L = T - 1
    pos = cov_full[..., : L + 1]                              # lags 0..L
    neg = cov_full[..., n_fft - L : n_fft] if L > 0 else np.zeros(  # lags -L..-1
        cov_full.shape[:2] + (0,), dtype=cov_full.dtype
    )
    cov_window = np.concatenate([neg, pos], axis=2)         # lags -L..+L
    lags = np.arange(-L, L + 1, dtype=np.int64)
    # Normalise: r[i,j,lag] = cov[i,j,lag] / (n_eff * sd_i * sd_j)
    # We use a single normaliser N_pair[i,j] (zero-lag count) and the
    # per-row sd. This is the standard "Pearson-cross-correlation"
    # normalization; for short lags (max_lag << T), the bias from missing
    # tail samples is small.
    denom = np.outer(sd, sd) * np.maximum(N_pair, 1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_window = cov_window / denom[..., None]
    r_window = np.where(
        (N_pair[..., None] > 1) & (np.outer(sd, sd)[..., None] > 0),
        r_window,
        np.nan,
    )
    # Find peak |r| per pair, store r at that lag (signed).
    abs_r = np.abs(r_window)
    # Replace NaNs with -inf so they don't win argmax.
    abs_safe = np.where(np.isnan(abs_r), -np.inf, abs_r)
    peak_idx = np.argmax(abs_safe, axis=2)
    ii, jj = np.indices(peak_idx.shape)
    peak_r = r_window[ii, jj, peak_idx]
    peak_lag_s = lags[peak_idx].astype(np.float64) * dt
    # Pairs where everything was NaN should remain NaN.
    all_nan = np.all(np.isnan(r_window), axis=2)
    peak_r = np.where(all_nan, np.nan, peak_r)
    peak_lag_s = np.where(all_nan, np.nan, peak_lag_s)
    return peak_r, peak_lag_s, N_pair


def lagged_pairwise_corr(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    *,
    states: tuple[str, ...] = ("NREM", "Wake"),
    max_lag_s: float = 2.0,
    bout_strategy: str = "all_segments",
    bout_kwargs: dict | None = None,
    min_bout_length_s: float | None = None,
    channel: int = 0,
    trace: str = "denoised",
    validity_mode: str = "all",
) -> pl.DataFrame:
    """Per-pair, per-state lagged cross-correlation aggregated across bouts.

    For each bout, computes cross-correlations on z-scored, mean-centered
    traces and finds the peak |r| and corresponding lag in the window
    ``[-max_lag_s, +max_lag_s]``. Across bouts, aggregates Fisher-z
    weighted by ``(n_b − 3)`` and returns the median lag.

    Parameters
    ----------
    subject, exp, loc, acq, soma_id : str
    states : tuple of str
    max_lag_s : float
        Lag-search half-window in seconds. Default 2.0.
    bout_strategy : str
        ``"all_segments"`` (default — NaN-free, exact) or
        ``"fixed_valid"`` (interior-NaN-tolerant).
    bout_kwargs : dict | None
    min_bout_length_s : float | None
        If provided, drops bouts shorter than this. Default: 4 s for
        all_segments, 10 s for fixed_valid.
    channel, trace, validity_mode : standard.

    Returns
    -------
    pl.DataFrame
        One row per (pair, state). Columns: standard pair metadata
        (``subject, exp, loc, acq, soma_id, cell_id, recording_id, syn_i,
        syn_j, dmd_i, dmd_j, dend_i, dend_j, dend_type_i, dend_type_j,
        soma_i, soma_j, pair_type, same_type, dend_pair``) plus ``state``,
        ``peak_r`` (Fisher-z-weighted average across bouts, then ``tanh``),
        ``peak_z`` (Fisher-z), ``peak_se_z``, ``peak_lag_s_median`` (median
        across bouts of the lag at which peak |r| was achieved),
        ``n_samples`` (sum of per-bout joint-valid sample counts),
        ``n_bouts`` (bouts contributing).
    """
    if bout_strategy not in _BOUT_BUILDERS:
        raise ValueError(
            f"bout_strategy must be one of {list(_BOUT_BUILDERS)}, got {bout_strategy!r}"
        )

    recording_id = f"{subject}|{exp}|{loc}|{acq}"
    cell_id = f"{recording_id}|{soma_id}"

    dn = _load_dn(subject, exp, loc, acq, soma_id, channel=channel, trace=trace)
    if dn.sizes.get("syn_id", 0) < 2:
        raise ValueError("Need ≥2 synapses to correlate.")
    val_mask = validity_mask(dn, mode=validity_mode)
    hypno, *_ = _get.acq_sleep_coverage(subject, exp, loc, acq)

    # Estimate dt from the time axis.
    times = np.asarray(dn["time"].values, dtype=np.float64)
    if times.size < 2:
        raise ValueError("Insufficient time samples to estimate dt.")
    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Bad dt={dt!r}; cannot compute lagged correlation.")
    max_lag_samples = int(round(max_lag_s / dt))

    builder = _BOUT_BUILDERS[bout_strategy]
    if bout_strategy == "fixed_valid":
        eff_kwargs = {**_DEFAULT_FIXED_BOUT_KWARGS, **(bout_kwargs or {})}
    elif bout_strategy == "all_segments":
        eff_kwargs = {**_DEFAULT_ALL_SEGMENT_KWARGS, **(bout_kwargs or {})}
    else:
        eff_kwargs = {**_DEFAULT_HYPNO_BOUT_KWARGS, **(bout_kwargs or {})}

    if min_bout_length_s is not None:
        eff_kwargs["min_bout_length"] = float(min_bout_length_s)

    n_syn = dn.sizes["syn_id"]
    syn_ids = np.asarray(dn["syn_id"].values, dtype=str)
    dmds = _coord_to_int_array(dn, "dmd", n_syn)
    dends = _coord_to_str_array(dn, "dend-ID", n_syn)
    dend_types = _coord_to_str_array(dn, "dend_type", n_syn)
    somas = _coord_to_str_array(dn, "soma-ID", n_syn)
    poses = _coord_to_float_array(dn, "pos", n_syn)

    combined = _to_combined(dn)

    out_dfs: list[pl.DataFrame] = []
    for state in states:
        bouts = builder(hypno, val_mask, state, **eff_kwargs)
        if bouts.height == 0:
            continue
        # Aggregate Fisher-z-weighted peak r and median peak lag across bouts.
        weight_sum = np.zeros((n_syn, n_syn), dtype=np.float64)
        weighted_z = np.zeros((n_syn, n_syn), dtype=np.float64)
        n_total = np.zeros((n_syn, n_syn), dtype=np.int64)
        bout_count = np.zeros((n_syn, n_syn), dtype=np.int64)
        per_bout_lag = np.full((bouts.height, n_syn, n_syn), np.nan)
        for bi, b in enumerate(bouts.iter_rows(named=True)):
            sl = combined.sel(time=slice(float(b["start_time"]), float(b["end_time"])))
            if sl.sizes.get("time", 0) < 4:
                continue
            peak_r, peak_lag_s, N_pair = _peak_lagged_r_per_bout(
                sl, dt=dt, max_lag_samples=max_lag_samples,
            )
            z_b = _r_to_z(peak_r)
            valid_pair = (N_pair > 3) & np.isfinite(z_b)
            eff_n = np.where(valid_pair, (N_pair - 3).astype(np.float64), 0.0)
            weighted_z += np.where(valid_pair, eff_n * z_b, 0.0)
            weight_sum += eff_n
            n_total += np.where(valid_pair, N_pair, 0)
            bout_count += valid_pair.astype(np.int64)
            per_bout_lag[bi] = np.where(valid_pair, peak_lag_s, np.nan)

        with np.errstate(invalid="ignore", divide="ignore"):
            z_avg = np.where(weight_sum > 0, weighted_z / weight_sum, np.nan)
            se_z = np.where(weight_sum > 0, 1.0 / np.sqrt(weight_sum), np.nan)
        r_avg = np.tanh(z_avg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lag_med = np.nanmedian(per_bout_lag, axis=0)

        # Build (n_syn, n_syn) DataArrays so we can reuse _flatten_upper.
        from .core import _make_pair_coords  # local import to avoid cycle
        coords = _make_pair_coords(_prep_combined(combined))
        r_xr = xr.DataArray(r_avg, dims=("syn_1", "syn_2"), coords=coords)
        n_xr = xr.DataArray(n_total, dims=("syn_1", "syn_2"), coords=coords)
        z_xr = xr.DataArray(z_avg, dims=("syn_1", "syn_2"), coords=coords)
        se_z_xr = xr.DataArray(se_z, dims=("syn_1", "syn_2"), coords=coords)
        lag_xr = xr.DataArray(lag_med, dims=("syn_1", "syn_2"), coords=coords)

        flat = _flatten_upper(
            r_xr, n_xr, z_xr, se_z_xr,
            syn_ids=syn_ids, dmds=dmds, dends=dends,
            dend_types=dend_types, somas=somas, poses=poses,
        )
        iu, ju = np.triu_indices(n_syn, k=1)
        n_pairs = len(flat["syn_i"])

        pair_type = np.empty(n_pairs, dtype=object)
        same_type = np.empty(n_pairs, dtype=object)
        dend_pair = np.empty(n_pairs, dtype=object)
        for k in range(n_pairs):
            pt, st = _classify_pair(
                flat["dend_i"][k], flat["dend_j"][k],
                flat["dend_type_i"][k], flat["dend_type_j"][k],
            )
            pair_type[k] = pt
            same_type[k] = st
            dend_pair[k] = "|".join(sorted([str(flat["dend_i"][k]), str(flat["dend_j"][k])]))

        df = pl.DataFrame({
            "subject": [subject] * n_pairs,
            "exp": [exp] * n_pairs,
            "loc": [loc] * n_pairs,
            "acq": [acq] * n_pairs,
            "soma_id": [soma_id] * n_pairs,
            "cell_id": [cell_id] * n_pairs,
            "recording_id": [recording_id] * n_pairs,
            "state": [state] * n_pairs,
            "syn_i": flat["syn_i"],
            "syn_j": flat["syn_j"],
            "soma_i": flat["soma_i"],
            "soma_j": flat["soma_j"],
            "peak_r": flat["r"],
            "peak_z": flat["z"],
            "peak_se_z": flat["se_z"],
            "peak_lag_s_median": lag_xr.values[iu, ju],
            "n_samples": flat["n_samples"],
            "n_bouts": [int(b) for b in bout_count[iu, ju]],
            "dmd_i": flat["dmd_i"],
            "dmd_j": flat["dmd_j"],
            "dend_i": flat["dend_i"],
            "dend_j": flat["dend_j"],
            "dend_type_i": flat["dend_type_i"],
            "dend_type_j": flat["dend_type_j"],
            "pair_type": pair_type.astype(str),
            "same_type": same_type.astype(str),
            "dend_pair": dend_pair.astype(str),
            "max_lag_s": [float(max_lag_s)] * n_pairs,
            "bout_strategy": [bout_strategy] * n_pairs,
        })
        out_dfs.append(df)

    if not out_dfs:
        return _empty_lagged_df()
    return pl.concat(out_dfs, how="vertical")


def _empty_lagged_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "subject": pl.String, "exp": pl.String, "loc": pl.String,
            "acq": pl.String, "soma_id": pl.String, "cell_id": pl.String,
            "recording_id": pl.String, "state": pl.String,
            "syn_i": pl.String, "syn_j": pl.String,
            "soma_i": pl.String, "soma_j": pl.String,
            "peak_r": pl.Float64, "peak_z": pl.Float64,
            "peak_se_z": pl.Float64, "peak_lag_s_median": pl.Float64,
            "n_samples": pl.Int64, "n_bouts": pl.Int64,
            "dmd_i": pl.Int64, "dmd_j": pl.Int64,
            "dend_i": pl.String, "dend_j": pl.String,
            "dend_type_i": pl.String, "dend_type_j": pl.String,
            "pair_type": pl.String, "same_type": pl.String,
            "dend_pair": pl.String, "max_lag_s": pl.Float64,
            "bout_strategy": pl.String,
        }
    )


def lagged_pairwise_corr_multi(
    rows: pl.DataFrame | Iterable[dict],
    **kwargs,
) -> pl.DataFrame:
    """Run :func:`lagged_pairwise_corr` over many cells and concat."""
    if isinstance(rows, pl.DataFrame):
        iterator = list(rows.iter_rows(named=True))
    else:
        iterator = list(rows)
    out: list[pl.DataFrame] = []
    for row in iterator:
        try:
            df = lagged_pairwise_corr(
                row["subject"], row["exp"], row["loc"], row["acq"], row["soma_id"],
                **kwargs,
            )
            if df.height > 0:
                out.append(df)
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"lagged_pairwise_corr failed for "
                f"{row.get('subject')}/{row.get('exp')}/{row.get('loc')}/"
                f"{row.get('acq')}/{row.get('soma_id')}: "
                f"{type(e).__name__}: {e}",
                stacklevel=2,
            )
    if not out:
        return _empty_lagged_df()
    return pl.concat(out, how="vertical")
