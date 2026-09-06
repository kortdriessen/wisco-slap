"""Matched-pursuit calcium event detection on soma ROI traces (jRGECO1a).

Exposed as ``wis.scope.gecopur``. The calcium sibling of
``wis.scope.pursuit`` (:mod:`._GLUT_PURSUIT_PIPELINE`): same greedy
matched-pursuit principle, adapted to a slow indicator on a heavily
bleaching soma trace. Developed and validated against the 34 single-soma
acquisitions (``slap_mi_2_sleep/geco_pursuit_dev/``) and planted-event
synthetics (``tests/test_geco_pursuit.py``).

Usage
-----
>>> import wisco_slap as wis
>>> F = wis.scope.gecopur.load_soma_F(subject, exp, loc, acq)  # Fsvd ch0
>>> res = wis.scope.gecopur.detect_events(F)
>>> res.events        # one row per event: onset_time, peak_time, peak_dff, ...
>>> res.per_soma      # per-soma noise / bleaching / template / rate diagnostics
>>> res.fdr           # empirical false-discovery table vs z
>>> wis.scope.gecopur.plot_events(res, soma_id="soma1", t0=1000, t1=1060)
>>> wis.scope.gecopur.rates_at(res, z=6)    # re-slice at another threshold

Method summary
--------------
1. **Bleach-tracking baseline.** ``F0(t)`` = rolling low percentile
   (``baseline_percentile``) of the 1-s bin medians of F over
   ``baseline_window_s``, smoothed; ``dff = (F - F0) / F0``. The window is
   long relative to any calcium event (~1 s decay) and short relative to
   bleaching, so a 50x fluorescence decline (alnilam exp_1) and a flat
   recording both give a dF/F trace with a flat zero baseline.
2. **Gain stabilisation.** Per-sample noise ``sigma_dff(t)`` from the MAD
   of first differences in 1-s bins, median-filtered over
   ``sigma_window_s``; dF/F is divided by ``sigma_dff(t) / median`` before
   detection so the statistic is measured against the LOCAL noise (shot
   noise in dF/F units grows as ``1/sqrt(F0)`` as the soma bleaches).
3. **Matched filter.** Rise-and-decay template
   ``(1 - exp(-t/tau_rise)) exp(-t/tau_decay)``, plain (positive) kernel.
   Its filtered output can only go negative through noise — events are
   positive — which is what the sign-flip null relies on. The decay is
   ``"auto"`` by default: a first pass with a 0.5-s template, then each
   soma's decay is the median half-decay of its largest clean events
   (1/e decays span 0.15-1.5 s across somas, and a badly mismatched
   template leaves explaining-away residuals).
4. **Greedy matched pursuit.** Accept the largest filtered peak, subtract
   its template footprint (explaining-away resolves stacked events),
   ``min_sep_s`` refractory, repeat down to ``z_floor``.
5. **Peak / onset on the data.** From each pursuit onset the smoothed dF/F
   is searched forward (bounded by the next onset) for the first prominent
   local maximum = ``peak``; the minimum between the previous peak and
   this peak = ``foot``; robust levels are medians around foot / peak;
   ``onset`` = walking BACK from the peak, the first sample at or below
   ``foot + onset_frac * rise`` (10 % rise point). Structural rules: a
   component still rising when the next one starts is MERGED into it; a
   component whose rise is below ``min_rise_sigma`` background sigmas is
   DROPPED. The drop rule is essential — explaining-away residuals of big,
   imperfectly matched events reach z of 5-15 and a noise-only null
   cannot see them; requiring a real rise in the data removes them.
6. **Level correction and shot-noise deflation.** The positive kernel also
   integrates the level the trace already sits at. Tails of earlier events
   are explained away by the pursuit, so what remains is the UNEXPLAINED
   level (residual after the fitted events); its excess beyond the local
   quiet band — the regime the null is measured in — times the kernel's DC
   gain is subtracted from the event's amplitude and z. This is what
   removes bumps on the slow wander of a bleached-out tail (there that
   response is 4-5x the actual rise) without touching events on a flat
   baseline or stacked on explained tails. z is then divided by
   ``sqrt(1 + pre_dff)`` (capped) — the pure-shot upper bound on the noise
   increase at an elevated pre-event level.
7. **Empirical FDR.** The same statistic on sign-flipped quiet samples
   (whose whole forward template window stays near the level of the
   locally quietest bins) counts false positives by construction;
   ``res.fdr`` tabulates expected false events vs z and the operating
   point is the lowest z above which the set FDR stays <= ``fdr_alpha``.
   The z scale is soma- and time-specific: sigma_null is a ROLLING
   (5-min) lower-half MAD of the filtered trace on quiet samples, so z is
   measured against the local background (Fsvd traces carry slow
   fluctuations 10-50x the white per-sample noise, and a bleached-out tail
   wanders 10x more than the early trace). The FDR choice — not a fixed
   z — is what makes somas comparable.

Every event carries: ``onset_time`` / ``peak_time`` (data-derived),
``peak_dff`` (absolute dF/F at the peak), ``pre_dff`` (foot level),
``rise_dff = peak_dff - pre_dff`` (this event's own rise), ``amp`` (pursuit
template amplitude in dF/F), ``z`` / ``z_shot`` (detection statistics —
threshold on ``z_shot``), ``rise_sigma`` (data-side significance),
``rise_time_s``, ``half_decay_s`` (NaN when the next event interrupts the
decay), ``n_merged``, ``gap_bridged``, ``f0`` and ``f0_frac`` (baseline F
at the event, absolute and as a fraction of the soma's first-2-min
baseline — filter on this to drop events from bleached-out stretches,
where wander of a near-dead trace is all that is left).
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Literal, NamedTuple

import numpy as np
import polars as pl
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import fftconvolve, find_peaks

MAD_TO_SIGMA = 1.4826

#: Rolling window (s) for the per-sample noise estimate / gain profile.
SIGMA_WINDOW_S_DEFAULT = 120.0
#: Rolling window (s) for the bleach-tracking baseline percentile.
BASELINE_WINDOW_S_DEFAULT = 60.0
#: Clip range for the auto-estimated decay constant (s).
TAU_DECAY_AUTO_RANGE = (0.15, 2.0)
TAU_DECAY_FALLBACK_S = 0.5


class GecoPursuitResult(NamedTuple):
    """Everything :func:`detect_events` computes.

    events : one row per calcium event (catalog down to ``z_floor``; filter
        on ``z_shot`` yourself or via :func:`rates_at`).
    per_soma : per-soma diagnostics (noise, bleaching, template, quiet
        fraction, rate at ``z_star``).
    fdr : expected false events vs z threshold (all somas pooled).
    dff : bleach-corrected dF/F, dims ``(soma_id, time)``.
    f0 : the baseline used, same dims (raw F units).
    snr : matched-filter trace in ``sigma_null`` units (NaN where the
        detector cannot place an event) — the trace the pursuit runs on.
    params : resolved parameters (fs, z_star, per-soma tau, ...).
    """

    events: pl.DataFrame
    per_soma: pl.DataFrame
    fdr: pl.DataFrame
    dff: xr.DataArray
    f0: xr.DataArray
    snr: xr.DataArray
    params: dict


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------


def _as_rows(
    da: xr.DataArray, time_dim: str
) -> tuple[xr.DataArray, str, np.ndarray, np.ndarray, float]:
    """Coerce to a 2-D ``(soma_dim, time)`` DataArray (1-D input -> one row)."""
    if time_dim not in da.dims:
        raise ValueError(f"input must have a {time_dim!r} dim; got {da.dims}")
    if "channel" in da.dims and da.sizes["channel"] == 1:
        da = da.squeeze("channel")
    if da.ndim == 1:
        if "soma_id" in da.coords and da.coords["soma_id"].ndim == 0:
            da = da.expand_dims("soma_id")
        else:
            da = da.expand_dims({"soma_id": ["soma"]})
    if da.ndim != 2:
        raise ValueError(
            "input must be (soma_id, time) — select the channel first "
            f"(e.g. .sel(channel=0)); got dims {da.dims}"
        )
    soma_dim = next(d for d in da.dims if d != time_dim)
    da = da.transpose(soma_dim, time_dim)
    t = np.asarray(da[time_dim].values, dtype=np.float64)
    if t.size < 2:
        raise ValueError("need at least two samples")
    fs = 1.0 / float(np.median(np.diff(t)))
    ids = np.asarray(da[soma_dim].values).astype(str)
    return da, soma_dim, ids, t, fs


def _bin_rows(x: np.ndarray, nb: int) -> np.ndarray:
    """Reshape a 1-D array into ``(n_bins, nb)`` rows, NaN-padded at the end."""
    n_bins = int(np.ceil(x.size / nb))
    pad = n_bins * nb - x.size
    if pad:
        x = np.concatenate([x, np.full(pad, np.nan)])
    return x.reshape(n_bins, nb)


def _bin_centers(n_t: int, nb: int) -> np.ndarray:
    n_bins = int(np.ceil(n_t / nb))
    return (np.arange(n_bins) + 0.5) * nb - 0.5


def _rolling_nan(v: np.ndarray, win: int, stat: str, min_count: int, q: float = 0.5):
    """Centered rolling NaN-aware statistic over a short (binned) series."""
    win = max(1, int(win))
    if win % 2 == 0:
        win += 1
    half = win // 2
    padded = np.concatenate([np.full(half, np.nan), v, np.full(half, np.nan)])
    W = sliding_window_view(padded, win)
    cnt = np.isfinite(W).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if stat == "quantile":
            out = np.nanquantile(W, q, axis=1)
        elif stat == "median":
            out = np.nanmedian(W, axis=1)
        elif stat == "mean":
            out = np.nanmean(W, axis=1)
        else:
            raise ValueError(stat)
    out = np.asarray(out, dtype=np.float64)
    out[cnt < min_count] = np.nan
    return out


def _fill_nan_1d(v: np.ndarray) -> np.ndarray:
    """Linear interpolation / edge extension of NaNs (all-NaN stays NaN)."""
    fin = np.isfinite(v)
    if fin.all():
        return v
    if not fin.any():
        return v
    idx = np.arange(v.size)
    return np.interp(idx, idx[fin], v[fin])


def _bins_to_samples(vb: np.ndarray, n_t: int, nb: int) -> np.ndarray:
    return np.interp(np.arange(n_t), _bin_centers(n_t, nb), vb)


def _nan_boxcar(x: np.ndarray, w: int) -> np.ndarray:
    """Centered boxcar mean ignoring NaNs (NaN where fewer than half finite)."""
    w = max(1, int(w))
    if w == 1:
        return x.copy()
    fin = np.isfinite(x)
    k = np.ones(w)
    num = np.convolve(np.where(fin, x, 0.0), k, mode="same")
    cnt = np.convolve(fin.astype(np.float64), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / cnt
    out[cnt < max(1, w / 2)] = np.nan
    return out


def _bridge_gaps(x: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaN runs of length <= ``max_gap`` samples.

    Returns the filled array and a bool mask of the bridged samples. Longer
    runs (and runs touching either end) stay NaN.
    """
    out = x.copy()
    bridged = np.zeros(x.size, dtype=bool)
    bad = ~np.isfinite(x)
    if not bad.any() or max_gap < 1:
        return out, bridged
    edges = np.diff(np.r_[0, bad.astype(np.int8), 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # exclusive
    for s, e in zip(starts, ends, strict=True):
        if s == 0 or e >= x.size or (e - s) > max_gap:
            continue
        out[s:e] = np.interp(np.arange(s, e), [s - 1, e], [x[s - 1], x[e]])
        bridged[s:e] = True
    return out, bridged


def _pooled_window_stat(
    values: np.ndarray,
    mask: np.ndarray,
    nb_bin: int,
    n_pool: int,
    fn,
    min_count: int,
) -> np.ndarray:
    """Per-bin statistic ``fn(values[mask] pooled over the surrounding
    n_pool bins)``; NaN where fewer than ``min_count`` samples pool.

    Pooling raw samples over the whole window (rather than a statistic per
    small bin, then a rolling median of those) matters for autocorrelated
    series such as the matched-filtered trace: a 10-s bin of a 3-s-template
    filter holds a handful of independent values and its within-bin spread
    underestimates the true scale ~2x.
    """
    n_t = values.size
    n_bins = int(np.ceil(n_t / nb_bin))
    half = max(1, n_pool // 2)
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        lo = max(0, (b - half) * nb_bin)
        hi = min(n_t, (b + half + 1) * nb_bin)
        v = values[lo:hi][mask[lo:hi]]
        v = v[np.isfinite(v)]
        if v.size >= min_count:
            out[b] = fn(v)
    return out


def _lower_half_sigma(v: np.ndarray) -> float:
    """Robust sd from the lower half around the median (events positive)."""
    med = np.median(v)
    return MAD_TO_SIGMA * float(np.median(med - v[v <= med]))


def _neg_side_sigma(r: np.ndarray) -> float:
    """One-sided robust sd of residuals around a reference at zero."""
    neg = r[r <= 0]
    return MAD_TO_SIGMA * float(np.median(-neg)) if neg.size else np.nan


def _forward_all(mask: np.ndarray, klen: int) -> np.ndarray:
    """``out[t] = all(mask[t : t + klen])`` (window running off the end -> False)."""
    n_t = mask.size
    bad = (~mask).astype(np.int8)
    cov = np.convolve(bad, np.ones(klen), "full")[klen - 1 : klen - 1 + n_t]
    out = cov == 0
    if klen > 1:
        out[n_t - klen + 1 :] = False
    return out


def _matched_filter(arr_1d: np.ndarray, k_unit: np.ndarray) -> np.ndarray:
    """Causal correlation: ``m[t] = sum_k x[t + k] k_unit[k]``."""
    klen = k_unit.size
    full = fftconvolve(np.nan_to_num(arr_1d), k_unit[::-1], mode="full")
    return full[klen - 1 : klen - 1 + arr_1d.size]


def _template(
    tau_rise_s: float, tau_decay_s: float, fs: float, zero_mean: bool = False
) -> SimpleNamespace:
    """Zero-mean rise-and-decay kernel plus its explaining-away footprint.

    ``h_shape`` peaks at 1 so that a fitted amplitude is the template's peak
    dF/F. ``k_unit`` is the unit-norm ZERO-MEAN kernel the filter uses;
    ``e_norm = <h_shape, k_unit>`` converts filtered peak height to
    amplitude; ``footprint`` is the filtered response to one unit event
    (normalised to 1 at its peak, ``peak_off`` samples after its start).
    """
    tr = max(tau_rise_s * fs, 1e-3)
    td = max(tau_decay_s * fs, 1.0)
    klen = int(np.ceil(4 * tr + 4 * td))
    klen = max(klen, 8)
    k = np.arange(klen, dtype=np.float64)
    h = (1.0 - np.exp(-k / tr)) * np.exp(-k / td)
    h_shape = h / h.max()
    k0 = h_shape - h_shape.mean() if zero_mean else h_shape
    k_unit = k0 / np.linalg.norm(k0)
    e_norm = float(np.dot(h_shape, k_unit))
    padded = np.r_[np.zeros(klen), h_shape, np.zeros(klen)]
    foot_full = np.correlate(padded, k_unit, mode="valid")
    peak_off = int(np.argmax(foot_full))
    footprint = foot_full / foot_full[peak_off]
    return SimpleNamespace(
        klen=klen,
        h_shape=h_shape,
        k_unit=k_unit,
        e_norm=e_norm,
        footprint=footprint,
        peak_off=peak_off,
        rise_peak_off=int(np.argmax(h_shape)),
        snr_factor=float(np.linalg.norm(k0) / np.linalg.norm(h_shape)),
        dc_gain=float(k_unit.sum()),  # filtered response to a unit constant level
        zero_mean=zero_mean,
    )


# --------------------------------------------------------------------------
# baseline / dF/F
# --------------------------------------------------------------------------


def _slow_baseline(
    x: np.ndarray, nb: int, win_bins: int, smooth_bins: int, pct: float
) -> np.ndarray:
    n_t = x.size
    rows = _bin_rows(x, nb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bin_med = np.nanmedian(rows, axis=1)
    frac = np.isfinite(rows).mean(axis=1)
    bin_med[frac < 0.25] = np.nan
    if not np.isfinite(bin_med).any():
        return np.full(n_t, np.nan)
    base = _rolling_nan(
        bin_med, win_bins, "quantile", min_count=max(3, win_bins // 4), q=pct / 100
    )
    base = _fill_nan_1d(base)
    base = _rolling_nan(base, smooth_bins, "mean", min_count=1)
    return _bins_to_samples(base, n_t, nb)


def baseline_dff(
    F: xr.DataArray,
    baseline_window_s: float = BASELINE_WINDOW_S_DEFAULT,
    baseline_percentile: float = 10.0,
    baseline_smooth_s: float = 30.0,
    bin_s: float = 1.0,
    time_dim: str = "time",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Bleach-tracking baseline and dF/F for soma fluorescence traces.

    ``F0(t)`` is the rolling ``baseline_percentile`` of the ``bin_s``-bin
    medians of F over a centered ``baseline_window_s`` window, then
    smoothed with a ``baseline_smooth_s`` running mean and interpolated back
    to the sample grid. Binning first makes the percentile a percentile of
    *levels* rather than of noise samples, so the baseline sits at the
    quiet level, not below it.

    Returns ``(dff, f0)`` with the dims/coords of ``F`` (1-D input is
    promoted to a one-row ``(soma_id, time)`` array). NaN where F is NaN.
    """
    da, soma_dim, _ids, _t, fs = _as_rows(F, time_dim)
    X = np.asarray(da.values, dtype=np.float64)
    nb = max(1, int(round(bin_s * fs)))
    win_bins = max(3, int(round(baseline_window_s / bin_s)))
    smooth_bins = max(1, int(round(baseline_smooth_s / bin_s)))
    F0 = np.full_like(X, np.nan)
    for i in range(X.shape[0]):
        F0[i] = _slow_baseline(X[i], nb, win_bins, smooth_bins, baseline_percentile)
    nonpos = np.isfinite(F0) & (F0 <= 0)
    if nonpos.any():
        warnings.warn(
            f"baseline_dff: {int(nonpos.sum())} samples have F0 <= 0 "
            "(set to NaN) — is this really a fluorescence trace?",
            stacklevel=2,
        )
        F0[nonpos] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        dff = (X - F0) / F0
    dff[~np.isfinite(X)] = np.nan
    attrs = {
        "baseline_window_s": baseline_window_s,
        "baseline_percentile": baseline_percentile,
        "baseline_smooth_s": baseline_smooth_s,
    }
    dff_da = xr.DataArray(
        dff.astype(np.float32),
        dims=da.dims,
        coords=da.coords,
        name="dff",
        attrs={**attrs, "signal_type": "dFF"},
    )
    f0_da = xr.DataArray(
        F0.astype(np.float32),
        dims=da.dims,
        coords=da.coords,
        name="f0",
        attrs={**attrs, "signal_type": "F0"},
    )
    return dff_da, f0_da


# --------------------------------------------------------------------------
# noise profile / kinetics
# --------------------------------------------------------------------------


def _noise_profile(x: np.ndarray, nb: int, win_bins: int) -> np.ndarray:
    """Per-sample white-equivalent noise sigma of a dF/F trace.

    MAD of first differences within ``nb``-sample bins (robust to the slow
    calcium signal, whose per-sample increments are tiny outside the few
    rising samples), ``/ sqrt(2)`` for the differencing, rolling-median over
    ``win_bins`` bins (kills event-inflated bins), interpolated to samples.
    """
    d = np.diff(x)
    rows = _bin_rows(d, nb)
    fin = np.isfinite(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(rows, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(rows - med), axis=1)
    sig_b = MAD_TO_SIGMA * mad / np.sqrt(2.0)
    sig_b[fin.sum(axis=1) < max(4, nb // 2)] = np.nan
    sig_b = _rolling_nan(sig_b, win_bins, "median", min_count=max(2, win_bins // 4))
    sig_b = _fill_nan_1d(sig_b)
    if not np.isfinite(sig_b).any():
        return np.full(x.size, np.nan)
    return _bins_to_samples(sig_b, x.size, nb)


def _tau_from_events(
    rows: list[dict], xs: np.ndarray, fs: float, n_top: int = 25, max_s: float = 4.0
) -> tuple[float, int, float]:
    """Decay constant from the detector's own events.

    The post-peak dF/F of the ``n_top`` largest uninterrupted events (finite
    ``half_decay_s``) is normalised by each event's rise, AVERAGED, and the
    averaged curve's half-crossing time ``/ ln 2`` is the decay constant
    (clipped to :data:`TAU_DECAY_AUTO_RANGE`). Averaging first avoids the
    first-passage bias of per-event half-decays on noisy traces (a slowly
    decaying trace hovering near the half level crosses it early). Returns
    ``(tau_s, n_used, half_decay_s)``; the fallback when fewer than 5
    events qualify.
    """
    cand = [r for r in rows if np.isfinite(r["half_decay_s"]) and r["rise_dff"] > 0]
    if len(cand) < 5:
        return TAU_DECAY_FALLBACK_S, len(cand), np.nan
    cand.sort(key=lambda r: -r["rise_dff"])
    cand = cand[:n_top]
    L = int(max_s * fs)
    segs = np.full((len(cand), L), np.nan)
    for k, r in enumerate(cand):
        seg = (xs[r["idx_peak"] : r["idx_peak"] + L] - r["pre_dff"]) / r["rise_dff"]
        segs[k, : seg.size] = seg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_curve = np.nanmean(segs, axis=0)
    below = np.flatnonzero(np.nan_to_num(mean_curve, nan=np.inf) <= 0.5)
    if below.size == 0:
        return TAU_DECAY_FALLBACK_S, len(cand), np.nan
    hd = float(below[0] / fs)
    tau = float(np.clip(hd / np.log(2.0), *TAU_DECAY_AUTO_RANGE))
    return tau, len(cand), hd


def estimate_kinetics(
    dff: xr.DataArray,
    n_top: int = 25,
    time_dim: str = "time",
    **kwargs,
) -> pl.DataFrame:
    """Per-soma event kinetics from a first-pass detection on a dF/F trace
    (template decay fixed at :data:`TAU_DECAY_FALLBACK_S`) — the estimate
    :func:`detect_events` uses for ``tau_decay_s="auto"``. Columns:
    ``soma_id``, ``n_events`` (at the first pass's operating point),
    ``n_used`` (largest uninterrupted events entering the median),
    ``rise_time_s`` / ``half_decay_s`` (medians over ``n_used``),
    ``tau_decay_s`` (the clipped template estimate), ``rise_dff`` (median
    rise of those events). ``kwargs`` go to :func:`detect_events`.
    """
    res = detect_events(
        dff,
        input_is_dff=True,
        tau_decay_s=TAU_DECAY_FALLBACK_S,
        time_dim=time_dim,
        verbose=False,
        **kwargs,
    )
    out = []
    fs = res.params["fs"]
    smooth_w = max(1, int(round(res.params["smooth_s"] * fs)))
    for i, sid in enumerate(np.asarray(res.dff["soma_id"].values).astype(str)):
        ev = res.events.filter(
            (pl.col("soma_id") == sid) & (pl.col("z_shot") >= res.params["z_star"])
        )
        xs = _nan_boxcar(np.asarray(res.dff.values[i], dtype=np.float64), smooth_w)
        tau, n_used, hd = _tau_from_events(ev.to_dicts(), xs, fs, n_top)
        fin = (
            ev
            .filter(pl.col("half_decay_s").is_finite())
            .sort("rise_dff", descending=True)
            .head(n_top)
        )
        out.append({
            "soma_id": sid,
            "n_events": ev.height,
            "n_used": n_used,
            "rise_time_s": float(fin["rise_time_s"].median()) if fin.height else np.nan,
            "half_decay_s": hd,
            "tau_decay_s": tau,
            "rise_dff": float(fin["rise_dff"].median()) if fin.height else np.nan,
        })
    return pl.DataFrame(out)


# --------------------------------------------------------------------------
# core: per-soma preparation
# --------------------------------------------------------------------------


def _prepare_soma(
    dff_1d: np.ndarray,
    fs: float,
    *,
    tau_rise_s: float,
    tau_decay: float,
    sigma_window_s: float,
    quiet_window_s: float,
    quiet_level_s: float,
    quiet_k: float,
    quiet_ref_frac: float,
    null_window_s: float,
    bg_window_s: float,
    max_gap_s: float,
    smooth_s: float,
    zero_mean: bool,
) -> SimpleNamespace:
    n_t = dff_1d.size
    nb = max(1, int(round(fs)))
    smooth_w = max(1, int(round(smooth_s * fs)))
    xs = _nan_boxcar(dff_1d, smooth_w)  # smoothed dF/F (data units)

    # ---- noise profile & gain stabilisation ------------------------------
    sig = _noise_profile(dff_1d, nb, max(3, int(round(sigma_window_s))))
    sig_scalar = float(np.nanmedian(sig)) if np.isfinite(sig).any() else np.nan
    if not np.isfinite(sig_scalar) or sig_scalar <= 0:
        return SimpleNamespace(ok=False, n_t=n_t)
    gain = np.clip(sig / sig_scalar, 0.2, 5.0)
    gain = np.where(np.isfinite(gain), gain, 1.0)

    # ---- template ---------------------------------------------------------
    td = float(tau_decay)
    T = _template(tau_rise_s, td, fs, zero_mean=zero_mean)
    if T.klen >= n_t:
        raise ValueError("trace shorter than the template")

    # ---- stabilised, gap-bridged trace and its matched filter -------------
    x_raw = dff_1d / gain
    x, bridged = _bridge_gaps(x_raw, max(0, int(round(max_gap_s * fs))))
    finite = np.isfinite(x)
    valid = _forward_all(finite, T.klen)
    m = _matched_filter(x, T.k_unit)

    # ---- quiet mask (rolling) ----------------------------------------------
    # The glutamate pipeline reads quietness off the noise-free denoised
    # trace; here the level is a ``quiet_level_s`` boxcar of the stabilised
    # trace. Everything is LOCAL (``null_window_s`` windows): the reference
    # is the rolling ``quiet_ref_frac`` quantile of the 1-s bin levels (the
    # quietest bins around each time), sigma_level the rolling one-sided
    # robust sd of the level around that reference inside those quiet bins.
    # A sample is quiet when its level is within ``quiet_k`` level-sigmas of
    # the reference, and a detection sample is quiet when its whole template
    # window is. Events (positive) are excluded; noise passes ~99 % per
    # level sample, so the selection barely touches the NEGATIVE tail of m
    # that sigma_null and the FDR null are read from. Local rather than
    # global so that a bleached-out tail, where slow wander of a near-dead
    # trace is 10x the early background, gets its own (large) scale instead
    # of being measured against the early one.
    lvl_w = max(smooth_w, int(round(quiet_level_s * fs)))
    level = _nan_boxcar(x, lvl_w)
    rows = _bin_rows(level, nb)
    n_bins = rows.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lvl_b = np.nanmedian(rows, axis=1)
    win_q = max(5, int(round(null_window_s / 1.0)))
    ref_b = _rolling_nan(
        lvl_b, win_q, "quantile", min_count=max(5, win_q // 10), q=quiet_ref_frac
    )
    ref_b = _fill_nan_1d(ref_b)
    if not np.isfinite(ref_b).any():
        ref_b = np.zeros(n_bins)
    ref = _bins_to_samples(ref_b, n_t, nb)
    # one-sided sd of the level around the reference, pooled over the
    # window's locally quiet bins (level at or below the reference)
    in_quiet_bin = np.repeat(lvl_b <= ref_b, nb)[:n_t] & np.isfinite(level)
    nb_pool = max(nb, int(round(30 * fs)))
    n_pool = max(3, int(round(null_window_s / (nb_pool / fs))))
    sl_b = _pooled_window_stat(
        level - ref, in_quiet_bin, nb_pool, n_pool, _neg_side_sigma, min_count=200
    )
    sl_b = _fill_nan_1d(sl_b)
    if not np.isfinite(sl_b).any():
        sl_b = np.full(sl_b.size, sig_scalar / np.sqrt(lvl_w))
    sigma_level_t = _bins_to_samples(sl_b, n_t, nb_pool)
    sigma_level = float(np.nanmedian(sigma_level_t))
    resid = level - ref
    quiet_pt = np.isfinite(resid) & (resid <= quiet_k * sigma_level_t)
    quiet = valid & _forward_all(quiet_pt, T.klen)

    # ---- sigma_null(t): rolling lower-half MAD of m on quiet samples --------
    # pooled over the full window (see _pooled_window_stat)
    nb_n = nb_pool
    sn_b = _pooled_window_stat(
        m, quiet, nb_n, n_pool, _lower_half_sigma, min_count=1000
    )
    sn_b = _fill_nan_1d(sn_b)
    if np.isfinite(sn_b).any() and np.nanmedian(sn_b) > 0:
        null_source = "quiet_rolling"
    else:
        # hyperactive soma with no quiet windows: lower half of everything
        v = m[valid]
        if v.size < 100:
            return SimpleNamespace(ok=False, n_t=n_t)
        sn_b = np.full(sn_b.size, _lower_half_sigma(v))
        null_source = "all_valid_lower_half"
    sigma_null_t = _bins_to_samples(sn_b, n_t, nb_n)
    sigma_null_t = np.where(
        np.isfinite(sigma_null_t) & (sigma_null_t > 0), sigma_null_t, np.nan
    )
    if not np.isfinite(sigma_null_t).any():
        return SimpleNamespace(ok=False, n_t=n_t)
    sigma_null_t = _fill_nan_1d(sigma_null_t)
    sigma_null = float(np.nanmedian(sigma_null_t))
    zt = m / sigma_null_t

    # ---- background-fluctuation yardstick for the refinement step ----------
    # One-sided robust sd of the smoothed trace around its ``bg_window_s``
    # local median (events are positive, so the lower side is background
    # only): white noise plus the coloured sub-second fluctuations of Fsvd
    # traces (10-50x the white per-sample sigma) — what a candidate's rise
    # must beat to count as "an event in the data".
    xs_stab = _nan_boxcar(x, smooth_w)
    nb_bg = max(1, int(round(0.1 * fs)))
    rows_bg = _bin_rows(xs_stab, nb_bg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bg_b = np.nanmedian(rows_bg, axis=1)
    win_bg = max(3, int(round(bg_window_s / 0.1)))
    bg_b = _fill_nan_1d(
        _rolling_nan(bg_b, win_bg, "median", min_count=max(2, win_bg // 4))
    )
    bg = (
        _bins_to_samples(bg_b, n_t, nb_bg) if np.isfinite(bg_b).any() else np.zeros(n_t)
    )
    rf = xs_stab - bg
    rf = rf[np.isfinite(rf)]
    if rf.size > 100:
        mrf = float(np.median(rf))
        sigma_bg = MAD_TO_SIGMA * float(np.median(mrf - rf[rf <= mrf]))
    else:
        sigma_bg = np.nan
    if not np.isfinite(sigma_bg) or sigma_bg <= 0:
        sigma_bg = sig_scalar / np.sqrt(smooth_w)

    return SimpleNamespace(
        ok=True,
        n_t=n_t,
        fs=fs,
        xs=xs,
        level=level,
        ref=ref,
        sigma_level_t=sigma_level_t,
        lvl_w=lvl_w,
        quiet_k=quiet_k,
        sig=sig,
        sig_scalar=sig_scalar,
        gain=gain,
        T=T,
        tau_decay_s=td,
        x=x,
        bridged=bridged,
        finite=finite,
        valid=valid,
        m=m,
        quiet=quiet,
        sigma_level=sigma_level,
        sigma_bg=sigma_bg,
        sigma_null=sigma_null,
        sigma_null_t=sigma_null_t,
        zt=zt,
        null_source=null_source,
        smooth_w=smooth_w,
    )


def _pursue(
    P: SimpleNamespace, z_floor: float, min_sep: int
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy matched pursuit on one soma, on the z-normalised filtered
    trace ``m / sigma_null(t)``. Returns (onset_idx, amp_stab) sorted."""
    T = P.T
    z = P.zt.astype(np.float64).copy()
    z[~P.valid] = -np.inf
    sn = P.sigma_null_t
    n_t = z.size
    acc_i: list[int] = []
    acc_a: list[float] = []
    for _sweep in range(4):
        pk, _ = find_peaks(z, height=z_floor)
        if pk.size == 0:
            break
        changed = False
        for p in pk[np.argsort(-z[pk])]:
            val = z[p]
            if not np.isfinite(val) or val < z_floor:
                continue
            lo, hi = max(0, p - 1), min(n_t, p + 2)
            if val < np.max(z[lo:hi]) - 1e-9:
                continue  # no longer a local max after earlier subtractions
            amp = val * sn[p] / T.e_norm
            a0 = p - T.peak_off
            b0 = a0 + T.footprint.size
            fa, fb = max(0, a0), min(n_t, b0)
            z[fa:fb] -= amp * T.e_norm * T.footprint[fa - a0 : fb - a0] / sn[fa:fb]
            z[max(0, p - min_sep + 1) : p + min_sep] = -np.inf
            acc_i.append(int(p))
            acc_a.append(float(amp))
            changed = True
        if not changed:
            break
    idx = np.array(acc_i, dtype=int)
    amp = np.array(acc_a, dtype=np.float64)
    order = np.argsort(idx)
    return idx[order], amp[order]


def _unexplained_level(
    P: SimpleNamespace, idx: np.ndarray, amp_stab: np.ndarray
) -> np.ndarray:
    """Local level (``quiet_level_s`` boxcar) of the stabilised trace after
    subtracting the fitted pursuit events — what the pursuit did NOT explain
    (baseline error, plateaus, slow wander)."""
    model = np.zeros(P.n_t)
    h = P.T.h_shape
    for i0, a in zip(idx, amp_stab, strict=True):
        seg = h[: P.n_t - i0]
        model[i0 : i0 + seg.size] += a * seg
    return _nan_boxcar(P.x - model, P.lvl_w)


def _refine(
    P: SimpleNamespace,
    idx: np.ndarray,
    amp_stab: np.ndarray,
    t: np.ndarray,
    *,
    max_rise_s: float,
    pre_window_s: float,
    onset_frac: float,
    prom_frac: float,
    min_rise_sigma: float,
    level_s: float,
    shot_correction: bool,
    shot_cap: float,
    f0_1d: np.ndarray,
) -> list[dict]:
    """Peak / foot / onset on the smoothed dF/F for each pursuit component,
    with the merge (still rising into the next component) and drop (no rise)
    rules. See the module docstring, step 5."""
    xs, fs, n_t = P.xs, P.fs, P.n_t
    T = P.T
    P.level_resid = _unexplained_level(P, idx, amp_stab)
    max_rise = max(2, int(round(max_rise_s * fs)))
    pre_win = max(1, int(round(pre_window_s * fs)))
    w_lvl = max(1, int(round(level_s * fs)))  # half-window for robust levels
    w_pk = max(1, w_lvl // 4)
    n = idx.size
    onset = [int(i) for i in idx]
    amps = [float(a) for a in amp_stab]
    n_merged = [1] * n
    rows: list[dict] = []
    prev_peak = -1
    j = 0
    while j < n:
        i0 = onset[j]
        nxt = onset[j + 1] if j + 1 < n else n_t
        hi = min(nxt, i0 + max_rise, n_t)
        seg = xs[i0:hi]
        if seg.size < 2 or not np.isfinite(seg).any():
            j += 1
            continue
        lo = max(prev_peak + 1, i0 - pre_win, 0)
        pre_seg = xs[lo:hi]
        if not np.isfinite(pre_seg).any():
            j += 1
            continue
        foot0 = float(np.nanmin(pre_seg))
        rise_max = float(np.nanmax(seg)) - foot0
        # noise yardstick in dF/F units at this time: the robust sd of the
        # 0.5-s level (captures the coloured, slow background fluctuations
        # that the white per-sample sigma misses by 10-50x on Fsvd traces)
        sig_bg = float(P.sigma_bg * P.gain[i0])
        tol = max(prom_frac * rise_max, 2.0 * sig_bg)
        pk, _ = find_peaks(np.nan_to_num(seg, nan=-np.inf), prominence=tol)
        if pk.size:
            p = i0 + int(pk[0])
        else:
            imax = i0 + int(np.nanargmax(seg))
            end_val = xs[hi - 1]
            still_rising = (
                nxt < n_t
                and hi == nxt
                and np.isfinite(end_val)
                and end_val >= float(np.nanmax(seg)) - tol
                and rise_max > tol
            )
            if still_rising:
                # over-split slow rise: fold this component into the next one
                onset[j + 1] = i0
                amps[j + 1] += amps[j]
                n_merged[j + 1] += n_merged[j]
                j += 1
                continue
            if rise_max <= tol:
                j += 1  # no rise of its own: explaining-away residual
                continue
            p = imax
        pre_seg = xs[lo : p + 1]
        foot_i = lo + int(np.nanargmin(pre_seg))
        # Robust levels: the sample minimum / maximum are range statistics
        # (biased by ~2-3 noise sd each), so pre-event level = median of a
        # +-level_s window centred on the foot (clipped to [lo, p]) and peak
        # level = median of a quarter-size window around the peak.
        foot = float(
            np.nanmedian(xs[max(lo, foot_i - w_lvl) : min(p, foot_i + w_lvl) + 1])
        )
        peak = float(np.nanmedian(xs[max(foot_i, p - w_pk) : min(n_t, p + w_pk + 1)]))
        rise = peak - foot
        rise_sigma = rise / sig_bg if sig_bg > 0 else np.inf
        if not (rise > 0) or rise_sigma < min_rise_sigma:
            j += 1  # no significant rise of its own in the data
            continue
        # onset: walk BACK from the peak to the first sample at or below the
        # onset level. Searching forward from the foot would trigger on the
        # first noise blip after a noise trough (biases onsets ~0.2 s early
        # on noisy traces); from the peak side the crossing is on the rise.
        lvl = foot + onset_frac * rise
        seg2 = np.nan_to_num(xs[foot_i : p + 1], nan=np.inf)
        under = np.flatnonzero(seg2 <= lvl)
        on = foot_i + int(under[-1]) if under.size else foot_i
        # Level correction. The (positive) kernel integrates whatever level
        # the trace sits at, so a candidate on an elevated level has that
        # level's response folded into its pursuit amplitude — on the slow
        # wander of a bleached-out tail 4-5x the actual rise. Tails of
        # earlier events are already explained away by the pursuit, so the
        # relevant level is the UNEXPLAINED one (residual after the fitted
        # events), and only its excess beyond the local quiet band (what
        # the null is measured in) is removed. Healthy events on a flat
        # baseline and stacked events on explained tails are untouched.
        excess = max(
            float(P.level_resid[foot_i] - P.ref[foot_i])
            - P.quiet_k * float(P.sigma_level_t[foot_i]),
            0.0,
        )
        amp_stab_ev = max(amps[j] - excess * T.dc_gain / T.e_norm, 0.0)
        amp_dff = amp_stab_ev * float(P.gain[i0])
        z = amp_stab_ev * T.e_norm / P.sigma_null_t[i0]
        if shot_correction:
            infl = min(np.sqrt(1.0 + max(foot, 0.0)), shot_cap)
        else:
            infl = 1.0
        rows.append({
            "idx_onset": int(on),
            "idx_peak": int(p),
            "idx_pursuit": int(i0),
            "onset_time": float(t[on]),
            "peak_time": float(t[p]),
            "peak_dff": peak,
            "pre_dff": foot,
            "rise_dff": rise,
            "rise_sigma": float(rise_sigma),
            "amp": amp_dff,
            "z": float(z),
            "z_shot": float(z / infl),
            "rise_time_s": float((p - on) / fs),
            "half_decay_s": np.nan,  # filled below, needs the ACCEPTED neighbour
            "n_merged": int(n_merged[j]),
            "gap_bridged": bool(P.bridged[on : p + 1].any()),
            "f0": float(f0_1d[p]),
        })
        prev_peak = p
        j += 1
    # Half-decay: time from the peak down to foot + rise/2, NaN when the next
    # ACCEPTED event starts first. (Bounding by the next pursuit component
    # instead would censor most slow decays: mismatched big events spawn
    # residual components on their own tails that are dropped above.)
    for k, r in enumerate(rows):
        p = r["idx_peak"]
        nxt_acc = rows[k + 1]["idx_onset"] if k + 1 < len(rows) else n_t
        tail = xs[p:nxt_acc]
        lvl = r["pre_dff"] + 0.5 * r["rise_dff"]
        below = np.flatnonzero(np.nan_to_num(tail, nan=np.inf) <= lvl)
        r["half_decay_s"] = float(below[0] / fs) if below.size else np.nan
    return rows


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def detect_events(
    F: xr.DataArray,
    *,
    input_is_dff: bool = False,
    tau_rise_s: float = 0.08,
    tau_decay_s: float | Literal["auto"] = "auto",
    z_threshold: float | None = None,
    z_floor: float = 3.0,
    fdr_alpha: float = 0.05,
    min_sep_s: float = 0.1,
    max_rise_s: float = 1.5,
    pre_window_s: float = 0.3,
    onset_frac: float = 0.1,
    prom_frac: float = 0.15,
    min_rise_sigma: float = 4.0,
    bg_window_s: float = 0.5,
    level_s: float = 0.1,
    smooth_s: float = 0.05,
    max_gap_s: float = 0.5,
    sigma_window_s: float = SIGMA_WINDOW_S_DEFAULT,
    quiet_window_s: float = 60.0,
    quiet_level_s: float = 0.5,
    quiet_k: float = 2.5,
    quiet_ref_frac: float = 0.3,
    null_window_s: float = 300.0,
    zero_mean: bool = False,
    baseline_window_s: float = BASELINE_WINDOW_S_DEFAULT,
    baseline_percentile: float = 10.0,
    baseline_smooth_s: float = 30.0,
    shot_correction: bool = True,
    shot_cap: float = 3.0,
    time_dim: str = "time",
    verbose: bool = True,
) -> GecoPursuitResult:
    """Detect somatic calcium events by calibrated matched pursuit.

    Parameters
    ----------
    F : xr.DataArray
        Raw soma fluorescence, dims ``(soma_id, time)`` (a 1-D ``(time,)``
        trace is fine too). Typically ``load_soma_F(...)`` = the jRGECO
        channel of ``ROI_Fsvd``. Pass ``input_is_dff=True`` if it is
        already dF/F (the baseline step is skipped, ``f0`` is all ones).
    tau_rise_s, tau_decay_s : float / float or "auto"
        Template kinetics. ``tau_rise_s`` = 0.08 s (10-90 % rise ≈ 0.18 s,
        between single-AP and burst rises). ``tau_decay_s="auto"`` runs a
        first pass with :data:`TAU_DECAY_FALLBACK_S` and sets each soma's
        decay to the median half-decay of its 25 largest uninterrupted
        events ``/ ln 2`` (clipped to :data:`TAU_DECAY_AUTO_RANGE`; fallback
        when fewer than 5 qualify) — see :func:`estimate_kinetics`.
    z_threshold : float or None
        Fixed operating point on ``z_shot``. None (default): choose it from
        the empirical FDR table instead (``fdr_alpha``). The z scale is
        soma-specific — sigma_null is measured on each soma's own quiet
        stretches, and Fsvd traces carry slow background fluctuations 10-50x
        the white per-sample noise — so a fixed z means different things on
        different somas; the FDR choice does not. Re-slice cheaply with
        :func:`rates_at` — the catalog keeps everything >= ``z_floor``.
    z_floor : float
        Lowest z retained in the catalog (pursuit base threshold).
    fdr_alpha : float
        Operating point = the lowest z above which the estimated set FDR
        stays <= alpha (bounded below by ``z_floor``); on most somas the
        null is empty above z ~ 4-5, so this lands at 3-5.
    min_sep_s : float
        Pursuit refractory (s) between accepted components.
    max_rise_s : float
        How far after a pursuit onset the data peak may be (s).
    pre_window_s : float
        How far before a pursuit onset the foot (pre-event minimum) may be.
    onset_frac : float
        Onset = first sample after the foot at ``foot + onset_frac * rise``.
    prom_frac : float
        A local maximum counts as an event's own peak if its prominence is
        at least ``prom_frac`` of the rise (or 2 fast-noise sigma) — drives
        the merge / drop rules.
    min_rise_sigma, bg_window_s : float
        Data-side significance every candidate must pass: the rise between
        the robust pre-event level and the robust peak level must be at
        least ``min_rise_sigma`` background sigmas. ``sigma_bg`` (per soma,
        scaled by the local gain) is the one-sided robust sd of the smoothed
        dF/F around its ``bg_window_s`` local median: with the 0.5-s default
        it captures white noise plus the coloured sub-second fluctuations of
        Fsvd traces. This is what removes explaining-away residuals (the
        pursuit's own z cannot: subtracting a big, imperfectly matched event
        leaves positive residuals that a noise-only null never sees). Raise
        ``bg_window_s`` to ~5 s to also demand that events beat slow
        baseline wander (useful on bleached-out stretches, at the cost of
        sensitivity during dense bursting, where the local median itself
        rides on the events); the cleaner tool for bleached stretches is
        the ``f0_frac`` column. Lower ``min_rise_sigma`` to admit smaller
        events; ``rise_sigma`` lets you re-slice later.
    level_s : float
        Half-window (s) for the robust pre-event level (median around the
        foot); the peak level uses a quarter of it.
    smooth_s : float
        Boxcar (s) applied to dF/F before peak / onset measurement.
    max_gap_s : float
        NaN gaps up to this length are linearly bridged before filtering
        (motion-discard gaps are mostly < 0.25 s; a 2-s template would
        otherwise blind the detector for 2 s before each of them). Events
        whose onset-to-peak span touches a bridged gap get
        ``gap_bridged=True``. Longer gaps stay NaN and exclude one template
        length before them.
    sigma_window_s, quiet_window_s : float
        Rolling windows for the noise profile and the quiet-level median.
    quiet_level_s, quiet_k, quiet_ref_frac, null_window_s : float
        Quiet samples (for sigma_null and the FDR null) are those whose
        ``quiet_level_s`` boxcar level stays within ``quiet_k`` level-sigmas
        of the local quiet reference (the ``quiet_ref_frac`` quantile of the
        1-s bin levels in a ``null_window_s`` window) over the whole
        template window. sigma_null itself is rolling on the same window
        (lower-half MAD of the filtered trace on the quiet samples pooled
        over the window), so z is always measured against the LOCAL
        background: a bleached-out
        tail whose slow wander is 10x the early background gets its own
        scale (``per_soma.sigma_null_drift`` = last/first 2 min).
    zero_mean : bool
        Use a zero-mean (DC-blind) kernel instead of the plain positive
        template. Off by default: the plain template's negative excursions
        are noise-only (events are positive), which is what the sign-flip
        null and the lower-half sigma_null rely on; a zero-mean kernel has
        negative lobes around every event that contaminate both and leave
        lobe residuals after explaining-away. Kept as an option for traces
        with strong slow baseline error.
    baseline_window_s, baseline_percentile, baseline_smooth_s : float
        Passed to :func:`baseline_dff`.
    shot_correction, shot_cap : bool, float
        Deflate z by ``min(sqrt(1 + pre_dff), shot_cap)``.
    """
    da, soma_dim, ids, t, fs = _as_rows(F, time_dim)
    n_soma, n_t = da.shape
    dur = float(t[-1] - t[0])
    if input_is_dff:
        dff_da = da.astype(np.float32)
        f0_da = xr.ones_like(dff_da)
    else:
        dff_da, f0_da = baseline_dff(
            da,
            baseline_window_s=baseline_window_s,
            baseline_percentile=baseline_percentile,
            baseline_smooth_s=baseline_smooth_s,
            time_dim=time_dim,
        )
    DFF = np.asarray(dff_da.values, dtype=np.float64)
    F0 = np.asarray(f0_da.values, dtype=np.float64)
    min_sep = max(1, int(round(min_sep_s * fs)))

    rows: list[dict] = []
    per_soma: list[dict] = []
    snr = np.full((n_soma, n_t), np.nan, dtype=np.float32)
    z_grid = np.arange(z_floor, 40.01, 0.25)
    null_tot = np.zeros(z_grid.size)
    tau_by_soma: dict[str, float] = {}
    if isinstance(tau_decay_s, str) and tau_decay_s != "auto":
        raise ValueError("tau_decay_s must be a float or 'auto'")
    refine_kw = dict(
        max_rise_s=max_rise_s,
        pre_window_s=pre_window_s,
        onset_frac=onset_frac,
        prom_frac=prom_frac,
        min_rise_sigma=min_rise_sigma,
        level_s=level_s,
        shot_correction=shot_correction,
        shot_cap=shot_cap,
    )
    prep_kw = dict(
        tau_rise_s=tau_rise_s,
        sigma_window_s=sigma_window_s,
        quiet_window_s=quiet_window_s,
        quiet_level_s=quiet_level_s,
        quiet_k=quiet_k,
        quiet_ref_frac=quiet_ref_frac,
        null_window_s=null_window_s,
        bg_window_s=bg_window_s,
        max_gap_s=max_gap_s,
        smooth_s=smooth_s,
        zero_mean=zero_mean,
    )
    for i, sid in enumerate(ids):
        # -- template decay: fixed, or two-pass (first pass with the fallback
        #    decay, tau from the half-decays of its largest clean events)
        tau_n_used, tau_hd = None, None
        if tau_decay_s == "auto":
            P0 = _prepare_soma(DFF[i], fs, tau_decay=TAU_DECAY_FALLBACK_S, **prep_kw)
            if P0.ok:
                idx0, amp0 = _pursue(P0, z_floor, min_sep)
                ev0 = _refine(P0, idx0, amp0, t, f0_1d=F0[i], **refine_kw)
                tau_i, tau_n_used, tau_hd = _tau_from_events(ev0, P0.xs, fs)
            else:
                tau_i = TAU_DECAY_FALLBACK_S
        else:
            tau_i = float(tau_decay_s)
        P = _prepare_soma(DFF[i], fs, tau_decay=tau_i, **prep_kw)
        fin = np.isfinite(DFF[i])
        base = {
            "soma_id": sid,
            "valid_frac": float(fin.mean()),
            "f0_start": float(np.nanmedian(F0[i][: int(120 * fs)]))
            if fin.any()
            else np.nan,
            "f0_end": float(np.nanmedian(F0[i][-int(120 * fs) :]))
            if fin.any()
            else np.nan,
        }
        base["bleach_ratio"] = (
            base["f0_start"] / base["f0_end"] if base["f0_end"] else np.nan
        )
        if not P.ok:
            if verbose:
                print(f"{sid}: no usable signal — skipped")
            per_soma.append({**base, "n_candidates": 0})
            continue
        idx, amp = _pursue(P, z_floor, min_sep)
        ev = _refine(P, idx, amp, t, f0_1d=F0[i], **refine_kw)
        f0_ref = base["f0_start"]
        for r in ev:
            r["f0_frac"] = (
                r["f0"] / f0_ref if f0_ref and np.isfinite(f0_ref) else np.nan
            )
            rows.append({"soma_id": sid, **r})
        snr_i = P.zt.astype(np.float32)
        snr_i[~P.valid] = np.nan
        snr[i] = snr_i
        # FDR null: negative peaks on quiet samples, same deflation
        qfrac = float(P.quiet.mean())
        if qfrac > 0:
            neg = np.where(P.quiet, -P.zt, -np.inf)
            pk, _ = find_peaks(neg)
            if pk.size:
                zc = neg[pk]
                if shot_correction:
                    pre = np.maximum(np.nan_to_num(P.xs[pk]), 0.0)
                    zc = zc / np.minimum(np.sqrt(1.0 + pre), shot_cap)
                null_tot += np.array([(zc >= zz).sum() for zz in z_grid]) / qfrac
        tau_by_soma[sid] = P.tau_decay_s
        per_soma.append({
            **base,
            "sigma_dff": P.sig_scalar,
            "sigma_bg": P.sigma_bg * float(np.nanmedian(P.gain)),
            "sigma_null": P.sigma_null,
            "sigma_null_drift": float(
                np.nanmedian(P.sigma_null_t[-int(120 * fs) :])
                / np.nanmedian(P.sigma_null_t[: int(120 * fs)])
            ),
            "noise_color": P.sigma_null / P.sig_scalar,
            "null_source": P.null_source,
            "template_len_s": P.T.klen / fs,
            "template_snr_factor": P.T.snr_factor,
            "quiet_frac": qfrac,
            "detectable_frac": float(P.valid.mean()),
            "tau_decay_s": P.tau_decay_s,
            "tau_source": "auto" if tau_decay_s == "auto" else "fixed",
            "tau_n_used": tau_n_used,
            "tau_half_decay_med_s": tau_hd,
            "n_pursuit": int(idx.size),
            "n_candidates": len(ev),
        })
        if verbose:
            print(
                f"{sid}: tau_decay {P.tau_decay_s:.2f}s | sigma_dff {P.sig_scalar:.4f}"
                f" | bleach x{base['bleach_ratio']:.1f} | quiet {qfrac:.2f}"
                f" | pursuit {idx.size} -> {len(ev)} candidates (z >= {z_floor})"
            )

    schema = {
        "soma_id": pl.Utf8,
        "idx_onset": pl.Int64,
        "idx_peak": pl.Int64,
        "idx_pursuit": pl.Int64,
        "onset_time": pl.Float64,
        "peak_time": pl.Float64,
        "peak_dff": pl.Float64,
        "pre_dff": pl.Float64,
        "rise_dff": pl.Float64,
        "rise_sigma": pl.Float64,
        "amp": pl.Float64,
        "z": pl.Float64,
        "z_shot": pl.Float64,
        "rise_time_s": pl.Float64,
        "half_decay_s": pl.Float64,
        "n_merged": pl.Int64,
        "gap_bridged": pl.Boolean,
        "f0": pl.Float64,
        "f0_frac": pl.Float64,
    }
    events = pl.DataFrame(rows, schema=schema).sort(["soma_id", "onset_time"])

    zs_all = events["z_shot"].to_numpy() if events.height else np.array([])
    det_tot = np.array([(zs_all >= zz).sum() for zz in z_grid])
    fdr_curve = null_tot / np.maximum(det_tot, 1)
    fdr = pl.DataFrame({
        "z": z_grid,
        "detections": det_tot.astype(int),
        "expected_fp": null_tot,
        "fdr": fdr_curve,
    })
    if z_threshold is None:
        # lowest z above which the FDR STAYS <= alpha (robust to a bumpy curve)
        bad = np.flatnonzero(fdr_curve > fdr_alpha)
        if bad.size == 0:
            z_star = float(z_grid[0])
        elif bad[-1] + 1 < z_grid.size:
            z_star = float(z_grid[bad[-1] + 1])
        else:
            z_star = float(z_grid[-1])
    else:
        z_star = float(z_threshold)
    if verbose:
        j = min(int(np.searchsorted(z_grid, z_star)), z_grid.size - 1)
        mode = "z_threshold" if z_threshold is not None else f"FDR<={fdr_alpha}"
        print(
            f"z* = {z_star} ({mode}) | events {int(det_tot[j])}"
            f" | expected FP {null_tot[j]:.1f} | set FDR {fdr_curve[j]:.3f}"
        )

    params = {
        "fs": fs,
        "duration_s": dur,
        "z_star": z_star,
        "z_floor": z_floor,
        "fdr_alpha": fdr_alpha,
        "tau_rise_s": tau_rise_s,
        "tau_decay_s": tau_decay_s,
        "tau_decay_by_soma": tau_by_soma,
        "min_sep_s": min_sep_s,
        "max_rise_s": max_rise_s,
        "onset_frac": onset_frac,
        "prom_frac": prom_frac,
        "min_rise_sigma": min_rise_sigma,
        "bg_window_s": bg_window_s,
        "level_s": level_s,
        "smooth_s": smooth_s,
        "max_gap_s": max_gap_s,
        "sigma_window_s": sigma_window_s,
        "null_window_s": null_window_s,
        "baseline_window_s": baseline_window_s,
        "baseline_percentile": baseline_percentile,
        "shot_correction": shot_correction,
        "shot_cap": shot_cap,
        "zero_mean": zero_mean,
        "input_is_dff": input_is_dff,
        "soma_dim": soma_dim,
    }
    snr_da = xr.DataArray(
        snr,
        dims=da.dims,
        coords=da.coords,
        name="matchfilt_snr",
        attrs={"units": "sigma_null", "tau_rise_s": tau_rise_s},
    )
    per_soma_df = pl.DataFrame(per_soma)
    res = GecoPursuitResult(events, per_soma_df, fdr, dff_da, f0_da, snr_da, params)
    rates = rates_at(res, z_star)
    per_soma_df = per_soma_df.join(
        rates.select(["soma_id", "n", "rate"]), on="soma_id", how="left"
    ).with_columns(pl.col("n").fill_null(0), pl.col("rate").fill_null(0.0))
    return GecoPursuitResult(events, per_soma_df, fdr, dff_da, f0_da, snr_da, params)


def rates_at(res: GecoPursuitResult, z: float) -> pl.DataFrame:
    """Per-soma event counts / rates (ev/s of DETECTABLE time) at any
    ``z_shot`` threshold — no recompute."""
    dur = res.params["duration_s"]
    ps = res.per_soma.select([
        "soma_id",
        *[c for c in ("detectable_frac",) if c in res.per_soma.columns],
    ])
    got = (
        res.events
        .filter(pl.col("z_shot") >= z)
        .group_by("soma_id")
        .agg(pl.len().alias("n"))
    )
    out = ps.join(got, on="soma_id", how="left").with_columns(pl.col("n").fill_null(0))
    if "detectable_frac" in out.columns:
        eff = pl.col("detectable_frac").fill_null(0.0) * dur
    else:
        eff = pl.lit(dur)
    return out.with_columns(
        pl.when(eff > 0).then(pl.col("n") / eff).otherwise(0.0).alias("rate"),
        pl.lit(float(z)).alias("z_threshold"),
    )


def matchfilt_snr(
    F: xr.DataArray,
    input_is_dff: bool = False,
    tau_rise_s: float = 0.08,
    tau_decay_s: float | Literal["auto"] = "auto",
    time_dim: str = "time",
    **kwargs,
) -> xr.DataArray:
    """The matched-filter trace in ``sigma_null`` units without running the
    pursuit — the same array as ``detect_events(...).snr``.
    ``kwargs`` are forwarded to :func:`detect_events` (baseline, windows...).
    """
    res = detect_events(
        F,
        input_is_dff=input_is_dff,
        tau_rise_s=tau_rise_s,
        tau_decay_s=tau_decay_s,
        time_dim=time_dim,
        verbose=False,
        z_floor=1e9,  # nothing accepted: pursuit / refinement are skipped
        **kwargs,
    )
    return res.snr


def event_train(
    res: GecoPursuitResult,
    like: xr.DataArray,
    z: float | None = None,
    at: Literal["onset", "peak"] = "onset",
    value: Literal["amp", "rise_dff", "ones"] = "rise_dff",
    time_dim: str = "time",
) -> xr.DataArray:
    """Rasterize the catalog into an impulse train with ``like``'s
    ``(soma_id, time)`` axes: ``value`` at each event's onset (or peak)
    sample, zero elsewhere."""
    if z is None:
        z = res.params["z_star"]
    da, _sd, ids, _t, _fs = _as_rows(like, time_dim)
    row = {s: i for i, s in enumerate(ids)}
    out = np.zeros(da.shape, dtype=np.float32)
    col = "idx_onset" if at == "onset" else "idx_peak"
    for r in res.events.filter(pl.col("z_shot") >= z).iter_rows(named=True):
        i = row.get(r["soma_id"])
        if i is not None:
            out[i, r[col]] += 1.0 if value == "ones" else r[value]
    return xr.DataArray(out, dims=da.dims, coords=da.coords, name="event_train")


def load_soma_F(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str | None = None,
    trace: str = "Fsvd",
    channel: int = 0,
    apply_ephys_offset: bool = True,
) -> xr.DataArray:
    """Soma ROI fluorescence for :func:`detect_events`: dims ``(soma_id, time)``.

    Somas are taken from every DMD listed in ``dmd_info.yaml`` for the
    acquisition (a ``dmd`` coord records which); ``soma_id`` restricts to
    one. Channel 0 is jRGECO (calcium) in the ROI zarrs. Raises if the soma
    has no saved ROI trace (no ROI mask was drawn for it).
    """
    import wisco_slap as wis

    info = wis.meta.get.dmd_info()[subject][exp][loc][acq]
    parts = []
    for dmd_key, dmd_meta in info.items():
        somas = list(dmd_meta.get("somas") or [])
        if soma_id is not None and soma_id not in somas:
            continue
        if not somas:
            continue
        dmd_num = int(str(dmd_key).split("-")[1])
        try:
            roi = wis.get.roi_F(
                subject,
                exp,
                loc,
                acq,
                trace=trace,
                dmd=dmd_num,
                channels=channel,
                apply_ephys_offset=apply_ephys_offset,
            )[f"dmd_{dmd_num}"]
        except FileNotFoundError:
            continue
        have = [s for s in somas if s in roi["soma_id"].values.astype(str)]
        if soma_id is not None:
            have = [s for s in have if s == soma_id]
        if not have:
            continue
        sub = roi.sel(soma_id=have)
        if "channel" in sub.dims:
            sub = sub.sel(channel=channel)
        sub = sub.assign_coords(dmd=("soma_id", [dmd_num] * len(have)))
        parts.append(sub)
    if not parts:
        raise ValueError(
            f"no ROI trace for soma {soma_id!r} in {subject}/{exp}/{loc}--{acq}"
            " (is there an ROI_Fsvd zarr and a soma mask for it?)"
        )
    out = xr.concat(parts, dim="soma_id") if len(parts) > 1 else parts[0]
    return out.transpose("soma_id", "time")


def plot_events(
    res: GecoPursuitResult,
    soma_id: str | None = None,
    t0: float | None = None,
    t1: float | None = None,
    z: float | None = None,
    show_snr: bool = True,
    figsize: tuple[float, float] = (14, 6),
    axes=None,
):
    """QC plot: dF/F with onset (green bar) / peak (red dot) markers for the
    events at threshold ``z`` (default ``z_star``); below it the matched-
    filter SNR trace with the threshold line. Returns the figure."""
    import matplotlib.pyplot as plt

    if z is None:
        z = res.params["z_star"]
    ids = res.dff["soma_id"].values.astype(str)
    if soma_id is None:
        soma_id = str(ids[0])
    dff = res.dff.sel(soma_id=soma_id)
    t = np.asarray(dff["time"].values, dtype=float)
    lo = 0 if t0 is None else int(np.searchsorted(t, t0))
    hi = t.size if t1 is None else int(np.searchsorted(t, t1))
    ev = res.events.filter(
        (pl.col("soma_id") == soma_id)
        & (pl.col("z_shot") >= z)
        & (pl.col("peak_time") >= t[lo])
        & (pl.col("peak_time") <= t[hi - 1])
    )
    if axes is None:
        n_ax = 2 if show_snr else 1
        fig, axes = plt.subplots(
            n_ax,
            1,
            sharex=True,
            figsize=figsize,
            squeeze=False,
            gridspec_kw={"height_ratios": [2, 1][:n_ax]},
        )
        axes = axes[:, 0]
    else:
        fig = axes[0].figure
    ax = axes[0]
    ax.plot(t[lo:hi], dff.values[lo:hi], color="k", lw=0.5, label="dF/F")
    if ev.height:
        on = ev["onset_time"].to_numpy()
        pk = ev["peak_time"].to_numpy()
        ax.plot(pk, ev["peak_dff"].to_numpy(), "o", color="tab:red", ms=4, label="peak")
        ax.plot(
            on,
            ev["pre_dff"].to_numpy(),
            "|",
            color="tab:green",
            ms=12,
            mew=1.5,
            label="onset",
        )
    ax.set_ylabel(f"{soma_id} dF/F")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    if show_snr:
        s = res.snr.sel(soma_id=soma_id).values
        axes[1].plot(t[lo:hi], s[lo:hi], color="0.3", lw=0.5)
        axes[1].axhline(z, color="tab:red", lw=0.8, ls="--")
        axes[1].set_ylabel("matched-filter z")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


__all__ = [
    "GecoPursuitResult",
    "baseline_dff",
    "detect_events",
    "estimate_kinetics",
    "event_train",
    "load_soma_F",
    "matchfilt_snr",
    "plot_events",
    "rates_at",
]
