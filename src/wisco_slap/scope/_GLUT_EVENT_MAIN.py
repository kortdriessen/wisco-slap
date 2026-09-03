"""Principled noise / baseline estimators for denoised SLAP2 traces.

Goal
----
Decide which peaks in the ``denoised`` trace to trust. The recipe is always:

    sigma = <one of the estimators below>(denoised, ls)   # noise scale
    snr   = peak_amplitude / sigma                          # signal-to-noise
    trust = snr >= snr_threshold                            # keep the peak

The key principle behind every estimator here
---------------------------------------------
The ``denoised`` trace is the upstream OASIS deconvolution: strictly
non-negative, floored at ~0, with the white noise already removed. You
therefore cannot reliably estimate noise *from* the denoised trace — it
collapses to ~0 during quiet periods (exactly where you need a baseline
reference) and, where it isn't quiet, its "spread" is the event-amplitude
distribution, not noise. So every estimator below measures the noise scale
from the ``ls`` (pre-denoise) trace, where the noise still lives, and returns
it as the yardstick for the denoised trace.

Two empirical facts shape the methods (characterized on bellatrix/exp_1/
loc_Z/acq_1, 200 Hz, ~49 min):

1. Glutamate events are strictly *positive* excursions, so the below-baseline
   half of ``ls`` is pure noise. Estimators that use only the lower half are
   immune to event contamination.
2. The noise is signal-dependent shot noise: ``var(noise) ≈ a + b·F̂``. For a
   detection *threshold* you want the baseline noise ``σ0 = √a`` (noise at
   F≈0). For the *uncertainty of a measured amplitude* you want the
   per-sample ``σ(F̂) = √(a + b·F̂)``. They are different — see
   :func:`shot_noise_sigma`.

Return-type convention
----------------------
Every estimator takes a ``denoised`` and an ``ls`` :class:`xarray.DataArray`
with a ``time`` dimension (plus any leading dims such as ``syn_id`` /
``channel``) and returns an :class:`xarray.DataArray` ``sigma``:

* **per-synapse** estimators reduce over ``time`` -> dims = input minus
  ``time`` (e.g. ``(syn_id,)``). Use these as a flat per-synapse threshold.
* **per-sample** estimators (rolling lower-half MAD with ``window_s`` set, and
  ``shot_noise_sigma(per_sample=True)``) keep the ``time`` dim.

Because xarray broadcasts, ``denoised / sigma`` works for both. Helpers
:func:`snr` and :func:`master_event_detector` wrap the full "which peaks to
trust" workflow.

Typical use
-----------
>>> import wisco_slap as wis
>>> import principled_noise_estimators as pne
>>> dn = wis.get.syn_dF("bellatrix","exp_1","loc_Z","acq_1", trace="denoised",
...                      channels=0, merge_info=True)["dmd_1"]
>>> ls = wis.get.syn_dF("bellatrix","exp_1","loc_Z","acq_1", trace="ls",
...                      channels=0, merge_info=True)["dmd_1"]
>>> sigma = pne.lower_half_mad_sigma(dn, ls, window_s=10.0)   # per-sample σ(t)
>>> all_peaks, final_peaks, peaks_df = pne.master_event_detector(
...     dn, sigma, snr_threshold=5.0)
>>> peaks_df                  # one row per detected event (df_scope='final')
"""

from __future__ import annotations

from typing import NamedTuple

import bottleneck as bn
import numpy as np
import polars as pl
import xarray as xr
from scipy.signal import find_peaks

MAD_TO_SIGMA: float = 1.4826  # median-abs-dev -> Gaussian sigma


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _infer_fs(da: xr.DataArray, time_dim: str = "time") -> float:
    """Sampling rate in Hz from the ``time`` coordinate (median spacing)."""
    t = np.asarray(da.coords[time_dim].values, dtype=float)
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Could not infer fs from '{time_dim}' coord.")
    return 1.0 / dt


def _odd_window(window_s: float, fs: float) -> int:
    """Convert a window in seconds to an odd, >=3 sample count."""
    w = max(3, int(round(window_s * fs)))
    return w + 1 if w % 2 == 0 else w


def _centered_move_median(
    arr: np.ndarray, window: int, axis: int, min_count: int
) -> np.ndarray:
    """NaN-aware centered rolling median along ``axis`` (bottleneck-backed).

    Mirrors the padding trick in ``wis.util.snr.rolling_mad`` so the output is
    centered and NaNs are ignored within each window.
    """
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    pads = [(0, 0)] * arr.ndim
    pads[axis] = (pad_left, pad_right)
    padded = np.pad(arr, pads, mode="constant", constant_values=np.nan)
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(window - 1, None)
    return bn.move_median(padded, window=window, min_count=min_count, axis=axis)[
        tuple(sl)
    ]


def _empty_like_reduced(
    da: xr.DataArray, time_dim: str, name: str
) -> tuple[np.ndarray, tuple[str, ...], dict]:
    """Return a NaN array + dims/coords for a per-synapse (time-reduced) result."""
    out_dims = tuple(d for d in da.dims if d != time_dim)
    shape = tuple(da.sizes[d] for d in out_dims)
    coords = {d: da.coords[d] for d in out_dims if d in da.coords}
    return np.full(shape, np.nan, dtype=float), out_dims, coords


def _move_time_first(a: np.ndarray, time_axis: int) -> np.ndarray:
    return np.moveaxis(a, time_axis, 0)


# ---------------------------------------------------------------------------
# A. lower-half MAD on ls   (recommended default; global or rolling)
# ---------------------------------------------------------------------------

# Moved 2026-08-20 to its general home in wis.util.snr; re-exported here so
# existing callers (and the ESTIMATORS registry below) keep working.
from wisco_slap.util.snr import (
    lower_half_mad_sigma as lower_half_mad_sigma,  # noqa: E402
)

# ---------------------------------------------------------------------------
# C. residual robust scale on baseline samples
# ---------------------------------------------------------------------------


def residual_baseline_sigma(
    denoised: xr.DataArray,
    ls: xr.DataArray,
    baseline_quantile: float = 0.5,
    time_dim: str = "time",
    scale: float = MAD_TO_SIGMA,
) -> xr.DataArray:
    """Per-synapse baseline σ from the residual ``r = ls - denoised``.

    The residual is, by construction, what the denoiser treats as noise. We
    restrict to *baseline* samples — those where the denoised level is in its
    lowest ``baseline_quantile`` fraction — so the estimate reflects the noise
    floor at F≈0 rather than the (larger, shot-noise-inflated) noise during
    events. Robust scale = ``scale * MAD`` about the residual's own median
    (the residual carries a non-zero offset; using its median absorbs that).

    Parameters
    ----------
    baseline_quantile : float
        Fraction of lowest-denoised-level samples treated as baseline.
        ``0.5`` uses the quieter half of each synapse.

    Returns
    -------
    xr.DataArray
        Per-synapse σ (dims = input minus ``time``).
    """
    d = np.asarray(denoised.values, dtype=float)
    r = np.asarray(ls.values, dtype=float) - d
    axis = denoised.get_axis_num(time_dim)

    d_t = _move_time_first(d, axis)
    r_t = _move_time_first(r, axis)
    nt = d_t.shape[0]
    flat_d = d_t.reshape(nt, -1)
    flat_r = r_t.reshape(nt, -1)

    out_flat = np.full(flat_d.shape[1], np.nan)
    for j in range(flat_d.shape[1]):
        dj, rj = flat_d[:, j], flat_r[:, j]
        m = np.isfinite(dj) & np.isfinite(rj)
        if m.sum() < 20:
            continue
        dj, rj = dj[m], rj[m]
        thr = np.quantile(dj, baseline_quantile)
        base = rj[dj <= thr]
        if base.size < 10:
            continue
        out_flat[j] = scale * np.median(np.abs(base - np.median(base)))

    out, dims, coords = _empty_like_reduced(denoised, time_dim, "sigma")
    out[...] = out_flat.reshape(out.shape)
    return xr.DataArray(out, dims=dims, coords=coords, name="sigma")


# ---------------------------------------------------------------------------
# D. higher-order finite-difference (Immerkaer / estimatenoise) on ls
# ---------------------------------------------------------------------------


def finite_diff_sigma(
    denoised: xr.DataArray,
    ls: xr.DataArray,
    order: int = 2,
    time_dim: str = "time",
    robust: bool = True,
    scale: float = MAD_TO_SIGMA,
) -> xr.DataArray:
    """Per-synapse white-noise σ from higher-order finite differences of ``ls``.

    An ``order``-th difference annihilates any locally-polynomial signal of
    degree < ``order`` while leaving the noise variance unchanged up to the
    kernel's L2 norm. This is the principle behind the upstream pipeline's
    ``estimatenoise.m`` (D'Errico) and Immerkær's classic estimator.

    For order 2 the kernel is ``[1,-2,1]`` (‖·‖₂ = √6); order 3 is
    ``[-1,3,-3,1]`` (‖·‖₂ = √20). ``σ = robust_scale(diff) / ‖kernel‖₂``.

    Parameters
    ----------
    order : int
        Difference order (2 recommended; 3 kills curved baselines too).
    robust : bool
        If True, scale = ``scale * MAD`` (resists the occasional event that
        survives differencing). If False, plain ``std``.

    Returns
    -------
    xr.DataArray
        Per-synapse σ (dims = input minus ``time``).

    Notes
    -----
    Runs on ``ls``; ``denoised`` is accepted only for signature uniformity.
    Unlike ``std(diff)/√2``, higher orders are robust to smooth drift; but like
    any difference estimator it must be applied to ``ls`` — on the denoised
    (autocorrelated, noise-removed) trace it severely under-reports.
    """
    x = np.asarray(ls.values, dtype=float)
    axis = ls.get_axis_num(time_dim)

    # kernel = order-th finite difference; coefficients are signed binomials.
    kernel = np.array([(-1) ** k * _binom(order, k) for k in range(order + 1)])
    knorm = np.linalg.norm(kernel)

    d = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), axis, x)
    if robust:
        med = np.nanmedian(d, axis=axis, keepdims=True)
        s = scale * np.nanmedian(np.abs(d - med), axis=axis)
    else:
        s = np.nanstd(d, axis=axis)

    out, dims, coords = _empty_like_reduced(ls, time_dim, "sigma")
    out[...] = s / knorm
    return xr.DataArray(out, dims=dims, coords=coords, name="sigma")


def _binom(n: int, k: int) -> int:
    from math import comb

    return comb(n, k)


# ---------------------------------------------------------------------------
# B. shot-noise model  var = a + b*F̂
# ---------------------------------------------------------------------------


def shot_noise_sigma(
    denoised: xr.DataArray,
    ls: xr.DataArray,
    per_sample: bool = False,
    n_bins: int = 12,
    min_bin_count: int = 50,
    level_cap_quantile: float = 0.99,
    robust_bins: bool = True,
    floor: float = 1e-3,
    time_dim: str = "time",
) -> xr.DataArray:
    """Signal-dependent noise from the shot-noise model ``var = a + b·F̂``.

    For each synapse, bins samples by denoised level F̂, estimates the residual
    (``ls - denoised``) variance per bin, and fits ``var = a + b·F̂`` by
    weighted least squares. The fitted per-synapse parameters ``a`` (baseline
    variance), ``b`` (shot-noise slope) and ``sigma0 = √a`` are always attached
    to the result as non-dimension coords on the synapse dim.

    Parameters
    ----------
    per_sample : bool
        If ``False`` (default) returns the per-synapse **baseline** noise
        ``σ0 = √a`` — the right quantity to *threshold events against*.
        If ``True`` returns the per-sample ``σ(F̂) = √(a + b·F̂)`` — the
        local uncertainty, which correctly inflates during events (use this to
        put an error bar on a measured amplitude, **not** as a detection
        threshold: thresholding against the inflated during-event σ is
        circular and suppresses real events).
    n_bins : int
        Number of level bins (quantile-spaced) for the var-vs-level fit.
    min_bin_count : int
        Bins with fewer samples are dropped from the fit.
    level_cap_quantile : float
        Ignore samples above this denoised-level quantile when binning, so a
        few huge events don't dominate the slope.
    robust_bins : bool
        If True, per-bin scale is ``(1.4826·MAD)²`` (resists onset
        misalignment); else plain variance.
    floor : float
        σ is clipped to be ≥ ``floor`` to avoid divide-by-zero in SNR.

    Returns
    -------
    xr.DataArray
        ``per_sample=False`` -> per-synapse σ0 (dims = input minus ``time``).
        ``per_sample=True``  -> per-sample σ(F̂) (same dims as input).
        Both carry ``shot_a``, ``shot_b``, ``sigma0`` coords on the synapse dim.
    """
    d = np.asarray(denoised.values, dtype=float)
    r = np.asarray(ls.values, dtype=float) - d
    axis = denoised.get_axis_num(time_dim)

    d_t = _move_time_first(d, axis)
    r_t = _move_time_first(r, axis)
    nt = d_t.shape[0]
    lead_shape = d_t.shape[1:]
    flat_d = d_t.reshape(nt, -1)
    flat_r = r_t.reshape(nt, -1)
    nsrc = flat_d.shape[1]

    a_arr = np.full(nsrc, np.nan)
    b_arr = np.full(nsrc, np.nan)
    for j in range(nsrc):
        dj, rj = flat_d[:, j], flat_r[:, j]
        m = np.isfinite(dj) & np.isfinite(rj)
        if m.sum() < (min_bin_count * 3):
            continue
        dj, rj = dj[m], rj[m]
        cap = np.quantile(dj, level_cap_quantile)
        sel = dj <= cap
        dj, rj = dj[sel], rj[sel]
        edges = np.unique(np.quantile(dj, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 4:
            continue
        centers, variances, weights = [], [], []
        for i in range(edges.size - 1):
            bsel = (dj >= edges[i]) & (dj < edges[i + 1])
            n = int(bsel.sum())
            if n < min_bin_count:
                continue
            rr = rj[bsel]
            if robust_bins:
                v = (MAD_TO_SIGMA * np.median(np.abs(rr - np.median(rr)))) ** 2
            else:
                v = float(np.var(rr))
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            variances.append(v)
            weights.append(n)
        if len(centers) < 3:
            continue
        A = np.vstack([np.ones(len(centers)), np.array(centers)]).T
        w = np.sqrt(np.array(weights, dtype=float))
        coef, *_ = np.linalg.lstsq(A * w[:, None], np.array(variances) * w, rcond=None)
        a_arr[j], b_arr[j] = float(coef[0]), float(coef[1])

    a_arr = np.where(np.isfinite(a_arr), np.maximum(a_arr, floor**2), np.nan)
    sigma0 = np.sqrt(a_arr)

    syn_dim = next((dd for dd in denoised.dims if dd != time_dim), None)
    a_re = a_arr.reshape(lead_shape)
    b_re = b_arr.reshape(lead_shape)
    s0_re = sigma0.reshape(lead_shape)

    def _attach(da: xr.DataArray) -> xr.DataArray:
        if syn_dim is not None and da[syn_dim].ndim == 1 and a_re.ndim == 1:
            da = da.assign_coords(
                shot_a=(syn_dim, a_re),
                shot_b=(syn_dim, b_re),
                sigma0=(syn_dim, s0_re),
            )
        return da

    if not per_sample:
        out, dims, coords = _empty_like_reduced(denoised, time_dim, "sigma")
        out[...] = s0_re
        return _attach(xr.DataArray(out, dims=dims, coords=coords, name="sigma"))

    # per-sample sigma(F̂) = sqrt(a + b*max(F̂,0))
    a_b = np.broadcast_to(np.expand_dims(a_arr.reshape(lead_shape), axis), d.shape)
    b_b = np.broadcast_to(np.expand_dims(b_arr.reshape(lead_shape), axis), d.shape)
    var = a_b + b_b * np.clip(d, 0.0, None)
    sig = np.sqrt(np.clip(var, floor**2, None))
    return _attach(
        xr.DataArray(sig, dims=denoised.dims, coords=denoised.coords, name="sigma")
    )


# ---------------------------------------------------------------------------
# E. rolling percentile  (signal LEVEL reference, not a noise scale)
# ---------------------------------------------------------------------------


def rolling_percentile(
    signal: xr.DataArray,
    window_s: float,
    percentile: float = 0.8,
    time_dim: str = "time",
    min_valid_frac: float = 0.25,
    interpolation: str = "linear",
) -> xr.DataArray:
    """Centered rolling percentile of ``signal`` along ``time``.

    A windowed *level* reference (e.g. a rolling 80th-percentile of the denoised
    trace), returned with the **same shape** as the input. Unlike the σ
    estimators above this is not a noise scale — just a moving quantile of the
    signal itself, useful as a slow baseline / "typical high level" track.

    NaN-aware: NaNs are skipped within each window (not sorted as large values),
    and a window with fewer than ``min_valid_frac`` of its samples valid yields
    NaN — relevant here, as the denoised trace carries NaN gaps.

    Parameters
    ----------
    signal : xr.DataArray
        Trace to summarize; dims include ``time`` plus any leading dims (e.g.
        ``syn_id`` / ``channel``). 1-D (time-only) input is fine too.
    window_s : float
        Window length in seconds (converted to an odd sample count via the
        ``time`` coord's sampling rate).
    percentile : float
        Quantile as a **fraction in [0, 1]** (``0.8`` = 80th percentile).
    time_dim : str
        Name of the time dimension.
    min_valid_frac : float
        Minimum fraction of a window that must be non-NaN, else NaN there
        (governs the partial windows at the edges and across NaN gaps).
    interpolation : {"linear", "nearest", "lower", "higher", "midpoint"}
        Interpolation between order statistics (passed to polars; ``"linear"``
        matches :func:`numpy.percentile`'s default).

    Returns
    -------
    xr.DataArray
        Per-sample rolling percentile, same dims/coords as ``signal``.
    """
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be a fraction in [0, 1] (e.g. 0.8).")

    fs = _infer_fs(signal, time_dim)
    win = _odd_window(window_s, fs)
    axis = signal.get_axis_num(time_dim)

    # flatten every non-time slice into columns; roll each over time at once
    xt = _move_time_first(np.asarray(signal.values, dtype=float), axis)
    nt = xt.shape[0]
    lead = xt.shape[1:]
    flat = np.ascontiguousarray(xt.reshape(nt, -1))

    min_samples = max(1, int(round(min_valid_frac * win)))
    df = pl.from_numpy(
        flat, schema=[f"c{i}" for i in range(flat.shape[1])], orient="row"
    )
    out = (
        df
        .select(
            pl
            .all()
            .fill_nan(None)  # NaN -> null so it is skipped, not treated as +inf
            .rolling_quantile(
                quantile=percentile,
                interpolation=interpolation,
                window_size=win,
                min_samples=min_samples,
                center=True,
            )
        )
        .to_numpy()
        .reshape((nt, *lead))
    )
    out = np.moveaxis(out, 0, axis)
    return xr.DataArray(
        out, dims=signal.dims, coords=signal.coords, name="rolling_pctl"
    )


# ---------------------------------------------------------------------------
# SNR + peak-trust helpers
# ---------------------------------------------------------------------------


def snr(
    denoised: xr.DataArray, sigma: xr.DataArray, floor: float = 1e-6
) -> xr.DataArray:
    """SNR = denoised / sigma, broadcasting a per-synapse or per-sample σ."""
    safe = xr.where(np.abs(sigma) < floor, np.nan, sigma)
    out = denoised / safe
    return out.rename("snr")


def _enforce_min_distance(
    positions: np.ndarray, priority: np.ndarray, min_dist: int
) -> np.ndarray:
    """Greedy refractory filter — the logic behind ``find_peaks(distance=...)``.

    Given peak ``positions`` (ascending sample indices) and a ``priority`` per
    peak (higher = keep preferentially), repeatedly take the highest-priority
    surviving peak and drop every still-surviving peak within ``min_dist``
    samples of it, until no two survivors are closer than ``min_dist``. This
    removes the *fewest* possible peaks, always sacrificing the lower-priority
    member of any too-close pair. Returns a boolean keep-mask over ``positions``.
    """
    n = positions.shape[0]
    keep = np.ones(n, dtype=bool)
    order = np.argsort(priority, kind="stable")  # ascending priority
    for i in range(n - 1, -1, -1):  # walk highest-priority first
        j = order[i]
        if not keep[j]:
            continue
        k = j - 1
        while k >= 0 and positions[j] - positions[k] < min_dist:
            keep[k] = False
            k -= 1
        k = j + 1
        while k < n and positions[k] - positions[j] < min_dist:
            keep[k] = False
            k += 1
    return keep


class EventDetectionResult(NamedTuple):
    """Return bundle for :func:`master_event_detector`.

    Attributes
    ----------
    all_peaks : xr.DataArray
        Boolean mask, same dims/coords as ``signal``; True at every first-pass
        (permissive) peak sample.
    final_peaks : xr.DataArray
        Boolean mask, same dims/coords as ``signal``; True only at peaks that
        survived the SNR gate and (if enabled) the refractory filter.
    peaks_df : pl.DataFrame
        Per-peak table; membership set by ``df_scope`` (default ``"final"`` ->
        one row per surviving event, matching ``final_peaks``). Carries every
        synapse coordinate plus per-peak amplitude / sigma / SNR and the
        ``passed_snr`` / ``final`` flags.
    """

    all_peaks: xr.DataArray
    final_peaks: xr.DataArray
    peaks_df: pl.DataFrame


def master_event_detector(
    signal: xr.DataArray,
    sigma: xr.DataArray,
    snr_threshold: float = 5.0,
    min_distance_s: float | None = 0.02,
    distance_priority: str = "snr",
    df_scope: str = "final",
    time_dim: str = "time",
    syn_dim: str | None = None,
) -> EventDetectionResult:
    """Detect events in ``signal`` by per-sample SNR against ``sigma``.

    Pipeline, per synapse:

    1. **Permissive first pass** — every local maximum of ``signal``
       (``scipy.signal.find_peaks`` with no constraints). The intent is recall,
       not precision: real events should all be in here, alongside many false
       positives.
    2. **SNR gate** — score each peak ``snr = signal_at_peak / sigma_at_peak``
       (a plain per-sample division, each peak with its own denominator) and
       keep those with ``snr >= snr_threshold``.
    3. **Refractory filter** (optional) — if ``min_distance_s`` is set, greedily
       drop the lowest-``distance_priority`` peaks until no two survivors are
       closer than ``min_distance_s``.

    ``sigma`` must be a per-sample noise estimate sharing ``signal``'s ``time``
    and synapse axes — choosing the *right* noise estimate (e.g. a baseline σ(t)
    that does **not** inflate during events, to avoid circular suppression) is
    entirely the caller's responsibility. See the module docstring.

    Parameters
    ----------
    signal : xr.DataArray
        Trace to detect on (typically ``denoised``); dims = synapse axis (any
        non-``time`` dims) + ``time``, with whatever synapse coordinates
        (``syn_id``, ``dend_id``, ``pos``, ``channel`` …) you want propagated.
    sigma : xr.DataArray
        Per-sample noise scale. Must carry ``time`` and align exactly with
        ``signal`` on every shared index coordinate.
    snr_threshold : float
        Keep peaks whose ``signal/sigma`` is at or above this. 5.0 is a sensible
        starting point; sweep it.
    min_distance_s : float or None
        Minimum allowed spacing between final peaks (seconds). ``0.02`` ≈ the
        iGluSnFR transient width. ``None`` disables the refractory filter
        (``final`` then equals the SNR gate result).
    distance_priority : {"snr", "amplitude"}
        Which quantity ranks peaks when resolving too-close pairs in the
        refractory filter. ``"snr"`` (default) keeps the more *trustworthy*
        peak under a time-varying σ; ``"amplitude"`` keeps the literally taller
        one (matches ``scipy``'s native ``distance`` behaviour). They coincide
        when σ is locally flat.
    df_scope : {"all", "passed", "final"}
        Which peaks populate ``peaks_df``. ``"final"`` (default) returns only
        the surviving events — its row count equals ``final_peaks.sum()``.
        ``"passed"`` returns all SNR-gate survivors (pre-refractory); ``"all"``
        returns every first-pass peak (parallels ``all_peaks``; can be millions
        of rows). The two masks are unaffected — they always reflect the full
        first-pass and final sets.
    time_dim : str
        Name of the time dimension.
    syn_dim : str or None
        Optional explicit synapse dim name (validated if given). Not otherwise
        required — all synapse coordinates are propagated automatically.

    Returns
    -------
    EventDetectionResult
        Named tuple ``(all_peaks, final_peaks, peaks_df)``.

    Notes
    -----
    A peak landing on a sample where ``sigma`` is NaN or ≤ 0 gets ``snr = null``
    and is excluded from ``final`` (and ``passed_snr``) — relevant for rolling
    baseline estimators, which return NaN inside long fully-active stretches
    (exactly where large events can live).
    """
    if time_dim not in signal.dims:
        raise ValueError(f"signal has no '{time_dim}' dim (dims={signal.dims}).")
    if time_dim not in sigma.dims:
        raise ValueError(
            f"sigma has no '{time_dim}' dim (dims={sigma.dims}). "
            "master_event_detector requires a per-sample sigma sharing signal's "
            "time axis."
        )
    if distance_priority not in ("snr", "amplitude"):
        raise ValueError("distance_priority must be 'snr' or 'amplitude'.")
    if df_scope not in ("all", "passed", "final"):
        raise ValueError("df_scope must be 'all', 'passed' or 'final'.")

    # Require identical time + synapse index coords; raises on any mismatch.
    signal, sigma = xr.align(signal, sigma, join="exact")

    non_time = [d for d in signal.dims if d != time_dim]
    if not non_time:
        raise ValueError("signal needs at least one non-time (synapse) dim.")
    if syn_dim is not None and syn_dim not in signal.dims:
        raise ValueError(f"syn_dim '{syn_dim}' not in signal dims {signal.dims}.")

    fs = _infer_fs(signal, time_dim)
    min_dist = (
        None if min_distance_s is None else max(1, int(round(min_distance_s * fs)))
    )
    tvals = np.asarray(signal.coords[time_dim].values, dtype=float)

    # Flatten every (syn_id x channel x …) slice into one synapse axis.
    sig_st = signal.stack(_s=non_time).transpose("_s", time_dim)
    sigma_st = sigma.stack(_s=non_time).transpose("_s", time_dim)

    n_s, n_t = sig_st.shape
    all_arr = np.zeros((n_s, n_t), dtype=bool)
    final_arr = np.zeros((n_s, n_t), dtype=bool)

    # Columnar accumulators — appended per synapse, concatenated once at the end.
    # (A per-peak list-of-dicts would mean millions of Python dicts; this stays
    # in numpy and attaches the synapse coordinates via one small join below.)
    sidx_parts: list[np.ndarray] = []
    pkidx_parts: list[np.ndarray] = []
    amp_parts: list[np.ndarray] = []
    sig_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []
    pass_parts: list[np.ndarray] = []
    final_parts: list[np.ndarray] = []
    meta_rows: list[dict] = []  # one record per synapse (joined onto peaks)

    for k in range(n_s):
        col = sig_st.isel(_s=k)
        y = np.asarray(col.values, dtype=float)
        sg = np.asarray(sigma_st.isel(_s=k).values, dtype=float)

        # 1. permissive first pass: every local maximum
        yy = np.nan_to_num(y, nan=0.0)
        pk, _ = find_peaks(yy)
        if pk.size == 0:
            continue
        all_arr[k, pk] = True

        amp = y[pk]
        sg_pk = sg[pk]
        with np.errstate(invalid="ignore", divide="ignore"):
            snr_pk = np.where(sg_pk > 0, amp / sg_pk, np.nan)
            passed = np.isfinite(snr_pk) & (snr_pk >= snr_threshold)

        # 2. SNR gate, then 3. optional refractory filter
        final = passed.copy()
        if min_dist is not None and passed.any():
            sel = np.flatnonzero(passed)
            prio = snr_pk[sel] if distance_priority == "snr" else amp[sel]
            keep = _enforce_min_distance(pk[sel], prio, min_dist)
            final = np.zeros_like(passed)
            final[sel[keep]] = True
        final_arr[k, pk[final]] = True

        sidx_parts.append(np.full(pk.size, k, dtype=np.int64))
        pkidx_parts.append(pk.astype(np.int64))
        amp_parts.append(amp)
        sig_parts.append(sg_pk)
        snr_parts.append(snr_pk)
        pass_parts.append(passed)
        final_parts.append(final)

        # one metadata record per synapse: every scalar coordinate on the slice
        meta: dict = {"_sidx": k}
        for name in col["_s"].coords:  # multiindex levels (syn_id, channel, …)
            if name == "_s":
                continue
            try:
                meta[name] = col["_s"].coords[name].item()
            except Exception:
                pass
        for name, c in col.coords.items():  # non-dim coords (dend-ID, pos, …)
            if name in (time_dim, "_s") or name in meta:
                continue
            if c.ndim == 0:
                try:
                    meta[name] = c.item()
                except Exception:
                    pass
        meta_rows.append(meta)

    # Rebuild the masks on ``signal``'s own grid rather than via ``unstack``.
    # ``unstack`` re-derives coords from the ``_s`` MultiIndex, which broadcasts
    # any coord living on a *subset* of the stacked dims (e.g. a ``dend_id``
    # along ``syn`` only) up to the full non-time grid — changing its
    # dimensionality and column order. Reshaping the flat ``(_s, time)`` arrays
    # back into the non-time grid (C-order, exactly inverting ``stack``) and
    # copying onto ``signal`` preserves every coordinate's original dims,
    # values, order, and scalars.
    def _to_signal_grid(arr: np.ndarray, name: str) -> xr.DataArray:
        shaped = arr.reshape(*[signal.sizes[d] for d in non_time], n_t)
        return (
            signal
            .transpose(*non_time, time_dim)
            .copy(data=shaped)
            .rename(name)
            .transpose(*signal.dims)
        )

    all_peaks = _to_signal_grid(all_arr, "all_peaks")
    final_peaks = _to_signal_grid(final_arr, "final_peaks")

    if not meta_rows:
        return EventDetectionResult(all_peaks, final_peaks, pl.DataFrame())

    sidx_all = np.concatenate(sidx_parts)
    pkidx_all = np.concatenate(pkidx_parts)
    amp_all = np.concatenate(amp_parts)
    sig_all = np.concatenate(sig_parts)
    snr_all = np.concatenate(snr_parts)  # NaN where sigma was invalid
    pass_all = np.concatenate(pass_parts)
    final_all = np.concatenate(final_parts)

    if df_scope == "final":
        keep = final_all
    elif df_scope == "passed":
        keep = pass_all
    else:  # "all"
        keep = np.ones(sidx_all.shape, dtype=bool)

    pkidx_keep = pkidx_all[keep]
    core = pl.DataFrame({
        "_sidx": sidx_all[keep],
        "peak_idx": pkidx_keep,
        "peak_time": tvals[pkidx_keep],
        "amplitude": amp_all[keep],
        "sigma": sig_all[keep],
        "snr": snr_all[keep],
        "passed_snr": pass_all[keep],
        "final": final_all[keep],
    })
    # small per-synapse table (handles mixed str/None coords) joined onto peaks
    meta_df = pl.DataFrame(meta_rows, infer_schema_length=None)
    coord_cols = [c for c in meta_df.columns if c != "_sidx"]
    peak_cols = [
        "peak_idx",
        "peak_time",
        "amplitude",
        "sigma",
        "snr",
        "passed_snr",
        "final",
    ]
    peaks_df = (
        core
        .join(meta_df, on="_sidx", how="left")
        .select(coord_cols + peak_cols)
        .sort((["syn_id"] if "syn_id" in coord_cols else []) + ["peak_idx"])
    )

    return EventDetectionResult(all_peaks, final_peaks, peaks_df)


# all per-(denoised, ls) estimators, for convenience / iteration
ESTIMATORS = {
    "lower_half_mad": lower_half_mad_sigma,
    "residual_baseline": residual_baseline_sigma,
    "finite_diff": finite_diff_sigma,
    "shot_noise": shot_noise_sigma,
}
