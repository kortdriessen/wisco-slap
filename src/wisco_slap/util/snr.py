"""SNR estimation for SLAP2 activity traces.

Implements the SNR estimation method described in the SLAP2 paper draft
(Methods → "SNR estimation"):

- noise  = std(diff(trace)) / sqrt(2)
- signal = `percentile`-th percentile of peak amplitudes detected by
           scipy.signal.find_peaks
- snr    = signal / noise

Three computation modes, mutually exclusive:
    - whole-trace (default): one value per non-time dimension.
    - non-overlapping chunks (``chunk_size``).
    - rolling window (``window_size``, ``step_size``).

The function operates on xr.DataArray and broadcasts over any non-time dims
(e.g. ``syn_id``, ``channel``), so a single call covers an entire DMD's worth
of synapses.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import bottleneck as bn
import numpy as np
import polars as pl
import xarray as xr
from scipy.signal import find_peaks

MAD_TO_SIGMA: float = 1.4826  # median-abs-dev -> Gaussian sigma


def compute_snr(
    da: xr.DataArray,
    *,
    chunk_size: float | None = None,
    window_size: float | None = None,
    step_size: float | None = None,
    time_dim: str = "time",
    units: Literal["seconds", "samples"] = "seconds",
    percentile: float = 95.0,
    min_peaks: int = 1,
    find_peaks_kwargs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Estimate signal-to-noise ratio per the SLAP2 paper definition.

    Parameters
    ----------
    da
        Input trace(s). Must have ``time_dim`` as one of its dims; any
        additional dims (e.g. ``syn_id``, ``channel``) are broadcast over.
    chunk_size
        Length of non-overlapping chunks in ``units``. Mutually exclusive
        with ``window_size`` / ``step_size``.
    window_size, step_size
        Length and stride of rolling windows in ``units``. Both required
        together; mutually exclusive with ``chunk_size``.
    time_dim
        Name of the time dimension. Defaults to ``"time"``.
    units
        Units for ``chunk_size`` / ``window_size`` / ``step_size``.
        ``"seconds"`` (default) requires ``time_dim`` to have a numeric
        coordinate; sample period is inferred as
        ``np.median(np.diff(time))``. ``"samples"`` interprets sizes as
        sample counts.
    percentile
        Percentile of peak amplitudes used as the signal estimate. Paper
        default is 95.
    min_peaks
        Minimum number of detected peaks required to report a signal. Below
        this, ``signal`` and ``snr`` are NaN (``noise`` is still reported).
    find_peaks_kwargs
        Forwarded to ``scipy.signal.find_peaks``. Default ``{}`` matches the
        paper. See "Notes" for caveats.

    Returns
    -------
    xr.Dataset
        Variables ``signal``, ``noise``, ``snr``. In whole-trace mode
        ``time_dim`` is dropped; in chunked / rolling mode it is replaced by
        a new coordinate giving the *center* time of each window.

    Notes
    -----
    Whole-trace mode reduces over all of ``time_dim``; chunked and rolling
    modes return one (signal, noise, snr) triple per window.

    NaNs are tolerated: diffs spanning a NaN are dropped before computing
    noise, and peaks are searched only within contiguous finite segments.
    Chunks / windows that fall entirely on NaN samples produce NaN outputs.

    Caveats faithful to the paper that are worth knowing:

    1. The bare ``find_peaks`` default finds every local maximum, including
       noise fluctuations. The 95th-percentile-of-peaks signal estimate
       works well when events are frequent enough that the upper tail of
       the peak-amplitude distribution is dominated by real events; for
       sparse activity it can be unreliable. Pass e.g.
       ``find_peaks_kwargs={"prominence": 3 * sigma}`` for sparser, more
       meaningful peaks.
    2. ``std(diff)/sqrt(2)`` assumes consecutive noise samples are
       uncorrelated. On filtered or denoised traces (e.g.
       ``syn_dF-denoised``) noise is temporally smoothed and this estimator
       under-reports it, inflating SNR.
    3. SNR here is a linear ratio (not dB).

    Examples
    --------
    >>> # Whole-trace SNR for one synapse
    >>> g = dn["dmd_2"].sx.sid(54).sx.ch(0)
    >>> compute_snr(g)

    >>> # Whole-DMD: one (signal, noise, snr) per synapse, broadcast over
    >>> # syn_id and channel
    >>> compute_snr(dn["dmd_2"])

    >>> # 30 s non-overlapping chunks
    >>> compute_snr(g, chunk_size=30.0)

    >>> # 60 s windows stepped every 5 s
    >>> compute_snr(g, window_size=60.0, step_size=5.0)
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"`da` must be xr.DataArray, got {type(da).__name__}.")
    if time_dim not in da.dims:
        raise ValueError(f"DataArray has no '{time_dim}' dim. dims: {da.dims}")
    if find_peaks_kwargs is None:
        find_peaks_kwargs = {}

    using_chunk = chunk_size is not None
    using_window = window_size is not None or step_size is not None
    if using_chunk and using_window:
        raise ValueError("Specify chunk_size OR window_size/step_size, not both.")
    if using_window and (window_size is None or step_size is None):
        raise ValueError(
            "Both window_size and step_size are required for rolling mode."
        )

    if not (using_chunk or using_window):
        return _apply_whole(
            da,
            time_dim=time_dim,
            percentile=percentile,
            min_peaks=min_peaks,
            find_peaks_kwargs=find_peaks_kwargs,
        )

    if units == "seconds":
        if time_dim not in da.coords:
            raise ValueError(
                f"units='seconds' requires the '{time_dim}' coord. "
                "Pass units='samples' to size in samples."
            )
        time_vals = np.asarray(da[time_dim].values, dtype=float)
        dt = float(np.median(np.diff(time_vals)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"Could not infer sample period from '{time_dim}'.")
    elif units == "samples":
        dt = 1.0
        time_vals = (
            np.asarray(da[time_dim].values, dtype=float)
            if time_dim in da.coords
            else None
        )
    else:
        raise ValueError(f"`units` must be 'seconds' or 'samples', got {units!r}.")

    if using_chunk:
        win_n = int(round(chunk_size / dt))
        step_n = win_n
    else:
        win_n = int(round(window_size / dt))
        step_n = int(round(step_size / dt))

    if win_n < 2:
        raise ValueError(
            f"chunk_size/window_size resolves to {win_n} samples; need >= 2."
        )
    if step_n < 1:
        raise ValueError(f"step_size resolves to {step_n} samples; need >= 1.")

    n = da.sizes[time_dim]
    starts = np.arange(0, n - win_n + 1, step_n)
    if starts.size == 0:
        raise ValueError(
            f"Trace of length {n} is shorter than window of {win_n} samples."
        )

    if time_vals is not None:
        center_times = np.array([
            (time_vals[s] + time_vals[s + win_n - 1]) / 2.0 for s in starts
        ])
    else:
        center_times = starts.astype(float) + (win_n - 1) / 2.0

    pieces: list[xr.Dataset] = []
    for s in starts:
        sub = da.isel({time_dim: slice(int(s), int(s) + win_n)})
        ds_s = _apply_whole(
            sub,
            time_dim=time_dim,
            percentile=percentile,
            min_peaks=min_peaks,
            find_peaks_kwargs=find_peaks_kwargs,
        )
        pieces.append(ds_s.expand_dims({time_dim: 1}))

    out = xr.concat(pieces, dim=time_dim)
    out = out.assign_coords({time_dim: center_times})
    return out


def _apply_whole(
    da: xr.DataArray,
    *,
    time_dim: str,
    percentile: float,
    min_peaks: int,
    find_peaks_kwargs: dict[str, Any],
) -> xr.Dataset:
    """Reduce over ``time_dim``, returning a Dataset of signal/noise/snr."""

    def _kernel(arr: np.ndarray) -> tuple[float, float, float]:
        return _snr_1d(
            arr,
            percentile=percentile,
            min_peaks=min_peaks,
            find_peaks_kwargs=find_peaks_kwargs,
        )

    sig, noise, snr = xr.apply_ufunc(
        _kernel,
        da,
        input_core_dims=[[time_dim]],
        output_core_dims=[[], [], []],
        vectorize=True,
        output_dtypes=[float, float, float],
    )
    return xr.Dataset({"signal": sig, "noise": noise, "snr": snr})


def _snr_1d(
    trace: np.ndarray,
    *,
    percentile: float,
    min_peaks: int,
    find_peaks_kwargs: dict[str, Any],
) -> tuple[float, float, float]:
    """SNR triple for a single 1D NaN-tolerant trace."""
    trace = np.asarray(trace, dtype=float)
    if trace.size < 2:
        return np.nan, np.nan, np.nan

    diffs = np.diff(trace)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return np.nan, np.nan, np.nan
    noise = float(np.std(diffs) / np.sqrt(2.0))

    peak_amps = _peak_amps_nan_safe(trace, find_peaks_kwargs)
    if peak_amps.size < min_peaks:
        return np.nan, noise, np.nan

    signal = float(np.percentile(peak_amps, percentile))
    snr = signal / noise if noise > 0 else np.nan
    return signal, noise, snr


def _peak_amps_nan_safe(
    trace: np.ndarray, find_peaks_kwargs: dict[str, Any]
) -> np.ndarray:
    """Concatenated peak amplitudes from find_peaks across contiguous finite
    segments of ``trace``."""
    finite = np.isfinite(trace)
    if not finite.any():
        return np.array([], dtype=float)

    edges = np.diff(finite.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if finite[0]:
        starts = np.concatenate(([0], starts))
    if finite[-1]:
        ends = np.concatenate((ends, [trace.size]))

    amps: list[np.ndarray] = []
    for s, e in zip(starts, ends, strict=True):
        seg = trace[s:e]
        if seg.size < 3:
            continue
        peaks, _ = find_peaks(seg, **find_peaks_kwargs)
        if peaks.size:
            amps.append(seg[peaks])
    if not amps:
        return np.array([], dtype=float)
    return np.concatenate(amps)


def rolling_mad(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    center: bool = True,
    scale: bool = True,
    min_count: int | None = None,
    lower_envelope: bool = False,
) -> xr.DataArray:
    """Centered rolling MAD via two passes of bottleneck.move_median.

    ~10-50x faster than xr.rolling().construct().median() for large windows.

    NaN-tolerant: NaNs in `da` are ignored within each window. Output is
    NaN only where fewer than `min_count` real samples sit in the window
    (i.e. the center of a gap longer than ~window samples). Defaults to
    `max(1, window // 4)` — at least 25% of the window must be real.

    Parameters
    ----------
    da
        Input trace(s); any dims besides `dim` are broadcast over.
    window
        Window length in samples.
    dim
        Dimension to roll along.
    center
        Center the window on each sample (via NaN padding) rather than
        trailing it.
    scale
        Multiply by 1.4826 so the MAD estimates Gaussian sigma.
    min_count
        Minimum real samples per window; below it the output is NaN.
        Defaults to ``max(1, window // 4)``.
    lower_envelope
        If True, estimate scale from the *lower envelope* only: the
        deviation pool keeps just the samples at/below the local rolling
        median (``median - x`` where ``x <= median``), discarding all
        above-median samples. For noise symmetric about baseline this is
        statistically identical to the two-sided MAD (a deviation's
        magnitude is independent of its sign), so the 1.4826 scaling
        applies unchanged — but positive-only transients (glutamate /
        calcium events) can never enter the lower deviation pool, so
        event contamination only reaches the estimate indirectly, through
        the median shift, instead of directly corrupting the deviations.
        Use it on traces whose events are strictly positive and whose
        baseline noise is symmetric: ``syn_dF-ls`` and matched-filter
        output. Do NOT use it on rectified traces (``syn_dF-denoised``):
        their below-median half hugs the zero floor and the estimate
        collapses to ~0. Note both variants still inflate when sustained
        activity drags the rolling median itself above baseline; for
        synapses active most of the window, use the denoised-conditioned
        :func:`lower_half_mad_sigma` in this module instead.
        The deviation pass runs with ``min_count // 2`` (only ~half of
        real samples yield a lower deviation), preserving `min_count`'s
        "fraction of the window that must be real" meaning.

    Notes
    -----
    Runs per-row on a thread pool (bottleneck's move_median releases the
    GIL), so the two rolling-median passes parallelize across the non-time
    dims (synapses / channels).
    """
    axis = da.get_axis_num(dim)
    arr = np.asarray(da.values)
    if arr.dtype.kind != "f":
        arr = arr.astype(np.float64)
    if min_count is None:
        min_count = max(1, window // 4)

    moved = np.moveaxis(arr, axis, -1)
    rows = np.ascontiguousarray(moved.reshape(-1, moved.shape[-1]))
    out_rows = np.empty_like(rows)
    dev_mc = max(1, min_count // 2) if lower_envelope else min_count
    nanv = np.array(np.nan, dtype=rows.dtype)

    if center:
        pad_left = window // 2
        pad_right = window - 1 - pad_left

        def _move_median(x: np.ndarray, mc: int) -> np.ndarray:
            # Pad with NaN, not 'edge': edge mode replicates leading/trailing
            # NaNs into a stripe across the padding. With min_count<window,
            # NaN padding is simply ignored inside each window.
            padded = np.pad(
                x, (pad_left, pad_right), mode="constant", constant_values=np.nan
            )
            return bn.move_median(padded, window=window, min_count=mc)[window - 1 :]

    else:

        def _move_median(x: np.ndarray, mc: int) -> np.ndarray:
            return bn.move_median(x, window=window, min_count=mc)

    def _one(i: int) -> None:
        x = rows[i]
        med = _move_median(x, min_count)
        if lower_envelope:
            dev = np.where(x <= med, med - x, nanv)
        else:
            dev = np.abs(x - med)
        out_rows[i] = _move_median(dev, dev_mc)

    _run_threaded(_one, rows.shape[0])
    mad = np.moveaxis(out_rows.reshape(moved.shape), -1, axis)

    if scale:
        mad = mad * MAD_TO_SIGMA

    return xr.DataArray(mad, dims=da.dims, coords=da.coords, name="mad")


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

    Same padding trick as :func:`rolling_mad`: NaN padding keeps the output
    centered while NaNs are ignored within each window.
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


def _as_float_rows(
    da: xr.DataArray, time_dim: str
) -> tuple[np.ndarray, tuple[int, ...]]:
    """View/copy of ``da`` as a float (n_rows, n_time) array, time last.

    Preserves float32 (no silent float64 upcast — halves memory traffic);
    integer input is upcast to float64 so NaN masking works. Also returns the
    time-last shape needed to reshape a per-sample result back.
    """
    x = np.asarray(da.values)
    if x.dtype.kind != "f":
        x = x.astype(np.float64)
    axis = da.get_axis_num(time_dim)
    moved = np.moveaxis(x, axis, -1)
    return moved.reshape(-1, moved.shape[-1]), moved.shape


def _run_threaded(fn: Callable[[int], None], n: int) -> None:
    """Run ``fn(i)`` for i in range(n) on a thread pool.

    Effective because the heavy per-row work (numpy partition-based
    median/quantile, bottleneck move_median) releases the GIL.
    """
    workers = min(32, os.cpu_count() or 1, n)
    if workers <= 1:
        for i in range(n):
            fn(i)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fn, range(n)))


def _rolling_quantile(
    x: np.ndarray, win: int, q: float, min_count: int, n_grid: int = 16
) -> np.ndarray | None:
    """Centered rolling quantile of ``x`` (NaN-aware), evaluated on a coarse
    grid of ``win / n_grid`` steps and linearly interpolated to every sample.

    Windows with fewer than ``min_count`` finite samples are skipped and
    bridged by the interpolation; returns ``None`` when no window qualifies.
    The threshold varies on the window timescale, so the coarse grid loses
    nothing meaningful and keeps the cost negligible.
    """
    n = x.size
    half = win // 2
    step = max(1, win // n_grid)
    centers = np.arange(0, n, step)
    if centers[-1] != n - 1:
        centers = np.append(centers, n - 1)
    vals = np.full(centers.size, np.nan)
    for k, c in enumerate(centers):
        seg = x[max(0, c - half) : c + half + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= min_count:
            vals[k] = np.quantile(seg, q)
    fin = np.isfinite(vals)
    if not fin.any():
        return None
    return np.interp(np.arange(n), centers[fin], vals[fin]).astype(x.dtype)


def lower_half_mad_sigma(
    denoised: xr.DataArray,
    ls: xr.DataArray,
    baseline_quantile: float | None = 0.5,
    window_s: float | None = None,
    time_dim: str = "time",
    scale: float = MAD_TO_SIGMA,
    min_valid_frac: float = 0.25,
) -> xr.DataArray:
    """Baseline noise σ from the *lower half* of the ``ls`` trace.

    The standard yardstick for expressing a denoised trace in noise units
    (``denoised / sigma``): the denoised trace is a fit with the noise already
    removed, so its scale must come from the ``ls`` data it was fitted to.

    Two ideas combine here:

    * **Lower half** — robust scale from only the samples at or below the
      median, i.e. deviations that events (strictly positive) cannot create.
      This is the one-sided alternative to a two-sided MAD; see
      ``rolling_mad(lower_envelope=True)`` for the generic unconditioned
      version.
    * **Baseline conditioning** — first keep only samples whose *denoised*
      level is in its lowest ``baseline_quantile`` fraction, so the median sits
      at the true baseline center. Without this, a synapse that is active most
      of the time has its ls median pulled far above baseline and the estimate
      inflates badly (≈2× here). This is why ``denoised`` is needed, not just
      ``ls``.

    For symmetric baseline noise the lower-half MAD equals the ordinary MAD, so
    the standard ``1.4826`` Gaussian calibration applies unchanged.

    Parameters
    ----------
    denoised, ls : xr.DataArray
        Denoised and LS traces (dims include ``time``). ``denoised`` selects
        baseline samples; the σ is computed on ``ls``.
    baseline_quantile : float or None
        Keep samples with denoised level ≤ this per-synapse quantile before
        estimating σ. ``0.5`` uses the quieter half. ``None`` disables
        conditioning (pure one-sided scale of the whole ls trace — only sound
        for synapses that are quiet most of the time).
    window_s : float or None
        ``None`` (default) -> one σ per synapse (a flat, stable noise floor,
        ideal as a detection threshold). A number -> centered rolling estimate
        of that length (seconds) to track slow baseline drift / bleaching.
        Baseline conditioning is then LOCAL (the quantile of the denoised
        level within each window), so windows always contribute their own
        quiet half; returns per-sample σ (NaN only where a window has fewer
        than ``min_valid_frac`` finite baseline samples, e.g. long NaN gaps).
    scale : float
        MAD→σ factor. ``1.4826`` calibrates to Gaussian σ.
    min_valid_frac : float
        Rolling mode: minimum fraction of a window that must be real baseline
        samples, else NaN.

    Returns
    -------
    xr.DataArray
        ``window_s=None`` -> per-synapse σ (dims = input minus ``time``).
        ``window_s`` set -> per-sample σ (same dims as input).

    Notes
    -----
    Implementation is per-synapse and threaded (numpy's partition-based
    median/quantile and bottleneck's move_median release the GIL), works on
    compressed baseline samples instead of full-size NaN-masked copies, and
    preserves float32 input without upcasting. Same estimator as the original
    vectorized version to within float32 rounding.
    """
    xt, moved_shape = _as_float_rows(ls, time_dim)
    dt = _as_float_rows(denoised, time_dim)[0]
    n_rows = xt.shape[0]
    conditioning = baseline_quantile is not None and baseline_quantile < 1.0
    bq: float = float(baseline_quantile) if conditioning else 1.0

    if window_s is None:
        out_flat = np.full(n_rows, np.nan)

        def _one(i: int) -> None:
            xv, dv = xt[i], dt[i]
            if conditioning:
                dfin = dv[np.isfinite(dv)]
                if dfin.size == 0:
                    return
                thr = np.quantile(dfin, bq)
                sel = (dv <= thr) & np.isfinite(xv)
            else:
                sel = np.isfinite(xv)
            xs = xv[sel]
            if xs.size < 2:
                return
            med = np.median(xs)
            out_flat[i] = scale * np.median(med - xs[xs <= med])

        _run_threaded(_one, n_rows)
        out, dims, coords = _empty_like_reduced(ls, time_dim, "sigma")
        out[...] = out_flat.reshape(out.shape)
        return xr.DataArray(out, dims=dims, coords=coords, name="sigma")

    fs = _infer_fs(ls, time_dim)
    win = _odd_window(window_s, fs)
    med_mc = max(1, win // 8)
    mad_mc = max(1, int(round(min_valid_frac * win / 4)))
    nanv = np.array(np.nan, dtype=xt.dtype)
    mad_rows = np.full_like(xt, np.nan)

    def _one_roll(i: int) -> None:
        xv, dv = xt[i], dt[i]
        if conditioning:
            # LOCAL baseline threshold: the quantile of the denoised level
            # within each window, not of the whole recording. A global
            # threshold starves windows where the denoised level sits above
            # its recording-wide quantile for a long time (e.g. slow calcium
            # early in a bleaching recording: <10% "quiet" samples -> NaN
            # strips), even though the window has a perfectly good baseline.
            thr = _rolling_quantile(dv, win, bq, min_count=med_mc)
            if thr is None:
                return
            xb = np.where(dv <= thr, xv, nanv)
        else:
            xb = xv
        med = _centered_move_median(xb, win, 0, min_count=med_mc)
        lower = np.where(xb <= med, np.abs(xb - med), nanv)
        mad_rows[i] = _centered_move_median(lower, win, 0, min_count=mad_mc)

    _run_threaded(_one_roll, n_rows)
    axis = ls.get_axis_num(time_dim)
    mad = np.moveaxis(mad_rows.reshape(moved_shape), -1, axis)
    return xr.DataArray(
        scale * mad.astype(np.float64), dims=ls.dims, coords=ls.coords, name="sigma"
    )


def rolling_std(
    da: xr.DataArray,
    window: int,
    dim: str = "time",
    center: bool = True,
    ddof: int = 0,
    min_count: int | None = None,
    post_smoothing: bool = False,
    smoothing_window: int | None = None,
    smoothing_method: Literal["median", "mean"] = "median",
) -> xr.DataArray:
    """Centered rolling standard deviation via bottleneck.move_std.

    The std analogue of :func:`rolling_mad`: same windowing, centering and
    NaN handling, but a single pass since move_std needs no median subtraction.

    NaN-tolerant: NaNs in `da` are ignored within each window. Output is
    NaN only where fewer than `min_count` real samples sit in the window
    (i.e. the center of a gap longer than ~window samples). Defaults to
    `max(1, window // 4)` — at least 25% of the window must be real.

    Parameters
    ----------
    ddof : int
        Delta degrees of freedom passed to ``bottleneck.move_std``. 0 (default)
        gives the population std; 1 gives the sample (Bessel-corrected) std.
        This replaces `rolling_mad`'s MAD-specific `scale` argument.
    post_smoothing : bool
        If True, run a second moving pass over the std estimate itself to
        produce a cleaner noise-floor curve. Off by default (the raw rolling
        std is returned unchanged).
    smoothing_window : int | None
        Width, in samples, of the post-smoothing pass. ``None`` (default)
        resolves to ``3 * window``. The reason it is a *multiple* of ``window``
        and not equal to it: a brief event inflates the std for ~``window``
        samples (as long as it sits anywhere inside the sliding window), so the
        bump in the std series has width ~``window``. A moving median of width
        ``w`` only rejects a positive bump of width ``b`` when ``w > 2 * b``;
        ``3 * window`` clears that with margin (empirically the std snaps back to
        the true floor at ~3x and barely moves at 1x). Widen further for more
        aggressive rejection, narrow toward ``window`` to merely de-jitter.
    smoothing_method : {"median", "mean"}
        How to post-smooth. ``"median"`` (default) is the right choice for a
        noise *floor*: a moving median *rejects* the event-induced bump
        outright, so the curve tracks baseline noise rather than activity.
        ``"mean"`` is a plain box smoother — it can only dilute a bump, never
        remove it, so transients always leak into the floor; use it only to
        de-jitter an already event-free estimate. Both are NaN-tolerant
        (bottleneck ``move_*`` with the module's ``max(1, smoothing_window // 4)``
        min-count rule), so short gaps in the std series are bridged while long
        gaps stay NaN.

    Notes
    -----
    Post-smoothing is intentionally distinct from simply widening ``window``.
    A larger ``window`` averages *squared deviations*, so a single event raises
    the estimate proportionally and lingers; a moving median over the std
    series *discards* those excursions. Reach for ``post_smoothing`` when you
    want the noise floor to ignore activity, not just a wider average.
    """
    axis = da.get_axis_num(dim)
    arr = np.ascontiguousarray(da.values)
    if min_count is None:
        min_count = max(1, window // 4)

    if center:
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        pads = [(0, 0)] * arr.ndim
        pads[axis] = (pad_left, pad_right)
        # Pad with NaN, not 'edge' (see rolling_mad): with min_count < window
        # the NaN padding is simply ignored inside each window, whereas 'edge'
        # would smear leading/trailing NaNs across the padding.
        padded = np.pad(arr, pads, mode="constant", constant_values=np.nan)
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(window - 1, None)
        sl = tuple(sl)

        std = bn.move_std(
            padded, window=window, min_count=min_count, axis=axis, ddof=ddof
        )[sl]
    else:
        std = bn.move_std(arr, window=window, min_count=min_count, axis=axis, ddof=ddof)

    out = xr.DataArray(std, dims=da.dims, coords=da.coords, name="std")

    if post_smoothing:
        out = _smooth_estimate(
            out,
            window=3 * window if smoothing_window is None else smoothing_window,
            dim=dim,
            method=smoothing_method,
            center=center,
        )

    return out


def _smooth_estimate(
    da: xr.DataArray,
    *,
    window: int,
    dim: str,
    method: Literal["median", "mean"],
    center: bool,
) -> xr.DataArray:
    """Post-smooth a rolling-statistic series with a NaN-tolerant moving pass.

    Reuses the centered NaN-padding trick from :func:`rolling_std` so the
    smoother is centered and edge-safe, and ignores NaN gaps inside each
    window. ``min_count`` follows the module convention of
    ``max(1, window // 4)``: short gaps in the input series are bridged, long
    gaps stay NaN.
    """
    if window < 2:
        return da

    move_fn = {"median": bn.move_median, "mean": bn.move_mean}.get(method)
    if move_fn is None:
        raise ValueError(
            f"smoothing_method must be 'median' or 'mean', got {method!r}."
        )

    axis = da.get_axis_num(dim)
    arr = np.ascontiguousarray(da.values)
    min_count = max(1, window // 4)

    if center:
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        pads = [(0, 0)] * arr.ndim
        pads[axis] = (pad_left, pad_right)
        padded = np.pad(arr, pads, mode="constant", constant_values=np.nan)
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(window - 1, None)
        smoothed = move_fn(padded, window=window, min_count=min_count, axis=axis)[
            tuple(sl)
        ]
    else:
        smoothed = move_fn(arr, window=window, min_count=min_count, axis=axis)

    return xr.DataArray(smoothed, dims=da.dims, coords=da.coords, name=da.name)


def robust_minmax(df, df_filt, cols_to_normalize, qlower=1, qupper=99):
    """Apply robust min-max normalization to specified columns in a DataFrame."""
    normalized_df = df.clone()
    for col in cols_to_normalize:
        col_data = df_filt[col].to_numpy()
        q1 = np.percentile(col_data, qlower)
        q99 = np.percentile(col_data, qupper)
        denom = q99 - q1
        full_data = df[col].to_numpy()
        norm_data = (full_data - q1) / denom
        # clip to [0, 1]
        norm_data = np.clip(norm_data, 0, 1)
        normalized_df = normalized_df.with_columns(
            pl.Series(name=f"{col}_norm", values=norm_data)
        )
    return normalized_df


def min_max_dlc_dfs(eye_df, whisk_df):
    eye_df = robust_minmax(
        eye_df, eye_df.filter(pl.col("pup_likelihood") > 0.7), ["diameter", "motion"]
    )
    whisk_df = robust_minmax(whisk_df, whisk_df, ["whis"], qlower=1, qupper=99.9)
    return eye_df, whisk_df


def get_sigmas(matchfilt, denoised):
    """
    Compute SNRs for matched filter and denoised traces.
    On matchfilt, we compute rolling_MAD, and on denoised we compute rolling_std.
    """
    mf_sigma = rolling_mad(matchfilt, window=(30 * 200))

    dn_sigma = rolling_std(denoised, window=(300 * 200))

    return mf_sigma, dn_sigma


def get_snrs(matchfilt, denoised):
    """
    Compute SNRs for matched filter and denoised traces.
    On matchfilt, we compute rolling_MAD, and on denoised we compute rolling_std.
    """
    mf_sigma = rolling_mad(matchfilt, window=(30 * 200))
    mfsnr = matchfilt / mf_sigma
    dn_sigma = rolling_std(denoised, window=(300 * 200))
    dnsnr = denoised / dn_sigma
    return mfsnr, dnsnr
