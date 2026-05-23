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

from typing import Any, Literal

import bottleneck as bn
import numpy as np
import xarray as xr
from scipy.signal import find_peaks


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
    for s, e in zip(starts, ends):
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
) -> xr.DataArray:
    """Centered rolling MAD via two passes of bottleneck.move_median.

    ~10-50x faster than xr.rolling().construct().median() for large windows.

    NaN-tolerant: NaNs in `da` are ignored within each window. Output is
    NaN only where fewer than `min_count` real samples sit in the window
    (i.e. the center of a gap longer than ~window samples). Defaults to
    `max(1, window // 4)` — at least 25% of the window must be real.
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
        # Pad with NaN, not 'edge': edge mode replicates leading/trailing NaNs
        # into a stripe across the padding. With min_count<window, NaN padding
        # is simply ignored inside each window.
        padded = np.pad(arr, pads, mode="constant", constant_values=np.nan)
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(window - 1, None)
        sl = tuple(sl)

        med = bn.move_median(padded, window=window, min_count=min_count, axis=axis)[sl]
        abs_dev = np.abs(arr - med)
        abs_dev_padded = np.pad(abs_dev, pads, mode="constant", constant_values=np.nan)
        mad = bn.move_median(
            abs_dev_padded, window=window, min_count=min_count, axis=axis
        )[sl]
    else:
        med = bn.move_median(arr, window=window, min_count=min_count, axis=axis)
        mad = bn.move_median(
            np.abs(arr - med), window=window, min_count=min_count, axis=axis
        )

    if scale:
        mad = mad * 1.4826

    return xr.DataArray(mad, dims=da.dims, coords=da.coords, name="mad")
