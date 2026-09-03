"""Matched-filter and rolling-MAD noise primitives for fluorescence traces.

General-purpose signal-processing helpers: match-filter LS traces with the
iGluSnFR decay kernel and estimate per-sample noise with a rolling MAD.
Relocated from the retired ``pns.glu_ev_basic_gen`` event-detection pipeline
(2026-08); the on-disk ``event_detection/`` artifacts that pipeline produced
no longer exist, but these primitives remain in use (e.g. the WISynaptic
Soma-Data Loupe preset match-filters traces on the fly).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import ceil

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import fftconvolve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGLUSNFR_TAU_S: float = 0.03  # sensor decay constant (seconds)
DEFAULT_NOISE_WINDOW_S: float = 8.0  # rolling MAD window (seconds)
DEFAULT_MIN_VALID_FRAC: float = 0.5  # min valid samples in kernel window
MAD_TO_SIGMA: float = 1.4826  # scale factor: MAD -> sigma for Gaussian


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------


def _build_kernel(tau_s: float, fs: float) -> np.ndarray:
    """Build an exponential decay kernel, normalized to unit energy.

    Parameters
    ----------
    tau_s : float
        Decay time constant in seconds.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        1-D kernel of shape ``(kernel_len,)``.
    """
    tau_samps = tau_s * fs
    kernel_len = max(2, int(ceil(5 * tau_samps)))
    kernel = np.exp(-np.arange(kernel_len) / tau_samps)
    kernel /= np.linalg.norm(kernel)
    return kernel


def _batch_match_filter(
    data_filled: np.ndarray,
    valid: np.ndarray,
    kernel: np.ndarray,
    min_valid_frac: float,
) -> np.ndarray:
    """Batch matched filter across all synapses simultaneously.

    Parameters
    ----------
    data_filled : np.ndarray, shape ``(n_syn, n_time)``
        Trace data with NaNs replaced by 0.
    valid : np.ndarray, shape ``(n_syn, n_time)``
        Binary mask of valid (non-NaN) samples (float64).
    kernel : np.ndarray, shape ``(kernel_len,)``
        1-D matched-filter kernel.
    min_valid_frac : float
        Minimum fraction of valid samples within the kernel window.
        Output is NaN where this threshold is not met.

    Returns
    -------
    np.ndarray, shape ``(n_syn, n_time)``
        Matched-filter output with NaN where validity is insufficient.
    """
    # Convolve all synapses at once along the time axis
    kernel_2d = kernel[::-1][np.newaxis, :]
    filtered = fftconvolve(data_filled, kernel_2d, mode="same", axes=-1)

    # Count valid samples contributing to each output sample
    ones_2d = np.ones((1, len(kernel)))
    valid_count = fftconvolve(valid, ones_2d, mode="same", axes=-1)

    # Mask where too few valid samples contributed
    filtered[valid_count < min_valid_frac * len(kernel)] = np.nan

    return filtered


def _rolling_mad_single(
    x: np.ndarray, window: int, min_periods: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling MAD noise estimate for a single synapse trace.

    Parameters
    ----------
    x : np.ndarray, shape ``(n_time,)``
        Matched-filter output for one synapse.
    window : int
        Rolling window size (should be odd).
    min_periods : int
        Minimum number of non-NaN observations required.

    Returns
    -------
    noise_std : np.ndarray, shape ``(n_time,)``
        MAD scaled to sigma.
    median : np.ndarray, shape ``(n_time,)``
        Rolling median.
    """
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(
        window, center=True, min_periods=min_periods
    ).median() * MAD_TO_SIGMA
    return mad.values, med.values


def _batch_rolling_mad(
    filtered: np.ndarray,
    noise_win: int,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parallel rolling MAD across all synapses using ThreadPoolExecutor.

    Parameters
    ----------
    filtered : np.ndarray, shape ``(n_syn, n_time)``
        Matched-filter output.
    noise_win : int
        Window size in samples (must be odd).
    max_workers : int or None
        Thread pool size. Defaults to ``min(n_syn, 32)``.

    Returns
    -------
    noise_std : np.ndarray, shape ``(n_syn, n_time)``
    median : np.ndarray, shape ``(n_syn, n_time)``
    """
    n_syn = filtered.shape[0]
    if max_workers is None:
        max_workers = min(n_syn, 32)

    min_periods = max(1, noise_win // 4)
    noise_std = np.empty_like(filtered)
    median = np.empty_like(filtered)

    def _process(i: int) -> tuple[int, np.ndarray, np.ndarray]:
        ns, md = _rolling_mad_single(filtered[i], noise_win, min_periods)
        return i, ns, md

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, ns, md in pool.map(_process, range(n_syn)):
            noise_std[i] = ns
            median[i] = md

    return noise_std, median


# ---------------------------------------------------------------------------
# Core match-filter + noise estimation
# ---------------------------------------------------------------------------


def _match_filter_core(
    ls: xr.DataArray,
    tau_s: float,
    min_valid_frac: float,
) -> tuple[np.ndarray, float]:
    """Shared matched-filter computation for the public wrappers.

    Parameters
    ----------
    ls : xr.DataArray
        LS traces with dims ``(syn_id, time)`` (or 1-D ``(time,)``).
        May contain NaNs (censored / motion-artifact samples).
    tau_s : float
        Transient decay time constant in seconds.
    min_valid_frac : float
        Minimum fraction of valid (non-NaN) samples within the kernel
        window. Output is NaN where this threshold is not met.

    Returns
    -------
    filtered : np.ndarray
        Float64 matched-filter output, same shape as the input (1-D
        input stays 1-D).
    fs : float
        Sampling rate inferred from the time coordinate (Hz).
    """
    time = ls.coords["time"].values
    dt = float(np.median(np.diff(time)))
    fs = 1.0 / dt

    kernel = _build_kernel(tau_s, fs)

    # Ensure 2D (syn_id, time)
    squeezed = ls.ndim == 1
    if squeezed:
        data = ls.values[np.newaxis, :]
    else:
        data = ls.values

    # Prepare NaN-safe inputs
    nan_mask = np.isnan(data)
    data_filled = np.where(nan_mask, 0.0, data).astype(np.float64)
    valid = (~nan_mask).astype(np.float64)

    # Batch matched filter (single FFT call for all synapses)
    filtered = _batch_match_filter(data_filled, valid, kernel, min_valid_frac)

    if squeezed:
        filtered = filtered[0]
    return filtered, fs


def match_filter(
    ls: xr.DataArray,
    tau_s: float = IGLUSNFR_TAU_S,
    min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
) -> xr.DataArray:
    """Match-filter LS traces without estimating noise.

    Filter-only variant of :func:`match_filter_with_noise`, for callers
    that normalize the filtered traces themselves (e.g. the WISynaptic
    Soma-Data Loupe preset) and would otherwise pay for a rolling-MAD
    noise estimate they discard.

    Parameters
    ----------
    ls : xr.DataArray
        LS traces with dims ``(syn_id, time)`` (or 1-D ``(time,)``).
        May contain NaNs (censored / motion-artifact samples).
    tau_s : float
        Transient decay time constant in seconds.
    min_valid_frac : float
        Minimum fraction of valid (non-NaN) samples within the kernel
        window. Output is NaN where this threshold is not met.

    Returns
    -------
    xr.DataArray
        Matched-filter output (float32) with the same dims and
        dimension coords as the input.
    """
    filtered, _ = _match_filter_core(ls, tau_s, min_valid_frac)
    coords = {dim: ls.coords[dim] for dim in ls.dims}
    return xr.DataArray(filtered.astype(np.float32), dims=ls.dims, coords=coords)


def match_filter_with_noise(
    ls: xr.DataArray,
    tau_s: float = IGLUSNFR_TAU_S,
    noise_window_s: float = DEFAULT_NOISE_WINDOW_S,
    min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
) -> xr.Dataset:
    """Match-filter LS traces and estimate per-sample noise.

    Vectorized implementation: the matched filter is computed for all
    synapses in a single batched FFT call, and the rolling MAD noise
    estimate is parallelized across synapses using threads.

    Parameters
    ----------
    ls : xr.DataArray
        LS traces with dims ``(syn_id, time)``. May contain NaNs
        (censored / motion-artifact samples).
    tau_s : float
        Transient decay time constant in seconds.
    noise_window_s : float
        Window for rolling MAD noise estimate, in seconds.
    min_valid_frac : float
        Minimum fraction of valid (non-NaN) samples within the kernel
        window. Output is NaN where this threshold is not met.

    Returns
    -------
    xr.Dataset
        Variables ``filtered`` and ``noise_std``, each with the same
        dims/coords as the input.
    """
    filtered, fs = _match_filter_core(ls, tau_s, min_valid_frac)

    # Rolling MAD noise estimate (threaded across synapses)
    noise_win = max(3, int(np.round(noise_window_s * fs)))
    if noise_win % 2 == 0:
        noise_win += 1
    squeezed = filtered.ndim == 1
    noise_std, _ = _batch_rolling_mad(
        filtered[np.newaxis, :] if squeezed else filtered, noise_win
    )
    if squeezed:
        noise_std = noise_std[0]

    coords = {dim: ls.coords[dim] for dim in ls.dims}
    return xr.Dataset({
        "filtered": xr.DataArray(
            filtered.astype(np.float32), dims=ls.dims, coords=coords
        ),
        "noise_std": xr.DataArray(
            noise_std.astype(np.float32), dims=ls.dims, coords=coords
        ),
    })


def rolling_mad_noise(
    signal: xr.DataArray,
    noise_window_s: float = DEFAULT_NOISE_WINDOW_S,
) -> xr.Dataset:
    """Estimate per-sample rolling MAD noise on a pre-cleaned signal trace.

    For signals used directly as a detection signal (no matched filter),
    e.g. denoised traces. Mirrors the noise-estimation branch of
    :func:`match_filter_with_noise`.

    Parameters
    ----------
    signal : xr.DataArray
        Signal traces with dims ``(syn_id, time)``. May contain NaNs.
    noise_window_s : float
        Window for rolling MAD noise estimate, in seconds.

    Returns
    -------
    xr.Dataset
        Variable ``noise_std`` with the same dims/coords as the input.
    """
    time = signal.coords["time"].values
    dt = float(np.median(np.diff(time)))
    fs = 1.0 / dt

    squeezed = signal.ndim == 1
    if squeezed:
        data = signal.values[np.newaxis, :].astype(np.float64)
    else:
        data = signal.values.astype(np.float64)

    noise_win = max(3, int(np.round(noise_window_s * fs)))
    if noise_win % 2 == 0:
        noise_win += 1
    noise_std, _ = _batch_rolling_mad(data, noise_win)

    if squeezed:
        noise_std = noise_std[0]

    coords = {dim: signal.coords[dim] for dim in signal.dims}
    return xr.Dataset({
        "noise_std": xr.DataArray(
            noise_std.astype(np.float32), dims=signal.dims, coords=coords
        ),
    })
