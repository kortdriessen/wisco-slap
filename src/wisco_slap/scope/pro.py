"""
Trace post-processing: recreate NoLoCo-era trace variants from the LoCo LS trace.

Functions here apply temporal filters / deconvolutions to scopex xarray
DataArrays (typically dims ``["channel", "syn_id", "time"]``). This module
also includes ROI dF/F helpers for ScopeX user-defined ROIs.
"""

import warnings

import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import PchipInterpolator
from scipy.signal import fftconvolve, lfilter

IGLUSNFR_TAU_S: float = 0.026  # 26 ms decay constant (from iGluSnFR4f paper)
ROI_DFF_DENOISE_WINDOW_S: float = 0.2
ROI_DFF_BASELINE_WINDOW_GLU_S: float = 4.0
ROI_DFF_BASELINE_WINDOW_CA_S: float = 4.0
_ROI_DFF_DIV_ABS_EPS: float = 1e-12
_ROI_DFF_DIV_REL_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_fs(da: xr.DataArray) -> float:
    """Extract sampling rate (Hz) from the ``time`` coordinate."""
    time = da.coords["time"].values
    return 1.0 / float(np.median(np.diff(time)))


def _matched_exp_filter(data: np.ndarray, tau_frames: float) -> np.ndarray:
    """Backward exponential matched filter along the last axis.

    Equivalent to MATLAB ``matchedExpFilter.m``: runs backward in time,
    combining each sample with the next (future) sample using exponential
    decay weighting ``gamma = exp(-1/tau_frames)``::

        D[t] = gamma * D[t+1] + (1 - gamma) * D[t]

    Uses a fast ``scipy.signal.lfilter`` path when there are no NaNs.
    Falls back to a per-sample loop (faithful to the MATLAB NaN-aware
    weighting) when NaNs are present.
    """
    gamma = np.exp(-1.0 / tau_frames)
    out = data.astype(np.float64, copy=True)
    has_nans = np.isnan(out).any()

    if not has_nans:
        return _matched_exp_filter_fast(out, gamma)
    return _matched_exp_filter_nanaware(out, gamma)


def _matched_exp_filter_fast(out: np.ndarray, gamma: float) -> np.ndarray:
    """Fast path using ``lfilter`` on time-reversed data (no NaNs)."""
    # Boundary condition: D[T] = gamma * D[T]  (MATLAB line 6)
    out[..., -1] *= gamma

    # Reshape to 2D for lfilter: (n_sources, time)
    orig_shape = out.shape
    flat = out.reshape(-1, orig_shape[-1])

    # Flip time axis to convert anti-causal → causal IIR
    flipped = np.ascontiguousarray(flat[:, ::-1])

    b = np.array([1.0 - gamma])
    a = np.array([1.0, -gamma])

    # Initial condition: set so that y[0] = flipped[0] (preserves
    # the MATLAB boundary condition rather than starting from rest).
    # lfilter (direct-form II transposed):
    #   y[0] = (1-gamma)*x[0] + zi[0]
    # We want y[0] = x[0], so zi[0] = gamma * x[0].
    zi = gamma * flipped[:, 0:1]  # shape (n_sources, 1)
    filtered, _ = lfilter(b, a, flipped, axis=1, zi=zi)

    return filtered[:, ::-1].reshape(orig_shape)


def _matched_exp_filter_nanaware(out: np.ndarray, gamma: float) -> np.ndarray:
    """Slow loop path that mirrors MATLAB NaN-aware weighting exactly."""
    nan_mask = np.isnan(out)
    out[..., -1] *= gamma

    for t in range(out.shape[-1] - 2, -1, -1):
        next_valid = ~np.isnan(out[..., t + 1])
        curr_valid = ~np.isnan(out[..., t])

        w1 = gamma * next_valid.astype(np.float64)
        w2 = (1.0 - gamma) * curr_valid.astype(np.float64)

        next_val = np.where(next_valid, out[..., t + 1], 0.0)
        curr_val = np.where(curr_valid, out[..., t], 0.0)

        denom = w1 + w2
        safe = denom > 0
        out[..., t] = np.where(
            safe,
            (w1 * next_val + w2 * curr_val) / np.where(safe, denom, 1.0),
            0.0,
        )

    out[nan_mask] = np.nan
    return out


def _make_exp_kernel(tau_frames: float, n_tau: float = 8.0) -> np.ndarray:
    """Normalised forward exponential-decay kernel with leading zeros.

    Matches the MATLAB construction::

        kernel = [zeros(1, ceil(n_tau*tau)) exp(-(0:ceil(n_tau*tau))/tau)];
        kernel = kernel ./ sum(kernel);
    """
    pad = int(np.ceil(n_tau * tau_frames))
    decay = np.exp(-np.arange(pad + 1) / tau_frames)
    kernel = np.concatenate([np.zeros(pad), decay])
    kernel /= kernel.sum()
    return kernel


def _conv1d_last(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve *data* with a 1-D *kernel* along the last axis (mode='same')."""
    shape = [1] * (data.ndim - 1) + [len(kernel)]
    return fftconvolve(data, kernel.reshape(shape), mode="same", axes=-1)


def _richardson_lucy(
    observed: np.ndarray,
    psf: np.ndarray,
    n_iter: int,
    weights: np.ndarray,
    initial_estimate: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted Richardson-Lucy deconvolution along the last axis.

    Non-negativity is enforced at every iteration.  *weights* should be 1
    where data is valid and 0 where it is missing (NaN positions).
    """
    eps = 1e-12
    psf_flip = psf[::-1].copy()

    if initial_estimate is not None:
        estimate = np.maximum(initial_estimate.copy(), eps)
    else:
        estimate = np.maximum(observed.copy(), eps)

    for _ in range(n_iter):
        est_conv = _conv1d_last(estimate, psf)
        ratio = observed / np.maximum(est_conv, eps)
        ratio *= weights
        correction = _conv1d_last(ratio, psf_flip)
        estimate = np.maximum(estimate * correction, 0.0)

    return estimate


def _window_to_samples(window_s: float, fs: float) -> int:
    """Convert a window in seconds to a centered moving-window sample count."""
    return max(1, int(round(window_s * fs)))


def _centered_nan_windows(trace: np.ndarray, window: int) -> np.ndarray:
    """Return centered, edge-truncated windows via NaN padding."""
    window = max(1, int(window))
    left = (window - 1) // 2
    right = window - left - 1
    padded = np.pad(trace, (left, right), constant_values=np.nan)
    return sliding_window_view(padded, window)


def _moving_nanmedian_1d(trace: np.ndarray, window: int) -> np.ndarray:
    """Centered moving median that mirrors MATLAB ``medfilt1(..., 'truncate')``."""
    windows = _centered_nan_windows(trace, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(windows, axis=-1)


def _moving_nanmean_1d(trace: np.ndarray, window: int) -> np.ndarray:
    """Centered moving mean with NaN omission and edge truncation."""
    windows = _centered_nan_windows(trace, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(windows, axis=-1)


def _interp_linear_preserve_nan_gaps(
    x: np.ndarray,
    y: np.ndarray,
    xq: np.ndarray,
) -> np.ndarray:
    """Interpolate over contiguous finite segments, leaving NaN gaps unfilled."""
    out = np.full(xq.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(y)
    if not finite.any():
        return out

    start = 0
    while start < y.size:
        if not finite[start]:
            start += 1
            continue

        end = start
        while end + 1 < y.size and finite[end + 1]:
            end += 1

        x_seg = x[start : end + 1]
        y_seg = y[start : end + 1]
        if x_seg.size == 1:
            out[xq == x_seg[0]] = y_seg[0]
        else:
            mask = (xq >= x_seg[0]) & (xq <= x_seg[-1])
            out[mask] = np.interp(xq[mask], x_seg, y_seg)

        start = end + 1

    return out


def _safe_pchip_interp(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """PCHIP interpolation with graceful fallbacks for degenerate inputs."""
    finite = np.isfinite(y)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return np.full(xq.shape, np.nan, dtype=np.float64)
    if n_finite == 1:
        return np.full(xq.shape, y[finite][0], dtype=np.float64)
    return PchipInterpolator(x[finite], y[finite], extrapolate=True)(xq)


def _compute_f0_algo1_1d(
    trace: np.ndarray,
    denoise_window: int,
    hull_window: int,
) -> np.ndarray:
    """Port of MATLAB ``computeF0(..., algo1)`` for a single 1-D trace."""
    trace = np.asarray(trace, dtype=np.float64)
    nan_mask = np.isnan(trace)
    if nan_mask.all():
        return trace.copy()

    n_time = trace.size
    hull_window = max(1, min(int(hull_window), max(1, n_time // 4)))
    delta_des = max(4.0, denoise_window / 6.0)
    sample_times = np.rint(
        np.linspace(0, n_time - 1, int(np.ceil(n_time / delta_des)) + 1)
    ).astype(int)
    sample_times = np.unique(sample_times)
    n_samps_in_hull = max(
        1,
        min(int(np.ceil(hull_window / delta_des)), sample_times.size),
    )

    f0 = _moving_nanmedian_1d(trace, denoise_window)

    f00 = np.full((sample_times.size, n_samps_in_hull), np.nan, dtype=np.float64)
    for offset in range(n_samps_in_hull):
        xi = sample_times[offset::n_samps_in_hull]
        if xi.size == 0:
            continue
        yi = f0[xi]
        f00[:, offset] = _interp_linear_preserve_nan_gaps(xi, yi, sample_times)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ff = np.nanmin(f00, axis=1)

    doubt = np.sum(~np.isnan(f00), axis=1) < int(np.ceil(n_samps_in_hull / 2))
    f2 = ff.copy()
    if np.sum(~doubt) > 2:
        f2[doubt] = np.nan

    smooth_window = 2 * int(np.ceil(n_samps_in_hull / 2)) + 1
    fill = _moving_nanmean_1d(f2, smooth_window)
    missing = np.isnan(f2)
    f2[missing] = fill[missing]
    f2 = _moving_nanmean_1d(f2, smooth_window)

    f0 = _safe_pchip_interp(sample_times, f2, np.arange(n_time, dtype=np.float64))
    f0[nan_mask] = np.nan
    return f0


def _baseline_window_s_for_channel(channel_value: int | float) -> float:
    """Map ScopeX channel order to the matching processSLAP2 baseline window."""
    if int(channel_value) == 0:
        return ROI_DFF_BASELINE_WINDOW_GLU_S
    return ROI_DFF_BASELINE_WINDOW_CA_S


def _safe_f0_for_division(f0: np.ndarray) -> np.ndarray:
    """Clamp only near-zero baselines to avoid division blow-ups."""
    safe = f0.astype(np.float64, copy=True)
    finite = np.abs(safe[np.isfinite(safe)])
    if finite.size == 0:
        return safe

    floor = max(_ROI_DFF_DIV_ABS_EPS, float(np.nanmedian(finite)) * _ROI_DFF_DIV_REL_EPS)
    small = np.abs(safe) < floor
    safe[small] = np.where(safe[small] < 0, -floor, floor)
    return safe


def _channel_value_for_trace(
    roi_scopex_dataarray: xr.DataArray,
    trace_index: tuple[int, ...],
) -> int | float:
    """Resolve the ScopeX channel label for one flattened trace."""
    non_time_dims = roi_scopex_dataarray.dims[:-1]
    if "channel" in non_time_dims:
        channel_axis = non_time_dims.index("channel")
        channel_values = np.asarray(roi_scopex_dataarray.coords["channel"].values)
        return channel_values[trace_index[channel_axis]].item()

    if "channel" in roi_scopex_dataarray.coords:
        channel_values = np.asarray(roi_scopex_dataarray.coords["channel"].values)
        if channel_values.size == 1:
            return channel_values.reshape(()).item()

    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ls_to_matchFilt(
    ls_scopex_dataarray: xr.DataArray,
    tau_s: float = IGLUSNFR_TAU_S,
) -> xr.DataArray:
    """Convert an LS trace to a matched-filter trace.

    Recreates the ``matchFilt`` trace from the old NoLoCo pipeline by
    applying a backward exponential filter matched to the indicator decay
    kinetics.  The filter runs backward in time, combining each sample with
    future samples weighted by exponential decay
    (``gamma = exp(-1 / (tau_s * fs))``), boosting SNR for transient events
    with the known decay shape.

    Because the per-timepoint spatial LS solve and the temporal matched
    filter commute, applying the filter *after* the LS solve gives an
    equivalent result to the old pipeline (which filtered raw pixels first).

    Parameters
    ----------
    ls_scopex_dataarray : xr.DataArray
        LS scopex DataArray (typically dims ``["channel", "syn_id", "time"]``).
    tau_s : float
        Indicator decay time constant in seconds.
        Default ``IGLUSNFR_TAU_S`` (0.026 s).  Use ~0.15 for jRGECO1a calcium.

    Returns
    -------
    xr.DataArray
        Matched-filtered trace with the same shape, dims, and coords.
    """
    fs = _get_fs(ls_scopex_dataarray)
    tau_frames = tau_s * fs

    filtered = _matched_exp_filter(ls_scopex_dataarray.values, tau_frames)

    return xr.DataArray(
        filtered.astype(np.float32),
        dims=ls_scopex_dataarray.dims,
        coords=ls_scopex_dataarray.coords,
        attrs={**ls_scopex_dataarray.attrs, "trace_type": "matchFilt", "tau_s": tau_s},
    )


def ls_to_nonneg(
    ls_scopex_dataarray: xr.DataArray,
    tau_s: float = IGLUSNFR_TAU_S,
    rl_iterations: int = 20,
    nan_fill_rounds: int = 3,
) -> xr.DataArray:
    """Convert an LS trace to a nonneg trace via matched filter + Richardson-Lucy.

    Recreates the ``nonneg`` trace from the old NoLoCo pipeline:

    1. Apply matched exponential filter to the LS trace.
    2. Shift the signal positive (required by Richardson-Lucy).
    3. Deconvolve with the backward (time-reversed) exponential kernel using
       weighted Richardson-Lucy, enforcing non-negativity.
    4. Iteratively fill NaN positions with reconstructed values and
       re-deconvolve (``nan_fill_rounds`` times), continuing from the
       previous estimate each round.
    5. Remove the positivity shift.

    The result retains some temporal smoothing from the forward exponential
    (only the backward component is deconvolved out) while imposing weak
    non-negativity via RL — sitting between ``matchFilt`` and fully
    deconvolved events.

    Parameters
    ----------
    ls_scopex_dataarray : xr.DataArray
        LS scopex DataArray (typically dims ``["channel", "syn_id", "time"]``).
    tau_s : float
        Indicator decay time constant in seconds.
        Default ``IGLUSNFR_TAU_S`` (0.026 s).  Use ~0.15 for jRGECO1a calcium.
    rl_iterations : int
        Richardson-Lucy iterations per round.  Default ``20``.
    nan_fill_rounds : int
        Number of reconstruct → re-deconvolve rounds.  Default ``3``.

    Returns
    -------
    xr.DataArray
        Nonneg trace with the same shape, dims, and coords.
    """
    fs = _get_fs(ls_scopex_dataarray)
    tau_frames = tau_s * fs

    # 1. Matched filter
    match_data = _matched_exp_filter(ls_scopex_dataarray.values, tau_frames)

    # 2. Build backward (time-reversed) kernel
    kernel = _make_exp_kernel(tau_frames)
    f_kernel = kernel[::-1].copy()

    # 3. Shift for positivity  (mirrors Hsub in processTrialAsync.m:
    #    Hsub = median(H, 2, 'omitnan') - 2*prctile(H, 1, 2);  H = H + Hsub)
    nan_mask = np.isnan(match_data)
    hsub = np.nanmedian(match_data, axis=-1, keepdims=True) - 2.0 * np.nanpercentile(
        match_data, 1, axis=-1, keepdims=True
    )
    observed = np.where(nan_mask, 0.0, match_data) + hsub
    observed = np.maximum(observed, 0.0)

    # Weights: 1 = valid, 0 = NaN position
    weights = (~nan_mask).astype(np.float64)

    # 4. Initial RL deconvolution
    estimate = _richardson_lucy(observed, f_kernel, rl_iterations, weights)

    # 5. Iterative NaN-fill rounds (continue from previous estimate).
    #    Only needed when NaNs are present — skip otherwise.
    if nan_mask.any():
        for _ in range(nan_fill_rounds):
            recon = _conv1d_last(estimate, f_kernel)
            observed[nan_mask] = recon[nan_mask]
            estimate = _richardson_lucy(
                observed, f_kernel, rl_iterations, weights, initial_estimate=estimate
            )

    # 6. Remove shift, restore NaNs
    result = estimate - hsub
    result[nan_mask] = np.nan

    return xr.DataArray(
        result.astype(np.float32),
        dims=ls_scopex_dataarray.dims,
        coords=ls_scopex_dataarray.coords,
        attrs={**ls_scopex_dataarray.attrs, "trace_type": "nonneg", "tau_s": tau_s},
    )


def roi_to_dff(
    roi_scopex_dataarray: xr.DataArray,
    *,
    trace: str | None = None,
    return_f0: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """Convert ScopeX ROI fluorescence traces to dF/F.

    This uses a trace-level port of MATLAB ``computeF0(..., algo1)`` on the
    ROI trace itself. That is the most principled loader-side analogue for
    user-defined ROIs, because the source-level NMF background model used to
    compute synaptic ``F0`` is not available for ROI traces after extraction.

    Parameters
    ----------
    roi_scopex_dataarray : xr.DataArray
        ROI trace DataArray. ScopeX ROI stores are typically shaped
        ``(channel, soma_id, time)``.
    trace : str or None
        Optional source trace label (for example ``"Fsvd"`` or ``"Fraw"``).
        This is recorded in the output attrs for provenance only.
    return_f0 : bool
        If True, also return the computed baseline ``F0`` DataArray.

    Returns
    -------
    xr.DataArray or tuple[xr.DataArray, xr.DataArray]
        dF/F DataArray with the same dims and coordinates as the input, and
        optionally the matching ``F0`` DataArray.
    """
    if "time" not in roi_scopex_dataarray.dims:
        raise ValueError("roi_scopex_dataarray must have a 'time' dimension")

    original_dims = roi_scopex_dataarray.dims
    working = roi_scopex_dataarray
    if working.get_axis_num("time") != working.ndim - 1:
        working = working.transpose(
            *[dim for dim in working.dims if dim != "time"],
            "time",
        )

    fs = _get_fs(working)
    denoise_window = _window_to_samples(ROI_DFF_DENOISE_WINDOW_S, fs)
    values = np.asarray(working.values, dtype=np.float64)
    f0_values = np.full_like(values, np.nan, dtype=np.float64)
    dff_values = np.full_like(values, np.nan, dtype=np.float64)

    non_time_shape = values.shape[:-1]
    index_iter = np.ndindex(non_time_shape) if non_time_shape else [()]
    for trace_index in index_iter:
        channel_value = _channel_value_for_trace(working, trace_index)
        baseline_window = _window_to_samples(
            _baseline_window_s_for_channel(channel_value),
            fs,
        )
        trace_values = values[trace_index]
        trace_f0 = _compute_f0_algo1_1d(
            trace_values,
            denoise_window=denoise_window,
            hull_window=baseline_window,
        )
        trace_f0_safe = _safe_f0_for_division(trace_f0)
        trace_dff = (trace_values - trace_f0) / trace_f0_safe
        trace_dff[np.isnan(trace_values)] = np.nan

        f0_values[trace_index] = trace_f0
        dff_values[trace_index] = trace_dff

    base_attrs = dict(roi_scopex_dataarray.attrs)
    if trace is not None:
        base_attrs["source_trace"] = trace

    dff_da = xr.DataArray(
        dff_values.astype(np.float32),
        dims=working.dims,
        coords=working.coords,
        attrs={**base_attrs, "signal_type": "dFF"},
    )
    f0_da = xr.DataArray(
        f0_values.astype(np.float32),
        dims=working.dims,
        coords=working.coords,
        attrs={**base_attrs, "signal_type": "F0"},
    )

    if working.dims != original_dims:
        dff_da = dff_da.transpose(*original_dims)
        f0_da = f0_da.transpose(*original_dims)

    if return_f0:
        return dff_da, f0_da
    return dff_da
