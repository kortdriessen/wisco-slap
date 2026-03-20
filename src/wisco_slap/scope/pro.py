"""
Trace post-processing: recreate NoLoCo-era trace variants from the LoCo LS trace.

Functions here apply temporal filters / deconvolutions to scopex xarray
DataArrays (typically dims ``["channel", "syn_id", "time"]``).
"""

import numpy as np
import xarray as xr
from scipy.signal import fftconvolve, lfilter

IGLUSNFR_TAU_S: float = 0.026  # 26 ms decay constant (from iGluSnFR4f paper)


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
