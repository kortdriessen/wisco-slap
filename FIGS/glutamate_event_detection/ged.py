"""
Glutamate release event detection pipeline based on constrained deconvolution.

The implementation follows the instructions in
`FIGS/glutamate_event_detection/detection_instructions.md`:
 - split traces around NaN gaps (gaps are motion artifacts)
 - rolling percentile baseline (20th percentile over 1 s)
 - normalize to dF/F
 - constrained deconvolution with AR(1) kernel (FOOPSI-style, non-negative)
 - spike/event extraction with a noise-adaptive threshold
 - QC plotting with reconstruction, spike train, and residuals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import lsq_linear


@dataclass
class Event:
    timestamp: float  # seconds
    magnitude: float  # spike amplitude (a.u.)
    index: int  # sample index in the original trace


@dataclass
class DetectionResult:
    raw_trace: np.ndarray
    baseline: np.ndarray
    dff: np.ndarray
    reconstruction: np.ndarray
    spikes: np.ndarray
    events: List[Event]
    sigma: float
    snr: float
    g: float
    fs: float
    tau: float


def _split_contiguous_segments(trace: np.ndarray) -> List[Tuple[int, int]]:
    """Return (start, end) indices of finite contiguous segments."""
    mask = np.isfinite(trace)
    n = len(trace)
    segments: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        segments.append((start, i))
    return segments


def _rolling_percentile_baseline(
    segment: np.ndarray,
    window_frames: int,
    percentile: float,
) -> np.ndarray:
    """Compute rolling percentile baseline for a NaN-free segment."""
    if window_frames < 1:
        raise ValueError("window_frames must be >= 1")
    series = pd.Series(segment)
    # center=True keeps the baseline aligned to each point
    baseline = (
        series.rolling(window_frames, min_periods=1, center=True)
        .quantile(percentile / 100.0)
        .to_numpy()
    )
    return baseline


def _safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Elementwise division that protects against zeros in the denominator."""
    eps = np.finfo(float).eps
    safe = np.where(np.abs(denom) < eps, eps, denom)
    return numer / safe


def _estimate_noise_sigma(trace: np.ndarray) -> float:
    """Robust noise estimate from the first derivative using MAD."""
    if trace.size < 2:
        return 0.0
    diff = np.diff(trace)
    mad = np.median(np.abs(diff - np.median(diff)))
    return 1.4826 * mad


def _build_ar1_convolution_matrix(
    length: int, g: float, impulse_len: Optional[int]
) -> sparse.csr_matrix:
    """
    Build a sparse lower-triangular convolution matrix for an AR(1) kernel.

    Each spike contributes g^k to subsequent samples; we truncate the kernel
    at impulse_len to keep the matrix banded and efficient.
    """
    if impulse_len is None or impulse_len < 1:
        impulse_len = length
    impulse_len = min(impulse_len, length)
    diagonals = []
    offsets = []
    for lag in range(impulse_len):
        offsets.append(-lag)
        diagonals.append(np.full(length - lag, g**lag, dtype=float))
    return sparse.diags(diagonals, offsets, shape=(length, length), format="csr")


def _deconvolve_segment(
    segment_dff: np.ndarray,
    g: float,
    sigma: float,
    *,
    impulse_len: Optional[int],
    s_min_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Non-negative constrained deconvolution via NNLS on a banded AR(1) matrix.

    Returns (spikes, reconstruction).
    """
    n = len(segment_dff)
    if n == 0:
        return np.array([]), np.array([])
    A = _build_ar1_convolution_matrix(n, g, impulse_len)
    # Fit spikes with non-negativity
    res = lsq_linear(A, segment_dff, bounds=(0.0, np.inf), lsmr_tol="auto")
    spikes = res.x
    # Enforce minimum spike size based on noise
    if sigma > 0 and s_min_factor > 0:
        thresh = s_min_factor * sigma
        spikes[spikes < thresh] = 0.0
    reconstruction = A @ spikes
    return spikes, reconstruction


def _extract_events(
    spikes: np.ndarray,
    *,
    fs: float,
    start_index: int,
    threshold: float,
) -> List[Event]:
    """Collapse contiguous supra-threshold bins into discrete events."""
    events: List[Event] = []
    if spikes.size == 0:
        return events
    mask = spikes > threshold
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return events
    # group contiguous indices
    group_start = idx[0]
    current_group = [group_start]
    for k in idx[1:]:
        if k == current_group[-1] + 1:
            current_group.append(k)
        else:
            peak_idx = current_group[np.argmax(spikes[current_group])]
            events.append(
                Event(
                    timestamp=(start_index + peak_idx) / fs,
                    magnitude=float(spikes[peak_idx]),
                    index=start_index + peak_idx,
                )
            )
            current_group = [k]
    peak_idx = current_group[np.argmax(spikes[current_group])]
    events.append(
        Event(
            timestamp=(start_index + peak_idx) / fs,
            magnitude=float(spikes[peak_idx]),
            index=start_index + peak_idx,
        )
    )
    return events


def detect_glutamate_events(
    trace: Sequence[float],
    *,
    fs: float = 200.0,
    tau: float = 0.026,
    baseline_window_s: float = 4.0,
    baseline_percentile: float = 20.0,
    detrend: bool = False,
    s_min_factor: float = 3.0,
    min_spike_threshold_sigma: float = 0.1,
    impulse_len: Optional[int] = None,
) -> DetectionResult:
    """
    Full pipeline to detect glutamate release events from a 1D trace.

    Parameters
    ----------
    trace : sequence of float
        Raw fluorescence trace with NaNs marking invalid intervals.
    fs : float
        Sampling rate in Hz.
    tau : float
        Decay constant in seconds (26 ms for iGluSnFR4f).
    baseline_window_s : float
        Rolling percentile window length in seconds.
    baseline_percentile : float
        Percentile for baseline estimation (default 20th).
    detrend : bool
        Apply linear detrending per finite segment if True.
    s_min_factor : float
        Minimum spike magnitude multiplier relative to noise sigma.
    min_spike_threshold_sigma : float
        Threshold (in sigma units) for declaring events from the spike train.
    impulse_len : int or None
        Truncation length of the AR(1) impulse response in samples. Defaults to
        ~5 tau (in samples) when None.
    """
    arr = np.asarray(trace, dtype=float)
    n = arr.size
    g = float(np.exp(-1.0 / (tau * fs)))
    baseline = np.full(n, np.nan, dtype=float)
    dff = np.full(n, np.nan, dtype=float)
    spikes_full = np.full(n, np.nan, dtype=float)
    recon_full = np.full(n, np.nan, dtype=float)
    all_events: List[Event] = []

    segments = _split_contiguous_segments(arr)
    for start, end in segments:
        seg = arr[start:end]
        if detrend and seg.size > 1:
            x = np.arange(seg.size)
            coeffs = np.polyfit(x, seg, deg=1)
            seg = seg - np.polyval(coeffs, x)
        window_frames = max(1, int(round(baseline_window_s * fs)))
        f0 = _rolling_percentile_baseline(seg, window_frames, baseline_percentile)
        dff_seg = _safe_divide(seg - f0, f0)
        sigma = _estimate_noise_sigma(dff_seg)
        # limit impulse response to ~5 tau by default to keep NNLS fast
        default_impulse = max(10, int(np.ceil(fs * tau * 5)))
        impulse_use = impulse_len if impulse_len is not None else default_impulse
        spikes, recon = _deconvolve_segment(
            dff_seg,
            g,
            sigma,
            impulse_len=impulse_use,
            s_min_factor=s_min_factor,
        )
        event_thresh = (min_spike_threshold_sigma * sigma) if sigma > 0 else 0.0
        events = _extract_events(
            spikes,
            fs=fs,
            start_index=start,
            threshold=event_thresh,
        )
        baseline[start:end] = f0
        dff[start:end] = dff_seg
        spikes_full[start:end] = spikes
        recon_full[start:end] = recon
        all_events.extend(events)

    # Aggregate global noise metric from finite dF/F portion
    sigma_global = _estimate_noise_sigma(dff[np.isfinite(dff)])
    finite_spikes = spikes_full[np.isfinite(spikes_full) & (spikes_full > 0)]
    snr = float(
        np.nan if sigma_global == 0 else (np.mean(finite_spikes) / sigma_global)
    )
    return DetectionResult(
        raw_trace=arr,
        baseline=baseline,
        dff=dff,
        reconstruction=recon_full,
        spikes=spikes_full,
        events=all_events,
        sigma=sigma_global,
        snr=snr,
        g=g,
        fs=fs,
        tau=tau,
    )


def plot_qc_dashboard(
    result: DetectionResult,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = False,
) -> Dict[str, object]:
    """
    Generate QC dashboard with raw vs reconstruction, spikes, and residuals.
    """
    t = np.arange(result.raw_trace.size) / result.fs
    resids = result.dff - result.reconstruction

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ax0, ax1, ax2 = axes
    ax0.plot(t, result.raw_trace, color="k", lw=1, label="Raw")
    ax0.plot(t, result.reconstruction, color="r", lw=1.2, label="Reconstruction")
    ax0.set_ylabel("Fluorescence")
    ax0.legend(loc="upper right")
    ax0.set_title(title or "Glutamate Event Detection QC")

    markerline, stemlines, baseline_line = ax1.stem(
        t,
        result.spikes,
        linefmt="C0-",
        markerfmt=" ",
        basefmt=" ",
    )
    _ = (markerline, stemlines, baseline_line)
    ax1.set_ylabel("Spikes (a.u.)")

    ax2.plot(t, resids, color="0.2", lw=0.8)
    ax2.axhline(0, color="r", lw=0.8, ls="--")
    ax2.set_ylabel("Residual (dF/F)")
    ax2.set_xlabel("Time (s)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return {"fig": fig, "axes": axes}


def run_detection_on_file(
    file_path: str,
    *,
    fs: float = 200.0,
    tau: float = 0.026,
    save_qc_path: Optional[str] = None,
    detrend: bool = False,
) -> DetectionResult:
    """
    Convenience entry point to load a .npy trace and run the pipeline.
    """
    data = np.load(file_path)
    if data.ndim != 1:
        raise ValueError(f"Expected 1D trace from {file_path}, got shape {data.shape}")
    result = detect_glutamate_events(
        data,
        fs=fs,
        tau=tau,
        detrend=detrend,
    )
    if save_qc_path:
        plot_qc_dashboard(
            result,
            title=f"QC: {file_path}",
            save_path=save_qc_path,
            show=False,
        )
    return result


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Glutamate release event detection (FOOPSI-style)"
    )
    parser.add_argument("npy_path", type=str, help="Path to 1D .npy trace")
    parser.add_argument(
        "--fs", type=float, default=200.0, help="Sampling rate in Hz (default 200)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.026,
        help="Decay constant in seconds (default 0.026 s)",
    )
    parser.add_argument(
        "--detrend",
        action="store_true",
        help="Apply per-segment linear detrending before baseline",
    )
    parser.add_argument(
        "--save-qc",
        type=str,
        default=None,
        help="Optional path to save QC dashboard (PNG)",
    )
    args = parser.parse_args()
    res = run_detection_on_file(
        args.npy_path,
        fs=args.fs,
        tau=args.tau,
        save_qc_path=args.save_qc,
        detrend=args.detrend,
    )
    print(
        f"Detected {len(res.events)} events; noise sigma={res.sigma:.4f}; SNR={res.snr:.2f}"
    )
