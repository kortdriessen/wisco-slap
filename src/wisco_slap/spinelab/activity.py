"""Wright-style activity detection for spine-size measurement.

Detects "active" frames in a spine's fluorescence trace so they can be
excluded from structural (inactive-frame) mean projections, following the
event-detection recipe in Wright et al. 2025 (science.ads4706) adapted for
iGluSnFR4f at the registered-TIFF frame rate. This is measurement-side code:
the labelling app never needs it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde


@dataclass(frozen=True)
class ActivityDetection:
    """Intermediate and final products of Wright-style event exclusion."""

    active: np.ndarray
    dff: np.ndarray
    detection_trace: np.ndarray
    baseline: float
    noise: float
    high_threshold: float
    low_threshold: float


def kde_mode(values: np.ndarray) -> float:
    """Estimate a trace baseline as the mode of its finite-value KDE."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    if finite.size < 8 or np.ptp(finite) <= np.finfo(float).eps:
        return float(np.median(finite))
    low, high = np.quantile(finite, [0.005, 0.995])
    if not high > low:
        return float(np.median(finite))
    try:
        density = gaussian_kde(finite)
        grid = np.linspace(low, high, 512)
        return float(grid[int(np.argmax(density(grid)))])
    except (np.linalg.LinAlgError, ValueError):
        return float(np.median(finite))


def _odd_window(samples: float, minimum: int) -> int:
    window = max(minimum, int(round(samples)))
    if window % 2 == 0:
        window += 1
    return window


def detect_active_frames(
    trace: np.ndarray,
    *,
    sample_rate_hz: float = 80.0,
    valid_mask: np.ndarray | None = None,
    smoothing_seconds: float = 0.5,
    savgol_seconds: float = 1.0,
    savgol_order: int = 4,
    noise_multiplier: float = 2.0,
    minimum_amplitude: float = 0.2,
) -> ActivityDetection:
    """Detect active spine frames using the published Wright-style recipe.

    The detection trace is the Savitzky-Golay smoothing of the dF/F0 trace,
    applied directly per the paper's event-detection wording (the 0.5-second
    boxcar ``smoothing_seconds`` is only a fallback when the trace is too
    short to form a Savitzky-Golay window).  Invalid frames are filled by
    linear interpolation before filtering so gaps do not drag the smoothed
    trace toward baseline, but they never become active themselves; callers
    should exclude them from projections separately.  Event regions crossing
    the high threshold are expanded in both directions until the unsmoothed
    dF/F reaches one noise SD.
    """

    values = np.asarray(trace, dtype=float)
    if values.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    valid = np.isfinite(values)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    baseline = kde_mode(values[valid])
    if not np.isfinite(baseline) or abs(baseline) <= np.finfo(float).eps:
        nan_trace = np.full(values.shape, np.nan)
        return ActivityDetection(
            active=np.zeros(values.shape, dtype=bool),
            dff=nan_trace,
            detection_trace=nan_trace.copy(),
            baseline=float(baseline),
            noise=math.nan,
            high_threshold=math.nan,
            low_threshold=math.nan,
        )

    filled = values.copy()
    indices = np.arange(len(values))
    if valid.any():
        filled[~valid] = np.interp(indices[~valid], indices[valid], values[valid])
    dff_filled = (filled - baseline) / baseline
    dff = dff_filled.copy()
    dff[~valid] = np.nan
    sg_window = _odd_window(savgol_seconds * sample_rate_hz, savgol_order + 2)
    if sg_window > len(values):
        sg_window = len(values) if len(values) % 2 == 1 else len(values) - 1
    if sg_window >= savgol_order + 2:
        detection_trace = savgol_filter(
            dff_filled, window_length=sg_window, polyorder=savgol_order, mode="interp"
        )
    else:
        smooth_samples = max(1, int(round(smoothing_seconds * sample_rate_hz)))
        detection_trace = uniform_filter1d(
            dff_filled, size=smooth_samples, mode="nearest"
        )
    detection_trace = np.asarray(detection_trace, dtype=float)
    detection_trace[~valid] = np.nan

    negative = dff[valid & (dff < 0)]
    if negative.size >= 2:
        mirrored = np.concatenate([negative, -negative])
        noise = float(np.std(mirrored, ddof=1))
    else:
        centered = dff[valid] - np.nanmedian(dff[valid])
        noise = float(1.4826 * np.nanmedian(np.abs(centered)))
    low_threshold = noise
    high_threshold = max(noise_multiplier * noise, minimum_amplitude)
    seeds = valid & (detection_trace >= high_threshold)
    active = np.zeros(values.shape, dtype=bool)
    seed_indices = np.flatnonzero(seeds)
    if seed_indices.size:
        breaks = np.flatnonzero(np.diff(seed_indices) > 1) + 1
        for group in np.split(seed_indices, breaks):
            left = int(group[0])
            right = int(group[-1])
            while left > 0 and valid[left - 1] and dff[left - 1] > low_threshold:
                left -= 1
            while (
                right + 1 < len(values)
                and valid[right + 1]
                and dff[right + 1] > low_threshold
            ):
                right += 1
            active[left : right + 1] = True
    active &= valid
    return ActivityDetection(
        active=active,
        dff=dff,
        detection_trace=detection_trace,
        baseline=float(baseline),
        noise=float(noise),
        high_threshold=float(high_threshold),
        low_threshold=float(low_threshold),
    )
