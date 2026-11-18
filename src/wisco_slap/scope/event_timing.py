"""Event timing kernels with explicit observation support and grouped windows.

Times and window widths use seconds. Observation intervals are half-open; count
windows include both endpoints (within numerical tolerance). Jitter preserves
an event's sampling-grid phase and never crosses its assigned support interval.
"""
from __future__ import annotations

import numpy as np


def contiguous_support(time: np.ndarray, valid: np.ndarray, fs: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return sample-index [start, end) runs, splitting invalid samples and time gaps."""
    time, valid = np.asarray(time, float), np.asarray(valid, bool)
    if time.ndim != 1 or valid.shape != time.shape:
        raise ValueError("time and valid must be equal-length one-dimensional arrays")
    if not time.size:
        return np.array([], dtype=int), np.array([], dtype=int)
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("time must be finite and strictly increasing")
    dt = 1 / fs if fs is not None else float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("A positive sampling rate is required")
    breaks = np.r_[True, np.diff(time) > 1.5 * dt]
    starts = np.flatnonzero(valid & (np.r_[True, ~valid[:-1]] | breaks))
    ends = np.flatnonzero(valid & (np.r_[~valid[1:], True] | np.r_[breaks[1:], True])) + 1
    return starts, ends


def reflected_jitter(t: np.ndarray, lo: np.ndarray, hi: np.ndarray, fs: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Reflect independent uniform integer-frame displacements inside [lo, hi).

    This is a local timing surrogate, not an exact conditional Poisson null;
    reflection at endpoints is not claimed to yield uniform stationary occupancy.
    A one-frame interval necessarily leaves its event unchanged.
    """
    t, lo, hi = (np.asarray(x, float) for x in (t, lo, hi))
    if t.shape != lo.shape or t.shape != hi.shape or fs <= 0 or jitter < 0:
        raise ValueError("Equal array shapes, positive fs and nonnegative jitter required")
    if np.any(~np.isfinite(t + lo + hi)) or np.any(t < lo - 1e-8) or np.any(t >= hi):
        raise ValueError("Every event must lie inside its finite half-open support")
    phase = t - np.round(t * fs) / fs
    left = np.ceil((lo - phase) * fs - 1e-5)
    right = np.ceil((hi - phase) * fs - 1e-5) - 1
    width = right - left
    if np.any(width < 0):
        raise ValueError("Support contains no sampling frame")
    base = np.round((t - phase) * fs) - left
    radius = int(round(jitter * fs))
    trial = base + rng.integers(-radius, radius + 1, t.size)
    folded = np.mod(trial, np.maximum(2 * width, 1))
    result = (left + np.where(width == 0, 0, np.where(folded <= width, folded, 2 * width - folded))) / fs + phase
    if np.any(result < lo - 1e-8) or np.any(result >= hi + 1e-8):
        raise RuntimeError("Jitter escaped support")
    return result


def count_partners(tq: np.ndarray, bq: np.ndarray, tr: np.ndarray, br: np.ndarray, window: float, *, atol: float = 1e-6) -> np.ndarray:
    """Count reference events within ±window in the same bout; no self-removal."""
    tq, tr = np.asarray(tq, float), np.asarray(tr, float)
    bq, br = np.asarray(bq), np.asarray(br)
    if tq.shape != bq.shape or tr.shape != br.shape or window < 0 or atol < 0:
        raise ValueError("Mismatched arrays or negative window/tolerance")
    out = np.zeros(tq.size, dtype=np.int64)
    if not tq.size or not tr.size:
        return out
    # Sort references once within integer-coded groups, then search only that
    # group's float time array. This preserves exact boundary arithmetic without
    # either arbitrary duration sentinels or costly structured-array comparisons.
    _, codes = np.unique(np.concatenate([bq, br]), return_inverse=True)
    query_groups, reference_groups = codes[:tq.size], codes[tq.size:]
    reference_order = np.lexsort((tr, reference_groups))
    query_order = np.argsort(query_groups, kind='stable')
    rs, qs = reference_groups[reference_order], query_groups[query_order]
    reference_bounds = np.r_[0, np.flatnonzero(np.diff(rs)) + 1, rs.size]
    query_bounds = np.r_[0, np.flatnonzero(np.diff(qs)) + 1, qs.size]
    reference_labels = rs[reference_bounds[:-1]]
    lookup = np.searchsorted(reference_labels, qs[query_bounds[:-1]])
    reference_times = tr[reference_order]
    for query_group, reference_group in enumerate(lookup):
        if reference_group >= reference_labels.size or reference_labels[reference_group] != qs[query_bounds[query_group]]:
            continue
        positions = query_order[query_bounds[query_group]:query_bounds[query_group + 1]]
        query_times = tq[positions]
        refs = reference_times[reference_bounds[reference_group]:reference_bounds[reference_group + 1]]
        out[positions] = (np.searchsorted(refs, query_times + window + atol, side='right')
                          - np.searchsorted(refs, query_times - window - atol, side='left'))
    return out



def count_other_events(t: np.ndarray, bout: np.ndarray, synapse: np.ndarray, window: float, *, atol: float = 1e-6) -> np.ndarray:
    """Count events on other synapses in the same bout, including boundary frames."""
    t, bout, synapse = np.asarray(t, float), np.asarray(bout), np.asarray(synapse)
    if t.shape != bout.shape or t.shape != synapse.shape:
        raise ValueError("Time, bout and synapse arrays must have identical shapes")
    _, bc = np.unique(bout, return_inverse=True)
    _, sc = np.unique(synapse, return_inverse=True)
    groups = bc * max(1, int(sc.max()) + 1) + sc if t.size else bc
    return count_partners(t, bout, t, bout, window, atol=atol) - count_partners(t, groups, t, groups, window, atol=atol)
