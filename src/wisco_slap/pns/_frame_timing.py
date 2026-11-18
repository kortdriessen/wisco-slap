"""Trial-aware clocks for registered TIFFs and shaft-calcium traces."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SHAFT_TIME_AXIS_VERSION = 2
SHAFT_TIME_CONVENTION = (
    "scopex trial boundaries, including dropped-trial placeholders; "
    "TIFF samples spaced within each analyzed trial; multi-epoch offsets applied"
)


def _counts(values: Sequence[int], name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not len(raw):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(raw)) or np.any(raw < 0) or np.any(raw != np.floor(raw)):
        raise ValueError(f"{name} must contain nonnegative integer counts")
    return raw.astype(np.int64)


def estimate_frame_rate(
    analyze_hz: float,
    frame_counts: Sequence[int],
    analyzed_counts: Sequence[int],
) -> float:
    """Estimate TIFF rate using the same surviving trials in both populations.

    A dropped ExSum trial has zero analyzed samples, even when its TIFF exists.
    Exclude that trial's TIFF frames as well. Prefix/smoke-test builds pass only
    their actual trial population; never extrapolate it to the full acquisition.
    """
    frames = _counts(frame_counts, "frame_counts")
    analyzed = _counts(analyzed_counts, "analyzed_counts")
    if frames.shape != analyzed.shape:
        raise ValueError("TIFF and analyzed counts must describe the same trials")
    if not np.isfinite(analyze_hz) or analyze_hz <= 0:
        raise ValueError("analyze_hz must be positive and finite")
    usable = (analyzed > 0) & (frames > 0)
    if not usable.any():
        raise ValueError("Cannot infer TIFF rate without a surviving analyzed trial")
    return float(analyze_hz * frames[usable].sum() / analyzed[usable].sum())


def fill_dropped_trial_counts(
    analyzed_counts: Sequence[int], epochs: Sequence[int]
) -> np.ndarray:
    """Use Scopex's first-clean-trial template within each epoch for missing trials.

    Real trial lengths are retained, including partial trials. An epoch with no
    template is ambiguous and must fail, rather than borrowing another epoch's
    duration or silently compressing time.
    """
    counts = _counts(analyzed_counts, "analyzed_counts").copy()
    epoch = _counts(epochs, "epochs")
    if counts.shape != epoch.shape or np.any(epoch < 1):
        raise ValueError("epochs must be positive, with one value per trial")
    for e in np.unique(epoch):
        own = epoch == e
        valid = np.flatnonzero(own & (counts > 0))
        if not len(valid):
            raise ValueError(f"Epoch {e} has no analyzed trial to define placeholders")
        counts[own & (counts == 0)] = counts[valid[0]]
    return counts


def scopex_frame_times(
    frame_counts: Sequence[int],
    analyzed_counts: Sequence[int],
    analyze_hz: float,
    epochs: Sequence[int],
    epoch_offsets_s: dict[int, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Place TIFF frames inside the exact Scopex trial intervals.

    A global TIFF rate cannot represent independently rounded/partial TIFF
    trials without accumulated boundary drift. Each trial therefore starts at
    its analyzed-sample boundary and retains its complete duration, even if all
    its samples are discarded. Returned rate is the acquisition-average rate;
    actual timestamps are authoritative when frame counts vary.
    """
    frames = _counts(frame_counts, "frame_counts")
    counts = fill_dropped_trial_counts(analyzed_counts, epochs)
    epoch = np.asarray(epochs, dtype=int)
    if frames.shape != counts.shape:
        raise ValueError("TIFF and analyzed counts must describe the same trials")
    if not frames.sum() or not np.isfinite(analyze_hz) or analyze_hz <= 0:
        raise ValueError(
            "At least one TIFF frame and a positive analyze_hz are required"
        )
    if epoch_offsets_s is None:
        if np.unique(epoch).tolist() != [1]:
            raise ValueError("Multi-epoch clocks require explicit epoch offsets")
        epoch_offsets_s = {1: 0.0}
    counters: dict[int, int] = {}
    pieces = []
    for nf, ns, e in zip(frames, counts, epoch, strict=True):
        if int(e) not in epoch_offsets_s:
            raise ValueError(f"Missing offset for epoch {e}")
        offset = float(epoch_offsets_s[int(e)])
        if not np.isfinite(offset):
            raise ValueError(f"Nonfinite offset for epoch {e}")
        start = offset + counters.get(int(e), 0) / analyze_hz
        if nf:
            pieces.append(start + np.arange(nf, dtype=float) * (ns / analyze_hz / nf))
        counters[int(e)] = counters.get(int(e), 0) + int(ns)
    times = np.concatenate(pieces)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Trial/epoch offsets produce a nonmonotonic frame clock")
    average_hz = float(frames.sum() * analyze_hz / counts.sum())
    return times, average_hz
