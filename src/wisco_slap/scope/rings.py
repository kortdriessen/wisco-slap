"""Detector for phasic, coordinated glutamate input packets ("rings").

A ring is a brief window in which many recorded synapses of one group (the
whole population, a dendrite class, or a single dendrite) cross their activity
threshold together, with fresh arrivals at the leading edge. The detector only
looks at the *input* signal: no EEG, sleep state, soma calcium, or prior ON/OFF
logic enters detection. Thresholds are morphological review settings, not
calibrated p-values, and none of the reported integrals is a glutamate
concentration or synaptic efficacy.

Typical use::

    import wisco_slap as wis

    dn = wis.get.fetch_data_pipe("synapse", trace="denoised", channel=1, ...)
    ls = wis.get.fetch_data_pipe("synapse", trace="ls", channel=1, ...)
    sig = wis.scope.pursuit.denoised_sigma_units(dn, ls)
    rings = wis.scope.rings.detect_rings(sig, level="dend-ID")
    rings, diag = wis.scope.rings.detect_rings(sig, return_diagnostics=True)

The detector was developed and validated in
``slap_mi_2_sleep/ring_detector/`` (see its README for the validation
report and parameter guidance); that folder now imports from this module.

Defaults (since 2026-09-11) are the EEG-associated ``previous_correlation``
review presets selected by the four-acquisition sweep in
``slap_mi_2_sleep/ring_detector/eeg_sweep/`` (``review_presets.json``): a
population preset for ``level="syn_id"`` and a per-group preset for every
other level (fitted on ``dend-ID`` groups). Only ``activity_threshold`` and
``min_prominence`` differ between the two; see :data:`LEVEL_DEFAULTS` and
:func:`default_params`. They are review settings chosen for association with
EEG slow power, not validated physiological thresholds.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
import xarray as xr
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths

__all__ = [
    "LEVEL_DEFAULTS",
    "RING_SCHEMA",
    "RingParams",
    "default_params",
    "detect_rings",
]


@dataclass(frozen=True)
class RingParams:
    """Detection settings. All times are seconds; fractions refer to the group.

    ``activity_threshold`` is in input units with ``normalization="none"``.
    With ``"robust"`` it is in units of a per-ROI lower-distribution spread;
    that spread is neither measurement noise nor a calibrated SNR.

    The field defaults are the whole-population (``level="syn_id"``) review
    preset. ``activity_threshold`` and ``min_prominence`` differ for grouped
    detection (any other level); use :func:`default_params` to obtain the
    preset for a given level instead of relying on the field defaults there.

    Attributes
    ----------
    activity_threshold
        Per-synapse amplitude required to count as active.
    min_active_synapses
        Minimum simultaneously active *and* newly recruited distinct synapses.
    min_peak_fraction
        Minimum instantaneous fraction of the group above threshold.
    min_prominence
        Minimum envelope-peak prominence, in active-fraction units.
    min_excess_synapses
        Additional prominence floor expressed as ``min_excess_synapses / n``.
    smooth_s
        Centered boxcar width applied to the active fraction (rounded to an
        odd number of samples).
    min_duration_s, max_duration_s
        Accepted packet duration range.
    min_separation_s
        Minimum spacing between envelope peaks.
    boundary_fraction
        Height above the prominence baseline at which packet edges are placed.
    prominence_window_s
        Local context window for peak prominence.
    min_onset_fraction
        Required fraction of distinct recent arrivals (also bounded below by
        ``min_active_synapses``).
    onset_window_s
        Width of the causal distinct-arrival window.
    rearm_fraction
        Per-synapse rearming level relative to ``activity_threshold``; a new
        arrival requires an observed sample below this since the last one.
    normalization
        ``"none"`` (thresholds in input units) or ``"robust"`` (per-ROI q20
        centre and ``(q50 - q20) / 0.8416`` scale, fitted on this recording).
    """

    activity_threshold: float = 3.0
    min_active_synapses: int = 3
    min_peak_fraction: float = 0.15
    min_prominence: float = 0.15
    min_excess_synapses: float = 2.0
    smooth_s: float = 0.20
    min_duration_s: float = 0.30
    max_duration_s: float = 2.0
    min_separation_s: float = 0.06
    boundary_fraction: float = 0.20
    prominence_window_s: float = 2.0
    min_onset_fraction: float = 0.05
    onset_window_s: float = 0.06
    rearm_fraction: float = 0.5
    normalization: str = "none"

    def __post_init__(self) -> None:
        for k, v in asdict(self).items():
            if k != "normalization" and not np.isfinite(v):
                raise ValueError(f"{k} must be finite")
        if self.activity_threshold <= 0:
            raise ValueError("activity_threshold must be positive")
        if (
            isinstance(self.min_active_synapses, bool)
            or int(self.min_active_synapses) != self.min_active_synapses
            or self.min_active_synapses < 1
        ):
            raise ValueError("min_active_synapses must be a positive integer")
        for k in ("min_peak_fraction", "min_onset_fraction"):
            if not 0 <= getattr(self, k) <= 1:
                raise ValueError(f"{k} must be between 0 and 1")
        if not 0 < self.min_prominence <= 1:
            raise ValueError("min_prominence must be in (0, 1]")
        if self.min_excess_synapses < 0:
            raise ValueError("min_excess_synapses cannot be negative")
        for k in ("boundary_fraction", "rearm_fraction"):
            if not 0 < getattr(self, k) < 1:
                raise ValueError(f"{k} must be in (0, 1)")
        if self.smooth_s < 0 or self.min_separation_s < 0:
            raise ValueError("smooth_s and min_separation_s cannot be negative")
        if not 0 < self.min_duration_s <= self.max_duration_s:
            raise ValueError("require 0 < min_duration_s <= max_duration_s")
        if self.prominence_window_s <= 0 or self.onset_window_s <= 0:
            raise ValueError("prominence_window_s and onset_window_s must be positive")
        if self.normalization not in ("none", "robust"):
            raise ValueError("normalization must be 'none' or 'robust'")


LEVEL_DEFAULTS: dict[str, dict[str, float]] = {
    "syn_id": {"activity_threshold": 3.0, "min_prominence": 0.15},
    "dend-ID": {"activity_threshold": 2.5, "min_prominence": 0.20},
}
"""Level-dependent default settings (the ``previous_correlation`` review presets).

``"syn_id"`` is the whole-population preset; ``"dend-ID"`` is the per-group
preset, which :func:`default_params` applies to every level other than
``"syn_id"`` (it was fitted on ``dend-ID`` groups; other groupings such as
``dend_type`` have not been swept separately). All remaining settings are
shared and live as the :class:`RingParams` field defaults.
"""


def default_params(level: str = "syn_id", **overrides: Any) -> RingParams:
    """Return the default :class:`RingParams` for ``level``, with overrides applied.

    Parameters
    ----------
    level
        ``"syn_id"`` selects the population preset; any other value selects the
        per-group preset (see :data:`LEVEL_DEFAULTS`).
    **overrides
        Any :class:`RingParams` field. Explicit values always win over the
        level-dependent defaults.

    Returns
    -------
    RingParams
        Validated settings, exactly what :func:`detect_rings` uses when the
        corresponding arguments are left at their defaults.
    """
    preset = LEVEL_DEFAULTS["syn_id" if level == "syn_id" else "dend-ID"]
    return RingParams(**{**preset, **overrides})


RING_SCHEMA: dict[str, pl.DataType] = {
    "ring_id": pl.Int64,
    "level": pl.String,
    "group": pl.String,
    "start_s": pl.Float64,
    "onset_s": pl.Float64,
    "end_s": pl.Float64,
    "duration_s": pl.Float64,
    "peak_s": pl.Float64,
    "start_index": pl.Int64,
    "end_index": pl.Int64,
    "n_synapses": pl.Int64,
    "n_participating": pl.Int64,
    "fraction_participating": pl.Float64,
    "participant_ids": pl.List(pl.String),
    "n_newly_active": pl.Int64,
    "newly_active_ids": pl.List(pl.String),
    "peak_active_count": pl.Int64,
    "peak_active_fraction": pl.Float64,
    "mean_active_fraction": pl.Float64,
    "peak_new_fraction": pl.Float64,
    "peak_envelope": pl.Float64,
    "prominence": pl.Float64,
    "boundary_level": pl.Float64,
    "rise_s": pl.Float64,
    "decay_s": pl.Float64,
    "signal_sum": pl.Float64,
    "signal_integral": pl.Float64,
    "positive_signal_integral": pl.Float64,
    "excess_integral": pl.Float64,
    "peak_signal_sum": pl.Float64,
    "participant_signal_fraction": pl.Float64,
    "largest_synapse_signal_fraction": pl.Float64,
    "onset_spread_s": pl.Float64,
    "left_censored": pl.Boolean,
    "right_censored": pl.Boolean,
    "valid_segment": pl.Int64,
    "sample_interval_s": pl.Float64,
    "signal_units": pl.String,
    "normalization": pl.String,
    "activity_threshold": pl.Float64,
}
"""Polars schema of the table returned by :func:`detect_rings` (also when empty)."""


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` index runs of contiguous ``True`` samples."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    # Padded Boolean transitions always supply paired start/end edges.
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _groups(a: xr.DataArray, level: str) -> list[tuple[str, np.ndarray]]:
    """Return ``(group_name, syn_id positional indices)`` for each group.

    ``level="syn_id"`` yields a single ``"population"`` group holding every
    synapse. Any other level must be a coordinate on ``syn_id``; synapses with
    a missing annotation (None / NaN / empty string) are excluded with a warning.
    """
    if level == "syn_id":
        return [("population", np.arange(a.sizes["syn_id"]))]
    if level not in a.coords or a[level].dims != ("syn_id",):
        raise ValueError(
            f"{level!r} must be a coordinate on syn_id; available: "
            f"{[k for k in a.coords if a[k].dims == ('syn_id',)]}"
        )
    result: dict[str, list[int]] = {}
    missing = 0
    for i, value in enumerate(a[level].values):
        if (
            value is None
            or (isinstance(value, (float, np.floating)) and np.isnan(value))
            or str(value).strip().lower() in ("", "nan", "none")
        ):
            missing += 1
            continue
        result.setdefault(str(value), []).append(i)
    if missing:
        warnings.warn(
            f"Excluded {missing} synapses without {level!r} annotation", stacklevel=3
        )
    return [(key, np.array(indices)) for key, indices in result.items()]


def _activity(
    y: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    rearm_fraction: float,
    onset_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-ROI occupancy, hysteretically rearmed arrivals, and recent arrivals.

    Parameters
    ----------
    y
        ``(n_roi, n_time)`` signal already in threshold units.
    valid
        ``(n_time,)`` Boolean mask of observed samples.
    threshold, rearm_fraction
        A ROI becomes active at ``y >= threshold``; a *new* arrival is only
        counted if a sample ``<= threshold * rearm_fraction`` was observed
        since the previous excursion.
    onset_width
        Causal window (samples) over which distinct arrivals are counted.

    Returns
    -------
    active, onsets
        ``(n_roi, n_time)`` Boolean arrays.
    recent_count
        ``(n_time,)`` number of ROIs with an arrival in the trailing window.

    No arrival is invented at a finite-segment boundary: a low sample must
    actually have been observed since the previous above-threshold excursion.
    """
    active = (y >= threshold) & valid[None, :]
    onsets = np.zeros_like(active)
    recent_count = np.zeros(y.shape[1], dtype=np.int32)
    for i in range(len(y)):
        low = (y[i] <= threshold * rearm_fraction) & valid
        arm_epoch = np.cumsum(low, dtype=np.int64)
        rising = np.flatnonzero(active[i] & ~np.r_[False, active[i, :-1]])
        if rising.size:
            # Every gap disarms the detector until a real low sample follows it.
            last_gap = np.maximum.accumulate(
                np.where(~valid, np.arange(len(valid)), -1)
            )
            last_low = np.maximum.accumulate(np.where(low, np.arange(len(valid)), -1))
            epochs = arm_epoch[rising]
            armed = (epochs != np.r_[0, epochs[:-1]]) & (
                last_low[rising] > last_gap[rising]
            )
            onsets[i, rising[armed]] = True
        cs = np.r_[0, np.cumsum(onsets[i], dtype=np.int32)]
        count = cs[1:] - cs[np.maximum(0, np.arange(len(valid)) + 1 - onset_width)]
        recent_count += count > 0
    return active, onsets, recent_count


def _new_support(onsets: np.ndarray, lo: int, stop: int, width: int) -> int:
    """Maximum distinct arrivals in a causal window restricted to ``[lo, stop)``."""
    if stop <= lo:
        return 0
    arrivals = onsets[:, lo:stop]
    cs = np.pad(np.cumsum(arrivals, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
    counts = cs[:, 1:] - cs[:, np.maximum(0, np.arange(stop - lo) + 1 - width)]
    return int((counts > 0).sum(axis=0).max(initial=0))


def _empty_table() -> pl.DataFrame:
    return pl.DataFrame(schema=RING_SCHEMA)


def detect_rings(
    scopex_array: xr.DataArray,
    level: str = "syn_id",
    *,
    activity_threshold: float | None = None,
    min_active_synapses: int = 3,
    min_peak_fraction: float = 0.15,
    min_prominence: float | None = None,
    min_excess_synapses: float = 2.0,
    smooth_s: float = 0.20,
    min_duration_s: float = 0.30,
    max_duration_s: float = 2.0,
    min_separation_s: float = 0.06,
    boundary_fraction: float = 0.20,
    prominence_window_s: float = 2.0,
    min_onset_fraction: float = 0.05,
    onset_window_s: float = 0.06,
    rearm_fraction: float = 0.5,
    normalization: str = "none",
    return_diagnostics: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, xr.Dataset]]:
    """Detect phasic coordinated input packets independently in each group.

    Algorithm: threshold each ROI; smooth the fraction active; find locally
    prominent peaks; require both simultaneous support and newly recruited
    distinct ROIs; bound each packet at a fraction of its prominence; divide
    overlapping packets at the intervening trough. No prior ON/OFF logic,
    Gaussian assumption, EEG, or postsynaptic response is used.

    Parameters
    ----------
    scopex_array
        Two-dimensional array with exactly the ``syn_id`` and ``time`` (seconds)
        dimensions, in either order. Scalar coordinates such as ``channel`` are
        allowed. The recommended input is
        :func:`wisco_slap.scope.pursuit.denoised_sigma_units` (denoised trace
        in LS-noise units), on which the default thresholds were selected; for
        ``dn_dff`` or raw ``dn`` pass ``normalization="robust"`` or choose a
        threshold in input units. Detection never spans missing samples, an explicit
        ``is_valid=False`` coordinate (on ``time`` or ``syn_id x time``), or
        time jumps greater than 1.5 sample intervals. All ROIs in a group must
        be observed at a sample for it to count; the denominator never changes
        within a group.
    level
        ``"syn_id"`` treats the whole supplied population as one group named
        ``"population"``. Any other value must name a coordinate on ``syn_id``
        (e.g. ``"dend-ID"``, ``"dend_type"``); one detection pass runs per
        distinct value. Groups smaller than ``min_active_synapses`` yield no
        detections. The level also selects the default ``activity_threshold``
        and ``min_prominence`` (see below).
    activity_threshold, min_prominence
        ``None`` (default) resolves to the level's review preset via
        :func:`default_params`: 3.0 / 0.15 for ``"syn_id"``, 2.5 / 0.20 for
        any other level. Pass a value to override.
    min_active_synapses, ..., normalization
        See :class:`RingParams` for the meaning and validation of each setting;
        the signature defaults equal the :class:`RingParams` field defaults.
    return_diagnostics
        Also return ``{group: xr.Dataset}`` with the per-sample active
        fraction, smoothed envelope, recent-recruitment fraction, valid mask,
        and per-ROI input thresholds, plus the parameters and valid exposure
        in ``attrs``. Descriptive only, not calibrated probabilities.

    Returns
    -------
    pl.DataFrame
        One row per ring with schema :data:`RING_SCHEMA` (typed also when
        empty), sorted by ``start_s`` then ``group``, with ``ring_id`` numbered
        from 0. ``start_s``/``onset_s`` are the first included sample and
        ``end_s`` is the exclusive right edge (same half-open convention for
        ``start_index``/``end_index``). Integral columns are input-units x
        seconds; ``signal_sum`` is the plain sample sum (sampling-rate
        dependent); ``excess_integral`` integrates only signal above each
        ROI's threshold in original input units. ``participant_ids`` are ROIs
        above threshold at any included sample; ``newly_active_ids`` require a
        rearmed threshold crossing inside the packet. Onset precision is
        limited by sampling, smoothing, and indicator kinetics.
    tuple[pl.DataFrame, dict[str, xr.Dataset]]
        When ``return_diagnostics=True``.

    Raises
    ------
    ValueError
        On an invalid parameter, a non-DataArray or wrongly-dimensioned input,
        a non-finite / non-increasing / irregular (>5 % jitter) time axis,
        duplicate ``syn_id`` values, a missing ``level`` coordinate, or an
        ``is_valid`` coordinate on unsupported dimensions.
    """
    level_dependent = {
        k: v
        for k, v in {
            "activity_threshold": activity_threshold,
            "min_prominence": min_prominence,
        }.items()
        if v is not None
    }
    params = default_params(
        level,
        **level_dependent,
        min_active_synapses=min_active_synapses,
        min_peak_fraction=min_peak_fraction,
        min_excess_synapses=min_excess_synapses,
        smooth_s=smooth_s,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        min_separation_s=min_separation_s,
        boundary_fraction=boundary_fraction,
        prominence_window_s=prominence_window_s,
        min_onset_fraction=min_onset_fraction,
        onset_window_s=onset_window_s,
        rearm_fraction=rearm_fraction,
        normalization=normalization,
    )
    activity_threshold = params.activity_threshold
    min_prominence = params.min_prominence
    if not isinstance(scopex_array, xr.DataArray) or set(scopex_array.dims) != {
        "syn_id",
        "time",
    }:
        raise ValueError("scopex_array must have exactly syn_id and time dimensions")
    a = scopex_array.transpose("syn_id", "time")
    t = np.asarray(a.time.values, dtype=np.float64)
    if t.size < 3 or not a.sizes["syn_id"]:
        empty = _empty_table()
        return (empty, {}) if return_diagnostics else empty
    d = np.diff(t)
    if not np.isfinite(t).all() or np.any(d <= 0):
        raise ValueError("time must contain finite, strictly increasing seconds")
    dt = float(np.median(d))
    ordinary = d <= 1.5 * dt
    if np.any(np.abs(d[ordinary] / dt - 1) > 0.05):
        raise ValueError(
            "time must be regularly sampled within segments (5% tolerance)"
        )
    jumps = np.flatnonzero(~ordinary) + 1
    source = np.asarray(a.values)
    ids = np.asarray(a.syn_id.values).astype(str)
    if np.unique(ids).size != ids.size:
        raise ValueError("syn_id values must be unique")
    valid_coord = a.coords.get("is_valid")
    if valid_coord is not None and valid_coord.dims not in [
        ("time",),
        ("syn_id", "time"),
        ("time", "syn_id"),
    ]:
        raise ValueError("is_valid must be a time or syn_id-by-time coordinate")
    smooth_width = max(1, int(round(smooth_s / dt)))
    smooth_width += smooth_width % 2 == 0  # centered odd window
    onset_width = max(1, int(round(onset_window_s / dt)))
    peak_wlen = max(3, int(round(prominence_window_s / dt)) | 1)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, xr.Dataset] = {}
    for group, ix in _groups(a, level):
        raw = source[ix]
        valid = np.all(np.isfinite(raw), axis=0)
        if valid_coord is not None:
            vv = valid_coord.fillna(False)
            if vv.dims == ("time",):
                valid &= np.asarray(vv, dtype=bool)
            else:
                valid &= np.asarray(vv.transpose("syn_id", "time"), dtype=bool)[ix].all(
                    axis=0
                )
        # Split at jumps without throwing out an observed endpoint sample.
        runs: list[tuple[int, int]] = []
        for start, stop in _runs(valid):
            breaks = jumps[(jumps > start) & (jumps < stop)]
            bounds = np.r_[start, breaks, stop]
            runs.extend(zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True))
        n = len(ix)
        center = np.zeros(n)
        scale = np.ones(n)
        if normalization == "robust" and valid.any():
            q = np.quantile(raw[:, valid], [0.2, 0.5], axis=1)
            center = q[0]
            scale = (q[1] - q[0]) / 0.8416212336
            floor = np.maximum(
                np.max(np.abs(raw[:, valid]), axis=1) * 1e-6, np.finfo(np.float32).eps
            )
            scale = np.maximum(scale, floor)
        thresholds = center + activity_threshold * scale
        y = (
            (raw - center[:, None]) / scale[:, None]
            if normalization == "robust"
            else raw
        )
        frac = np.full(t.size, np.nan, dtype=np.float32)
        envelope = frac.copy()
        recent = frac.copy()
        for segment_id, (start, stop) in enumerate(runs):
            length = stop - start
            if length < 3:
                continue
            yy = y[:, start:stop]
            active, onsets, new_count = _activity(
                yy,
                np.ones(length, dtype=bool),
                activity_threshold,
                rearm_fraction,
                onset_width,
            )
            ff = active.mean(axis=0)
            env = (
                uniform_filter1d(ff, smooth_width, mode="nearest")
                if smooth_width > 1
                else ff
            )
            frac[start:stop] = ff
            envelope[start:stop] = env
            recent[start:stop] = new_count / n
            if n < min_active_synapses:
                continue
            peaks, prop = find_peaks(
                env,
                prominence=max(min_prominence, min_excess_synapses / n),
                wlen=peak_wlen,
                distance=max(1, int(np.ceil(min_separation_s / dt))),
            )
            if not len(peaks):
                continue
            widths = peak_widths(
                env,
                peaks,
                rel_height=1 - boundary_fraction,
                prominence_data=(
                    prop["prominences"],
                    prop["left_bases"],
                    prop["right_bases"],
                ),
            )
            min_support = max(min_active_synapses, int(np.ceil(n * min_peak_fraction)))
            min_new = max(min_active_synapses, int(np.ceil(n * min_onset_fraction)))
            candidates: list[list[int]] = []
            for k, peak in enumerate(peaks):
                lo = int(np.ceil(widths[2][k]))
                hi = int(np.floor(widths[3][k])) + 1
                if hi <= lo:
                    continue
                # Simultaneous and arrival support are separate requirements.
                support = active[:, lo:hi].sum(axis=0)
                if support.max() < min_support:
                    continue
                new_peak = _new_support(onsets, lo, peak + 1, onset_width)
                if new_peak < min_new:
                    continue
                candidates.append([lo, hi, int(peak), k, int(new_peak)])
            # Resolve overlapping prominence widths at a real intervening trough.
            for k in range(1, len(candidates)):
                left, right = candidates[k - 1], candidates[k]
                if left[1] > right[0]:
                    trough = left[2] + int(np.argmin(env[left[2] : right[2] + 1]))
                    cut = trough + 1
                    left[1] = min(left[1], cut)
                    right[0] = max(right[0], cut)
            for lo, hi, peak, k, _ in candidates:
                duration = (hi - lo) * dt
                if not min_duration_s <= duration <= max_duration_s:
                    continue
                # Revalidate support after any split.
                active_slice = active[:, lo:hi]
                counts = active_slice.sum(axis=0)
                if counts.max(initial=0) < min_support:
                    continue
                new_peak = _new_support(onsets, lo, peak + 1, onset_width)
                if new_peak < min_new:
                    continue
                part = active_slice.any(axis=1)
                new_slice = onsets[:, lo:hi]
                newly = new_slice.any(axis=1)
                first = np.flatnonzero(newly)
                first_times = np.array([
                    np.flatnonzero(new_slice[j])[0] * dt for j in first
                ])
                values = raw[:, start + lo : start + hi].astype(np.float64)
                positive = np.maximum(values, 0)
                by_syn = positive.sum(axis=1) * dt
                total_positive = by_syn.sum()
                left_edge = float(t[start + lo])
                right_edge = float(t[start + hi - 1] + dt)
                rows.append(
                    dict(
                        ring_id=0,
                        level=level,
                        group=group,
                        start_s=left_edge,
                        onset_s=left_edge,
                        end_s=right_edge,
                        duration_s=right_edge - left_edge,
                        peak_s=float(t[start + peak]),
                        start_index=int(start + lo),
                        end_index=int(start + hi),
                        n_synapses=n,
                        n_participating=int(part.sum()),
                        fraction_participating=float(part.mean()),
                        participant_ids=ids[ix][part].tolist(),
                        n_newly_active=int(newly.sum()),
                        newly_active_ids=ids[ix][newly].tolist(),
                        peak_active_count=int(counts.max()),
                        peak_active_fraction=float(counts.max() / n),
                        mean_active_fraction=float(counts.mean() / n),
                        peak_new_fraction=float(new_peak / n),
                        peak_envelope=float(env[peak]),
                        prominence=float(prop["prominences"][k]),
                        boundary_level=float(widths[1][k]),
                        rise_s=float(t[start + peak] - left_edge),
                        decay_s=float(right_edge - t[start + peak]),
                        signal_sum=float(values.sum()),
                        signal_integral=float(values.sum() * dt),
                        positive_signal_integral=float(total_positive),
                        excess_integral=float(
                            np.maximum(values - thresholds[:, None], 0).sum() * dt
                        ),
                        peak_signal_sum=float(values.sum(axis=0).max()),
                        participant_signal_fraction=float(
                            by_syn[part].sum() / total_positive
                        )
                        if total_positive
                        else 0.0,
                        largest_synapse_signal_fraction=float(
                            by_syn.max() / total_positive
                        )
                        if total_positive
                        else 0.0,
                        onset_spread_s=float(
                            np.quantile(first_times, 0.9)
                            - np.quantile(first_times, 0.1)
                        )
                        if len(first_times) > 1
                        else 0.0,
                        left_censored=bool(lo <= smooth_width // 2),
                        right_censored=bool(hi >= length - smooth_width // 2),
                        valid_segment=segment_id,
                        sample_interval_s=dt,
                        signal_units=str(a.attrs.get("units", a.name or "input units")),
                        normalization=normalization,
                        activity_threshold=activity_threshold,
                    )
                )
        if return_diagnostics:
            diagnostics[group] = xr.Dataset(
                {
                    "active_fraction": ("time", frac),
                    "envelope": ("time", envelope),
                    "new_fraction": ("time", recent),
                    "valid": ("time", valid),
                    "input_threshold": ("syn_id", thresholds),
                },
                coords={"time": t, "syn_id": ids[ix]},
                attrs={
                    "level": level,
                    "group": group,
                    "params": asdict(params),
                    "valid_exposure_s": float(valid.sum() * dt),
                    "n_valid_segments": len(runs),
                    "sample_interval_s": dt,
                },
            )
    result = (
        pl.DataFrame(rows, schema=RING_SCHEMA).sort(["start_s", "group"])
        if rows
        else _empty_table()
    )
    if result.height:
        result = result.with_columns(pl.int_range(0, pl.len()).alias("ring_id"))
    return (result, diagnostics) if return_diagnostics else result
