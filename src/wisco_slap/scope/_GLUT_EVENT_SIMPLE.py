"""Simple threshold-run event detection on scopex-style arrays.

Exposed as ``wis.scope.detect_basic_events``.

The detector is deliberately minimal: a sample is "active" when its value is
strictly above ``threshold``, and every maximal contiguous run of active
samples (per unit — synapse / channel / …) that is at least ``min_samples``
long is one event. There is no noise model, refractory logic or template
here — those live in :mod:`wisco_slap.scope._GLUT_EVENT_MAIN` and
:mod:`wisco_slap.scope._GLUT_PURSUIT_PIPELINE`. This is the
"threshold the SNR trace and count the bumps" baseline to compare them
against, and a quick first-look tool.

Typical use
-----------
>>> import wisco_slap as wis
>>> dn = wis.get.combine_scopex_arrays(
...     wis.get.syn_dF(subject, exp, loc, acq, trace="denoised")).sel(channel=1)
>>> ls = wis.get.combine_scopex_arrays(
...     wis.get.syn_dF(subject, exp, loc, acq, trace="ls")).sel(channel=1)
>>> snr = wis.scope.pursuit.matchfilt_snr(dn, ls)
>>> events = wis.scope.detect_basic_events(snr, threshold=5.0, min_samples=2)
>>> events.group_by("syn_id").len()        # events per synapse
>>> events.filter(pl.col("dend-ID") == "A-1")["duration"].mean()

The per-DMD dict returned by the loaders can be passed directly; the result
then carries a ``dmd`` column::

>>> snr_by_dmd = {k: wis.scope.pursuit.matchfilt_snr(dn[k], ls[k]) for k in dn}
>>> events = wis.scope.detect_basic_events(snr_by_dmd, threshold=5.0)
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping

import numpy as np
import polars as pl
import xarray as xr

__all__ = ["detect_basic_events"]


# ---------------------------------------------------------------------------
# Vectorized numpy core
# ---------------------------------------------------------------------------


def _threshold_runs_2d(
    b2d: np.ndarray,
    v2d: np.ndarray,
    min_samples: int,
) -> dict[str, np.ndarray]:
    """Find every contiguous ``True`` run per row and summarise it.

    Fully vectorized across rows and runs (no Python loop over units or
    events). Runs never cross a row boundary.

    Parameters
    ----------
    b2d : np.ndarray, shape ``(n_units, n_time)``, dtype bool
        Active-sample mask.
    v2d : np.ndarray, shape ``(n_units, n_time)``
        Values summarised within each run (peak location / value and mean).
        NaNs are treated as ``-inf`` for the peak search so they are never
        chosen unless a whole run is NaN (cannot happen when the mask was
        built by ``v2d > threshold``).
    min_samples : int
        Runs shorter than this are dropped.

    Returns
    -------
    dict[str, np.ndarray]
        One entry per run (in row-major order, i.e. by unit then start
        sample): ``unit``, ``start``, ``end`` (inclusive), ``peak`` (all
        int64 sample indices), ``n_samples`` (int64), ``peak_value`` and
        ``mean_value`` (float64).

    Notes
    -----
    On ties (several samples equal to the run maximum) the earliest sample
    is the peak, deterministically.
    """
    n_units, n_time = b2d.shape

    # Pad each row with an inactive sample on both sides so every run has a
    # rising and a falling edge; then diff: +1 at a run's first sample, -1 one
    # past its last. np.nonzero scans row-major, and within a row edges strictly
    # alternate (+1, -1, +1, ...), so the k-th rise pairs with the k-th fall.
    padded = np.zeros((n_units, n_time + 2), dtype=np.int8)
    padded[:, 1:-1] = b2d
    d = np.diff(padded, axis=1)
    unit, start = np.nonzero(d == 1)
    _, end_excl = np.nonzero(d == -1)
    end = end_excl - 1
    n_runs = int(unit.shape[0])

    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=np.float64)
    if n_runs == 0:
        return {
            "unit": empty_i,
            "start": empty_i.copy(),
            "end": empty_i.copy(),
            "peak": empty_i.copy(),
            "n_samples": empty_i.copy(),
            "peak_value": empty_f,
            "mean_value": empty_f.copy(),
        }

    unit = unit.astype(np.int64)
    start = start.astype(np.int64)
    end = end.astype(np.int64)
    n_samples = end - start + 1

    # Ascending flat indices of all active samples: because runs are maximal
    # and ordered row-major this is exactly the runs concatenated in order, so
    # segment boundaries are the cumulative run lengths.
    true_pos = np.flatnonzero(b2d)
    seg_starts = np.zeros(n_runs, dtype=np.int64)
    np.cumsum(n_samples[:-1], out=seg_starts[1:])

    vals = v2d.reshape(-1)[true_pos].astype(np.float64)
    seg_sum = np.add.reduceat(vals, seg_starts)
    nan_mask = np.isnan(vals)
    if nan_mask.any():
        vals = np.where(nan_mask, -np.inf, vals)
    seg_max = np.maximum.reduceat(vals, seg_starts)

    # argmax per segment = first sample equal to its segment's max
    is_max = vals == np.repeat(seg_max, n_samples)
    max_pos = np.flatnonzero(is_max)
    max_run = np.repeat(np.arange(n_runs), n_samples)[max_pos]
    first = np.empty(max_pos.shape[0], dtype=bool)
    first[0] = True
    np.not_equal(max_run[1:], max_run[:-1], out=first[1:])
    peak_flat = true_pos[max_pos[first]]  # one per run, in run order
    peak = peak_flat - unit * n_time

    peak_value = seg_max
    mean_value = seg_sum / n_samples

    if min_samples > 1:
        keep = n_samples >= min_samples
        unit, start, end, peak = unit[keep], start[keep], end[keep], peak[keep]
        n_samples = n_samples[keep]
        peak_value, mean_value = peak_value[keep], mean_value[keep]

    return {
        "unit": unit,
        "start": start,
        "end": end,
        "peak": peak,
        "n_samples": n_samples,
        "peak_value": peak_value,
        "mean_value": mean_value,
    }


# ---------------------------------------------------------------------------
# DataFrame assembly
# ---------------------------------------------------------------------------


def _coord_series(name: str, values: np.ndarray) -> pl.Series:
    """Turn per-event coordinate values into a polars Series.

    Object coords (the synapse-label coords are mixed ``str`` / ``None`` /
    ``NaN``) become ``Utf8`` with nulls; float NaNs become nulls; other numeric
    / bool / datetime dtypes pass straight through.
    """
    kind = values.dtype.kind
    if kind == "O":
        cleaned = [
            None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
            for v in values.tolist()
        ]
        return pl.Series(name, cleaned, dtype=pl.Utf8)
    if kind == "U":
        return pl.Series(name, values.tolist(), dtype=pl.Utf8)
    if kind == "f":
        return pl.Series(name, values).fill_nan(None)
    if kind in "iubM":
        return pl.Series(name, values)
    return pl.Series(name, values.tolist())


def _infer_dt(t: np.ndarray) -> float:
    """Sample spacing from a time coordinate (median diff; NaN if < 2 samples)."""
    if t.shape[0] < 2:
        return float("nan")
    return float(np.median(np.diff(t)))


def _event_columns(value_name: str) -> list[str]:
    return [
        "start_time",
        "peak_time",
        "end_time",
        "duration",
        "n_samples",
        f"peak_{value_name}",
        f"mean_{value_name}",
        "start_idx",
        "peak_idx",
        "end_idx",
    ]


def _detect_one(
    snr: xr.DataArray,
    threshold: float | xr.DataArray,
    min_samples: int,
    time_dim: str,
    value_name: str,
) -> pl.DataFrame:
    """Run detection on a single DataArray (see :func:`detect_basic_events`)."""
    if time_dim not in snr.dims:
        raise ValueError(f"time_dim={time_dim!r} not in snr dims {snr.dims}")
    if time_dim not in snr.coords:
        raise ValueError(
            f"snr has no {time_dim!r} coordinate; event times cannot be assigned."
        )

    event_cols = _event_columns(value_name)
    coord_names = [str(c) for c in snr.coords if c != time_dim]
    clash = sorted(set(coord_names) & set(event_cols))
    if clash:
        raise ValueError(
            f"coordinate name(s) {clash} collide with the event columns "
            f"{event_cols}; rename the coordinate or pass a different value_name."
        )

    # Strictly-above mask. NaN compares False, so NaN samples are never active
    # and always break a run. A DataArray threshold broadcasts via xarray
    # (exact alignment required so no unit is silently dropped).
    if isinstance(threshold, xr.DataArray):
        snr, threshold = xr.align(snr, threshold, join="exact")
        if any(d not in snr.dims for d in threshold.dims):
            raise ValueError(
                f"threshold dims {threshold.dims} must be a subset of snr dims "
                f"{snr.dims}"
            )
    with np.errstate(invalid="ignore"):
        mask = snr > threshold

    # Orient as (*unit_dims, time) and flatten the unit dims into one axis.
    unit_dims: list[Hashable] = [d for d in snr.dims if d != time_dim]
    order = (*unit_dims, time_dim)
    da_t = snr.transpose(*order)
    mask_t = mask.transpose(*order)
    unit_shape = tuple(int(da_t.sizes[d]) for d in unit_dims)
    n_units = int(np.prod(unit_shape)) if unit_dims else 1
    n_time = int(da_t.sizes[time_dim])

    b2d = np.ascontiguousarray(mask_t.values, dtype=bool).reshape(n_units, n_time)
    v2d = np.ascontiguousarray(da_t.values).reshape(n_units, n_time)
    runs = _threshold_runs_2d(b2d, v2d, min_samples)
    n_events = int(runs["unit"].shape[0])

    # Per-event index into every original dimension (time -> the peak sample).
    event_idx: dict[Hashable, np.ndarray] = {time_dim: runs["peak"]}
    if unit_dims:
        unit_multi = np.unravel_index(runs["unit"], unit_shape)
        event_idx.update(dict(zip(unit_dims, unit_multi, strict=True)))

    columns: list[pl.Series] = []
    for cname, coord in da_t.coords.items():
        if cname == time_dim:
            continue  # replaced by start_time / peak_time / end_time
        cvals = np.asarray(coord.values)
        if coord.ndim == 0:  # scalar coordinate -> broadcast constant
            col = np.repeat(cvals.reshape(1), n_events)
        else:  # index by the event's position along each of the coord's dims
            col = cvals[tuple(event_idx[d] for d in coord.dims)]
        columns.append(_coord_series(str(cname), col))

    t = np.asarray(da_t.coords[time_dim].values, dtype=np.float64)
    dt = _infer_dt(t)
    columns.extend([
        pl.Series("start_time", t[runs["start"]]),
        pl.Series("peak_time", t[runs["peak"]]),
        pl.Series("end_time", t[runs["end"]]),
        pl.Series("duration", runs["n_samples"].astype(np.float64) * dt),
        pl.Series("n_samples", runs["n_samples"]),
        pl.Series(f"peak_{value_name}", runs["peak_value"]),
        pl.Series(f"mean_{value_name}", runs["mean_value"]),
        pl.Series("start_idx", runs["start"]),
        pl.Series("peak_idx", runs["peak"]),
        pl.Series("end_idx", runs["end"]),
    ])
    return pl.DataFrame(columns)


_DMD_KEY = re.compile(r"^dmd[_-]?(\d+)$")


def _dmd_labels(keys: list[str]) -> list[int | str]:
    """``["dmd_1", "dmd_2"] -> [1, 2]``; any other key set is kept as strings."""
    matches = [_DMD_KEY.match(str(k)) for k in keys]
    if all(m is not None for m in matches):
        return [int(m.group(1)) for m in matches]  # type: ignore[union-attr]
    return [str(k) for k in keys]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_basic_events(
    snr: xr.DataArray | Mapping[str, xr.DataArray],
    threshold: float | xr.DataArray,
    min_samples: int = 2,
    time_dim: str = "time",
    value_name: str = "snr",
) -> pl.DataFrame:
    """Simple threshold-run event detection on a scopex-style array.

    A sample is *active* when its value is strictly above ``threshold``. For
    every unit (each combination of the non-``time`` dims, e.g. each
    ``(channel, syn_id)``) independently, every maximal contiguous run of
    active samples along ``time`` that contains at least ``min_samples``
    samples is one event. NaN samples are never active, so a NaN gap always
    ends a run.

    Each event is summarised by its first active sample (start), its last
    active sample (end), the sample with the largest value (peak; earliest on
    ties), the number of active samples, the peak value and the mean value
    over the run.

    Parameters
    ----------
    snr : xr.DataArray or mapping of str -> xr.DataArray
        Scopex-style array with a ``time`` dimension and coordinate, plus any
        unit dims (``channel``, ``syn_id``, …) and any non-dimension
        coordinates (``soma-ID``, ``dend-ID``, ``pos``, ``dmd``, …). Typically
        an SNR trace (:func:`wisco_slap.scope.pursuit.matchfilt_snr` or
        ``denoised / sigma``), but any array works. A per-DMD dict as returned
        by the ``wis.get`` loaders is also accepted: each entry is processed
        and the results are concatenated with a ``dmd`` column prepended
        (``"dmd_1" -> 1``) unless the arrays already carry a ``dmd`` coord.
    threshold : float or xr.DataArray
        Samples must be **strictly greater** than this to count as active. A
        DataArray broadcastable against ``snr`` (e.g. one value per
        ``syn_id``) gives per-unit thresholds; it must align exactly with
        ``snr`` on its shared coordinates.
    min_samples : int, default 2
        Minimum number of consecutive active samples for a run to count as an
        event. Runs shorter than this are dropped.
    time_dim : str, default ``"time"``
        Name of the time dimension / coordinate.
    value_name : str, default ``"snr"``
        Suffix for the two value columns (``peak_<value_name>`` and
        ``mean_<value_name>``). Set this when ``snr`` is not actually an SNR
        trace (e.g. ``"dff"``).

    Returns
    -------
    pl.DataFrame
        One row per event, ordered by unit (row-major over the non-time dims,
        e.g. ``channel`` then ``syn_id``) and then by start sample. Columns:

        * every coordinate of ``snr`` except ``time`` itself, under its own
          name — unit coordinates (``syn_id``, ``channel``, ``soma-ID``,
          ``dend-ID``, ``pos``, …) identify the unit, coordinates that vary
          along ``time`` are evaluated at the event's peak sample, and scalar
          coordinates are broadcast. Object-dtype coordinates become ``Utf8``
          (``None``/NaN -> null); float NaNs become nulls;
        * ``start_time``, ``peak_time``, ``end_time`` — the ``time`` coordinate
          at the first active, peak and last active sample (same units as the
          ``time`` coord, i.e. ephys seconds for the standard loaders);
        * ``duration`` — the span the active samples occupy,
          ``n_samples * dt`` (``dt`` = median spacing of the ``time`` coord).
          This equals ``end_time - start_time + dt`` and is never zero; use
          ``end_time - start_time`` if you want the peak-to-peak-free span
          between the first and last crossing instead;
        * ``n_samples`` — number of active samples in the run;
        * ``peak_<value_name>``, ``mean_<value_name>`` — the array's value at
          the peak sample and its mean over the run;
        * ``start_idx``, ``peak_idx``, ``end_idx`` — the corresponding
          0-based sample indices along ``time`` (inclusive end), for indexing
          back into the array.

        With no events the frame has zero rows but the full schema.

    Raises
    ------
    ValueError
        If ``time_dim`` is not a dimension (or has no coordinate), if
        ``min_samples < 1``, if ``threshold`` is not finite, or if a coordinate
        name collides with one of the event columns.

    Notes
    -----
    Fully vectorized across units and events; a whole
    ``(channel, syn_id, time)`` acquisition runs in seconds. Events are
    reported per unit — the same glutamate event seen in both channels (or in
    neighbouring synapses) yields separate rows; no merging is attempted.
    """
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}")
    if not isinstance(threshold, xr.DataArray):
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError(f"threshold must be finite, got {threshold}")

    if isinstance(snr, xr.DataArray):
        return _detect_one(snr, threshold, min_samples, time_dim, value_name)

    if not isinstance(snr, Mapping):
        raise TypeError(
            "snr must be an xr.DataArray or a mapping of str -> xr.DataArray, "
            f"got {type(snr).__name__}"
        )
    keys = list(snr.keys())
    frames: list[pl.DataFrame] = []
    for key, label in zip(keys, _dmd_labels(keys), strict=True):
        df = _detect_one(snr[key], threshold, min_samples, time_dim, value_name)
        if "dmd" not in df.columns:
            dtype = pl.Int64 if isinstance(label, int) else pl.Utf8
            df = df.select(pl.lit(label, dtype=dtype).alias("dmd"), pl.all())
        frames.append(df)
    if not frames:
        raise ValueError("snr mapping is empty")
    # diagonal_relaxed: the per-DMD label coords can differ in dtype (e.g.
    # ``pos`` int32 on one DMD, float64 on the other) — cast to the supertype.
    return pl.concat(frames, how="diagonal_relaxed")
