"""Refined glutamate event-detection pipeline (scopex arrays).

This module holds the finalized event-detection primitives that operate
directly on scopex ``xr.DataArray``\\ s (dims ``(syn_id, time)``, optionally
with a leading ``channel`` dim, plus the usual non-dimension coordinates
``soma-ID``/``dend-ID``/``dmd``/...).

The first stage of the pipeline is :func:`masked_bools_to_peak_events`, which
collapses a supra-threshold boolean mask into discrete, single-timepoint
events (one per contiguous run, located at the run's peak).
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import polars as pl
import xarray as xr

__all__ = ["masked_bools_to_peak_events"]


# ---------------------------------------------------------------------------
# Vectorized numpy core
# ---------------------------------------------------------------------------


def _peak_events_2d(
    b2d: np.ndarray,
    v2d: np.ndarray,
    min_run_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find, per row, the peak sample of every contiguous ``True`` run.

    Fully vectorized across all rows and runs (no Python-level loop over
    synapses or events).

    Parameters
    ----------
    b2d : np.ndarray, shape ``(n_units, n_time)``, dtype bool
        Boolean mask. Runs are contiguous ``True`` stretches along axis 1
        and never cross the row (unit) boundary.
    v2d : np.ndarray, shape ``(n_units, n_time)``
        Values used to locate each run's peak (its argmax). NaNs are treated
        as ``-inf`` so they are never chosen unless a whole run is NaN.
    min_run_length : int
        Minimum number of ``True`` samples a run must contain to yield an
        event.

    Returns
    -------
    peak_unit : np.ndarray, shape ``(n_events,)``, int64
        Row index of each event's peak.
    peak_time : np.ndarray, shape ``(n_events,)``, int64
        Column (time) index of each event's peak.
    run_length : np.ndarray, shape ``(n_events,)``, int64
        Number of ``True`` samples in the run the event came from.
    peak_values : np.ndarray, shape ``(n_events,)``
        Value of ``v2d`` at each peak (original dtype of ``v2d``).

    Notes
    -----
    On ties (several samples equal to the run maximum) the *first*
    (earliest-in-time) sample is kept, deterministically.
    """
    n_time = b2d.shape[1]

    # Run starts / ends along time: a start is a True whose left neighbour is
    # False (or the row edge); an end is a True whose right neighbour is False
    # (or the row edge). np.nonzero scans row-major, so the k-th start pairs
    # with the k-th end.
    prev = np.zeros_like(b2d)
    prev[:, 1:] = b2d[:, :-1]
    nxt = np.zeros_like(b2d)
    nxt[:, :-1] = b2d[:, 1:]
    starts = b2d & ~prev
    ends = b2d & ~nxt

    su, sc = np.nonzero(starts)
    _, ec = np.nonzero(ends)
    n_runs = su.shape[0]

    if n_runs == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), empty_i.copy(), np.empty(0, dtype=v2d.dtype)

    run_len = (ec - sc + 1).astype(np.int64)

    # Ascending flat indices of all True samples. Because runs are maximal and
    # ordered row-major, this is exactly the runs concatenated in order, so the
    # run boundaries within it are the cumulative run lengths.
    true_pos = np.flatnonzero(b2d)
    seg_starts = np.zeros(n_runs, dtype=np.int64)
    np.cumsum(run_len[:-1], out=seg_starts[1:])

    # Gather the detection values at the True samples (NaN-safe for argmax).
    gathered = v2d.reshape(-1)[true_pos].astype(np.float64)
    nan_mask = np.isnan(gathered)
    if nan_mask.any():
        gathered = np.where(nan_mask, -np.inf, gathered)

    # Segmented max, then the first sample equal to its run's max = argmax.
    seg_max = np.maximum.reduceat(gathered, seg_starts)
    is_max = gathered == np.repeat(seg_max, run_len)
    max_pos = np.flatnonzero(is_max)
    max_run = np.repeat(np.arange(n_runs), run_len)[max_pos]
    first = np.empty(max_pos.shape[0], dtype=bool)
    first[0] = True
    np.not_equal(max_run[1:], max_run[:-1], out=first[1:])
    peak_flat = true_pos[max_pos[first]]  # one entry per run, in run order

    # Drop runs shorter than the minimum.
    if min_run_length > 1:
        keep = run_len >= min_run_length
        peak_flat = peak_flat[keep]
        run_len = run_len[keep]

    peak_unit = peak_flat // n_time
    peak_time = peak_flat - peak_unit * n_time
    peak_values = v2d.reshape(-1)[peak_flat]
    return peak_unit, peak_time, run_len, peak_values


# ---------------------------------------------------------------------------
# DataFrame assembly
# ---------------------------------------------------------------------------


def _events_dataframe(
    source: xr.DataArray,
    unit_dims: list[Hashable],
    time_dim: Hashable,
    unit_shape: tuple[int, ...],
    peak_unit: np.ndarray,
    peak_time: np.ndarray,
    run_length: np.ndarray,
    peak_values: np.ndarray,
    peak_value_name: str,
) -> pl.DataFrame:
    """Build a one-row-per-event polars DataFrame.

    Every coordinate of ``source`` becomes a column, evaluated at each event's
    peak position, followed by ``peak_value_name`` (the value of the detection
    array at the peak) and ``run_length``.
    """
    n_events = int(peak_unit.shape[0])

    # Per-event index into each original dimension.
    event_idx: dict[Hashable, np.ndarray] = {time_dim: peak_time}
    if unit_dims:
        unit_multi = np.unravel_index(peak_unit, unit_shape)
        event_idx.update(dict(zip(unit_dims, unit_multi, strict=True)))

    data: dict[str, object] = {}
    for cname, coord in source.coords.items():
        cdims = coord.dims
        cvals = coord.values
        if len(cdims) == 0:  # scalar coordinate -> broadcast constant
            col = np.full(n_events, cvals.item())
        else:  # index by the event's position along each of the coord's dims
            col = cvals[tuple(event_idx[d] for d in cdims)]
        # polars infers Utf8 cleanly from python lists, not numpy object arrays
        data[str(cname)] = col.tolist() if col.dtype == object else col

    if peak_value_name in data:
        raise ValueError(
            f"peak_value_name={peak_value_name!r} collides with an existing "
            "coordinate name; choose a different peak_value_name."
        )
    data[peak_value_name] = peak_values
    data["run_length"] = run_length.astype(np.int64)
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def masked_bools_to_peak_events(
    bool_array: xr.DataArray,
    values_array: xr.DataArray,
    min_run_length: int = 1,
    return_polars_df: bool = False,
    time_dim: str = "time",
    peak_value_name: str = "peak_value",
) -> xr.DataArray | tuple[xr.DataArray, pl.DataFrame]:
    """Collapse a supra-threshold boolean mask into discrete peak events.

    For every unit (e.g. synapse) independently, each contiguous run of
    ``True`` along ``time_dim`` is reduced to the single sample at which
    ``values_array`` is largest within that run. The result is a boolean
    DataArray of the same shape as ``bool_array`` that is ``True`` only at
    those per-run peaks — i.e. one discrete event per run, located at one
    timepoint, rather than every supra-threshold sample.

    This is the first stage of the refined event-detection pipeline. A typical
    call thresholds a matched-filter SNR trace and locates each event's peak on
    either the SNR trace or the raw matched-filter trace::

        mf_bool = mfsnr > 3
        peaks = masked_bools_to_peak_events(mf_bool, mfsnr)

    Parameters
    ----------
    bool_array : xr.DataArray
        Boolean (or boolean-coercible) scopex mask, e.g. ``mfsnr > thresh``.
        Must contain ``time_dim``; every other dimension is treated as an
        independent unit axis (runs never cross unit boundaries). Non-NaN /
        masking is assumed already folded into the mask (NaN samples are
        ``False``).
    values_array : xr.DataArray
        Array whose per-run argmax defines each event's peak location. Must
        have the same dims and shape as ``bool_array`` (e.g. the raw matched
        filter ``mf`` or the SNR trace ``mfsnr``). NaNs are treated as ``-inf``
        and so are never selected unless a whole run is NaN.
    min_run_length : int, default 1
        Minimum number of consecutive ``True`` samples a run must contain to
        produce an event. Runs shorter than this are dropped entirely. Must be
        ``>= 1``.
    return_polars_df : bool, default False
        If True, additionally return a one-row-per-event :class:`polars.DataFrame`
        (see Returns).
    time_dim : str, default ``"time"``
        Dimension along which runs are found.
    peak_value_name : str, default ``"peak_value"``
        Column name for the peak detection value in the returned DataFrame.
        Must not collide with an existing coordinate name.

    Returns
    -------
    peaks : xr.DataArray
        Boolean DataArray with the same dims, shape, coordinates, name, and
        attrs as ``bool_array``, ``True`` only at the per-run peak samples.
        Two marker attrs are added: ``event_peaks=True`` and ``min_run_length``.
    events_df : pl.DataFrame, optional
        Returned only when ``return_polars_df`` is True. One row per detected
        event, in (unit, time) order. Columns: every coordinate of
        ``bool_array`` evaluated at the event's peak (so ``time`` is the peak
        time and ``syn_id`` etc. identify the unit), then ``peak_value_name``
        (the ``values_array`` value at the peak) and ``run_length`` (number of
        ``True`` samples in the run).

    Notes
    -----
    The implementation is fully vectorized across units and runs. On ties
    within a run (multiple samples equal to the maximum) the earliest sample is
    kept.
    """
    if min_run_length < 1:
        raise ValueError(f"min_run_length must be >= 1, got {min_run_length}")
    if time_dim not in bool_array.dims:
        raise ValueError(
            f"time_dim={time_dim!r} not found in bool_array dims {bool_array.dims}"
        )
    if set(bool_array.dims) != set(values_array.dims):
        raise ValueError(
            "bool_array and values_array must share the same dims; got "
            f"{bool_array.dims} vs {values_array.dims}"
        )
    if any(bool_array.sizes[d] != values_array.sizes[d] for d in bool_array.dims):
        raise ValueError("bool_array and values_array must have the same shape")

    # Orient as (*unit_dims, time) and flatten units into one axis.
    unit_dims = [d for d in bool_array.dims if d != time_dim]
    order = (*unit_dims, time_dim)
    bt = bool_array.transpose(*order)
    vt = values_array.transpose(*order)

    unit_shape = tuple(int(bt.sizes[d]) for d in unit_dims)
    n_units = int(np.prod(unit_shape)) if unit_dims else 1
    n_time = int(bt.sizes[time_dim])

    b2d = np.ascontiguousarray(bt.values, dtype=bool).reshape(n_units, n_time)
    v2d = np.ascontiguousarray(vt.values).reshape(n_units, n_time)

    peak_unit, peak_time, run_length, peak_values = _peak_events_2d(
        b2d, v2d, min_run_length
    )

    # Scatter the peaks into an all-False mask, back in the original layout.
    mask2d = np.zeros((n_units, n_time), dtype=bool)
    mask2d[peak_unit, peak_time] = True
    peaks = xr.DataArray(
        mask2d.reshape(bt.shape),
        dims=order,
        coords=bt.coords,
        attrs={
            **bool_array.attrs,
            "event_peaks": True,
            "min_run_length": int(min_run_length),
        },
        name=bool_array.name,
    ).transpose(*bool_array.dims)

    if not return_polars_df:
        return peaks

    events_df = _events_dataframe(
        bt, unit_dims, time_dim, unit_shape,
        peak_unit, peak_time, run_length, peak_values, peak_value_name,
    )
    return peaks, events_df
