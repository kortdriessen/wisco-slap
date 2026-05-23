"""Validity mask construction and simple integrals over masks.

A "validity mask" is a boolean ``xarray.DataArray`` aligned with a scopex
DataArray's time axis. ``True`` means the timepoint is real, recorded data;
``False`` means it should be excluded from any analysis (NaN sample,
rejected, between-trial gap, etc.).

This module is the foundation of :mod:`wisco_slap.util.validity`. Higher-
level helpers (hypnogram tiling, event rates) consume the masks produced
here.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xarray as xr

# A scopex DataArray's reduce dimension is normally ``syn_id``, but ROI
# arrays use ``soma_id`` instead. We auto-detect.
_REDUCE_DIM_CANDIDATES = ("syn_id", "soma_id")

_VALID_MODES = ("all", "any", "per_syn")


def _detect_reduce_dim(da: xr.DataArray) -> str | None:
    """Return the dim to collapse over (``syn_id`` or ``soma_id``), or None."""
    for cand in _REDUCE_DIM_CANDIDATES:
        if cand in da.dims:
            return cand
    return None


def _check_single_channel(da: xr.DataArray) -> xr.DataArray:
    """Drop a singleton channel dim if present; raise on multi-channel."""
    if "channel" in da.dims:
        if da.sizes["channel"] == 1:
            return da.squeeze("channel", drop=True)
        raise ValueError(
            f"`da` has {da.sizes['channel']} channels — select one before "
            "computing a validity mask (NaN patterns differ across "
            "indicators)."
        )
    return da


def _median_dt(time_values: np.ndarray) -> float:
    """Median sample spacing, used as the canonical dt for a time axis."""
    if time_values.size < 2:
        raise ValueError(
            "Need at least 2 time samples to determine sampling interval."
        )
    dt = float(np.median(np.diff(time_values)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            f"Could not determine a positive median sampling interval (got {dt})."
        )
    return dt


def validity_mask(da: xr.DataArray, mode: str = "all") -> xr.DataArray:
    """Boolean validity mask aligned with ``da``'s time axis.

    Parameters
    ----------
    da
        A scopex DataArray with a ``time`` dim and (typically) a ``syn_id``
        or ``soma_id`` dim. A singleton ``channel`` dim is squeezed; a
        multi-channel array is rejected (NaN patterns differ across
        indicators, so collapsing them silently would be misleading).
    mode
        - ``'all'`` (default): 1-D bool over time; ``True`` where every
          synapse is non-NaN. The right default for analyses that pool
          across syns.
        - ``'any'``: 1-D bool over time; ``True`` where at least one
          synapse is non-NaN. Useful when you'll mask per-syn anyway and
          just want to know "was the scope on?".
        - ``'per_syn'``: 2-D bool ``(syn_id, time)`` (or ``(soma_id, time)``);
          per-syn validity, no collapse. The raw form, for analyses that
          will reduce over syns themselves.

    Returns
    -------
    xr.DataArray
        Boolean DataArray. For 1-D modes, dims are ``('time',)`` with the
        same time coord as ``da``. For ``'per_syn'``, dims are
        ``(reduce_dim, 'time')`` with both coords copied from ``da``.

    Notes
    -----
    Filter the array to your analysis subset (one DMD, one soma, the
    dendrites you care about) *before* calling this function. The mask is
    derived from whatever syns are in the input — if you include
    chronically NaN syns, the ``'all'`` mode mask will be near-empty.

    For 1-D inputs (no syn dim), the output is just ``~isnan(da)`` regardless
    of mode.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"`mode` must be one of {_VALID_MODES}, got {mode!r}."
        )

    da = _check_single_channel(da)

    if "time" not in da.dims:
        raise ValueError(
            f"`da` must have a 'time' dim (got {da.dims!r})."
        )

    reduce_dim = _detect_reduce_dim(da)

    is_finite = ~np.isnan(da.values).astype(bool) if np.issubdtype(da.dtype, np.floating) else np.ones(
        da.shape, dtype=bool
    )
    is_finite = xr.DataArray(is_finite, dims=da.dims, coords=da.coords)

    if reduce_dim is None:
        # 1-D input (no syn/soma dim) — mode is irrelevant.
        return is_finite.transpose("time")

    if mode == "per_syn":
        return is_finite.transpose(reduce_dim, "time")

    if mode == "all":
        collapsed = is_finite.all(dim=reduce_dim)
    else:  # mode == "any"
        collapsed = is_finite.any(dim=reduce_dim)

    return collapsed.transpose("time")


def attach_validity(
    da: xr.DataArray,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Return ``da`` with an ``is_valid`` coord on its time dim.

    The coord travels along when callers do ``da.sel(time=...)`` so a
    downstream consumer that supports the auto-detect pattern (see
    :func:`resolve_mask`) can recover the mask without an explicit argument.

    Parameters
    ----------
    da
        A DataArray with a ``time`` dim.
    mask
        Optional precomputed 1-D mask. When omitted, a mask is derived via
        ``validity_mask(da, mode='all')``. The mask must be 1-D over time
        and align with ``da``'s time coord.

    Returns
    -------
    xr.DataArray
        ``da`` with ``is_valid`` attached as a non-dim coord on the time
        dim.

    Warning
    -------
    Most binary xarray operations drop non-dim coords. After arithmetic
    (``a + b``, ``a - a.mean('time')``, ``a.where(...)``) the
    ``is_valid`` coord may silently disappear. For analyses that
    transform the array, prefer passing ``mask`` explicitly to consumers
    rather than relying on the attached coord.
    """
    if "time" not in da.dims:
        raise ValueError("`da` must have a 'time' dim to attach validity.")

    if mask is None:
        mask = validity_mask(da, mode="all")

    if "time" not in mask.dims:
        raise ValueError("`mask` must have a 'time' dim.")
    if mask.sizes["time"] != da.sizes["time"]:
        raise ValueError(
            f"Mask time length ({mask.sizes['time']}) does not match "
            f"da time length ({da.sizes['time']})."
        )

    return da.assign_coords(is_valid=("time", np.asarray(mask.values, dtype=bool)))


def resolve_mask(
    da: xr.DataArray,
    mask: xr.DataArray | None,
) -> xr.DataArray:
    """Internal mask-resolution policy used by consuming functions.

    Resolution order:

    1. If ``mask`` is provided, validate and return it.
    2. Else if ``da`` has an ``is_valid`` coord on its time dim, lift it
       to a 1-D bool DataArray.
    3. Else if ``da`` is float-typed, derive a mask via
       ``validity_mask(da, mode='all')``.
    4. Else (boolean / integer array, no ``is_valid`` coord) raise — there
       is no way to determine validity from the array alone, and silently
       assuming "all valid" would produce wrong denominators in rate
       calculations.

    Parameters
    ----------
    da
        The DataArray whose time axis the mask must align with.
    mask
        Explicit mask, or None.

    Returns
    -------
    xr.DataArray
        A 1-D boolean DataArray with the same time coord as ``da``.
    """
    if mask is not None:
        if "time" not in mask.dims:
            raise ValueError("`mask` must have a 'time' dim.")
        if mask.sizes["time"] != da.sizes["time"]:
            raise ValueError(
                "Explicit mask time length does not match da time length."
            )
        return mask.astype(bool)

    if "is_valid" in da.coords:
        # Coord must already be aligned with the time dim — that's the
        # contract of attach_validity.
        return xr.DataArray(
            np.asarray(da["is_valid"].values, dtype=bool),
            dims=("time",),
            coords={"time": da["time"]},
            name="is_valid",
        )

    if np.issubdtype(da.dtype, np.floating):
        return validity_mask(da, mode="all")

    raise ValueError(
        "Cannot determine validity for a non-float array without an "
        "explicit `mask=` argument or an attached `is_valid` coord. The "
        "array's NaN pattern (the usual fallback) is unavailable for "
        f"dtype {da.dtype}. Pass `mask=` explicitly."
    )


def valid_duration(
    mask: xr.DataArray,
    t1: float | None = None,
    t2: float | None = None,
) -> float:
    """Seconds of valid time within ``[t1, t2]``.

    Parameters
    ----------
    mask
        1-D boolean DataArray with a ``time`` coord.
    t1, t2
        Optional time bounds (inclusive). Default: full extent of ``mask``.

    Returns
    -------
    float
        ``n_true_samples * dt`` where ``dt`` is the median sample spacing.

    Notes
    -----
    The result has sub-``dt`` precision in expectation — there is no
    fractional-sample accounting at the boundaries. For typical 200 Hz
    scopex data, dt = 5 ms, which is well below the precision of any
    downstream science.
    """
    if "time" not in mask.dims:
        raise ValueError("`mask` must have a 'time' dim.")

    time_values = np.asarray(mask["time"].values)
    dt = _median_dt(time_values)

    if t1 is None and t2 is None:
        return float(np.count_nonzero(mask.values)) * dt

    lo = -np.inf if t1 is None else t1
    hi = np.inf if t2 is None else t2
    if lo > hi:
        raise ValueError(f"t1 ({t1}) must be <= t2 ({t2}).")

    in_range = (time_values >= lo) & (time_values <= hi)
    return float(np.count_nonzero(np.asarray(mask.values, dtype=bool) & in_range)) * dt


def validity_intervals(mask: xr.DataArray) -> pl.DataFrame:
    """Contiguous-True runs of ``mask`` as (start_time, end_time, duration).

    Each run corresponds to a stretch of consecutive ``True`` samples. The
    start and end times are the time-coord values of the first and last
    ``True`` sample of the run, and ``duration`` is
    ``(n_samples_in_run) * dt`` (so a single-sample run has ``duration =
    dt``, not zero).

    Parameters
    ----------
    mask
        1-D boolean DataArray with a ``time`` coord.

    Returns
    -------
    pl.DataFrame
        Columns: ``start_time`` (f64), ``end_time`` (f64), ``duration`` (f64).
        Empty DataFrame if ``mask`` has no True samples.
    """
    if "time" not in mask.dims:
        raise ValueError("`mask` must have a 'time' dim.")

    time_values = np.asarray(mask["time"].values, dtype=float)
    bools = np.asarray(mask.values, dtype=bool)

    if bools.size == 0:
        return pl.DataFrame(
            schema={"start_time": pl.Float64, "end_time": pl.Float64, "duration": pl.Float64},
        )

    dt = _median_dt(time_values)

    # Find run boundaries via diff on the bool array.
    padded = np.concatenate([[False], bools, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]      # index of first True sample of each run
    ends = np.where(edges == -1)[0] - 1   # index of last True sample of each run

    if starts.size == 0:
        return pl.DataFrame(
            schema={"start_time": pl.Float64, "end_time": pl.Float64, "duration": pl.Float64},
        )

    return pl.DataFrame(
        {
            "start_time": time_values[starts],
            "end_time": time_values[ends],
            # Use sample count for duration so single-sample runs are dt-long, not zero.
            "duration": (ends - starts + 1).astype(np.float64) * dt,
        }
    )
