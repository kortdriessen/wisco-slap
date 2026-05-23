"""NaN-aware event-rate primitives.

These functions live one step above :mod:`mask` and :mod:`hypno`: they
combine a per-syn boolean event array with a hypnogram and a validity
mask to produce per-syn-per-state event rates with the correct (valid-
time) denominator.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import polars as pl
import xarray as xr

from .hypno import _state_iter
from .mask import _median_dt, resolve_mask


def _resolve_states(hypno, states: str | Iterable[str]) -> tuple[str, ...]:
    """Expand ``'all'`` to every state present in the hypno."""
    if isinstance(states, str) and states == "all":
        return tuple(hypno.states)
    return tuple(_state_iter(states))


def valid_duration_per_state(
    mask: xr.DataArray,
    hypno,
    states: str | Iterable[str] = "all",
) -> pl.DataFrame:
    """Per-state valid duration (seconds).

    Parameters
    ----------
    mask
        1-D boolean DataArray on the analysis sample grid.
    hypno
        ``electro_py.hypno.Hypnogram`` with a float time axis.
    states
        ``'all'`` (default) for every state in the hypnogram, a single
        state name, or an iterable of state names.

    Returns
    -------
    pl.DataFrame
        Columns: ``state`` (str), ``valid_duration_s`` (f64). One row per
        requested state; states with zero valid samples appear with 0.0.
    """
    if "time" not in mask.dims:
        raise ValueError("`mask` must have a 'time' dim.")

    time_values = np.asarray(mask["time"].values, dtype=float)
    valid = np.asarray(mask.values, dtype=bool)
    dt = _median_dt(time_values)

    requested = _resolve_states(hypno, states)
    rows = []
    for s in requested:
        s_mask = hypno.mask_times_by_state(time_values, [s])
        rows.append(
            {
                "state": s,
                "valid_duration_s": float(np.count_nonzero(s_mask & valid)) * dt,
            }
        )

    return pl.DataFrame(
        rows,
        schema={"state": pl.String, "valid_duration_s": pl.Float64},
    )


def event_rate_per_state(
    event_array: xr.DataArray,
    hypno,
    *,
    mask: xr.DataArray | None = None,
    states: str | Iterable[str] = "all",
) -> pl.DataFrame:
    """Per-syn event rate per hypnogram state, with NaN-aware denominator.

    For each (syn_id, state) pair, counts events in valid time within the
    state and divides by the valid-time duration of that state. Events
    that fall on NaN samples are excluded from the count by intersecting
    the event array with the mask.

    Parameters
    ----------
    event_array
        Boolean DataArray with dims ``(syn_id, time)`` (or ``(soma_id,
        time)``). ``True`` where an event was detected. Typically
        produced by ``detect_validated_peaks``, which sets NaN samples to
        ``False`` — meaning the array on its own has no validity
        information; the ``mask=`` argument or an attached ``is_valid``
        coord is required to recover it.
    hypno
        ``electro_py.hypno.Hypnogram`` with a float time axis.
    mask
        1-D boolean validity DataArray. See :func:`resolve_mask` for the
        full resolution policy. For boolean event arrays without an
        attached ``is_valid`` coord this argument is **required** —
        otherwise the function raises rather than silently divide by
        wall-clock duration.
    states
        ``'all'`` (default) or a state name / iterable of state names.

    Returns
    -------
    pl.DataFrame
        Columns: ``syn_id`` (carries dtype of input), ``state`` (str),
        ``n_events`` (i64), ``valid_duration_s`` (f64), ``rate_hz`` (f64).
        ``rate_hz`` is ``NaN`` when ``valid_duration_s == 0`` (no valid
        time in that state for that syn — division would be undefined).
    """
    if "time" not in event_array.dims:
        raise ValueError("`event_array` must have a 'time' dim.")

    # Find the syn-like dim.
    reduce_dim = None
    for cand in ("syn_id", "soma_id"):
        if cand in event_array.dims:
            reduce_dim = cand
            break
    if reduce_dim is None:
        raise ValueError(
            "`event_array` must have a 'syn_id' or 'soma_id' dim."
        )

    resolved_mask = resolve_mask(event_array, mask)

    # Align dim order so we can broadcast cleanly.
    ev = event_array.transpose(reduce_dim, "time")
    ev_bool = np.asarray(ev.values, dtype=bool)
    mask_bool = np.asarray(resolved_mask.values, dtype=bool)

    time_values = np.asarray(ev["time"].values, dtype=float)
    dt = _median_dt(time_values)

    # Pre-compute (events ∧ valid) once: a NaN sample can't host a real
    # event, so exclude it from the numerator.
    ev_valid = ev_bool & mask_bool[None, :]

    requested = _resolve_states(hypno, states)
    syn_values = ev[reduce_dim].values

    rows = []
    for s in requested:
        s_mask = hypno.mask_times_by_state(time_values, [s])
        valid_in_state = float(np.count_nonzero(s_mask & mask_bool)) * dt
        # Per-syn count of events in this state.
        events_in_state = (ev_valid & s_mask[None, :]).sum(axis=1)
        for i, syn in enumerate(syn_values):
            n_events = int(events_in_state[i])
            if valid_in_state > 0:
                rate = n_events / valid_in_state
            else:
                rate = float("nan")
            rows.append(
                {
                    "syn_id": syn,
                    "state": s,
                    "n_events": n_events,
                    "valid_duration_s": valid_in_state,
                    "rate_hz": rate,
                }
            )

    # Use schema_overrides instead of schema so the syn_id dtype follows
    # the input (could be str like "dmd_1-5" or int).
    return pl.DataFrame(
        rows,
        schema_overrides={
            "state": pl.String,
            "n_events": pl.Int64,
            "valid_duration_s": pl.Float64,
            "rate_hz": pl.Float64,
        },
    )
