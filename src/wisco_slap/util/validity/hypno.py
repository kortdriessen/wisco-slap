"""Hypnogram × validity-mask integration.

Functions here resolve a hypnogram and a validity mask onto the same
sample grid, then provide two primitives:

- :func:`valid_state_intervals` — contiguous (state ∧ valid) runs.
- :func:`valid_state_epochs`     — fixed-length-epoch tiling, NaN-aware.

The tiling has two modes (see :func:`valid_state_epochs` for details):

- ``'span'`` (default): each epoch contains exactly ``epoch_length`` seconds
  of valid data; wall-clock window may be longer when interior NaN.
- ``'strict'``: tile only within contiguous (state ∧ valid) runs, so every
  epoch has wall_duration == valid_duration == epoch_length.

A ``max_nan_span`` parameter on span mode caps the longest contiguous NaN
run that an epoch may swallow — useful for "span small NaN gaps but skip
big ones."
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import polars as pl
import xarray as xr

from .mask import _median_dt


def _state_iter(states: str | Iterable[str]) -> tuple[str, ...]:
    """Normalize a state argument into a tuple of state names."""
    if isinstance(states, str):
        return (states,)
    return tuple(states)


def _resolve_to_grid(
    hypno,
    mask: xr.DataArray,
    state: str | Iterable[str],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Project hypno + mask onto a single sample grid.

    Returns
    -------
    state_mask : np.ndarray[bool]
        ``True`` where the time sample falls inside any bout of the
        requested ``state`` (or any of the requested states).
    valid_mask : np.ndarray[bool]
        ``True`` where the validity mask is True.
    dt : float
        The sample-grid dt.

    All work is on integer indices into the mask's time coord — no float-
    boundary rounding mismatch is possible.
    """
    if "time" not in mask.dims:
        raise ValueError("`mask` must have a 'time' dim.")
    time_values = np.asarray(mask["time"].values, dtype=float)
    dt = _median_dt(time_values)

    state_list = list(_state_iter(state))
    state_mask = hypno.mask_times_by_state(time_values, state_list)
    valid_mask = np.asarray(mask.values, dtype=bool)

    if state_mask.shape != valid_mask.shape:
        raise ValueError(
            f"Internal shape mismatch: state_mask {state_mask.shape} vs "
            f"valid_mask {valid_mask.shape}."
        )
    return state_mask, valid_mask, dt


def _runs_of_true(arr: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start_idx, end_idx_inclusive)`` of each contiguous True run."""
    if arr.size == 0:
        return []
    padded = np.concatenate([[False], arr.astype(bool), [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def valid_state_intervals(
    hypno,
    mask: xr.DataArray,
    state: str | Iterable[str],
) -> pl.DataFrame:
    """Continuous (state ∧ valid) intervals.

    Returns one row per maximal stretch of time where the hypnogram says
    a sample is in one of the requested states *and* the mask says the
    sample is valid.

    Parameters
    ----------
    hypno
        ``electro_py.hypno.Hypnogram`` with a float time axis.
    mask
        1-D boolean DataArray on the analysis sample grid.
    state
        Single state name or iterable of state names. Multiple states are
        unioned (an interval that flips between any of them with no
        validity break stays as one interval).

    Returns
    -------
    pl.DataFrame
        Columns: ``state`` (the requested label, joined with ``|`` if
        multiple), ``start_time`` (f64), ``end_time`` (f64), ``duration``
        (f64). Empty if no qualifying samples exist.
    """
    state_mask, valid_mask, dt = _resolve_to_grid(hypno, mask, state)
    eligible = state_mask & valid_mask

    state_label = "|".join(_state_iter(state))
    time_values = np.asarray(mask["time"].values, dtype=float)

    runs = _runs_of_true(eligible)
    if not runs:
        return pl.DataFrame(
            schema={
                "state": pl.String,
                "start_time": pl.Float64,
                "end_time": pl.Float64,
                "duration": pl.Float64,
            }
        )

    starts, ends = zip(*runs)
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    return pl.DataFrame(
        {
            "state": [state_label] * len(runs),
            "start_time": time_values[starts],
            "end_time": time_values[ends],
            "duration": (ends - starts + 1).astype(np.float64) * dt,
        }
    )


def _tile_strict(
    state_mask: np.ndarray,
    valid_mask: np.ndarray,
    epoch_samples: int,
) -> list[tuple[int, int]]:
    """Strict tiling: tile non-overlapping epochs inside (state ∧ valid) runs.

    Returns list of ``(a_idx, b_idx_inclusive)`` for each emitted epoch.
    """
    eligible = state_mask & valid_mask
    out: list[tuple[int, int]] = []
    for run_start, run_end in _runs_of_true(eligible):
        run_len = run_end - run_start + 1
        n_fit = run_len // epoch_samples
        for k in range(n_fit):
            a = run_start + k * epoch_samples
            b = a + epoch_samples - 1
            out.append((a, b))
    return out


def _tile_span(
    state_mask: np.ndarray,
    valid_mask: np.ndarray,
    epoch_samples: int,
    max_nan_samples: int | float,
) -> list[tuple[int, int]]:
    """Spanning tile with optional max-NaN-run bailout.

    Iterates over contiguous bouts of the requested state (runs of
    ``state_mask``). Within each bout, walks a cursor: advance past
    leading invalid samples, then walk the window end forward
    accumulating valid samples until the target is reached. If a
    contiguous invalid-run inside the window exceeds ``max_nan_samples``,
    abandon that attempt and restart the cursor past the offending run.

    Returns list of ``(a_idx, b_idx_inclusive)`` for each emitted epoch.
    """
    out: list[tuple[int, int]] = []

    # Bouts of the state on this grid (one or more). Tiling never spans
    # state boundaries, even if validity is contiguous across them.
    for bout_start, bout_end_inclusive in _runs_of_true(state_mask):
        bout_end = bout_end_inclusive + 1  # half-open upper bound
        cursor = bout_start
        while cursor < bout_end:
            # Advance cursor past leading invalid samples within the bout.
            a = cursor
            while a < bout_end and not valid_mask[a]:
                a += 1
            if a >= bout_end:
                break

            # Walk the window end forward until we hit the target valid count
            # or a too-long NaN run.
            b = a
            valid_count = 0
            nan_run_len = 0
            bailed = False
            while b < bout_end and valid_count < epoch_samples:
                if valid_mask[b]:
                    valid_count += 1
                    nan_run_len = 0
                else:
                    nan_run_len += 1
                    if nan_run_len > max_nan_samples:
                        bailed = True
                        break
                b += 1

            if bailed:
                # b is currently inside the offending NaN run. Advance to
                # the first valid sample past it (still inside this bout).
                while b < bout_end and not valid_mask[b]:
                    b += 1
                cursor = b
                continue

            if valid_count < epoch_samples:
                # Ran out of bout before hitting target — done with this bout.
                break

            # b is one past the last valid sample we counted; the window ends
            # on the previous index.
            b -= 1
            out.append((a, b))
            cursor = b + 1

    return out


def valid_state_epochs(
    hypno,
    mask: xr.DataArray,
    state: str,
    epoch_length: float,
    *,
    mode: str = "span",
    max_nan_span: float | None = None,
    min_bout_length: float = 0.0,
    n_epochs: int | str = "all",
    seed: int | None = None,
) -> pl.DataFrame:
    """NaN-aware tiling of fixed-length epochs from hypnogram bouts of a state.

    Parameters
    ----------
    hypno
        ``electro_py.hypno.Hypnogram`` with a float time axis.
    mask
        1-D boolean DataArray on the analysis sample grid.
    state
        State name to tile (single state — multi-state union is a separate
        primitive at :func:`valid_state_intervals`).
    epoch_length
        Target valid-time content per epoch, in seconds.
    mode
        Tiling policy:

        - ``'span'`` (default): each epoch contains exactly ``epoch_length``
          seconds of valid data. ``wall_duration ≥ epoch_length`` when
          interior NaN. With ``max_nan_span`` set, the cursor refuses to
          swallow any contiguous NaN run longer than that, abandoning the
          attempt and restarting past the offending run.
        - ``'strict'``: tile only within contiguous (state ∧ valid) runs.
          Every epoch has ``wall_duration == valid_duration == epoch_length``.
          Conceptually equivalent to ``mode='span', max_nan_span=0`` but
          kept as a separate mode for clarity and slightly faster path.
    max_nan_span
        Span-mode only. Maximum contiguous NaN run (seconds) the cursor
        will tolerate inside an epoch. ``None`` (default) → no limit
        (pure span). Ignored when ``mode='strict'``.
    min_bout_length
        Drop hypno bouts shorter than this from the underlying iteration.
        The check is on the bout's wall-clock duration (its
        ``end_time - start_time``), not its valid duration. Default 0
        (no filter).
    n_epochs
        ``'all'`` (default) returns every epoch found. An int caps the
        result to that many epochs, sampled uniformly at random across all
        bouts; this requires ``seed`` to be set.
    seed
        Random seed for the ``n_epochs=int`` subsample.

    Returns
    -------
    pl.DataFrame
        Columns: ``state`` (the input state, repeated), ``start_time``
        (f64, the time-coord value of the first sample of the epoch),
        ``end_time`` (f64, the time-coord value of the last *valid* sample
        of the epoch — see Notes), ``valid_duration`` (f64, always
        ``epoch_length``), ``wall_duration`` (f64, wall-clock seconds
        spanned by the epoch including interior NaN), ``bout_idx`` (i64,
        which hypno bout this epoch came from, by sort order in
        ``hypno.df``). Empty if no epochs can be cut.

    Notes
    -----
    ``end_time`` lands on the **last sample whose value contributed to the
    valid count**, not one ``dt`` past it. That means for adjacent
    spanning epochs (e.g. ``[a1, b1]`` and ``[a2, b2]`` with ``a2 = b1 +
    dt``), ``end_time(epoch_1) + dt == start_time(epoch_2)``. Compute
    ``wall_duration`` from the column or from the underlying time-coord
    values, not by subtracting ``end_time - start_time`` directly (that
    would be ``epoch_samples * dt - dt`` rather than the actual run
    length).

    Multi-bout iteration: the algorithm never spans across a state
    boundary in the hypnogram. Two NREM bouts with a Wake interruption
    are tiled independently; their internal NaN runs are independent.

    See also
    --------
    :func:`valid_state_intervals` — the primitive of "find runs of
    (state ∧ valid)" without epoch tiling.
    """
    if mode not in ("span", "strict"):
        raise ValueError(f"`mode` must be 'span' or 'strict', got {mode!r}.")
    if epoch_length <= 0:
        raise ValueError(f"`epoch_length` must be positive, got {epoch_length}.")
    if max_nan_span is not None and max_nan_span < 0:
        raise ValueError(
            f"`max_nan_span` must be >= 0 or None, got {max_nan_span}."
        )

    if isinstance(n_epochs, str) and n_epochs != "all":
        raise ValueError(f"`n_epochs` must be 'all' or int, got {n_epochs!r}.")
    if isinstance(n_epochs, int) and n_epochs < 0:
        raise ValueError(f"`n_epochs` int must be >= 0, got {n_epochs}.")
    if isinstance(n_epochs, int) and seed is None:
        raise ValueError(
            "`n_epochs=int` requires a `seed` to make the subsample reproducible."
        )

    state_mask, valid_mask, dt = _resolve_to_grid(hypno, mask, state)

    # Apply min_bout_length: zero out runs of state_mask that are shorter
    # than the threshold. This is the correct semantics — the mask we feed
    # to the tilers is "samples in a long-enough state bout".
    if min_bout_length > 0:
        min_bout_samples = max(int(round(min_bout_length / dt)), 1)
        runs = _runs_of_true(state_mask)
        filtered = state_mask.copy()
        for run_start, run_end in runs:
            if (run_end - run_start + 1) < min_bout_samples:
                filtered[run_start : run_end + 1] = False
        state_mask = filtered

    epoch_samples = max(int(round(epoch_length / dt)), 1)
    if mode == "strict":
        epochs = _tile_strict(state_mask, valid_mask, epoch_samples)
    else:
        if max_nan_span is None:
            max_nan_samples: int | float = float("inf")
        else:
            max_nan_samples = int(round(max_nan_span / dt))
        epochs = _tile_span(state_mask, valid_mask, epoch_samples, max_nan_samples)

    # Map each emitted epoch back to a bout_idx (the index of the bout
    # whose state run contains the epoch's first sample). This requires
    # the bouts to have been sorted by start_time, which Hypnogram
    # guarantees via its constructor.
    bout_starts = hypno.df["start_time"].to_numpy()
    bout_ends = hypno.df["end_time"].to_numpy()
    time_values = np.asarray(mask["time"].values, dtype=float)

    rows = []
    for a, b in epochs:
        a_t = float(time_values[a])
        b_t = float(time_values[b])
        # Find which bout contains a_t (binary search on starts).
        idx = int(np.searchsorted(bout_starts, a_t, side="right") - 1)
        if idx < 0 or idx >= len(bout_starts) or a_t > bout_ends[idx]:
            # Should not happen given how a was selected, but guard anyway.
            idx = -1
        rows.append(
            {
                "state": state,
                "start_time": a_t,
                "end_time": b_t,
                "valid_duration": float(epoch_samples) * dt,
                "wall_duration": float(b - a + 1) * dt,
                "bout_idx": idx,
            }
        )

    df = pl.DataFrame(
        rows,
        schema={
            "state": pl.String,
            "start_time": pl.Float64,
            "end_time": pl.Float64,
            "valid_duration": pl.Float64,
            "wall_duration": pl.Float64,
            "bout_idx": pl.Int64,
        },
    )

    if isinstance(n_epochs, int) and df.height > n_epochs:
        df = df.sample(n=n_epochs, seed=seed).sort("start_time")

    return df


def add_valid_duration(hypno, mask: xr.DataArray):
    """Add a per-bout ``valid_duration`` column to a hypnogram.

    For each bout, counts the mask samples whose time-coord falls inside
    the bout and whose validity is True, then multiplies by the mask's
    median sample spacing to get seconds.

    Sample-to-bout assignment uses the same boundary semantics as
    ``Hypnogram.covers_time``: a sample at time ``t`` belongs to bout
    ``i`` iff ``start_time[i] <= t <= end_time[i]``, with adjacent bouts
    (``end_time[i] == start_time[i+1]``) splitting the shared sample to
    neither side.

    Parameters
    ----------
    hypno
        ``electro_py.hypno.Hypnogram`` with a float time axis aligned to
        ``mask``'s time coord (TDT-block-relative seconds in this
        project).
    mask
        1-D boolean DataArray with a ``time`` coord. Typically the
        output of :func:`validity_mask` on a scopex DataArray.

    Returns
    -------
    Hypnogram
        New hypnogram (same class as input) with a ``valid_duration``
        (f64, seconds) column added. All other columns and bout ordering
        are preserved. ``valid_duration <= duration`` per bout; bouts
        falling entirely outside the mask's time coverage get 0.0.
    """
    if "time" not in mask.dims:
        raise ValueError("`mask` must have a 'time' dim.")

    time_values = np.asarray(mask["time"].values, dtype=float)
    dt = _median_dt(time_values)
    valid = np.asarray(mask.values, dtype=bool)

    starts = hypno.df["start_time"].to_numpy()
    ends = hypno.df["end_time"].to_numpy()

    start_idx = np.searchsorted(starts, time_values, side="right") - 1
    end_idx = np.searchsorted(ends, time_values, side="left")
    in_bout = (
        (start_idx == end_idx)
        & (start_idx >= 0)
        & (start_idx < len(starts))
    )

    eligible = in_bout & valid
    if eligible.any():
        valid_counts = np.bincount(
            start_idx[eligible], minlength=len(starts)
        )
    else:
        valid_counts = np.zeros(len(starts), dtype=np.int64)

    valid_durations = valid_counts.astype(np.float64) * dt

    new_df = hypno.df.with_columns(
        pl.Series("valid_duration", valid_durations)
    )
    return type(hypno)(new_df)


def merge_brief_breaking_bouts(hypno, *, max_break_s: float):
    """Iteratively merge brief "breaking" bouts in a hypnogram.

    A bout is a "breaking bout" if its duration is below ``max_break_s`` AND
    its immediate neighbours (the bout before and the bout after) are the
    same state. Merging absorbs the breaker into a single bout spanning the
    flanking pair, with state inherited from the flanks.

    Repeatedly applies the rule until no more merges are possible. With
    ``max_break_s == 0`` the hypnogram is returned unchanged.

    Use case: a ~2 s arousal interrupting two ~80 s NREM bouts should be
    treated as a single ~162 s NREM bout for within-bout-drift analyses.

    Parameters
    ----------
    hypno : electro_py.hypno.Hypnogram
        Source hypnogram. Must have ``state, start_time, end_time`` columns.
    max_break_s : float
        Maximum duration (seconds) of a "brief" bout that is eligible to
        be merged. Strict less-than comparison: a bout of exactly
        ``max_break_s`` is kept.

    Returns
    -------
    electro_py.hypno.Hypnogram
        New hypnogram of the same class as the input, with brief breakers
        merged. The first/last bouts are never merged (they have no flank
        on one side). Other columns are preserved from the leading flank
        of each merged group.
    """
    if max_break_s <= 0:
        return hypno

    df = hypno.df.sort("start_time")
    schema = df.schema
    while True:
        n = df.height
        if n < 3:
            break
        durs = (df["end_time"] - df["start_time"]).to_numpy()
        states = df["state"].to_list()
        merge_at = -1
        for i in range(1, n - 1):
            if durs[i] < max_break_s and states[i - 1] == states[i + 1]:
                merge_at = i
                break
        if merge_at == -1:
            break
        i = merge_at
        rows = df.to_dicts()
        merged = dict(rows[i - 1])  # inherit other columns from the leading flank
        merged["state"] = states[i - 1]
        merged["start_time"] = rows[i - 1]["start_time"]
        merged["end_time"] = rows[i + 1]["end_time"]
        new_rows = rows[: i - 1] + [merged] + rows[i + 2:]
        df = pl.DataFrame(new_rows, schema=schema)
    # Reconstruct a Hypnogram of the original class.
    return type(hypno)(df)
