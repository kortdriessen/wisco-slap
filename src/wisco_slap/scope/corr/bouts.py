"""Bout-list builders for state-conditioned correlation analysis.

Each function in this module returns a polars DataFrame with the same schema,
so downstream aggregators (:mod:`wisco_slap.scope.corr.aggregate`) and the
table builder (:mod:`wisco_slap.scope.corr.state_compare`) don't need to
branch on the bout-selection strategy.

Common output schema:

================  =======  =========================================================
column            dtype    meaning
================  =======  =========================================================
``state``         str      hypnogram state label (or ``"|".join(states)``)
``start_time``    f64      first sample of the bout (matches ``data['time']``)
``end_time``      f64      last sample of the bout (inclusive both ends)
``valid_duration``f64      seconds of valid (non-NaN) samples in the bout
``wall_duration`` f64      seconds spanned by the bout including interior NaN
``bout_idx``      i64      index of the source hypnogram bout (sorted by start)
================  =======  =========================================================

Four primitives are provided:

- :func:`fixed_valid_bouts` — equal valid-duration tiles (current notebook
  behaviour). Wraps
  :func:`wisco_slap.util.validity.hypno.valid_state_epochs`.
- :func:`all_segments_in_state` — variable-length bouts; one per maximal
  contiguous (state ∧ valid) run. Wraps
  :func:`wisco_slap.util.validity.hypno.valid_state_intervals`.
- :func:`state_hypno_bouts` — one bout per hypnogram state-bout, regardless
  of interior NaN. Use when you want the full wall-range of a state bout
  (e.g. for within-bout sliding-window analyses where the sliding window
  itself handles NaN gaps via ``min_valid_frac``).
- :func:`random_subsample_bouts` — uniformly subsample an existing bout
  DataFrame to a fixed count. Used to equalise across states.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xarray as xr

from ...util.validity.hypno import valid_state_epochs, valid_state_intervals


def fixed_valid_bouts(
    hypno,
    mask: xr.DataArray,
    state: str,
    epoch_length: float,
    *,
    min_bout_length: float | None = None,
    max_nan_span: float | None = 2.0,
    mode: str = "span",
    n_epochs: int | str = "all",
    seed: int | None = None,
) -> pl.DataFrame:
    """Equal valid-duration epochs from hypnogram bouts of a state.

    Thin wrapper around :func:`valid_state_epochs`. Each emitted bout contains
    exactly ``epoch_length`` seconds of valid (non-NaN) data; ``wall_duration``
    can exceed ``epoch_length`` when the cursor spanned interior NaN gaps
    (only allowed up to ``max_nan_span`` seconds at a time).

    Parameters
    ----------
    hypno : electro_py.hypno.Hypnogram
    mask : xr.DataArray
        1-D boolean validity mask on the analysis sample grid (use
        :func:`wisco_slap.util.validity.validity_mask`).
    state : str
        Single hypnogram state name.
    epoch_length : float
        Target valid-time content per epoch, in seconds.
    min_bout_length : float | None
        Drop hypno bouts shorter than this (seconds). Default
        ``epoch_length`` (a bout shorter than one epoch can never produce
        any output anyway).
    max_nan_span : float | None
        Max contiguous NaN gap the cursor will tolerate (seconds). Default
        2.0 (matches the existing notebook).
    mode : str
        ``"span"`` (default) or ``"strict"``. See
        :func:`valid_state_epochs`.
    n_epochs : int | str
        ``"all"`` (default) or an integer cap. ``int`` requires ``seed``.
    seed : int | None
        Required when ``n_epochs`` is an integer.

    Returns
    -------
    pl.DataFrame
        Bouts in the standard schema. ``valid_duration == epoch_length`` for
        every row.
    """
    if min_bout_length is None:
        min_bout_length = epoch_length
    return valid_state_epochs(
        hypno,
        mask,
        state,
        epoch_length,
        mode=mode,
        max_nan_span=max_nan_span,
        min_bout_length=min_bout_length,
        n_epochs=n_epochs,
        seed=seed,
    )


def all_segments_in_state(
    hypno,
    mask: xr.DataArray,
    state: str,
    *,
    min_bout_length: float = 0.0,
) -> pl.DataFrame:
    """Variable-length bouts: one per maximal contiguous (state ∧ valid) run.

    Wraps :func:`valid_state_intervals` and re-shapes the output to match the
    bouts schema used elsewhere in this module.

    Parameters
    ----------
    hypno : electro_py.hypno.Hypnogram
    mask : xr.DataArray
        1-D boolean validity mask on the analysis sample grid.
    state : str
        Single hypnogram state name.
    min_bout_length : float
        Drop runs shorter than this (seconds). Default 0 = keep all.

    Returns
    -------
    pl.DataFrame
        Bouts in the standard schema. ``valid_duration == wall_duration``
        for every row (segments are by construction NaN-free).
    """
    raw = valid_state_intervals(hypno, mask, state)
    if raw.height == 0:
        return pl.DataFrame(
            schema={
                "state": pl.String,
                "start_time": pl.Float64,
                "end_time": pl.Float64,
                "valid_duration": pl.Float64,
                "wall_duration": pl.Float64,
                "bout_idx": pl.Int64,
            }
        )

    # Map each segment to the underlying hypnogram bout it lives in.
    bout_starts = hypno.df["start_time"].to_numpy()
    bout_ends = hypno.df["end_time"].to_numpy()
    seg_starts = raw["start_time"].to_numpy()
    bout_idx = []
    for s in seg_starts:
        idx = int(np.searchsorted(bout_starts, s, side="right") - 1)
        if idx < 0 or idx >= len(bout_starts) or s > bout_ends[idx]:
            idx = -1
        bout_idx.append(idx)

    out = (
        raw.with_columns(
            pl.col("duration").alias("valid_duration"),
            pl.col("duration").alias("wall_duration"),
            pl.Series("bout_idx", bout_idx, dtype=pl.Int64),
        )
        .drop("duration")
        .select(
            "state",
            "start_time",
            "end_time",
            "valid_duration",
            "wall_duration",
            "bout_idx",
        )
    )

    if min_bout_length > 0:
        out = out.filter(pl.col("valid_duration") >= min_bout_length)

    return out


def state_hypno_bouts(
    hypno,
    mask: xr.DataArray,
    state: str,
    *,
    min_bout_length: float = 0.0,
) -> pl.DataFrame:
    """One bout per hypnogram state-bout, regardless of interior NaN gaps.

    Returns the original hypnogram's state-bouts (filtered to ``state``) as
    bouts in the standard schema. ``valid_duration`` is computed from the
    mask (sum of valid samples × dt) and ``wall_duration`` is the bout's
    wall extent. Use this when you want the full wall range of a state bout
    and plan to handle NaN tolerance downstream (e.g. via
    ``min_valid_frac`` in :func:`sliding_window_corr_in_bout`).

    Parameters
    ----------
    hypno : electro_py.hypno.Hypnogram
    mask : xr.DataArray
        1-D boolean validity mask on the analysis sample grid. Used only to
        compute ``valid_duration`` per bout.
    state : str
    min_bout_length : float
        Drop bouts whose ``wall_duration`` is below this. Default 0.

    Returns
    -------
    pl.DataFrame
        Bouts in the standard schema, sorted by ``start_time``.
    """
    df = hypno.df.filter(pl.col("state") == state).sort("start_time")
    if df.height == 0:
        return pl.DataFrame(
            schema={
                "state": pl.String,
                "start_time": pl.Float64,
                "end_time": pl.Float64,
                "valid_duration": pl.Float64,
                "wall_duration": pl.Float64,
                "bout_idx": pl.Int64,
            }
        )

    times = np.asarray(mask["time"].values, dtype=np.float64)
    valid = np.asarray(mask.values, dtype=bool)
    if times.size >= 2:
        dt = float(np.median(np.diff(times)))
    else:
        dt = 0.0

    starts = df["start_time"].to_numpy()
    ends = df["end_time"].to_numpy()
    bout_starts = np.asarray(hypno.df["start_time"].to_numpy())
    valid_durs = np.zeros(df.height, dtype=np.float64)
    wall_durs = np.zeros(df.height, dtype=np.float64)
    bout_idx = np.zeros(df.height, dtype=np.int64)
    for k, (s, e) in enumerate(zip(starts, ends)):
        wall = float(e - s)
        wall_durs[k] = wall
        if dt > 0 and times.size:
            in_window = (times >= s) & (times <= e)
            valid_durs[k] = float(valid[in_window].sum()) * dt
        else:
            valid_durs[k] = wall
        # find this bout's index in the original hypno
        idx = int(np.searchsorted(bout_starts, s, side="right") - 1)
        if idx < 0 or idx >= len(bout_starts) or s > hypno.df["end_time"].to_numpy()[idx]:
            idx = -1
        bout_idx[k] = idx

    out = pl.DataFrame({
        "state": df["state"],
        "start_time": pl.Series("start_time", starts, dtype=pl.Float64),
        "end_time": pl.Series("end_time", ends, dtype=pl.Float64),
        "valid_duration": pl.Series("valid_duration", valid_durs, dtype=pl.Float64),
        "wall_duration": pl.Series("wall_duration", wall_durs, dtype=pl.Float64),
        "bout_idx": pl.Series("bout_idx", bout_idx, dtype=pl.Int64),
    })
    if min_bout_length > 0:
        out = out.filter(pl.col("wall_duration") >= min_bout_length)
    return out


def random_subsample_bouts(
    bouts: pl.DataFrame,
    n: int,
    *,
    seed: int,
) -> pl.DataFrame:
    """Uniformly subsample ``n`` rows from a bouts DataFrame.

    Used to equalise bout counts across states. Returns the input unchanged if
    ``bouts.height <= n``.

    Parameters
    ----------
    bouts : pl.DataFrame
        Bouts in the standard schema.
    n : int
        Target row count.
    seed : int
        Required for reproducibility.

    Returns
    -------
    pl.DataFrame
        Subsampled bouts, sorted by ``start_time``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if bouts.height <= n:
        return bouts.sort("start_time")
    return bouts.sample(n=n, seed=seed).sort("start_time")
