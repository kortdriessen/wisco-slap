"""Time-evolution analyses for state-conditioned correlations.

How do per-pair correlations change over the course of a recording? Two
complementary views:

- :func:`split_bouts_into_periods` — partition a recording's bouts into N
  equal-duration time chunks (e.g. early/middle/late thirds of the recording)
  and label them. Use these labels with
  :func:`build_state_corr_table_periods`.
- :func:`build_state_corr_table_periods` — build the long-form correlation
  table separately within each period of each recording, so each pair has
  one (state × period) row.
- :func:`bout_synchrony_over_time` — sliding-window per-bout off-diagonal
  mean r over time. The continuous view that
  :func:`wisco_slap.scope.corr.bout_level_synchrony` already supports —
  this module just adds plotting/aggregation conveniences.

A typical time-evolution workflow:

  1. Build period-labelled tables for each acq.
  2. Compare period-1 NREM r to period-3 NREM r per pair (drift within state).
  3. Compare Δz (NREM − Wake) early vs late (does the state effect change?).
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import polars as pl
import xarray as xr

from ... import get as _get
from ...util.validity.mask import validity_mask
from .aggregate import (
    aggregate_fisher_z,
    aggregate_pooled_sums,
    aggregate_simple_mean,
)
from .bouts import all_segments_in_state, fixed_valid_bouts, random_subsample_bouts
from .state_compare import (
    _AGGREGATORS,
    _BOUT_BUILDERS,
    _DEFAULT_ALL_SEGMENT_KWARGS,
    _DEFAULT_FIXED_BOUT_KWARGS,
    _classify_pair,
    _coord_to_float_array,
    _coord_to_int_array,
    _coord_to_str_array,
    _flatten_upper,
    _load_dn,
)


def split_bouts_into_periods(
    bouts: pl.DataFrame,
    n_periods: int = 3,
    *,
    period_labels: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Add a ``period`` column partitioning bouts by their start time.

    Splits the time range covered by the bouts into ``n_periods`` equal-duration
    chunks and labels each bout by which chunk its midpoint falls in.

    Parameters
    ----------
    bouts : pl.DataFrame
        Bouts in the standard schema (``start_time``, ``end_time``, ...).
    n_periods : int
        How many equal-duration time periods to split into. Default 3
        (early / middle / late thirds).
    period_labels : iterable of str, optional
        Labels to use for each period. Default: ``"P1", "P2", ..., "P<n>"``
        if ``n_periods != 3``, or ``"early", "middle", "late"`` for
        ``n_periods == 3``.

    Returns
    -------
    pl.DataFrame
        Same as input plus a ``period`` (string) column.
    """
    if period_labels is None:
        if n_periods == 3:
            period_labels = ("early", "middle", "late")
        else:
            period_labels = tuple(f"P{i + 1}" for i in range(n_periods))
    else:
        period_labels = tuple(period_labels)
    if len(period_labels) != n_periods:
        raise ValueError(
            f"period_labels has length {len(period_labels)}, expected {n_periods}"
        )
    if bouts.height == 0:
        return bouts.with_columns(pl.lit(None).cast(pl.String).alias("period"))

    t_start = float(bouts["start_time"].min())
    t_end = float(bouts["end_time"].max())
    if t_end <= t_start:
        return bouts.with_columns(pl.lit(period_labels[0]).alias("period"))

    edges = np.linspace(t_start, t_end, n_periods + 1)
    midpoints = (
        (bouts["start_time"].to_numpy() + bouts["end_time"].to_numpy()) / 2
    )
    period_idx = np.clip(
        np.searchsorted(edges[1:], midpoints, side="right"),
        0,
        n_periods - 1,
    )
    period_str = np.array(period_labels)[period_idx]
    return bouts.with_columns(pl.Series("period", period_str, dtype=pl.String))


def build_state_corr_table_periods(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str | list[str],
    *,
    n_periods: int = 3,
    period_labels: Iterable[str] | None = None,
    channel: int = 0,
    trace: str = "denoised",
    bout_strategy: str = "fixed_valid",
    aggregation: str = "fisher_z_weighted",
    bout_kwargs: dict | None = None,
    states: tuple[str, ...] = ("NREM", "Wake"),
    validity_mode: str = "all",
) -> pl.DataFrame:
    """Like :func:`build_state_corr_table` but split into time periods.

    For each (state, period) pair, computes a separate per-pair correlation
    aggregate from the bouts that fall into that period. Output schema is the
    same as :func:`build_state_corr_table` plus a ``period`` column.

    Periods partition the wall-clock duration of the recording into
    ``n_periods`` equal-duration chunks (default 3 = early/middle/late).

    Parameters
    ----------
    subject, exp, loc, acq, soma_id
        See :func:`build_state_corr_table`.
    n_periods, period_labels
        Forwarded to :func:`split_bouts_into_periods`.
    channel, trace, bout_strategy, aggregation, bout_kwargs, states, validity_mode
        See :func:`build_state_corr_table`.

    Returns
    -------
    pl.DataFrame
        One row per (pair, state, period). Note: ``equalise_states`` is not
        supported here — equalisation across both states and periods would
        confound the analysis. ``aggregation`` defaults to
        ``"fisher_z_weighted"`` which handles unequal n correctly anyway.
    """
    if bout_strategy not in _BOUT_BUILDERS:
        raise ValueError(f"bout_strategy must be one of {list(_BOUT_BUILDERS)}")
    if aggregation not in _AGGREGATORS:
        raise ValueError(f"aggregation must be one of {list(_AGGREGATORS)}")

    recording_id = f"{subject}|{exp}|{loc}|{acq}"
    dn = _load_dn(subject, exp, loc, acq, soma_id, channel=channel, trace=trace)
    if dn.sizes.get("syn_id", 0) < 2:
        raise ValueError("Need ≥2 synapses to correlate.")

    val_mask = validity_mask(dn, mode=validity_mode)
    hypno, *_ = _get.acq_sleep_coverage(subject, exp, loc, acq)

    builder = _BOUT_BUILDERS[bout_strategy]
    aggregator = _AGGREGATORS[aggregation]

    if bout_strategy == "fixed_valid":
        eff_bout_kwargs = {**_DEFAULT_FIXED_BOUT_KWARGS, **(bout_kwargs or {})}
    else:
        eff_bout_kwargs = {**_DEFAULT_ALL_SEGMENT_KWARGS, **(bout_kwargs or {})}

    n_syn = dn.sizes["syn_id"]
    syn_ids = np.asarray(dn["syn_id"].values, dtype=str)
    dmds = _coord_to_int_array(dn, "dmd", n_syn)
    dends = _coord_to_str_array(dn, "dend-ID", n_syn)
    dend_types = _coord_to_str_array(dn, "dend_type", n_syn)
    somas = _coord_to_str_array(dn, "soma-ID", n_syn)
    poses = _coord_to_float_array(dn, "pos", n_syn)

    # Build bouts per state once and label each row with its period (based on
    # the recording's wall-clock duration, not the state's bouts alone, so
    # NREM-period-1 and Wake-period-1 cover the same time slice).
    all_bouts: dict[str, pl.DataFrame] = {}
    for state in states:
        all_bouts[state] = builder(hypno, val_mask, state, **eff_bout_kwargs)

    # Use the FULL recording duration as the partition range so periods
    # match across states.
    rec_start = float(np.asarray(val_mask["time"].values, dtype=float).min())
    rec_end = float(np.asarray(val_mask["time"].values, dtype=float).max())
    period_labels_resolved: tuple[str, ...]
    if period_labels is None:
        period_labels_resolved = (
            ("early", "middle", "late") if n_periods == 3
            else tuple(f"P{i + 1}" for i in range(n_periods))
        )
    else:
        period_labels_resolved = tuple(period_labels)
    edges = np.linspace(rec_start, rec_end, n_periods + 1)

    def _label_bout(b: pl.DataFrame) -> pl.DataFrame:
        if b.height == 0:
            return b.with_columns(pl.lit(None).cast(pl.String).alias("period"))
        midpoints = (b["start_time"].to_numpy() + b["end_time"].to_numpy()) / 2
        idx = np.clip(np.searchsorted(edges[1:], midpoints, side="right"),
                      0, n_periods - 1)
        return b.with_columns(
            pl.Series("period", np.array(period_labels_resolved)[idx], dtype=pl.String)
        )

    rows: list[pl.DataFrame] = []
    for state, bouts in all_bouts.items():
        bouts = _label_bout(bouts)
        for period in period_labels_resolved:
            sub = bouts.filter(pl.col("period") == period)
            if sub.height == 0:
                continue
            r_xr, n_xr, z_xr, se_z_xr = aggregator(dn, sub)
            flat = _flatten_upper(
                r_xr, n_xr, z_xr, se_z_xr,
                syn_ids=syn_ids, dmds=dmds, dends=dends,
                dend_types=dend_types, somas=somas, poses=poses,
            )
            n_pairs = len(flat["syn_i"])

            pair_type = np.empty(n_pairs, dtype=object)
            same_type = np.empty(n_pairs, dtype=object)
            dend_pair = np.empty(n_pairs, dtype=object)
            for k in range(n_pairs):
                pt, st = _classify_pair(
                    flat["dend_i"][k], flat["dend_j"][k],
                    flat["dend_type_i"][k], flat["dend_type_j"][k],
                )
                pair_type[k] = pt
                same_type[k] = st
                dend_pair[k] = "|".join(sorted([
                    str(flat["dend_i"][k]), str(flat["dend_j"][k])
                ]))

            soma_i_arr = flat["soma_i"]
            soma_j_arr = flat["soma_j"]
            cell_id_i_arr = np.asarray(
                [f"{recording_id}|{s}" for s in soma_i_arr], dtype=object
            )
            cell_id_j_arr = np.asarray(
                [f"{recording_id}|{s}" for s in soma_j_arr], dtype=object
            )
            same_soma_arr = soma_i_arr == soma_j_arr
            cell_id_arr = np.where(same_soma_arr, cell_id_i_arr, recording_id)
            soma_id_arr = np.where(
                same_soma_arr,
                soma_i_arr,
                np.array(
                    [
                        "+".join(sorted([str(a), str(b)]))
                        for a, b in zip(soma_i_arr, soma_j_arr)
                    ],
                    dtype=object,
                ),
            )
            soma_type_class = np.array(
                [
                    f"{'same_soma' if same_soma_arr[k] else 'cross_soma'}_"
                    f"{same_type[k]}"
                    for k in range(n_pairs)
                ],
                dtype=object,
            )

            df = pl.DataFrame(
                {
                    "subject": [subject] * n_pairs,
                    "exp": [exp] * n_pairs,
                    "loc": [loc] * n_pairs,
                    "acq": [acq] * n_pairs,
                    "soma_id": soma_id_arr.astype(str),
                    "cell_id": cell_id_arr.astype(str),
                    "recording_id": [recording_id] * n_pairs,
                    "syn_i": flat["syn_i"],
                    "syn_j": flat["syn_j"],
                    "soma_i": soma_i_arr.astype(str),
                    "soma_j": soma_j_arr.astype(str),
                    "cell_id_i": cell_id_i_arr.astype(str),
                    "cell_id_j": cell_id_j_arr.astype(str),
                    "same_soma": same_soma_arr.astype(bool),
                    "state": [state] * n_pairs,
                    "period": [period] * n_pairs,
                    "r": flat["r"],
                    "z": flat["z"],
                    "se_z": flat["se_z"],
                    "n_samples": flat["n_samples"],
                    "n_bouts": [sub.height] * n_pairs,
                    "dmd_i": flat["dmd_i"],
                    "dmd_j": flat["dmd_j"],
                    "dend_i": flat["dend_i"],
                    "dend_j": flat["dend_j"],
                    "dend_type_i": flat["dend_type_i"],
                    "dend_type_j": flat["dend_type_j"],
                    "pair_type": pair_type.astype(str),
                    "same_type": same_type.astype(str),
                    "soma_type_class": soma_type_class.astype(str),
                    "dend_pair": dend_pair.astype(str),
                    "bout_strategy": [bout_strategy] * n_pairs,
                    "aggregation": [aggregation] * n_pairs,
                }
            )
            rows.append(df)

    if not rows:
        return pl.DataFrame()
    return pl.concat(rows, how="vertical")


def build_state_corr_table_periods_multi(
    rows: pl.DataFrame | Iterable[dict],
    **build_kwargs,
) -> pl.DataFrame:
    """Run :func:`build_state_corr_table_periods` over multiple inputs."""
    if isinstance(rows, pl.DataFrame):
        iterator = list(rows.iter_rows(named=True))
    else:
        iterator = list(rows)
    tables: list[pl.DataFrame] = []
    for row in iterator:
        try:
            df = build_state_corr_table_periods(
                row["subject"], row["exp"], row["loc"], row["acq"], row["soma_id"],
                **build_kwargs,
            )
            if df.height > 0:
                tables.append(df)
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"build_state_corr_table_periods failed for "
                f"{row.get('subject')}/{row.get('exp')}/{row.get('loc')}/"
                f"{row.get('acq')}/{row.get('soma_id')}: "
                f"{type(e).__name__}: {e}",
                stacklevel=2,
            )
    if not tables:
        return pl.DataFrame()
    return pl.concat(tables, how="vertical")
