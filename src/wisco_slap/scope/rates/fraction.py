"""Rate in the first / middle / last N-th of each state or bout.

The within-container time-evolution layer of :mod:`wisco_slap.scope.rates`.
Splits each container (a whole state's valid timeline, or a single bout's) into
equal-*valid*-time bins so early/middle/late comparisons aren't skewed by NaN
gaps. The rate analogue of
:func:`wisco_slap.scope.corr.split_bouts_into_periods`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl
import xarray as xr

from ...util.validity.events import _resolve_states
from .core import (
    _assign_samples_to_bouts,
    _order_columns,
    _prepare,
    _rate_over_partition,
)


def _frac_label(k: int, n: int) -> str:
    """Human-readable name for fraction bin ``k`` of ``n``."""
    if n == 2:
        return ("early", "late")[k]
    if n == 3:
        return ("early", "middle", "late")[k]
    return f"bin_{k + 1}of{n}"


def compute_rate_by_fraction(
    peaks_df: pl.DataFrame,
    hypno,
    final_peaks: xr.DataArray,
    n_bins: int = 3,
    *,
    within: str = "state",
    mask: xr.DataArray | None = None,
    states: str | Iterable[str] = "all",
    complete_grid: bool = True,
    time_col: str = "peak_time",
) -> pl.DataFrame:
    """Rate in the first / middle / last N-th of each state or bout.

    Each container (a whole state's valid timeline, or a single bout's) is
    split into ``n_bins`` consecutive bins holding an **equal number of valid
    samples** — so the split is by *valid* time, not wall-clock time, and a
    NaN gap in the middle of a bout doesn't skew the thirds. ``n_bins=3`` gives
    the classic early/middle/late breakdown; ``n_bins=2`` gives early/late.

    Parameters
    ----------
    peaks_df, hypno, final_peaks, mask, states, complete_grid, time_col
        As in :func:`~wisco_slap.scope.rates.compute_rate`.
    n_bins : int
        Number of equal-valid-time bins per container (>= 2). Default 3.
    within : {'state', 'bout'}
        Split each state's pooled valid timeline (``'state'``), or split each
        individual bout (``'bout'``).

    Returns
    -------
    pl.DataFrame
        One row per (synapse, container, fraction bin). Columns: per-synapse
        metadata; then ``state`` and (for ``within='bout'``) ``bout_idx`` /
        ``start_time`` / ``end_time``; then ``frac_bin`` (0-indexed i64),
        ``frac_label`` (``early``/``middle``/``late`` for ``n_bins<=3``);
        then ``n_events``, ``valid_duration_s``, ``rate_hz``.

    Notes
    -----
    A container with fewer than ``n_bins`` valid samples still yields
    ``n_bins`` rows; the empty trailing bins get ``valid_duration_s = 0`` and
    ``rate_hz = NaN``.
    """
    if within not in ("state", "bout"):
        raise ValueError(f"`within` must be 'state' or 'bout', got {within!r}.")
    if n_bins < 2:
        raise ValueError(f"`n_bins` must be >= 2, got {n_bins}.")

    (
        syn_key,
        syn_meta,
        time_values,
        dt,
        n_samples,
        valid_bool,
        pk_idx,
        pk_valid,
    ) = _prepare(peaks_df, hypno, final_peaks, mask, time_col)

    bout_of_sample = _assign_samples_to_bouts(time_values, hypno)
    bout_states = hypno.df["state"].to_list()
    bout_starts = hypno.df["start_time"].to_list()
    bout_ends = hypno.df["end_time"].to_list()
    requested = _resolve_states(hypno, states)
    req_set = set(requested)

    bin_of_sample = np.full(n_samples, -1, dtype=np.int64)
    descriptor_rows: list[dict] = []
    bin_id = 0

    def _emit_container(elig_idx: np.ndarray, identity: dict) -> None:
        """Split one container's eligible (state ∧ valid) samples into n_bins."""
        nonlocal bin_id
        n = elig_idx.size
        edges = np.linspace(0, n, n_bins + 1)
        edges = np.round(edges).astype(int)
        for k in range(n_bins):
            seg = elig_idx[edges[k]:edges[k + 1]]
            if seg.size:
                bin_of_sample[seg] = bin_id
            row = {"_bin_id": bin_id, **identity,
                   "frac_bin": k, "frac_label": _frac_label(k, n_bins)}
            descriptor_rows.append(row)
            bin_id += 1

    if within == "state":
        for s in requested:
            state_mask = hypno.mask_times_by_state(time_values, [s])
            elig_idx = np.flatnonzero(state_mask & valid_bool)
            _emit_container(elig_idx, {"state": s})
    else:  # within == "bout"
        for b in range(hypno.df.height):
            if bout_states[b] not in req_set:
                continue
            elig_idx = np.flatnonzero((bout_of_sample == b) & valid_bool)
            _emit_container(
                elig_idx,
                {
                    "bout_idx": b,
                    "state": bout_states[b],
                    "start_time": bout_starts[b],
                    "end_time": bout_ends[b],
                },
            )

    if not descriptor_rows:
        # No qualifying containers → empty frame with a sensible schema.
        descriptor = pl.DataFrame(
            schema={"_bin_id": pl.Int64, "state": pl.String,
                    "frac_bin": pl.Int64, "frac_label": pl.String}
        )
    else:
        descriptor = pl.DataFrame(descriptor_rows)

    out = _rate_over_partition(
        peaks_df, syn_key, pk_idx, pk_valid, bin_of_sample, valid_bool,
        dt, bin_id, descriptor, syn_meta, complete_grid,
    )
    return _order_columns(out, syn_meta.columns)
