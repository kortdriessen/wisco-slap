"""Rate per fixed-valid-duration epoch.

The confound-controlled layer of :mod:`wisco_slap.scope.rates`. Cuts each
state's bouts into epochs that all hold exactly the same amount of *valid*
data, so per-epoch rates are directly comparable across states and bouts
without a bout-length or dead-time confound. The rate analogue of
:func:`wisco_slap.scope.corr.fixed_valid_bouts`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl
import xarray as xr

from ...util.validity.events import _resolve_states
from ...util.validity.hypno import valid_state_epochs
from ...util.validity.mask import resolve_mask
from .core import _order_columns, _prepare, _rate_over_partition


def compute_rate_by_epoch(
    peaks_df: pl.DataFrame,
    hypno,
    final_peaks: xr.DataArray,
    epoch_length_s: float,
    *,
    mask: xr.DataArray | None = None,
    states: str | Iterable[str] = "all",
    mode: str = "span",
    max_nan_span: float | None = None,
    min_bout_length: float = 0.0,
    complete_grid: bool = True,
    time_col: str = "peak_time",
) -> pl.DataFrame:
    """Rate per fixed-valid-duration epoch, tiled NaN-aware from each state.

    Wraps :func:`~wisco_slap.util.validity.valid_state_epochs` to cut each
    state's bouts into epochs that each hold exactly ``epoch_length_s`` of
    valid data, then counts valid events per (synapse, epoch). Because every
    epoch has the same valid denominator, epoch rates are directly comparable
    across states and bouts without a bout-length or dead-time confound —
    the rigorous unit for state comparisons.

    Parameters
    ----------
    peaks_df, hypno, final_peaks, mask, states, complete_grid, time_col
        As in :func:`~wisco_slap.scope.rates.compute_rate`.
    epoch_length_s : float
        Valid seconds per epoch.
    mode, max_nan_span, min_bout_length
        Forwarded to :func:`~wisco_slap.util.validity.valid_state_epochs`.
        ``mode='span'`` (default) lets an epoch span interior NaN so its valid
        content is exactly ``epoch_length_s``; ``mode='strict'`` tiles only
        within contiguous (state ∧ valid) runs.

    Returns
    -------
    pl.DataFrame
        One row per (synapse, epoch). Columns: per-synapse metadata; then
        ``state``, ``epoch_idx`` (0-indexed within the whole result, i64),
        ``bout_idx`` (which hypno bout the epoch came from), ``start_time`` /
        ``end_time`` (f64), ``wall_duration_s``; then ``n_events``,
        ``valid_duration_s`` (≈ ``epoch_length_s``), ``rate_hz``.
    """
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

    resolved_mask = resolve_mask(final_peaks, mask)
    requested = _resolve_states(hypno, states)

    bin_of_sample = np.full(n_samples, -1, dtype=np.int64)
    descriptor_rows: list[dict] = []
    bin_id = 0
    for s in requested:
        epochs = valid_state_epochs(
            hypno, resolved_mask, s, epoch_length_s,
            mode=mode, max_nan_span=max_nan_span, min_bout_length=min_bout_length,
        )
        for ep in epochs.iter_rows(named=True):
            # Samples in [start_time, end_time]; validity is applied in the
            # denominator so a spanning epoch's interior NaN doesn't count.
            i0 = int(np.searchsorted(time_values, ep["start_time"], side="left"))
            i1 = int(np.searchsorted(time_values, ep["end_time"], side="right"))
            bin_of_sample[i0:i1] = bin_id
            descriptor_rows.append(
                {
                    "_bin_id": bin_id,
                    "state": s,
                    "epoch_idx": bin_id,
                    "bout_idx": ep["bout_idx"],
                    "start_time": ep["start_time"],
                    "end_time": ep["end_time"],
                    "wall_duration_s": ep["wall_duration"],
                }
            )
            bin_id += 1

    if not descriptor_rows:
        descriptor = pl.DataFrame(
            schema={"_bin_id": pl.Int64, "state": pl.String, "epoch_idx": pl.Int64,
                    "bout_idx": pl.Int64, "start_time": pl.Float64,
                    "end_time": pl.Float64, "wall_duration_s": pl.Float64}
        )
    else:
        descriptor = pl.DataFrame(descriptor_rows)

    out = _rate_over_partition(
        peaks_df, syn_key, pk_idx, pk_valid, bin_of_sample, valid_bool,
        dt, bin_id, descriptor, syn_meta, complete_grid,
    )
    return _order_columns(out, syn_meta.columns)
