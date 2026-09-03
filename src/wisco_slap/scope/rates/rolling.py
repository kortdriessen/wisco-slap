"""Sliding-window event rate over time.

The time-course layer of :mod:`wisco_slap.scope.rates`: instead of pooling by
hypnogram state or bout, tile the recording into fixed-length windows and
compute a rate in each. Feeds rate-vs-time plots.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xarray as xr

from .core import _assign_samples_to_bouts, _order_columns, _prepare, _rate_expr


def compute_rate_rolling(
    peaks_df: pl.DataFrame,
    hypno,
    final_peaks: xr.DataArray,
    window_s: float,
    step_s: float | None = None,
    *,
    mask: xr.DataArray | None = None,
    min_valid_frac: float = 0.0,
    label_state: bool = True,
    time_col: str = "peak_time",
) -> pl.DataFrame:
    """Sliding-window event rate over time, per synapse.

    Tiles the acquisition into windows of ``window_s`` seconds stepped by
    ``step_s`` and, for each window and synapse, counts valid events and
    divides by the window's *valid* seconds. Ideal for time-course plots of
    rate, optionally coloured by the hypnogram state at each window's centre.

    Parameters
    ----------
    peaks_df, hypno, final_peaks, mask, time_col
        As in :func:`~wisco_slap.scope.rates.compute_rate`. The hypnogram is
        only used (when ``label_state``) to tag each window with the state at
        its centre.
    window_s : float
        Window length in wall-clock seconds. Its valid content may be less
        when the window straddles NaN gaps — the denominator reflects that.
    step_s : float, optional
        Step between consecutive window starts. Defaults to ``window_s``
        (contiguous, non-overlapping tiling). Use a fraction of ``window_s``
        for a smooth overlapping trace.
    min_valid_frac : float
        Drop windows whose valid fraction (``valid_duration_s / window_s``) is
        below this. Default 0 keeps every window with any valid data.
    label_state : bool
        If True (default) add a ``state`` column: the hypnogram state covering
        each window's centre sample (``None`` if the centre is unscored).

    Returns
    -------
    pl.DataFrame
        One row per (synapse, window). Columns: per-synapse metadata,
        ``window_idx`` (i64), ``center_time`` / ``start_time`` / ``end_time``
        (f64, seconds), ``wall_duration_s``, optional ``state``, then
        ``n_events``, ``valid_duration_s``, ``rate_hz``.

    Notes
    -----
    Windows are cut on sample-index boundaries (``round(window_s / dt)``
    samples), so every window has the same wall duration. A trailing stub
    shorter than one window is dropped.
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

    w_samp = max(1, int(round(window_s / dt)))
    s_samp = w_samp if step_s is None else max(1, int(round(step_s / dt)))
    if w_samp > n_samples:
        raise ValueError(
            f"window_s ({window_s}) is longer than the recording "
            f"({n_samples * dt:.1f}s)."
        )

    starts_i = np.arange(0, n_samples - w_samp + 1, s_samp, dtype=np.int64)
    ends_i = starts_i + w_samp  # half-open [start, end)
    n_win = starts_i.size

    # Valid seconds per window via a prefix sum over the validity mask.
    cum = np.concatenate([[0], np.cumsum(valid_bool)])
    valid_secs = (cum[ends_i] - cum[starts_i]).astype(np.float64) * dt
    center_idx = starts_i + w_samp // 2

    win = pl.DataFrame(
        {
            "window_idx": np.arange(n_win, dtype=np.int64),
            "center_time": time_values[center_idx],
            "start_time": time_values[starts_i],
            "end_time": time_values[ends_i - 1],
            "wall_duration_s": np.full(n_win, w_samp * dt),
            "valid_duration_s": valid_secs,
        }
    )
    if label_state:
        bout_of_sample = _assign_samples_to_bouts(time_values, hypno)
        bout_states = hypno.df["state"].to_list()
        center_state = [
            bout_states[b] if b >= 0 else None for b in bout_of_sample[center_idx]
        ]
        win = win.with_columns(pl.Series("state", center_state, dtype=pl.String))

    # Numerator: for each synapse, count its valid peaks in each window.
    valid_peaks = (
        peaks_df.select(syn_key, "peak_idx")
        .with_columns(_valid=pl.Series("_valid", pk_valid))
        .filter(pl.col("_valid"))
    )
    syn_order = syn_meta[syn_key].to_list()
    row_of_syn = {s: r for r, s in enumerate(syn_order)}
    counts_mat = np.zeros((len(syn_order), n_win), dtype=np.int64)
    for srow in (
        valid_peaks.group_by(syn_key).agg(pl.col("peak_idx").sort())
    ).iter_rows(named=True):
        r = row_of_syn.get(srow[syn_key])
        if r is None:
            continue
        idxs = np.asarray(srow["peak_idx"], dtype=np.int64)
        counts_mat[r] = np.searchsorted(idxs, ends_i, side="left") - np.searchsorted(
            idxs, starts_i, side="left"
        )

    long = pl.DataFrame(
        {
            syn_key: np.repeat(np.asarray(syn_order), n_win),
            "window_idx": np.tile(np.arange(n_win, dtype=np.int64), len(syn_order)),
            "n_events": counts_mat.reshape(-1),
        }
    )
    out = (
        syn_meta.join(long, on=syn_key, how="left")
        .join(win, on="window_idx", how="left")
        .with_columns(_rate_expr())
    )
    if min_valid_frac > 0:
        out = out.filter(
            pl.col("valid_duration_s") >= min_valid_frac * w_samp * dt
        )
    return _order_columns(out, syn_meta.columns)
