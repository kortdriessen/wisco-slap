"""Temporal-dynamics analyses for state-conditioned correlations.

The complement to :mod:`wisco_slap.scope.corr.time_evolution`. Where
``time_evolution`` partitions a recording into a small number of equal-duration
periods, this module asks finer-grained temporal questions:

- **Bout temporal context**: where does this bout sit in the per-state sequence?
  Cumulative valid time spent in the state up to this point. Gap to the
  previous same-state bout. (:func:`annotate_bout_temporal_context`)
- **Within-bout drift**: inside a single long-ish bout, do pairwise
  correlations rise, fall, or stay flat? Sliding-window pairwise correlation
  across one bout. (:func:`sliding_window_corr_in_bout`,
  :func:`within_bout_correlation_timeline`)
- **State-clock dependence**: as the animal accumulates time in a state across
  many bouts, does the correlation level drift with cumulative state-time?
  Cumulative-time bins span across bout boundaries. (:func:`state_clock_table`)
- **State-onset transients**: align bouts on a state-onset transition and
  compute mean correlation as a function of relative time around the
  transition. (:func:`state_onset_aligned_synchrony`)
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import polars as pl
import xarray as xr

from ... import get as _get
from ...util.validity.hypno import merge_brief_breaking_bouts
from ...util.validity.mask import validity_mask
from .aggregate import (
    aggregate_fisher_z,
    aggregate_pooled_sums,
    aggregate_simple_mean,
)
from .bouts import all_segments_in_state, fixed_valid_bouts, state_hypno_bouts
from .core import _to_combined, pairwise_pearson_corr
from .state_compare import (
    _classify_pair,
    _coord_to_float_array,
    _coord_to_int_array,
    _coord_to_str_array,
    _flatten_upper,
    _load_dn,
)

_BOUT_BUILDERS = {
    "fixed_valid": fixed_valid_bouts,
    "all_segments": all_segments_in_state,
    "hypno_bouts": state_hypno_bouts,
}
_AGGREGATORS = {
    "simple_mean": aggregate_simple_mean,
    "fisher_z_weighted": lambda d, b, **k: aggregate_fisher_z(d, b, weighted=True, **k),
    "fisher_z_unweighted": lambda d, b, **k: aggregate_fisher_z(d, b, weighted=False, **k),
    "pooled_sums": aggregate_pooled_sums,
}
_DEFAULT_FIXED_BOUT_KWARGS = {
    "epoch_length": 10.0,
    "min_bout_length": 10.0,
    "max_nan_span": 2.0,
    "mode": "span",
}
_DEFAULT_ALL_SEGMENT_KWARGS = {"min_bout_length": 4.0}
_DEFAULT_HYPNO_BOUT_KWARGS = {"min_bout_length": 0.0}


def _eff_bout_kwargs(strategy: str, user_kwargs: dict | None) -> dict:
    if strategy == "fixed_valid":
        base = _DEFAULT_FIXED_BOUT_KWARGS
    elif strategy == "all_segments":
        base = _DEFAULT_ALL_SEGMENT_KWARGS
    else:
        base = _DEFAULT_HYPNO_BOUT_KWARGS
    return {**base, **(user_kwargs or {})}


def _compute_cumulative_clocks(
    hypno,
    val_mask: xr.DataArray,
    *,
    states: tuple[str, ...] = ("NREM", "Wake", "REM"),
    use_valid: bool = False,
) -> dict:
    """Per state, return a 1-D array on ``val_mask``'s time grid giving
    cumulative seconds spent in that state up to (and including) each
    timepoint.

    Parameters
    ----------
    hypno : electro_py.hypno.Hypnogram
    val_mask : xr.DataArray
        1-D boolean mask along ``time``.
    states : tuple of str
    use_valid : bool
        If True, only valid (non-NaN-data) samples count toward the clock.
        If False (default), every sample inside a state bout counts —
        equivalent to wall-clock time in that state, which is the more
        biologically natural "how much state has the animal experienced"
        metric. Use ``True`` for cross-checks against
        :func:`state_clock_table` which uses valid time.

    Returns
    -------
    dict
        ``{'_times': times_array, '_dt': dt, state_name: cum_array, ...}``
    """
    times = np.asarray(val_mask["time"].values, dtype=np.float64)
    valid = np.asarray(val_mask.values, dtype=bool)
    if times.size < 2:
        dt = 0.0
    else:
        dt = float(np.median(np.diff(times)))
    out: dict = {"_times": times, "_dt": dt}
    df = hypno.df
    for state in states:
        bouts = df.filter(pl.col("state") == state)
        in_state = np.zeros(times.size, dtype=bool)
        if bouts.height > 0:
            starts = bouts["start_time"].to_numpy()
            ends = bouts["end_time"].to_numpy()
            for s, e in zip(starts, ends):
                in_state |= (times >= s) & (times < e)
        contributing = in_state & valid if use_valid else in_state
        out[state] = np.cumsum(contributing.astype(np.float64)) * dt
    return out


# ---------------------------------------------------------------- annotate


def annotate_bout_temporal_context(
    bouts: pl.DataFrame,
    *,
    use_valid_duration: bool = True,
) -> pl.DataFrame:
    """Annotate a bouts DataFrame with per-state temporal context columns.

    Per state independently (so the NREM clock isn't perturbed by intervening
    Wake bouts), adds:

    - ``bout_index_in_state`` (i64): 0 for the earliest bout of this state,
      1 for the next, …
    - ``cum_state_time_at_start`` (f64): seconds of state-bout duration that
      elapsed in *this state* before this bout's start.
    - ``cum_state_time_at_end`` (f64): cumulative including this bout.
    - ``bout_duration`` (f64): alias of ``valid_duration`` if
      ``use_valid_duration``, else ``wall_duration``. Used as the per-bout
      contribution to the cumulative.
    - ``prev_bout_gap_s`` (f64): wall-clock seconds between the *end* of the
      previous same-state bout and the *start* of this one. Null for the
      first bout of each state.

    Pure function: takes a bouts DataFrame, returns a copy with extra columns.
    No data load.

    Parameters
    ----------
    bouts : pl.DataFrame
        Bouts in the standard schema, possibly containing multiple states.
    use_valid_duration : bool
        If True (default), cumulative time and ``bout_duration`` use
        ``valid_duration``; otherwise ``wall_duration``.

    Returns
    -------
    pl.DataFrame
        Copy of input with five additional columns. Output is sorted by
        ``(state, start_time)``.
    """
    if bouts.height == 0:
        return bouts.with_columns(
            pl.lit(None).cast(pl.Int64).alias("bout_index_in_state"),
            pl.lit(None).cast(pl.Float64).alias("cum_state_time_at_start"),
            pl.lit(None).cast(pl.Float64).alias("cum_state_time_at_end"),
            pl.lit(None).cast(pl.Float64).alias("bout_duration"),
            pl.lit(None).cast(pl.Float64).alias("prev_bout_gap_s"),
        )

    dur_col = "valid_duration" if use_valid_duration else "wall_duration"
    if dur_col not in bouts.columns:
        raise ValueError(f"`bouts` is missing required column {dur_col!r}.")

    out = bouts.sort(["state", "start_time"]).with_columns(
        pl.col(dur_col).alias("bout_duration"),
    )
    # Per-state cumulative within an ordered window
    out = out.with_columns(
        pl.col("bout_duration").cum_sum().over("state").alias("cum_state_time_at_end"),
        pl.col("start_time").cum_count().over("state").alias("_n_so_far"),
        pl.col("end_time").shift(1).over("state").alias("_prev_end"),
    )
    out = out.with_columns(
        (pl.col("cum_state_time_at_end") - pl.col("bout_duration")).alias(
            "cum_state_time_at_start"
        ),
        (pl.col("_n_so_far") - 1).cast(pl.Int64).alias("bout_index_in_state"),
        (pl.col("start_time") - pl.col("_prev_end")).alias("prev_bout_gap_s"),
    ).drop(["_n_so_far", "_prev_end"])

    return out


# -------------------------------------------------- within-bout sliding


def _summarise_window(
    r_xr: xr.DataArray,
    n_xr: xr.DataArray,
    *,
    iu: np.ndarray,
    ju: np.ndarray,
    pair_type_arr: np.ndarray,
    same_type_arr: np.ndarray | None = None,
    high_threshold: float,
) -> dict:
    """Reduce a (syn_1, syn_2) r matrix into the same summary shape used by
    :func:`bout_level_synchrony` — kept consistent so notebook code can stack
    these windows next to per-bout summaries without column-shape mismatches.

    When ``same_type_arr`` is provided (basal_basal / apical_apical /
    basal_apical / unknown labels per pair), three additional columns are
    emitted so callers can stratify within-bout drift by dendrite type.
    """
    r_off = r_xr.values[iu, ju]
    with np.errstate(invalid="ignore"):
        mean_off = float(np.nanmean(r_off))
        median_off = float(np.nanmedian(r_off))
        frac_high = float(np.nanmean(r_off > high_threshold))

        def _strat_mean(name: str) -> float:
            sel = pair_type_arr == name
            if not sel.any():
                return float("nan")
            return float(np.nanmean(r_off[sel]))

        n_samples = int(np.nanmax(n_xr.values)) if n_xr.values.size else 0
        out = {
            "n_samples": n_samples,
            "mean_r_offdiag": mean_off,
            "median_r_offdiag": median_off,
            "frac_r_above": frac_high,
            "mean_r_within_dend": _strat_mean("within_dend"),
            "mean_r_between_dend_same_type": _strat_mean("between_dend_same_type"),
            "mean_r_between_dend_cross_type": _strat_mean("between_dend_cross_type"),
        }
        if same_type_arr is not None:
            offdiag_sel = pair_type_arr != "within_dend"
            for st_label in ("basal_basal", "apical_apical", "basal_apical"):
                sel = (same_type_arr == st_label) & offdiag_sel
                out[f"mean_r_{st_label}"] = (
                    float(np.nanmean(r_off[sel])) if sel.any() else float("nan")
                )
                out[f"n_pairs_{st_label}"] = int(sel.sum())
        return out


def _per_synapse_pair_classifications(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(iu, ju, pair_type_arr, same_type_arr)`` for the i<j upper triangle of ``da``."""
    n_syn = da.sizes["syn_id"]
    iu, ju = np.triu_indices(n_syn, k=1)

    def _coord_strs(name: str) -> np.ndarray:
        if name in da.coords:
            return np.asarray([str(v) for v in da.coords[name].values])
        return np.asarray(["?"] * n_syn)

    dends = _coord_strs("dend-ID")
    dtypes = _coord_strs("dend_type")
    pair_type_arr = np.empty(len(iu), dtype=object)
    same_type_arr = np.empty(len(iu), dtype=object)
    for k, (i, j) in enumerate(zip(iu, ju)):
        pt, st = _classify_pair(dends[i], dends[j], dtypes[i], dtypes[j])
        pair_type_arr[k] = pt
        same_type_arr[k] = st
    return iu, ju, pair_type_arr, same_type_arr


def sliding_window_corr_in_bout(
    da: xr.DataArray,
    bout_row: dict,
    *,
    window_s: float = 30.0,
    step_s: float = 10.0,
    min_valid_frac: float = 0.8,
    high_threshold: float = 0.2,
    iu: np.ndarray | None = None,
    ju: np.ndarray | None = None,
    pair_type_arr: np.ndarray | None = None,
    same_type_arr: np.ndarray | None = None,
) -> pl.DataFrame:
    """Sliding-window pairwise Pearson correlation across one bout.

    For a single bout, slides a window of width ``window_s`` (seconds) across
    its time range with step ``step_s`` (seconds), running
    :func:`pairwise_pearson_corr` on the synapses in ``da`` for each window.

    Parameters
    ----------
    da : xr.DataArray
        ``(syn_id, time)`` array, single channel, on the same time axis as
        ``bout_row['start_time']/['end_time']`` (TDT-block-relative seconds in
        the canonical pipeline).
    bout_row : dict
        One row from a bouts DataFrame (``start_time``, ``end_time``,
        ``valid_duration`` etc.) — call ``bouts.iter_rows(named=True)``.
    window_s : float
        Window width in seconds.
    step_s : float
        Window stride in seconds.
    min_valid_frac : float
        Minimum fraction of non-NaN samples (over the union mask, which equals
        ``validity_mode='all'``) required for a window to be emitted. Default
        0.8.
    high_threshold : float
        Threshold for ``frac_r_above`` column. Default 0.2.
    iu, ju, pair_type_arr : np.ndarray, optional
        Pre-computed upper-triangle indices and pair-type classifications for
        ``da``. If omitted they are computed once. Pass them when calling this
        function repeatedly across many bouts on the same ``da``.

    Returns
    -------
    pl.DataFrame
        One row per emitted window. Columns: ``window_idx``,
        ``window_center_in_bout_s``, ``window_start_time``, ``window_end_time``,
        ``n_samples``, ``mean_r_offdiag``, ``median_r_offdiag``,
        ``frac_r_above``, ``mean_r_within_dend``,
        ``mean_r_between_dend_same_type``, ``mean_r_between_dend_cross_type``.
        Returns an empty (typed) DataFrame if no windows pass the validity
        threshold.
    """
    if window_s <= 0:
        raise ValueError(f"window_s must be > 0, got {window_s}")
    if step_s <= 0:
        raise ValueError(f"step_s must be > 0, got {step_s}")

    if iu is None or ju is None or pair_type_arr is None:
        iu, ju, pair_type_arr, same_type_arr = _per_synapse_pair_classifications(da)

    t_start = float(bout_row["start_time"])
    t_end = float(bout_row["end_time"])
    if t_end - t_start < window_s:
        return _empty_window_df()

    starts = np.arange(t_start, t_end - window_s + 1e-9, step_s, dtype=float)
    if starts.size == 0:
        return _empty_window_df()

    rows: list[dict] = []
    for w_idx, w_start in enumerate(starts):
        w_end = w_start + window_s
        sl = da.sel(time=slice(w_start, w_end))
        if sl.sizes.get("time", 0) == 0:
            continue
        # Validity check on the union mask: a sample is "valid" iff every
        # synapse is non-NaN at that time. _load_dn enforces that all-syn-NaN
        # invariant, so checking syn 0 here is sufficient.
        vals = np.asarray(sl.values, dtype=np.float64)
        # n_total samples in this slice
        n_total = vals.shape[1]
        if n_total == 0:
            continue
        n_valid = int(np.sum(~np.isnan(vals[0])))
        if n_valid / n_total < min_valid_frac:
            continue
        r_xr, n_xr = pairwise_pearson_corr(sl, return_n=True)
        summ = _summarise_window(
            r_xr, n_xr,
            iu=iu, ju=ju, pair_type_arr=pair_type_arr, same_type_arr=same_type_arr,
            high_threshold=high_threshold,
        )
        rows.append({
            "window_idx": w_idx,
            "window_start_time": float(w_start),
            "window_end_time": float(w_end),
            "window_center_in_bout_s": float((w_start + w_end) / 2 - t_start),
            **summ,
        })

    if not rows:
        return _empty_window_df()
    return pl.DataFrame(rows)


def _empty_window_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "window_idx": pl.Int64,
            "window_start_time": pl.Float64,
            "window_end_time": pl.Float64,
            "window_center_in_bout_s": pl.Float64,
            "n_samples": pl.Int64,
            "mean_r_offdiag": pl.Float64,
            "median_r_offdiag": pl.Float64,
            "frac_r_above": pl.Float64,
            "mean_r_within_dend": pl.Float64,
            "mean_r_between_dend_same_type": pl.Float64,
            "mean_r_between_dend_cross_type": pl.Float64,
            "mean_r_basal_basal": pl.Float64,
            "n_pairs_basal_basal": pl.Int64,
            "mean_r_apical_apical": pl.Float64,
            "n_pairs_apical_apical": pl.Int64,
            "mean_r_basal_apical": pl.Float64,
            "n_pairs_basal_apical": pl.Int64,
        }
    )


def within_bout_correlation_timeline(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    *,
    states: tuple[str, ...] = ("NREM", "Wake"),
    window_s: float = 30.0,
    step_s: float = 10.0,
    min_bout_length_s: float = 60.0,
    min_valid_frac: float = 0.8,
    high_threshold: float = 0.2,
    bout_strategy: str = "all_segments",
    bout_kwargs: dict | None = None,
    merge_brief_max_s: float = 0.0,
    attach_cumulative_clocks: bool = True,
    cumulative_clock_states: tuple[str, ...] = ("NREM", "Wake", "REM"),
    cumulative_clock_use_valid: bool = False,
    channel: int = 0,
    trace: str = "denoised",
    validity_mode: str = "all",
) -> pl.DataFrame:
    """Within-bout correlation timeline for one (subject, soma).

    For each requested state, derive bouts, filter to bouts ≥
    ``min_bout_length_s``, and run :func:`sliding_window_corr_in_bout` on each.
    Concatenate with bout context columns attached.

    Parameters
    ----------
    subject, exp, loc, acq, soma_id : str
    states : tuple of str
        States to include. Default ``("NREM", "Wake")``. Pass
        ``("NREM", "Wake", "REM")`` to include REM.
    window_s, step_s : float
        Sliding-window parameters.
    min_bout_length_s : float
        Drop bouts whose duration is below this. Per the bout strategy in
        question, this filters on ``valid_duration``.
    min_valid_frac : float
        Forwarded to :func:`sliding_window_corr_in_bout`.
    high_threshold : float
        Forwarded to :func:`sliding_window_corr_in_bout`.
    bout_strategy : str
        ``"all_segments"`` (default — variable-length contiguous valid runs)
        or ``"fixed_valid"``.
    bout_kwargs : dict, optional
        Forwarded to the bout builder.
    channel, trace, validity_mode : see :func:`build_state_corr_table`.

    Returns
    -------
    pl.DataFrame
        Long-form. One row per emitted window. Columns:

        - identifiers: ``subject, exp, loc, acq, soma_id, cell_id, recording_id``
        - bout context: ``state, bout_idx, bout_index_in_state,
          bout_start_time, bout_end_time, bout_duration,
          cum_state_time_at_start, prev_bout_gap_s``
        - window context: ``window_idx, window_start_time, window_end_time,
          window_center_in_bout_s, window_center_in_bout_norm`` (= window
          center / bout_duration ∈ [0,1])
        - correlation summaries: ``n_samples, mean_r_offdiag,
          median_r_offdiag, frac_r_above, mean_r_within_dend,
          mean_r_between_dend_same_type, mean_r_between_dend_cross_type``
        - parameters: ``window_s, step_s, bout_strategy``

    Notes
    -----
    REM-feasibility: this function does not warn when REM has zero qualifying
    bouts; it just returns no REM rows. Callers iterating across many cells
    should aggregate REM panel feasibility from the returned table itself.
    """
    if bout_strategy not in _BOUT_BUILDERS:
        raise ValueError(
            f"bout_strategy must be one of {list(_BOUT_BUILDERS)}, got {bout_strategy!r}"
        )

    recording_id = f"{subject}|{exp}|{loc}|{acq}"
    cell_id = f"{recording_id}|{soma_id}"

    dn = _load_dn(subject, exp, loc, acq, soma_id, channel=channel, trace=trace)
    if dn.sizes.get("syn_id", 0) < 2:
        raise ValueError("Need ≥2 synapses to correlate.")
    val_mask = validity_mask(dn, mode=validity_mode)
    hypno, *_ = _get.acq_sleep_coverage(subject, exp, loc, acq)
    if merge_brief_max_s > 0:
        hypno = merge_brief_breaking_bouts(hypno, max_break_s=merge_brief_max_s)

    builder = _BOUT_BUILDERS[bout_strategy]
    eff_kwargs = _eff_bout_kwargs(bout_strategy, bout_kwargs)

    iu, ju, pair_type_arr, same_type_arr = _per_synapse_pair_classifications(dn)
    combined = _to_combined(dn)

    all_bouts: list[pl.DataFrame] = []
    for state in states:
        b = builder(hypno, val_mask, state, **eff_kwargs)
        if b.height == 0:
            continue
        all_bouts.append(b)
    if not all_bouts:
        return _empty_timeline_df()
    bouts_all = pl.concat(all_bouts, how="vertical")
    bouts_annot = annotate_bout_temporal_context(bouts_all, use_valid_duration=True)
    bouts_annot = bouts_annot.filter(pl.col("valid_duration") >= min_bout_length_s)
    if bouts_annot.height == 0:
        return _empty_timeline_df()

    rows: list[pl.DataFrame] = []
    for b in bouts_annot.iter_rows(named=True):
        win = sliding_window_corr_in_bout(
            combined, b,
            window_s=window_s, step_s=step_s,
            min_valid_frac=min_valid_frac, high_threshold=high_threshold,
            iu=iu, ju=ju, pair_type_arr=pair_type_arr, same_type_arr=same_type_arr,
        )
        if win.height == 0:
            continue
        bout_duration = float(b["valid_duration"])
        win = win.with_columns(
            (pl.col("window_center_in_bout_s") / bout_duration).alias(
                "window_center_in_bout_norm"
            ),
            pl.lit(subject).alias("subject"),
            pl.lit(exp).alias("exp"),
            pl.lit(loc).alias("loc"),
            pl.lit(acq).alias("acq"),
            pl.lit(soma_id).alias("soma_id"),
            pl.lit(cell_id).alias("cell_id"),
            pl.lit(recording_id).alias("recording_id"),
            pl.lit(b["state"]).alias("state"),
            pl.lit(int(b["bout_idx"])).alias("bout_idx"),
            pl.lit(int(b["bout_index_in_state"])).alias("bout_index_in_state"),
            pl.lit(float(b["start_time"])).alias("bout_start_time"),
            pl.lit(float(b["end_time"])).alias("bout_end_time"),
            pl.lit(bout_duration).alias("bout_duration"),
            pl.lit(float(b["cum_state_time_at_start"])).alias("cum_state_time_at_start"),
            pl.lit(b["prev_bout_gap_s"]).cast(pl.Float64).alias("prev_bout_gap_s"),
            pl.lit(float(window_s)).alias("window_s"),
            pl.lit(float(step_s)).alias("step_s"),
            pl.lit(bout_strategy).alias("bout_strategy"),
        )
        rows.append(win)

    if not rows:
        return _empty_timeline_df()
    out = pl.concat(rows, how="vertical")

    if attach_cumulative_clocks and out.height > 0:
        clocks = _compute_cumulative_clocks(
            hypno, val_mask,
            states=cumulative_clock_states,
            use_valid=cumulative_clock_use_valid,
        )
        clock_times = clocks["_times"]
        # Window-center wall time = bout_start_time + window_center_in_bout_s.
        centers = (
            out["bout_start_time"].to_numpy()
            + out["window_center_in_bout_s"].to_numpy()
        )
        idx = np.clip(
            np.searchsorted(clock_times, centers, side="right") - 1,
            0, max(clock_times.size - 1, 0),
        )
        new_cols: list[pl.Series] = []
        for state in cumulative_clock_states:
            new_cols.append(pl.Series(
                f"cum_{state}_s_at_window", clocks[state][idx], dtype=pl.Float64
            ))
        out = out.with_columns(new_cols)
    return out


def _empty_timeline_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "window_idx": pl.Int64,
            "window_start_time": pl.Float64,
            "window_end_time": pl.Float64,
            "window_center_in_bout_s": pl.Float64,
            "window_center_in_bout_norm": pl.Float64,
            "n_samples": pl.Int64,
            "mean_r_offdiag": pl.Float64,
            "median_r_offdiag": pl.Float64,
            "frac_r_above": pl.Float64,
            "mean_r_within_dend": pl.Float64,
            "mean_r_between_dend_same_type": pl.Float64,
            "mean_r_between_dend_cross_type": pl.Float64,
            "mean_r_basal_basal": pl.Float64,
            "n_pairs_basal_basal": pl.Int64,
            "mean_r_apical_apical": pl.Float64,
            "n_pairs_apical_apical": pl.Int64,
            "mean_r_basal_apical": pl.Float64,
            "n_pairs_basal_apical": pl.Int64,
            "subject": pl.String,
            "exp": pl.String,
            "loc": pl.String,
            "acq": pl.String,
            "soma_id": pl.String,
            "cell_id": pl.String,
            "recording_id": pl.String,
            "state": pl.String,
            "bout_idx": pl.Int64,
            "bout_index_in_state": pl.Int64,
            "bout_start_time": pl.Float64,
            "bout_end_time": pl.Float64,
            "bout_duration": pl.Float64,
            "cum_state_time_at_start": pl.Float64,
            "prev_bout_gap_s": pl.Float64,
            "window_s": pl.Float64,
            "step_s": pl.Float64,
            "bout_strategy": pl.String,
        }
    )


def within_bout_correlation_timeline_multi(
    rows: pl.DataFrame | Iterable[dict],
    **kwargs,
) -> pl.DataFrame:
    """Run :func:`within_bout_correlation_timeline` over many cells and concat.

    On error, prints a warning and skips. Use this when iterating the
    notebook's testbed.
    """
    if isinstance(rows, pl.DataFrame):
        iterator = list(rows.iter_rows(named=True))
    else:
        iterator = list(rows)
    out: list[pl.DataFrame] = []
    for row in iterator:
        try:
            df = within_bout_correlation_timeline(
                row["subject"], row["exp"], row["loc"], row["acq"], row["soma_id"],
                **kwargs,
            )
            if df.height > 0:
                out.append(df)
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"within_bout_correlation_timeline failed for "
                f"{row.get('subject')}/{row.get('exp')}/{row.get('loc')}/"
                f"{row.get('acq')}/{row.get('soma_id')}: "
                f"{type(e).__name__}: {e}",
                stacklevel=2,
            )
    if not out:
        return _empty_timeline_df()
    return pl.concat(out, how="vertical")


# ----------------------------------------------------------- state-clock


def _build_state_clock_bins(
    bouts: pl.DataFrame,
    *,
    window_s: float,
) -> pl.DataFrame:
    """For one state's bouts, partition the cumulative-valid-time axis into
    fixed-width bins and return per-bin slices on wall-time.

    Each bin spans ``window_s`` seconds of valid state-time. A single bin can
    span across bout boundaries; conversely a single bout can be split across
    several bins. Returns a DataFrame with columns ``clock_bin_idx``,
    ``clock_bin_start_s`` (cumulative-state-time at bin start),
    ``clock_bin_end_s``, ``clock_bin_center_s``, plus, for each (bin, bout)
    intersection, ``slice_start_time`` / ``slice_end_time`` on wall time.

    Notes
    -----
    Wall-time slicing is approximate when the underlying state-bout contains
    interior NaN gaps: we proportionally interpolate cumulative-state-time
    fractions onto the bout's [start_time, end_time] wall range. For
    ``all_segments`` bouts (NaN-free by construction) this is exact; for
    ``fixed_valid`` bouts in ``span`` mode it slightly inflates the wall
    duration of bins that happen to land on NaN gaps. Acceptable for the
    state-clock view.
    """
    if bouts.height == 0:
        return pl.DataFrame(
            schema={
                "clock_bin_idx": pl.Int64,
                "clock_bin_start_s": pl.Float64,
                "clock_bin_end_s": pl.Float64,
                "clock_bin_center_s": pl.Float64,
                "slice_start_time": pl.Float64,
                "slice_end_time": pl.Float64,
            }
        )

    annot = annotate_bout_temporal_context(bouts, use_valid_duration=True).sort("start_time")
    total_state_s = float(annot["cum_state_time_at_end"].max())
    if total_state_s <= 0:
        return pl.DataFrame(
            schema={
                "clock_bin_idx": pl.Int64,
                "clock_bin_start_s": pl.Float64,
                "clock_bin_end_s": pl.Float64,
                "clock_bin_center_s": pl.Float64,
                "slice_start_time": pl.Float64,
                "slice_end_time": pl.Float64,
            }
        )

    n_bins = int(np.ceil(total_state_s / window_s))
    bin_edges_clock = np.arange(n_bins + 1, dtype=float) * window_s
    # Clip last edge to total state time to avoid out-of-range windowing.
    bin_edges_clock[-1] = min(bin_edges_clock[-1], total_state_s)

    # For every (bin, bout) overlap, emit a wall-time slice.
    rows: list[dict] = []
    for b in annot.iter_rows(named=True):
        c0 = float(b["cum_state_time_at_start"])
        c1 = float(b["cum_state_time_at_end"])
        wall_start = float(b["start_time"])
        wall_end = float(b["end_time"])
        wall_dur = max(wall_end - wall_start, 1e-12)
        valid_dur = float(b["valid_duration"])
        if valid_dur <= 0:
            continue

        # The bins this bout overlaps:
        first = int(np.floor(c0 / window_s))
        last = int(np.floor((c1 - 1e-12) / window_s))
        for bidx in range(first, last + 1):
            bin_lo = bidx * window_s
            bin_hi = (bidx + 1) * window_s
            ov_lo = max(bin_lo, c0)
            ov_hi = min(bin_hi, c1)
            if ov_hi <= ov_lo:
                continue
            # Map cumulative-state-time fractions back onto wall time, using
            # this bout's wall range and total valid duration as the linear
            # mapping target.
            f_lo = (ov_lo - c0) / valid_dur
            f_hi = (ov_hi - c0) / valid_dur
            slice_start = wall_start + f_lo * wall_dur
            slice_end = wall_start + f_hi * wall_dur
            rows.append({
                "clock_bin_idx": int(bidx),
                "clock_bin_start_s": float(bin_lo),
                "clock_bin_end_s": float(min(bin_hi, total_state_s)),
                "clock_bin_center_s": float(0.5 * (bin_lo + min(bin_hi, total_state_s))),
                "slice_start_time": float(slice_start),
                "slice_end_time": float(slice_end),
            })
    return pl.DataFrame(rows)


def state_clock_table(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    *,
    state: str,
    window_s: float = 60.0,
    aggregation: str = "fisher_z_weighted",
    bout_strategy: str = "all_segments",
    bout_kwargs: dict | None = None,
    merge_brief_max_s: float = 0.0,
    channel: int = 0,
    trace: str = "denoised",
    validity_mode: str = "all",
) -> pl.DataFrame:
    """State-clock-tiled long-form correlation table for one (cell, state).

    Cumulative valid time spent in ``state`` (across the recording) is
    partitioned into ``window_s``-second bins. Each bin can collect data from
    one or more state bouts. Per bin, the chosen aggregator is applied to the
    set of wall-time slices that fall inside the bin.

    Returns a long-form per-(pair, bin) table compatible with
    :func:`mixed_state_test` (the formula will use ``clock_bin_center_s`` as
    the predictor).

    Parameters
    ----------
    subject, exp, loc, acq, soma_id : str
    state : str
        Single state to tile; call once per state.
    window_s : float
        Bin width on the cumulative-state-time axis. Default 60s.
    aggregation : str
        One of ``"simple_mean"``, ``"fisher_z_weighted"`` (default),
        ``"fisher_z_unweighted"``, ``"pooled_sums"``.
    bout_strategy : str
        Default ``"all_segments"`` (NaN-free segments give exact wall-time
        proportional mapping).
    bout_kwargs : dict, optional
        Forwarded to bout builder.

    Returns
    -------
    pl.DataFrame
        One row per (pair, clock_bin). Columns: standard pair metadata
        (``subject, exp, loc, acq, soma_id, cell_id, recording_id, syn_i,
        syn_j, dmd_i, dmd_j, dend_i, dend_j, dend_type_i, dend_type_j,
        pair_type, same_type, dend_pair, soma_i, soma_j``) + ``state``,
        ``clock_bin_idx``, ``clock_bin_center_s``, ``clock_bin_start_s``,
        ``clock_bin_end_s``, ``r``, ``z``, ``se_z``, ``n_samples``,
        ``n_slices`` (number of wall-time slices summed into this bin),
        ``window_s``, ``bout_strategy``, ``aggregation``.
    """
    if bout_strategy not in _BOUT_BUILDERS:
        raise ValueError(
            f"bout_strategy must be one of {list(_BOUT_BUILDERS)}, got {bout_strategy!r}"
        )
    if aggregation not in _AGGREGATORS:
        raise ValueError(
            f"aggregation must be one of {list(_AGGREGATORS)}, got {aggregation!r}"
        )

    recording_id = f"{subject}|{exp}|{loc}|{acq}"
    cell_id = f"{recording_id}|{soma_id}"

    dn = _load_dn(subject, exp, loc, acq, soma_id, channel=channel, trace=trace)
    if dn.sizes.get("syn_id", 0) < 2:
        raise ValueError("Need ≥2 synapses to correlate.")
    val_mask = validity_mask(dn, mode=validity_mode)
    hypno, *_ = _get.acq_sleep_coverage(subject, exp, loc, acq)
    if merge_brief_max_s > 0:
        hypno = merge_brief_breaking_bouts(hypno, max_break_s=merge_brief_max_s)

    builder = _BOUT_BUILDERS[bout_strategy]
    aggregator = _AGGREGATORS[aggregation]
    eff_kwargs = _eff_bout_kwargs(bout_strategy, bout_kwargs)

    bouts = builder(hypno, val_mask, state, **eff_kwargs)
    if bouts.height == 0:
        return _empty_state_clock_df()

    slices = _build_state_clock_bins(bouts, window_s=window_s)
    if slices.height == 0:
        return _empty_state_clock_df()

    n_syn = dn.sizes["syn_id"]
    syn_ids = np.asarray(dn["syn_id"].values, dtype=str)
    dmds = _coord_to_int_array(dn, "dmd", n_syn)
    dends = _coord_to_str_array(dn, "dend-ID", n_syn)
    dend_types = _coord_to_str_array(dn, "dend_type", n_syn)
    somas = _coord_to_str_array(dn, "soma-ID", n_syn)
    poses = _coord_to_float_array(dn, "pos", n_syn)

    out_rows: list[pl.DataFrame] = []
    for bin_idx, group in slices.group_by("clock_bin_idx", maintain_order=True):
        bin_idx = int(bin_idx[0]) if isinstance(bin_idx, tuple) else int(bin_idx)
        slice_df = group.select(
            pl.col("slice_start_time").alias("start_time"),
            pl.col("slice_end_time").alias("end_time"),
        )
        # Aggregator expects standard "bouts" with start_time/end_time cols.
        r_xr, n_xr, z_xr, se_z_xr = aggregator(dn, slice_df)
        flat = _flatten_upper(
            r_xr, n_xr, z_xr, se_z_xr,
            syn_ids=syn_ids, dmds=dmds, dends=dends,
            dend_types=dend_types, somas=somas, poses=poses,
        )
        n_pairs = len(flat["syn_i"])
        if n_pairs == 0:
            continue
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
            dend_pair[k] = "|".join(sorted([str(flat["dend_i"][k]), str(flat["dend_j"][k])]))

        bin_center = float(group["clock_bin_center_s"][0])
        bin_start = float(group["clock_bin_start_s"][0])
        bin_end = float(group["clock_bin_end_s"][0])
        df = pl.DataFrame({
            "subject": [subject] * n_pairs,
            "exp": [exp] * n_pairs,
            "loc": [loc] * n_pairs,
            "acq": [acq] * n_pairs,
            "soma_id": [soma_id] * n_pairs,
            "cell_id": [cell_id] * n_pairs,
            "recording_id": [recording_id] * n_pairs,
            "state": [state] * n_pairs,
            "clock_bin_idx": [bin_idx] * n_pairs,
            "clock_bin_center_s": [bin_center] * n_pairs,
            "clock_bin_start_s": [bin_start] * n_pairs,
            "clock_bin_end_s": [bin_end] * n_pairs,
            "syn_i": flat["syn_i"],
            "syn_j": flat["syn_j"],
            "soma_i": flat["soma_i"],
            "soma_j": flat["soma_j"],
            "r": flat["r"],
            "z": flat["z"],
            "se_z": flat["se_z"],
            "n_samples": flat["n_samples"],
            "n_slices": [group.height] * n_pairs,
            "dmd_i": flat["dmd_i"],
            "dmd_j": flat["dmd_j"],
            "dend_i": flat["dend_i"],
            "dend_j": flat["dend_j"],
            "dend_type_i": flat["dend_type_i"],
            "dend_type_j": flat["dend_type_j"],
            "pair_type": pair_type.astype(str),
            "same_type": same_type.astype(str),
            "dend_pair": dend_pair.astype(str),
            "window_s": [float(window_s)] * n_pairs,
            "bout_strategy": [bout_strategy] * n_pairs,
            "aggregation": [aggregation] * n_pairs,
        })
        out_rows.append(df)

    if not out_rows:
        return _empty_state_clock_df()
    return pl.concat(out_rows, how="vertical")


def _empty_state_clock_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "subject": pl.String, "exp": pl.String, "loc": pl.String,
            "acq": pl.String, "soma_id": pl.String, "cell_id": pl.String,
            "recording_id": pl.String, "state": pl.String,
            "clock_bin_idx": pl.Int64, "clock_bin_center_s": pl.Float64,
            "clock_bin_start_s": pl.Float64, "clock_bin_end_s": pl.Float64,
            "syn_i": pl.String, "syn_j": pl.String,
            "soma_i": pl.String, "soma_j": pl.String,
            "r": pl.Float64, "z": pl.Float64, "se_z": pl.Float64,
            "n_samples": pl.Int64, "n_slices": pl.UInt32,
            "dmd_i": pl.Int64, "dmd_j": pl.Int64,
            "dend_i": pl.String, "dend_j": pl.String,
            "dend_type_i": pl.String, "dend_type_j": pl.String,
            "pair_type": pl.String, "same_type": pl.String,
            "dend_pair": pl.String, "window_s": pl.Float64,
            "bout_strategy": pl.String, "aggregation": pl.String,
        }
    )


def state_clock_table_multi(
    rows: pl.DataFrame | Iterable[dict],
    *,
    state: str,
    **kwargs,
) -> pl.DataFrame:
    """Run :func:`state_clock_table` over many cells for one state and concat."""
    if isinstance(rows, pl.DataFrame):
        iterator = list(rows.iter_rows(named=True))
    else:
        iterator = list(rows)
    out: list[pl.DataFrame] = []
    for row in iterator:
        try:
            df = state_clock_table(
                row["subject"], row["exp"], row["loc"], row["acq"], row["soma_id"],
                state=state, **kwargs,
            )
            if df.height > 0:
                out.append(df)
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"state_clock_table failed for "
                f"{row.get('subject')}/{row.get('exp')}/{row.get('loc')}/"
                f"{row.get('acq')}/{row.get('soma_id')} state={state}: "
                f"{type(e).__name__}: {e}",
                stacklevel=2,
            )
    if not out:
        return _empty_state_clock_df()
    return pl.concat(out, how="vertical")


# ------------------------------------------------------- state-onset


def state_onset_aligned_synchrony(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    *,
    onset_state: str = "NREM",
    pre_state: str | None = "Wake",
    pre_window_s: float = 60.0,
    post_window_s: float = 120.0,
    window_s: float = 30.0,
    step_s: float = 10.0,
    min_pre_state_s: float = 30.0,
    min_post_state_s: float = 60.0,
    min_valid_frac: float = 0.8,
    high_threshold: float = 0.2,
    merge_brief_max_s: float = 0.0,
    channel: int = 0,
    trace: str = "denoised",
    validity_mode: str = "all",
) -> pl.DataFrame:
    """Compute synchrony aligned on state-onset transitions.

    For each transition where the previous bout's state is ``pre_state`` (or
    any state if ``pre_state is None``) and the next bout's state is
    ``onset_state``, slide a window across [-pre_window_s, +post_window_s]
    relative to the transition and compute :func:`pairwise_pearson_corr`
    summary stats per window.

    Both bouts must meet their respective minimum-duration requirements; the
    pre-window and post-window are clamped within the bouts' wall ranges, so
    if the transition is closer to a bout end than ``pre_window_s``, the
    available pre-time is shorter (windows that would start before the bout
    are skipped). Same on the post side.

    Parameters
    ----------
    subject, exp, loc, acq, soma_id : str
    onset_state : str
        Target post-transition state. Default ``"NREM"``.
    pre_state : str | None
        Required pre-transition state, or None to allow any. Default
        ``"Wake"``.
    pre_window_s, post_window_s : float
        Time range to scan around the transition.
    window_s, step_s : float
        Sliding-window parameters.
    min_pre_state_s, min_post_state_s : float
        Minimum durations the pre-bout and post-bout must satisfy.
    min_valid_frac, high_threshold : see :func:`sliding_window_corr_in_bout`.
    channel, trace, validity_mode : standard.

    Returns
    -------
    pl.DataFrame
        One row per (transition, window). Columns include ``onset_idx``,
        ``transition_time`` (wall-time of the boundary), ``t_rel_s`` (window
        center − transition_time), ``window_phase`` ('pre' or 'post'),
        ``pre_state``, ``onset_state``, plus the standard window-summary
        columns and identifiers.
    """
    recording_id = f"{subject}|{exp}|{loc}|{acq}"
    cell_id = f"{recording_id}|{soma_id}"

    dn = _load_dn(subject, exp, loc, acq, soma_id, channel=channel, trace=trace)
    if dn.sizes.get("syn_id", 0) < 2:
        raise ValueError("Need ≥2 synapses to correlate.")
    val_mask = validity_mask(dn, mode=validity_mode)
    hypno, *_ = _get.acq_sleep_coverage(subject, exp, loc, acq)
    if merge_brief_max_s > 0:
        hypno = merge_brief_breaking_bouts(hypno, max_break_s=merge_brief_max_s)

    # Use the FULL hypnogram-state bouts (regardless of interior NaN), since the
    # pre-/post-window slicer crosses interior NaN gaps via the sliding window's
    # min_valid_frac, and we need genuine state-boundary transitions (not the
    # tighter all_segments NaN-cut boundaries).
    pre_bouts = (
        state_hypno_bouts(hypno, val_mask, pre_state, min_bout_length=min_pre_state_s)
        if pre_state is not None
        else None
    )
    post_bouts = state_hypno_bouts(
        hypno, val_mask, onset_state, min_bout_length=min_post_state_s
    )
    if post_bouts.height == 0:
        return _empty_onset_df()

    iu, ju, pair_type_arr, same_type_arr = _per_synapse_pair_classifications(dn)
    combined = _to_combined(dn)

    rows: list[dict] = []
    for onset_idx, post in enumerate(post_bouts.iter_rows(named=True)):
        post_start = float(post["start_time"])
        post_end = float(post["end_time"])
        # Find pre-bout that ends right at (or just before) post_start.
        if pre_bouts is None:
            pre_start = None
            pre_end = post_start
        else:
            cands = pre_bouts.filter(
                (pl.col("end_time") <= post_start + 0.01)
                & (pl.col("end_time") >= post_start - 0.5)
            )
            if cands.height == 0:
                continue
            pre = cands.row(cands.height - 1, named=True)
            pre_start = float(pre["start_time"])
            pre_end = float(pre["end_time"])
            if (pre_end - pre_start) < min_pre_state_s:
                continue

        transition_time = post_start
        # Pre-window: [transition_time - pre_window_s, transition_time]
        pre_lo = max(
            transition_time - pre_window_s,
            pre_start if pre_start is not None else transition_time - pre_window_s,
        )
        pre_hi = transition_time
        # Post-window: [transition_time, transition_time + post_window_s]
        post_lo = transition_time
        post_hi = min(transition_time + post_window_s, post_end)

        # Pre side: windows ending at successive offsets from transition_time
        for w_start in np.arange(pre_lo, pre_hi - window_s + 1e-9, step_s):
            w_end = w_start + window_s
            if w_end > pre_hi + 1e-9:
                break
            row = _eval_window(
                combined, w_start, w_end, iu, ju, pair_type_arr,
                same_type_arr=same_type_arr,
                min_valid_frac=min_valid_frac, high_threshold=high_threshold,
            )
            if row is None:
                continue
            rows.append({
                **row,
                "onset_idx": int(onset_idx),
                "transition_time": float(transition_time),
                "window_phase": "pre",
                "pre_state": pre_state if pre_state is not None else "ANY",
                "onset_state": onset_state,
                "subject": subject, "exp": exp, "loc": loc, "acq": acq,
                "soma_id": soma_id, "cell_id": cell_id, "recording_id": recording_id,
            })

        # Post side
        for w_start in np.arange(post_lo, post_hi - window_s + 1e-9, step_s):
            w_end = w_start + window_s
            if w_end > post_hi + 1e-9:
                break
            row = _eval_window(
                combined, w_start, w_end, iu, ju, pair_type_arr,
                same_type_arr=same_type_arr,
                min_valid_frac=min_valid_frac, high_threshold=high_threshold,
            )
            if row is None:
                continue
            rows.append({
                **row,
                "onset_idx": int(onset_idx),
                "transition_time": float(transition_time),
                "window_phase": "post",
                "pre_state": pre_state if pre_state is not None else "ANY",
                "onset_state": onset_state,
                "subject": subject, "exp": exp, "loc": loc, "acq": acq,
                "soma_id": soma_id, "cell_id": cell_id, "recording_id": recording_id,
            })

    if not rows:
        return _empty_onset_df()
    df = pl.DataFrame(rows)
    df = df.with_columns(
        ((pl.col("window_start_time") + pl.col("window_end_time")) / 2
         - pl.col("transition_time")).alias("t_rel_s")
    )
    return df


def _eval_window(
    combined: xr.DataArray,
    w_start: float,
    w_end: float,
    iu: np.ndarray,
    ju: np.ndarray,
    pair_type_arr: np.ndarray,
    same_type_arr: np.ndarray | None = None,
    *,
    min_valid_frac: float,
    high_threshold: float,
) -> dict | None:
    sl = combined.sel(time=slice(w_start, w_end))
    if sl.sizes.get("time", 0) == 0:
        return None
    vals = np.asarray(sl.values, dtype=np.float64)
    n_total = vals.shape[1]
    if n_total == 0:
        return None
    n_valid = int(np.sum(~np.isnan(vals[0])))
    if n_valid / n_total < min_valid_frac:
        return None
    r_xr, n_xr = pairwise_pearson_corr(sl, return_n=True)
    summ = _summarise_window(
        r_xr, n_xr, iu=iu, ju=ju,
        pair_type_arr=pair_type_arr, same_type_arr=same_type_arr,
        high_threshold=high_threshold,
    )
    return {
        "window_start_time": float(w_start),
        "window_end_time": float(w_end),
        **summ,
    }


def _empty_onset_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "window_start_time": pl.Float64,
            "window_end_time": pl.Float64,
            "n_samples": pl.Int64,
            "mean_r_offdiag": pl.Float64,
            "median_r_offdiag": pl.Float64,
            "frac_r_above": pl.Float64,
            "mean_r_within_dend": pl.Float64,
            "mean_r_between_dend_same_type": pl.Float64,
            "mean_r_between_dend_cross_type": pl.Float64,
            "onset_idx": pl.Int64,
            "transition_time": pl.Float64,
            "window_phase": pl.String,
            "pre_state": pl.String,
            "onset_state": pl.String,
            "subject": pl.String, "exp": pl.String, "loc": pl.String,
            "acq": pl.String, "soma_id": pl.String, "cell_id": pl.String,
            "recording_id": pl.String,
            "t_rel_s": pl.Float64,
        }
    )


def state_onset_aligned_synchrony_multi(
    rows: pl.DataFrame | Iterable[dict],
    **kwargs,
) -> pl.DataFrame:
    """Run :func:`state_onset_aligned_synchrony` over many cells and concat."""
    if isinstance(rows, pl.DataFrame):
        iterator = list(rows.iter_rows(named=True))
    else:
        iterator = list(rows)
    out: list[pl.DataFrame] = []
    for row in iterator:
        try:
            df = state_onset_aligned_synchrony(
                row["subject"], row["exp"], row["loc"], row["acq"], row["soma_id"],
                **kwargs,
            )
            if df.height > 0:
                out.append(df)
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"state_onset_aligned_synchrony failed for "
                f"{row.get('subject')}/{row.get('exp')}/{row.get('loc')}/"
                f"{row.get('acq')}/{row.get('soma_id')}: "
                f"{type(e).__name__}: {e}",
                stacklevel=2,
            )
    if not out:
        return _empty_onset_df()
    return pl.concat(out, how="vertical")


# ----------------------------------------- head-vs-tail per-bout drift


def head_tail_bout_drift(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    *,
    states: tuple[str, ...] = ("NREM", "Wake", "REM"),
    head_tail_seconds: float = 30.0,
    min_bout_seconds: float = 90.0,
    bout_strategy: str = "hypno_bouts",
    bout_kwargs: dict | None = None,
    high_threshold: float = 0.2,
    merge_brief_max_s: float = 0.0,
    channel: int = 0,
    trace: str = "denoised",
    validity_mode: str = "all",
) -> pl.DataFrame:
    """Per-bout head-vs-tail correlation drift.

    For each bout with ``wall_duration >= min_bout_seconds`` and enough valid
    samples to fit non-overlapping head and tail windows, compute pairwise
    Pearson r over **the first ``head_tail_seconds`` of valid data** (head)
    and **the last ``head_tail_seconds`` of valid data** (tail). Returns
    one row per bout with summary stats and the head→tail change.

    The head/tail are constructed by walking forward (head) or backward
    (tail) from the bout's wall boundary, accumulating non-NaN samples
    until ``head_tail_seconds × fs`` valid samples are gathered. So head
    and tail are each guaranteed to contain exactly ``head_tail_seconds``
    of valid signal — their wall extent may be larger if NaN gaps lie
    inside. Bouts where the resulting head and tail would overlap (i.e.
    the bout doesn't have ``2 × head_tail_seconds`` of valid data with
    non-overlapping placement) are skipped.

    Parameters
    ----------
    subject, exp, loc, acq, soma_id : str
    states : tuple of str
    head_tail_seconds : float
        Seconds of valid data to gather at each end. Default 30.
    min_bout_seconds : float
        Minimum bout wall_duration (seconds) to qualify. Bouts shorter
        than this are skipped before any sample-counting. Default 90 s.
    bout_strategy : str
        Default ``"hypno_bouts"`` (full hypnogram state-bouts).
    bout_kwargs : dict | None
    high_threshold : float
        For the per-summary ``frac_r_above`` columns. Default 0.2.

    Returns
    -------
    pl.DataFrame
        One row per surviving bout. Columns:

        - identifiers: ``subject, exp, loc, acq, soma_id, cell_id,
          recording_id, state, bout_idx``
        - bout extents: ``bout_start_time, bout_end_time, wall_duration,
          valid_duration``
        - head/tail extents: ``head_start_time, head_end_time,
          head_wall_span, tail_start_time, tail_end_time, tail_wall_span,
          gap_seconds`` (= ``tail_start_time - head_end_time``)
        - head correlations: ``mean_r_offdiag_head, median_r_offdiag_head,
          frac_r_above_head, mean_r_within_dend_head,
          mean_r_between_dend_same_type_head,
          mean_r_between_dend_cross_type_head, n_pairs_head``
        - tail correlations: same prefix replaced with ``_tail``
        - delta: ``delta_mean_r_offdiag = mean_r_offdiag_tail -
          mean_r_offdiag_head``, plus per-stratum deltas and
          ``delta_z_offdiag = arctanh(r_tail) - arctanh(r_head)``.
    """
    if bout_strategy not in _BOUT_BUILDERS:
        raise ValueError(
            f"bout_strategy must be one of {list(_BOUT_BUILDERS)}, got {bout_strategy!r}"
        )

    recording_id = f"{subject}|{exp}|{loc}|{acq}"
    cell_id = f"{recording_id}|{soma_id}"

    dn = _load_dn(subject, exp, loc, acq, soma_id, channel=channel, trace=trace)
    if dn.sizes.get("syn_id", 0) < 2:
        raise ValueError("Need ≥2 synapses to correlate.")
    val_mask = validity_mask(dn, mode=validity_mode)
    hypno, *_ = _get.acq_sleep_coverage(subject, exp, loc, acq)
    if merge_brief_max_s > 0:
        hypno = merge_brief_breaking_bouts(hypno, max_break_s=merge_brief_max_s)

    builder = _BOUT_BUILDERS[bout_strategy]
    eff_kwargs = _eff_bout_kwargs(bout_strategy, bout_kwargs)
    iu, ju, pair_type_arr, same_type_arr = _per_synapse_pair_classifications(dn)
    combined = _to_combined(dn)

    times = np.asarray(dn["time"].values, dtype=np.float64)
    if times.size < 2:
        raise ValueError("Insufficient time samples to estimate dt.")
    dt = float(np.median(np.diff(times)))
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError(f"Bad dt={dt!r}.")
    n_target = int(round(head_tail_seconds / dt))

    rows: list[dict] = []
    for state in states:
        b = builder(hypno, val_mask, state, **eff_kwargs)
        if b.height == 0:
            continue
        b = b.filter(pl.col("wall_duration") >= min_bout_seconds)
        for bout in b.iter_rows(named=True):
            t0, t1 = float(bout["start_time"]), float(bout["end_time"])
            sl = combined.sel(time=slice(t0, t1))
            if sl.sizes.get("time", 0) == 0:
                continue
            slice_times = np.asarray(sl["time"].values, dtype=np.float64)
            valid = ~np.isnan(np.asarray(sl.values[0], dtype=np.float64))
            n_valid_total = int(valid.sum())
            # Need >= 2 × n_target valid samples to place non-overlapping head & tail.
            if n_valid_total < 2 * n_target or n_target <= 0:
                continue
            cum = np.cumsum(valid)
            # Head ends at the smallest k where cum[k] == n_target (1-indexed).
            head_end_idx = int(np.searchsorted(cum, n_target))
            if head_end_idx >= len(slice_times):
                continue
            # Tail starts at the smallest j (counted from end) where
            # cum_rev[j] == n_target. Equivalently: the largest k where
            # (n_valid_total - cum[k]) >= n_target  — use the first index
            # where n_valid_total - cum[k-1] == n_target, i.e. cum[k-1] ==
            # n_valid_total - n_target.
            cum_threshold = n_valid_total - n_target
            tail_start_idx = int(np.searchsorted(cum, cum_threshold, side="right"))
            if tail_start_idx <= head_end_idx:
                # Head and tail would overlap — skip.
                continue
            head_slice = sl.isel(time=slice(0, head_end_idx + 1))
            tail_slice = sl.isel(time=slice(tail_start_idx, len(slice_times)))

            # Sanity: each side has at least n_target valid samples.
            head_valid_cnt = int((~np.isnan(head_slice.values[0])).sum())
            tail_valid_cnt = int((~np.isnan(tail_slice.values[0])).sum())
            if head_valid_cnt < n_target or tail_valid_cnt < n_target:
                continue

            head_summ = _summarise_window(
                *pairwise_pearson_corr(head_slice, return_n=True),
                iu=iu, ju=ju, pair_type_arr=pair_type_arr, same_type_arr=same_type_arr,
                high_threshold=high_threshold,
            )
            tail_summ = _summarise_window(
                *pairwise_pearson_corr(tail_slice, return_n=True),
                iu=iu, ju=ju, pair_type_arr=pair_type_arr, same_type_arr=same_type_arr,
                high_threshold=high_threshold,
            )

            head_t0 = float(slice_times[0])
            head_t1 = float(slice_times[head_end_idx])
            tail_t0 = float(slice_times[tail_start_idx])
            tail_t1 = float(slice_times[-1])

            r_head = head_summ["mean_r_offdiag"]
            r_tail = tail_summ["mean_r_offdiag"]
            with np.errstate(invalid="ignore"):
                z_head = float(np.arctanh(np.clip(r_head, -0.999999, 0.999999))) if np.isfinite(r_head) else float("nan")
                z_tail = float(np.arctanh(np.clip(r_tail, -0.999999, 0.999999))) if np.isfinite(r_tail) else float("nan")

            row = {
                "subject": subject, "exp": exp, "loc": loc, "acq": acq,
                "soma_id": soma_id, "cell_id": cell_id, "recording_id": recording_id,
                "state": state,
                "bout_idx": int(bout["bout_idx"]),
                "bout_start_time": t0, "bout_end_time": t1,
                "wall_duration": float(bout["wall_duration"]),
                "valid_duration": float(bout["valid_duration"]),
                "head_start_time": head_t0, "head_end_time": head_t1,
                "head_wall_span": head_t1 - head_t0,
                "tail_start_time": tail_t0, "tail_end_time": tail_t1,
                "tail_wall_span": tail_t1 - tail_t0,
                "gap_seconds": tail_t0 - head_t1,
                "head_tail_seconds": float(head_tail_seconds),
                "n_pairs": int(len(iu)),
                # head
                "n_samples_head": int(head_summ["n_samples"]),
                "mean_r_offdiag_head": r_head,
                "median_r_offdiag_head": head_summ["median_r_offdiag"],
                "frac_r_above_head": head_summ["frac_r_above"],
                "mean_r_within_dend_head": head_summ["mean_r_within_dend"],
                "mean_r_between_dend_same_type_head": head_summ["mean_r_between_dend_same_type"],
                "mean_r_between_dend_cross_type_head": head_summ["mean_r_between_dend_cross_type"],
                "z_head": z_head,
                # tail
                "n_samples_tail": int(tail_summ["n_samples"]),
                "mean_r_offdiag_tail": r_tail,
                "median_r_offdiag_tail": tail_summ["median_r_offdiag"],
                "frac_r_above_tail": tail_summ["frac_r_above"],
                "mean_r_within_dend_tail": tail_summ["mean_r_within_dend"],
                "mean_r_between_dend_same_type_tail": tail_summ["mean_r_between_dend_same_type"],
                "mean_r_between_dend_cross_type_tail": tail_summ["mean_r_between_dend_cross_type"],
                "z_tail": z_tail,
                # delta = tail - head
                "delta_mean_r_offdiag": r_tail - r_head,
                "delta_median_r_offdiag": tail_summ["median_r_offdiag"] - head_summ["median_r_offdiag"],
                "delta_frac_r_above": tail_summ["frac_r_above"] - head_summ["frac_r_above"],
                "delta_mean_r_within_dend": (
                    tail_summ["mean_r_within_dend"] - head_summ["mean_r_within_dend"]
                ),
                "delta_mean_r_between_dend_same_type": (
                    tail_summ["mean_r_between_dend_same_type"]
                    - head_summ["mean_r_between_dend_same_type"]
                ),
                "delta_mean_r_between_dend_cross_type": (
                    tail_summ["mean_r_between_dend_cross_type"]
                    - head_summ["mean_r_between_dend_cross_type"]
                ),
                "delta_z_offdiag": z_tail - z_head,
            }
            # Stratified-by-dendrite-type head/tail/delta columns.
            for st_label in ("basal_basal", "apical_apical", "basal_apical"):
                row[f"mean_r_{st_label}_head"] = head_summ.get(f"mean_r_{st_label}", float("nan"))
                row[f"mean_r_{st_label}_tail"] = tail_summ.get(f"mean_r_{st_label}", float("nan"))
                row[f"n_pairs_{st_label}"] = head_summ.get(f"n_pairs_{st_label}", 0)
                row[f"delta_mean_r_{st_label}"] = (
                    row[f"mean_r_{st_label}_tail"] - row[f"mean_r_{st_label}_head"]
                )
            rows.append(row)

    if not rows:
        return _empty_head_tail_df()
    return pl.DataFrame(rows)


def _empty_head_tail_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "subject": pl.String, "exp": pl.String, "loc": pl.String,
            "acq": pl.String, "soma_id": pl.String, "cell_id": pl.String,
            "recording_id": pl.String, "state": pl.String,
            "bout_idx": pl.Int64,
            "bout_start_time": pl.Float64, "bout_end_time": pl.Float64,
            "wall_duration": pl.Float64, "valid_duration": pl.Float64,
            "head_start_time": pl.Float64, "head_end_time": pl.Float64,
            "head_wall_span": pl.Float64,
            "tail_start_time": pl.Float64, "tail_end_time": pl.Float64,
            "tail_wall_span": pl.Float64,
            "gap_seconds": pl.Float64,
            "head_tail_seconds": pl.Float64,
            "n_pairs": pl.Int64,
            "n_samples_head": pl.Int64,
            "mean_r_offdiag_head": pl.Float64, "median_r_offdiag_head": pl.Float64,
            "frac_r_above_head": pl.Float64,
            "mean_r_within_dend_head": pl.Float64,
            "mean_r_between_dend_same_type_head": pl.Float64,
            "mean_r_between_dend_cross_type_head": pl.Float64,
            "z_head": pl.Float64,
            "n_samples_tail": pl.Int64,
            "mean_r_offdiag_tail": pl.Float64, "median_r_offdiag_tail": pl.Float64,
            "frac_r_above_tail": pl.Float64,
            "mean_r_within_dend_tail": pl.Float64,
            "mean_r_between_dend_same_type_tail": pl.Float64,
            "mean_r_between_dend_cross_type_tail": pl.Float64,
            "z_tail": pl.Float64,
            "delta_mean_r_offdiag": pl.Float64,
            "delta_median_r_offdiag": pl.Float64,
            "delta_frac_r_above": pl.Float64,
            "delta_mean_r_within_dend": pl.Float64,
            "delta_mean_r_between_dend_same_type": pl.Float64,
            "delta_mean_r_between_dend_cross_type": pl.Float64,
            "delta_z_offdiag": pl.Float64,
            "mean_r_basal_basal_head": pl.Float64,
            "mean_r_basal_basal_tail": pl.Float64,
            "delta_mean_r_basal_basal": pl.Float64,
            "n_pairs_basal_basal": pl.Int64,
            "mean_r_apical_apical_head": pl.Float64,
            "mean_r_apical_apical_tail": pl.Float64,
            "delta_mean_r_apical_apical": pl.Float64,
            "n_pairs_apical_apical": pl.Int64,
            "mean_r_basal_apical_head": pl.Float64,
            "mean_r_basal_apical_tail": pl.Float64,
            "delta_mean_r_basal_apical": pl.Float64,
            "n_pairs_basal_apical": pl.Int64,
        }
    )


def head_tail_bout_drift_multi(
    rows: pl.DataFrame | Iterable[dict],
    **kwargs,
) -> pl.DataFrame:
    """Run :func:`head_tail_bout_drift` over many cells and concat."""
    if isinstance(rows, pl.DataFrame):
        iterator = list(rows.iter_rows(named=True))
    else:
        iterator = list(rows)
    out: list[pl.DataFrame] = []
    for row in iterator:
        try:
            df = head_tail_bout_drift(
                row["subject"], row["exp"], row["loc"], row["acq"], row["soma_id"],
                **kwargs,
            )
            if df.height > 0:
                out.append(df)
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"head_tail_bout_drift failed for "
                f"{row.get('subject')}/{row.get('exp')}/{row.get('loc')}/"
                f"{row.get('acq')}/{row.get('soma_id')}: "
                f"{type(e).__name__}: {e}",
                stacklevel=2,
            )
    if not out:
        return _empty_head_tail_df()
    return pl.concat(out, how="vertical")
