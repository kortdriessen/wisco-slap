"""Outlier discovery on the long-form state-correlation table.

- :func:`outlier_pairs` — rank pairs by ``Δz`` (or ``|Δz|``) and return the
  top N along with metadata. Use the result with
  :func:`wisco_slap.scope.corr.plot_outlier_pairs`.
- :func:`bout_level_synchrony` — re-run pairwise correlation per bout and
  return per-bout summary statistics (off-diagonal mean r, fraction above
  threshold, stratified means). Lets you see whether some NREM bouts are
  much more synchronous than others. Output feeds
  :func:`wisco_slap.scope.corr.plot_bout_synchrony_timeline`.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ... import get as _get
from ...util.validity.mask import validity_mask
from .bouts import all_segments_in_state, fixed_valid_bouts, state_hypno_bouts
from .core import _to_combined, pairwise_pearson_corr
from .state_compare import _classify_pair, _load_dn

_BOUT_BUILDERS = {
    "fixed_valid": fixed_valid_bouts,
    "all_segments": all_segments_in_state,
    "hypno_bouts": state_hypno_bouts,
}


def outlier_pairs(
    table: pl.DataFrame,
    *,
    direction: str = "nrem_higher",
    top_n: int = 20,
    states: tuple[str, str] = ("NREM", "Wake"),
    min_n_samples: int | None = None,
    max_se_z: float | None = None,
    drop_within_dend: bool = True,
) -> pl.DataFrame:
    """Rank pairs by Δz to surface state-dependent correlation outliers.

    For each pair, joins the two state rows and computes
    ``Δz = z_<states[0]> − z_<states[1]>``. Sorts by ``Δz`` (or ``|Δz|``)
    and returns the top ``top_n`` rows with all metadata.

    Parameters
    ----------
    table : pl.DataFrame
        Long-form state-correlation table.
    direction : str
        ``"nrem_higher"`` (default; sort Δz descending),
        ``"wake_higher"`` (ascending), or
        ``"abs"`` (largest |Δz|).
    top_n : int
        Max rows to return.
    states : tuple of two str
        Δz = z[states[0]] − z[states[1]]. Default ``('NREM', 'Wake')``.
    min_n_samples : int | None
        Drop pairs whose smaller per-state ``n_samples`` is below this.
    max_se_z : float | None
        Drop pairs whose larger per-state ``se_z`` exceeds this.
    drop_within_dend : bool
        Default True — within-dendrite pairs are biological-noise-confounded
        and rarely the outliers you want.

    Returns
    -------
    pl.DataFrame
        One row per top pair, sorted as requested. Columns: pair metadata,
        ``z_<state>``, ``se_z_<state>``, ``n_<state>``, ``delta_z``,
        ``rank``.
    """
    s0, s1 = states
    df = table
    if drop_within_dend:
        df = df.filter(pl.col("pair_type") != "within_dend")

    keep_meta = [
        "cell_id", "recording_id",
        "subject", "exp", "loc", "acq", "soma_id",
        "syn_i", "syn_j",
        "dmd_i", "dmd_j",
        "dend_i", "dend_j",
        "dend_type_i", "dend_type_j",
        "pair_type", "same_type", "dend_pair",
    ]

    parts = []
    for state in states:
        sub = df.filter(pl.col("state") == state).select(
            *keep_meta,
            pl.col("z").alias(f"z_{state}"),
            pl.col("se_z").alias(f"se_z_{state}"),
            pl.col("n_samples").alias(f"n_{state}"),
        )
        parts.append(sub)
    wide = parts[0]
    for sub in parts[1:]:
        wide = wide.join(sub, on=keep_meta, how="inner")

    wide = wide.with_columns(
        (pl.col(f"z_{s0}") - pl.col(f"z_{s1}")).alias("delta_z"),
    ).filter(
        pl.col("delta_z").is_not_null() & pl.col("delta_z").is_finite()
    )

    if min_n_samples is not None:
        wide = wide.filter(
            pl.min_horizontal(pl.col(f"n_{s0}"), pl.col(f"n_{s1}")) >= min_n_samples
        )
    if max_se_z is not None:
        wide = wide.filter(
            pl.max_horizontal(pl.col(f"se_z_{s0}"), pl.col(f"se_z_{s1}")) <= max_se_z
        )

    if direction == "nrem_higher":
        wide = wide.sort("delta_z", descending=True)
    elif direction == "wake_higher":
        wide = wide.sort("delta_z", descending=False)
    elif direction == "abs":
        wide = wide.with_columns(pl.col("delta_z").abs().alias("_abs_delta")).sort(
            "_abs_delta", descending=True
        ).drop("_abs_delta")
    else:
        raise ValueError(f"direction must be 'nrem_higher', 'wake_higher', or 'abs', got {direction!r}")

    out = wide.head(top_n).with_row_index("rank", offset=1)
    return out


def bout_level_synchrony(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    *,
    channel: int = 0,
    trace: str = "denoised",
    bout_strategy: str = "fixed_valid",
    bout_kwargs: dict | None = None,
    states: tuple[str, ...] = ("NREM", "Wake"),
    high_threshold: float = 0.2,
    validity_mode: str = "all",
    annotate_temporal: bool = False,
) -> pl.DataFrame:
    """Per-bout correlation summaries for one (subject, soma).

    For every bout, runs :func:`pairwise_pearson_corr` on that bout's data and
    summarises the resulting matrix: off-diagonal mean / median r, fraction
    of pairs with ``r > high_threshold``, plus stratified means for
    within-dend / between-dend-same-type / between-dend-cross-type.

    Useful for asking "are some NREM bouts much more synchronous than
    others?" — pair the output with
    :func:`wisco_slap.scope.corr.plot_bout_synchrony_timeline`.

    Parameters
    ----------
    subject, exp, loc, acq, soma_id : str
    channel : int
    trace : str
    bout_strategy : str
        ``"fixed_valid"`` or ``"all_segments"``. Default fixed_valid.
    bout_kwargs : dict | None
        Forwarded to the bout builder.
    states : tuple of str
        Default ``('NREM', 'Wake')``.
    high_threshold : float
        Threshold for the ``frac_r_above`` column. Default 0.2.
    validity_mode : str
        See :func:`build_state_corr_table`. Default ``"all"``.
    annotate_temporal : bool
        If True, also add the per-state temporal-context columns produced
        by :func:`wisco_slap.scope.corr.annotate_bout_temporal_context`
        (``bout_index_in_state``, ``cum_state_time_at_start``,
        ``cum_state_time_at_end``, ``bout_duration``, ``prev_bout_gap_s``).
        Default False (kept off for backwards compatibility).

    Returns
    -------
    pl.DataFrame
        One row per bout. Columns: ``state, start_time, end_time,
        valid_duration, wall_duration, n_samples, mean_r_offdiag,
        median_r_offdiag, frac_r_above, mean_r_within_dend,
        mean_r_between_dend_same_type, mean_r_between_dend_cross_type``.
        With ``annotate_temporal=True`` also adds per-state context
        columns; see :func:`annotate_bout_temporal_context`.
    """
    if bout_strategy not in _BOUT_BUILDERS:
        raise ValueError(
            f"bout_strategy must be one of {list(_BOUT_BUILDERS)}, got {bout_strategy!r}"
        )
    builder = _BOUT_BUILDERS[bout_strategy]

    dn = _load_dn(subject, exp, loc, acq, soma_id, channel=channel, trace=trace)
    if dn.sizes.get("syn_id", 0) < 2:
        raise ValueError("Need ≥2 synapses to correlate.")
    val_mask = validity_mask(dn, mode=validity_mode)
    hypno, *_ = _get.acq_sleep_coverage(subject, exp, loc, acq)

    if bout_strategy == "fixed_valid":
        eff_kwargs = {
            "epoch_length": 10.0, "min_bout_length": 10.0, "max_nan_span": 2.0,
            "mode": "span",
        }
    elif bout_strategy == "all_segments":
        eff_kwargs = {"min_bout_length": 4.0}
    else:  # hypno_bouts
        eff_kwargs = {"min_bout_length": 0.0}
    eff_kwargs.update(bout_kwargs or {})

    # Pre-compute per-pair classifications once.
    n_syn = dn.sizes["syn_id"]
    iu, ju = np.triu_indices(n_syn, k=1)
    dends = np.asarray([str(v) for v in dn.coords.get("dend-ID", np.full(n_syn, "?")).values])
    dtypes = np.asarray([str(v) for v in dn.coords.get("dend_type", np.full(n_syn, "?")).values])
    pair_type_arr = np.empty(len(iu), dtype=object)
    for k, (i, j) in enumerate(zip(iu, ju)):
        pt, _st = _classify_pair(dends[i], dends[j], dtypes[i], dtypes[j])
        pair_type_arr[k] = pt

    combined = _to_combined(dn)

    rows: list[dict] = []
    for state in states:
        bouts = builder(hypno, val_mask, state, **eff_kwargs)
        for b in bouts.iter_rows(named=True):
            t1, t2 = float(b["start_time"]), float(b["end_time"])
            sl = combined.sel(time=slice(t1, t2))
            r_xr, n_xr = pairwise_pearson_corr(sl, return_n=True)
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

                row = {
                    "state": state,
                    "start_time": t1,
                    "end_time": t2,
                    "valid_duration": float(b["valid_duration"]),
                    "wall_duration": float(b["wall_duration"]),
                    "n_samples": int(np.nanmax(n_xr.values)) if n_xr.values.size else 0,
                    "mean_r_offdiag": mean_off,
                    "median_r_offdiag": median_off,
                    "frac_r_above": frac_high,
                    "mean_r_within_dend": _strat_mean("within_dend"),
                    "mean_r_between_dend_same_type": _strat_mean("between_dend_same_type"),
                    "mean_r_between_dend_cross_type": _strat_mean("between_dend_cross_type"),
                }
            rows.append(row)

    out = pl.DataFrame(rows).sort("start_time")
    if annotate_temporal:
        from .temporal import annotate_bout_temporal_context
        # The per-bout summary table already has state/start_time/end_time/
        # valid_duration/wall_duration. We need a synthetic bout_idx column
        # for annotate_bout_temporal_context to be happy with the schema —
        # the value is opaque downstream and we keep the per-state ordering.
        if "bout_idx" not in out.columns:
            out = out.with_columns(
                pl.int_range(0, pl.len()).cast(pl.Int64).alias("bout_idx")
            )
        out = annotate_bout_temporal_context(out, use_valid_duration=True)
    return out
