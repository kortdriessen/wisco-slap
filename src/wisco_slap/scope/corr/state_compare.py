"""Long-form state-conditioned correlation tables.

The single entry point for building a tidy per-pair-per-state DataFrame ready
for plotting, stratified summaries, and statistical inference. One row per
``(pair, state)``; one (subject, soma) per call to
:func:`build_state_corr_table`; multiple via
:func:`build_state_corr_table_multi`.

See module docstrings of :mod:`wisco_slap.scope.corr.bouts` and
:mod:`wisco_slap.scope.corr.aggregate` for the bout-strategy and aggregation
options. This module orchestrates them into a tidy table.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field

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

_DEFAULT_FIXED_BOUT_KWARGS: dict = {
    "epoch_length": 10.0,
    "min_bout_length": 10.0,
    "max_nan_span": 2.0,
    "mode": "span",
}
_DEFAULT_ALL_SEGMENT_KWARGS: dict = {
    "min_bout_length": 4.0,
}

_BOUT_BUILDERS = {
    "fixed_valid": fixed_valid_bouts,
    "all_segments": all_segments_in_state,
}
_AGGREGATORS = {
    "simple_mean": aggregate_simple_mean,
    "fisher_z_weighted": lambda d, b, **k: aggregate_fisher_z(d, b, weighted=True, **k),
    "fisher_z_unweighted": lambda d, b, **k: aggregate_fisher_z(
        d, b, weighted=False, **k
    ),
    "pooled_sums": aggregate_pooled_sums,
}

_TABLE_SCHEMA = {
    "subject": pl.String,
    "exp": pl.String,
    "loc": pl.String,
    "acq": pl.String,
    "soma_id": pl.String,
    # f"{subject}|{exp}|{loc}|{acq}|{soma_id}" — globally unique cell key.
    # For single-soma analyses: cell_id == cell_id_i == cell_id_j on every row.
    # For multi-soma analyses: equals cell_id_i for same-soma rows; equals
    # recording_id for cross-soma rows (the natural unit when a pair spans
    # two cells).
    "cell_id": pl.String,
    # f"{subject}|{exp}|{loc}|{acq}" — shared across somas in same recording
    "recording_id": pl.String,
    "syn_i": pl.String,
    "syn_j": pl.String,
    # Per-synapse soma & cell identity (always populated, including for
    # single-soma analyses where soma_i == soma_j).
    "soma_i": pl.String,
    "soma_j": pl.String,
    "cell_id_i": pl.String,
    "cell_id_j": pl.String,
    "same_soma": pl.Boolean,
    "state": pl.String,
    "r": pl.Float64,
    "z": pl.Float64,
    "se_z": pl.Float64,
    "n_samples": pl.Int64,
    "n_bouts": pl.Int64,
    "dmd_i": pl.Int64,
    "dmd_j": pl.Int64,
    "dend_i": pl.String,
    "dend_j": pl.String,
    "dend_type_i": pl.String,
    "dend_type_j": pl.String,
    # Proximal-to-distal index along each dendrite (0 = most proximal).
    # NaN when the synapse's dendrite has no prox_lines annotation yet.
    "pos_i": pl.Float64,
    "pos_j": pl.Float64,
    "pair_type": pl.String,
    "same_type": pl.String,
    # New stratum that combines same/cross-soma with same/cross dend-type.
    # One of: same_soma_basal_basal, same_soma_apical_apical, same_soma_basal_apical,
    # cross_soma_basal_basal, cross_soma_apical_apical, cross_soma_basal_apical,
    # plus _unknown variants when dend_type is missing on either synapse.
    "soma_type_class": pl.String,
    "dend_pair": pl.String,
    "bout_strategy": pl.String,
    "aggregation": pl.String,
    "equalise_states": pl.Boolean,
}


def _load_dn(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str | list[str],
    *,
    channel: int = 0,
    trace: str = "denoised",
) -> xr.DataArray:
    """Load and filter the scopex DataArray for one or more somas in a recording.

    ``soma_id`` may be a single string (single-cell analysis) or a list/tuple
    of strings (multi-cell analysis covering the listed somas in this
    acquisition). All synapses are filtered to those with non-null ``dend-ID``.
    """
    sx = _get.syn_dF(subject, exp, loc, acq, trace=trace, channels=[channel])
    dn = _get.combine_scopex_arrays(sx)
    dn = dn.sx.ch(channel)
    if isinstance(soma_id, str):
        dn = dn.sx.somaid(soma_id)
    else:
        # Multi-soma: filter to the union of requested somas.
        soma_ids = [str(s) for s in soma_id]
        soma_coord = np.asarray([str(v) for v in dn["soma-ID"].values], dtype=object)
        mask = np.isin(soma_coord, soma_ids)
        if not mask.any():
            raise ValueError(
                f"No synapses match any of soma_id={soma_ids!r} in "
                f"{subject}/{exp}/{loc}/{acq}."
            )
        dn = dn.sel(syn_id=dn["syn_id"].values[mask])
    dn = dn.sx.dendid("any")
    # for any timepoints that are nan for any synapse, we want them to be nan for all synapses
    dn_nan_eq = dn.where(np.isfinite(dn).all(dim="syn_id"), other=np.nan)
    return dn_nan_eq


def _classify_pair(
    dend_i: str, dend_j: str, dend_type_i: str, dend_type_j: str
) -> tuple[str, str]:
    """Return ``(pair_type, same_type)`` labels for one pair."""
    if dend_i == dend_j:
        pair_type = "within_dend"
    elif dend_type_i == dend_type_j and dend_type_i not in ("?", None):
        pair_type = "between_dend_same_type"
    else:
        pair_type = "between_dend_cross_type"

    di = (dend_type_i or "?").lower()
    dj = (dend_type_j or "?").lower()
    if "?" in (di, dj):
        same_type = "unknown"
    elif di == dj == "basal":
        same_type = "basal_basal"
    elif di == dj == "apical":
        same_type = "apical_apical"
    else:
        same_type = "basal_apical"

    return pair_type, same_type


def _coord_to_str_array(da: xr.DataArray, name: str, n: int) -> np.ndarray:
    if name in da.coords:
        vals = da.coords[name].values
        return np.asarray([str(v) if v is not None else "?" for v in vals])
    return np.asarray(["?"] * n)


def _coord_to_int_array(da: xr.DataArray, name: str, n: int) -> np.ndarray:
    if name in da.coords:
        return np.asarray(da.coords[name].values, dtype=np.int64)
    return np.zeros(n, dtype=np.int64)


def _coord_to_float_array(da: xr.DataArray, name: str, n: int) -> np.ndarray:
    if name in da.coords:
        return np.asarray(da.coords[name].values, dtype=float)
    return np.full(n, np.nan, dtype=float)


def _flatten_upper(
    r: xr.DataArray,
    n: xr.DataArray,
    z: xr.DataArray,
    se_z: xr.DataArray,
    *,
    syn_ids: np.ndarray,
    dmds: np.ndarray,
    dends: np.ndarray,
    dend_types: np.ndarray,
    somas: np.ndarray,
    poses: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pull the i<j upper triangle of the per-pair matrices into flat arrays."""
    n_syn = len(syn_ids)
    iu, ju = np.triu_indices(n_syn, k=1)

    return {
        "syn_i": syn_ids[iu],
        "syn_j": syn_ids[ju],
        "r": r.values[iu, ju],
        "z": z.values[iu, ju],
        "se_z": se_z.values[iu, ju],
        "n_samples": n.values[iu, ju].astype(np.int64),
        "dmd_i": dmds[iu],
        "dmd_j": dmds[ju],
        "dend_i": dends[iu],
        "dend_j": dends[ju],
        "dend_type_i": dend_types[iu],
        "dend_type_j": dend_types[ju],
        "pos_i": poses[iu],
        "pos_j": poses[ju],
        "soma_i": somas[iu],
        "soma_j": somas[ju],
    }


def build_state_corr_table(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str | list[str],
    *,
    channel: int = 0,
    trace: str = "denoised",
    bout_strategy: str = "fixed_valid",
    aggregation: str = "fisher_z_weighted",
    bout_kwargs: dict | None = None,
    equalise_states: bool = False,
    equalise_seed: int = 0,
    states: tuple[str, ...] = ("NREM", "Wake"),
    validity_mode: str = "all",
) -> pl.DataFrame:
    """Long-form per-pair-per-state correlation table for one (subject, recording).

    Pipeline:

    1. Load ``syn_dF`` (channel ``channel``, trace ``trace``); filter to
       the requested soma(s) and to dendrite synapses only.
    2. Build per-state bouts via ``bout_strategy`` ∈ ``{"fixed_valid",
       "all_segments"}``. Defaults match the existing notebook for
       ``"fixed_valid"`` (10-s valid epochs, 2-s NaN tolerance, drop bouts
       <10 s); ``min_bout_length=4`` for ``"all_segments"``.
    3. Optionally equalise bout counts between states by random subsample
       (``equalise_states=True``).
    4. Aggregate per state via ``aggregation`` ∈ ``{"simple_mean",
       "fisher_z_weighted", "fisher_z_unweighted", "pooled_sums"}``.
    5. Flatten the upper triangle (i<j) into long format, attach metadata
       (per-synapse ``soma_i``/``soma_j``/``cell_id_i``/``cell_id_j``,
       plus pair-level ``same_soma`` and ``soma_type_class``), and stamp
       provenance columns.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Target acquisition.
    soma_id : str | list[str]
        Single soma identifier (single-cell analysis) or list of soma
        identifiers (multi-cell analysis covering several somas in this
        recording). When a list is passed, every pair of synapses is
        considered — including cross-soma pairs — and the output table
        carries ``same_soma = soma_i == soma_j`` so you can stratify.
    channel : int
        Trace channel index. ``0`` = iGluSnFR4f (default), ``1`` = jRGECO1a.
    trace : str
        Which scopex trace variant: ``"denoised"`` (default) or ``"ls"``.
    bout_strategy : str
        ``"fixed_valid"`` (equal-valid-duration tiles) or ``"all_segments"``
        (variable-length contiguous (state ∧ valid) runs).
    aggregation : str
        ``"simple_mean"``: nanmean of per-bout r (current notebook default;
        not recommended for variable-length bouts).
        ``"fisher_z_weighted"``: per-bout Fisher-z averaging weighted by
        ``(n−3)`` (default; principled for variable-length bouts).
        ``"fisher_z_unweighted"``: per-bout Fisher-z averaging, equal weight
        per bout.
        ``"pooled_sums"``: per-bout-centered pooled sums (gold-standard
        "use all the data" mode).
    bout_kwargs : dict | None
        Passes through to the chosen bout builder. Merges with the
        strategy-specific defaults.
    equalise_states : bool
        If True, subsample the longer state's bouts to match the shorter's
        count. Off by default; ``fisher_z_weighted`` and ``pooled_sums``
        handle unequal totals correctly without equalisation.
    equalise_seed : int
        Seed used for the equalising subsample.
    states : tuple of str
        Which states to compute. Default ``('NREM', 'Wake')``.
    validity_mode : str
        ``"all"`` (default) — a timepoint counts as valid only if every
        synapse in the filtered set is non-NaN at that timepoint. See
        :func:`wisco_slap.util.validity.validity_mask`.

    Returns
    -------
    pl.DataFrame
        Long-form table; one row per (pair, state). Columns described in
        the module docstring.
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

    dn = _load_dn(subject, exp, loc, acq, soma_id, channel=channel, trace=trace)
    if dn.sizes.get("syn_id", 0) < 2:
        raise ValueError(
            f"Only {dn.sizes.get('syn_id', 0)} synapses survived filtering — "
            "no pairs to correlate."
        )

    val_mask = validity_mask(dn, mode=validity_mode)
    hypno, *_ = _get.acq_sleep_coverage(subject, exp, loc, acq)

    builder = _BOUT_BUILDERS[bout_strategy]
    aggregator = _AGGREGATORS[aggregation]

    if bout_strategy == "fixed_valid":
        eff_bout_kwargs = {**_DEFAULT_FIXED_BOUT_KWARGS, **(bout_kwargs or {})}
    else:
        eff_bout_kwargs = {**_DEFAULT_ALL_SEGMENT_KWARGS, **(bout_kwargs or {})}

    state_bouts: dict[str, pl.DataFrame] = {}
    for state in states:
        state_bouts[state] = builder(hypno, val_mask, state, **eff_bout_kwargs)

    if equalise_states and len(state_bouts) >= 2:
        n_target = min(b.height for b in state_bouts.values())
        for state, bouts in list(state_bouts.items()):
            state_bouts[state] = random_subsample_bouts(
                bouts, n_target, seed=equalise_seed
            )

    # Pre-compute per-synapse metadata once.
    n_syn = dn.sizes["syn_id"]
    syn_ids = np.asarray(dn["syn_id"].values, dtype=str)
    dmds = _coord_to_int_array(dn, "dmd", n_syn)
    dends = _coord_to_str_array(dn, "dend-ID", n_syn)
    dend_types = _coord_to_str_array(dn, "dend_type", n_syn)
    somas = _coord_to_str_array(dn, "soma-ID", n_syn)
    poses = _coord_to_float_array(dn, "pos", n_syn)
    cell_ids_per_syn = np.asarray([f"{recording_id}|{s}" for s in somas], dtype=object)

    rows: list[pl.DataFrame] = []
    for state, bouts in state_bouts.items():
        if bouts.height == 0:
            warnings.warn(
                f"{subject}/{exp}/{loc}/{acq} {soma_id} state={state!r} has 0 bouts — "
                "skipping this state.",
                stacklevel=2,
            )
            continue
        r_xr, n_xr, z_xr, se_z_xr = aggregator(dn, bouts)
        flat = _flatten_upper(
            r_xr,
            n_xr,
            z_xr,
            se_z_xr,
            syn_ids=syn_ids,
            dmds=dmds,
            dends=dends,
            dend_types=dend_types,
            somas=somas,
            poses=poses,
        )
        n_pairs = len(flat["syn_i"])

        pair_type = np.empty(n_pairs, dtype=object)
        same_type = np.empty(n_pairs, dtype=object)
        dend_pair = np.empty(n_pairs, dtype=object)
        for k in range(n_pairs):
            pt, st = _classify_pair(
                flat["dend_i"][k],
                flat["dend_j"][k],
                flat["dend_type_i"][k],
                flat["dend_type_j"][k],
            )
            pair_type[k] = pt
            same_type[k] = st
            di, dj = flat["dend_i"][k], flat["dend_j"][k]
            dend_pair[k] = "|".join(sorted([str(di), str(dj)]))

        soma_i_arr = flat["soma_i"]
        soma_j_arr = flat["soma_j"]
        cell_id_i_arr = np.asarray(
            [f"{recording_id}|{s}" for s in soma_i_arr], dtype=object
        )
        cell_id_j_arr = np.asarray(
            [f"{recording_id}|{s}" for s in soma_j_arr], dtype=object
        )
        same_soma_arr = soma_i_arr == soma_j_arr
        # cell_id (single-string) — for backwards compat. Equals cell_id_i
        # for same-soma rows (or every row in single-soma analyses) and
        # falls back to recording_id when a pair spans two cells.
        cell_id_arr = np.where(same_soma_arr, cell_id_i_arr, recording_id)
        # soma_id (single-string) — equals soma_i for same-soma rows;
        # for cross-soma rows we use the canonical i+j combo.
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

        # soma_type_class: 6-way (or _unknown) cross of same/cross-soma and
        # same/cross dend-type.
        soma_type_class = np.empty(n_pairs, dtype=object)
        for k in range(n_pairs):
            soma_part = "same_soma" if same_soma_arr[k] else "cross_soma"
            soma_type_class[k] = f"{soma_part}_{same_type[k]}"

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
                "r": flat["r"],
                "z": flat["z"],
                "se_z": flat["se_z"],
                "n_samples": flat["n_samples"],
                "n_bouts": [bouts.height] * n_pairs,
                "dmd_i": flat["dmd_i"],
                "dmd_j": flat["dmd_j"],
                "dend_i": flat["dend_i"],
                "dend_j": flat["dend_j"],
                "dend_type_i": flat["dend_type_i"],
                "dend_type_j": flat["dend_type_j"],
                "pos_i": flat["pos_i"],
                "pos_j": flat["pos_j"],
                "pair_type": pair_type.astype(str),
                "same_type": same_type.astype(str),
                "soma_type_class": soma_type_class.astype(str),
                "dend_pair": dend_pair.astype(str),
                "bout_strategy": [bout_strategy] * n_pairs,
                "aggregation": [aggregation] * n_pairs,
                "equalise_states": [equalise_states] * n_pairs,
            },
            schema=_TABLE_SCHEMA,
        )
        rows.append(df)

    if not rows:
        return pl.DataFrame(schema=_TABLE_SCHEMA)
    return pl.concat(rows, how="vertical")


@dataclass
class StateCorrTableBundle:
    """Result of :func:`build_state_corr_table_multi`.

    Attributes
    ----------
    table : pl.DataFrame
        The concatenated long-form table.
    errors : list[dict]
        One entry per failed input row, with the input identifiers and the
        exception message. Empty if everything succeeded.
    """

    table: pl.DataFrame
    errors: list[dict] = field(default_factory=list)

    def __iter__(self):
        # Allow `df, errors = build_state_corr_table_multi(...)` if you prefer.
        yield self.table
        yield self.errors


def build_state_corr_table_multi(
    rows: pl.DataFrame | Iterable[dict],
    **build_kwargs,
) -> StateCorrTableBundle:
    """Run :func:`build_state_corr_table` over multiple inputs.

    Parameters
    ----------
    rows : pl.DataFrame | Iterable[dict]
        Either a polars DataFrame with columns ``subject, exp, loc, acq,
        soma_id`` (one row per call) or any iterable of dict-like rows.
        ``soma_id`` may be a single string (per-cell analysis) or a list
        of strings (multi-cell analysis covering several somas in the
        same recording — in which case the row's call to
        :func:`build_state_corr_table` includes both within-soma and
        cross-soma pairs in the output).
    **build_kwargs
        Forwarded verbatim to :func:`build_state_corr_table`.

    Returns
    -------
    StateCorrTableBundle
        ``.table`` is the concatenated long-form table; ``.errors`` is a list
        of dicts describing per-row failures (each with ``subject, exp, loc,
        acq, soma_id, error``).
    """
    if isinstance(rows, pl.DataFrame):
        iterator = list(rows.iter_rows(named=True))
    else:
        iterator = list(rows)

    tables: list[pl.DataFrame] = []
    errors: list[dict] = []
    for row in iterator:
        try:
            df = build_state_corr_table(
                row["subject"],
                row["exp"],
                row["loc"],
                row["acq"],
                row["soma_id"],
                **build_kwargs,
            )
            tables.append(df)
        except Exception as e:  # noqa: BLE001
            errors.append({
                "subject": row.get("subject"),
                "exp": row.get("exp"),
                "loc": row.get("loc"),
                "acq": row.get("acq"),
                "soma_id": row.get("soma_id"),
                "error": f"{type(e).__name__}: {e}",
            })

    if tables:
        full = pl.concat(tables, how="vertical")
    else:
        full = pl.DataFrame(schema=_TABLE_SCHEMA)
    return StateCorrTableBundle(table=full, errors=errors)


def stratified_summaries(
    table: pl.DataFrame,
    *,
    drop_within_dend_from_offdiag: bool = True,
    unit: str = "cell_id",
    include_soma_strata: bool | None = None,
) -> pl.DataFrame:
    """Per-(unit, state, stratum) means/medians of r and z.

    Strata produced (one row per combination, when present in the table):

    - ``pair_type`` cuts: ``within_dend``, ``between_dend_same_type``,
      ``between_dend_cross_type``.
    - ``same_type`` cuts (over between-dend pairs only): ``basal_basal``,
      ``apical_apical``, ``basal_apical``.
    - ``all_offdiag`` — everything except ``within_dend``.
    - **(when both same-soma and cross-soma pairs are present in the table —
      ``include_soma_strata=True``, or auto-detected by default)**:
      same-vs-cross-soma cuts: ``same_soma``, ``cross_soma``,
      and the orthogonal product ``soma_type_class`` values:
      ``same_soma_basal_basal``, ``cross_soma_basal_basal``, etc.

    Parameters
    ----------
    table : pl.DataFrame
        Long-form table from :func:`build_state_corr_table` /
        :func:`build_state_corr_table_multi`.
    drop_within_dend_from_offdiag : bool
        Affects only ``"all_offdiag"``. Default True.
    unit : str
        Grouping unit. ``"cell_id"`` (default — single-cell analyses) or
        ``"recording_id"`` (preferred for multi-cell analyses where pairs
        can span two cells in the same recording). For multi-cell tables,
        ``"recording_id"`` is the natural unit because cross-soma pairs
        live at the recording level.
    include_soma_strata : bool | None
        If True, emit the same/cross-soma strata. If False, skip them.
        Default ``None`` means "auto" — include them iff the table contains
        both same-soma and cross-soma pairs.

    Returns
    -------
    pl.DataFrame
        Columns: ``<unit>, subject, [exp, loc, acq, soma_id,] recording_id,
        state, stratum, mean_r, median_r, mean_z, median_z, n_pairs,
        n_pairs_with_data``.
    """
    if unit not in ("cell_id", "recording_id"):
        raise ValueError(f"unit must be 'cell_id' or 'recording_id', got {unit!r}")
    grouping = [unit, "state"]
    stratum_dfs: list[pl.DataFrame] = []

    if include_soma_strata is None:
        if "same_soma" in table.columns and table.height > 0:
            include_soma_strata = bool(table["same_soma"].any()) and bool(
                ~table["same_soma"].all()
            )
        else:
            include_soma_strata = False

    # The grouping column will already be present as a key after group_by;
    # don't duplicate it inside agg.
    carry_cols = [
        c
        for c in (
            "subject",
            "exp",
            "loc",
            "acq",
            "soma_id",
            "recording_id",
        )
        if c != unit
    ]

    def _summarise(sub: pl.DataFrame, stratum: str) -> pl.DataFrame:
        carry_aggs = [pl.col(c).first().alias(c) for c in carry_cols]
        return (
            sub
            .group_by(grouping)
            .agg(
                *carry_aggs,
                pl.col("r").mean().alias("mean_r"),
                pl.col("r").median().alias("median_r"),
                pl.col("z").mean().alias("mean_z"),
                pl.col("z").median().alias("median_z"),
                pl.len().alias("n_pairs"),
                pl.col("r").is_not_null().sum().alias("n_pairs_with_data"),
            )
            .with_columns(pl.lit(stratum).alias("stratum"))
        )

    for pair_type_value in (
        "within_dend",
        "between_dend_same_type",
        "between_dend_cross_type",
    ):
        sub = table.filter(pl.col("pair_type") == pair_type_value)
        if sub.height == 0:
            continue
        stratum_dfs.append(_summarise(sub, pair_type_value))

    for same_type_value in ("basal_basal", "apical_apical", "basal_apical"):
        sub = table.filter(
            (pl.col("same_type") == same_type_value)
            & (pl.col("pair_type") != "within_dend")
        )
        if sub.height == 0:
            continue
        stratum_dfs.append(_summarise(sub, same_type_value))

    if include_soma_strata and "same_soma" in table.columns:
        # Same-vs-cross-soma cuts (over between-dend pairs).
        offdiag_for_soma = table.filter(pl.col("pair_type") != "within_dend")
        for soma_label, mask_expr in (
            ("same_soma", pl.col("same_soma")),
            ("cross_soma", ~pl.col("same_soma")),
        ):
            sub = offdiag_for_soma.filter(mask_expr)
            if sub.height == 0:
                continue
            stratum_dfs.append(_summarise(sub, soma_label))

        # Full orthogonal product: soma_type_class values.
        if "soma_type_class" in table.columns:
            for stc in (
                offdiag_for_soma["soma_type_class"]
                .drop_nulls()
                .unique()
                .sort()
                .to_list()
            ):
                sub = offdiag_for_soma.filter(pl.col("soma_type_class") == stc)
                if sub.height == 0:
                    continue
                stratum_dfs.append(_summarise(sub, stc))

    if drop_within_dend_from_offdiag:
        offdiag = table.filter(pl.col("pair_type") != "within_dend")
    else:
        offdiag = table
    if offdiag.height > 0:
        stratum_dfs.append(_summarise(offdiag, "all_offdiag"))

    if not stratum_dfs:
        return pl.DataFrame(
            schema={
                unit: pl.String,
                "state": pl.String,
                "subject": pl.String,
                "exp": pl.String,
                "loc": pl.String,
                "acq": pl.String,
                "soma_id": pl.String,
                "recording_id": pl.String,
                "mean_r": pl.Float64,
                "median_r": pl.Float64,
                "mean_z": pl.Float64,
                "median_z": pl.Float64,
                "n_pairs": pl.UInt32,
                "n_pairs_with_data": pl.UInt32,
                "stratum": pl.String,
            }
        )

    return pl.concat(stratum_dfs, how="vertical").sort(unit, "stratum", "state")
