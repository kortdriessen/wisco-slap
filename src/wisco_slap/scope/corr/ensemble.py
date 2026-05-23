"""Bout-to-bout ensemble and pair-rank stability.

Two complementary measures of "is the co-activity pattern the same across
time?", separating *how much* synapses co-vary (Pearson magnitude) from
*which* synapses co-vary together (the structure of the correlation matrix
or the leading PCA direction):

- :func:`per_bout_pair_r_vectors` and :func:`pair_rank_stability`:
  Spearman-rank correlation between the (n_pairs)-long upper-triangle
  z-vectors of two bouts. If the same pairs are consistently the most
  correlated, this is high. Independent of overall mean magnitude.

- :func:`per_bout_pca` and :func:`ensemble_subspace_similarity`: per-bout
  PCA on the synapse-by-time matrix; principal-angle similarity between
  the top-K subspaces of two bouts. If the dominant co-activity directions
  point the same way in synapse-space, this is near 1; if orthogonal, near
  0.

Both pair with bout temporal-context annotation (see
:mod:`wisco_slap.scope.corr.temporal`) so analyses can ask whether stability
is higher within a state than across, or whether it changes with cumulative
state-time.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import polars as pl
from scipy import stats as _stats

from ... import get as _get
from ...util.validity.mask import validity_mask
from .aggregate import _r_to_z
from .bouts import all_segments_in_state, fixed_valid_bouts, state_hypno_bouts
from .core import _to_combined, pairwise_pearson_corr
from .state_compare import _load_dn
from .temporal import annotate_bout_temporal_context

_BOUT_BUILDERS = {
    "fixed_valid": fixed_valid_bouts,
    "all_segments": all_segments_in_state,
    "hypno_bouts": state_hypno_bouts,
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


# -------------------------------------------------- per-bout r vectors


def per_bout_pair_r_vectors(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    *,
    states: tuple[str, ...] = ("NREM", "Wake"),
    bout_strategy: str = "all_segments",
    bout_kwargs: dict | None = None,
    min_bout_length_s: float = 30.0,
    channel: int = 0,
    trace: str = "denoised",
    validity_mode: str = "all",
) -> tuple[pl.DataFrame, np.ndarray]:
    """Per-bout flattened upper-triangle z-vectors for one (cell).

    For each bout meeting ``min_bout_length_s``, computes
    :func:`pairwise_pearson_corr` and stacks the i<j Fisher-z values into a
    ``(n_bouts, n_pairs)`` ndarray. Returns the array alongside per-bout
    metadata, including the temporal-context columns from
    :func:`annotate_bout_temporal_context`. Pairs with NaN r in any bout
    contribute NaN columns; downstream rank-stability handles NaN by
    pairwise complete observations.

    Parameters
    ----------
    subject, exp, loc, acq, soma_id : str
    states : tuple of str
    bout_strategy : str
    bout_kwargs : dict | None
    min_bout_length_s : float
        Bouts shorter than this are excluded.
    channel, trace, validity_mode : standard.

    Returns
    -------
    bout_meta : pl.DataFrame
        One row per bout; columns include ``state``, ``start_time``,
        ``end_time``, ``valid_duration``, ``bout_idx``,
        ``bout_index_in_state``, ``cum_state_time_at_start``, ``cell_id``,
        ``recording_id``, ``subject``, ``soma_id``.
    z_matrix : np.ndarray
        Shape ``(n_bouts, n_pairs)``. Order along axis 0 matches
        ``bout_meta``; order along axis 1 is the i<j upper triangle.
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

    builder = _BOUT_BUILDERS[bout_strategy]
    eff_kwargs = _eff_bout_kwargs(bout_strategy, bout_kwargs)

    parts = []
    for state in states:
        b = builder(hypno, val_mask, state, **eff_kwargs)
        if b.height > 0:
            parts.append(b)
    if not parts:
        return _empty_bout_meta(), np.zeros((0, 0))
    bouts_all = pl.concat(parts, how="vertical")
    bouts_annot = annotate_bout_temporal_context(bouts_all, use_valid_duration=True)
    bouts_annot = bouts_annot.filter(pl.col("valid_duration") >= min_bout_length_s)
    if bouts_annot.height == 0:
        return _empty_bout_meta(), np.zeros((0, 0))
    bouts_annot = bouts_annot.sort(["start_time"])

    combined = _to_combined(dn)
    n_syn = dn.sizes["syn_id"]
    iu, ju = np.triu_indices(n_syn, k=1)
    n_pairs = len(iu)

    z_matrix = np.full((bouts_annot.height, n_pairs), np.nan, dtype=np.float64)
    n_samples = np.zeros(bouts_annot.height, dtype=np.int64)
    for i, b in enumerate(bouts_annot.iter_rows(named=True)):
        sl = combined.sel(time=slice(float(b["start_time"]), float(b["end_time"])))
        if sl.sizes.get("time", 0) == 0:
            continue
        r_xr, n_xr = pairwise_pearson_corr(sl, return_n=True)
        z_matrix[i, :] = _r_to_z(r_xr.values[iu, ju])
        if n_xr.values.size:
            n_samples[i] = int(np.nanmax(n_xr.values))

    bout_meta = bouts_annot.with_columns(
        pl.lit(subject).alias("subject"),
        pl.lit(exp).alias("exp"),
        pl.lit(loc).alias("loc"),
        pl.lit(acq).alias("acq"),
        pl.lit(soma_id).alias("soma_id"),
        pl.lit(cell_id).alias("cell_id"),
        pl.lit(recording_id).alias("recording_id"),
        pl.Series("n_samples", n_samples, dtype=pl.Int64),
    )
    return bout_meta, z_matrix


def _empty_bout_meta() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "state": pl.String, "start_time": pl.Float64, "end_time": pl.Float64,
            "valid_duration": pl.Float64, "wall_duration": pl.Float64,
            "bout_idx": pl.Int64, "bout_duration": pl.Float64,
            "cum_state_time_at_end": pl.Float64,
            "cum_state_time_at_start": pl.Float64,
            "bout_index_in_state": pl.Int64,
            "prev_bout_gap_s": pl.Float64,
            "subject": pl.String, "exp": pl.String, "loc": pl.String,
            "acq": pl.String, "soma_id": pl.String, "cell_id": pl.String,
            "recording_id": pl.String, "n_samples": pl.Int64,
        }
    )


# --------------------------------------------------- pair-rank stability


def pair_rank_stability(
    z_matrix: np.ndarray,
    bout_meta: pl.DataFrame,
    *,
    lags: tuple[int, ...] = (1,),
) -> pl.DataFrame:
    """Spearman correlation between bouts' z-vectors at given bout-index lags.

    For every ordered pair of bouts ``(i, j)`` where ``j = i + lag`` for some
    ``lag ∈ lags``, computes the Spearman correlation between
    ``z_matrix[i]`` and ``z_matrix[j]`` over pairs that are non-NaN in both.

    Parameters
    ----------
    z_matrix : np.ndarray
        ``(n_bouts, n_pairs)`` from :func:`per_bout_pair_r_vectors`.
    bout_meta : pl.DataFrame
        One row per bout, in the same order as ``z_matrix``.
    lags : tuple of int
        Bout-index lags to compute. Default ``(1,)`` (consecutive bouts).
        Use e.g. ``(1, 2, 5, 10)`` to look at how stability decays with
        bout separation.

    Returns
    -------
    pl.DataFrame
        One row per (bout_i, bout_j, lag). Columns: ``bout_i``, ``bout_j``,
        ``lag``, ``state_i``, ``state_j``, ``state_pair``
        (e.g. ``"NREM->NREM"``, ``"NREM->Wake"``), ``cell_id``, ``subject``,
        ``soma_id``, ``time_gap_s`` (wall-clock seconds between bouts),
        ``cum_state_gap_s`` (cumulative valid-state-time gap, NaN if states
        differ), ``spearman_r``, ``n_pairs_used`` (non-NaN pair count).
    """
    n_bouts, n_pairs = z_matrix.shape
    if n_bouts == 0:
        return _empty_rank_df()

    rows: list[dict] = []
    starts = bout_meta["start_time"].to_numpy()
    states = bout_meta["state"].to_list()
    cum_at_end = bout_meta["cum_state_time_at_end"].to_numpy()
    cum_at_start = bout_meta["cum_state_time_at_start"].to_numpy()

    cell_id = bout_meta["cell_id"][0] if bout_meta.height else None
    subject = bout_meta["subject"][0] if bout_meta.height else None
    soma_id = bout_meta["soma_id"][0] if bout_meta.height else None

    for lag in lags:
        for i in range(n_bouts - lag):
            j = i + lag
            zi = z_matrix[i]
            zj = z_matrix[j]
            mask = np.isfinite(zi) & np.isfinite(zj)
            n_used = int(mask.sum())
            if n_used < 5:
                continue
            try:
                rho, _p = _stats.spearmanr(zi[mask], zj[mask])
            except Exception:  # noqa: BLE001
                rho = np.nan
            state_i = states[i]
            state_j = states[j]
            state_pair = f"{state_i}->{state_j}"
            time_gap = float(starts[j] - starts[i])
            cum_gap = (
                float(cum_at_start[j] - cum_at_end[i])
                if state_i == state_j
                else float("nan")
            )
            rows.append({
                "bout_i": i, "bout_j": j, "lag": int(lag),
                "state_i": state_i, "state_j": state_j,
                "state_pair": state_pair,
                "cell_id": cell_id, "subject": subject, "soma_id": soma_id,
                "time_gap_s": time_gap,
                "cum_state_gap_s": cum_gap,
                "spearman_r": float(rho) if rho is not None else float("nan"),
                "n_pairs_used": n_used,
            })
    if not rows:
        return _empty_rank_df()
    return pl.DataFrame(rows)


def _empty_rank_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "bout_i": pl.Int64, "bout_j": pl.Int64, "lag": pl.Int64,
            "state_i": pl.String, "state_j": pl.String, "state_pair": pl.String,
            "cell_id": pl.String, "subject": pl.String, "soma_id": pl.String,
            "time_gap_s": pl.Float64, "cum_state_gap_s": pl.Float64,
            "spearman_r": pl.Float64, "n_pairs_used": pl.Int64,
        }
    )


# ----------------------------------------------------------- per-bout PCA


def per_bout_pca(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    *,
    states: tuple[str, ...] = ("NREM", "Wake"),
    n_components: int = 5,
    standardize: bool = True,
    min_bout_length_s: float = 30.0,
    bout_strategy: str = "all_segments",
    bout_kwargs: dict | None = None,
    channel: int = 0,
    trace: str = "denoised",
    validity_mode: str = "all",
) -> tuple[pl.DataFrame, list[np.ndarray], list[np.ndarray]]:
    """Per-bout PCA on synapse activity for one (cell).

    For each bout meeting ``min_bout_length_s``: optionally z-score each
    synapse over the bout's time, then run truncated SVD (PCA) and return
    the top-``n_components`` synapse-loading vectors plus the explained
    variance ratios. Bouts where any synapse has zero variance over the
    bout are skipped (no rotation defined).

    Parameters
    ----------
    subject, exp, loc, acq, soma_id : str
    states : tuple of str
    n_components : int
        Number of PCs to keep per bout. Defaults to 5.
    standardize : bool
        If True (default), z-score each synapse over the bout.
        If False, only mean-center.
    min_bout_length_s : float
    bout_strategy, bout_kwargs : standard.
    channel, trace, validity_mode : standard.

    Returns
    -------
    bout_meta : pl.DataFrame
        One row per surviving bout; same shape as
        :func:`per_bout_pair_r_vectors`'s ``bout_meta``.
    loadings : list of np.ndarray
        Length ``n_bouts``. Each entry has shape ``(n_components, n_synapses)``
        — the right-singular vectors (synapse loadings).
    explained_variance_ratio : list of np.ndarray
        Length ``n_bouts``. Each entry has shape ``(n_components,)``.
    """
    if bout_strategy not in _BOUT_BUILDERS:
        raise ValueError(
            f"bout_strategy must be one of {list(_BOUT_BUILDERS)}, got {bout_strategy!r}"
        )

    recording_id = f"{subject}|{exp}|{loc}|{acq}"
    cell_id = f"{recording_id}|{soma_id}"

    dn = _load_dn(subject, exp, loc, acq, soma_id, channel=channel, trace=trace)
    if dn.sizes.get("syn_id", 0) < 2:
        raise ValueError("Need ≥2 synapses for PCA.")
    val_mask = validity_mask(dn, mode=validity_mode)
    hypno, *_ = _get.acq_sleep_coverage(subject, exp, loc, acq)

    builder = _BOUT_BUILDERS[bout_strategy]
    eff_kwargs = _eff_bout_kwargs(bout_strategy, bout_kwargs)

    parts = []
    for state in states:
        b = builder(hypno, val_mask, state, **eff_kwargs)
        if b.height > 0:
            parts.append(b)
    if not parts:
        return _empty_bout_meta(), [], []
    bouts_all = pl.concat(parts, how="vertical")
    bouts_annot = annotate_bout_temporal_context(bouts_all, use_valid_duration=True)
    bouts_annot = bouts_annot.filter(pl.col("valid_duration") >= min_bout_length_s)
    if bouts_annot.height == 0:
        return _empty_bout_meta(), [], []
    bouts_annot = bouts_annot.sort(["start_time"])

    combined = _to_combined(dn)
    n_syn = dn.sizes["syn_id"]
    k = min(n_components, n_syn)

    keep_idx: list[int] = []
    loadings: list[np.ndarray] = []
    evrs: list[np.ndarray] = []

    for i, b in enumerate(bouts_annot.iter_rows(named=True)):
        sl = combined.sel(time=slice(float(b["start_time"]), float(b["end_time"])))
        if sl.sizes.get("time", 0) < 4:
            continue
        X = np.asarray(sl.values, dtype=np.float64)  # (syn_id, time)
        # Drop NaN columns (timepoints) so the SVD is on the dense subset.
        col_mask = ~np.isnan(X).any(axis=0)
        Xv = X[:, col_mask]
        if Xv.shape[1] < 4:
            continue
        # Per-synapse mean-center; optionally divide by std.
        mu = Xv.mean(axis=1, keepdims=True)
        Xc = Xv - mu
        if standardize:
            sd = Xc.std(axis=1, keepdims=True, ddof=1)
            if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
                continue
            Xc = Xc / sd
        # SVD: Xc = U S Vt with Xc shape (n_syn, n_time). U columns are PCs in
        # synapse-space; we want the right-singular orientation in synapse-space,
        # i.e. U[:, :k].T (each row a PC's loading over synapses).
        try:
            U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if S.size < 1:
            continue
        kk = min(k, S.size)
        load = U[:, :kk].T  # (kk, n_syn)
        var = (S**2) / max(Xc.shape[1] - 1, 1)
        evr = var[:kk] / var.sum() if var.sum() > 0 else np.full(kk, np.nan)
        # Pad to n_components if needed for stack uniformity.
        if kk < n_components:
            pad = np.full((n_components - kk, n_syn), np.nan)
            load = np.vstack([load, pad])
            evr = np.concatenate([evr, np.full(n_components - kk, np.nan)])
        keep_idx.append(i)
        loadings.append(load)
        evrs.append(evr)

    if not keep_idx:
        return _empty_bout_meta(), [], []

    bout_meta = (
        bouts_annot.with_row_index("_orig_idx")
        .filter(pl.col("_orig_idx").is_in(keep_idx))
        .drop("_orig_idx")
        .with_columns(
            pl.lit(subject).alias("subject"),
            pl.lit(exp).alias("exp"),
            pl.lit(loc).alias("loc"),
            pl.lit(acq).alias("acq"),
            pl.lit(soma_id).alias("soma_id"),
            pl.lit(cell_id).alias("cell_id"),
            pl.lit(recording_id).alias("recording_id"),
            pl.lit(0).cast(pl.Int64).alias("n_samples"),
        )
    )
    return bout_meta, loadings, evrs


# ------------------------------------------------ subspace similarity


def _grassmann_similarity(A: np.ndarray, B: np.ndarray, *, k: int) -> float:
    """Mean cos² of principal angles between top-k subspaces of A and B.

    A, B are ``(n_components, n_features)``. Rows that contain NaN are
    dropped before computation; returns NaN if fewer than ``k`` valid rows
    on either side.
    """
    A_valid = A[~np.isnan(A).any(axis=1)]
    B_valid = B[~np.isnan(B).any(axis=1)]
    if A_valid.shape[0] < k or B_valid.shape[0] < k:
        return float("nan")
    A_top = A_valid[:k].T  # (n_features, k)
    B_top = B_valid[:k].T  # (n_features, k)
    # Orthonormalise each subspace (rows of A_top columns might already be
    # orthonormal coming from SVD's U).
    Q_a, _ = np.linalg.qr(A_top)
    Q_b, _ = np.linalg.qr(B_top)
    # Singular values of Q_a^T Q_b are cos(principal angles).
    sigmas = np.linalg.svd(Q_a.T @ Q_b, compute_uv=False)
    # Numerical clamp.
    sigmas = np.clip(sigmas, -1.0, 1.0)
    return float(np.mean(sigmas**2))


def ensemble_subspace_similarity(
    loadings: list[np.ndarray],
    bout_meta: pl.DataFrame,
    *,
    k: int = 3,
) -> pl.DataFrame:
    """Pairwise principal-angle similarity between bouts' top-k subspaces.

    For every ordered pair ``(i, j)`` with ``i < j``, compute the mean
    ``cos²`` of the principal angles between bouts' top-``k`` PC subspaces.
    A value of 1 means identical subspaces; 0 means orthogonal.

    Parameters
    ----------
    loadings : list of np.ndarray
        Per-bout PC loadings from :func:`per_bout_pca`.
    bout_meta : pl.DataFrame
        Per-bout metadata aligned with ``loadings``.
    k : int
        Subspace dimension. Default 3.

    Returns
    -------
    pl.DataFrame
        One row per (bout_i, bout_j) with i<j. Columns: ``bout_i``, ``bout_j``,
        ``state_i``, ``state_j``, ``state_pair``, ``time_gap_s``,
        ``cum_state_gap_s`` (NaN when states differ), ``cell_id``,
        ``subject``, ``soma_id``, ``grassmann_sim`` (mean cos²),
        ``k_used`` (effective k after dropping NaN-padded rows).
    """
    n = len(loadings)
    if n == 0 or bout_meta.height != n:
        return _empty_subspace_df()

    starts = bout_meta["start_time"].to_numpy()
    states = bout_meta["state"].to_list()
    cum_at_end = bout_meta["cum_state_time_at_end"].to_numpy()
    cum_at_start = bout_meta["cum_state_time_at_start"].to_numpy()
    cell_id = bout_meta["cell_id"][0]
    subject = bout_meta["subject"][0]
    soma_id = bout_meta["soma_id"][0]

    rows: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = _grassmann_similarity(loadings[i], loadings[j], k=k)
            time_gap = float(starts[j] - starts[i])
            cum_gap = (
                float(cum_at_start[j] - cum_at_end[i])
                if states[i] == states[j]
                else float("nan")
            )
            rows.append({
                "bout_i": i, "bout_j": j,
                "state_i": states[i], "state_j": states[j],
                "state_pair": f"{states[i]}->{states[j]}",
                "time_gap_s": time_gap, "cum_state_gap_s": cum_gap,
                "cell_id": cell_id, "subject": subject, "soma_id": soma_id,
                "grassmann_sim": sim,
                "k_used": int(k),
            })
    if not rows:
        return _empty_subspace_df()
    return pl.DataFrame(rows)


def _empty_subspace_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "bout_i": pl.Int64, "bout_j": pl.Int64,
            "state_i": pl.String, "state_j": pl.String,
            "state_pair": pl.String,
            "time_gap_s": pl.Float64, "cum_state_gap_s": pl.Float64,
            "cell_id": pl.String, "subject": pl.String, "soma_id": pl.String,
            "grassmann_sim": pl.Float64, "k_used": pl.Int64,
        }
    )


# ----------------------------------------------------- batch helpers


def per_bout_pair_r_vectors_multi(
    rows: pl.DataFrame | Iterable[dict],
    **kwargs,
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Run :func:`per_bout_pair_r_vectors` over many cells.

    Returns one combined ``bout_meta`` (with all per-cell rows concatenated)
    and a dict mapping ``cell_id`` → ``(n_bouts_per_cell, n_pairs)`` z-matrix.
    The pair indexing is intrinsic to each cell (different n_synapses across
    cells), so the matrices are not concatenated across cells.
    """
    if isinstance(rows, pl.DataFrame):
        iterator = list(rows.iter_rows(named=True))
    else:
        iterator = list(rows)
    metas: list[pl.DataFrame] = []
    z_by_cell: dict[str, np.ndarray] = {}
    for row in iterator:
        try:
            meta, z = per_bout_pair_r_vectors(
                row["subject"], row["exp"], row["loc"], row["acq"], row["soma_id"],
                **kwargs,
            )
            if meta.height > 0:
                metas.append(meta)
                z_by_cell[meta["cell_id"][0]] = z
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"per_bout_pair_r_vectors failed for "
                f"{row.get('subject')}/{row.get('exp')}/{row.get('loc')}/"
                f"{row.get('acq')}/{row.get('soma_id')}: "
                f"{type(e).__name__}: {e}",
                stacklevel=2,
            )
    if not metas:
        return _empty_bout_meta(), {}
    return pl.concat(metas, how="vertical"), z_by_cell


def per_bout_pca_multi(
    rows: pl.DataFrame | Iterable[dict],
    **kwargs,
) -> tuple[pl.DataFrame, dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    """Run :func:`per_bout_pca` over many cells.

    Returns ``(combined_meta, loadings_by_cell, evr_by_cell)``.
    """
    if isinstance(rows, pl.DataFrame):
        iterator = list(rows.iter_rows(named=True))
    else:
        iterator = list(rows)
    metas: list[pl.DataFrame] = []
    loadings_by_cell: dict[str, list[np.ndarray]] = {}
    evr_by_cell: dict[str, list[np.ndarray]] = {}
    for row in iterator:
        try:
            meta, loadings, evrs = per_bout_pca(
                row["subject"], row["exp"], row["loc"], row["acq"], row["soma_id"],
                **kwargs,
            )
            if meta.height > 0:
                metas.append(meta)
                loadings_by_cell[meta["cell_id"][0]] = loadings
                evr_by_cell[meta["cell_id"][0]] = evrs
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"per_bout_pca failed for "
                f"{row.get('subject')}/{row.get('exp')}/{row.get('loc')}/"
                f"{row.get('acq')}/{row.get('soma_id')}: "
                f"{type(e).__name__}: {e}",
                stacklevel=2,
            )
    if not metas:
        return _empty_bout_meta(), {}, {}
    return pl.concat(metas, how="vertical"), loadings_by_cell, evr_by_cell
