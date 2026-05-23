"""Bout-aggregation strategies for state-conditioned correlations.

Three modes, all consuming ``(data, bouts)`` and returning the same four-array
result so callers downstream don't branch:

- :func:`aggregate_simple_mean` — unweighted ``np.nanmean`` of per-bout r.
  The current notebook behaviour. Kept as a baseline; biased near r=0 and
  gives equal weight to short and long bouts (avoid for variable-length
  bouts).
- :func:`aggregate_fisher_z` — per-bout Fisher z-transform, averaged across
  bouts, optionally weighted by ``(n−3)``. Inverse-variance-optimal for
  variable-length bouts when ``weighted=True``.
- :func:`aggregate_pooled_sums` — per-bout-centered pooled sufficient
  statistics. Equivalent to "concatenate all bouts after subtracting each
  bout's per-synapse mean, then compute one r." Each sample contributes
  with equal weight; long bouts naturally contribute more.

All three return:

- ``r`` : ``xr.DataArray`` (syn_1, syn_2) — aggregated correlation
- ``n`` : ``xr.DataArray`` (syn_1, syn_2), int64 — total joint-valid samples
- ``z`` : ``xr.DataArray`` (syn_1, syn_2) — Fisher-z (NaN where r is NaN)
- ``se_z`` : ``xr.DataArray`` (syn_1, syn_2) — std-err of z
  (``1/sqrt(n - 3)`` for pooled / weighted Fisher-z; NaN for simple_mean
  since the per-bout-mean has no closed-form variance).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xarray as xr

from .core import (
    _make_pair_coords,
    _pairwise_pooled_sums,
    _prep_combined,
    _r_from_sums,
    _to_combined,
)

_FISHER_FLOOR = 1.0 - 1e-7


def _slice_bouts(combined: xr.DataArray, bouts: pl.DataFrame, start_col: str, end_col: str):
    """Yield per-bout slices of ``combined`` along ``time`` (inclusive ends)."""
    for row in bouts.iter_rows(named=True):
        t1 = float(row[start_col])
        t2 = float(row[end_col])
        yield combined.sel(time=slice(t1, t2))


def _r_to_z(r: np.ndarray) -> np.ndarray:
    """Fisher z-transform, with a small floor to keep ``arctanh`` finite."""
    r_safe = np.clip(r, -_FISHER_FLOOR, _FISHER_FLOOR)
    return np.arctanh(r_safe)


def _wrap_pair(arr: np.ndarray, coords: dict, name: str) -> xr.DataArray:
    return xr.DataArray(arr, dims=("syn_1", "syn_2"), coords=coords, name=name)


def _empty_result(coords: dict) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    n_syn = len(coords["syn_1"][1])
    nan_mat = np.full((n_syn, n_syn), np.nan)
    n_mat = np.zeros((n_syn, n_syn), dtype=np.int64)
    return (
        _wrap_pair(nan_mat.copy(), coords, "pearson_r"),
        _wrap_pair(n_mat, coords, "n_samples"),
        _wrap_pair(nan_mat.copy(), coords, "fisher_z"),
        _wrap_pair(nan_mat.copy(), coords, "se_fisher_z"),
    )


def aggregate_simple_mean(
    data,
    bouts: pl.DataFrame,
    *,
    start_col: str = "start_time",
    end_col: str = "end_time",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Unweighted ``nanmean`` of per-bout Pearson r.

    Backwards-compatible with :func:`pairwise_pearson_corr_by_bouts` (yields the
    same ``r`` matrix). ``n`` is the **sum** of per-bout joint-valid counts.
    ``z = arctanh(r_avg)`` for table-schema compatibility, and ``se_z`` is NaN
    (the simple-mean of r has no closed-form variance — use
    :func:`aggregate_fisher_z` instead if you want one).
    """
    combined = _to_combined(data)
    template = _prep_combined(combined)
    coords = _make_pair_coords(template)

    if bouts.height == 0:
        return _empty_result(coords)

    rs: list[np.ndarray] = []
    ns: list[np.ndarray] = []
    for sl in _slice_bouts(combined, bouts, start_col, end_col):
        sums = _pairwise_pooled_sums(sl)
        rs.append(_r_from_sums(sums))
        ns.append(sums["N"])

    r_avg = np.nanmean(np.stack(rs, axis=0), axis=0)
    n_total = np.sum(np.stack(ns, axis=0), axis=0)

    return (
        _wrap_pair(r_avg, coords, "pearson_r"),
        _wrap_pair(n_total.astype(np.int64), coords, "n_samples"),
        _wrap_pair(_r_to_z(r_avg), coords, "fisher_z"),
        _wrap_pair(np.full_like(r_avg, np.nan), coords, "se_fisher_z"),
    )


def aggregate_fisher_z(
    data,
    bouts: pl.DataFrame,
    *,
    weighted: bool = True,
    start_col: str = "start_time",
    end_col: str = "end_time",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Per-bout Fisher-z averaging, optionally weighted by per-pair (n − 3).

    For each bout: compute pairwise r and per-pair joint-valid count n, then
    ``z_b = arctanh(r_b)``. Aggregate across bouts:

    - ``weighted=True`` (recommended for variable-length bouts):
      ``z_avg[i,j] = Σ_b (n_b[i,j] − 3) · z_b[i,j] / Σ_b (n_b[i,j] − 3)``,
      with ``se_z = 1 / sqrt(Σ_b (n_b − 3))``.
    - ``weighted=False``: simple mean of ``z`` across bouts.

    Bouts with ``n_b ≤ 3`` for a pair contribute nothing to that pair.

    Returns ``(r, n, z, se_z)`` where ``r = tanh(z_avg)`` and ``n`` is the sum
    of per-bout joint-valid counts.
    """
    combined = _to_combined(data)
    template = _prep_combined(combined)
    coords = _make_pair_coords(template)
    n_syn = template.sizes["syn_id"]

    if bouts.height == 0:
        return _empty_result(coords)

    weight_sum = np.zeros((n_syn, n_syn), dtype=np.float64)
    weighted_z = np.zeros((n_syn, n_syn), dtype=np.float64)
    n_total = np.zeros((n_syn, n_syn), dtype=np.int64)
    bout_count = np.zeros((n_syn, n_syn), dtype=np.int64)

    for sl in _slice_bouts(combined, bouts, start_col, end_col):
        sums = _pairwise_pooled_sums(sl)
        r_b = _r_from_sums(sums)
        n_b = sums["N"].astype(np.int64)
        z_b = _r_to_z(r_b)

        valid_pair = (n_b > 3) & np.isfinite(z_b)
        eff_n = np.where(valid_pair, (n_b - 3).astype(np.float64), 0.0)
        w_b = eff_n if weighted else valid_pair.astype(np.float64)

        contrib = np.where(valid_pair, w_b * z_b, 0.0)
        weighted_z += contrib
        weight_sum += w_b
        n_total += np.where(valid_pair, n_b, 0)
        bout_count += valid_pair.astype(np.int64)

    with np.errstate(invalid="ignore", divide="ignore"):
        z_avg = np.where(weight_sum > 0, weighted_z / weight_sum, np.nan)

    if weighted:
        with np.errstate(invalid="ignore", divide="ignore"):
            se_z = np.where(weight_sum > 0, 1.0 / np.sqrt(weight_sum), np.nan)
    else:
        # Unweighted: SE of the mean of K iid normal-ish z's is std/√K.
        # Without per-bout z stored, approximate with 1/sqrt(Σ(n_b−3))
        # (which is the precision-weighted bound — conservative-ish).
        with np.errstate(invalid="ignore", divide="ignore"):
            se_z = np.where(bout_count > 0, 1.0 / np.sqrt(weight_sum.clip(min=1.0)), np.nan)

    r_avg = np.tanh(z_avg)

    return (
        _wrap_pair(r_avg, coords, "pearson_r"),
        _wrap_pair(n_total, coords, "n_samples"),
        _wrap_pair(z_avg, coords, "fisher_z"),
        _wrap_pair(se_z, coords, "se_fisher_z"),
    )


def aggregate_pooled_sums(
    data,
    bouts: pl.DataFrame,
    *,
    center_per_bout: bool = True,
    start_col: str = "start_time",
    end_col: str = "end_time",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Pooled per-bout-centered sums. The "concatenate all data" mode.

    For each bout, build per-pair sufficient statistics
    ``Sx, Sxx, Sxy, N`` after subtracting the bout's per-synapse mean
    (``center_per_bout=True``, default). Pool across bouts:
    ``Sxx_pool = Σ_b Sxx_b``, etc. Compute one r per pair from the pooled
    sums.

    With ``center_per_bout=True`` (recommended), this is mathematically
    equivalent to: concatenate every bout's data, subtract each bout's
    per-synapse mean within its segment, then compute a single Pearson r —
    i.e. a within-bout-detrended single-shot correlation. With
    ``center_per_bout=False``, the pooled means cross bout boundaries.

    Returns ``(r, n, z, se_z)`` with ``z = arctanh(r)`` and
    ``se_z = 1 / sqrt(n − 3)`` (single-r asymptotic SE on Fisher-z).
    """
    combined = _to_combined(data)
    template = _prep_combined(combined)
    coords = _make_pair_coords(template)
    n_syn = template.sizes["syn_id"]

    if bouts.height == 0:
        return _empty_result(coords)

    Sxx_pool = np.zeros((n_syn, n_syn), dtype=np.float64)
    Sxy_pool = np.zeros((n_syn, n_syn), dtype=np.float64)
    Sx_pool = np.zeros((n_syn, n_syn), dtype=np.float64)
    N_pool = np.zeros((n_syn, n_syn), dtype=np.int64)

    for sl in _slice_bouts(combined, bouts, start_col, end_col):
        # Skip bouts that ended up empty after slicing (shouldn't normally happen).
        if sl.sizes.get("time", 0) == 0:
            continue
        if center_per_bout:
            # Subtract each synapse's nanmean over this bout's time, then
            # call _pairwise_pooled_sums on the centered data. Sx contribution
            # is then ≈ 0 per pair (up to NaN-mask asymmetry at the pair level).
            X = np.asarray(_prep_combined(sl).values, dtype=np.float64)
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(X, axis=1, keepdims=True)
            X_centered = X - mu
            sl_centered = xr.DataArray(
                X_centered,
                dims=("syn_id", "time"),
                coords=_prep_combined(sl).coords,
            )
            sums = _pairwise_pooled_sums(sl_centered)
        else:
            sums = _pairwise_pooled_sums(sl)

        Sxx_pool += sums["Sxx"]
        Sxy_pool += sums["Sxy"]
        Sx_pool += sums["Sx"]
        N_pool += sums["N"]

    with np.errstate(invalid="ignore", divide="ignore"):
        N_f = N_pool.astype(np.float64)
        mu_x = Sx_pool / N_f
        mu_y = Sx_pool.T / N_f
        cov = Sxy_pool / N_f - mu_x * mu_y
        var_x = Sxx_pool / N_f - mu_x**2
        var_y = Sxx_pool.T / N_f - mu_y**2
        denom = np.sqrt(var_x * var_y)
        r = np.where((N_pool > 1) & (denom > 0), cov / denom, np.nan)

    z = _r_to_z(r)
    with np.errstate(invalid="ignore", divide="ignore"):
        se_z = np.where(N_pool > 3, 1.0 / np.sqrt(N_pool - 3), np.nan)

    return (
        _wrap_pair(r, coords, "pearson_r"),
        _wrap_pair(N_pool, coords, "n_samples"),
        _wrap_pair(z, coords, "fisher_z"),
        _wrap_pair(se_z, coords, "se_fisher_z"),
    )
