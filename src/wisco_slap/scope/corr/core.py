"""Pairwise Pearson correlation across SLAP2 synapses.

The pure-math layer of `wisco_slap.scope.corr`. Two public entry points:

- :func:`pairwise_pearson_corr` — synapse-by-synapse Pearson correlation,
  pairwise complete observations, optionally returning the per-pair count
  matrix.
- :func:`pairwise_pearson_corr_by_bouts` — apply the above to each bout in
  a polars DataFrame and return the simple mean across bouts plus the
  per-bout matrices. Backwards-compatible with the original API.

Plus one internal helper that the aggregation layer needs:

- :func:`_pairwise_pooled_sums` — the raw sufficient statistics
  ``(Sx, Sxx, Sxy, N)`` for a single time slice. Aggregators in
  ``aggregate.py`` accumulate these across bouts to compute pooled-sums
  correlations.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xarray as xr


def _to_combined(data) -> xr.DataArray:
    """Accept a ``{'dmd_1': DataArray, 'dmd_2': DataArray}`` dict or an already-combined DataArray."""
    if isinstance(data, dict):
        arrs = []
        for dmd_key in sorted(data.keys()):
            dmd_num = int(str(dmd_key).split("_")[-1])
            da = data[dmd_key]
            if "dmd" not in da.coords:
                da = da.assign_coords(
                    dmd=("syn_id", np.full(da.sizes["syn_id"], dmd_num, dtype=int))
                )
            new_ids = [f"{dmd_num}-{s}" for s in da["syn_id"].values]
            da = da.assign_coords(syn_id=new_ids)
            arrs.append(da)
        return xr.concat(arrs, dim="syn_id")
    return data


def _prep_combined(data) -> xr.DataArray:
    """Resolve ``data`` to a ``(syn_id, time)`` DataArray with channel squeezed."""
    combined = _to_combined(data)

    if "channel" in combined.dims:
        if combined.sizes["channel"] == 1:
            combined = combined.squeeze("channel", drop=True)
        else:
            raise ValueError(
                f"`data` has {combined.sizes['channel']} channels — "
                "select one before computing correlations."
            )

    if "time" not in combined.dims:
        raise ValueError("`data` must have a `time` dim.")
    if "syn_id" not in combined.dims:
        raise ValueError("`data` must have a `syn_id` dim.")

    return combined.transpose("syn_id", "time")


def _make_pair_coords(da: xr.DataArray) -> dict:
    """Mirror every 1-D ``syn_id`` coord on ``da`` onto ``(syn_1, syn_2)`` pair dims."""
    syn_ids = np.asarray(da["syn_id"].values)
    coords: dict = {"syn_1": ("syn_1", syn_ids), "syn_2": ("syn_2", syn_ids)}
    for name, c in da.coords.items():
        if name == "syn_id":
            continue
        if c.dims != ("syn_id",):
            continue
        coords[f"{name}_1"] = ("syn_1", c.values)
        coords[f"{name}_2"] = ("syn_2", c.values)
    return coords


def _pairwise_pooled_sums(data) -> dict:
    """Compute the raw pairwise sufficient statistics for a single time slice.

    Returns a dict of ``(syn_i, syn_j)``-shaped numpy arrays plus the coord
    bundle for reconstructing an :class:`xr.DataArray` later. Aggregators in
    ``aggregate.py`` accumulate ``Sx``, ``Sxx``, ``Sxy``, ``N`` across bouts
    (after per-bout centering, if requested) to compute pooled-sums correlations
    that respect bout boundaries.

    Parameters
    ----------
    data : dict[str, xr.DataArray] | xr.DataArray
        Same format as :func:`pairwise_pearson_corr`.

    Returns
    -------
    dict
        Keys: ``Sx`` (n×n), ``Sxx`` (n×n), ``Sxy`` (n×n), ``N`` (n×n int),
        ``coords`` (dict for the pair dims), ``syn_ids`` (1-D array).

    Notes
    -----
    All four sums use the pairwise-complete-observations rule: per (i, j),
    only timepoints where *both* traces are non-NaN contribute. ``Sx[i, j]``
    is therefore the sum of ``X_i`` over the joint mask, **not** over the
    full time axis — and ``Sx[i, j] != Sx[j, i]`` in general (they come
    from different joint masks). ``Sxx[i, j]`` is the sum of ``X_i**2`` over
    the joint mask of (i, j); ``Syy[i, j] = Sxx[j, i]``.
    """
    da = _prep_combined(data)
    X = np.asarray(da.values, dtype=np.float64)
    mask = ~np.isnan(X)
    X_zero = np.where(mask, X, 0.0)
    M = mask.astype(np.float64)

    return {
        "N": (M @ M.T).astype(np.int64),
        "Sx": X_zero @ M.T,
        "Sxx": (X_zero**2) @ M.T,
        "Sxy": X_zero @ X_zero.T,
        "coords": _make_pair_coords(da),
        "syn_ids": np.asarray(da["syn_id"].values),
    }


def _r_from_sums(sums: dict) -> np.ndarray:
    """Compute the Pearson r matrix from a dict of pairwise sufficient statistics."""
    N = sums["N"].astype(np.float64)
    Sx, Sxx, Sxy = sums["Sx"], sums["Sxx"], sums["Sxy"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_x = Sx / N
        mu_y = Sx.T / N
        cov = Sxy / N - mu_x * mu_y
        var_x = Sxx / N - mu_x**2
        var_y = Sxx.T / N - mu_y**2
        denom = np.sqrt(var_x * var_y)
        r = np.where((N > 1) & (denom > 0), cov / denom, np.nan)
    return r


def pairwise_pearson_corr(
    data,
    *,
    return_n: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """Pairwise Pearson correlation across synapses over time.

    Computes a synapse-by-synapse correlation matrix from a scopex DataArray.
    NaNs are handled with pairwise complete observations: for each pair (i, j),
    only timepoints where both traces are non-NaN contribute.

    Parameters
    ----------
    data : dict[str, xr.DataArray] | xr.DataArray
        Either ``{'dmd_1': DataArray, 'dmd_2': DataArray}`` as returned by
        ``wis.get.syn_dF``, or a single DataArray that already concatenates the
        DMDs along ``syn_id`` with a ``dmd`` coord attached. Must have a
        ``time`` dim and a single channel (extra channel dim of size 1 is
        squeezed; >1 errors).
    return_n : bool, keyword-only
        If True, also return the per-pair count of jointly-valid timepoints
        as a DataArray with the same dims/coords as the correlation matrix.
        Default False (backwards compatible).

    Returns
    -------
    r : xr.DataArray
        Correlation matrix with dims ``('syn_1', 'syn_2')``. ``syn_1`` /
        ``syn_2`` carry the original ``syn_id`` values; every 1-D coord that
        lived on ``syn_id`` in the input is mirrored onto both dims as
        ``<name>_1`` and ``<name>_2``. Pairs with fewer than 2 jointly-valid
        timepoints, or zero variance on either side, return NaN.
    n : xr.DataArray, optional
        Returned only when ``return_n=True``. Per-pair count of jointly-valid
        timepoints, same dims and coords as ``r``, dtype int64.
    """
    sums = _pairwise_pooled_sums(data)
    r = _r_from_sums(sums)

    r_xr = xr.DataArray(
        r,
        dims=("syn_1", "syn_2"),
        coords=sums["coords"],
        name="pearson_r",
    )
    if not return_n:
        return r_xr

    n_xr = xr.DataArray(
        sums["N"],
        dims=("syn_1", "syn_2"),
        coords=sums["coords"],
        name="n_samples",
    )
    return r_xr, n_xr


def pairwise_pearson_corr_by_bouts(
    data,
    bouts: pl.DataFrame,
    *,
    start_col: str = "start_time",
    end_col: str = "end_time",
) -> tuple[xr.DataArray, list[xr.DataArray]]:
    """Pairwise Pearson correlation per bout, plus the unweighted mean across bouts.

    For each row of ``bouts``, slices ``data`` to ``[start_col, end_col]`` along
    ``time`` and computes the synapse-by-synapse correlation matrix via
    :func:`pairwise_pearson_corr`. Returns the element-wise NaN-mean across the
    per-bout matrices and the per-bout matrices themselves, in row order.

    For a more principled average (Fisher-z weighted by per-pair sample count,
    or pooled sums of centered traces), see
    :func:`wisco_slap.scope.corr.aggregate_fisher_z` and
    :func:`wisco_slap.scope.corr.aggregate_pooled_sums`. This function is kept
    for backwards compatibility and as the simple-mean baseline.

    Parameters
    ----------
    data : dict[str, xr.DataArray] | xr.DataArray
        Either ``{'dmd_1': DataArray, 'dmd_2': DataArray}`` as returned by
        ``wis.get.syn_dF``, or a single DataArray that already concatenates the
        DMDs along ``syn_id`` with a ``dmd`` coord attached. Must have a
        ``time`` dim; a single ``channel`` is squeezed automatically.
    bouts : pl.DataFrame
        Bouts to correlate over. Each row defines one window via ``start_col``
        and ``end_col`` on the same time axis as ``data['time']`` (inclusive on
        both ends, matching ``DataArray.sel(time=slice(...))``).
    start_col, end_col : str
        Column names in ``bouts`` for the bout endpoints. Default
        ``'start_time'`` / ``'end_time'``.

    Returns
    -------
    avg : xr.DataArray
        ``(syn_1, syn_2)`` element-wise NaN-mean of the per-bout matrices, with
        the same coords as a single :func:`pairwise_pearson_corr` output.
    per_bout : list[xr.DataArray]
        One correlation matrix per row of ``bouts``, indexed in row order.
        ``per_bout[i]`` corresponds to ``bouts.row(i, named=True)``. Bouts with
        too few jointly-valid timepoints (or zero variance on either side)
        contribute all-NaN matrices that the average ignores.
    """
    if bouts.height == 0:
        raise ValueError("`bouts` is empty; nothing to correlate.")
    for col in (start_col, end_col):
        if col not in bouts.columns:
            raise ValueError(f"`bouts` is missing required column {col!r}.")

    combined = _to_combined(data)

    per_bout: list[xr.DataArray] = []
    for row in bouts.iter_rows(named=True):
        t1 = float(row[start_col])
        t2 = float(row[end_col])
        per_bout.append(pairwise_pearson_corr(combined.sel(time=slice(t1, t2))))

    stacked = np.stack([c.values for c in per_bout], axis=0)
    mean_vals = np.nanmean(stacked, axis=0)
    template = per_bout[0]
    avg = xr.DataArray(
        mean_vals,
        dims=template.dims,
        coords=template.coords,
        name="pearson_r",
    )
    return avg, per_bout
