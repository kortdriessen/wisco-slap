"""Loading functions for scopex (xarray/zarr) datasets.

All scopex zarrs are chunked as ``(1, 1, n_time)`` — one channel, one
synapse/soma per chunk.  This means selecting a subset of syn_ids or
channels reads only the necessary chunks from disk, so passing ``syn_ids``
or ``channels`` is very efficient.
"""

import os
from collections.abc import Sequence

import numpy as np
import polars as pl
import xarray as xr
from slap2_py.core.xarr_summ import load_xr_from_zarr

from wisco_slap.defs import anmat_root
from wisco_slap.meta.get import ephys_offset as _get_ephys_offset


def _scopex_dir(subject: str, exp: str, loc: str, acq: str) -> str:
    return os.path.join(anmat_root, subject, exp, "scopex", f"{loc}--{acq}")


def _build_sel(
    ids: Sequence[int] | None,
    id_dim: str,
    channels: Sequence[int] | int | None,
) -> dict | None:
    """Build a sel dict from optional id/channel subsets."""
    sel = {}
    if ids is not None:
        sel[id_dim] = list(ids)
    if channels is not None:
        sel["channel"] = channels
    return sel or None


def _resolve_dmd(dmd: int | None) -> str | None:
    """Convert user-facing dmd int (1 or 2) to zarr group key, or None for both."""
    if dmd is None:
        return None
    return f"dmd_{dmd}"


def _apply_ephys_offset(
    result: dict[str, xr.DataArray],
    subject: str,
    exp: str,
    loc: str,
    acq: str,
) -> dict[str, xr.DataArray]:
    """Shift the ``time`` coordinate by the ephys offset so that times are
    aligned to the ephys clock."""
    offset = _get_ephys_offset(subject, exp, loc, acq)
    return {
        key: da.assign_coords(time=da.coords["time"].values + offset)
        for key, da in result.items()
    }


def merge_syn_info_to_scopex(
    scopex: dict[str, xr.DataArray],
    idf: pl.DataFrame,
) -> dict[str, xr.DataArray]:
    """Merge synapse info labels onto scopex DataArrays as non-dimension coordinates.

    For each DMD key in *scopex*, filters *idf* to that DMD and assigns each
    metadata column (``dend-ID``, ``soma-ID``, ``soma-depth``, etc.) as a
    coordinate on the ``syn_id`` dimension.  This lets you use
    ``da.groupby('dend-ID')``, boolean indexing, and similar xarray operations
    with the synapse metadata attached.

    Parameters
    ----------
    scopex : dict[str, xr.DataArray]
        Scopex dict as returned by :func:`syn_dF`, :func:`syn_F0`, etc.
    idf : pl.DataFrame
        Synapse info DataFrame as returned by
        :func:`wisco_slap.get.synid_labels`.

    Returns
    -------
    dict[str, xr.DataArray]
        Same structure as input, with additional non-dimension coordinates on
        the ``syn_id`` dimension.
    """
    info_cols = [c for c in idf.columns if c not in ("syn_id", "dmd")]
    result = {}

    for key, da in scopex.items():
        dmd_num = int(key.split("_")[1])
        dmd_info = idf.filter(pl.col("dmd") == dmd_num).sort("syn_id")

        # Build a target frame of the DataArray's syn_ids and left-join the info
        da_syn_ids = da.coords["syn_id"].values
        target = pl.DataFrame({"syn_id": da_syn_ids.astype(np.int32)})
        aligned = target.join(
            dmd_info.select("syn_id", *info_cols), on="syn_id", how="left"
        )

        new_coords: dict[str, tuple[str, np.ndarray]] = {}
        for col in info_cols:
            new_coords[col] = ("syn_id", aligned[col].to_numpy())

        result[key] = da.assign_coords(new_coords)

    return result


def syn_dF(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    trace: str = "denoised",
    dmd: int | None = None,
    syn_ids: Sequence[int] | None = None,
    channels: Sequence[int] | int | None = None,
    merge_info: bool = True,
    apply_ephys_offset: bool = True,
) -> dict[str, xr.DataArray]:
    """Load synaptic dF/F traces from scopex.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    trace : str
        Trace type: ``"denoised"`` or ``"ls"``.
    dmd : int or None
        DMD number (1 or 2).  None loads both.
    syn_ids : sequence of int or None
        Subset of synapse IDs to load.  None loads all.
    channels : sequence of int, int, or None
        Subset of channels to load.  None loads all.
    merge_info : bool
        If True, load synapse labels via
        :func:`~wisco_slap.get.synid_labels` and attach them as
        non-dimension coordinates on the ``syn_id`` dimension.
    apply_ephys_offset : bool
        If True (default), shift the ``time`` coordinate by the ephys
        offset so that times are aligned to the ephys clock.

    Returns
    -------
    dict[str, xr.DataArray]
        Mapping from ``"dmd_1"`` / ``"dmd_2"`` to DataArray with dims
        ``(channel, syn_id, time)``.
    """
    path = os.path.join(_scopex_dir(subject, exp, loc, acq), f"syn_dF-{trace}.zarr")
    result = load_xr_from_zarr(
        path, dmd=_resolve_dmd(dmd), sel=_build_sel(syn_ids, "syn_id", channels)
    )
    if merge_info:
        from wisco_slap.get._get_syn_info import synid_labels

        idf = synid_labels(subject, exp, loc, acq)
        if idf is not None:
            result = merge_syn_info_to_scopex(result, idf)
    if apply_ephys_offset:
        result = _apply_ephys_offset(result, subject, exp, loc, acq)
    return result


def syn_F0(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    dmd: int | None = None,
    syn_ids: Sequence[int] | None = None,
    channels: Sequence[int] | int | None = None,
    apply_ephys_offset: bool = True,
) -> dict[str, xr.DataArray]:
    """Load synaptic baseline fluorescence (F0) from scopex.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    dmd : int or None
        DMD number (1 or 2).  None loads both.
    syn_ids : sequence of int or None
        Subset of synapse IDs to load.  None loads all.
    channels : sequence of int, int, or None
        Subset of channels to load.  None loads all.
    apply_ephys_offset : bool
        If True (default), shift the ``time`` coordinate by the ephys
        offset so that times are aligned to the ephys clock.

    Returns
    -------
    dict[str, xr.DataArray]
        Mapping from ``"dmd_1"`` / ``"dmd_2"`` to DataArray with dims
        ``(channel, syn_id, time)``.
    """
    path = os.path.join(_scopex_dir(subject, exp, loc, acq), "syn_F0.zarr")
    result = load_xr_from_zarr(
        path, dmd=_resolve_dmd(dmd), sel=_build_sel(syn_ids, "syn_id", channels)
    )
    if apply_ephys_offset:
        result = _apply_ephys_offset(result, subject, exp, loc, acq)
    return result


def roi_F(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    trace: str = "Fsvd",
    dmd: int | None = None,
    soma_ids: Sequence[str] | None = None,
    channels: Sequence[int] | int | None = None,
    apply_ephys_offset: bool = True,
) -> dict[str, xr.DataArray]:
    """Load ROI (soma) fluorescence traces from scopex.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    trace : str
        Trace type: ``"Fsvd"`` or ``"Fraw"``.
    dmd : int or None
        DMD number (1 or 2).  None loads both.
    soma_ids : sequence of str or None
        Subset of soma IDs to load (e.g. ``["soma1", "soma2"]``).
        None loads all.
    channels : sequence of int, int, or None
        Subset of channels to load.  None loads all.
    apply_ephys_offset : bool
        If True (default), shift the ``time`` coordinate by the ephys
        offset so that times are aligned to the ephys clock.

    Returns
    -------
    dict[str, xr.DataArray]
        Mapping from ``"dmd_1"`` / ``"dmd_2"`` to DataArray with dims
        ``(channel, soma_id, time)``.
    """
    if trace == "Fraw":
        name = "ROI_Fraw"
    else:
        name = f"ROI_{trace}"
    path = os.path.join(_scopex_dir(subject, exp, loc, acq), f"{name}.zarr")
    result = load_xr_from_zarr(
        path, dmd=_resolve_dmd(dmd), sel=_build_sel(soma_ids, "soma_id", channels)
    )
    if apply_ephys_offset:
        result = _apply_ephys_offset(result, subject, exp, loc, acq)
    return result
