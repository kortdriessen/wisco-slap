"""Loading functions for scopex (xarray/zarr) datasets.

All scopex zarrs are chunked as ``(1, 1, n_time)`` — one channel, one
synapse/soma per chunk.  This means selecting a subset of syn_ids or
channels reads only the necessary chunks from disk, so passing ``syn_ids``
or ``channels`` is very efficient.
"""

import os
import warnings
from collections.abc import Sequence

import numpy as np
import polars as pl
import xarray as xr
from slap2_py.core.xarr_summ import load_xr_from_zarr

from wisco_slap.defs import anmat_root
from wisco_slap.meta.get import ephys_offset as _get_ephys_offset
from wisco_slap.meta.get import sync_info as _get_sync_info


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
    """Shift the ``time`` coordinate by the ephys offset.

    Single-epoch acqs and multi-epoch single-sync-block acqs: the shifted
    axis is "seconds from the TDT block's ephys_start" — the usual ephys
    timebase. Straightforward.

    Multi-epoch cross-sync-block acqs: the offset we apply is epoch 1's
    (which is what ``sync_info.yaml`` stores by convention). The resulting
    axis is "seconds from sync_block-1's ephys_start" — a wall-clock-like
    axis that is NOT directly indexable into sync_block-2's ephys data
    for epochs that fall in block 2. Consumers who need per-epoch ephys
    alignment should compute it via ``epoch_info.yaml``. A ``UserWarning``
    is emitted so this isn't silent.
    """
    offset = _get_ephys_offset(subject, exp, loc, acq)
    try:
        acq_entry = _get_sync_info()[subject][exp]["acquisitions"][f"{loc}--{acq}"]
        n_epochs = acq_entry.get("n_epochs", 1) or 1
    except Exception:
        n_epochs = 1
    if n_epochs > 1:
        warnings.warn(
            f"{subject}/{exp}/{loc}/{acq} is multi-epoch (n_epochs={n_epochs}): "
            f"apply_ephys_offset=True shifts the scopex time axis by epoch 1's "
            f"ephys_offset. The resulting axis is 'seconds from sync_block-1's "
            f"ephys_start' — valid for epoch 1, but later epochs' data lie at "
            f"times that may fall in sync_block-1's gap or in sync_block-2. "
            f"For per-epoch ephys alignment, read epoch_info.yaml (via "
            f"wis.meta.get.epoch_info()) and apply each epoch's own "
            f"sync_block + ephys_offset_s to the corresponding slice.",
            UserWarning,
            stacklevel=2,
        )
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
    """Load synaptic dF (delta-F) traces from scopex.

    Note: these are *not* dF/F — they are raw delta-F as output by the
    preprocessing pipeline.  To obtain dF/F, divide by the time-varying
    baseline returned by :func:`~wisco_slap.get.syn_F0`.

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
    merge_info: bool = True,
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
    merge_info : bool
        If True, load synapse labels via :func:`~wisco_slap.get.synid_labels` and attach them as
        non-dimension coordinates on the ``syn_id`` dimension.

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

    if merge_info:
        from wisco_slap.get._get_syn_info import synid_labels

        idf = synid_labels(subject, exp, loc, acq)
        if idf is not None:
            result = merge_syn_info_to_scopex(result, idf)

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
    return_dFF: bool = False,
) -> dict[str, xr.DataArray]:
    """Load ROI (soma) fluorescence traces from scopex.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    trace : str
        Trace type: ``"Fsvd"`` or ``"Fraw"``. ``"Fsvd"`` is the preferred
        default for ROI dF/F because it is the lower-noise, low-rank ROI
        estimate from processSLAP2. ``"Fraw"`` remains useful for raw/QC
        inspection.
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
    return_dFF : bool
        If True, convert the requested ROI trace to dF/F before returning it.
        ROI dF/F is computed lazily from the loaded trace using a trace-level
        analogue of MATLAB ``computeF0(..., algo1)`` because source-level NMF
        ``F0`` is only available for synaptic traces, not for saved user-ROI
        traces.

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
    if return_dFF:
        from wisco_slap.scope.pro import roi_to_dff

        result = {key: roi_to_dff(da, trace=trace) for key, da in result.items()}
    return result
