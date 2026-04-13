"""Loading functions for event detection outputs.

All event detection outputs live under
``{scopex_dir}/event_detection/`` for each acquisition.
"""

import os
from collections.abc import Sequence

import polars as pl
import xarray as xr
from slap2_py.core.xarr_summ import load_xr_from_zarr

from wisco_slap.defs import anmat_root


def _ev_dir(subject: str, exp: str, loc: str, acq: str) -> str:
    """Return the event_detection directory for a given acquisition."""
    return os.path.join(
        anmat_root, subject, exp, "scopex", f"{loc}--{acq}", "event_detection"
    )


def _build_sel(
    syn_ids: Sequence[int] | None,
    channels: Sequence[int] | int | None,
) -> dict | None:
    """Build a sel dict from optional id/channel subsets."""
    sel = {}
    if syn_ids is not None:
        sel["syn_id"] = list(syn_ids)
    if channels is not None:
        sel["channel"] = channels
    return sel or None


def _resolve_dmd(dmd: int | None) -> str | None:
    """Convert user-facing dmd int (1 or 2) to zarr group key, or None for both."""
    if dmd is None:
        return None
    return f"dmd_{dmd}"


def _load_glu_events(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    parquet_name: str,
    apply_ephys_offset: bool,
    merge_info: bool,
) -> pl.DataFrame:
    """Shared body for matchfilt / denoised event-DataFrame loaders."""
    path = os.path.join(_ev_dir(subject, exp, loc, acq), parquet_name)
    df = pl.read_parquet(path)
    if merge_info:
        from wisco_slap.get._get_syn_info import synid_labels

        idf = synid_labels(subject, exp, loc, acq)
        if idf is not None:
            df = df.join(idf, on=["dmd", "syn_id"], how="left")
    if apply_ephys_offset:
        from wisco_slap.meta.get import ephys_offset

        offset = ephys_offset(subject, exp, loc, acq)
        df = df.with_columns(
            (pl.col("time") + offset).alias("time"),
            (pl.col("peak_time") + offset).alias("peak_time"),
        )
    return df


def glu_events_basic(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    apply_ephys_offset: bool = True,
    merge_info: bool = True,
) -> pl.DataFrame:
    """Load the basic glutamate events DataFrame (matchfilt mode).

    Loads ``glu_events_basic.parquet`` — events detected from the
    match-filtered LS trace. For the denoised-trace variant, use
    :func:`glu_events_basic_denoised`.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    apply_ephys_offset : bool
        If True (default), shift the ``time`` and ``peak_time`` columns by
        the ephys offset so that times are in the shared ephys clock.
    merge_info : bool
        If True (default), left-join synapse identity/annotation labels
        (dendrite ID, soma ID, depth, etc.) onto the events via
        ``dmd`` and ``syn_id``.

    Returns
    -------
    pl.DataFrame
        Events with columns: dmd, syn_id, start_sample, end_sample,
        n_samples, time, peak_time, event_duration, peak_snr,
        average_snr, peak_filtered_value, integral — plus synapse
        annotation columns when *merge_info* is True.
    """
    return _load_glu_events(
        subject, exp, loc, acq,
        parquet_name="glu_events_basic.parquet",
        apply_ephys_offset=apply_ephys_offset,
        merge_info=merge_info,
    )


def glu_events_basic_denoised(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    apply_ephys_offset: bool = True,
    merge_info: bool = True,
) -> pl.DataFrame:
    """Load the basic glutamate events DataFrame (denoised mode).

    Loads ``glu_events_basic_denoised.parquet`` — events detected by
    running the basic pipeline on the preprocessed denoised trace
    (``syn_dF-denoised.zarr``) instead of the match-filtered LS trace.
    The schema is identical to :func:`glu_events_basic`; the column
    ``peak_filtered_value`` holds the peak of the denoised trace within
    each event.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    apply_ephys_offset : bool
        If True (default), shift the ``time`` and ``peak_time`` columns by
        the ephys offset so that times are in the shared ephys clock.
    merge_info : bool
        If True (default), left-join synapse identity/annotation labels
        (dendrite ID, soma ID, depth, etc.) onto the events via
        ``dmd`` and ``syn_id``.

    Returns
    -------
    pl.DataFrame
        Same schema as :func:`glu_events_basic`.
    """
    return _load_glu_events(
        subject, exp, loc, acq,
        parquet_name="glu_events_basic_denoised.parquet",
        apply_ephys_offset=apply_ephys_offset,
        merge_info=merge_info,
    )


def _load_ev_zarr(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    zarr_name: str,
    dmd: int | None,
    syn_ids: Sequence[int] | None,
    channels: Sequence[int] | int | None,
    merge_info: bool,
    apply_ephys_offset: bool,
) -> dict[str, xr.DataArray]:
    """Shared body for event-detection zarr loaders.

    Applies the same ergonomics as :func:`~wisco_slap.get.syn_dF`
    (ephys-offset alignment and optional synapse-info merging).
    """
    path = os.path.join(_ev_dir(subject, exp, loc, acq), zarr_name)
    result = load_xr_from_zarr(
        path, dmd=_resolve_dmd(dmd), sel=_build_sel(syn_ids, channels)
    )
    if merge_info:
        from wisco_slap.get._get_scopex import merge_syn_info_to_scopex
        from wisco_slap.get._get_syn_info import synid_labels

        idf = synid_labels(subject, exp, loc, acq)
        if idf is not None:
            result = merge_syn_info_to_scopex(result, idf)
    if apply_ephys_offset:
        from wisco_slap.get._get_scopex import _apply_ephys_offset

        result = _apply_ephys_offset(result, subject, exp, loc, acq)
    return result


def matchFilt_traces(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    dmd: int | None = None,
    syn_ids: Sequence[int] | None = None,
    channels: Sequence[int] | int | None = None,
    merge_info: bool = True,
    apply_ephys_offset: bool = True,
) -> dict[str, xr.DataArray]:
    """Load matched-filter traces used in event detection (matchfilt mode).

    Convenience wrapper around the ``filtered.zarr`` output of the
    matchfilt event-detection pipeline.

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
    return _load_ev_zarr(
        subject, exp, loc, acq,
        zarr_name="filtered.zarr",
        dmd=dmd, syn_ids=syn_ids, channels=channels,
        merge_info=merge_info,
        apply_ephys_offset=apply_ephys_offset,
    )


def matchFilt_noise_std(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    dmd: int | None = None,
    syn_ids: Sequence[int] | None = None,
    channels: Sequence[int] | int | None = None,
    merge_info: bool = True,
    apply_ephys_offset: bool = True,
) -> dict[str, xr.DataArray]:
    """Load rolling MAD noise estimate from event detection (matchfilt mode).

    Convenience wrapper around the ``noise_std.zarr`` output of the
    matchfilt event-detection pipeline. For the denoised-mode equivalent,
    use :func:`denoised_noise_std`.

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
    return _load_ev_zarr(
        subject, exp, loc, acq,
        zarr_name="noise_std.zarr",
        dmd=dmd, syn_ids=syn_ids, channels=channels,
        merge_info=merge_info,
        apply_ephys_offset=apply_ephys_offset,
    )


def denoised_noise_std(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    dmd: int | None = None,
    syn_ids: Sequence[int] | None = None,
    channels: Sequence[int] | int | None = None,
    merge_info: bool = True,
    apply_ephys_offset: bool = True,
) -> dict[str, xr.DataArray]:
    """Load rolling MAD noise estimate from event detection (denoised mode).

    Convenience wrapper around the ``noise_std_denoised.zarr`` output
    written by :func:`~wisco_slap.pns.glu_ev_basic_gen.detect_and_save`
    when ``use_denoised=True``. The noise estimate was computed on the
    preprocessed denoised trace (``syn_dF-denoised.zarr``) rather than
    on a matched-filter output.

    For the denoised trace itself, use
    ``wis.get.syn_dF(trace="denoised")``.

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
    return _load_ev_zarr(
        subject, exp, loc, acq,
        zarr_name="noise_std_denoised.zarr",
        dmd=dmd, syn_ids=syn_ids, channels=channels,
        merge_info=merge_info,
        apply_ephys_offset=apply_ephys_offset,
    )


