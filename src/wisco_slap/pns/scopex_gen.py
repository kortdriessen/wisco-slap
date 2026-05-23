"""Pure generator functions for scopex (xarray/zarr) datasets.

These functions produce output unconditionally when called. All existence
checking, version tracking, and orchestration is handled by scopex_mon.py.
"""

import os
import shutil

import numpy as np
import slap2_py as spy

import wisco_slap as wis
from wisco_slap.defs import anmat_root


def write_ExSum_version_to_scopex(
    subject: str, exp: str, loc: str, acq: str, esum_p: str, name: str = "BLANK",
) -> None:
    """Write a version-tracking text file for a scopex zarr output.

    Creates a file named ``{name}__{esum_version}.txt`` in the scopex
    directory, where ``esum_version`` is the ExperimentSummary filename
    without its ``.mat`` extension (e.g. ``SummaryLoCo-260306-233829``).
    The file's *contents* are just that same version string on one line,
    matching the convention used by annotation material marker files at
    ``annotation_mat_mon._write_marker``. Any existing version file whose
    name contains ``name`` is removed first so that only one version file
    per output exists at a time.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    esum_p : str
        Full path to the ExperimentSummary .mat file used.
    name : str
        Output name prefix (e.g. ``"syn_dF-denoised"``).
    """
    act_dir = os.path.join(anmat_root, subject, exp, "scopex", f"{loc}--{acq}")
    wis.util.check_dir(act_dir)
    esum_version = os.path.basename(esum_p).split(".mat")[0]
    file_to_write = os.path.join(act_dir, f"{name}__{esum_version}.txt")
    for f in os.listdir(act_dir):
        if name in f and f.endswith(".txt"):
            os.remove(os.path.join(act_dir, f))
    with open(file_to_write, "w") as f:
        f.write(f"{esum_version}\n")


def save_dF_xarrays(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_p: str,
    trial_data: dict,
    fs: float,
    traces: tuple[str, ...] = ("ls", "denoised"),
    trial_epochs: np.ndarray | None = None,
    epoch_offsets_s: dict[int, float] | None = None,
) -> None:
    """Save synaptic dF (delta-F) trace xarrays as zarr stores.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    esum_p : str
        Path to the ExperimentSummary .mat file.
    trial_data : dict
        Trial data dictionary from ``spy.xsum.read_full_trial_data_dict``.
    fs : float
        Sampling rate in Hz.
    traces : tuple of str
        Which dF trace types to save.
    trial_epochs, epoch_offsets_s :
        For multi-epoch acqs, pass both so the time coord is built with
        real inter-epoch gaps. For single-epoch acqs leave as ``None``
        (identical to the pre-multi-epoch behavior).
    """
    acq_id = f"{loc}--{acq}"
    act_dir = os.path.join(anmat_root, subject, exp, "scopex", acq_id)
    wis.util.check_dir(act_dir)

    for trace_type in traces:
        xr_dict = spy.core.xarr_summ.dF_data_to_xr(
            trial_data, trace_type, fs,
            trial_epochs=trial_epochs, epoch_offsets_s=epoch_offsets_s,
        )
        name = f"syn_dF-{trace_type}"
        sx_path = os.path.join(act_dir, f"{name}.zarr")
        if os.path.exists(sx_path):
            shutil.rmtree(sx_path)
        spy.core.xarr_summ.save_xr_to_zarr(xr_dict, sx_path)
        write_ExSum_version_to_scopex(subject, exp, loc, acq, esum_p, name=name)


def save_F0_xarray(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_p: str,
    trial_data: dict,
    fs: float,
    trial_epochs: np.ndarray | None = None,
    epoch_offsets_s: dict[int, float] | None = None,
) -> None:
    """Save synaptic baseline fluorescence (F0) xarray as a zarr store.

    See :func:`save_dF_xarrays` for parameter semantics.
    """
    acq_id = f"{loc}--{acq}"
    act_dir = os.path.join(anmat_root, subject, exp, "scopex", acq_id)
    wis.util.check_dir(act_dir)
    xr_dict = spy.core.xarr_summ.F0_data_to_xr(
        trial_data, fs,
        trial_epochs=trial_epochs, epoch_offsets_s=epoch_offsets_s,
    )
    name = "syn_F0"
    sx_path = os.path.join(act_dir, f"{name}.zarr")
    if os.path.exists(sx_path):
        shutil.rmtree(sx_path)
    spy.core.xarr_summ.save_xr_to_zarr(xr_dict, sx_path)
    write_ExSum_version_to_scopex(subject, exp, loc, acq, esum_p, name=name)


def save_ROI_xarrays(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_p: str,
    trial_data: dict,
    fs: float,
    traces: tuple[str, ...] = ("F", "Fsvd"),
    trial_epochs: np.ndarray | None = None,
    epoch_offsets_s: dict[int, float] | None = None,
) -> None:
    """Save ROI (soma) fluorescence xarrays as zarr stores.

    Generates ``ROI_Fraw.zarr`` and ``ROI_Fsvd.zarr`` (by default).
    Requires that at least one DMD has user-drawn ROIs in the
    ExperimentSummary. See :func:`save_dF_xarrays` for the multi-epoch
    parameter semantics.
    """
    acq_id = f"{loc}--{acq}"
    act_dir = os.path.join(anmat_root, subject, exp, "scopex", acq_id)
    wis.util.check_dir(act_dir)

    roi_info = [spy.xsum.get_roi_list(esum_p, dmd) for dmd in [1, 2]]

    for trace_type in traces:
        xr_dict = spy.core.xarr_summ.ROI_data_to_xr(
            trial_data, trace_type, fs, roi_info,
            trial_epochs=trial_epochs, epoch_offsets_s=epoch_offsets_s,
        )
        if trace_type == "F":
            name = "ROI_Fraw"
        else:
            name = f"ROI_{trace_type}"
        sx_path = os.path.join(act_dir, f"{name}.zarr")
        if os.path.exists(sx_path):
            shutil.rmtree(sx_path)
        spy.core.xarr_summ.save_xr_to_zarr(xr_dict, sx_path)
        write_ExSum_version_to_scopex(subject, exp, loc, acq, esum_p, name=name)
