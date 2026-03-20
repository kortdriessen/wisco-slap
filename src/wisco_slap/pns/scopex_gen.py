"""Pure generator functions for scopex (xarray/zarr) datasets.

These functions produce output unconditionally when called. All existence
checking, version tracking, and orchestration is handled by scopex_mon.py.
"""

import os
import shutil

import slap2_py as spy

import wisco_slap as wis
from wisco_slap.defs import anmat_root


def write_ExSum_version_to_scopex(
    subject: str, exp: str, loc: str, acq: str, esum_p: str, name: str = "BLANK",
) -> None:
    """Write a version-tracking text file for a scopex zarr output.

    Creates a file named ``{name}__{esum_basename}.txt`` in the scopex
    directory.  Any existing version file for the same ``name`` is removed
    first so that only one version file per output exists at a time.

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
    esum_p_name = os.path.basename(esum_p).split(".mat")[0]
    file_to_write = os.path.join(act_dir, f"{name}__{esum_p_name}.txt")
    for f in os.listdir(act_dir):
        if name in f and f.endswith(".txt"):
            os.remove(os.path.join(act_dir, f))
    with open(file_to_write, "w") as f:
        f.write(esum_p)


def save_dF_xarrays(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_p: str,
    trial_data: dict,
    fs: float,
    traces: tuple[str, ...] = ("ls", "denoised"),
) -> None:
    """Save synaptic dF/F trace xarrays as zarr stores.

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
    """
    acq_id = f"{loc}--{acq}"
    act_dir = os.path.join(anmat_root, subject, exp, "scopex", acq_id)
    wis.util.check_dir(act_dir)

    for trace_type in traces:
        xr_dict = spy.core.xarr_summ.dF_data_to_xr(trial_data, trace_type, fs)
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
) -> None:
    """Save synaptic baseline fluorescence (F0) xarray as a zarr store.

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
    """
    acq_id = f"{loc}--{acq}"
    act_dir = os.path.join(anmat_root, subject, exp, "scopex", acq_id)
    wis.util.check_dir(act_dir)
    xr_dict = spy.core.xarr_summ.F0_data_to_xr(trial_data, fs)
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
) -> None:
    """Save ROI (soma) fluorescence xarrays as zarr stores.

    Generates ``ROI_Fraw.zarr`` and ``ROI_Fsvd.zarr`` (by default).
    Requires that at least one DMD has user-drawn ROIs in the
    ExperimentSummary.

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
        Which ROI trace types to save.
    """
    acq_id = f"{loc}--{acq}"
    act_dir = os.path.join(anmat_root, subject, exp, "scopex", acq_id)
    wis.util.check_dir(act_dir)

    roi_info = [spy.xsum.get_roi_list(esum_p, dmd) for dmd in [1, 2]]

    for trace_type in traces:
        xr_dict = spy.core.xarr_summ.ROI_data_to_xr(
            trial_data, trace_type, fs, roi_info
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
