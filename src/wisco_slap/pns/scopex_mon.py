"""Monitor and orchestrate scopex (xarray/zarr) dataset generation.

This module is the sole decision-maker for what scopex outputs need to be
generated per acquisition. It checks existence AND version freshness: each
zarr has a companion ``.txt`` file encoding which ExperimentSummary version
produced it. If any output is missing or stale (version mismatch), ALL
outputs are regenerated because the data-loading step is expensive and
shared across all outputs.

The generating functions in scopex_gen.py should NOT contain any existence
checks or orchestration logic — they simply produce output when called.
"""

import os

import slap2_py as spy

import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.pns import scopex_gen

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _scopex_dir(subject: str, exp: str, loc: str, acq: str) -> str:
    """Return the scopex directory for a given acquisition."""
    return os.path.join(DEFS.anmat_root, subject, exp, "scopex", f"{loc}--{acq}")


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def _check_esum_mirror(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
) -> str | None:
    """Check whether a valid ExperimentSummary mirror exists.

    Returns the esum path if valid, None otherwise.
    """
    try:
        esum_path = wis.meta.get.esum_mirror_path(subject, exp, loc, acq)
    except (FileNotFoundError, ValueError):
        return None
    if esum_path == "NO_ESUM_MIRROR":
        return None
    return esum_path


def _get_esum_version(esum_path: str) -> str:
    """Extract the version string from an esum mirror path.

    For example, given a path ending in ``SummaryLoCo-260308-135713.mat``,
    returns ``"SummaryLoCo-260308-135713"``.
    """
    return os.path.basename(esum_path).split(".mat")[0]


def _check_zarr_exists(scopex_dir: str, name: str) -> bool:
    """Check whether a zarr directory exists."""
    return os.path.isdir(os.path.join(scopex_dir, f"{name}.zarr"))


def _check_version_match(
    scopex_dir: str,
    name: str,
    expected_version: str,
) -> bool:
    """Check whether the version txt file for a given output matches."""
    return os.path.isfile(os.path.join(scopex_dir, f"{name}__{expected_version}.txt"))


def _has_rois(esum_path: str) -> bool:
    """Check whether any DMD has user-drawn ROIs."""
    for dmd in [1, 2]:
        if len(spy.xsum.get_roi_list(esum_path, dmd)) > 0:
            return True
    return False


def _check_all_outputs(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_path: str,
) -> bool:
    """Check whether ALL expected scopex outputs exist with correct versions.

    Returns True only if every expected zarr is present AND every version
    txt file matches the current esum mirror version.
    """
    sd = _scopex_dir(subject, exp, loc, acq)
    version = _get_esum_version(esum_path)

    # Synaptic outputs (always required)
    required = ["syn_dF-denoised", "syn_dF-ls", "syn_dF-events", "syn_F0"]

    # ROI outputs (required only if ROIs exist)
    if _has_rois(esum_path):
        required.extend(["ROI_Fraw", "ROI_Fsvd"])

    for name in required:
        if not _check_zarr_exists(sd, name):
            return False
        if not _check_version_match(sd, name, version):
            return False

    return True


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def acq_data(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    overwrite: bool = False,
) -> None:
    """Check and generate all scopex outputs for a single acquisition.

    If any output is missing or has a stale version (ExperimentSummary has
    been updated), ALL outputs are regenerated. This is an all-or-nothing
    approach because the data-loading step is expensive and shared.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    overwrite : bool, optional
        If True, regenerate all outputs regardless of existence/version.
    """
    tag = f"{subject} {exp} {loc} {acq}"

    # Prerequisite: ExperimentSummary mirror must exist
    esum_path = _check_esum_mirror(subject, exp, loc, acq)
    if esum_path is None:
        print(f"[{tag}] Skipping — no ExperimentSummary mirror found")
        return

    # Fast path: everything present and version matches
    if not overwrite and _check_all_outputs(subject, exp, loc, acq, esum_path):
        return

    # Load trial data (expensive, done once)
    print(f"[{tag}] Loading trial data and generating scopex arrays...")
    trial_data, refdata, fs, ntrials = spy.xsum.read_full_trial_data_dict(esum_path)
    clean_trials = spy.xsum.get_clean_trial_dict(trial_data)
    trial_data = spy.xsum.replace_bad_trials_with_null_data(trial_data, clean_trials)
    spy.xsum.check_all_trial_shapes_match(trial_data, clean_trials)

    # Generate all synaptic trace outputs
    scopex_gen.save_dF_xarrays(
        subject,
        exp,
        loc,
        acq,
        esum_path,
        trial_data,
        fs,
        traces=("ls", "denoised", "events"),
    )
    scopex_gen.save_F0_xarray(
        subject,
        exp,
        loc,
        acq,
        esum_path,
        trial_data,
        fs,
    )

    # Generate ROI outputs (only if ROIs exist)
    if _has_rois(esum_path):
        scopex_gen.save_ROI_xarrays(
            subject,
            exp,
            loc,
            acq,
            esum_path,
            trial_data,
            fs,
            traces=("F", "Fsvd"),
        )
    else:
        print(f"[{tag}] No ROIs found — skipping ROI zarrs")


def exp_data(subject: str, exp: str, overwrite: bool = False) -> None:
    """Run acq_data for all acquisitions in an experiment.

    Parameters
    ----------
    subject : str
        Subject name.
    exp : str
        Experiment name.
    overwrite : bool, optional
        If True, regenerate all outputs regardless of existence/version.
    """
    locacqs = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
    for la in locacqs:
        loc, acq = la.split("--")
        acq_data(subject, exp, loc, acq, overwrite=overwrite)


def all_subjects(overwrite: bool = False) -> None:
    """Run acq_data for all subjects, experiments, and acquisitions.

    Parameters
    ----------
    overwrite : bool, optional
        If True, regenerate all outputs regardless of existence/version.
    """
    si = wis.meta.get.sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            try:
                locacqs = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
            except (KeyError, FileNotFoundError):
                print(f"[{subject} {exp}] No acquisitions found, skipping")
                continue
            for la in locacqs:
                loc, acq = la.split("--")
                try:
                    acq_data(subject, exp, loc, acq, overwrite=overwrite)
                except Exception as e:
                    print(f"[{subject} {exp} {loc} {acq}] Error: {e}")
                    continue
