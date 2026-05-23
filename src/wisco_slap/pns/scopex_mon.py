"""Monitor and orchestrate scopex (xarray/zarr) dataset generation.

This module is the sole decision-maker for what scopex outputs need to be
generated per acquisition. It checks existence AND version freshness: each
zarr has a companion ``.txt`` file encoding which ExperimentSummary version
produced it. If any output is missing or stale (version mismatch), ALL
outputs are regenerated because the data-loading step is expensive and
shared across all outputs.

The generating functions in scopex_gen.py should NOT contain any existence
checks or orchestration logic — they simply produce output when called.

For read-only status queries that do NOT regenerate anything (e.g. for the
project-wide ExSum audit consumed by WISynaptic), use
``check_outputs_status`` rather than ``acq_data``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import slap2_py as spy

import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.pns import scopex_gen


@dataclass(slots=True)
class ScopexCheckResult:
    """Structured result of a scopex freshness check.

    Attributes
    ----------
    status : str
        One of ``"fresh"``, ``"stale"``, ``"missing"``, ``"missing_mirror"``.
        ``"stale"`` means at least one output exists but with a version
        mismatch; ``"missing"`` means at least one output is absent (with
        no stale outputs). If both conditions are present, ``"stale"``
        takes precedence.
    esum_version : str | None
        The current mirror version string, or None if no mirror exists.
    missing_zarrs : tuple[str, ...]
        Names of expected zarrs that are not present on disk.
    stale_zarrs : tuple[str, ...]
        Names of zarrs whose version flag does not match the current
        mirror.
    expected_zarrs : tuple[str, ...]
        Full list of zarrs that were expected for this acquisition (varies
        based on whether user ROIs exist).
    """

    status: str
    esum_version: str | None
    missing_zarrs: tuple[str, ...] = ()
    stale_zarrs: tuple[str, ...] = ()
    expected_zarrs: tuple[str, ...] = ()

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

    Kept as a thin bool-returning wrapper around
    :func:`check_outputs_status` for the ``acq_data`` fast-path.
    """
    return check_outputs_status(
        subject, exp, loc, acq, esum_path=esum_path
    ).status == "fresh"


def check_outputs_status(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_path: str | None = None,
) -> ScopexCheckResult:
    """Return the read-only freshness status of scopex outputs for an acq.

    Does not regenerate anything. If ``esum_path`` is omitted, the mirror
    is looked up; if no mirror is present, ``status="missing_mirror"``.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    esum_path : str, optional
        Path to the current ExperimentSummary mirror. Looked up if omitted.

    Returns
    -------
    ScopexCheckResult
        Structured status. See class docstring for field semantics.
    """
    if esum_path is None:
        esum_path = _check_esum_mirror(subject, exp, loc, acq)
    if esum_path is None:
        return ScopexCheckResult(status="missing_mirror", esum_version=None)

    sd = _scopex_dir(subject, exp, loc, acq)
    version = _get_esum_version(esum_path)

    # Synaptic outputs (always required)
    required = ["syn_dF-denoised", "syn_dF-ls", "syn_dF-events", "syn_F0"]

    # ROI outputs (required only if ROIs exist)
    try:
        if _has_rois(esum_path):
            required.extend(["ROI_Fraw", "ROI_Fsvd"])
    except Exception:
        # If we can't read the ExSum, we can't know which zarrs to expect;
        # treat as missing_mirror-equivalent (unsupported mirror).
        return ScopexCheckResult(
            status="missing_mirror",
            esum_version=version,
            expected_zarrs=tuple(required),
        )

    missing: list[str] = []
    stale: list[str] = []
    for name in required:
        if not _check_zarr_exists(sd, name):
            missing.append(name)
            continue
        if not _check_version_match(sd, name, version):
            stale.append(name)

    if stale:
        status = "stale"
    elif missing:
        status = "missing"
    else:
        status = "fresh"

    return ScopexCheckResult(
        status=status,
        esum_version=version,
        missing_zarrs=tuple(missing),
        stale_zarrs=tuple(stale),
        expected_zarrs=tuple(required),
    )


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
    trial_data, refdata, fs, ntrials, trial_epochs = (
        spy.xsum.read_full_trial_data_dict(esum_path)
    )
    multi_epoch = int(trial_epochs.max()) > 1
    # For multi-epoch acqs, per-trial shapes can differ BETWEEN epochs but
    # not within; feed trial_epochs to the shape helpers so the NaN-fill
    # template for each bad trial is drawn from a clean trial in the same
    # epoch.
    te_kw = trial_epochs if multi_epoch else None
    clean_trials = spy.xsum.get_clean_trial_dict(trial_data, trial_epochs=te_kw)
    trial_data = spy.xsum.replace_bad_trials_with_null_data(
        trial_data, clean_trials, trial_epochs=te_kw
    )
    spy.xsum.check_all_trial_shapes_match(
        trial_data, clean_trials, trial_epochs=te_kw
    )

    # Resolve per-epoch time offsets for multi-epoch acqs from epoch_info.yaml.
    # Single-epoch acqs pass None → identical behavior to pre-multi-epoch code.
    epoch_offsets_s: dict[int, float] | None = None
    if multi_epoch:
        acq_id = f"{loc}--{acq}"
        ei = wis.meta.get.epoch_info()
        try:
            acq_ei = ei[subject][exp][acq_id]
        except KeyError as e:
            raise KeyError(
                f"[{tag}] Multi-epoch acq but no entry in epoch_info.yaml. "
                f"Run wis.meta.update() first to populate per-epoch offsets."
            ) from e
        epoch_offsets_s = {
            int(k): float(v["duration_since_epoch_1_s"]) for k, v in acq_ei.items()
        }

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
        trial_epochs=te_kw,
        epoch_offsets_s=epoch_offsets_s,
    )
    scopex_gen.save_F0_xarray(
        subject,
        exp,
        loc,
        acq,
        esum_path,
        trial_data,
        fs,
        trial_epochs=te_kw,
        epoch_offsets_s=epoch_offsets_s,
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
            trial_epochs=te_kw,
            epoch_offsets_s=epoch_offsets_s,
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
