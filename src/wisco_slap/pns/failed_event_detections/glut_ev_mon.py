"""Monitor and orchestrate glutamate event detection.

Checks whether event detection outputs exist and are up to date with
the current ExperimentSummary version, and triggers regeneration via
glut_ev_gen when needed. Requires scopex zarrs (ls + F0) to have been
generated first (via scopex_mon).

Follows the same mon/gen pattern as scopex_mon.py.
"""

from __future__ import annotations

import os
from typing import Any

import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.pns import glut_ev_gen
from wisco_slap.pns.glut_ev_defs import OUTPUT_DIR_NAME

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _glut_ev_dir(subject: str, exp: str, loc: str, acq: str) -> str:
    """Return the output directory for glutamate event detection."""
    return os.path.join(
        DEFS.anmat_root, subject, exp, "activity_data", loc, acq, OUTPUT_DIR_NAME
    )


def _scopex_dir(subject: str, exp: str, loc: str, acq: str) -> str:
    return os.path.join(DEFS.anmat_root, subject, exp, "scopex", f"{loc}--{acq}")


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def _check_esum_mirror(
    subject: str, exp: str, loc: str, acq: str
) -> str | None:
    """Return esum path if a valid mirror exists, else None."""
    try:
        esum_path = wis.meta.get.esum_mirror_path(subject, exp, loc, acq)
    except (FileNotFoundError, ValueError):
        return None
    if esum_path == "NO_ESUM_MIRROR":
        return None
    return esum_path


def _get_esum_version(esum_path: str) -> str:
    return os.path.basename(esum_path).split(".mat")[0]


def _check_scopex_available(subject: str, exp: str, loc: str, acq: str) -> bool:
    """Verify that the required scopex zarrs exist (ls + F0)."""
    sx_dir = _scopex_dir(subject, exp, loc, acq)
    for name in ("syn_dF-ls", "syn_F0"):
        if not os.path.isdir(os.path.join(sx_dir, f"{name}.zarr")):
            return False
    return True


def _check_all_outputs(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_path: str,
) -> bool:
    """Check whether all event detection outputs exist with correct version."""
    out_dir = _glut_ev_dir(subject, exp, loc, acq)
    version = _get_esum_version(esum_path)

    # Required parquet files
    for dmd in [1, 2]:
        if not os.path.isfile(os.path.join(out_dir, f"events_dmd{dmd}.parquet")):
            return False
        if not os.path.isfile(os.path.join(out_dir, f"synapse_summary_dmd{dmd}.parquet")):
            return False

    # Version tracking file
    version_file = os.path.join(out_dir, f"{OUTPUT_DIR_NAME}__{version}.txt")
    if not os.path.isfile(version_file):
        return False

    return True


def _write_version_file(
    subject: str, exp: str, loc: str, acq: str, esum_path: str
) -> None:
    """Write a version-tracking text file for the event detection output."""
    out_dir = _glut_ev_dir(subject, exp, loc, acq)
    wis.util.check_dir(out_dir)
    version = _get_esum_version(esum_path)

    # Remove old version files
    for f in os.listdir(out_dir):
        if f.startswith(OUTPUT_DIR_NAME + "__") and f.endswith(".txt"):
            os.remove(os.path.join(out_dir, f))

    file_to_write = os.path.join(out_dir, f"{OUTPUT_DIR_NAME}__{version}.txt")
    with open(file_to_write, "w") as f:
        f.write(esum_path)


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def acq_data(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    overwrite: bool = False,
    **detection_kwargs: Any,
) -> None:
    """Check and generate glutamate event detection for a single acquisition.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    overwrite : bool
        If True, regenerate regardless of existing outputs.
    **detection_kwargs
        Additional keyword arguments passed to
        ``glut_ev_gen.detect_and_save``.
    """
    tag = f"{subject} {exp} {loc} {acq}"

    # Prerequisite: esum mirror
    esum_path = _check_esum_mirror(subject, exp, loc, acq)
    if esum_path is None:
        print(f"[{tag}] Skipping — no ExperimentSummary mirror")
        return

    # Prerequisite: scopex zarrs (ls + F0)
    if not _check_scopex_available(subject, exp, loc, acq):
        print(f"[{tag}] Skipping — scopex zarrs not available (run scopex_mon first)")
        return

    # Fast path: already up to date
    if not overwrite and _check_all_outputs(subject, exp, loc, acq, esum_path):
        return

    # Generate
    print(f"[{tag}] Detecting glutamate events...")
    glut_ev_gen.detect_and_save(
        subject, exp, loc, acq, **detection_kwargs
    )
    _write_version_file(subject, exp, loc, acq, esum_path)


def exp_data(
    subject: str,
    exp: str,
    overwrite: bool = False,
    **detection_kwargs: Any,
) -> None:
    """Run acq_data for all acquisitions in an experiment.

    Parameters
    ----------
    subject : str
    exp : str
    overwrite : bool
    **detection_kwargs
        Passed through to ``acq_data``.
    """
    locacqs = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
    for la in locacqs:
        loc, acq = la.split("--")
        acq_data(
            subject, exp, loc, acq,
            overwrite=overwrite, **detection_kwargs
        )


def all_subjects(
    overwrite: bool = False,
    **detection_kwargs: Any,
) -> None:
    """Run acq_data for all subjects, experiments, and acquisitions.

    Parameters
    ----------
    overwrite : bool
    **detection_kwargs
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
                    acq_data(
                        subject, exp, loc, acq,
                        overwrite=overwrite, **detection_kwargs
                    )
                except Exception as e:
                    print(f"[{subject} {exp} {loc} {acq}] Error: {e}")
                    continue
