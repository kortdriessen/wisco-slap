"""Monitor and orchestrate basic glutamate event detection.

This module checks whether event detection outputs exist with correct
versions, and calls the generator (glu_ev_basic_gen) when they need to
be (re-)created.  It mirrors the structure of scopex_mon.py.
"""

from __future__ import annotations

import os

import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.pns import glu_ev_basic_gen

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _scopex_dir(subject: str, exp: str, loc: str, acq: str) -> str:
    """Return the scopex directory for a given acquisition."""
    return os.path.join(DEFS.anmat_root, subject, exp, "scopex", f"{loc}--{acq}")


def _ev_dir(subject: str, exp: str, loc: str, acq: str) -> str:
    """Return the event_detection output directory."""
    return os.path.join(
        _scopex_dir(subject, exp, loc, acq), glu_ev_basic_gen.OUTPUT_DIR_NAME
    )


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
    """Extract the version string from an esum mirror path."""
    return os.path.basename(esum_path).split(".mat")[0]


def _check_scopex_available(
    subject: str, exp: str, loc: str, acq: str
) -> bool:
    """Verify that the required scopex LS zarr exists."""
    sd = _scopex_dir(subject, exp, loc, acq)
    return os.path.isdir(os.path.join(sd, "syn_dF-ls.zarr"))


def _check_all_outputs(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_path: str,
) -> bool:
    """Check whether all event detection outputs exist with correct version.

    Checks for:
      - glu_events_basic.parquet
      - filtered.zarr (directory)
      - noise_std.zarr (directory)
      - glu_ev_basic__{version}.txt
    """
    out_dir = _ev_dir(subject, exp, loc, acq)
    if not os.path.isdir(out_dir):
        return False

    version = _get_esum_version(esum_path)

    if not os.path.isfile(
        os.path.join(out_dir, glu_ev_basic_gen.PARQUET_NAME)
    ):
        return False
    if not os.path.isdir(
        os.path.join(out_dir, glu_ev_basic_gen.FILTERED_ZARR_NAME)
    ):
        return False
    if not os.path.isdir(
        os.path.join(out_dir, glu_ev_basic_gen.NOISE_STD_ZARR_NAME)
    ):
        return False
    if not os.path.isfile(
        os.path.join(
            out_dir, f"{glu_ev_basic_gen.VERSION_PREFIX}__{version}.txt"
        )
    ):
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
    **detection_kwargs,
) -> None:
    """Check and generate basic glutamate event detection for one acquisition.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    overwrite : bool, optional
        If True, regenerate regardless of existing outputs.
    **detection_kwargs
        Forwarded to ``glu_ev_basic_gen.detect_and_save`` (e.g.
        ``tau_s``, ``noise_window_s``, ``snr_threshold``).
    """
    tag = f"{subject} {exp} {loc} {acq}"

    # Prerequisite: ExperimentSummary mirror
    esum_path = _check_esum_mirror(subject, exp, loc, acq)
    if esum_path is None:
        print(f"[{tag}] Skipping — no ExperimentSummary mirror found")
        return

    # Prerequisite: scopex LS zarr
    if not _check_scopex_available(subject, exp, loc, acq):
        print(f"[{tag}] Skipping — syn_dF-ls.zarr not available (run scopex_mon first)")
        return

    # Fast path: everything present and version matches
    if not overwrite and _check_all_outputs(subject, exp, loc, acq, esum_path):
        return

    # Generate
    print(f"[{tag}] Detecting basic glutamate events...")
    glu_ev_basic_gen.detect_and_save(
        subject, exp, loc, acq, esum_p=esum_path, **detection_kwargs
    )


def exp_data(
    subject: str,
    exp: str,
    overwrite: bool = False,
    **detection_kwargs,
) -> None:
    """Run acq_data for all acquisitions in an experiment.

    Parameters
    ----------
    subject : str
        Subject name.
    exp : str
        Experiment name.
    overwrite : bool, optional
        If True, regenerate all outputs regardless of existence/version.
    **detection_kwargs
        Forwarded to ``acq_data``.
    """
    locacqs = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
    for la in locacqs:
        loc, acq = la.split("--")
        acq_data(subject, exp, loc, acq, overwrite=overwrite, **detection_kwargs)


def all_subjects(
    overwrite: bool = False,
    **detection_kwargs,
) -> None:
    """Run acq_data for all subjects, experiments, and acquisitions.

    Parameters
    ----------
    overwrite : bool, optional
        If True, regenerate all outputs regardless of existence/version.
    **detection_kwargs
        Forwarded to ``acq_data``.
    """
    si = wis.meta.get.sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            try:
                locacqs = wis.meta.get.unique_acquisitions_per_experiment(
                    subject, exp
                )
            except (KeyError, FileNotFoundError):
                print(f"[{subject} {exp}] No acquisitions found, skipping")
                continue
            for la in locacqs:
                loc, acq = la.split("--")
                try:
                    acq_data(
                        subject, exp, loc, acq,
                        overwrite=overwrite, **detection_kwargs,
                    )
                except Exception as e:
                    print(f"[{subject} {exp} {loc} {acq}] Error: {e}")
                    continue
