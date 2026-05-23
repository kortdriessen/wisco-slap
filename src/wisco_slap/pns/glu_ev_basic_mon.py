"""Monitor and orchestrate basic glutamate event detection.

This module checks whether event detection outputs exist with correct
versions, and calls the generator (glu_ev_basic_gen) when they need to
be (re-)created.  It mirrors the structure of scopex_mon.py.

For read-only status queries that do NOT regenerate anything (e.g. for
the project-wide ExSum audit consumed by WISynaptic), use
:func:`check_outputs_status` rather than :func:`acq_data`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.pns import glu_ev_basic_gen


@dataclass(slots=True)
class EventDetectionCheckResult:
    """Structured result of an event-detection freshness check for one mode.

    Attributes
    ----------
    status : str
        One of ``"fresh"``, ``"stale"``, ``"missing"``, ``"missing_mirror"``,
        ``"scopex_unavailable"``. The last value indicates the upstream
        scopex zarr for this mode (denoised or ls) is absent, so event
        detection cannot be assessed for freshness (the outputs may or
        may not exist but we have no basis to call them fresh).
    mode : str
        ``"matchfilt"`` or ``"denoised"``.
    esum_version : str | None
        The current mirror version string, or None if no mirror exists.
    missing_files : tuple[str, ...]
        Relative filenames of expected outputs that are absent.
    stale_files : tuple[str, ...]
        Relative filenames whose version flag does not match the current
        mirror.
    """

    status: str
    mode: str
    esum_version: str | None
    missing_files: tuple[str, ...] = ()
    stale_files: tuple[str, ...] = ()

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
    subject: str, exp: str, loc: str, acq: str, use_denoised: bool = False
) -> bool:
    """Verify that the required scopex trace zarr exists.

    Checks for ``syn_dF-denoised.zarr`` in denoised mode, otherwise
    ``syn_dF-ls.zarr``.
    """
    sd = _scopex_dir(subject, exp, loc, acq)
    zarr_name = "syn_dF-denoised.zarr" if use_denoised else "syn_dF-ls.zarr"
    return os.path.isdir(os.path.join(sd, zarr_name))


def _check_all_outputs(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_path: str,
    use_denoised: bool = False,
) -> bool:
    """Check whether all event detection outputs exist with correct version.

    In **matchfilt mode** (``use_denoised=False``), checks for:
      - glu_events_basic.parquet
      - filtered.zarr (directory)
      - noise_std.zarr (directory)
      - glu_ev_basic__{version}.txt

    In **denoised mode** (``use_denoised=True``), checks for:
      - glu_events_basic_denoised.parquet
      - noise_std_denoised.zarr (directory)
      - glu_ev_basic_denoised__{version}.txt

    Kept as a thin bool-returning wrapper around
    :func:`check_outputs_status` for the ``acq_data`` fast-path.
    """
    return check_outputs_status(
        subject,
        exp,
        loc,
        acq,
        esum_path=esum_path,
        mode="denoised" if use_denoised else "matchfilt",
    ).status == "fresh"


def check_outputs_status(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_path: str | None = None,
    mode: str = "matchfilt",
) -> EventDetectionCheckResult:
    """Return the read-only freshness status of event-detection outputs.

    Does not regenerate anything. If ``esum_path`` is omitted, the mirror
    is looked up; if no mirror is present, ``status="missing_mirror"``.
    If the upstream scopex zarr for this mode is absent, status is
    ``"scopex_unavailable"``.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    esum_path : str, optional
        Path to the current ExperimentSummary mirror. Looked up if omitted.
    mode : {"matchfilt", "denoised"}
        Which pipeline variant to assess. Both modes have disjoint outputs
        and coexist in the same ``event_detection/`` directory.

    Returns
    -------
    EventDetectionCheckResult
        Structured status. See class docstring for field semantics.
    """
    if mode not in ("matchfilt", "denoised"):
        raise ValueError(f"mode must be 'matchfilt' or 'denoised'; got {mode!r}")
    use_denoised = mode == "denoised"

    if esum_path is None:
        esum_path = _check_esum_mirror(subject, exp, loc, acq)
    if esum_path is None:
        return EventDetectionCheckResult(
            status="missing_mirror", mode=mode, esum_version=None
        )

    version = _get_esum_version(esum_path)

    # Upstream scopex must be available to assess freshness meaningfully.
    if not _check_scopex_available(
        subject, exp, loc, acq, use_denoised=use_denoised
    ):
        return EventDetectionCheckResult(
            status="scopex_unavailable",
            mode=mode,
            esum_version=version,
        )

    out_dir = _ev_dir(subject, exp, loc, acq)

    if use_denoised:
        expected_parquet = glu_ev_basic_gen.PARQUET_DENOISED_NAME
        expected_zarrs = (glu_ev_basic_gen.NOISE_STD_DENOISED_ZARR_NAME,)
        version_prefix = glu_ev_basic_gen.VERSION_PREFIX_DENOISED
    else:
        expected_parquet = glu_ev_basic_gen.PARQUET_NAME
        expected_zarrs = (
            glu_ev_basic_gen.FILTERED_ZARR_NAME,
            glu_ev_basic_gen.NOISE_STD_ZARR_NAME,
        )
        version_prefix = glu_ev_basic_gen.VERSION_PREFIX

    version_file = f"{version_prefix}__{version}.txt"
    legacy_prefix = f"{version_prefix}__"

    def _mode_has_any_artifacts() -> bool:
        if not os.path.isdir(out_dir):
            return False
        if os.path.isfile(os.path.join(out_dir, expected_parquet)):
            return True
        for zname in expected_zarrs:
            if os.path.isdir(os.path.join(out_dir, zname)):
                return True
        for f in os.listdir(out_dir):
            if f.startswith(legacy_prefix) and f.endswith(".txt"):
                return True
        return False

    # "Never run" case: event-detection output dir absent OR no mode-specific
    # artifacts at all. Treat as not_applicable so consumers can distinguish
    # "pipeline never ran" from "pipeline ran but is inconsistent/stale".
    if not _mode_has_any_artifacts():
        return EventDetectionCheckResult(
            status="not_applicable",
            mode=mode,
            esum_version=version,
        )

    missing: list[str] = []
    stale: list[str] = []

    if not os.path.isfile(os.path.join(out_dir, expected_parquet)):
        missing.append(expected_parquet)

    for zname in expected_zarrs:
        if not os.path.isdir(os.path.join(out_dir, zname)):
            missing.append(zname)

    # Version file: absent means either missing (if nothing else exists) or
    # stale (if outputs are present under an older version name). We
    # distinguish by looking for *any* matching-prefix txt file.
    version_match = os.path.isfile(os.path.join(out_dir, version_file))
    if not version_match:
        has_legacy_version = any(
            f.startswith(legacy_prefix) and f.endswith(".txt")
            for f in os.listdir(out_dir)
        )
        if has_legacy_version:
            stale.append(version_file)
        else:
            missing.append(version_file)

    if stale:
        status = "stale"
    elif missing:
        status = "missing"
    else:
        status = "fresh"

    return EventDetectionCheckResult(
        status=status,
        mode=mode,
        esum_version=version,
        missing_files=tuple(missing),
        stale_files=tuple(stale),
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
    use_denoised: bool = False,
    **detection_kwargs,
) -> None:
    """Check and generate basic glutamate event detection for one acquisition.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    overwrite : bool, optional
        If True, regenerate regardless of existing outputs.
    use_denoised : bool, optional
        If True, run the denoised-trace variant of the pipeline (uses
        ``syn_dF-denoised.zarr`` as the detection signal; writes to
        ``glu_events_basic_denoised.parquet`` /
        ``noise_std_denoised.zarr``). Default False (matchfilt mode).
        The two modes have disjoint outputs and check/regenerate
        independently.
    **detection_kwargs
        Forwarded to ``glu_ev_basic_gen.detect_and_save`` (e.g.
        ``tau_s``, ``noise_window_s``, ``snr_threshold``).
    """
    mode_label = "denoised" if use_denoised else "matchfilt"
    tag = f"{subject} {exp} {loc} {acq}"

    # Prerequisite: ExperimentSummary mirror
    esum_path = _check_esum_mirror(subject, exp, loc, acq)
    if esum_path is None:
        print(f"[{tag}] Skipping — no ExperimentSummary mirror found")
        return

    # Prerequisite: scopex trace zarr (denoised or ls, depending on mode)
    if not _check_scopex_available(
        subject, exp, loc, acq, use_denoised=use_denoised
    ):
        zarr_name = (
            "syn_dF-denoised.zarr" if use_denoised else "syn_dF-ls.zarr"
        )
        print(
            f"[{tag}] Skipping — {zarr_name} not available "
            f"(run scopex_mon first)"
        )
        return

    # Fast path: everything present and version matches
    if not overwrite and _check_all_outputs(
        subject, exp, loc, acq, esum_path, use_denoised=use_denoised
    ):
        return

    # Generate
    print(f"[{tag}] Detecting basic glutamate events ({mode_label})...")
    glu_ev_basic_gen.detect_and_save(
        subject, exp, loc, acq,
        esum_p=esum_path,
        use_denoised=use_denoised,
        **detection_kwargs,
    )


def exp_data(
    subject: str,
    exp: str,
    overwrite: bool = False,
    use_denoised: bool = False,
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
    use_denoised : bool, optional
        Forwarded to ``acq_data``. See its docstring.
    **detection_kwargs
        Forwarded to ``acq_data``.
    """
    locacqs = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
    for la in locacqs:
        loc, acq = la.split("--")
        acq_data(
            subject, exp, loc, acq,
            overwrite=overwrite,
            use_denoised=use_denoised,
            **detection_kwargs,
        )


def all_subjects(
    overwrite: bool = False,
    use_denoised: bool = False,
    **detection_kwargs,
) -> None:
    """Run acq_data for all subjects, experiments, and acquisitions.

    Parameters
    ----------
    overwrite : bool, optional
        If True, regenerate all outputs regardless of existence/version.
    use_denoised : bool, optional
        Forwarded to ``acq_data``. See its docstring.
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
                        overwrite=overwrite,
                        use_denoised=use_denoised,
                        **detection_kwargs,
                    )
                except Exception as e:
                    print(f"[{subject} {exp} {loc} {acq}] Error: {e}")
                    continue
