# ==================================================================================
# Main entry point for getting information from or associated with the meta files
# ==================================================================================

import csv
import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import yaml

from wisco_slap.defs import anmat_root, data_root, exsum_mirror_root


# ==================================================================================
# 1. Sync Info
# ==================================================================================
def sync_info():
    si_path = f"{anmat_root}/sync_info.yaml"
    with open(si_path) as f:
        si = yaml.load(f, Loader=yaml.SafeLoader)
    return si


def ephys_offset(subject, exp, loc, acq):
    """Get the ephys offset for a given recording.

    This is the amount of time (in seconds) that the ephys data is shifted
    relative to the other data streams. This is necessary because the ephys
    data is recorded on a separate system and may not be perfectly
    synchronized with the other data streams.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    loc : str
        location name
    acq : str
        acquisition name

    Returns
    -------
    offset : float
        ephys offset in seconds

    Raises
    ------
    ValueError
        If the sync pipeline was unable to resolve an offset for this acquisition
        (e.g. the SYNC file is corrupt and no non-corrupt acq exists in the
        experiment to anchor against). In `sync_info.yaml` this is stored as
        `null`.
    """
    si = sync_info()
    acq_id = f"{loc}--{acq}"
    offset = si[subject][exp]["acquisitions"][acq_id]["ephys_offset"]
    if offset is None:
        raise ValueError(
            f"ephys_offset is unresolved for {subject}/{exp}/{loc}/{acq}. "
            f"The sync pipeline could not compute or interpolate an offset "
            f"(likely all sync blocks in this experiment are corrupt). "
            f"Re-run wis.meta.sync.update_sync_info after recovering at least "
            f"one non-corrupt SYNC file, or pass apply_ephys_offset=False to "
            f"bypass alignment."
        )
    return offset


def ephys_datetime(subject, exp, loc, acq):
    """Get the datetime start of the ephys recording for a given acquisition.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    loc : str
        location name
    acq : str
        acquisition name

    Returns
    -------
    datetime : str
        ephys recording datetime in ISO format (YYYY-MM-DDTHH:MM:SS)

    Raises
    ------
    ValueError
        If the ephys datetime is unresolved for this acquisition. In `sync_info.yaml`
        this is stored as `null`.
    """
    si = sync_info()
    sb = _resolve_acq_sync_block(si, subject, exp, loc, acq)
    if sb is None:
        raise ValueError(
            f"sync_block is unresolved for {subject}/{exp}/{loc}/{acq}, so "
            f"ephys_datetime cannot be resolved either. Re-run "
            f"wis.meta.sync.update_sync_info after recovering the necessary "
            f"metadata, or bypass alignment if you don't need the datetime."
        )
    dt = si[subject][exp]["sync_blocks"][sb]["ephys_start"]

    if dt is None:
        raise ValueError(
            f"ephys_datetime is unresolved for {subject}/{exp}/{loc}/{acq}. "
            f"Re-run wis.meta.sync.update_sync_info after recovering the "
            f"necessary metadata, or bypass alignment if you don't need "
            f"the datetime."
        )
    return dt


# ==================================================================================
# 1b. Epoch Info (per-epoch metadata for multi-epoch acqs)
# ==================================================================================
def epoch_info():
    """Load ``epoch_info.yaml``. Returns an empty dict if the file is absent."""
    ei_path = f"{anmat_root}/epoch_info.yaml"
    if not os.path.exists(ei_path):
        return {}
    with open(ei_path) as f:
        ei = yaml.load(f, Loader=yaml.SafeLoader)
    return ei or {}


# ==================================================================================
# 2. prepro info
# ==================================================================================
def prepro_info():
    prepro_path = f"{anmat_root}/prepro_info.yaml"
    with open(prepro_path) as f:
        prepro_info = yaml.load(f, Loader=yaml.SafeLoader)
    return prepro_info


# ==================================================================================
# 3. dmd info
# ==================================================================================
def dmd_info():
    path = os.path.join(anmat_root, "dmd_info.yaml")
    with open(path) as f:
        dmd_info = yaml.safe_load(f)
    return dmd_info


def soma_dmd(subject, exp, loc, acq, soma_id):
    """Get the DMD number (1 or 2) for a given soma ID.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    loc : str
        location name
    acq : str
        acquisition name
    soma_id : int or str
        soma ID to look up

    Returns
    -------
    dmd : int
        DMD number (1 or 2) for the given soma ID

    Raises
    ------
    ValueError
        If the soma ID is not found in either DMD for this acquisition.
    """
    dmd_info_data = dmd_info()
    try:
        acq_info = dmd_info_data[subject][exp][loc][acq]
    except (KeyError, TypeError):
        raise ValueError(f"No DMD info found for {subject}/{exp}/{loc}/{acq}")

    for dmd in ("dmd-1", "dmd-2"):
        dmd_somas = acq_info[dmd]["somas"]
        if soma_id in dmd_somas:
            return int(dmd[-1])  # Extract the DMD number from the key


# ==================================================================================
# 4. acquisition master
# ==================================================================================
def acq_master():
    path = os.path.join(anmat_root, "acquisition_master.yaml")
    with open(path) as f:
        master_acq = yaml.safe_load(f)
    return master_acq


def _load_exsum_audit_lookup(
    *, refresh_audit: bool = False
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    """Load the status-monitor ExSum audit CSV, keyed by acquisition tuple."""
    path = Path(anmat_root) / "ExSum_audits.csv"
    if refresh_audit:
        from wisco_slap.meta import status as status_mod

        status_mod.refresh_exsum_audit(csv_path=path)

    if not path.is_file():
        raise FileNotFoundError(
            f"No ExSum audit CSV at {path}. Run "
            "wis.meta.status.refresh_exsum_audit() first, or call "
            "wis.meta.get.valid_acquisitions(refresh_audit=True)."
        )

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return {
            (
                row.get("subject", ""),
                row.get("exp", ""),
                row.get("loc", ""),
                row.get("acq", ""),
            ): row
            for row in reader
        }


def _audit_status_is_fresh(
    audit_row: dict[str, str] | None,
    status_field: str,
) -> bool:
    return audit_row is not None and (audit_row.get(status_field) or "") == "fresh"


def _events_exsum_is_complete(audit_row: dict[str, str] | None) -> bool:
    """Match WISynaptic's rolled-up Events ExSum green/not-green semantics."""
    if audit_row is None:
        return False

    matchfilt = (audit_row.get("events_matchfilt_status") or "").strip()
    denoised = (audit_row.get("events_denoised_status") or "").strip()
    if not matchfilt and not denoised:
        return False

    severity_order = {
        "stale": 3,
        "missing": 2,
        "fresh": 1,
        "not_applicable": 0,
        "missing_mirror": 0,
        "": 0,
    }
    severity = max(
        severity_order.get(matchfilt, 0),
        severity_order.get(denoised, 0),
    )
    rollup = "fresh"
    for status in (matchfilt, denoised):
        if severity_order.get(status, 0) == severity and status:
            rollup = status
            break

    return rollup == "fresh"


def _audit_scoring_is_complete(audit_row: dict[str, str] | None) -> bool:
    if audit_row is None:
        return False
    complete = (audit_row.get("scoring_complete") or "").strip().lower()
    return complete in {"true", "1", "yes"}


def _annotation_dir(subject: str, exp: str, loc: str, acq: str) -> Path:
    return Path(anmat_root) / "annotation_materials" / subject / exp / loc / acq


def _paired_files_exist(path1: Path, path2: Path) -> bool:
    return path1.is_file() and path2.is_file()


def _label_completeness_is_complete(annotation_dir: Path) -> bool:
    """Return True when every numeric synapse PNG has a non-empty soma label."""
    total_labelled = 0
    total_count = 0
    any_synapses = False

    for dmd in ("dmd-1", "dmd-2"):
        dmd_dir = annotation_dir / "synapse_ids" / dmd
        png_count = sum(1 for p in dmd_dir.glob("*.png") if p.stem.isdigit())
        if png_count == 0:
            continue

        any_synapses = True
        total_count += png_count

        csv_path = dmd_dir / "synapse_labels.csv"
        if not csv_path.is_file():
            continue

        try:
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    source_id = (row.get("source-ID") or "").strip()
                    if not source_id.isdigit():
                        continue
                    soma_id = (row.get("soma-ID") or "").strip()
                    if soma_id:
                        total_labelled += 1
        except Exception:
            return False

    return any_synapses and total_count > 0 and total_labelled == total_count


def _prox_lines_are_complete(annotation_dir: Path) -> bool:
    return _paired_files_exist(
        annotation_dir / "source_sorting" / "prox_lines_dmd1.csv",
        annotation_dir / "source_sorting" / "prox_lines_dmd2.csv",
    )


def _depth_values_are_complete(
    dmd_info_data: dict[str, Any],
    subject: str,
    exp: str,
    loc: str,
    acq: str,
) -> bool:
    try:
        acq_info = dmd_info_data[subject][exp][loc][acq]
    except (KeyError, TypeError):
        return False

    for dmd in ("dmd-1", "dmd-2"):
        dmd_data = acq_info.get(dmd, {}) if isinstance(acq_info, dict) else {}
        depth = dmd_data.get("depth") if isinstance(dmd_data, dict) else None
        if depth is None or depth == -1:
            return False
    return True


def _resolve_acq_sync_block(
    sync_info_data: dict[str, Any],
    subject: str,
    exp: str,
    loc: str,
    acq: str,
) -> int | None:
    try:
        return int(
            sync_info_data[subject][exp]["acquisitions"][f"{loc}--{acq}"]["sync_block"]
        )
    except (KeyError, TypeError, ValueError):
        return None


def _sync_block_dir(subject: str, exp: str, sync_block: int) -> Path:
    return (
        Path(anmat_root)
        / subject
        / exp
        / "sync_block_data"
        / f"sync_block-{sync_block}"
    )


def _ephys_is_complete(sync_block_dir: Path) -> bool:
    ephys_dir = sync_block_dir / "ephys"
    if not ephys_dir.is_dir():
        return False
    return any(ephys_dir.glob("*.nc"))


def _split_acq_id(acq_id: str) -> tuple[str, str]:
    parts = acq_id.split("--")
    if len(parts) != 2:
        raise ValueError(f"Invalid acquisition id in acquisition_master.yaml: {acq_id}")
    return parts[0], parts[1]


def valid_acquisitions(
    *, refresh_audit: bool = False, exclude_cross_sb: bool = True
) -> list[str]:
    """Return acquisition-master entries with all critical monitor items green.

    The returned strings have the form ``"subject--exp--loc--acq"``. The
    critical checklist currently corresponds to:

    - Scopex ExSum
    - Events ExSum
    - Annotation ExSum
    - Scoring Complete
    - Label Completeness
    - Prox Lines
    - Depth Values
    - Whisk DF
    - Eye Metrics
    - Ephys

    ExSum-derived columns and scoring status are read from the same
    ``ExSum_audits.csv`` cache used by the WISynaptic status monitor. Pass
    ``refresh_audit=True`` to rebuild that cache before filtering.

    Parameters
    ----------
    refresh_audit : bool
        If True, rebuild ``ExSum_audits.csv`` before filtering.
    exclude_cross_sb : bool
        If True (default), exclude multi-epoch acqs whose epochs span more
        than one TDT sync block. Cross-sync-block acqs require special-case
        handling for any cross-stream alignment (per-sync-block artifacts
        like hypnograms, ephys, eye/whisk live on the spanned blocks'
        individual TDT axes), so by default they are not considered
        analysis-ready even when all material checks pass. Set False to
        include them and accept that downstream alignment is the caller's
        problem.
    """
    master = acq_master()
    audit_lookup = _load_exsum_audit_lookup(refresh_audit=refresh_audit)
    dmd_info_data = dmd_info()
    sync_info_data = sync_info()
    epoch_info_data = epoch_info() if exclude_cross_sb else {}

    valid: list[str] = []
    for subject, experiments in master.items():
        for exp, acq_ids in experiments.items():
            for acq_id in acq_ids:
                loc, acq = _split_acq_id(acq_id)
                audit_row = audit_lookup.get((subject, exp, loc, acq))
                annotation_dir = _annotation_dir(subject, exp, loc, acq)
                sync_block = _resolve_acq_sync_block(
                    sync_info_data, subject, exp, loc, acq
                )
                if sync_block is None:
                    continue

                if exclude_cross_sb:
                    acq_ei = (
                        epoch_info_data.get(subject, {}).get(exp, {}).get(acq_id, {})
                    )
                    if acq_ei:
                        sbs = {int(acq_ei[k]["sync_block"]) for k in acq_ei}
                        if len(sbs) > 1:
                            continue

                sb_dir = _sync_block_dir(subject, exp, sync_block)
                if all((
                    _audit_status_is_fresh(audit_row, "scopex_status"),
                    _events_exsum_is_complete(audit_row),
                    _audit_status_is_fresh(audit_row, "annotation_status"),
                    _audit_scoring_is_complete(audit_row),
                    _label_completeness_is_complete(annotation_dir),
                    _prox_lines_are_complete(annotation_dir),
                    _depth_values_are_complete(dmd_info_data, subject, exp, loc, acq),
                    (sb_dir / "whisking" / "whisk_df.parquet").is_file(),
                    (
                        sb_dir / "pupil" / "eye_metrics" / "eye_metrics.parquet"
                    ).is_file(),
                    _ephys_is_complete(sb_dir),
                )):
                    valid.append(f"{subject}--{exp}--{loc}--{acq}")

    return valid


def unique_acquisitions_per_experiment(subject, exp):
    """For a given subject and experiment, this function will return the unique
    acquisitions from acquisition_master.yaml. There may be more actual acquisitions in
    the raw data directory, but if they are not listed in acquisition_master.yaml,
    then they are not returned here!

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name

    Returns
    -------
    acqs : list[str]
        List of unique acquisitions for the given subject and experiment.
    """

    macq = acq_master()
    return macq[subject][exp]


def locacq(subject: str, exp: str):
    """For a given subject and experiment, returns loc, acq for that experiment

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name

    Returns
    -------
    list[tuple[str, str]]
        List of (location, acquisition) tuples for the given subject and experiment.

    Raises
    ------
    ValueError
        If the subject or experiment is not found in the acquisition_master.yaml file.
    ValueError
        _description_
    """
    macq = acq_master()
    if subject not in macq:
        raise ValueError(f"Subject {subject} not found in acquisition_master.yaml")
    if exp not in macq[subject]:
        raise ValueError(
            f"Experiment {exp} not found for subject {subject} "
            "in acquisition_master.yaml"
        )
    locacqs = macq[subject][exp]
    return [locacq.split("--") for locacq in locacqs]


# ==================================================================================
# 4. SyncBlock Scoring Times
# ==================================================================================
def sb_scoring_times():
    path = os.path.join(anmat_root, "sb_scoring_times.yaml")
    with open(path) as f:
        sb_scoring_times = yaml.safe_load(f)
    return sb_scoring_times


# ==================================================================================
# MISC META INFO
# ==================================================================================
def esum_mirror_path(subject, exp, loc, acq):
    mirror_dir = os.path.join(exsum_mirror_root, subject, exp, f"{loc}--{acq}")
    if not os.path.exists(mirror_dir):
        return 0
    esum_files = [f for f in os.listdir(mirror_dir) if f.endswith(".mat")]
    if len(esum_files) == 0:
        return "NO_ESUM_MIRROR"
    elif len(esum_files) == 1:
        return os.path.join(mirror_dir, esum_files[0])
    else:
        raise ValueError(
            f"Multiple esum files found in mirror for {subject} {exp} "
            f"{loc} {acq}! Fix this manually ASAP!!"
        )


class AcqTiming(NamedTuple):
    """Timing summary for one acquisition.

    Attributes
    ----------
    start_time : float
        Start of the acquisition, in seconds since the start of the TDT block
        named by ``sync_block``. Equals the acq's ``ephys_offset``.
    end_time : float
        End of the last microscope sample, on the same TDT time axis. For
        single-epoch acqs this is ``start_time + duration``; for single-
        sync-block multi-epoch acqs it includes the inter-epoch gap.
    duration : float
        Total *active* microscope time (sum of per-trial durations), seconds.
        Excludes inter-trial gaps and inter-epoch gaps.
    sync_block : int
        TDT sync block whose time axis ``start_time``/``end_time`` are on.
    """

    start_time: float
    end_time: float
    duration: float
    sync_block: int


def acq_timing(subject: str, exp: str, loc: str, acq: str) -> AcqTiming:
    """Start, end, and total active duration for one acquisition.

    Reads the canonical scopex zarr (``syn_dF-ls.zarr``) for sample timing
    and looks up ``ephys_offset`` from ``sync_info.yaml`` to translate the
    scopex time coord (which starts at 0) into TDT-block-relative seconds.

    Active duration is computed as ``len(time) * median(diff(time))`` —
    exact for both single- and multi-epoch zarrs (the median of ``diff``
    recovers ``1/fs`` cleanly because epoch-boundary jumps are rare).

    For multi-epoch acqs whose epochs span more than one TDT sync block,
    both ``start_time`` and ``end_time`` are reported on **the first sync
    block's TDT axis** (the same convention as ``wis.get.syn_dF`` with
    ``apply_ephys_offset=True``). The scopex time axis is FPGA-stitched
    across epochs, so ``end_time`` for a cross-sync-block acq is wall-clock-
    extrapolated past the first sync block's end — valid for aligning
    against anything anchored to that first block's start, but it may
    exceed the first block's actual ``ephys_end``. ``sync_block`` reports
    epoch 1's block (the anchor). For per-epoch ephys alignment in later
    blocks, use ``wis.meta.get.epoch_info()``. A ``UserWarning`` is emitted
    in this case so the limitation isn't silent.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Standard acquisition identifiers.

    Returns
    -------
    AcqTiming

    Raises
    ------
    FileNotFoundError
        If ``syn_dF-ls.zarr`` does not yet exist for this acquisition. Run
        ``wis.pns.scopex_mon.exp_data(subject, exp)`` first.
    ValueError
        If ``ephys_offset`` is unresolved (corrupt SYNC).
    """
    import warnings

    import xarray as xr

    zarr_path = os.path.join(
        anmat_root, subject, exp, "scopex", f"{loc}--{acq}", "syn_dF-ls.zarr"
    )
    if not os.path.isdir(zarr_path):
        raise FileNotFoundError(
            f"syn_dF-ls.zarr not found for {subject}/{exp}/{loc}/{acq}: "
            f"{zarr_path}. Run wis.pns.scopex_mon.exp_data first."
        )

    for group in ("dmd_1", "dmd_2"):
        if os.path.isdir(os.path.join(zarr_path, group)):
            ds = xr.open_zarr(zarr_path, group=group)
            break
    else:
        raise FileNotFoundError(f"{zarr_path} contains neither a dmd_1 nor dmd_2 group")

    n_samples = int(ds.sizes["time"])
    if n_samples < 2:
        raise ValueError(
            f"{subject}/{exp}/{loc}/{acq}: scopex time axis has < 2 samples; "
            f"cannot infer sample period."
        )
    time = ds.time.values
    dt = float(np.median(np.diff(time)))
    duration = n_samples * dt

    acq_id = f"{loc}--{acq}"
    acq_ei = epoch_info().get(subject, {}).get(exp, {}).get(acq_id, {})
    if acq_ei:
        sbs = sorted({int(acq_ei[k]["sync_block"]) for k in acq_ei})
        if len(sbs) > 1:
            warnings.warn(
                f"{subject}/{exp}/{loc}/{acq} is multi-epoch and spans "
                f"sync_blocks {sbs}. start_time/end_time are reported on "
                f"sync_block-{sbs[0]}'s TDT axis (epoch 1's anchor); "
                f"end_time is wall-clock-extrapolated past that block's "
                f"ephys_end for samples that physically belong to later "
                f"sync blocks. For per-epoch ephys alignment, use "
                f"wis.meta.get.epoch_info().",
                UserWarning,
                stacklevel=2,
            )

    eo = ephys_offset(subject, exp, loc, acq)
    sb = int(sync_info()[subject][exp]["acquisitions"][acq_id]["sync_block"])
    start_time = eo + float(time[0])
    end_time = eo + float(time[-1]) + dt
    return AcqTiming(
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        sync_block=sb,
    )


def esum_path_raw(subject: str, exp: str, loc: str, acq: str) -> str | None:
    """Get path to summary data for a given recording.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    loc : str
        location name
    acq : str
        acquisition name
    """
    esum_dir = os.path.join(data_root, subject, exp, loc, acq, "ExperimentSummary")
    if not os.path.exists(esum_dir):
        return None
    # Should only be one file inside this folder
    esum_files = [f for f in os.listdir(esum_dir) if f.endswith(".mat")]
    if len(esum_files) == 0:
        return None
    elif len(esum_files) == 1:
        return os.path.join(esum_dir, esum_files[0])
    else:
        raise ValueError(f"Multiple esum files found for {subject} {exp} {loc} {acq}")
