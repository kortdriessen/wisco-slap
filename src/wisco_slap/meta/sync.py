# ============================================
# Meta File that this handles: sync_info.yaml
# ============================================

import os

import electro_py as epy
import numpy as np
import pandas as pd
import slap2_py as spy
import yaml

import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.meta.get import sync_info


def load_sync_block(
    subject: str, exp: str, sync_block: int
) -> tuple[np.ndarray, np.ndarray] | tuple[str, str]:
    """Load a sync block for a given subject and experiment.

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    sync_block : int
        The sync block number.
    """
    try:
        sync_blocks, sync_paths = get_all_sync_paths(subject, exp)
        sync_path = sync_paths[sync_block - 1]
        print(f"Loading sync block {sync_block} from {sync_path}")
        full_sync = spy.utils.drec.load_datarec_file(sync_path)
        return (
            full_sync["slap2_acquiring_trigger"][:],
            full_sync["electrophysiology"][:],
        )
    except OSError:
        return "Corrupt", "Corrupt"


def get_all_sync_paths(subject, exp) -> tuple[list[int], list[str]]:
    """Get all sync paths for a given subject and experiment.

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.

    Returns
    -------
    tuple[list[int], list[str]]
        A list of sync paths.
    """
    exp_dir = f"{DEFS.data_root}/{subject}/{exp}"
    sync_files = [f for f in os.listdir(exp_dir) if "SYNC_" in f]
    sync_blocks = [int(f.split("SYNC_")[-1].split(".")[0]) for f in sync_files]
    sync_blocks.sort()
    sync_paths = [os.path.join(exp_dir, f) for f in sync_files]
    sync_paths.sort()
    return sync_blocks, sync_paths


def detect_ephys_start_sample(ephys, fs=5000):
    thresh = np.max(ephys[: int(fs * 2)]) * 3
    samples = np.where(ephys > thresh)[0]
    return samples[0]


def get_acq_sync_block(subject, exp, loc, acq):
    si = sync_info()
    key = f"{loc}--{acq}"
    return si[subject][exp]["acquisitions"][key]["sync_block"]


def get_ephys_coverage(subject, exp):
    coverages = {}

    sb, sp = get_all_sync_paths(subject, exp)
    block_paths = wis.peri.ephys.get_block_paths(subject, exp)
    for block in sb:
        path = block_paths[block - 1]
        td = epy.tdt.io.load_tdt_block(path, 0, 1)
        td_start = pd.Timestamp(td.info["start_date"])
        td_end = pd.Timestamp(td.info["stop_date"])
        coverages[block] = (td_start, td_end)
    return coverages


def assign_acqs_to_sync_blocks(subject, exp, acqs=None, coverages=None):
    assignments = {}
    if coverages is None:
        coverages = get_ephys_coverage(subject, exp)
    if acqs is None:
        acqs = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
    for a in acqs:
        loc, acq = a.split("--")
        est_start = _est_start_time_from_file_name(subject, exp, loc, acq)
        assert est_start is not None, f"Estimated start time could not be found for {a}"
        for block in coverages.keys():
            if coverages[block][0] <= est_start <= coverages[block][1]:
                assignments[a] = block
                break
    return assignments


def load_all_sync_blocks(subject, exp):
    scopes = {}
    ephyses = {}
    sb, sp = get_all_sync_paths(subject, exp)
    for block in sb:
        scopes[block], ephyses[block] = load_sync_block(subject, exp, block)
    return scopes, ephyses


def match_scope_up_window(
    scope, ephys, dat_start, ephys_start_ts, fs=5000, ephys_start_sample=None
):
    """Pure offset-matcher: given a SYNC file's scope and ephys channels, a
    microscope acquisition wall-clock start, and the TDT block's wall-clock
    start, return the best-matching ``ephys_offset`` in seconds.

    Extracted from ``compute_ephys_offset`` so it can be called per-epoch
    (with a different ``dat_start`` for each epoch) as well as per-acq.

    Parameters
    ----------
    scope : np.ndarray
        The ``slap2_acquiring_trigger`` channel from the SYNC HDF5 file.
    ephys : np.ndarray
        The ``electrophysiology`` channel from the SYNC HDF5 file.
    dat_start : pd.Timestamp
        Wall-clock start time of the microscope acquisition/epoch.
    ephys_start_ts : pd.Timestamp
        Wall-clock start time of the TDT block (from ``sync_info.yaml``
        ``sync_blocks[<block>].ephys_start``).
    fs : int, optional
        SYNC file sampling rate in Hz, by default 5000.
    ephys_start_sample : int, optional
        Pre-detected ephys-ON sample index. If None, auto-detected via
        :func:`detect_ephys_start_sample`.

    Returns
    -------
    float
        ``ephys_offset`` in seconds — i.e. seconds from the TDT block's start
        to this acquisition/epoch's start.
    """
    scope_df = spy.utils.drec.generate_scope_index_df(scope)
    if ephys_start_sample is None:
        ephys_start_sample = detect_ephys_start_sample(ephys)
    scope_df["ephys_up"] = ephys_start_sample
    scope_df["ephys_scope_diff"] = scope_df["start_idx"] - scope_df["ephys_up"]
    scope_df["ephys_scope_diff_s"] = scope_df["ephys_scope_diff"] / fs

    est_start = pd.Timestamp(dat_start)
    ephys_start = pd.Timestamp(ephys_start_ts)
    wall_gap_abs = np.abs((ephys_start - est_start).total_seconds())
    est_start_differential = np.abs(scope_df["ephys_scope_diff_s"] - wall_gap_abs)
    scope_df["est_start_differential"] = est_start_differential
    print(scope_df["est_start_differential"].min())
    ephys_offset = scope_df.loc[
        scope_df["est_start_differential"].idxmin(), "ephys_scope_diff"
    ]
    return ephys_offset / fs


def compute_ephys_offset(
    subject, exp, loc, acq, si, scope=None, ephys=None, fs=5000, ephys_start_sample=None
):
    """Compute ephys_offset for an acq by looking up sync-block + dat_start
    in sync_info, loading the SYNC file if needed, and running the matcher.

    For multi-epoch acqs this returns epoch 1's offset (since ``dat_start``
    in sync_info is defined as epoch 1's filename timestamp).
    """
    acq_id = f"{loc}--{acq}"
    block = si[subject][exp]["acquisitions"][acq_id]["sync_block"]
    if scope is None and ephys is None:
        scope, ephys = load_sync_block(subject, exp, block)
    dat_start = pd.Timestamp(si[subject][exp]["acquisitions"][acq_id]["dat_start"])
    ephys_start_ts = pd.Timestamp(
        si[subject][exp]["sync_blocks"][block]["ephys_start"]
    )
    return match_scope_up_window(
        scope, ephys, dat_start, ephys_start_ts, fs=fs,
        ephys_start_sample=ephys_start_sample,
    )


def find_corrupted_sync_offsets(subject, exp, loc, acq, si):
    start_time = pd.Timestamp(
        si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["dat_start"]
    )
    # find all the candidate start times of non-corrupt sync blocks
    starts = []
    acq_ids = []
    for acq_id in si[subject][exp]["acquisitions"].keys():
        if (
            si[subject][exp]["sync_blocks"][
                si[subject][exp]["acquisitions"][acq_id]["sync_block"]
            ]["corrupt"]
            is False
        ):
            starts.append(
                pd.Timestamp(si[subject][exp]["acquisitions"][acq_id]["dat_start"])
            )
            acq_ids.append(acq_id)

    if len(starts) == 0:
        print(f"No valid start times found for {subject} {exp} {loc} {acq}")
        return None

    # find the start time (and corresponding acq_id) that is closest to the acquisition start time
    closest_start = min(starts, key=lambda x: abs(x - start_time))
    closest_acq_id = acq_ids[starts.index(closest_start)]

    # get the ephys offset for the closest start time
    closest_ephys_offset = si[subject][exp]["acquisitions"][closest_acq_id][
        "ephys_offset"
    ]

    # find the pure difference in time between the closest start time and the acquisition start time
    if closest_start > start_time:
        pure_diff = (closest_start - start_time).total_seconds()
    else:
        pure_diff = (start_time - closest_start).total_seconds()

    # put that closest start time into the ephys time base, which should be precisely synchronized
    good_ephys_start = pd.Timestamp(
        si[subject][exp]["sync_blocks"][
            si[subject][exp]["acquisitions"][closest_acq_id]["sync_block"]
        ]["ephys_start"]
    )
    close_start_in_ephys = good_ephys_start + pd.Timedelta(seconds=closest_ephys_offset)

    # put the corrupt file in the same ephys time base
    if closest_start > start_time:
        corrupt_time_start_in_ephys = close_start_in_ephys - pd.Timedelta(
            seconds=pure_diff
        )
    else:
        corrupt_time_start_in_ephys = close_start_in_ephys + pd.Timedelta(
            seconds=pure_diff
        )

    # get the corrupt block's ephys start time
    corrupt_block_start = pd.Timestamp(
        si[subject][exp]["sync_blocks"][
            si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["sync_block"]
        ]["ephys_start"]
    )

    # find the difference in time between the corrupt start time and the corrupt time start in ephys (everything should now be synced into the ephys time base)
    diff_in_time = (corrupt_time_start_in_ephys - corrupt_block_start).total_seconds()
    return diff_in_time


def _list_epoch_start_times(subject, exp, loc, acq):
    """Return the per-epoch start times parsed from the DMD1 first-cycle
    .dat filenames in this acq's directory.

    Returns a list of ``(filename, timestamp)`` tuples, sorted ascending by
    ``timestamp``. Single-epoch acqs produce a list of length 1; multi-epoch
    acqs (paused and restarted during scanning, typically re-assembled by the
    preprocessing pipeline's epoch functionality) produce length > 1. Returns
    an empty list if no matching files are found.

    Sorting is by the parsed timestamp, NOT by filename string — the
    ``acq-N_`` prefix is not guaranteed to be monotonic with wall clock (and
    lexicographic sort breaks at ``acq-10`` anyway).
    """
    acq_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}"
    if not os.path.isdir(acq_dir):
        return []
    found = []
    for f in os.listdir(acq_dir):
        if "_DMD1-CYCLE-000000.dat" in f:
            ts_str = f.split("_DMD1-CYCLE-000000.dat")[0][-15:]
            try:
                ts = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
            except ValueError:
                continue
            found.append((f, ts))
    found.sort(key=lambda x: x[1])
    return found


def _est_start_time_from_file_name(subject, exp, loc, acq, verbose=True):
    """Parse the acquisition-start wall-clock time from the DMD1 first-cycle
    .dat filename.

    The SLAP2 acquisition software stamps the wall-clock start time into every
    raw .dat filename as ``_YYYYMMDD_HHMMSS_`` (1-second resolution). This is
    precise enough to anchor scope-up-window matching in the SYNC file,
    avoiding the need for a MATLAB-generated ``datStartTimes1.txt`` file.

    For multi-epoch acqs (multiple first-cycle .dat files, one per epoch),
    returns the earliest timestamp — i.e. epoch 1.
    """
    epochs = _list_epoch_start_times(subject, exp, loc, acq)
    if not epochs:
        if verbose:
            acq_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}"
            print(
                f"No file with _DMD1-CYCLE-000000.dat found in {acq_dir}, "
                f"returning None for estimated start time"
            )
        return None
    return epochs[0][1]


def _has_first_cycle_dat_file(subject, exp, loc, acq):
    """Return True if at least one DMD1 first-cycle .dat file exists for
    this acq.

    Used as the precondition check for sync-info population, replacing the
    previous ``datStartTimes1.txt`` existence check.
    """
    acq_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}"
    if not os.path.isdir(acq_dir):
        return False
    for f in os.listdir(acq_dir):
        if "_DMD1-CYCLE-000000.dat" in f:
            return True
    return False


def update_sync_info(subject, exp, redo=False, ephys_start_sample=None):
    """update the sync info file for a given experiment.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    redo : bool, optional
        whether to redo the sync info, by default False
    ephys_start_sample : dict, optional
        dictionary mapping acquisition IDs to their corresponding ephys start samples, by default None
    """
    si = sync_info()
    if si is None:
        si = {}
    if subject not in si:
        si[subject] = {}
    if exp not in si[subject]:
        si[subject][exp] = {}
        si[subject][exp]["sync_blocks"] = {}
        si[subject][exp]["acquisitions"] = {}

    acqs = []
    acqs_all = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
    for acq_id in acqs_all:
        if acq_id not in si[subject][exp]["acquisitions"].keys():
            loc, acq = acq_id.split("--")
            if _has_first_cycle_dat_file(subject, exp, loc, acq):
                acqs.append(acq_id)

    # Fast backfill of cheap-to-compute fields on existing entries. Only add
    # fields here that are (a) cheap — no SYNC loads / hysteresis / offset
    # math — and (b) a pure function of current raw-data state. For anything
    # expensive, or anything that re-touches the ephys_offset matcher, use
    # redo=True. Skips any entry whose raw files have disappeared (leaves
    # n_epochs absent rather than writing the non-sensical value 0, since an
    # acq by definition has >= 1 epoch).
    backfilled = False
    for acq_id, entry in si[subject][exp].get("acquisitions", {}).items():
        if acq_id in acqs:
            continue  # about to be freshly populated below
        loc, acq = acq_id.split("--")
        if "n_epochs" not in entry and _has_first_cycle_dat_file(
            subject, exp, loc, acq
        ):
            entry["n_epochs"] = len(
                _list_epoch_start_times(subject, exp, loc, acq)
            )
            backfilled = True

    if len(acqs) == 0 and not redo:
        if backfilled:
            with open(f"{DEFS.anmat_root}/sync_info.yaml", "w") as f:
                yaml.dump(si, f)
            print(
                f"Backfilled missing cheap fields for {subject} {exp} "
                f"(no new acqs to fully process)."
            )
        else:
            print(
                f"All acquisitions already have sync info for {subject} {exp}. "
                f"Use redo=True to recompute."
            )
        return

    sb, sp = get_all_sync_paths(subject, exp)
    coverage = get_ephys_coverage(subject, exp)
    scope, ephys = load_all_sync_blocks(subject, exp)
    for block in sb:
        if block in si[subject][exp]["sync_blocks"].keys() and not redo:
            continue
        si[subject][exp]["sync_blocks"][block] = {}
        si[subject][exp]["sync_blocks"][block]["ephys_start"] = str(coverage[block][0])
        si[subject][exp]["sync_blocks"][block]["ephys_end"] = str(coverage[block][1])
        if type(scope[block]) is str and type(ephys[block]) is str:
            if scope[block] == "Corrupt" and ephys[block] == "Corrupt":
                si[subject][exp]["sync_blocks"][block]["corrupt"] = True
            else:
                si[subject][exp]["sync_blocks"][block]["corrupt"] = (
                    "Unknown - check sync files"
                )
        else:
            si[subject][exp]["sync_blocks"][block]["corrupt"] = False

    # Now we update the specific acquisitions with block assignments and ephys offsets
    # acqs = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
    assignments = assign_acqs_to_sync_blocks(subject, exp, acqs, coverage)

    for acq_id in acqs:
        loc, acq = acq_id.split("--")
        epochs = _list_epoch_start_times(subject, exp, loc, acq)
        assert epochs, (
            f"No DMD1 first-cycle .dat file found for {subject}/{exp}/{loc}/{acq} "
            f"despite _has_first_cycle_dat_file passing — race condition?"
        )
        dat_start = epochs[0][1]  # earliest epoch (epoch 1)
        n_epochs = len(epochs)
        si[subject][exp]["acquisitions"][acq_id] = {}
        si[subject][exp]["acquisitions"][acq_id]["sync_block"] = assignments[acq_id]
        si[subject][exp]["acquisitions"][acq_id]["dat_start"] = str(dat_start)
        si[subject][exp]["acquisitions"][acq_id]["n_epochs"] = n_epochs
        if ephys_start_sample is not None and acq_id in ephys_start_sample.keys():
            start_sample_to_use = ephys_start_sample[acq_id]
        else:
            start_sample_to_use = None
        if si[subject][exp]["sync_blocks"][assignments[acq_id]]["corrupt"] is False:
            offset = compute_ephys_offset(
                subject,
                exp,
                loc,
                acq,
                si,
                scope[assignments[acq_id]],
                ephys[assignments[acq_id]],
                ephys_start_sample=start_sample_to_use,
            )
            offset = float(offset)
        else:
            offset = find_corrupted_sync_offsets(subject, exp, loc, acq, si)
            offset = float(offset) if offset is not None else None
        si[subject][exp]["acquisitions"][acq_id]["ephys_offset"] = offset

    # One final retry for any acquisitions still unresolved (offset is None),
    # which can happen when a corrupt acq was processed before any non-corrupt
    # reference was populated. By this pass, references are in place, so the
    # lookup usually succeeds. Anything still None after this means no
    # non-corrupt sync block exists anywhere in the experiment to anchor against.
    for acq_id in si[subject][exp]["acquisitions"].keys():
        loc, acq = acq_id.split("--")
        if si[subject][exp]["acquisitions"][acq_id]["ephys_offset"] is None:
            ephys_offset = find_corrupted_sync_offsets(subject, exp, loc, acq, si)
            si[subject][exp]["acquisitions"][acq_id]["ephys_offset"] = (
                float(ephys_offset) if ephys_offset is not None else None
            )

    # write the updated sync info to the yaml file
    with open(f"{DEFS.anmat_root}/sync_info.yaml", "w") as f:
        yaml.dump(si, f)
        f.close()
    return
