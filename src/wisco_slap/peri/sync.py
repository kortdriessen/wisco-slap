import os

import electro_py as epy
import numpy as np
import pandas as pd
import polars as pl
import slap2_py as spy
import yaml

import wisco_slap as wis
import wisco_slap.defs as DEFS


def load_datStartTimes(subject, exp, loc, acq):
    path = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}/datStartTimes1.txt"
    if not os.path.exists(path):
        print("DAT START TIMES FILE NOT FOUND, RUN MATLAB SCRIPT TO GENERATE")
        raise FileNotFoundError(f"File not found: {path}")
    with open(path) as f:
        datStartTimes = [pd.Timestamp(line.strip()) for line in f]
    return datStartTimes


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


def load_sync_info():
    si_path = f"{DEFS.anmat_root}/sync_info.yaml"
    with open(si_path) as f:
        si = yaml.load(f, Loader=yaml.SafeLoader)
    return si


def get_acq_sync_block(subject, exp, loc, acq):
    si = load_sync_info()
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
        acqs = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
    ei = wis.util.info.load_exp_info_spreadsheet()
    for a in acqs:
        loc, acq = a.split("--")
        est_start = (
            ei
            .filter(pl.col("subject") == subject)
            .filter(pl.col("experiment") == exp)
            .filter(pl.col("location") == loc)
            .filter(pl.col("acquisition") == acq)["estimated_start_time"]
            .to_numpy()[0]
        )
        d, t = est_start.split("_")
        est_start = pd.Timestamp(d + " " + t)
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


def compute_ephys_offset(subject, exp, loc, acq, si, scope=None, ephys=None, fs=5000):
    acq_id = f"{loc}--{acq}"
    block = si[subject][exp]["acquisitions"][acq_id]["sync_block"]
    if scope is None and ephys is None:
        scope, ephys = load_sync_block(subject, exp, block)
    scope_df = spy.utils.drec.generate_scope_index_df(scope)
    scope_df["sync_block"] = block
    ephys_start_sample = detect_ephys_start_sample(ephys)
    scope_df["ephys_up"] = ephys_start_sample
    scope_df["ephys_scope_diff"] = scope_df["start_idx"] - scope_df["ephys_up"]
    scope_df["ephys_scope_diff_s"] = scope_df["ephys_scope_diff"] / fs
    ephys_start_estimate = pd.Timestamp(
        si[subject][exp]["sync_blocks"][block]["ephys_start"]
    )
    scope_df["ephys_start_estimate"] = ephys_start_estimate

    est_start = pd.Timestamp(si[subject][exp]["acquisitions"][acq_id]["dat_start"])

    est_start_diff = [
        (scope_df["ephys_start_estimate"][i] - est_start).total_seconds()
        for i in range(len(scope_df))
    ]
    est_start_diff_abs = [np.abs(est_start_diff[i]) for i in range(len(est_start_diff))]
    est_start_differential = np.abs(scope_df["ephys_scope_diff_s"] - est_start_diff_abs)
    scope_df["est_start_differential"] = est_start_differential
    print(scope_df["est_start_differential"].min())
    ephys_offset = scope_df.loc[
        scope_df["est_start_differential"].idxmin(), "ephys_scope_diff"
    ]
    return ephys_offset / fs


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
        return -10

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


def update_sync_info(subject, exp, redo=False):
    si = load_sync_info()
    if si is None:
        si = {}
    if subject in si and exp in si[subject] and not redo:
        print(
            f"Something is already present for {subject}'s {exp}. Use redo=True to update."
        )
    if subject not in si:
        si[subject] = {}
    if exp not in si[subject]:
        si[subject][exp] = {}
    si[subject][exp]["sync_blocks"] = {}
    si[subject][exp]["acquisitions"] = {}
    sb, sp = get_all_sync_paths(subject, exp)
    coverage = get_ephys_coverage(subject, exp)
    scope, ephys = load_all_sync_blocks(subject, exp)
    for block in sb:
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
    acqs = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
    assignments = assign_acqs_to_sync_blocks(subject, exp, acqs, coverage)

    for acq_id in acqs:
        loc, acq = acq_id.split("--")
        dat_starts = load_datStartTimes(subject, exp, loc, acq)
        dat_start = pd.Timestamp(dat_starts[0])
        si[subject][exp]["acquisitions"][acq_id] = {}
        si[subject][exp]["acquisitions"][acq_id]["sync_block"] = assignments[acq_id]
        si[subject][exp]["acquisitions"][acq_id]["dat_start"] = str(dat_start)
        if si[subject][exp]["sync_blocks"][assignments[acq_id]]["corrupt"] is False:
            offset = compute_ephys_offset(
                subject,
                exp,
                loc,
                acq,
                si,
                scope[assignments[acq_id]],
                ephys[assignments[acq_id]],
            )
            offset = float(offset)
        else:
            offset = -10  # placeholder to find ephys offset for corrupt sync blocks
        si[subject][exp]["acquisitions"][acq_id]["ephys_offset"] = offset

    # check for corrupted sync blocks and find the ephys offset
    for acq_id in acqs:
        loc, acq = acq_id.split("--")
        if (
            si[subject][exp]["sync_blocks"][
                si[subject][exp]["acquisitions"][acq_id]["sync_block"]
            ]["corrupt"]
            is True
        ):
            ephys_offset = find_corrupted_sync_offsets(subject, exp, loc, acq, si)
            si[subject][exp]["acquisitions"][acq_id]["ephys_offset"] = float(
                ephys_offset
            )

    # write the updated sync info to the yaml file
    with open(f"{DEFS.anmat_root}/sync_info.yaml", "w") as f:
        yaml.dump(si, f)
        f.close()
    return
