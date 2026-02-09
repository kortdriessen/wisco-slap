import glob
import os

import h5py
import numpy as np
import polars as pl
import yaml

import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.util.info import load_exp_info_spreadsheet


def estimate_start_time(subject, exp, loc, acq):
    acq_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}"
    meta_name = None
    for f in os.listdir(acq_dir):
        if "_DMD1.meta" in f:
            meta_name = f
        else:
            continue
    if meta_name is None:
        raise ValueError(f"No meta file found for {subject} {exp} {loc} {acq}")
    day, tm = meta_name.split(".meta")[0].split("_DMD")[0].split("_")[-2:]
    return f"{day}_{tm}"


def estimate_acq_duration(subject, exp, loc, acq):
    """Estimate the duration of an acquisition.

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    loc : str
        The location ID.
    acq : str
        The acquisition ID.

    Returns
    -------
    float
        The duration of the acquisition in seconds.

    Raises
    ------
    ValueError
        No meta file found for the given acquisition.
    """
    acq_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}"
    meta_file = glob.glob(os.path.join(acq_dir, "*_DMD1.meta"))
    if len(meta_file) == 0:
        raise ValueError(f"No meta file found for {subject} {exp} {loc} {acq}")
    meta = h5py.File(meta_file[0], "r")

    # get the list of cycle numbers
    cycle_list = []
    for f in os.listdir(acq_dir):
        if "_DMD1" in f and "DOWNSAMPLED" not in f and f.endswith(".dat"):
            cyc = f.split("CYCLE-")[-1].split(".")[0]
            cycle_list.append(int(cyc))
    cycle_list.sort()
    cycles_per_file = np.median(np.diff(cycle_list))

    lines_per_cycle = meta["AcquisitionContainer"]["ParsePlan"]["linesPerCycle"][0]
    line_rate = meta["AcquisitionContainer"]["ParsePlan"]["lineRateHz"][0]
    time_per_line = 1 / line_rate
    time_per_cycle = time_per_line * lines_per_cycle
    time_per_file = time_per_cycle * cycles_per_file
    n_files = len(cycle_list)
    total_time = time_per_file * n_files
    return int(total_time[0] - time_per_file)


def get_exsum_date(subject, exp, loc, acq):
    """Get the date of the exsum file for a given acquisition."""
    data_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}/ExperimentSummary"
    exp_summary_files = glob.glob(os.path.join(data_dir, "*Summary-*"))
    if len(exp_summary_files) > 0:
        return (
            exp_summary_files[0].split("Summary-")[-1].split(".")[0].replace("-", "_")
        )
    else:
        return None


def get_refstack_used(subject, exp, loc, acq):
    esum_path = wis.util.info.get_esum_mirror_path(subject, exp, loc, acq)
    if esum_path is None:
        return None
    with h5py.File(esum_path, "r") as f:
        refstack_used = f["exptSummary"]["trialTable"]["alignParams"][
            "refStackTemplate"
        ][0][0]
        return "True" if refstack_used > 0 else "False"


def get_analyze_hz(subject, exp, loc, acq):
    esum_path = wis.util.info.get_esum_mirror_path(subject, exp, loc, acq)
    if esum_path is None:
        return None
    with h5py.File(esum_path, "r") as f:
        analyze_hz = f["exptSummary"]["params"]["analyzeHz"][0][0]
        return int(analyze_hz)


def get_draw_user_rois(subject, exp, loc, acq):
    esum_path = wis.util.info.get_esum_mirror_path(subject, exp, loc, acq)
    if esum_path is None:
        return None
    with h5py.File(esum_path, "r") as f:
        draw_user_rois = f["exptSummary"]["params"]["drawUserRois"][0][0]
        return "True" if draw_user_rois > 0 else "False"


def update_estimated_start_time(ei, subject, exp, loc, acq):
    ei = ei.with_columns(
        pl
        .when(
            (pl.col("subject") == subject)
            & (pl.col("experiment") == exp)
            & (pl.col("location") == loc)
            & (pl.col("acquisition") == acq)
        )
        .then(pl.lit(estimate_start_time(subject, exp, loc, acq)))
        .otherwise(pl.col("estimated_start_time"))
        .alias("estimated_start_time")
    )
    return ei


def update_estimated_duration(ei, subject, exp, loc, acq):
    """Update the estimated duration for a given acquisition."""
    ei = ei.with_columns(
        pl
        .when(
            (pl.col("subject") == subject)
            & (pl.col("experiment") == exp)
            & (pl.col("location") == loc)
            & (pl.col("acquisition") == acq)
        )
        .then(pl.lit(estimate_acq_duration(subject, exp, loc, acq)))
        .otherwise(pl.col("estimated_duration"))
        .alias("estimated_duration")
    )
    return ei


def update_processing_done(ei, subject, exp, loc, acq):
    pdone = wis.util.info.determine_processing_done(subject, exp, loc, acq)
    ei = ei.with_columns(
        pl
        .when(
            (pl.col("subject") == subject)
            & (pl.col("experiment") == exp)
            & (pl.col("location") == loc)
            & (pl.col("acquisition") == acq)
        )
        .then(pl.lit(pdone))
        .otherwise(pl.col("processed"))
        .alias("processed")
    )
    return ei, pdone


def update_exsum_date(ei, subject, exp, loc, acq):
    exsum_date = get_exsum_date(subject, exp, loc, acq)
    ei = ei.with_columns(
        pl
        .when(
            (pl.col("subject") == subject)
            & (pl.col("experiment") == exp)
            & (pl.col("location") == loc)
            & (pl.col("acquisition") == acq)
        )
        .then(pl.lit(exsum_date))
        .otherwise(pl.col("exsum_date"))
        .alias("exsum_date")
    )
    return ei


def update_refstack_used(ei, subject, exp, loc, acq):
    refstack_used = get_refstack_used(subject, exp, loc, acq)
    ei = ei.with_columns(
        pl
        .when(
            (pl.col("subject") == subject)
            & (pl.col("experiment") == exp)
            & (pl.col("location") == loc)
            & (pl.col("acquisition") == acq)
        )
        .then(pl.lit(refstack_used))
        .otherwise(pl.col("refstack_used_mrr"))
        .alias("refstack_used_mrr")
    )
    return ei


def update_analyze_hz(ei, subject, exp, loc, acq):
    analyze_hz = get_analyze_hz(subject, exp, loc, acq)
    ei = ei.with_columns(
        pl
        .when(
            (pl.col("subject") == subject)
            & (pl.col("experiment") == exp)
            & (pl.col("location") == loc)
            & (pl.col("acquisition") == acq)
        )
        .then(pl.lit(analyze_hz))
        .otherwise(pl.col("analyze_hz"))
        .alias("analyze_hz")
    )
    return ei


def update_draw_user_rois(ei, subject, exp, loc, acq):
    draw_user_rois = get_draw_user_rois(subject, exp, loc, acq)
    ei = ei.with_columns(
        pl
        .when(
            (pl.col("subject") == subject)
            & (pl.col("experiment") == exp)
            & (pl.col("location") == loc)
            & (pl.col("acquisition") == acq)
        )
        .then(pl.lit(draw_user_rois))
        .otherwise(pl.col("user_rois"))
        .alias("user_rois")
    )
    return ei


def update_exp_info_spreadsheet():
    """Update the exp info spreadsheet with the latest information."""
    ei = load_exp_info_spreadsheet()
    for subject in ei["subject"].unique():
        for exp in ei.filter(pl.col("subject") == subject)["experiment"].unique():
            for loc in (
                ei
                .filter(pl.col("subject") == subject)
                .filter(pl.col("experiment") == exp)["location"]
                .unique()
            ):
                for acq in (
                    ei
                    .filter(pl.col("subject") == subject)
                    .filter(pl.col("experiment") == exp)
                    .filter(pl.col("location") == loc)["acquisition"]
                    .unique()
                ):
                    print(f"Updating {subject} {exp} {loc} {acq}")
                    ei = update_estimated_start_time(ei, subject, exp, loc, acq)
                    ei = update_estimated_duration(ei, subject, exp, loc, acq)
                    ei, pdone = update_processing_done(ei, subject, exp, loc, acq)
                    if pdone == "NO":
                        continue
                    ei = update_exsum_date(ei, subject, exp, loc, acq)
                    ei = update_refstack_used(ei, subject, exp, loc, acq)
                    ei = update_analyze_hz(ei, subject, exp, loc, acq)
                    ei = update_draw_user_rois(ei, subject, exp, loc, acq)
    ei.write_csv(f"{DEFS.anmat_root}/exp_info.csv")
    return


def update_all_subject_sync_info(redo=False):
    ei = load_exp_info_spreadsheet()
    for subject in ei["subject"].unique():
        for exp in ei.filter(pl.col("subject") == subject)["experiment"].unique():
            wis.peri.sync.update_sync_info(subject, exp, redo=redo)
    return


def _update_dmd_info(subject, exp, loc, acq):
    dmd_info_path = f"{DEFS.anmat_root}/dmd_info.yaml"
    esum_path = wis.util.info.get_esum_mirror_path(subject, exp, loc, acq)
    if esum_path is None:
        print(f"{subject} {exp} {loc} {acq} has no esum path")
        return None
    with open(dmd_info_path) as f:
        dmd_info = yaml.safe_load(f)
    if subject not in dmd_info:
        dmd_info[subject] = {}
    if exp not in dmd_info[subject]:
        dmd_info[subject][exp] = {}
    if loc not in dmd_info[subject][exp]:
        dmd_info[subject][exp][loc] = {}
    if acq not in dmd_info[subject][exp][loc]:
        dmd_info[subject][exp][loc][acq] = {}
    if "dmd-1" not in dmd_info[subject][exp][loc][acq]:
        dmd_info[subject][exp][loc][acq]["dmd-1"] = {}
        dmd_info[subject][exp][loc][acq]["dmd-1"]["depth"] = -1
    if "dmd-2" not in dmd_info[subject][exp][loc][acq]:
        dmd_info[subject][exp][loc][acq]["dmd-2"] = {}
        dmd_info[subject][exp][loc][acq]["dmd-2"]["depth"] = -1
    for dmd in [1, 2]:
        roi = wis.pipes.annotation_materials._get_roi_container(esum_path, dmd)
        if roi[0] == 0:
            dmd_info[subject][exp][loc][acq][f"dmd-{dmd}"]["somas"] = []
        else:
            roi_names = []
            for r in roi:
                roi_names.append(r["Label"])
            dmd_info[subject][exp][loc][acq][f"dmd-{dmd}"]["somas"] = roi_names
    with open(dmd_info_path, "w") as f:
        yaml.dump(dmd_info, f)


def update_dmd_info():
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            acqs = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
            for a in acqs:
                loc, acq = a.split("--")
                _update_dmd_info(subject, exp, loc, acq)
    return
