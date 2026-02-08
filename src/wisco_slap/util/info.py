import glob
import os

import polars as pl
import yaml

import wisco_slap as wis
import wisco_slap.defs as DEFS


def get_unique_acquisitions_per_experiment(subject, exp):
    acqs = []
    ei = wis.pipes.exp_info.load_exp_info_spreadsheet()
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
            acqs.append(f"{loc}--{acq}")
    return acqs


def sub_esum_path(subject: str, exp: str, loc: str, acq: str) -> str | None:
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
    esum_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}/ExperimentSummary"
    if not os.path.exists(esum_dir):
        return None
    # Should only be one file inside this folder
    esum_files = [f for f in os.listdir(esum_dir) if f.endswith(".mat")]
    if len(esum_files) == 0:
        return None
    elif len(esum_files) == 1:
        return os.path.join(esum_dir, esum_files[0])
    else:
        print(f"Multiple esum files found for {subject} {exp} {loc} {acq}")
        return None


def load_exp_info_spreadsheet() -> pl.DataFrame:
    path = f"{DEFS.anmat_root}/exp_info.csv"
    return pl.read_csv(path)


def load_dmd_info():
    path = f"{DEFS.anmat_root}/dmd_info.yaml"
    with open(path) as f:
        dmd_info = yaml.safe_load(f)
    return dmd_info


def determine_processing_done(subject, exp, loc, acq):
    """Determine if the processing has been done for a given acquisition.

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
    """
    data_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}/ExperimentSummary"
    exp_summary_files = glob.glob(os.path.join(data_dir, "*Summary-*"))
    if len(exp_summary_files) > 0:
        return "YES"
    else:
        return "NO"
