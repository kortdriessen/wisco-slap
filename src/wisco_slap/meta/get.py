# ==================================================================================
# Main entry point for getting information from or associated with the meta files
# ==================================================================================

import os

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
    """Get the ephys offset for a given recording. This is the amount of time (in seconds)
    that the ephys data is shifted relative to the other data streams. This is necessary
    because the ephys data is recorded on a separate system and may not be perfectly
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
    """
    si = sync_info()
    acq_id = f"{loc}--{acq}"
    return si[subject][exp]["acquisitions"][acq_id]["ephys_offset"]


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


# ==================================================================================
# 4. acquisition master
# ==================================================================================
def acq_master():
    path = os.path.join(anmat_root, "acquisition_master.yaml")
    with open(path) as f:
        master_acq = yaml.safe_load(f)
    return master_acq


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
    esum_files = [f for f in os.listdir(mirror_dir) if f.endswith(".mat")]
    if len(esum_files) == 0:
        return "NO_ESUM_MIRROR"
    elif len(esum_files) == 1:
        return os.path.join(mirror_dir, esum_files[0])
    else:
        raise ValueError(
            f"Multiple esum files found in mirror for {subject} {exp} {loc} {acq}! Fix this manually ASAP!!"
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
