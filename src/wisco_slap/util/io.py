import os

import wisco_slap.defs as DEFS


def sub_esum_path(subject: str, exp: str, loc: str, acq: str) -> str:
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
    # Should only be one file inside this folder
    esum_files = [f for f in os.listdir(esum_dir) if f.endswith(".mat")]
    return os.path.join(esum_dir, esum_files[0]) if esum_files else "N/A"
