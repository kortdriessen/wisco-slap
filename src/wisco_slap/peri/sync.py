import os

import numpy as np
import pandas as pd
import slap2_py as spy

import wisco_slap as wis
import wisco_slap.defs as DEFS


def load_datStartTimes(subject, exp, loc, acq):
    print("test")
    path = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}/datStartTimes1.txt"
    with open(path) as f:
        datStartTimes = [pd.Timestamp(line.strip()) for line in f]
    return datStartTimes


def load_sync_block(
    subject: str, exp: str, sync_block: int
) -> tuple[np.ndarray, np.ndarray]:
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
    sync_path = get_all_sync_paths(subject, exp)[sync_block - 1]
    full_sync = spy.utils.drec.load_datarec_file(sync_path)
    return (full_sync["slap2_acquiring_trigger"][:], full_sync["electrophysiology"][:])


def get_all_sync_paths(subject, exp) -> list[str]:
    """Get all sync paths for a given subject and experiment.

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.

    Returns
    -------
    list[str]
        A list of sync paths.
    """
    exp_dir = f"{DEFS.data_root}/{subject}/{exp}"
    sync_files = [f for f in os.listdir(exp_dir) if "SYNC_" in f]
    sync_paths = [os.path.join(exp_dir, f) for f in sync_files]
    sync_paths.sort()
    return sync_paths


def save_exp_sync_dataframe(subject, exp, overwrite=False) -> None:
    """Save a sync dataframe for all sync files in an experiment.

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    """
    exp_dir = f"{DEFS.data_root}/{subject}/{exp}"
    # find all files containing 'SYNC_' in the name
    sync_files = [f for f in os.listdir(exp_dir) if "SYNC_" in f]
    for sync_file in sync_files:
        # load the sync file
        sync_number = sync_file.split("SYNC_")[-1].split(".")[0]
        print(f"Processing sync file {sync_number} for experiment {exp}")
        sync_path = os.path.join(exp_dir, sync_file)
        scope, ephys = spy.utils.drec.load_sync_file(sync_path)
        sdf = spy.utils.drec.generate_scope_index_df(scope)
        # save the sync data file
        deriv_dir = f"{DEFS.derivs_root}/{subject}/{exp}"
        wis.utils.gen.check_dir(deriv_dir)
        save_path = os.path.join(deriv_dir, f"sync_{sync_number}_df.csv")
        if not os.path.exists(save_path) or overwrite:
            sdf.to_csv(save_path, index=False)
            print(f"Saved sync dataframe to {save_path}")
        elif os.path.exists(save_path) and not overwrite:
            print(f"File {save_path} already exists. Use overwrite=True to overwrite.")
            continue
    return


def create_sync_manifest(subject, exp) -> None:
    """Create a manifest file listing the experiments and acquisitions that need synchronization information.

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    """
    deriv_dir = f"{DEFS.derivs_root}/{subject}/{exp}"
    # find all files containing 'sync_' in the name
    sync_files = [
        f for f in os.listdir(deriv_dir) if "sync_" in f and f.endswith("_df.csv")
    ]
    manifest_path = os.path.join(deriv_dir, "sync_manifest.txt")
    with open(manifest_path, "w") as f:
        for sync_file in sync_files:
            f.write(f"{sync_file}\n")
    print(f"Saved sync manifest to {manifest_path}")
    return


def detect_ephys_start_sample(ephys, fs=5000):
    thresh = np.max(ephys[: int(fs * 2)]) * 3
    samples = np.where(ephys > thresh)[0]
    return samples[0]
