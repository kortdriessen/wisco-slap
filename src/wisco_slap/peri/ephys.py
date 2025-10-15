import os

import electro_py as epy
import xarray as xr

import wisco_slap as wis
import wisco_slap.defs as DEFS


def get_ephys_sync_block_path(subject, exp, sync_block):
    return f"{DEFS.data_root}/{subject}/{exp}/ephys/ephys-{sync_block}"


def get_block_paths(subject, exp):
    sb, sp = wis.peri.sync.get_all_sync_paths(subject, exp)
    ephys_dir = f"{DEFS.data_root}/{subject}/{exp}/ephys"
    bp = []
    for block in sb:
        bp.append(f"{ephys_dir}/ephys-{block}")
    bp.sort()
    return bp


def load_single_ephys_block(
    subject: str, exp: str, stores: list[str] = None, sync_block: int = 1
):
    """Load ephys data for a given experiment.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    """
    if stores is None:
        stores = ["EEGr", "EEG_", "loal", "Wav1"]
    ephys_dir = f"{DEFS.data_root}/{subject}/{exp}/ephys"
    ephys_files = [
        f
        for f in os.listdir(ephys_dir)
        if os.path.isdir(os.path.join(ephys_dir, f)) and f == f"ephys-{sync_block}"
    ]
    data = {}
    block_path = os.path.join(ephys_dir, ephys_files[0])
    for store in stores:
        store_data = epy.tdt.io.get_data(
            block_path, store=store, channel=DEFS.store_chans[store], dt=False
        )
        data[store] = store_data
    return data


def load_exp_ephys_data(subject: str, exp: str, stores: list[str] = None):
    """Load ephys data for a given experiment.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    """
    if stores is None:
        stores = ["EEGr", "loal", "Wav1"]
    ephys_dir = f"{DEFS.data_root}/{subject}/{exp}/ephys"
    ephys_files = [
        f for f in os.listdir(ephys_dir) if os.path.isdir(os.path.join(ephys_dir, f))
    ]
    data = {}
    for store in stores:
        store_data = []
        for block in ephys_files:
            block_path = os.path.join(ephys_dir, block)
            store_data.append(
                epy.tdt.io.get_data(
                    block_path, store=store, channel=DEFS.store_chans[store], dt=False
                )
            )
        data[store] = xr.concat(store_data, dim="time")
    return data
