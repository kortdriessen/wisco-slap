import os

import electro_py as epy
import numpy as np
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
                    block_path, store=store, channel=store_chans[store], dt=False
                )
            )
        data[store] = xr.concat(store_data, dim="time")
    return data


def generate_ephys_scoring_data(
    subject: str,
    exp: str,
    stores: list[str] = None,
    sync_block: int = 1,
    overwrite=False,
):
    """Generate and save ephys data for sleep scoring.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    stores : list[str], optional
        stores to generate data for, by default None
    sync_block : int, optional
        sync block number, by default 1
    overwrite : bool, optional
        whether to overwrite existing files, by default False
    """
    if stores is None:
        stores = ["EEG_", "loal"]
    e = load_single_ephys_block(subject, exp, stores=stores, sync_block=sync_block)
    save_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}"
    if os.path.exists(save_dir) and not overwrite:
        print(f"{save_dir} directory already exists. Use overwrite=True to overwrite.")
        return
    wis.util.gen.check_dir(save_dir)
    for store in stores:
        data = e[store]
        if "channel" in data.dims:
            for channel in data.channel.values:
                data_channel = data.sel(channel=channel)
                t = data_channel["time"].values
                d = data_channel.values
                assert len(t) == len(d)
                t_path = f"{save_dir}/{store}--ch{channel}_t.npy"
                d_path = f"{save_dir}/{store}--ch{channel}_y.npy"
                np.save(t_path, t)
                np.save(d_path, d)
        else:
            t = data["time"].values
            d = data.values
            assert len(t) == len(d)
            t_path = f"{save_dir}/{store}--ch0_t.npy"
            d_path = f"{save_dir}/{store}--ch0_y.npy"
            np.save(t_path, t)
            np.save(d_path, d)
    return
