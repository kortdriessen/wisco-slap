"""Loading functions for sync_block_data outputs."""

import os

import numpy as np
import polars as pl
import xarray as xr

from wisco_slap.defs import anmat_root


def _sb_dir(subject: str, exp: str, sb: int) -> str:
    """Return the sync_block directory path."""
    return os.path.join(
        anmat_root, subject, exp, "sync_block_data", f"sync_block-{sb}"
    )


def ephys(
    subject: str,
    exp: str,
    sb: int,
    stores: tuple[str, ...] = ("EEG_", "EEGr", "loal"),
) -> dict[str, xr.DataArray]:
    """Load ephys data for a given subject, experiment, and sync block.

    Parameters
    ----------
    subject : str
        Subject name.
    exp : str
        Experiment name.
    sb : int
        Sync block number.
    stores : tuple of str
        Which ephys stores to load.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary of ephys data arrays keyed by store name.
    """
    ep_data = {}
    for store in stores:
        path = os.path.join(_sb_dir(subject, exp, sb), "ephys", f"{store}.nc")
        ep_data[store] = xr.open_dataarray(path)
    return ep_data


def frame_times(subject: str, exp: str, sb: int) -> np.ndarray:
    """Load pupil video frame times for a sync block.

    Parameters
    ----------
    subject : str
        Subject name.
    exp : str
        Experiment name.
    sb : int
        Sync block number.

    Returns
    -------
    np.ndarray
        1D array of frame timestamps in seconds.
    """
    path = os.path.join(_sb_dir(subject, exp, sb), "pupil", "frame_times.npy")
    return np.load(path)


def eye_metrics(subject: str, exp: str, sb: int) -> pl.DataFrame:
    """Load eye metrics dataframe for a sync block.

    Parameters
    ----------
    subject : str
        Subject name.
    exp : str
        Experiment name.
    sb : int
        Sync block number.

    Returns
    -------
    pl.DataFrame
        Eye metrics with columns: frame, diameter, motion, lid, lid_norm,
        pup_likelihood, lid_likelihood, time.
    """
    path = os.path.join(
        _sb_dir(subject, exp, sb), "pupil", "eye_metrics", "eye_metrics.parquet"
    )
    return pl.read_parquet(path)


def eye_metrics_xa(subject: str, exp: str, sb: int) -> xr.DataArray:
    """Load eye metrics as an xarray DataArray.

    Parameters
    ----------
    subject : str
        Subject name.
    exp : str
        Experiment name.
    sb : int
        Sync block number.

    Returns
    -------
    xr.DataArray
        DataArray with dims (time, metric) where metric
        coords are ['diameter', 'motion', 'lid_norm'].
    """
    metrics = ["diameter", "motion", "lid_norm"]
    df = eye_metrics(subject, exp, sb)
    return xr.DataArray(
        data=df.select(metrics).to_numpy(),
        dims=["time", "metric"],
        coords={
            "time": df["time"].to_numpy(),
            "metric": metrics,
        },
    )


def whisk_df(subject: str, exp: str, sb: int) -> pl.DataFrame:
    """Load whisking dataframe for a sync block.

    Parameters
    ----------
    subject : str
        Subject name.
    exp : str
        Experiment name.
    sb : int
        Sync block number.

    Returns
    -------
    pl.DataFrame
        Whisking data with columns: whis, time.
    """
    path = os.path.join(
        _sb_dir(subject, exp, sb), "whisking", "whisk_df.parquet"
    )
    return pl.read_parquet(path)
