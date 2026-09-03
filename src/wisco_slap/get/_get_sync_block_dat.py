"""Loading functions for sync_block_data outputs."""

import os

import h5py
import numpy as np
import polars as pl
import xarray as xr

from wisco_slap.defs import anmat_root

# HDF5 attributes that are netCDF-internal bookkeeping (dimension-scale wiring,
# fill values). xarray hides these; we drop them so a hand-read DataArray matches
# what ``xr.open_dataarray`` would have returned.
_NC_INTERNAL_ATTRS = frozenset({"CLASS", "NAME", "DIMENSION_LIST", "REFERENCE_LIST", "_FillValue"})


def _sb_dir(subject: str, exp: str, sb: int) -> str:
    """Return the sync_block directory path."""
    return os.path.join(anmat_root, subject, exp, "sync_block_data", f"sync_block-{sb}")


def _clean_attr(v: object) -> object:
    """Normalise an h5py attribute value to the plain-Python form xarray yields."""
    if isinstance(v, np.ndarray):
        v = v.item() if v.size == 1 else v.tolist()
    if isinstance(v, bytes):
        v = v.decode()
    return v


def _load_ephys_dataarray(path: str, store: str) -> xr.DataArray:
    """Read a single-store ephys ``.nc`` file into an in-memory DataArray.

    Historical ephys ``.nc`` files exist in two on-disk formats, and we pick the
    reader by sniffing the file's magic bytes:

    * **HDF5-based netCDF-4** (magic ``\\x89HDF``): read with ``h5py`` and an
      explicit ``locking=False``. We deliberately bypass ``xr.open_dataarray``'s
      netCDF4 engine for these. Because the ephys files live on the ``/data``
      RAID and this process loads both ``h5py`` and ``netCDF4`` (which ship
      conflicting HDF5 file-locking behaviour), the netCDF4 reader intermittently
      — and, once the filesystem is in the wrong state, deterministically —
      raises ``OSError: [Errno -101] NetCDF: HDF error`` when it tries to flock
      the file. ``HDF5_USE_FILE_LOCKING=FALSE`` only helps if set before HDF5 is
      imported, which we can't guarantee from inside a loader.
      ``h5py.File(..., locking=False)`` disables the lock per-open and is robust.
    * **netCDF-3 "classic"** (magic ``CDF``): ``h5py`` cannot read these — it
      rejects the non-HDF5 signature with
      ``OSError: ... file signature not found``. These files were written when
      xarray's ``to_netcdf`` fell back to the ``scipy`` engine (netCDF4 not yet
      installed); the data is complete and valid, only the container format
      differs. We read them with the pure-Python ``scipy`` engine, which touches
      no HDF5 library and so has no locking to conflict with.

    Stores are only tens of MB and always 1-D ``(time,)`` timeseries, so we read
    fully into memory. Either path returns a DataArray equivalent to the old
    ``xr.open_dataarray`` result (name, dims, dtype, values, ``time`` coord, attrs).
    """
    with open(path, "rb") as fh:
        magic = fh.read(8)

    if magic.startswith(b"CDF"):
        # netCDF-3 classic: read via scipy, fully materialise, and return an
        # independent copy so nothing references the store once it is closed.
        with xr.open_dataarray(path, engine="scipy") as da:
            return da.load().rename(store).copy(deep=True)

    # Default: HDF5-based netCDF-4 (the common, current-writer case).
    with h5py.File(path, "r", locking=False) as f:
        time = f["time"][:]
        var = f[store]
        data = var[:]
        attrs = {
            k: _clean_attr(v)
            for k, v in var.attrs.items()
            if k not in _NC_INTERNAL_ATTRS and not k.startswith("_Netcdf4")
        }
    return xr.DataArray(
        data, dims=("time",), coords={"time": time}, name=store, attrs=attrs
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
        ep_data[store] = _load_ephys_dataarray(path, store)
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
    path = os.path.join(_sb_dir(subject, exp, sb), "whisking", "whisk_df.parquet")
    return pl.read_parquet(path)


def _whisk_df_to_xarray(whisk_df: pl.DataFrame) -> xr.DataArray:
    """Convert whisking polars DataFrame to an xarray DataArray with a time dim."""
    return xr.DataArray(
        data=whisk_df["whis"].to_numpy(),
        dims=["time"],
        coords={"time": whisk_df["time"].to_numpy()},
        name="whis",
    )


def whisk_xa(subject: str, exp: str, sb: int) -> xr.DataArray:
    """Load whisking data as an xarray DataArray.

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
        DataArray with dims (time,) containing whisking values.
    """
    df = whisk_df(subject, exp, sb)
    return _whisk_df_to_xarray(df)
