"""Pure generator functions for basic glutamate event detection.

Match-filters LS (light-scattering) traces, estimates rolling noise, and
extracts contiguous supra-threshold periods as discrete events. All
existence checking, version tracking, and orchestration is handled by
glu_ev_basic_mon.py.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from math import ceil

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from scipy.signal import fftconvolve

import slap2_py as spy
import wisco_slap as wis
import wisco_slap.defs as DEFS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGLUSNFR_TAU_S: float = 0.03  # sensor decay constant (seconds)
SNR_THRESHOLD: float = 4.0  # noise_std multiplier for active threshold
DEFAULT_NOISE_WINDOW_S: float = 8.0  # rolling MAD window (seconds)
DEFAULT_MIN_VALID_FRAC: float = 0.5  # min valid samples in kernel window
MAD_TO_SIGMA: float = 1.4826  # scale factor: MAD -> sigma for Gaussian
OUTPUT_DIR_NAME: str = "event_detection"
PARQUET_NAME: str = "glu_events_basic.parquet"
FILTERED_ZARR_NAME: str = "filtered.zarr"
NOISE_STD_ZARR_NAME: str = "noise_std.zarr"
VERSION_PREFIX: str = "glu_ev_basic"


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------


def _build_kernel(tau_s: float, fs: float) -> np.ndarray:
    """Build an exponential decay kernel, normalized to unit energy.

    Parameters
    ----------
    tau_s : float
        Decay time constant in seconds.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        1-D kernel of shape ``(kernel_len,)``.
    """
    tau_samps = tau_s * fs
    kernel_len = max(2, int(ceil(5 * tau_samps)))
    kernel = np.exp(-np.arange(kernel_len) / tau_samps)
    kernel /= np.linalg.norm(kernel)
    return kernel


def _batch_match_filter(
    data_filled: np.ndarray,
    valid: np.ndarray,
    kernel: np.ndarray,
    min_valid_frac: float,
) -> np.ndarray:
    """Batch matched filter across all synapses simultaneously.

    Parameters
    ----------
    data_filled : np.ndarray, shape ``(n_syn, n_time)``
        Trace data with NaNs replaced by 0.
    valid : np.ndarray, shape ``(n_syn, n_time)``
        Binary mask of valid (non-NaN) samples (float64).
    kernel : np.ndarray, shape ``(kernel_len,)``
        1-D matched-filter kernel.
    min_valid_frac : float
        Minimum fraction of valid samples within the kernel window.
        Output is NaN where this threshold is not met.

    Returns
    -------
    np.ndarray, shape ``(n_syn, n_time)``
        Matched-filter output with NaN where validity is insufficient.
    """
    # Convolve all synapses at once along the time axis
    kernel_2d = kernel[::-1][np.newaxis, :]
    filtered = fftconvolve(data_filled, kernel_2d, mode="same", axes=-1)

    # Count valid samples contributing to each output sample
    ones_2d = np.ones((1, len(kernel)))
    valid_count = fftconvolve(valid, ones_2d, mode="same", axes=-1)

    # Mask where too few valid samples contributed
    filtered[valid_count < min_valid_frac * len(kernel)] = np.nan

    return filtered


def _rolling_mad_single(
    x: np.ndarray, window: int, min_periods: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling MAD noise estimate for a single synapse trace.

    Parameters
    ----------
    x : np.ndarray, shape ``(n_time,)``
        Matched-filter output for one synapse.
    window : int
        Rolling window size (should be odd).
    min_periods : int
        Minimum number of non-NaN observations required.

    Returns
    -------
    noise_std : np.ndarray, shape ``(n_time,)``
        MAD scaled to sigma.
    median : np.ndarray, shape ``(n_time,)``
        Rolling median.
    """
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=min_periods).median()
    mad = (
        (s - med)
        .abs()
        .rolling(window, center=True, min_periods=min_periods)
        .median()
        * MAD_TO_SIGMA
    )
    return mad.values, med.values


def _batch_rolling_mad(
    filtered: np.ndarray,
    noise_win: int,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parallel rolling MAD across all synapses using ThreadPoolExecutor.

    Parameters
    ----------
    filtered : np.ndarray, shape ``(n_syn, n_time)``
        Matched-filter output.
    noise_win : int
        Window size in samples (must be odd).
    max_workers : int or None
        Thread pool size. Defaults to ``min(n_syn, 32)``.

    Returns
    -------
    noise_std : np.ndarray, shape ``(n_syn, n_time)``
    median : np.ndarray, shape ``(n_syn, n_time)``
    """
    n_syn = filtered.shape[0]
    if max_workers is None:
        max_workers = min(n_syn, 32)

    min_periods = max(1, noise_win // 4)
    noise_std = np.empty_like(filtered)
    median = np.empty_like(filtered)

    def _process(i: int) -> tuple[int, np.ndarray, np.ndarray]:
        ns, md = _rolling_mad_single(filtered[i], noise_win, min_periods)
        return i, ns, md

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, ns, md in pool.map(_process, range(n_syn)):
            noise_std[i] = ns
            median[i] = md

    return noise_std, median


# ---------------------------------------------------------------------------
# Core match-filter + noise estimation
# ---------------------------------------------------------------------------


def match_filter_with_noise(
    ls: xr.DataArray,
    tau_s: float = IGLUSNFR_TAU_S,
    noise_window_s: float = DEFAULT_NOISE_WINDOW_S,
    min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
) -> xr.Dataset:
    """Match-filter LS traces and estimate per-sample noise.

    Vectorized implementation: the matched filter is computed for all
    synapses in a single batched FFT call, and the rolling MAD noise
    estimate is parallelized across synapses using threads.

    Parameters
    ----------
    ls : xr.DataArray
        LS traces with dims ``(syn_id, time)``. May contain NaNs
        (censored / motion-artifact samples).
    tau_s : float
        Transient decay time constant in seconds.
    noise_window_s : float
        Window for rolling MAD noise estimate, in seconds.
    min_valid_frac : float
        Minimum fraction of valid (non-NaN) samples within the kernel
        window. Output is NaN where this threshold is not met.

    Returns
    -------
    xr.Dataset
        Variables ``filtered`` and ``noise_std``, each with the same
        dims/coords as the input.
    """
    time = ls.coords["time"].values
    dt = float(np.median(np.diff(time)))
    fs = 1.0 / dt

    kernel = _build_kernel(tau_s, fs)

    # Ensure 2D (syn_id, time)
    squeezed = ls.ndim == 1
    if squeezed:
        data = ls.values[np.newaxis, :]
    else:
        data = ls.values

    # Prepare NaN-safe inputs
    nan_mask = np.isnan(data)
    data_filled = np.where(nan_mask, 0.0, data).astype(np.float64)
    valid = (~nan_mask).astype(np.float64)

    # Batch matched filter (single FFT call for all synapses)
    filtered = _batch_match_filter(data_filled, valid, kernel, min_valid_frac)

    # Rolling MAD noise estimate (threaded across synapses)
    noise_win = max(3, int(np.round(noise_window_s * fs)))
    if noise_win % 2 == 0:
        noise_win += 1
    noise_std, _ = _batch_rolling_mad(filtered, noise_win)

    if squeezed:
        filtered = filtered[0]
        noise_std = noise_std[0]

    coords = {dim: ls.coords[dim] for dim in ls.dims}
    return xr.Dataset(
        {
            "filtered": xr.DataArray(
                filtered.astype(np.float32), dims=ls.dims, coords=coords
            ),
            "noise_std": xr.DataArray(
                noise_std.astype(np.float32), dims=ls.dims, coords=coords
            ),
        }
    )


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------


def _extract_events_for_synapse(
    snr: np.ndarray,
    filtered: np.ndarray,
    time_coords: np.ndarray,
    fs: float,
    dmd: int,
    syn_id: int,
) -> list[dict]:
    """Extract contiguous supra-threshold events for one synapse.

    Parameters
    ----------
    snr : np.ndarray, shape ``(n_time,)``
        SNR trace (``filtered / (noise_std * snr_threshold)``).
    filtered : np.ndarray, shape ``(n_time,)``
        Raw matched-filter output for amplitude metrics.
    time_coords : np.ndarray, shape ``(n_time,)``
        Time in seconds.
    fs : float
        Sampling rate in Hz.
    dmd : int
        DMD number (1 or 2).
    syn_id : int
        Synapse index.

    Returns
    -------
    list of dict
        One dict per event with all DataFrame columns.
    """
    active = snr > 1
    if not np.any(active):
        return []

    diff = np.diff(active.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1  # inclusive

    events = []
    for s, e in zip(starts, ends):
        snr_slice = snr[s : e + 1]
        filt_slice = filtered[s : e + 1]
        peak_idx_local = int(np.nanargmax(snr_slice))

        events.append(
            {
                "dmd": dmd,
                "syn_id": syn_id,
                "start_sample": int(s),
                "end_sample": int(e),
                "n_samples": int(e - s + 1),
                "time": float(time_coords[s]),
                "peak_time": float(time_coords[s + peak_idx_local]),
                "event_duration": float((e - s + 1) / fs),
                "peak_snr": float(np.nanmax(snr_slice)),
                "average_snr": float(np.nanmean(snr_slice)),
                "peak_filtered_value": float(filt_slice[peak_idx_local]),
                "integral": float(np.nansum(filt_slice)),
            }
        )
    return events


def _build_events_dataframe(all_events: list[dict]) -> pl.DataFrame:
    """Convert event dicts to a polars DataFrame.

    Returns a DataFrame with the correct schema even when *all_events*
    is empty.
    """
    schema = {
        "dmd": pl.Int32,
        "syn_id": pl.Int32,
        "start_sample": pl.Int64,
        "end_sample": pl.Int64,
        "n_samples": pl.Int32,
        "time": pl.Float64,
        "peak_time": pl.Float64,
        "event_duration": pl.Float64,
        "peak_snr": pl.Float64,
        "average_snr": pl.Float64,
        "peak_filtered_value": pl.Float64,
        "integral": pl.Float64,
    }
    if not all_events:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(all_events, schema=schema)


# ---------------------------------------------------------------------------
# Version tracking
# ---------------------------------------------------------------------------


def _write_version_file(out_dir: str, esum_p: str) -> None:
    """Write a version-tracking txt file in the event_detection directory.

    Removes any existing version file for this output before writing.
    """
    esum_version = os.path.basename(esum_p).split(".mat")[0]
    new_file = os.path.join(out_dir, f"{VERSION_PREFIX}__{esum_version}.txt")

    # Remove old version files
    for f in os.listdir(out_dir):
        if f.startswith(VERSION_PREFIX) and f.endswith(".txt"):
            os.remove(os.path.join(out_dir, f))

    with open(new_file, "w") as fh:
        fh.write(esum_p)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def detect_and_save(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_p: str,
    channel: int = 0,
    tau_s: float = IGLUSNFR_TAU_S,
    noise_window_s: float = DEFAULT_NOISE_WINDOW_S,
    min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
    snr_threshold: float = SNR_THRESHOLD,
) -> None:
    """Run basic glutamate event detection and save all outputs.

    Loads LS traces from scopex zarrs, match-filters them, estimates
    rolling noise, thresholds SNR to find active samples, groups
    contiguous active periods into events, and saves:

    - ``glu_events_basic.parquet`` — polars DataFrame of all events
    - ``filtered.zarr`` — matched-filter output
    - ``noise_std.zarr`` — rolling MAD noise estimate

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    esum_p : str
        Path to the ExperimentSummary ``.mat`` file (for version tracking).
    channel : int
        Which channel to process (default 0 = iGluSnFR4f).
    tau_s : float
        Sensor decay time constant in seconds.
    noise_window_s : float
        Rolling MAD window in seconds.
    min_valid_frac : float
        Minimum valid fraction for matched filter output.
    snr_threshold : float
        Noise multiplier for SNR threshold. Samples where
        ``filtered / (noise_std * snr_threshold) > 1`` are active.
    """
    tag = f"{subject} {exp} {loc} {acq}"

    # Load LS traces
    print(f"[{tag}] Loading LS traces...")
    ls_dict = wis.get.syn_dF(
        subject, exp, loc, acq, trace="ls", channels=channel
    )

    # Set up output directory
    acq_id = f"{loc}--{acq}"
    scopex_dir = os.path.join(DEFS.anmat_root, subject, exp, "scopex", acq_id)
    out_dir = os.path.join(scopex_dir, OUTPUT_DIR_NAME)
    wis.util.check_dir(out_dir)

    all_events: list[dict] = []
    filtered_xr_dict: dict[str, xr.DataArray] = {}
    noise_xr_dict: dict[str, xr.DataArray] = {}

    for dmd_key, ls_da in ls_dict.items():
        dmd_num = int(dmd_key.split("_")[1])
        print(f"[{tag}] Processing {dmd_key} ({ls_da.sizes['syn_id']} synapses)...")

        # Match filter + noise estimation
        mfn = match_filter_with_noise(
            ls_da, tau_s=tau_s, noise_window_s=noise_window_s,
            min_valid_frac=min_valid_frac,
        )

        filtered_vals = mfn["filtered"].values
        noise_vals = mfn["noise_std"].values

        # Compute SNR (guard against zero noise)
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_noise = np.maximum(noise_vals * snr_threshold, 1e-12)
            snr = filtered_vals / safe_noise

        # Extract events per synapse
        time_coords = ls_da.coords["time"].values
        fs = 1.0 / float(np.median(np.diff(time_coords)))
        syn_ids = ls_da.coords["syn_id"].values

        for i, sid in enumerate(syn_ids):
            events = _extract_events_for_synapse(
                snr[i], filtered_vals[i], time_coords, fs, dmd_num, int(sid)
            )
            all_events.extend(events)

        # Prepare zarr DataArrays: add channel dim back for save_xr_to_zarr
        filtered_xr_dict[dmd_key] = mfn["filtered"].expand_dims("channel", axis=0)
        noise_xr_dict[dmd_key] = mfn["noise_std"].expand_dims("channel", axis=0)

    # Save events parquet
    events_df = _build_events_dataframe(all_events)
    events_df.write_parquet(os.path.join(out_dir, PARQUET_NAME))
    print(f"[{tag}] Saved {len(events_df)} events to {PARQUET_NAME}")

    # Save zarrs
    for name, xr_dict in [
        (FILTERED_ZARR_NAME, filtered_xr_dict),
        (NOISE_STD_ZARR_NAME, noise_xr_dict),
    ]:
        zarr_path = os.path.join(out_dir, name)
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        spy.core.xarr_summ.save_xr_to_zarr(xr_dict, zarr_path)

    # Write version tracking
    _write_version_file(out_dir, esum_p)
    print(f"[{tag}] Event detection complete.")
