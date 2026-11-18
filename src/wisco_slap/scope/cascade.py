"""CASCADE calcium spike-inference on soma ROI dF/F (scopex arrays).

This module runs the supervised, parameter-free CASCADE model
(Rupprecht et al. 2021 / CascadeTorch) on a soma's dF/F trace to obtain a
continuous spike-probability estimate and a set of discrete spike times.
It is the standalone, importable home for the pipeline that previously
lived in ``slap_mi_2_sleep/GLU_EVENT_DETECTION/cascade_crosscheck.py`` and
``cascade_full_pipe.ipynb``.

The two headline functions are:

* :func:`run_cascade_detection` -- takes a single-soma dF/F scopex
  ``xr.DataArray`` (e.g. ``wis.get.roi_F(..., return_dFF=True)[dmd].sel(soma_id=...)``)
  and returns ``(spike_prob, cascade_spikes)`` -- the continuous CASCADE
  prediction as an ``xr.DataArray`` (with the soma-ID tied in) and the
  discrete spikes as a ``pl.DataFrame``.
* :func:`filter_cascade_spikes` -- refractory / non-maximum-suppression
  thinning of a discrete-spike ``pl.DataFrame`` (kept separate so callers
  choose whether/how to enforce a minimum separation).

CASCADE wants dF/F as a *fraction* (not percent), which is exactly what
``roi_F(..., return_dFF=True)`` produces. 200 Hz SLAP2 traces have no
matching CASCADE model, so the input is anti-alias resampled to 30 Hz
(:func:`_prepare_for_cascade`) before inference.

NOTE (dependency handling, 2026-07): ``cascade2p`` (the CascadeTorch import
package) and ``torch`` are imported *lazily inside the functions*, and are
deliberately NOT declared as dependencies of ``wisco-slap``. ``cascade2p`` is
only present via the editable ``cascadetorch`` install in the
``slap_mi_2_sleep`` venv (our canonical run environment); importing it lazily
means ``import wisco_slap`` does not require ``cascade2p`` to be installed and
does not load CASCADE's model machinery (or ``ruamel.yaml``) until inference
actually runs. (``torch`` is already pulled in transitively by other wisco-slap
deps, so this is about ``cascade2p``, not about avoiding torch.) If this ever
causes trouble (e.g. running from a venv where CASCADE is absent), the fix is
to formally add it: ``uv add`` cascadetorch as an editable path source in
``wisco-slap`` (mirroring ``slap_mi_2_sleep``'s ``[tool.uv.sources]`` entry)
then ``uv sync`` in ``slap_mi_2_sleep``.
"""

from __future__ import annotations

import bisect
import os
from collections.abc import Callable, Sequence
from math import gcd

import numpy as np
import polars as pl
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly

import wisco_slap.defs as DEFS

__all__ = [
    "run_cascade_detection",
    "filter_cascade_spikes",
    "event_triggered_rate",
    "rate_correlation",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CASCADE model bundled in the CascadeTorch Pretrained_models folder
DEFAULT_MODEL_NAME: str = "Global_EXC_30Hz_smoothing25ms"
DEFAULT_FS_TARGET: float = 30.0  # Hz; matches the 30 Hz CASCADE model
DEFAULT_LONG_GAP_S: float = 0.25  # NaN gaps longer than this are flagged in longmask


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_for_cascade(
    t: np.ndarray,
    dff: np.ndarray,
    fs_target: float = DEFAULT_FS_TARGET,
    long_gap_s: float = DEFAULT_LONG_GAP_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Resample a dF/F trace to ``fs_target`` and prepare it for CASCADE.

    Ported from ``cascade_crosscheck.prepare_for_cascade``. NaNs are linearly
    interpolated for the model input (CASCADE cannot ingest NaNs), but any NaN
    gap longer than ``long_gap_s`` is recorded in ``longmask_ds`` so downstream
    comparisons can exclude those spans. The resample is an exact rational
    (anti-aliased) conversion, e.g. 200 -> 30 Hz via ``200 * 3 / 20``.

    Parameters
    ----------
    t : np.ndarray
        Time axis (seconds), 1-D, monotonically increasing.
    dff : np.ndarray
        dF/F trace (fraction), same length as ``t``. May contain NaNs.
    fs_target : float
        Target sampling rate in Hz (default 30, to match the CASCADE model).
    long_gap_s : float
        NaN gaps strictly longer than this (seconds) are marked in the returned
        long-gap mask.

    Returns
    -------
    dff_ds : np.ndarray
        Resampled, NaN-filled dF/F near ``fs_target`` (the exact input clock
        is preserved rather than rounded).
    t_ds : np.ndarray
        Resampled time axis (seconds), anchored at ``t[0]`` with the effective
        sampling rate ``fs_orig * round(fs_target) / round(fs_orig)``.
    longmask_ds : np.ndarray of bool
        True where the resampled sample falls inside a long (> ``long_gap_s``)
        original NaN gap.
    fs_orig : float
        The estimated original sampling rate (Hz).
    """
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all():
        raise ValueError("CASCADE needs at least two finite time samples.")
    delta = np.diff(t)
    dt = float(np.median(delta))
    if dt <= 0 or not np.allclose(delta, dt, rtol=1e-6, atol=1e-9):
        raise ValueError("Resample each regular, contiguous timestamp run separately.")
    fs_orig = 1.0 / dt
    y = np.asarray(dff, float).copy()
    if y.shape != t.shape:
        raise ValueError("The dF/F and time arrays must have the same 1-D shape.")
    nan = ~np.isfinite(y)
    idx = np.arange(len(y))
    if (~nan).any():
        y[nan] = np.interp(idx[nan], idx[~nan], y[~nan])
    else:
        y[:] = np.nan

    longmask = np.zeros(len(y), bool)  # mark long gaps to exclude later
    i = 0
    while i < len(nan):
        if nan[i]:
            j = i
            while j < len(nan) and nan[j]:
                j += 1
            if (j - i) / fs_orig > long_gap_s:
                longmask[i:j] = True
            i = j
        else:
            i += 1

    if not np.isfinite(fs_target) or round(fs_target) < 1 or round(fs_orig) < 1:
        raise ValueError("Source and target sampling rates must round to positive Hz.")
    # Use the nominal rational filter (e.g. 3/20), but preserve the actual
    # input clock. Clock-corrected SLAP2 fs is ~199.999206, not exactly 200;
    # assigning exactly 30 Hz would otherwise introduce accumulating drift.
    g = gcd(int(round(fs_target)), int(round(fs_orig)))
    up, down = int(round(fs_target)) // g, int(round(fs_orig)) // g
    dff_ds = resample_poly(y, up, down)
    fs_effective = fs_orig * up / down
    t_ds = t[0] + np.arange(len(dff_ds)) / fs_effective
    # resample_poly rounds length upward; its final sample can fall outside
    # the observed input interval for very short/non-divisible runs.
    inside = t_ds <= t[-1] + 1e-9
    dff_ds, t_ds = dff_ds[inside], t_ds[inside]
    longmask_ds = np.interp(t_ds, t, longmask.astype(float)) > 0.5
    return dff_ds, t_ds, longmask_ds, fs_orig


def _discrete_indices(
    spike_prob: np.ndarray, infer: Callable[..., tuple], **kwargs
) -> np.ndarray:
    """Infer independent finite spans and restore CascadeTorch's bin indices.

    CascadeTorch's discrete fitter removes NaNs and adds ``first_finite - 1``
    to its local indices. Passing an entirely finite span therefore returns
    indices one bin early. Correct that offset explicitly, then restore the
    span's original position. Splitting also prevents internal NaNs from
    compressing the time axis or joining independent supports.
    """
    finite = np.isfinite(spike_prob)
    edges = np.diff(np.r_[False, finite, False].astype(np.int8))
    indices = []
    for start, stop in zip(
        np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
    ):
        _, spikes = infer(spike_prob[None, start:stop], **kwargs)
        local = np.asarray(spikes[0], dtype=np.int64) + 1
        if ((local < 0) | (local >= stop - start)).any():
            raise RuntimeError(
                "CASCADE discrete inference returned an invalid sample index."
            )
        indices.append(local + start)
    return np.sort(np.concatenate(indices)) if indices else np.array([], dtype=np.int64)


def _as_single_soma_trace(
    soma_dff: xr.DataArray,
    soma_id: str | None,
) -> tuple[xr.DataArray, str]:
    """Reduce ``soma_dff`` to a 1-D ``(time,)`` trace and resolve its soma-ID.

    Accepts a ready 1-D ``(time,)`` array (the usual ``pureFF``), or one that
    still carries a length-1 ``channel`` and/or ``soma_id`` dimension (squeezed
    here). Raises if more than one soma or channel is present, since CASCADE is
    run one soma at a time.
    """
    da = soma_dff
    if "channel" in da.dims:
        if da.sizes["channel"] != 1:
            raise ValueError(
                f"run_cascade_detection expects a single channel; got "
                f"{da.sizes['channel']}. Select one, e.g. roi_F(..., channels=0)."
            )
        da = da.isel(channel=0)
    if "soma_id" in da.dims:
        if da.sizes["soma_id"] != 1:
            raise ValueError(
                f"run_cascade_detection expects a single soma; got "
                f"{da.sizes['soma_id']} somas. Select one via .sel(soma_id=...) "
                f"or loop over somas yourself."
            )
        da = da.isel(soma_id=0)
    if da.ndim != 1 or "time" not in da.dims:
        raise ValueError(
            f"Expected a 1-D trace over 'time' after reduction; got dims {da.dims}."
        )

    if soma_id is None:
        if "soma_id" in da.coords:
            val = da.coords["soma_id"].values
            soma_id = str(val) if val.ndim == 0 else str(val.reshape(-1)[0])
        else:
            raise ValueError(
                "Could not resolve soma_id from the array's coordinates; pass "
                "soma_id=... explicitly."
            )
    return da, soma_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_cascade_detection(
    soma_dff: xr.DataArray,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    fs_target: float = DEFAULT_FS_TARGET,
    model_folder: str | None = None,
    device: str | None = None,
    long_gap_s: float = DEFAULT_LONG_GAP_S,
    soma_id: str | None = None,
    download_if_missing: bool = True,
    verbosity: int = 1,
) -> tuple[xr.DataArray, pl.DataFrame]:
    """Run CASCADE spike inference on a single soma's dF/F trace.

    Mirrors the verified notebook pipeline: resample the dF/F to ``fs_target``,
    run ``cascade.predict`` to get a continuous spike-probability trace, then
    ``infer_discrete_spikes`` to obtain discrete spike times.

    Parameters
    ----------
    soma_dff : xr.DataArray
        A single soma's dF/F trace (a *fraction*, not percent), 1-D over
        ``time`` -- e.g. the ``pureFF`` array from a soma ROI dF/F loaded via
        ``roi_F(..., channels=0, return_dFF=True)[dmd].sel(soma_id=...)``.
        A length-1 ``channel``/``soma_id`` dimension is squeezed automatically.
    model_name : str
        CASCADE model name (default ``"Global_EXC_30Hz_smoothing25ms"``). Must
        match ``fs_target``.
    fs_target : float
        Target sampling rate (Hz) for the resample fed to CASCADE. Default 30.
    model_folder : str or None
        Folder holding the pretrained CASCADE models. ``None`` uses
        :data:`wisco_slap.defs.cascade_model_folder`.
    device : str or None
        Torch device string (e.g. ``"cuda"`` / ``"cpu"``). ``None`` auto-selects
        CUDA when available.
    long_gap_s : float
        Passed to :func:`_prepare_for_cascade`; NaN gaps longer than this are
        flagged internally (not currently attached to the outputs).
    soma_id : str or None
        Override for the soma identifier tied into the outputs. ``None`` reads
        it from the input array's ``soma_id`` coordinate.
    download_if_missing : bool
        If True and ``model_folder/model_name`` is absent, download it via
        ``cascade.download_model`` (needs network + a writable folder).
    verbosity : int
        Passed through to ``cascade.predict`` / ``infer_discrete_spikes``.

    Returns
    -------
    spike_prob : xr.DataArray
        Continuous CASCADE prediction (expected spikes per bin) near ``fs_target``,
        dims ``(time,)``, named ``"cascade_spike_prob"``, with ``soma_id`` tied
        in as a scalar coordinate and recorded in ``.attrs``. Edge windows are
        NaN (CASCADE ``padding=np.nan``). Actual timestamp gaps are split before
        inference; the returned axis retains those gaps. ``attrs['fs']`` is
        the effective, clock-preserving sample rate; ``model_fs`` is nominal.
    cascade_spikes : pl.DataFrame
        Discrete spikes with columns ``time`` (seconds, on the same axis as
        ``spike_prob``), ``soma_id`` (str), and ``spike_prob`` (the continuous
        CASCADE probability at each discrete spike -- usable as the ``priority``
        argument to :func:`filter_cascade_spikes`).

    Notes
    -----
    ``cascade2p`` and ``torch`` are imported lazily here (see the module
    docstring). No refractory thinning is applied -- use
    :func:`filter_cascade_spikes` for that.
    """
    da, soma_id = _as_single_soma_trace(soma_dff, soma_id)
    t = np.asarray(da.coords["time"].values, dtype=np.float64)
    dff = np.asarray(da.values, dtype=np.float64)
    if t.size < 2 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("CASCADE needs finite, strictly increasing time samples.")
    dt = float(np.median(np.diff(t)))
    fs_orig = 1.0 / dt
    # A missing interval must not become adjacent samples during filtering.
    run_edges = np.r_[0, np.flatnonzero(np.diff(t) > 1.5 * dt) + 1, t.size]

    if model_folder is None:
        model_folder = DEFS.cascade_model_folder

    # --- lazy CASCADE / torch imports (see module docstring) ---
    import torch
    from cascade2p import cascade, config
    from cascade2p.utils_discrete_spikes import infer_discrete_spikes

    model_dir = os.path.join(model_folder, model_name)
    if download_if_missing and not os.path.isdir(model_dir):
        cascade.download_model(model_name, model_folder=model_folder, verbose=verbosity)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model_config = config.read_config(os.path.join(model_dir, "config.yaml"))
    window_size = int(model_config["windowsize"])
    probabilities, times, spike_times, spike_values = [], [], [], []
    for start, stop in zip(run_edges[:-1], run_edges[1:], strict=True):
        if stop - start < 2:
            continue
        dff_ds, traces_time, _longmask_ds, _ = _prepare_for_cascade(
            t[start:stop], dff[start:stop], fs_target=fs_target, long_gap_s=long_gap_s
        )
        if dff_ds.size < window_size or not np.isfinite(dff_ds).any():
            prediction = np.full(dff_ds.size, np.nan)
        else:
            prediction = np.asarray(
                cascade.predict(
                    model_name,
                    dff_ds[None, :],
                    model_folder=model_folder,
                    device=device,
                    verbosity=verbosity,
                )
            )[0]
        idxs = _discrete_indices(
            prediction,
            infer_discrete_spikes,
            model_name=model_name,
            model_folder=model_folder,
            verbosity=verbosity,
        )
        probabilities.append(prediction)
        times.append(traces_time)
        spike_times.append(traces_time[idxs])
        spike_values.append(prediction[idxs])
    prediction = np.concatenate(probabilities) if probabilities else np.array([])
    traces_time = np.concatenate(times) if times else np.array([])
    fs_effective = fs_orig * round(fs_target) / round(fs_orig)

    spike_prob_da = xr.DataArray(
        prediction.astype(np.float32),
        dims=["time"],
        coords={"time": traces_time, "soma_id": soma_id},
        name="cascade_spike_prob",
        attrs={
            "trace_type": "cascade_spike_prob",
            "model_name": model_name,
            "fs": float(fs_effective),
            "model_fs": float(fs_target),
            "source_fs": float(fs_orig),
            "n_timestamp_runs": int(len(run_edges) - 1),
            "soma_id": soma_id,
        },
    )

    cascade_spikes = pl.DataFrame({
        "time": np.concatenate(spike_times) if spike_times else np.array([]),
        "soma_id": soma_id,
        "spike_prob": np.concatenate(spike_values) if spike_values else np.array([]),
    })

    return spike_prob_da, cascade_spikes


def filter_cascade_spikes(
    spikes: pl.DataFrame,
    min_separation_sec: float,
    *,
    time_col: str = "time",
    priority: str | Sequence[float] | np.ndarray | None = None,
) -> pl.DataFrame:
    """Thin spikes so no two kept spikes are closer than ``min_separation_sec``.

    A refractory-period filter with two collision-resolution strategies (ported
    from ``cascade_crosscheck.enforce_min_separation``):

    * ``priority=None`` (default) -- single forward pass: keep a spike iff it is
      >= ``min_separation_sec`` after the last *kept* spike ("first wins, then
      dead time"). The standard refractory rule; O(n) on time-sorted spikes.

    * ``priority`` set -- non-maximum suppression: repeatedly keep the
      highest-priority spike and drop every spike within ``min_separation_sec``
      of it. Use when you have a per-spike confidence/amplitude and want
      collisions resolved in favour of the stronger spike, not the earlier one.
      For CASCADE, pass ``priority="spike_prob"`` (the column produced by
      :func:`run_cascade_detection`).

    Parameters
    ----------
    spikes : polars.DataFrame
        Must contain ``time_col`` (spike times in seconds). Need not be sorted.
    min_separation_sec : float
        Minimum gap between consecutive kept spikes. ``<= 0`` is a no-op.
    time_col : str
        Name of the time column.
    priority : str, array-like, or None
        Per-spike priority, higher = preferred. A string selects a column of
        ``spikes``; an array must be row-aligned. ``None`` uses the temporal
        "first wins" rule.

    Returns
    -------
    polars.DataFrame
        Retained spikes, sorted by time, with the original columns preserved.
    """
    t = spikes[time_col].to_numpy().astype(float)
    n = t.size
    if n == 0 or min_separation_sec <= 0:
        return spikes.sort(time_col)

    finite = np.isfinite(t)
    keep = np.zeros(n, dtype=bool)

    if priority is None:
        # Temporal greedy: walk spikes in time order, accept when the gap from
        # the last accepted spike clears the refractory window.
        order = np.argsort(t, kind="stable")
        order = order[finite[order]]
        last = -np.inf
        for idx in order:
            if t[idx] - last >= min_separation_sec:
                keep[idx] = True
                last = t[idx]
    else:
        # Non-maximum suppression: accept spikes high-priority first, rejecting
        # any that fall within the window of an already-accepted spike. The
        # accepted times are kept sorted so each check is two bisect lookups.
        prio = (
            spikes[priority].to_numpy()
            if isinstance(priority, str)
            else np.asarray(priority)
        )
        prio = prio.astype(float)
        if prio.shape[0] != n:
            raise ValueError(
                f"priority length ({prio.shape[0]}) must match number of spikes ({n})."
            )
        cand = np.where(finite & np.isfinite(prio))[0]
        cand = cand[np.argsort(prio[cand], kind="stable")[::-1]]  # high -> low
        accepted: list[float] = []
        for idx in cand:
            x = t[idx]
            pos = bisect.bisect_left(accepted, x)
            left_ok = pos == 0 or x - accepted[pos - 1] >= min_separation_sec
            right_ok = pos == len(accepted) or accepted[pos] - x >= min_separation_sec
            if left_ok and right_ok:
                accepted.insert(pos, x)
                keep[idx] = True

    return spikes.filter(pl.Series(keep)).sort(time_col)


# ---------------------------------------------------------------------------
# Cross-check against an independent event detector
# ---------------------------------------------------------------------------
# Ported from cascade_crosscheck.py for completeness. These compare CASCADE's
# continuous rate against an independent set of event times (e.g. your glu
# event detector), so they consume the outputs run_cascade_detection already
# produces (spike_prob values + its time axis).


def event_triggered_rate(
    ev_t: np.ndarray,
    spike_prob: np.ndarray,
    t_ds: np.ndarray,
    fs_target: float,
    win_s: float = 2.0,
    n_shuffle: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average CASCADE ``spike_prob`` around each of your event times.

    A peak at lag 0 means your events coincide with CASCADE-inferred spiking.
    A shuffled baseline is returned for context.

    Parameters
    ----------
    ev_t : np.ndarray
        Your event times (seconds), on the same axis as ``t_ds``.
    spike_prob : np.ndarray
        Continuous CASCADE spike-probability trace (e.g.
        ``run_cascade_detection(...)[0].values``).
    t_ds : np.ndarray
        Time axis of ``spike_prob`` (seconds).
    fs_target : float
        Sampling rate of ``spike_prob`` (Hz).
    win_s : float
        Half-window (seconds) around each event.
    n_shuffle : int
        Number of shuffled events (as a multiple base) for the baseline stack.
    seed : int
        RNG seed.

    Returns
    -------
    lags : np.ndarray
        Lag axis (seconds), length ``2 * win_s * fs_target + 1``.
    real : np.ndarray
        Stacked event-triggered segments (n_events x n_lags).
    shuf : np.ndarray
        Stacked shuffled segments for the baseline.
    """
    w = int(win_s * fs_target)
    lags = np.arange(-w, w + 1) / fs_target
    finite = np.isfinite(spike_prob)

    def stack(times: np.ndarray) -> np.ndarray:
        rows = []
        for tt in times:
            c = int(round((tt - t_ds[0]) * fs_target))
            if c - w < 0 or c + w + 1 > len(spike_prob):
                continue
            seg = spike_prob[c - w : c + w + 1]
            if np.all(np.isfinite(seg)):
                rows.append(seg)
        return np.array(rows)

    real = stack(ev_t)
    rng = np.random.default_rng(seed)
    valid_t = t_ds[finite]
    shuf = stack(rng.uniform(valid_t.min(), valid_t.max(), size=len(ev_t) * 3))
    return lags, real, shuf


def rate_correlation(
    ev_idx_orig: np.ndarray,
    spike_prob: np.ndarray,
    t_ds: np.ndarray,
    t_orig: np.ndarray,
    fs_target: float,
    longmask_ds: np.ndarray,
    smooth_ms: float = 50.0,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin your events to the CASCADE rate grid and correlate the two rates.

    Parameters
    ----------
    ev_idx_orig : np.ndarray
        Indices of your events into ``t_orig``.
    spike_prob : np.ndarray
        Continuous CASCADE spike-probability trace.
    t_ds : np.ndarray
        Time axis of ``spike_prob`` (seconds).
    t_orig : np.ndarray
        Original (pre-resample) time axis (seconds).
    fs_target : float
        Sampling rate of ``spike_prob`` (Hz).
    longmask_ds : np.ndarray of bool
        Long-gap mask on the ``t_ds`` grid (from :func:`_prepare_for_cascade`);
        masked samples are excluded from the correlation.
    smooth_ms : float
        Gaussian smoothing (ms) applied to your binned event rate.

    Returns
    -------
    pearson : float
    spearman : float
    t_ds : np.ndarray
    ev_s : np.ndarray
        Your smoothed event rate (events/s) on the ``t_ds`` grid.
    cas : np.ndarray
        CASCADE rate (spikes/s) on the ``t_ds`` grid.
    good : np.ndarray of bool
        Mask of samples used in the correlation.
    """
    from scipy.stats import pearsonr, spearmanr

    edges = np.concatenate([t_ds - 0.5 / fs_target, [t_ds[-1] + 0.5 / fs_target]])
    ev_rate, _ = np.histogram(t_orig[ev_idx_orig], bins=edges)
    ev_rate = ev_rate.astype(float) * fs_target  # events/s
    sig = smooth_ms / 1e3 * fs_target
    ev_s = gaussian_filter1d(ev_rate, sig)
    cas = spike_prob * fs_target  # spikes/s
    good = np.isfinite(cas) & ~longmask_ds
    return (
        pearsonr(ev_s[good], cas[good])[0],
        spearmanr(ev_s[good], cas[good])[0],
        t_ds,
        ev_s,
        cas,
        good,
    )
