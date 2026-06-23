"""Render raw `_REGISTERED_DOWNSAMPLED-80Hz.tif` data as MP4 movies.

Given an acquisition and a time window `(t1, t2)` in the same axis that
``wis.get.syn_dF(..., apply_ephys_offset=True)`` returns (i.e. seconds since
sync_block-1's ephys_start, for the acq's first epoch), produce four MP4 files
covering that window: one per `(DMD, channel)` pair.

The TIFF files are per-trial: one file per `(epoch, trial, DMD)` with frames
at 80 Hz and the two channels interleaved along the first axis as
``[F1_C1, F1_C2, F2_C1, F2_C2, ...]``. The scopex zarrs are sampled at
``params.analyzeHz`` (typically 200 Hz) with their time coordinate built by
concatenating all trials without inter-trial gaps; the only gaps that appear
are at epoch boundaries (multi-epoch acqs). Mapping ``t1, t2`` → TIFF frames
therefore goes through a per-trial table built from the scopex time axis.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterator

import av
import numpy as np
import polars as pl
import tifffile
import xarray as xr

import slap2_py as spy

import wisco_slap.defs as DEFS
from wisco_slap.meta.get import ephys_offset as _get_ephys_offset
from wisco_slap.meta.get import esum_mirror_path as _esum_mirror_path

TIFF_FPS = 80.0  # the `_REGISTERED_DOWNSAMPLED-80Hz.tif` rate (fixed by preprocessing)
_CHANNEL_PAIRS = (("DMD1", 1, "ch1", 0), ("DMD1", 1, "ch2", 1),
                  ("DMD2", 2, "ch1", 0), ("DMD2", 2, "ch2", 1))


# ----------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------
def _raw_acq_dir(subject: str, exp: str, loc: str, acq: str) -> str:
    return os.path.join(DEFS.raw_mirror_root, subject, exp, loc, acq)


def _tiff_path(
    subject: str, exp: str, loc: str, acq: str,
    epoch: int, trial_one_based: int, dmd: int,
) -> str:
    fname = f"E{epoch}T{trial_one_based}DMD{dmd}_REGISTERED_DOWNSAMPLED-80Hz.tif"
    return os.path.join(_raw_acq_dir(subject, exp, loc, acq), fname)


def _scopex_time_array(subject: str, exp: str, loc: str, acq: str) -> np.ndarray:
    """Read just the scopex time coordinate (with ephys_offset applied) from
    ``syn_dF-ls.zarr``. This is the same axis ``wis.get.syn_dF`` returns by
    default. Reading only the ``time`` coord is fast — xarray/zarr will not
    pull the per-synapse data."""
    zarr_path = os.path.join(
        DEFS.anmat_root, subject, exp, "scopex", f"{loc}--{acq}", "syn_dF-ls.zarr",
    )
    if not os.path.isdir(zarr_path):
        raise FileNotFoundError(
            f"scopex zarr not found for {subject}/{exp}/{loc}/{acq}: {zarr_path}. "
            f"Run wis.pns.scopex_mon.exp_data(subject, exp) first."
        )
    for group in ("dmd_1", "dmd_2"):
        if os.path.isdir(os.path.join(zarr_path, group)):
            ds = xr.open_zarr(zarr_path, group=group, consolidated=False)
            t_internal = ds["time"].values.astype(np.float64, copy=True)
            ds.close()
            break
    else:
        raise FileNotFoundError(f"{zarr_path} contains neither dmd_1 nor dmd_2")
    # syn_dF-ls.zarr's time coord is stored WITHOUT the ephys_offset shift;
    # apply it here so the returned axis matches ``wis.get.syn_dF`` defaults.
    t_internal += float(_get_ephys_offset(subject, exp, loc, acq))
    return t_internal


# ----------------------------------------------------------------------
# Trial table
# ----------------------------------------------------------------------
def _build_trial_table(
    subject: str, exp: str, loc: str, acq: str,
) -> tuple[pl.DataFrame, float, np.ndarray]:
    """Build a per-trial table with scopex sample ranges, time-axis ranges, and
    per-trial TIFF frame counts.

    Returns
    -------
    table : pl.DataFrame
        Columns: ``trial_idx`` (0-based; ``trial_idx + 1`` = T# in the filename),
        ``epoch`` (1-based; matches E# in the filename), ``scopex_sample_start``,
        ``scopex_sample_end`` (exclusive), ``t_axis_start``, ``t_axis_end``
        (in the scopex time axis with ephys_offset applied), ``tiff_frames``
        (frame count at 80 Hz for this trial).
    fs_scopex : float
        Scopex sample rate (``params.analyzeHz`` from ExSum).
    time_axis : np.ndarray
        The scopex time array, with ephys_offset applied.
    """
    esum_path = _esum_mirror_path(subject, exp, loc, acq)
    if esum_path in (0, "NO_ESUM_MIRROR") or not isinstance(esum_path, str):
        raise FileNotFoundError(
            f"ExSum mirror missing for {subject}/{exp}/{loc}/{acq}. "
            f"Run wis.meta.update() first."
        )

    trial_epochs = spy.hf.load_any(esum_path, "/exptSummary/trialTable['epoch']")
    trial_epochs = np.asarray(trial_epochs).squeeze().astype(int, copy=False).ravel()

    analyze_hz = spy.hf.load_any(esum_path, "/exptSummary/params['analyzeHz']")
    fs_scopex = float(np.asarray(analyze_hz).squeeze())

    time_axis = _scopex_time_array(subject, exp, loc, acq)
    n_total_scopex = int(time_axis.size)

    # Probe one TIFF per epoch for its frame count (cheap — reads only headers).
    n_trials = int(trial_epochs.size)
    epochs_present = sorted({int(e) for e in trial_epochs})
    frames_per_trial_by_epoch: dict[int, int] = {}
    for e in epochs_present:
        first_trial_in_epoch = int(np.where(trial_epochs == e)[0][0]) + 1  # 1-based
        # DMD1 vs DMD2 will have the same frame count (synchronous acquisition).
        probe = _tiff_path(subject, exp, loc, acq, e, first_trial_in_epoch, 1)
        if not os.path.exists(probe):
            probe2 = _tiff_path(subject, exp, loc, acq, e, first_trial_in_epoch, 2)
            if not os.path.exists(probe2):
                raise FileNotFoundError(
                    f"No probe TIFF for epoch {e} of {subject}/{exp}/{loc}/{acq}: "
                    f"tried {probe} and {probe2}"
                )
            probe = probe2
        with tifffile.TiffFile(probe) as tf:
            n_pages = len(tf.pages)
        if n_pages % 2 != 0:
            warnings.warn(
                f"Probe TIFF {probe} has odd page count {n_pages}; expected an "
                f"even number (two interleaved channels). Using floor-divide.",
                stacklevel=2,
            )
        frames_per_trial_by_epoch[e] = n_pages // 2

    # Build per-trial rows by walking trial_epochs in order.
    rows: list[dict] = []
    cumulative_samples = 0
    for trial_idx in range(n_trials):
        e = int(trial_epochs[trial_idx])
        frames = int(frames_per_trial_by_epoch[e])
        samples = int(round(frames * fs_scopex / TIFF_FPS))
        s0 = cumulative_samples
        s1 = cumulative_samples + samples
        # Clamp the last trial in case of off-by-a-few-samples (the SLAP2 last
        # trial can legitimately be shorter than the canonical per-trial count).
        if trial_idx == n_trials - 1 and s1 > n_total_scopex:
            s1 = n_total_scopex
        rows.append({
            "trial_idx": trial_idx,
            "epoch": e,
            "scopex_sample_start": s0,
            "scopex_sample_end": s1,
            "tiff_frames": frames,
        })
        cumulative_samples = s1

    # Sanity-check overall sample count.
    discrepancy = abs(cumulative_samples - n_total_scopex)
    if discrepancy > max(10, fs_scopex):  # allow up to a second's worth of slop
        raise RuntimeError(
            f"Trial-table sample accounting drifted by {discrepancy} samples for "
            f"{subject}/{exp}/{loc}/{acq} (cumulative={cumulative_samples}, "
            f"scopex_total={n_total_scopex}). The per-trial frame count probed "
            f"from TIFF headers may be wrong for some epoch, or the ExSum's "
            f"trialTable does not match the saved scopex zarr."
        )

    # Attach t_axis_start / t_axis_end from the scopex time array.
    starts = np.array([r["scopex_sample_start"] for r in rows], dtype=np.int64)
    ends = np.array([r["scopex_sample_end"] for r in rows], dtype=np.int64)
    t_start = time_axis[starts]
    # For end: time at sample (end-1) plus one sample period; use the median dt
    # within each epoch for robustness (epoch-boundary jumps don't enter here
    # because we only use intra-trial samples).
    dt_scopex = 1.0 / fs_scopex
    t_end = time_axis[ends - 1] + dt_scopex

    table = pl.DataFrame({
        "trial_idx": [r["trial_idx"] for r in rows],
        "epoch": [r["epoch"] for r in rows],
        "scopex_sample_start": [r["scopex_sample_start"] for r in rows],
        "scopex_sample_end": [r["scopex_sample_end"] for r in rows],
        "t_axis_start": t_start,
        "t_axis_end": t_end,
        "tiff_frames": [r["tiff_frames"] for r in rows],
    })
    return table, fs_scopex, time_axis


def _select_trials_and_frames(
    trial_table: pl.DataFrame, t1: float, t2: float,
) -> pl.DataFrame:
    """Filter to trials overlapping [t1, t2] and add TIFF frame slice columns.

    Within each selected trial, the slice is `[tiff_frame_start, tiff_frame_end)`
    at 80 Hz. Conversion: ``tiff_frame = round((t - t_axis_start) * 80)``,
    clamped to ``[0, tiff_frames]``.
    """
    if not t2 > t1:
        raise ValueError(f"t2 ({t2}) must be greater than t1 ({t1}).")

    selected = trial_table.filter(
        (pl.col("t_axis_end") > t1) & (pl.col("t_axis_start") < t2)
    )
    if selected.is_empty():
        raise ValueError(
            f"No trials overlap [{t1}, {t2}]. Trial-table covers "
            f"[{trial_table['t_axis_start'].min():.3f}, "
            f"{trial_table['t_axis_end'].max():.3f}]."
        )

    starts_s = ((np.maximum(selected["t_axis_start"].to_numpy(), t1)
                 - selected["t_axis_start"].to_numpy()) * TIFF_FPS)
    ends_s = ((np.minimum(selected["t_axis_end"].to_numpy(), t2)
               - selected["t_axis_start"].to_numpy()) * TIFF_FPS)
    n_frames_per_row = selected["tiff_frames"].to_numpy()

    fstart = np.clip(np.round(starts_s).astype(np.int64), 0, n_frames_per_row)
    fend = np.clip(np.round(ends_s).astype(np.int64), 0, n_frames_per_row)
    # Drop rows that round to a zero-frame slice (e.g. user window touches the
    # very tail of a trial).
    keep = fend > fstart
    selected = selected.with_columns(
        pl.Series("tiff_frame_start", fstart, dtype=pl.Int64),
        pl.Series("tiff_frame_end", fend, dtype=pl.Int64),
    ).filter(pl.Series(keep))
    if selected.is_empty():
        raise ValueError(
            f"Window [{t1}, {t2}] resolved to zero TIFF frames after rounding."
        )
    return selected


# ----------------------------------------------------------------------
# TIFF loading
# ----------------------------------------------------------------------
def _load_trial_slice(
    path: str, frame_start: int, frame_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a slice of one trial's TIFF and deinterleave the two channels.

    The file's first axis is `[F1_C1, F1_C2, F2_C1, F2_C2, ...]` — so frame `f`
    of channel `c` lives at page `2*f + c`. Slicing the file in page units
    avoids reading frames we don't need.
    """
    page_start = 2 * int(frame_start)
    page_end = 2 * int(frame_end)
    data = tifffile.imread(path, key=range(page_start, page_end))
    # tifffile returns either (n_pages, H, W) when n_pages > 1 or (H, W) when 1.
    if data.ndim == 2:
        data = data[None, ...]
    ch1 = data[0::2]
    ch2 = data[1::2]
    return ch1, ch2


# ----------------------------------------------------------------------
# Contrast / encoding
# ----------------------------------------------------------------------
def _percentile_contrast(
    stack: np.ndarray, lo_pct: float, hi_pct: float, max_sample_frames: int = 200,
) -> tuple[float, float]:
    """Return ``(vmin, vmax)`` from a subsample of frames for uint8 scaling."""
    if stack.shape[0] > max_sample_frames:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(stack.shape[0], size=max_sample_frames, replace=False)
        sample = stack[idx]
    else:
        sample = stack
    lo, hi = np.nanpercentile(sample.astype(np.float64), [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _resolve_cmap(name: str | None):
    """Look up a matplotlib colormap by name; returns ``None`` to mean grayscale."""
    if name is None:
        return None
    from matplotlib import colormaps as _mpl_colormaps

    return _mpl_colormaps[name]


def _to_rgb_u8(
    frame: np.ndarray, vmin: float, vmax: float, cmap=None,
) -> np.ndarray:
    """Normalize a single frame to ``[0, 1]`` then render to ``(H, W, 3)`` uint8.

    ``cmap`` is either ``None`` (grayscale, replicated to three channels) or a
    matplotlib ``Colormap`` (RGBA → RGB). NaN pixels (typically the area
    outside the active DMD footprint in registered TIFFs) are forced to black
    in the output regardless of the colormap, so colormaps like ``Reds`` /
    ``Greens`` that map 0 to white don't bleed into the off-footprint region.
    """
    f32 = frame.astype(np.float32, copy=False)
    nan_mask = ~np.isfinite(f32)
    x = (f32 - vmin) / (vmax - vmin)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(x, 0.0, 1.0, out=x)
    if cmap is None:
        u8 = (x * 255.0 + 0.5).astype(np.uint8)
        rgb = np.stack([u8, u8, u8], axis=-1)
    else:
        rgba = cmap(x, bytes=True)  # (H, W, 4) uint8
        rgb = rgba[..., :3].copy()
    if nan_mask.any():
        rgb[nan_mask] = 0
    return rgb


def _apply_smoothing(stack: np.ndarray, window_frames: int) -> np.ndarray:
    """Box-filter along the time axis (axis 0). Returns ``stack`` unchanged
    when ``window_frames <= 1``. Operates in float32 to avoid uint16 overflow."""
    if window_frames <= 1:
        return stack
    from scipy.ndimage import uniform_filter1d

    return uniform_filter1d(
        stack.astype(np.float32, copy=False),
        size=int(window_frames),
        axis=0,
        mode="nearest",
    )


def _stream_mp4(
    frames_iter: Iterator[np.ndarray],
    out_path: str,
    fps: float,
    codec: str,
    crf: int,
    width: int,
    height: int,
    n_frames: int,
    progress: bool,
    progress_every: int = 100,
) -> None:
    """Encode a stream of ``(H, W, 3)`` uint8 RGB frames to an MP4 via PyAV.

    Mirrors the encoder setup pattern in ``wisco_slap.pns.synmovies`` (see
    `wisco-slap/src/wisco_slap/pns/synmovies.py:510`); yuv420p output for
    universal compatibility, with right/bottom zero-pad when dimensions are odd.
    """
    H = int(height); W = int(width)
    H_eff = H + (H % 2)
    W_eff = W + (W % 2)
    pix_fmt = "yuv420p"
    pad_needed = (H_eff != H) or (W_eff != W)

    container = av.open(out_path, mode="w")
    stream = container.add_stream(codec, rate=int(round(fps)))
    stream.width = W_eff
    stream.height = H_eff
    stream.pix_fmt = pix_fmt

    opts: dict[str, str] = {}
    if codec in ("h264", "libx264"):
        opts["crf"] = str(int(crf))
        opts["preset"] = "veryfast"
    elif codec == "h264_nvenc":
        opts["cq"] = str(int(crf))
        opts["preset"] = "p5"
    if hasattr(stream, "codec_context") and hasattr(stream.codec_context, "options"):
        stream.codec_context.options = opts

    label = os.path.basename(out_path)
    for i, rgb in enumerate(frames_iter):
        if pad_needed:
            padded = np.zeros((H_eff, W_eff, 3), dtype=np.uint8)
            padded[:H, :W, :] = rgb
            rgb = padded
        video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        if (
            video_frame.width != W_eff
            or video_frame.height != H_eff
            or video_frame.format.name != pix_fmt
        ):
            video_frame = video_frame.reformat(
                width=W_eff, height=H_eff, format=pix_fmt
            )
        for packet in stream.encode(video_frame):
            container.mux(packet)
        if progress and ((i + 1) % progress_every == 0 or (i + 1) == n_frames):
            print(f"[{label}] encoded {i + 1}/{n_frames}")

    for packet in stream.encode():  # flush
        container.mux(packet)
    container.close()


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def make(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    t1: float,
    t2: float,
    *,
    out_dir: str | None = None,
    fps: float = 80.0,
    cmaps: tuple[str | None, str | None] | None = None,
    smoothing_s: float | None = None,
    rotate_k: int = 1,
    contrast_percentiles: (
        tuple[float, float] | dict[str, tuple[float, float]]
    ) = (1.0, 99.5),
    codec: str = "h264",
    crf: int = 18,
    overwrite: bool = False,
    progress: bool = True,
) -> dict[str, str]:
    """Render four MP4 movies of the raw 80 Hz TIFF data for the window
    ``[t1, t2]``.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers (e.g. ``"alnair", "exp_5", "loc_H", "acq_1"``).
    t1, t2 : float
        Time window in the *scopex time axis with ephys_offset applied* —
        the same axis as ``wis.get.syn_dF(subject, exp, loc, acq)['dmd_1'].time``.
        For multi-epoch cross-sync-block acqs this axis is anchored to
        sync_block-1's ephys_start (see ``wis.get._get_scopex._apply_ephys_offset``).
    out_dir : str | None
        Directory to write the four MP4s into. Defaults to
        ``<anmat_root>/<subject>/<exp>/viz/tiff_movies/<loc>--<acq>/t{t1:.2f}-{t2:.2f}/``.
    fps : float
        Playback frame rate. Default 80 (= real-time). No interpolation — this
        only changes how fast the rendered frames are played.
    cmaps : (str | None, str | None) | None
        Per-channel matplotlib colormaps applied to both DMDs: ``cmaps[0]``
        for ch1, ``cmaps[1]`` for ch2. Example: ``("Reds", "Greens")``.
        Pass ``None`` (default) for grayscale on both channels, or e.g.
        ``("Reds", None)`` for cmap on ch1 only.
    smoothing_s : float | None
        If given, apply a uniform rolling-mean over time of width
        ``round(smoothing_s * 80)`` TIFF frames (axis 0 of each stack). Use
        e.g. ``0.05`` for a 50 ms box filter. ``None`` (default) disables.
        Note: smoothing crosses trial boundaries (which are contiguous in
        scopex time) and epoch boundaries (which are NOT — there's a true
        wall-clock gap), so windows that straddle epoch boundaries will
        average across a real time gap.
    rotate_k : int
        Number of CCW 90° rotations to apply to each frame (``np.rot90``).
        Default ``1`` orients the SLAP2 FOV with its x-axis horizontal.
        Set to ``0`` to disable.
    contrast_percentiles : (float, float) | dict[str, (float, float)]
        Lower / upper percentiles for uint8 scaling. Either a single
        ``(lo, hi)`` 2-tuple applied to every ``(DMD, channel)`` stack
        (default ``(1.0, 99.5)``), or a dict keyed by ``"DMD1"`` / ``"DMD2"``
        for per-DMD percentiles, e.g.
        ``{"DMD1": (1.0, 99.5), "DMD2": (0.5, 99.9)}``. Within each DMD the
        two channels still get their own ``(vmin, vmax)`` computed at those
        percentiles.
    codec : str
        FFmpeg codec name. ``"h264"`` (libx264, default) or ``"h264_nvenc"``.
    crf : int
        Constant-rate-factor for libx264 (also used as ``cq`` for NVENC).
        Default 18 ≈ visually lossless.
    overwrite : bool
        If False (default), refuse to clobber existing MP4s in ``out_dir``.
    progress : bool
        Print per-MP4 encoding progress.

    Returns
    -------
    dict
        ``{"DMD1_ch1": path, "DMD1_ch2": path, "DMD2_ch1": path, "DMD2_ch2": path}``.
    """
    # Resolve contrast_percentiles into a per-DMD lookup.
    percentiles_by_dmd: dict[str, tuple[float, float]]
    if isinstance(contrast_percentiles, dict):
        unknown = set(contrast_percentiles) - {"DMD1", "DMD2"}
        if unknown:
            raise ValueError(
                f"contrast_percentiles dict keys must be 'DMD1' / 'DMD2', "
                f"got unknown key(s): {sorted(unknown)}."
            )
        # Default any missing DMD to the original (1.0, 99.5).
        default_pcts = (1.0, 99.5)
        percentiles_by_dmd = {
            "DMD1": tuple(contrast_percentiles.get("DMD1", default_pcts)),  # type: ignore[arg-type]
            "DMD2": tuple(contrast_percentiles.get("DMD2", default_pcts)),  # type: ignore[arg-type]
        }
    else:
        if len(contrast_percentiles) != 2:
            raise ValueError(
                f"contrast_percentiles tuple must be length 2 (lo, hi), "
                f"got length {len(contrast_percentiles)}."
            )
        pct = tuple(contrast_percentiles)  # type: ignore[arg-type]
        percentiles_by_dmd = {"DMD1": pct, "DMD2": pct}

    # Validate cmaps tuple early; resolve names to matplotlib Colormap objects.
    if cmaps is not None and len(cmaps) != 2:
        raise ValueError(
            f"cmaps must be a 2-tuple (ch1, ch2), got length {len(cmaps)}."
        )
    cmap_ch1 = _resolve_cmap(cmaps[0]) if cmaps is not None else None
    cmap_ch2 = _resolve_cmap(cmaps[1]) if cmaps is not None else None
    cmap_by_key = {
        "DMD1_ch1": cmap_ch1, "DMD2_ch1": cmap_ch1,
        "DMD1_ch2": cmap_ch2, "DMD2_ch2": cmap_ch2,
    }

    window_frames = 0
    if smoothing_s is not None and smoothing_s > 0:
        window_frames = int(round(float(smoothing_s) * TIFF_FPS))
        if window_frames <= 1:
            window_frames = 0  # too small to matter

    rotate_k = int(rotate_k) % 4
    trial_table, _fs_scopex, _time_axis = _build_trial_table(subject, exp, loc, acq)
    selected = _select_trials_and_frames(trial_table, t1, t2)

    if out_dir is None:
        out_dir = os.path.join(
            DEFS.anmat_root, subject, exp, "viz", "tiff_movies",
            f"{loc}--{acq}", f"t{t1:.2f}-{t2:.2f}",
        )
    os.makedirs(out_dir, exist_ok=True)
    out_paths = {
        f"{dmd_label}_{ch_label}": os.path.join(out_dir, f"{dmd_label}_{ch_label}.mp4")
        for dmd_label, _dmd, ch_label, _ch_idx in _CHANNEL_PAIRS
    }
    if not overwrite:
        existing = [p for p in out_paths.values() if os.path.exists(p)]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing MP4(s): {existing}. "
                f"Pass overwrite=True to replace."
            )

    # Load all selected slices into per-(DMD,channel) lists. For 8s windows
    # this is fine in memory (~640 frames × ~750×400 × uint16 × 4 channels
    # ≈ 1.5 GB).  Missing TIFFs are deferred as ("BLACK", n_frames, dmd_label)
    # tokens and filled with zeros once we know the spatial shape from a
    # successfully-loaded slice.
    per_key: dict[str, list] = {k: [] for k in out_paths}
    shape_by_dmd: dict[str, tuple[int, int]] = {}
    for row in selected.iter_rows(named=True):
        epoch = int(row["epoch"])
        trial_one_based = int(row["trial_idx"]) + 1
        f0 = int(row["tiff_frame_start"])
        f1 = int(row["tiff_frame_end"])
        for dmd_label, dmd in (("DMD1", 1), ("DMD2", 2)):
            path = _tiff_path(subject, exp, loc, acq, epoch, trial_one_based, dmd)
            if not os.path.exists(path):
                warnings.warn(
                    f"Missing TIFF for epoch {epoch} trial {trial_one_based} "
                    f"{dmd_label}: {path}. That segment will be black.",
                    stacklevel=2,
                )
                per_key[f"{dmd_label}_ch1"].append(("BLACK", dmd_label, f1 - f0))
                per_key[f"{dmd_label}_ch2"].append(("BLACK", dmd_label, f1 - f0))
                continue
            ch1, ch2 = _load_trial_slice(path, f0, f1)
            per_key[f"{dmd_label}_ch1"].append(ch1)
            per_key[f"{dmd_label}_ch2"].append(ch2)
            shape_by_dmd.setdefault(dmd_label, (ch1.shape[1], ch1.shape[2]))

    if not shape_by_dmd:
        raise FileNotFoundError(
            f"All TIFFs in window [{t1}, {t2}] for {subject}/{exp}/{loc}/{acq} "
            f"were missing on disk."
        )

    fallback_shape = next(iter(shape_by_dmd.values()))

    def _resolve(piece):
        if isinstance(piece, tuple) and piece and piece[0] == "BLACK":
            _, dmd_label, n = piece
            H, W = shape_by_dmd.get(dmd_label, fallback_shape)
            return np.zeros((n, H, W), dtype=np.uint16)
        return piece

    stacks = {
        k: np.concatenate([_resolve(p) for p in v], axis=0)
        for k, v in per_key.items()
    }

    # Rotate every stack (axes 1, 2 are the spatial H, W) so the SLAP2 FOV is
    # oriented x-horizontal in the rendered MP4. After ``rot90(k=1)`` on shape
    # ``(N, H, W)`` the new shape is ``(N, W, H)`` — i.e. height and width swap.
    if rotate_k:
        stacks = {k: np.rot90(s, k=rotate_k, axes=(1, 2)) for k, s in stacks.items()}

    # Apply temporal smoothing BEFORE contrast — a tighter (vmin, vmax) on
    # smoothed data gives better visible dynamic range in the MP4.
    if window_frames:
        if progress:
            print(
                f"[smoothing] uniform_filter1d size={window_frames} frames "
                f"({window_frames / TIFF_FPS:.3f}s) along time axis"
            )
        stacks = {k: _apply_smoothing(s, window_frames) for k, s in stacks.items()}

    # Per-key contrast → encode. Percentiles are looked up per DMD; both
    # channels of a DMD share the percentile pair but each gets its own
    # (vmin, vmax) computed against its own data.
    for key, stack in stacks.items():
        dmd_label = key.split("_", 1)[0]
        lo_pct, hi_pct = percentiles_by_dmd[dmd_label]
        vmin, vmax = _percentile_contrast(stack, lo_pct, hi_pct)
        n_frames = stack.shape[0]
        H, W = stack.shape[1], stack.shape[2]
        cmap = cmap_by_key[key]
        if progress:
            cmap_label = cmap.name if cmap is not None else "gray"
            print(
                f"[{key}] {n_frames} frames, {W}x{H}, "
                f"contrast=({vmin:.1f}, {vmax:.1f}), cmap={cmap_label}, "
                f"→ {out_paths[key]}"
            )
        _stream_mp4(
            (_to_rgb_u8(stack[i], vmin, vmax, cmap) for i in range(n_frames)),
            out_paths[key], fps=fps, codec=codec, crf=crf,
            width=W, height=H, n_frames=n_frames, progress=progress,
        )

    return out_paths
