"""Stream registered TIFFs to lossless display movies with authoritative ephys clocks.

The saved float64 frame timestamps, not the container's nominal FPS, define time.
They use exactly the per-original-trial clock used by repaired shaft calcium.
Movies retain every source frame; smoothing, contrast clipping and the 8-bit
colormap are display transforms, not a replacement for quantitative raw TIFFs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import h5py
import numpy as np
import tifffile
import xarray as xr
from scipy.ndimage import uniform_filter1d

import slap2_py as spy
from wisco_slap import defs
from wisco_slap.meta import get as meta
from wisco_slap.pns._frame_timing import (
    SHAFT_TIME_AXIS_VERSION,
    fill_dropped_trial_counts,
    scopex_frame_times,
)
from wisco_slap.pns.aligned_frames import (
    per_trial_sample_counts,
    read_channel_schema,
    trial_epochs,
)


@dataclass(frozen=True)
class Trial:
    """One real TIFF and its original trial identity and output-frame interval."""

    path: str
    trial: int
    epoch: int
    start: int
    stop: int
    n_frames: int


@dataclass
class FramePlan:
    """A DMD's complete, trial-aware frame inventory."""

    dmd: int
    trials: list[Trial]
    times: np.ndarray
    analyzed_counts: list[int]
    shape: tuple[int, int]
    n_channels: int
    rate: float
    segment_edges: np.ndarray


def default_movie_dir(subject: str, exp: str, loc: str, acq: str) -> Path:
    """Canonical full recording: ``<subject>/<exp>/movies/tiffs__<loc>--<acq>``."""
    return Path(defs.anmat_root, subject, exp, "movies", f"tiffs__{loc}--{acq}")


def _tiff_path(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    epoch: int,
    trial_one_based: int,
    dmd: int,
) -> str:
    return str(
        Path(
            defs.raw_mirror_root,
            subject,
            exp,
            loc,
            acq,
            f"E{epoch}T{trial_one_based}DMD{dmd}_REGISTERED_DOWNSAMPLED-80Hz.tif",
        )
    )


def plan_acquisition(
    subject: str, exp: str, loc: str, acq: str
) -> tuple[list[FramePlan], dict]:
    """Read only metadata; validate TIFF counts against the stored Scopex clock.

    Missing TIFFs fail explicitly. Dropped analyzed trials retain their TIFFs
    and their original Scopex placeholder duration. Cross-sync-block recordings
    are rejected because one ephys axis cannot identify two different blocks.
    """
    keys = (subject, exp, loc, acq)
    exsum = Path(meta.esum_mirror_path(*keys))
    with h5py.File(exsum, "r") as f:
        n_trials = f["exptSummary/E"].shape[1]
    epochs = trial_epochs(exsum, n_trials)
    analyze_hz = float(
        np.asarray(
            spy.hf.load_any(str(exsum), "/exptSummary/params['analyzeHz']")
        ).squeeze()
    )
    schema = read_channel_schema(exsum)
    n_channels = len(schema["rawChannelNumbers"])
    sync = meta.sync_info()[subject][exp]["acquisitions"][f"{loc}--{acq}"]
    sync_block = int(sync["sync_block"])
    offsets = None
    if np.unique(epochs).tolist() != [1]:
        epoch_info = meta.epoch_info()[subject][exp][f"{loc}--{acq}"]
        blocks = {int(v["sync_block"]) for v in epoch_info.values()}
        if blocks != {sync_block}:
            raise ValueError(
                "Movies spanning different ephys sync blocks "
                "must be generated per block"
            )
        offsets = {
            int(k): float(v["duration_since_epoch_1_s"]) for k, v in epoch_info.items()
        }
    ratio, offset = float(meta.clock_ratio(*keys)), float(meta.ephys_offset(*keys))
    if not np.isfinite(ratio) or ratio <= 0 or not np.isfinite(offset):
        raise ValueError("Invalid acquisition clock metadata")
    plans = []
    zarr = Path(
        defs.anmat_root, subject, exp, "scopex", f"{loc}--{acq}", "syn_dF-ls.zarr"
    )
    for dmd in (1, 2):
        group = zarr / f"dmd_{dmd}"
        if not group.exists():
            continue
        counts = per_trial_sample_counts(exsum, dmd, n_trials)
        filled = fill_dropped_trial_counts(counts, epochs)
        paths = [_tiff_path(*keys, int(e), t + 1, dmd) for t, e in enumerate(epochs)]
        frame_counts, shape = [], None
        for path in paths:
            with tifffile.TiffFile(path) as tf:
                npages, this_shape = len(tf.pages), tuple(tf.pages[0].shape)
            if npages % n_channels:
                raise ValueError(f"Incomplete interleaved channels in {path}")
            if shape is not None and this_shape != shape:
                raise ValueError(f"Changing spatial dimensions at {path}")
            shape = this_shape
            frame_counts.append(npages // n_channels)
        internal, rate = scopex_frame_times(
            frame_counts, counts, analyze_hz, epochs, offsets
        )
        # Check every trial boundary against actual stored Scopex timestamps.
        with xr.open_zarr(str(zarr), group=f"dmd_{dmd}", consolidated=False) as ds:
            scope_times = np.asarray(ds.time.values, dtype=float)
        if len(scope_times) != int(filled.sum()):
            raise ValueError(f"DMD{dmd}: ExSum and Scopex trial counts disagree")
        sample_starts = np.r_[0, np.cumsum(filled)[:-1]]
        edges = np.r_[0, np.cumsum(frame_counts)]
        if not np.allclose(
            internal[edges[:-1]], scope_times[sample_starts], rtol=0, atol=1e-7
        ):
            raise ValueError(
                f"DMD{dmd}: canonical TIFF and stored Scopex trial boundaries disagree"
            )
        times = internal / ratio + offset
        # Smooth only inside continuous epoch runs; never through wall-clock gaps.
        breaks = np.flatnonzero(np.diff(epochs) != 0) + 1
        segment_edges = np.r_[0, edges[breaks], len(times)].astype(np.int64)
        trials = [
            Trial(p, i + 1, int(e), int(edges[i]), int(edges[i + 1]), int(n))
            for i, (p, e, n) in enumerate(zip(paths, epochs, frame_counts, strict=True))
        ]
        plans.append(
            FramePlan(
                dmd,
                trials,
                times,
                counts,
                shape,
                n_channels,
                float(rate * ratio),
                segment_edges,
            )
        )
    if not plans:
        raise FileNotFoundError(f"No Scopex DMD data in {zarr}")
    stat = exsum.stat()
    provenance = dict(
        schema_version=1,
        timebase="ephys",
        acquisition=dict(zip(("subject", "exp", "loc", "acq"), keys, strict=True)),
        sync_block=sync_block,
        clock_ratio=ratio,
        ephys_offset_s=offset,
        time_axis_version=SHAFT_TIME_AXIS_VERSION,
        channel_schema=schema,
        exsum=dict(path=str(exsum), size=stat.st_size, mtime_ns=stat.st_mtime_ns),
        timing_method=(
            "scopex_frame_times(frame_counts, analyzed_counts, analyze_hz, "
            "epochs, epoch_offsets) / clock_ratio + ephys_offset"
        ),
        timing_limit=(
            "Exact agreement with shaft and Scopex trial clocks; TIFF samples are "
            "uniformly placed within each analyzed trial, not independently "
            "measured exposure TTLs."
        ),
    )
    return plans, provenance


def _read(plan: FramePlan, channel: int, start: int, stop: int) -> np.ndarray:
    """Read a bounded float32 channel slice; memory-map uncompressed TIFF pages."""
    pieces = []
    for trial in plan.trials:
        if trial.stop <= start or trial.start >= stop:
            continue
        lo, hi = (
            max(start, trial.start) - trial.start,
            min(stop, trial.stop) - trial.start,
        )
        try:
            raw = tifffile.memmap(trial.path, mode="r")
            piece = raw[
                lo * plan.n_channels + channel : hi * plan.n_channels : plan.n_channels
            ]
        except ValueError:
            piece = tifffile.imread(
                trial.path,
                key=range(
                    lo * plan.n_channels + channel,
                    hi * plan.n_channels,
                    plan.n_channels,
                ),
            )
            if piece.ndim == 2:
                piece = piece[None]
        pieces.append(piece)
    if not pieces:
        raise ValueError("Empty source frame range")
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


def _packed_smooth(
    plan: FramePlan, channel: int, start: int, stop: int, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth real pixels only, with full halos and no mixing across epoch gaps.

    Returns (time by finite-pixel values, flattened pixel indices). Missing pixels
    stay absent; finite-weight filtering prevents a single NaN poisoning a trace.
    """
    seg = np.searchsorted(plan.segment_edges, start, side="right") - 1
    seg0, seg1 = plan.segment_edges[seg : seg + 2]
    if stop > seg1:
        raise ValueError("Smoothing chunks must not cross an epoch boundary")
    left, right = window // 2, (window - 1) // 2
    lo, hi = max(int(seg0), start - left), min(int(seg1), stop + right)
    raw = _read(plan, channel, lo, hi)
    mask = np.isfinite(raw).any(axis=0)
    coords = np.flatnonzero(mask)
    packed = np.asarray(raw[:, mask], dtype=np.float32)
    if window > 1 and packed.size:
        valid = np.isfinite(packed)
        values = uniform_filter1d(
            np.where(valid, packed, 0), window, axis=0, mode="nearest"
        )
        weights = uniform_filter1d(
            valid.astype(np.float32), window, axis=0, mode="nearest"
        )
        np.divide(values, weights, out=values, where=weights > 0)
        values[weights == 0] = np.nan
        # A missing center pixel is still missing, regardless of neighbouring frames.
        values[~valid] = np.nan
        packed = values
    return packed[start - lo : stop - lo], coords


def _contrast(
    plan: FramePlan,
    channel: int,
    window: int,
    pcts: tuple[float, float],
    max_sample_frames: int = 200,
) -> tuple[float, float]:
    """Original deterministic random 200-frame contrast sample, after smoothing."""
    n = len(plan.times)
    indexes = (
        np.random.default_rng(0).choice(n, max_sample_frames, replace=False)
        if n > max_sample_frames
        else np.arange(n)
    )
    samples = []
    for i in np.sort(indexes):
        values, _ = _packed_smooth(plan, channel, int(i), int(i + 1), window)
        samples.append(values.ravel())
    sample = np.concatenate(samples)
    sample = sample[np.isfinite(sample)]
    if not sample.size:
        raise ValueError(f"DMD{plan.dmd} channel {channel + 1} has no finite pixels")
    lo, hi = np.percentile(sample.astype(np.float64), pcts)
    return float(lo), float(hi if hi > lo else lo + 1)


def _palette(name: str | None) -> np.ndarray:
    if name is None:
        return np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    from matplotlib import colormaps

    cmap = colormaps[name]
    return cmap(np.arange(cmap.N), bytes=True)[:, :3].copy()


def _render_packed(
    values: np.ndarray,
    coords: np.ndarray,
    shape: tuple[int, int],
    vmin: float,
    vmax: float,
    palette: np.ndarray,
    rotate_k: int,
    grayscale: bool = False,
) -> np.ndarray:
    """Render one packed frame with the same 8-bit colormap lookup as Matplotlib."""
    scaled = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    valid = np.isfinite(scaled)
    scaled = np.nan_to_num(scaled)
    indices = np.minimum(
        (scaled * 255 + 0.5 if grayscale else scaled * len(palette)).astype(np.int32),
        len(palette) - 1,
    )
    rgb = np.zeros((shape[0] * shape[1], 3), dtype=np.uint8)
    rgb[coords[valid]] = palette[indices[valid]]
    return np.ascontiguousarray(np.rot90(rgb.reshape(*shape, 3), rotate_k))


def _encode_segment(
    path: str,
    plan: FramePlan,
    channel: int,
    start: int,
    stop: int,
    window: int,
    contrast: tuple[float, float],
    palette: np.ndarray,
    rotate_k: int,
    fps: float,
    codec: str,
    crf: int,
    preset: str,
    encoder_threads: int,
    grayscale: bool,
) -> dict:
    t0 = time.perf_counter()
    packed, coords = _packed_smooth(plan, channel, start, stop, window)
    height, width = plan.shape if rotate_k % 2 == 0 else plan.shape[::-1]
    pix_fmt = (
        "rgb24" if codec == "libx264rgb" else ("bgr0" if codec == "ffv1" else "yuv420p")
    )
    if pix_fmt == "yuv420p":
        width += width % 2
        height += height % 2
    with av.open(path, mode="w") as output:
        stream = output.add_stream(
            codec, rate=Fraction(str(fps)).limit_denominator(100000)
        )
        stream.width, stream.height, stream.pix_fmt = width, height, pix_fmt
        stream.codec_context.thread_count = encoder_threads
        if "264" in codec or "265" in codec:
            stream.options = dict(crf=str(crf), preset=preset)
        for i, values in enumerate(packed):
            rgb = _render_packed(
                values, coords, plan.shape, *contrast, palette, rotate_k, grayscale
            )
            if rgb.shape[:2] != (height, width):
                rgb = np.pad(
                    rgb, ((0, height - rgb.shape[0]), (0, width - rgb.shape[1]), (0, 0))
                )
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return dict(path=path, n_frames=stop - start, elapsed_s=time.perf_counter() - t0)


def _concat(paths: list[str], output: Path) -> None:
    """Remux independent keyframe segments; no decoding, re-encoding or frame loss."""
    listing = output.with_suffix(".concat.txt")
    listing.write_text(
        "".join(
            "file '" + str(Path(p).resolve()).replace("'", "'\\''") + "'\n"
            for p in paths
        )
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        listing.unlink(missing_ok=True)


def make(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    t1: float | None = None,
    t2: float | None = None,
    *,
    out_dir: str | Path | None = None,
    fps: float | None = None,
    cmaps: tuple[str | None, str | None] | None = None,
    smoothing_s: float | None = None,
    rotate_k: int = 1,
    contrast_percentiles: tuple[float, float] | dict[str, tuple[float, float]] = (
        1.0,
        99.5,
    ),
    codec: str = "libx264rgb",
    crf: int = 0,
    preset: str = "veryfast",
    workers: int = 8,
    chunk_frames: int = 512,
    encoder_threads: int = 2,
    overwrite: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Generate all DMD/channel movies and a Loupe-readable ``manifest.json``.

    ``t1/t2`` are inclusive/exclusive ephys seconds; both None selects the full
    acquisition. Timestamp sidecars preserve every original TIFF sample and
    epoch gap. The normal movie container uses nominal FPS, whereas Loupe uses
    the exact sidecar clock. No temporal downsampling or spatial resizing occurs.

    Default libx264rgb CRF 0 is pixel-exact lossless RGB encoding of the requested
    display transform (no YUV chroma subsampling). Quantization to 256 cmap
    entries, contrast clipping and requested temporal smoothing remain explicit
    display losses. ``workers`` independent chunks encode concurrently, keeping
    memory bounded rather than concatenating terabytes into acquisition arrays.

    Movie and timestamp paths inside the manifest are relative to its directory,
    so completed exports can move together. ``manifest_path`` in the returned
    dictionary identifies the written manifest. Explicit time windows default
    to ``clips/tSTART-END`` beneath the acquisition's movie directory.
    """
    started = time.perf_counter()
    if workers < 1 or chunk_frames < 1 or encoder_threads < 1:
        raise ValueError("workers, chunk_frames and encoder_threads must be positive")
    if codec not in ("libx264rgb", "libx264", "h264", "ffv1"):
        raise ValueError("Supported codecs: libx264rgb, libx264, h264, ffv1")
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be positive and finite")
    if smoothing_s is not None and (not np.isfinite(smoothing_s) or smoothing_s < 0):
        raise ValueError("smoothing_s must be nonnegative and finite")
    names = cmaps or (None, None)
    if len(names) != 2:
        raise ValueError("cmaps must contain one entry for each raw channel")
    palettes = [_palette(name) for name in names]
    plans, manifest = plan_acquisition(subject, exp, loc, acq)
    if any(p.n_channels != 2 for p in plans):
        raise ValueError(
            "Movie renderer currently expects two interleaved raw channels"
        )
    first, last = (
        min(p.times[0] for p in plans),
        max(p.times[-1] + 1 / p.rate for p in plans),
    )
    lo, hi = first if t1 is None else float(t1), last if t2 is None else float(t2)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("t1/t2 must define a finite increasing time window")
    destination = (
        Path(out_dir)
        if out_dir is not None
        else default_movie_dir(subject, exp, loc, acq)
    )
    if out_dir is None and (t1 is not None or t2 is not None):
        destination = destination / "clips" / f"t{lo:.3f}-{hi:.3f}"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    extension = ".mkv" if codec == "ffv1" else ".mp4"
    expected = [manifest_path] + [
        destination / f"DMD{p.dmd}_ch{c + 1}{extension}"
        for p in plans
        for c in range(2)
    ]
    if not overwrite and any(p.exists() for p in expected):
        raise FileExistsError(
            f"Movies already exist in {destination}; pass overwrite=True"
        )
    stage = Path(tempfile.mkdtemp(prefix=".render-", dir=destination))
    videos, jobs, results = [], [], {}
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            contrast_jobs = {}
            configs = []
            for plan in plans:
                start, stop = np.searchsorted(plan.times, [lo, hi], side="left")
                if start == stop:
                    continue
                # Subset time window still uses full source halos, so endpoints are
                # identical to the corresponding full-recording display frames.
                window = max(1, int(round((smoothing_s or 0) * plan.rate)))
                pcts = (
                    contrast_percentiles.get(f"DMD{plan.dmd}", (1.0, 99.5))
                    if isinstance(contrast_percentiles, dict)
                    else contrast_percentiles
                )
                if len(pcts) != 2 or not 0 <= pcts[0] < pcts[1] <= 100:
                    raise ValueError(
                        "Contrast percentiles must satisfy 0 <= lo < hi <= 100"
                    )
                for channel in range(2):
                    key = f"DMD{plan.dmd}_ch{channel + 1}"
                    config = (plan, channel, int(start), int(stop), window, key)
                    configs.append(config)
                    # Acquisition-wide contrast stays stable for short windows.
                    contrast_jobs[
                        pool.submit(_contrast, plan, channel, window, pcts)
                    ] = key
            if not configs:
                raise ValueError("Requested interval contains no TIFF frames")
            contrasts = {}
            for future in as_completed(contrast_jobs):
                key = contrast_jobs[future]
                contrasts[key] = future.result()
                if progress:
                    print(f"[{key}] contrast={contrasts[key]}", flush=True)
            for plan, channel, start, stop, window, key in configs:
                frame_times = plan.times[start:stop]
                np.save(stage / f"{key}.frame_times.npy", frame_times)
                selected_trials = []
                for trial in plan.trials:
                    a, b = max(start, trial.start), min(stop, trial.stop)
                    if b > a:
                        stat = Path(trial.path).stat()
                        selected_trials.append(
                            dict(
                                trial=trial.trial,
                                epoch=trial.epoch,
                                source_path=trial.path,
                                source_size=stat.st_size,
                                source_mtime_ns=stat.st_mtime_ns,
                                source_frame_start=a - trial.start,
                                source_frame_stop=b - trial.start,
                                movie_frame_start=a - start,
                                movie_frame_stop=b - start,
                                analyzed=bool(plan.analyzed_counts[trial.trial - 1]),
                            )
                        )
                signals = np.asarray(
                    manifest["channel_schema"]["signalByRawChannel"], dtype=object
                ).ravel()
                video = dict(
                    key=key,
                    name=f"DMD{plan.dmd} {signals[channel]}",
                    dmd=plan.dmd,
                    channel=channel + 1,
                    video_path=f"{key}{extension}",
                    frame_times_path=f"{key}.frame_times.npy",
                    n_frames=len(frame_times),
                    frame_time_start_s=float(frame_times[0]),
                    frame_time_end_s=float(frame_times[-1]),
                    nominal_fps=float(fps or plan.rate),
                    source_shape=list(plan.shape),
                    rotate_k=rotate_k % 4,
                    smoothing_s=smoothing_s,
                    smoothing_frames=window,
                    contrast=list(contrasts[key]),
                    colormap=names[channel],
                    trials=selected_trials,
                    gaps_after_frame=np.flatnonzero(
                        np.diff(frame_times) > 3 * np.median(np.diff(frame_times))
                    ).tolist(),
                )
                videos.append(video)
                boundaries = sorted({
                    start,
                    stop,
                    *[int(v) for v in plan.segment_edges if start < v < stop],
                })
                chunks = []
                for a, b in zip(boundaries[:-1], boundaries[1:], strict=True):
                    chunks.extend(
                        (i, min(i + chunk_frames, b)) for i in range(a, b, chunk_frames)
                    )
                paths = []
                for number, (a, b) in enumerate(chunks):
                    path = str(stage / f"{key}.{number:05d}{extension}")
                    paths.append(path)
                    args = (
                        path,
                        plan,
                        channel,
                        a,
                        b,
                        window,
                        contrasts[key],
                        palettes[channel],
                        rotate_k % 4,
                        float(fps or plan.rate),
                        codec,
                        crf,
                        preset,
                        encoder_threads,
                        names[channel] is None,
                    )
                    jobs.append((key, number, args))
                results[key] = paths
            # Interleave all streams, so a short recording also uses the full pool.
            jobs.sort(key=lambda job: job[1])
            futures = {
                pool.submit(_encode_segment, *args): (key, n) for key, n, args in jobs
            }
            done_frames = 0
            last_print = time.perf_counter()
            for future in as_completed(futures):
                result = future.result()
                done_frames += result["n_frames"]
                now = time.perf_counter()
                if progress and (
                    now - last_print > 10
                    or done_frames == sum(v["n_frames"] for v in videos)
                ):
                    print(
                        f"[movies] {done_frames:,}/"
                        f"{sum(v['n_frames'] for v in videos):,} frames; "
                        f"{now - started:.1f}s",
                        flush=True,
                    )
                    last_print = now
        for video in videos:
            key = video["key"]
            target = stage / f"{key}{extension}"
            _concat(results[key], target)
            with av.open(str(target)) as container:
                stream = container.streams.video[0]
                # Packet count checks remux output without decoding every RGB pixel.
                packet_count = sum(p.size > 0 for p in container.demux(stream))
            if packet_count != video["n_frames"]:
                raise RuntimeError(
                    f"{key}: encoded {packet_count} frames, "
                    f"expected {video['n_frames']}"
                )
            video["size_bytes"] = target.stat().st_size
        manifest.update(
            videos=videos,
            path_convention="relative_to_manifest",
            requested_time_window_s=[t1, t2],
            covered_time_window_s=[lo, hi],
            contrast_percentiles=contrast_percentiles,
            contrast_sample_frames=200,
            contrast_random_seed=0,
            codec=codec,
            crf=crf,
            preset=preset,
            lossless_display_encoding=codec == "ffv1"
            or (codec == "libx264rgb" and crf == 0),
            display_transform=(
                "All source frames retained. Temporal finite-weight box smoothing; "
                "independent fixed contrast; colormap lookup table; "
                "off-footprint/missing pixels black."
            ),
            workers=workers,
            chunk_frames=chunk_frames,
            encoder_threads=encoder_threads,
            elapsed_s=time.perf_counter() - started,
            manifest_path=str(manifest_path.resolve()),
        )
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for video in videos:
            for field in ("video_path", "frame_times_path"):
                target = destination / video[field]
                os.replace(stage / target.name, target)
        os.replace(stage / "manifest.json", manifest_path)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    if progress:
        print(
            f"[movies] ready: {manifest_path} ({manifest['elapsed_s']:.1f}s)",
            flush=True,
        )
    return manifest
