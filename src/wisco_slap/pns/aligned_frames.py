"""Aligned pixel-frame caches from registered TIFFs, for any channel.

Shared low-level machinery for pipelines that need per-frame pixel values in
the common cross-trial-aligned image space (the spine-size measurement in
``wisco_slap.spinelab`` and the shaft-calcium pipeline in
``wisco_slap.pns.shaft_ca_gen``): reading channel pages from the registered
80 Hz TIFFs, applying the ExperimentSummary's per-trial alignment offsets,
and caching the sampled pixels as a resumable zarr with per-frame quality
flags (preprocessing discard masks + registration-failure sidecars).

Everything operates on an :class:`AcqContext`, built either from a portable
spine-labelling dataset folder (:func:`context_from_dataset`) or directly
from a mirrored ExperimentSummary (:func:`context_from_exsum`).  New caches
are validated by reconstructing per-trial means and comparing them with the
ExperimentSummary's own aligned per-trial images, an independent path through
the same raw data that catches channel, orientation, and offset errors.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tifffile
import zarr
from numcodecs import Blosc
from scipy.ndimage import map_coordinates

import slap2_py as spy
from wisco_slap import defs

FRAME_CACHE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Acquisition context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcqContext:
    """Everything a frame-cache build needs to know about one acquisition."""

    exsum_path: Path
    identity: str
    acquisition: dict[str, str]
    channel_schema: dict[str, Any]
    number_of_channels: int
    n_trials: int
    analyze_hz: float
    dmds: tuple[int, ...]
    support: dict[int, np.ndarray] = field(repr=False)
    pixel_size_um: dict[int, float] = field(default_factory=dict)

    def image_shape(self, dmd: int) -> tuple[int, int]:
        shape = self.support[int(dmd)].shape
        return (int(shape[0]), int(shape[1]))


def file_identity(
    exsum_path: Path, provenance: Mapping[str, Any], salt: int
) -> str:
    """Hash the exact ExSum file identity (plus an integer salt).

    The recipe is shared with spine-labelling dataset ``bundle_id`` values
    (where the salt is ``bin_trials``), so a dataset generated from the same
    mirrored ExSum file carries the same identity.
    """

    stat = exsum_path.stat()
    payload = {
        "path": str(exsum_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "run": provenance.get("runIdentifier"),
        "ophys_commit": provenance.get("ophysGit", {}).get("commit"),
        "bin_trials": int(salt),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def signal_channel(
    channel_schema: Mapping[str, Any], signal: str = "glutamate"
) -> dict[str, int | str]:
    """Resolve a signal name ("glutamate"/"calcium") to its channel indices."""

    signals = [
        str(v)
        for v in np.asarray(channel_schema["signalByRawChannel"], dtype=object).ravel()
    ]
    raw_channels = [
        int(round(float(v)))
        for v in np.asarray(channel_schema["rawChannelNumbers"], dtype=float).ravel()
    ]
    if signal not in signals:
        raise ValueError(
            f"channel schema has no {signal!r} channel (signals: {signals})"
        )
    index = signals.index(signal)
    return {
        "signal": signal,
        "raw_channel": raw_channels[index],
        "array_index": index,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _jsonable(value.item())
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def read_channel_schema(exsum_path: Path | str) -> dict[str, Any]:
    """Read and JSON-ify the corrected channel schema; reject its absence."""

    exsum_path = Path(exsum_path)
    with h5py.File(exsum_path, "r") as handle:
        if "channelSchema" not in handle["exptSummary"]:
            raise ValueError(
                f"{exsum_path.name} has no channelSchema; regenerate the "
                "ExperimentSummary with the channel-fixed preprocessing first"
            )
    schema_array = np.asarray(
        spy.xsum.deref_h5_any("/exptSummary/channelSchema", exsum_path), dtype=object
    )
    if schema_array.size == 0 or not isinstance(schema_array.flat[0], dict):
        raise ValueError(f"{exsum_path.name} has no readable channelSchema")
    schema = _jsonable(schema_array.flat[0])
    if schema.get("schemaVersion") != "wisc_raw_channel_v1":
        raise ValueError(f"Unexpected channel schema: {schema.get('schemaVersion')!r}")
    if schema.get("savedTraceOrder") != "raw":
        raise ValueError(
            f"{exsum_path.name}: expected arrays in raw channel order"
        )
    return schema


def context_from_exsum(exsum_path: Path | str) -> AcqContext:
    """Build a context straight from a mirrored ExperimentSummary.

    Scan support per DMD comes from the finite pixels of the mean images;
    the acquisition identity is parsed from the provenance's
    ``acquisitionIdentifier`` (``subject--exp--loc--acq``).
    """

    exsum = Path(exsum_path).resolve()
    schema = read_channel_schema(exsum)
    provenance = _jsonable(spy.xsum.deref_h5_any("/exptSummary/provenance", exsum))
    if not isinstance(provenance, dict):
        provenance = {}
    parts = str(provenance.get("acquisitionIdentifier", "")).split("--")
    if len(parts) != 4:
        raise ValueError(
            f"{exsum.name}: provenance.acquisitionIdentifier does not parse "
            f"as subject--exp--loc--acq ({parts!r})"
        )
    acquisition = dict(
        zip(("subject", "experiment", "location", "acquisition"), parts, strict=True)
    )
    _, n_trials, analyze_hz = spy.xsum.get_exsum_basics(exsum)
    trial_table = spy.xsum.deref_h5_any("/exptSummary/trialTable", exsum)
    mean_images = spy.xsum.get_meanIM(exsum)
    support: dict[int, np.ndarray] = {}
    pixel_sizes: dict[int, float] = {}
    for dmd in sorted(mean_images):
        images = np.asarray(mean_images[dmd])
        support[int(dmd)] = np.any(np.isfinite(images), axis=0)
        pixel_sizes[int(dmd)] = _pixel_size_um(trial_table, int(dmd))
    n_raw = len(
        np.asarray(schema["rawChannelNumbers"], dtype=float).ravel()
    )
    return AcqContext(
        exsum_path=exsum,
        identity=file_identity(exsum, provenance, salt=0),
        acquisition=acquisition,
        channel_schema=schema,
        number_of_channels=n_raw,
        n_trials=int(n_trials),
        analyze_hz=float(analyze_hz),
        dmds=tuple(sorted(int(d) for d in mean_images)),
        support=support,
        pixel_size_um=pixel_sizes,
    )


def context_from_dataset(data_dir: Path | str) -> AcqContext:
    """Build a context from a spine-labelling dataset folder.

    Verifies that the ExperimentSummary behind the dataset still hashes to
    the dataset's ``bundle_id`` (i.e. was not regenerated since), and uses
    the dataset's stored support masks so cache pixel sets line up with the
    labeller's images exactly.
    """

    from spine_labeller.data import load_dataset_metadata, load_dmd_images

    directory = Path(data_dir)
    metadata = load_dataset_metadata(directory)
    exsum = Path(metadata["exsum_path"])
    if not exsum.exists():
        raise FileNotFoundError(
            f"ExperimentSummary behind this dataset not found: {exsum}"
        )
    derived = file_identity(
        exsum,
        metadata.get("exsum_provenance", {}),
        salt=int(metadata["bin_trials"]),
    )
    if derived != metadata["bundle_id"]:
        raise ValueError(
            f"{exsum.name} no longer matches this dataset's bundle_id (the "
            "ExperimentSummary changed since the dataset was generated); "
            "regenerate the dataset and re-check the annotations"
        )
    dmds = sorted(int(k) for k in metadata["dmds"])
    support = {
        dmd: load_dmd_images(directory, dmd)["support_mask"].astype(bool)
        for dmd in dmds
    }
    pixel_sizes = {
        dmd: float(metadata["dmds"][str(dmd)]["pixel_size_um"]) for dmd in dmds
    }
    return AcqContext(
        exsum_path=exsum.resolve(),
        identity=str(metadata["bundle_id"]),
        acquisition={
            str(k): str(v) for k, v in metadata.get("acquisition", {}).items()
        },
        channel_schema=metadata["channel_schema"],
        number_of_channels=int(metadata["number_of_channels"]),
        n_trials=int(metadata["n_trials"]),
        analyze_hz=float(metadata["analyze_hz"]),
        dmds=tuple(dmds),
        support=support,
        pixel_size_um=pixel_sizes,
    )


# ---------------------------------------------------------------------------
# ExperimentSummary / raw-data primitives
# ---------------------------------------------------------------------------


def _pixel_size_um(trial_table: Mapping[str, Any], dmd: int) -> float:
    """Pixel size from the DMD's pixel-to-sample transform (isotropy checked)."""

    reference = np.asarray(trial_table["refStack"], dtype=object).flat[dmd - 1]
    transform = np.asarray(reference["dmdPixel2SampleTransform"], dtype=float)
    row_step = float(np.linalg.norm(transform[0, :2]))
    col_step = float(np.linalg.norm(transform[1, :2]))
    if not np.isclose(row_step, col_step, atol=1e-6):
        raise ValueError(
            f"DMD{dmd} has anisotropic pixels ({row_step}, {col_step}) micrometers"
        )
    return float((row_step + col_step) / 2)


def per_trial_sample_counts(
    exsum_path: Path, dmd: int, n_trials: int
) -> list[int]:
    """Per-trial analyzed-sample counts, from ``discardFrames`` shapes.

    Reads only HDF5 dataset shapes (no data). Trials whose per-trial group
    cannot be dereferenced (dropped by preprocessing) count as 0.
    """

    counts: list[int] = []
    with h5py.File(exsum_path, "r") as handle:
        references = handle["exptSummary/E"][dmd - 1]
        for trial in range(n_trials):
            try:
                group = handle[references[trial]]
                counts.append(int(np.prod(group["discardFrames"].shape)))
            except (KeyError, ValueError, OSError):
                counts.append(0)
    return counts


def trial_epochs(exsum_path: Path | str, n_trials: int) -> np.ndarray:
    """Per-trial 1-based epoch indices (all ones for single-epoch acqs)."""

    exsum = Path(exsum_path)
    trial_table = spy.xsum.deref_h5_any("/exptSummary/trialTable", exsum)
    if "epoch" in trial_table:
        epochs = np.asarray(trial_table["epoch"], dtype=int).ravel()
        if len(epochs) == n_trials:
            return epochs
    return np.ones(n_trials, dtype=int)


def resolve_raw_dir(
    acquisition: Mapping[str, str],
    first_filename: str,
    raw_dir: Path | str | None = None,
) -> Path:
    """Find the raw acquisition directory holding the registered TIFFs."""

    relative = Path(
        str(acquisition.get("subject", "")),
        str(acquisition.get("experiment", "")),
        str(acquisition.get("location", "")),
        str(acquisition.get("acquisition", "")),
    )
    candidates = (
        [Path(raw_dir)]
        if raw_dir is not None
        else [
            Path(defs.raw_mirror_root) / relative,
            Path(defs.data_root) / relative,
        ]
    )
    for candidate in candidates:
        if (candidate / first_filename).exists():
            return candidate
    raise FileNotFoundError(
        f"Registered TIFF {first_filename!r} not found in any of: "
        + ", ".join(str(c) for c in candidates)
    )


def frame_count(path: Path, number_of_channels: int) -> int:
    with tifffile.TiffFile(path) as tiff:
        page_count = len(tiff.pages)
    if page_count % number_of_channels:
        raise ValueError(
            f"{path.name} has {page_count} pages, not divisible by "
            f"{number_of_channels}"
        )
    return page_count // number_of_channels


def _read_channel_stack(
    path: Path, *, page_offset: int, number_of_channels: int
) -> np.ndarray:
    """Read one channel's physical pages and return display orientation."""

    with tifffile.TiffFile(path) as tiff:
        page_indices = range(page_offset, len(tiff.pages), number_of_channels)
        stack = tiff.asarray(key=page_indices)
    return np.asarray(stack).transpose(0, 2, 1)


def load_discard_mask(
    exsum_path: Path, dmd: int, trial_zero: int, target_frames: int
) -> np.ndarray:
    """Nearest-neighbor resample the ExSum discard mask to the TIFF frames.

    Trials the preprocessing dropped are stored as MATLAB-empty ``[0, 0]``
    placeholder datasets instead of per-trial struct groups; those (and any
    entry without a readable ``discardFrames``) come back fully discarded so
    the whole trial is excluded from measurement.
    """

    with h5py.File(exsum_path, "r") as handle:
        references = handle["exptSummary/E"][dmd - 1]
        reference = references[trial_zero]
        group = handle[reference]
        if not isinstance(group, h5py.Group) or "discardFrames" not in group:
            return np.ones(target_frames, dtype=bool)
        saved = np.asarray(group["discardFrames"], dtype=float).ravel() > 0
    if saved.size == target_frames:
        return saved
    source_positions = np.linspace(0.0, 1.0, saved.size)
    target_positions = np.linspace(0.0, 1.0, target_frames)
    nearest = np.searchsorted(source_positions, target_positions, side="left")
    nearest = np.clip(nearest, 0, saved.size - 1)
    previous = np.maximum(nearest - 1, 0)
    choose_previous = np.abs(source_positions[previous] - target_positions) < np.abs(
        source_positions[nearest] - target_positions
    )
    nearest[choose_previous] = previous[choose_previous]
    return saved[nearest]


def registration_failed(sidecar_path: Path) -> bool:
    """Read the per-trial registration flag; unreadable counts as failed."""

    try:
        with h5py.File(sidecar_path, "r") as handle:
            return bool(np.asarray(handle["aData/registrationFailed"]).ravel()[0])
    except (OSError, KeyError, ValueError, IndexError):
        return True


def _sample_trial_frames(
    tiff_path: str,
    expected: int,
    page_offset: int,
    number_of_channels: int,
    rows: np.ndarray,
    cols: np.ndarray,
    frame_block: int,
) -> np.ndarray:
    """Read one trial's channel pages and sample the cached pixels, aligned.

    Pure function of its arguments (no zarr, no HDF5), so it can run in a
    worker process; the caller writes the returned ``(frames, pixels)`` block.
    """

    stack = _read_channel_stack(
        Path(tiff_path),
        page_offset=page_offset,
        number_of_channels=number_of_channels,
    )
    if stack.shape[0] != expected:
        raise ValueError(
            f"{Path(tiff_path).name}: expected {expected} frames, read "
            f"{stack.shape[0]}"
        )
    out = np.empty((expected, len(rows)), dtype=np.float32)
    for block_start in range(0, expected, frame_block):
        block_stop = min(expected, block_start + frame_block)
        out[block_start:block_stop] = np.stack([
            map_coordinates(
                frame,
                [rows, cols],
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
            for frame in stack[block_start:block_stop]
        ]).astype(np.float32, copy=False)
    return out


def _sample_trial_frames_star(args: tuple) -> np.ndarray:
    return _sample_trial_frames(*args)


def _finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    """Compute a NaN-ignoring mean without emitting all-NaN warnings."""

    array = np.asarray(values)
    finite = np.isfinite(array)
    counts = finite.sum(axis=axis)
    sums = np.where(finite, array, 0).sum(axis=axis, dtype=np.float64)
    output = np.full(counts.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=output, where=counts > 0)
    return output


# ---------------------------------------------------------------------------
# Cache build / validate
# ---------------------------------------------------------------------------


def build_frame_cache(
    context: AcqContext,
    dmd: int,
    cache_path: Path | str,
    *,
    pixel_mask: np.ndarray | None = None,
    raw_dir: Path | str | None = None,
    max_trials: int | None = None,
    overwrite: bool = False,
    frame_block: int = 64,
    signal: str = "glutamate",
    workers: int | None = None,
) -> Path:
    """Build (or resume) an aligned frame cache for one DMD and one channel.

    Parameters
    ----------
    pixel_mask : np.ndarray, optional
        Boolean mask of pixels to cache (intersected with scan support).
        Default: the full scan support.
    max_trials : int, optional
        Cache only the first N trials (smoke-test mode).
    workers : int, optional
        Worker processes for the TIFF-read/sampling stage (default
        ``min(16, cpu_count)``); pass 1 to run sequentially.

    The cache is resumable: rerunning after an interruption continues from
    the last completed trial.
    """

    dmd = int(dmd)
    exsum = context.exsum_path
    support = context.support[dmd]
    if pixel_mask is None:
        selection = support
        pixel_selection = "full_support"
    else:
        selection = np.asarray(pixel_mask, dtype=bool) & support
        pixel_selection = "annotation_relevant"
    if not selection.any():
        raise ValueError("No pixels selected for the frame cache")
    coords = np.column_stack(np.where(selection)).astype(np.int32)

    total_trials = int(context.n_trials)
    use_trials = (
        total_trials if max_trials is None else min(int(max_trials), total_trials)
    )
    if use_trials <= 0:
        raise ValueError("max_trials must be positive")
    target = Path(cache_path)
    channel = signal_channel(context.channel_schema, signal)
    page_offset = int(channel["raw_channel"]) - 1

    trial_table = spy.xsum.deref_h5_any("/exptSummary/trialTable", exsum)
    filenames = np.asarray(trial_table["fnRegDS"], dtype=object)
    sidecars = np.asarray(trial_table["fnAdata"], dtype=object)
    starts = np.asarray(trial_table["trialStartTimeInferred"], dtype=float).ravel()
    offsets = np.asarray(
        spy.xsum.deref_h5_any(
            f"/exptSummary/perTrialAlignmentOffsets[{dmd - 1}][0]", exsum
        ),
        dtype=float,
    )
    raw = resolve_raw_dir(context.acquisition, str(filenames[0, dmd - 1]), raw_dir)
    frame_counts = [
        frame_count(raw / str(filenames[t, dmd - 1]), context.number_of_channels)
        for t in range(use_trials)
    ]
    total_frames = int(sum(frame_counts))
    analyze_frames = sum(per_trial_sample_counts(exsum, dmd, total_trials))
    all_tiff_frames = (
        total_frames
        if use_trials == total_trials
        else int(round(total_frames / max(1, use_trials) * total_trials))
    )
    sample_rate_hz = context.analyze_hz * all_tiff_frames / max(1, analyze_frames)

    resume_from = 0
    if target.exists() and not overwrite:
        root = zarr.open_group(str(target), mode="a")
        if root.attrs.get("bundle_id") != context.identity:
            raise ValueError("Existing frame cache belongs to a different dataset")
        if int(root.attrs.get("dmd", -1)) != dmd:
            raise ValueError("Existing frame cache belongs to a different DMD")
        if str(root.attrs.get("signal", "glutamate")) != signal:
            raise ValueError("Existing frame cache holds a different channel")
        if int(root.attrs.get("n_trials", -1)) != use_trials:
            raise ValueError("Existing frame cache has a different trial count")
        if not np.array_equal(np.asarray(root["coords_rc"]), coords):
            raise ValueError(
                "Existing frame cache covers a different pixel set; pass "
                "overwrite=True to rebuild it"
            )
        if root["values"].shape != (total_frames, len(coords)):
            raise ValueError("Existing frame cache has an incompatible array shape")
        if bool(root.attrs.get("complete", False)):
            return target
        resume_from = int(root.attrs.get("processed_trials", 0))
    else:
        if target.exists():
            shutil.rmtree(target)
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        root = zarr.open_group(str(target), mode="w")
        root.create_dataset(
            "values",
            shape=(total_frames, len(coords)),
            chunks=(min(128, total_frames), min(4096, max(1, len(coords)))),
            dtype="f4",
            fill_value=np.nan,
            compressor=compressor,
        )
        root.create_dataset("coords_rc", data=coords, compressor=compressor)
        root.create_dataset(
            "quality_valid", shape=(total_frames,), chunks=(8192,), dtype="bool"
        )
        root.create_dataset(
            "trial_index", shape=(total_frames,), chunks=(8192,), dtype="i4"
        )
        root.create_dataset(
            "trial_frame_index", shape=(total_frames,), chunks=(8192,), dtype="i4"
        )
        root.create_dataset(
            "time_seconds", shape=(total_frames,), chunks=(8192,), dtype="f8"
        )
        root.attrs.update({
            "schema_version": FRAME_CACHE_SCHEMA_VERSION,
            "bundle_id": context.identity,
            "created_at": datetime.now(UTC).isoformat(),
            "dmd": dmd,
            "n_trials": use_trials,
            "processed_trials": 0,
            "complete": False,
            "full_acquisition": use_trials == total_trials,
            "sample_rate_hz": sample_rate_hz,
            "signal": signal,
            "raw_channel": int(channel["raw_channel"]),
            "tiff_page_offset": page_offset,
            "alignment": "registered TIFF plus ExSum perTrialAlignmentOffsets",
            "pixel_selection": pixel_selection,
            "raw_dir": str(raw),
        })

    values = root["values"]
    quality = root["quality_valid"]
    trial_index = root["trial_index"]
    trial_frame_index = root["trial_frame_index"]
    time_seconds = root["time_seconds"]
    cursor = int(sum(frame_counts[:resume_from]))
    time_origin = starts[0]
    remaining = list(range(resume_from, use_trials))

    def _write_trial(trial_zero: int, sampled: np.ndarray) -> None:
        nonlocal cursor
        expected = frame_counts[trial_zero]
        values[cursor : cursor + expected, :] = sampled
        discard = load_discard_mask(exsum, dmd, trial_zero, expected)
        valid = ~discard
        sidecar_path = raw / str(sidecars[trial_zero, dmd - 1])
        if valid.any() and registration_failed(sidecar_path):
            valid[:] = False
        if not valid.any():
            print(
                f"[aligned_frames] DMD{dmd}: trial {trial_zero + 1} excluded "
                "(dropped by preprocessing or failed registration)",
                flush=True,
            )
        destination = slice(cursor, cursor + expected)
        quality[destination] = valid
        trial_index[destination] = trial_zero + 1
        trial_frame_index[destination] = np.arange(expected, dtype=np.int32)
        trial_start_seconds = float((starts[trial_zero] - time_origin) * 86400)
        time_seconds[destination] = (
            trial_start_seconds + np.arange(expected) / sample_rate_hz
        )
        cursor += expected
        root.attrs["processed_trials"] = trial_zero + 1
        if trial_zero == resume_from or (trial_zero + 1) % 10 == 0:
            print(
                f"[aligned_frames] DMD{dmd}: cached trial "
                f"{trial_zero + 1}/{use_trials}",
                flush=True,
            )

    def _task(trial_zero: int) -> tuple:
        return (
            str(raw / str(filenames[trial_zero, dmd - 1])),
            frame_counts[trial_zero],
            page_offset,
            context.number_of_channels,
            coords[:, 0].astype(float) + offsets[trial_zero, 0],
            coords[:, 1].astype(float) + offsets[trial_zero, 1],
            frame_block,
        )

    if workers is None:
        workers = min(16, os.cpu_count() or 1)
    if workers <= 1 or len(remaining) <= 1:
        for trial_zero in remaining:
            _write_trial(trial_zero, _sample_trial_frames(*_task(trial_zero)))
    else:
        # Workers do the expensive TIFF read + aligned sampling; this process
        # consumes results in trial order and does every write, so resume
        # bookkeeping and zarr chunk boundaries need no synchronization.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(
                _sample_trial_frames_star,
                (_task(t) for t in remaining),
                chunksize=1,
            )
            for trial_zero, sampled in zip(remaining, results, strict=True):
                _write_trial(trial_zero, sampled)
    root.attrs["complete"] = True
    return target


def validate_frame_cache(
    context: AcqContext,
    dmd: int,
    cache_path: Path | str,
) -> dict[str, Any]:
    """Compare cached trial means to the ExSum's own aligned trial means.

    Reconstructs the first, middle, and last quality-bearing cached trials'
    mean images from the cache and correlates them with
    ``perTrialMeanIMsAligned`` — an independent path through the same raw
    data that catches channel, orientation, and alignment-offset errors.

    The reconstruction averages ALL cached frames of the trial, not just
    quality-valid ones, because ``perTrialMeanIMsAligned`` itself includes
    discarded frames. Filtering here would make the comparison
    apples-to-oranges: the subset mismatch scales with trial brightness and
    can push the absolute MAE over threshold on bright early trials even
    when alignment/channel identity are perfect (seen on alnilam exp_1
    trial 1, r=0.99999 but MAE 0.118). Candidate trials are still drawn
    from quality-bearing trials only, so dropped/registration-failed trials
    never anchor the check.
    """

    exsum = context.exsum_path
    cache = zarr.open_group(str(Path(cache_path)), mode="r")
    if cache.attrs.get("bundle_id") != context.identity:
        raise ValueError("Frame cache does not belong to this acquisition context")
    if int(cache.attrs.get("dmd", -1)) != int(dmd):
        raise ValueError("Frame cache DMD does not match requested DMD")
    cached_trial_count = int(cache.attrs["n_trials"])
    coordinates = np.asarray(cache["coords_rc"])
    quality = np.asarray(cache["quality_valid"], dtype=bool)
    trial_index = np.asarray(cache["trial_index"], dtype=int)
    quality_trials = np.unique(trial_index[quality])
    if quality_trials.size == 0:
        return {
            "bundle_id": context.identity,
            "cache_path": str(cache_path),
            "dmd": int(dmd),
            "cached_trials": cached_trial_count,
            "complete": bool(cache.attrs.get("complete", False)),
            "trial_checks": [],
            "passed": False,
            "note": "no quality-valid frames in the cache",
        }
    trial_candidates = sorted({
        int(quality_trials[0]),
        int(quality_trials[len(quality_trials) // 2]),
        int(quality_trials[-1]),
    })
    signal = str(cache.attrs.get("signal", "glutamate"))
    channel = signal_channel(context.channel_schema, signal)
    per_trial = np.asarray(
        spy.xsum.deref_h5_any(
            f"/exptSummary/perTrialMeanIMsAligned[{int(dmd) - 1}][0]", exsum
        )
    )[:, int(channel["array_index"])]
    minimum_overlap = min(1000, max(1, len(coordinates) // 2))
    records = []
    for trial in trial_candidates:
        indices = np.flatnonzero(trial_index == trial)
        if indices.size == 0:
            continue
        block = np.asarray(cache["values"][indices[0] : indices[-1] + 1])
        block_quality = quality[indices]
        cached_mean = _finite_mean(block, axis=0)
        expected_image = per_trial[trial - 1].T
        expected = expected_image[coordinates[:, 0], coordinates[:, 1]]
        finite = np.isfinite(cached_mean) & np.isfinite(expected)
        if finite.sum() >= 2:
            correlation = float(
                np.corrcoef(cached_mean[finite], expected[finite])[0, 1]
            )
            mean_absolute_error = float(
                np.mean(np.abs(cached_mean[finite] - expected[finite]))
            )
        else:
            correlation, mean_absolute_error = float("nan"), float("nan")
        records.append(
            {
                "trial": trial,
                "acquired_frames": int(indices.size),
                "quality_valid_frames": int(block_quality.sum()),
                "overlap_pixels": int(finite.sum()),
                "pearson_r": correlation,
                "mean_absolute_error": mean_absolute_error,
                "passed": bool(
                    finite.sum() >= minimum_overlap
                    and correlation >= 0.999
                    and mean_absolute_error <= 0.1
                ),
            }
        )
    return {
        "bundle_id": context.identity,
        "cache_path": str(cache_path),
        "dmd": int(dmd),
        "cached_trials": cached_trial_count,
        "complete": bool(cache.attrs.get("complete", False)),
        "trial_checks": records,
        "passed": bool(records and all(record["passed"] for record in records)),
    }
