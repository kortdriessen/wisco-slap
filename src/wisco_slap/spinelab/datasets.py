"""Generate portable spine-labelling datasets from an ExperimentSummary.

A spine-labelling dataset is a self-contained folder that the standalone
``spine-labeller`` napari app (the ``spine_labeller`` repo) can open on any
machine, with no access to raw data or to this analysis environment. By
default it lives in the analysis-materials tree (see
:func:`spinelab_dataset_dir`); pass ``output_root`` to write a portable copy
elsewhere (e.g. a drive for another machine), which gets a fully qualified
folder name instead::

    <anmat_root>/<subject>/<exp>/spinelab/<loc>--<acq>/     # default
    <output_root>/<subject>--<exp>--<loc>--<acq>/           # explicit root
        metadata.json      # dataset identity + per-DMD image info + time bins
        dmd1_images.npz    # mean_stack, reference, support_mask,
        dmd2_images.npz    #   support_components, source_labels
        dmd1_overlays.npz  # act_stack, act_reference (activity overlays)
        dmd2_overlays.npz

The labeller writes ``dmd{d}_spine_annotations.json`` back into the same
folder, keyed to the dataset's ``bundle_id`` (a hash of the exact
ExperimentSummary file identity plus the binning parameter), so annotations
can never silently be applied to a different acquisition or ExSum version.

Everything is derived from the mirrored ExperimentSummary alone:

- display planes are the glutamate-channel ``perTrialMeanIMsAligned`` images
  averaged into consecutive ``bin_trials``-trial time bins (a pixel needs
  finite support in at least ``minimum_valid_trial_fraction`` of a bin's
  trials to appear). Trials the summarizer never aligned (NaN
  ``perTrialAlignmentOffsets``) are treated as NaN images — see
  :func:`_conform_aligned_trials` for the two ExSum quirks this papers over;
- the reference image is the whole-acquisition glutamate ``meanIM``;
- scan support and its connected components come from the per-trial valid
  counts;
- source labels come from the functional footprints (``get_fp_info``);
- activity overlays come from ``actIM`` and ``perTrialActIMsAligned``.

This module is deliberately untracked: no staleness flags, no audit rows.
A non-empty target folder is never touched unless ``overwrite=True``, and
even then only the files this module writes (``metadata.json``,
``dmd*_images.npz``, ``dmd*_overlays.npz``) are replaced — annotations, frame
caches and measurements saved alongside them are left alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.ndimage import label

import slap2_py as spy
from wisco_slap import defs as DEFS

DATASET_SCHEMA_VERSION = 1

#: Glob patterns of the files this module writes into a dataset folder.
#: Anything else found there (annotations, frame caches, measurements, ...)
#: belongs to the labeller / measurement pipeline and is never removed here.
GENERATED_FILE_PATTERNS: tuple[str, ...] = (
    "metadata.json",
    "dmd*_images.npz",
    "dmd*_overlays.npz",
)


def spinelab_dataset_dir(subject: str, exp: str, loc: str, acq: str) -> Path:
    """Default in-tree location of an acquisition's spine-labelling dataset.

    ``<anmat_root>/<subject>/<exp>/spinelab/<loc>--<acq>`` — sibling of the
    ``scopex`` and ``sync_block_data`` directories for that experiment.
    """
    return Path(DEFS.anmat_root) / subject / exp / "spinelab" / f"{loc}--{acq}"


def _generated_files(dataset_dir: Path) -> list[Path]:
    """Files inside ``dataset_dir`` that this module is responsible for."""
    found: list[Path] = []
    for pattern in GENERATED_FILE_PATTERNS:
        found.extend(p for p in dataset_dir.glob(pattern) if p.is_file())
    return sorted(set(found))


def _prepare_target_dir(target: Path, *, overwrite: bool) -> None:
    """Make ``target`` ready to receive a freshly generated dataset.

    - missing → created (with parents);
    - existing and empty → used as-is;
    - existing and non-empty → ``FileExistsError`` unless ``overwrite`` is
      true, in which case only the generator-owned files (see
      :data:`GENERATED_FILE_PATTERNS`) are deleted and everything else
      (annotations, frame caches, measurements, ...) is preserved.
    """
    if not target.exists():
        target.mkdir(parents=True)
        return
    if not target.is_dir():
        raise NotADirectoryError(f"Dataset target is not a directory: {target}")
    contents = sorted(p.name for p in target.iterdir())
    if not contents:
        return
    if not overwrite:
        preview = ", ".join(contents[:6]) + (", ..." if len(contents) > 6 else "")
        raise FileExistsError(
            f"Dataset folder already exists and is not empty: {target} "
            f"(contains: {preview}). Nothing was written. Pass overwrite=True "
            "to regenerate the dataset files in place (metadata.json, "
            "dmd*_images.npz, dmd*_overlays.npz); annotations, frame caches "
            "and measurements in the folder are always kept."
        )
    for path in _generated_files(target):
        path.unlink()


def _guard_annotated_rebin(target: Path, bin_trials: int) -> None:
    """Refuse to regenerate an annotated dataset with a different ``bin_trials``.

    ``bin_trials`` enters the ``bundle_id``, so regenerating at another value
    silently orphans every annotation in the folder (the labeller and the
    measurer then reject them as belonging to a different dataset). Called
    before the overwrite deletes ``metadata.json``; folders without
    annotations or without readable metadata are left to the caller.
    """
    if not any(target.glob("dmd*_spine_annotations.json")):
        return
    metadata_path = target / "metadata.json"
    if not metadata_path.exists():
        return
    try:
        existing = int(json.loads(metadata_path.read_text())["bin_trials"])
    except (OSError, ValueError, KeyError, TypeError):
        return
    if existing != int(bin_trials):
        raise ValueError(
            f"{target} holds annotations made against bin_trials={existing}; "
            f"regenerating with bin_trials={bin_trials} would change the "
            "bundle_id and detach them. Pass bin_trials="
            f"{existing} to keep the annotations attached."
        )


# ---------------------------------------------------------------------------
# Small pure helpers (unit-tested with synthetic inputs)
# ---------------------------------------------------------------------------


def _chunk_records(
    *,
    start_seconds: np.ndarray,
    end_seconds: np.ndarray,
    epochs: np.ndarray,
    frame_counts: Sequence[int],
    bin_trials: int,
) -> list[dict[str, Any]]:
    """Split trials into consecutive time bins of ``bin_trials`` trials.

    Parameters
    ----------
    start_seconds, end_seconds : np.ndarray
        Per-trial wall-clock start/end, in seconds on any common axis. The
        output ``start_seconds`` are re-referenced to the first trial's start.
    epochs : np.ndarray
        Per-trial epoch number (all ones for single-epoch acquisitions).
    frame_counts : Sequence[int]
        Per-trial frame/sample counts (informational; summed per bin).
    bin_trials : int
        Number of trials per bin. The final bin keeps the remainder and is
        marked ``partial_bin`` when it holds fewer than ``bin_trials`` trials.
    """
    starts = np.asarray(start_seconds, dtype=float).ravel()
    ends = np.asarray(end_seconds, dtype=float).ravel()
    epochs = np.asarray(epochs, dtype=int).ravel()
    n_trials = len(starts)
    if not (len(ends) == len(epochs) == len(frame_counts) == n_trials):
        raise ValueError("Per-trial arrays must have equal lengths")
    if bin_trials <= 0:
        raise ValueError("bin_trials must be positive")
    origin = starts[0]
    chunks: list[dict[str, Any]] = []
    for bin_zero, first_zero in enumerate(range(0, n_trials, bin_trials)):
        last_exclusive = min(n_trials, first_zero + bin_trials)
        selected = list(range(first_zero, last_exclusive))
        chunks.append({
            "bin_index": bin_zero + 1,
            "first_trial": first_zero + 1,
            "last_trial": last_exclusive,
            "n_trials": len(selected),
            "n_frames": int(sum(int(frame_counts[i]) for i in selected)),
            "epoch": int(epochs[first_zero]),
            "start_seconds": float(starts[first_zero] - origin),
            "end_seconds": float(ends[last_exclusive - 1] - origin),
            "duration_seconds": float(ends[last_exclusive - 1] - starts[first_zero]),
            "partial_bin": len(selected) < bin_trials,
        })
    return chunks


def _binned_aligned_means(
    per_trial: np.ndarray,
    chunks: Iterable[Mapping[str, Any]],
    *,
    minimum_valid_trial_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware per-bin means of aligned per-trial images.

    Returns the ``(n_bins, rows, cols)`` float32 mean stack (NaN where a
    pixel has finite values in fewer than
    ``ceil(minimum_valid_trial_fraction * n_trials_in_bin)`` trials) and the
    matching uint16 per-pixel valid-trial counts.
    """
    means: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    for record in chunks:
        start = int(record["first_trial"]) - 1
        stop = int(record["last_trial"])
        block = np.asarray(per_trial[start:stop], dtype=np.float32)
        finite = np.isfinite(block)
        count = finite.sum(axis=0).astype(np.uint16)
        summed = np.where(finite, block, 0).sum(axis=0, dtype=np.float64)
        mean = np.full(block.shape[1:], np.nan, dtype=np.float32)
        minimum = max(
            1,
            int(math.ceil(minimum_valid_trial_fraction * int(record["n_trials"]))),
        )
        keep = count >= minimum
        mean[keep] = (summed[keep] / count[keep]).astype(np.float32)
        means.append(mean)
        counts.append(count)
    return np.stack(means), np.stack(counts)


def _conform_aligned_trials(images: np.ndarray, registered: np.ndarray) -> np.ndarray:
    """Return ``(n_trials, ...)`` aligned images with unregistered trials as NaN.

    ``summarize_LoCo.m`` builds ``meanAligned`` by indexed assignment into an
    empty array while looping trials from last to first and ``continue``-ing
    past trials it does not align (dropped by ``verifyFiles`` or with an
    all-NaN activity image). Two consequences for the saved
    ``perTrialMeanIMsAligned``:

    1. the array stops at the last *aligned* trial, so it is shorter than
       the trial table by the trailing run of unaligned trials;
    2. interior unaligned trials hold MATLAB's zero fill, not NaN, and would
       count as finite data in :func:`_binned_aligned_means`.

    ``perTrialAlignmentOffsets`` is preallocated NaN and only written for
    aligned trials, so ``registered`` (finite offsets, one flag per trial in
    the trial table) is the reliable record. Unaligned slices that are
    present must be all-zero or all-NaN — anything else means the mask does
    not describe this array and is an error rather than something to hide.
    """
    registered = np.asarray(registered, dtype=bool).ravel()
    n_trials = int(registered.size)
    n_images = int(images.shape[0])
    if n_images > n_trials:
        raise ValueError(f"{n_images} aligned trial images for {n_trials} trials")
    if n_images < n_trials and registered[n_images:].any():
        beyond = (np.flatnonzero(registered[n_images:]) + n_images + 1).tolist()
        raise ValueError(
            f"aligned trial images stop at trial {n_images} but trials {beyond} "
            "were aligned"
        )
    for trial_zero in np.flatnonzero(~registered[:n_images]):
        block = np.asarray(images[trial_zero])
        if not np.all((block == 0) | np.isnan(block)):
            raise ValueError(
                f"trial {trial_zero + 1} has no alignment offsets but a non-empty "
                "aligned image"
            )
    out = np.full((n_trials, *images.shape[1:]), np.nan, dtype=np.float32)
    out[:n_images] = images
    out[~registered] = np.nan
    return out


def _bundle_id(exsum_path: Path, provenance: Mapping[str, Any], bin_trials: int) -> str:
    """Hash the exact ExSum file identity plus the binning parameter.

    Matches the recipe used by the original local Bellatrix prototype bundle,
    so a dataset generated from the same mirrored ExSum file carries the same
    ``bundle_id`` and annotations transfer between the two.
    """
    stat = exsum_path.stat()
    identity = {
        "path": str(exsum_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "run": provenance.get("runIdentifier"),
        "ophys_commit": provenance.get("ophysGit", {}).get("commit"),
        "bin_trials": int(bin_trials),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


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


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# ExperimentSummary readers
# ---------------------------------------------------------------------------


def _glutamate_channel_contract(exsum_path: Path) -> dict[str, Any]:
    """Identify the glutamate channel from the ExSum's channel schema.

    Requires a channel-fixed ExperimentSummary (``wisc_raw_channel_v1``
    schema with arrays saved in raw channel order); older ExSums without a
    ``channelSchema`` are rejected because their channel identity is
    ambiguous.
    """
    with h5py.File(exsum_path, "r") as handle:
        if "channelSchema" not in handle["exptSummary"]:
            raise ValueError(
                f"{exsum_path.name} has no channelSchema; regenerate the "
                "ExperimentSummary with the channel-fixed preprocessing "
                "before building a labelling dataset"
            )
    schema_array = np.asarray(
        spy.xsum.deref_h5_any("/exptSummary/channelSchema", exsum_path), dtype=object
    )
    if schema_array.size == 0 or not isinstance(schema_array.flat[0], dict):
        raise ValueError(f"{exsum_path.name} has no readable channelSchema")
    native = schema_array.flat[0]
    schema = _jsonable(native)
    if schema.get("schemaVersion") != "wisc_raw_channel_v1":
        raise ValueError(f"Unexpected channel schema: {schema.get('schemaVersion')!r}")
    if schema.get("savedTraceOrder") != "raw":
        raise ValueError(
            f"{exsum_path.name}: expected ExperimentSummary arrays in raw channel order"
        )
    raw_channels = [
        int(round(v)) for v in np.asarray(native["rawChannelNumbers"]).ravel()
    ]
    signals = [
        str(v) for v in np.asarray(native["signalByRawChannel"], dtype=object).ravel()
    ]
    indicators = [
        str(v)
        for v in np.asarray(native["indicatorByRawChannel"], dtype=object).ravel()
    ]
    try:
        glu_index = signals.index("glutamate")
    except ValueError as exc:
        raise ValueError(
            f"{exsum_path.name}: channelSchema identifies no glutamate channel"
        ) from exc
    activity_raw = schema.get("activityRawChannel")
    return {
        "channel_schema": schema,
        "glutamate_raw_channel": raw_channels[glu_index],
        "glutamate_array_index": glu_index,
        "glutamate_indicator": indicators[glu_index],
        "activity_is_glutamate": (
            activity_raw is not None
            and int(round(float(activity_raw))) == raw_channels[glu_index]
        ),
        "number_of_channels": len(raw_channels),
    }


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


def _per_trial_sample_counts(exsum_path: Path, dmd: int, n_trials: int) -> list[int]:
    """Per-trial analyzed-sample counts, from ``discardFrames`` shapes.

    Reads only HDF5 dataset shapes (no data). Trials whose per-trial group
    cannot be dereferenced (occasionally present for one DMD) count as 0.
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


def _registered_trial_mask(exsum_path: Path, dmd: int, n_trials: int) -> np.ndarray:
    """``True`` for each trial the summarizer aligned to the DMD's template.

    Reads ``perTrialAlignmentOffsets`` (preallocated NaN in
    ``summarize_LoCo.m``, written only for aligned trials), so finite
    offsets mark aligned trials. Returned in trial-table order, length
    ``n_trials``.
    """
    offsets = np.asarray(
        spy.xsum.deref_h5_any(
            f"/exptSummary/perTrialAlignmentOffsets[{dmd - 1}][0]", exsum_path
        ),
        dtype=float,
    )
    if offsets.ndim != 2:
        raise ValueError(
            f"perTrialAlignmentOffsets DMD{dmd} has unexpected shape {offsets.shape}"
        )
    if offsets.shape[0] == 2 and offsets.shape[1] != 2:
        offsets = offsets.T
    if offsets.shape != (n_trials, 2):
        raise ValueError(
            f"perTrialAlignmentOffsets DMD{dmd}: shape {offsets.shape} does not "
            f"match {n_trials} trials"
        )
    return np.isfinite(offsets).all(axis=1)


def _aligned_per_trial_images(
    exsum_path: Path, *, key: str, dmd: int, channel_index: int | None
) -> np.ndarray:
    """Load aligned per-trial images in display orientation (trials, r, c)."""
    raw = np.asarray(
        spy.xsum.deref_h5_any(f"/exptSummary/{key}[{dmd - 1}][0]", exsum_path)
    )
    if raw.ndim != 4:
        raise ValueError(f"{key} DMD{dmd} has unexpected shape {raw.shape}")
    index = 0 if channel_index is None else channel_index
    if index >= raw.shape[1]:
        raise ValueError(
            f"{key} DMD{dmd}: channel index {index} out of range for shape {raw.shape}"
        )
    return raw[:, index].transpose(0, 2, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_spine_labelling_dataset(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    output_root: Path | str | None = None,
    *,
    bin_trials: int = 18,
    minimum_valid_trial_fraction: float = 0.5,
    include_overlays: bool = True,
    exsum_path: Path | str | None = None,
    folder_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Build a portable spine-labelling dataset folder for one acquisition.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identity (e.g. ``"bellatrix", "exp_1", "loc_Z", "acq_1"``).
    output_root : Path | str, optional
        Directory to create the dataset folder in. Default ``None`` writes
        the dataset to its canonical in-tree location,
        ``<anmat_root>/<subject>/<exp>/spinelab/<loc>--<acq>`` (see
        :func:`spinelab_dataset_dir`). Give a path (any location — an
        external drive is fine) to write a portable copy elsewhere; the
        folder is then named ``<subject>--<exp>--<loc>--<acq>`` so it stays
        unambiguous outside the tree. Created if missing.
    bin_trials : int
        Trials per display time bin (18 trials of ~10 s is roughly three
        minutes — the value every in-tree dataset uses). Enters the
        ``bundle_id``, so changing it produces a dataset that existing
        annotations will refuse to attach to.
    minimum_valid_trial_fraction : float
        Fraction of a bin's trials in which a pixel must be finite to appear
        in that bin's display image.
    include_overlays : bool
        Also write the activity-overlay sidecars (``actIM`` +
        per-bin-averaged ``perTrialActIMsAligned``). Skipped with a printed
        note when the ExSum lacks activity images or its activity channel is
        not the glutamate channel.
    exsum_path : Path | str, optional
        Explicit ExperimentSummary ``.mat`` to use. Default: the current
        mirror from ``wis.meta.get.esum_mirror_path``.
    folder_name : str, optional
        Override the dataset folder name (``"<loc>--<acq>"`` in-tree,
        ``"<subject>--<exp>--<loc>--<acq>"`` under an explicit
        ``output_root``).
    overwrite : bool
        What to do when the target folder already exists and is not empty.
        ``False`` (default): raise ``FileExistsError`` without writing
        anything. ``True``: delete only the files this function produces
        (``metadata.json``, ``dmd*_images.npz``, ``dmd*_overlays.npz``) and
        regenerate them in place — saved annotations, frame caches and
        measurements are never removed. An existing *empty* folder is
        simply used, regardless of this flag.

    Returns
    -------
    Path
        The created dataset folder.

    Raises
    ------
    FileExistsError
        The target folder exists, is not empty, and ``overwrite`` is false.
    ValueError
        ``overwrite`` is true but the folder holds annotations made against
        a different ``bin_trials`` (which would change the ``bundle_id`` and
        detach them); pass the folder's existing ``bin_trials``.

    Notes
    -----
    Bin ``start/end/duration_seconds`` are seconds since the first trial on
    the scopex time axis. For multi-epoch acquisitions each epoch is anchored
    at its ``duration_since_epoch_1_s`` from ``epoch_info.yaml`` (via
    :func:`wisco_slap.meta.get.unify_epoch_times`), because the ExSum's own
    per-trial times are anchored arbitrarily per epoch; bins spanning an
    epoch gap therefore report a duration that includes the gap. The offsets
    used are recorded as ``epoch_offsets_s`` in ``metadata.json``.
    """
    import wisco_slap as wis

    if exsum_path is None:
        exsum_path = wis.meta.get.esum_mirror_path(subject, exp, loc, acq)
        if exsum_path in (0, "NO_ESUM_MIRROR"):
            raise FileNotFoundError(
                f"No ExperimentSummary mirror for {subject}/{exp}/{loc}--{acq}; "
                "run wis.meta.update() first"
            )
    exsum = Path(exsum_path).resolve()
    if not exsum.exists():
        raise FileNotFoundError(f"ExperimentSummary not found: {exsum}")

    if output_root is None:
        default_dir = spinelab_dataset_dir(subject, exp, loc, acq)
        target = default_dir.parent / (folder_name or default_dir.name)
    else:
        target = Path(output_root) / (folder_name or f"{subject}--{exp}--{loc}--{acq}")
    if overwrite and target.is_dir():
        _guard_annotated_rebin(target, bin_trials)
    _prepare_target_dir(target, overwrite=overwrite)
    print(f"[spinelab] writing dataset to {target}")

    try:
        _write_dataset(
            target,
            subject=subject,
            exp=exp,
            loc=loc,
            acq=acq,
            exsum=exsum,
            bin_trials=bin_trials,
            minimum_valid_trial_fraction=minimum_valid_trial_fraction,
            include_overlays=include_overlays,
        )
    except BaseException:
        # Don't leave a half-written dataset behind: it would make the folder
        # non-empty and block the next attempt. Only our own files go.
        for path in _generated_files(target):
            path.unlink(missing_ok=True)
        raise
    _self_check(target)
    return target


def _write_dataset(
    target: Path,
    *,
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    exsum: Path,
    bin_trials: int,
    minimum_valid_trial_fraction: float,
    include_overlays: bool,
) -> None:
    """Write ``metadata.json`` + per-DMD ``.npz`` files into ``target``."""

    import wisco_slap as wis
    contract = _glutamate_channel_contract(exsum)
    provenance = _jsonable(spy.xsum.deref_h5_any("/exptSummary/provenance", exsum))
    if not isinstance(provenance, dict):
        provenance = {}
    exsum_data, n_trials, analyze_hz = spy.xsum.get_exsum_basics(exsum)
    n_dmds = int(exsum_data["E"].shape[0])
    trial_table = spy.xsum.deref_h5_any("/exptSummary/trialTable", exsum)
    starts_days = np.asarray(trial_table["trialStartTimeInferred"], dtype=float).ravel()
    ends_days = np.asarray(trial_table["trialEndTimeFromPC"], dtype=float).ravel()
    if "epoch" in trial_table:
        epochs = np.asarray(trial_table["epoch"], dtype=int).ravel()
    else:
        epochs = np.ones(n_trials, dtype=int)
    epoch_offsets_s = wis.meta.get.epoch_offsets(subject, exp, loc, acq)
    if epoch_offsets_s is None and int(epochs.max()) > 1:
        raise KeyError(
            f"{subject}/{exp}/{loc}--{acq} is multi-epoch but has no entry in "
            "epoch_info.yaml; run wis.meta.update() first"
        )
    starts_s, ends_s = wis.meta.get.unify_epoch_times(
        starts_days * 86400.0, ends_days * 86400.0, epochs, epoch_offsets_s
    )
    bundle_id = _bundle_id(exsum, provenance, bin_trials)
    mean_images = spy.xsum.get_meanIM(exsum)
    source_maps, _ = spy.xsum.get_fp_info(exsum)
    channel_index = int(contract["glutamate_array_index"])

    metadata: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "created_at": datetime.now(UTC).isoformat(),
        "generator": "wisco_slap.spinelab.generate_spine_labelling_dataset",
        "acquisition": {
            "subject": subject,
            "experiment": exp,
            "location": loc,
            "acquisition": acq,
        },
        "exsum_path": str(exsum),
        "exsum_provenance": provenance,
        "channel_schema": contract["channel_schema"],
        "glutamate_raw_channel": contract["glutamate_raw_channel"],
        "glutamate_array_index": channel_index,
        "glutamate_indicator": contract["glutamate_indicator"],
        "number_of_channels": contract["number_of_channels"],
        "n_trials": int(n_trials),
        "analyze_hz": float(analyze_hz),
        "n_frames_hz": float(analyze_hz),
        "bin_trials": int(bin_trials),
        "minimum_valid_trial_fraction": float(minimum_valid_trial_fraction),
        "n_epochs": int(epochs.max()),
        "epoch_offsets_s": (
            None
            if epoch_offsets_s is None
            else {str(k): float(v) for k, v in sorted(epoch_offsets_s.items())}
        ),
        "time_axis": (
            "seconds since first trial on the scopex axis; multi-epoch "
            "acquisitions anchored per epoch by epoch_info duration_since_epoch_1_s"
        ),
        "dmds": {},
    }

    overlay_ok = bool(include_overlays and contract["activity_is_glutamate"])
    if include_overlays and not contract["activity_is_glutamate"]:
        print(
            f"[spinelab] {subject}/{exp}/{loc}--{acq}: activity images are "
            "not from the glutamate channel; skipping overlays"
        )

    for dmd in range(1, n_dmds + 1):
        frame_counts = _per_trial_sample_counts(exsum, dmd, int(n_trials))
        chunks = _chunk_records(
            start_seconds=starts_s,
            end_seconds=ends_s,
            epochs=epochs,
            frame_counts=frame_counts,
            bin_trials=bin_trials,
        )
        registered = _registered_trial_mask(exsum, dmd, int(n_trials))
        unregistered_trials = (np.flatnonzero(~registered) + 1).tolist()
        if unregistered_trials:
            print(
                f"[spinelab] {subject}/{exp}/{loc}--{acq} DMD{dmd}: "
                f"{len(unregistered_trials)} of {n_trials} trials were not "
                "aligned by the summarizer; treating them as NaN images"
            )
        try:
            per_trial = _conform_aligned_trials(
                _aligned_per_trial_images(
                    exsum,
                    key="perTrialMeanIMsAligned",
                    dmd=dmd,
                    channel_index=channel_index,
                ),
                registered,
            )
        except ValueError as exc:
            raise ValueError(f"DMD{dmd}: {exc}") from exc
        stack, valid_trial_counts = _binned_aligned_means(
            per_trial,
            chunks,
            minimum_valid_trial_fraction=minimum_valid_trial_fraction,
        )
        reference = np.asarray(mean_images[dmd][channel_index], dtype=np.float32)
        if reference.shape != stack.shape[1:]:
            raise ValueError(
                f"DMD{dmd}: meanIM shape {reference.shape} does not match "
                f"aligned trial images {stack.shape[1:]}"
            )
        support = np.any(valid_trial_counts > 0, axis=0)
        components, n_components = label(support, structure=np.ones((3, 3), dtype=int))
        source_labels = np.asarray(source_maps[dmd], dtype=np.int32)
        if source_labels.shape != reference.shape:
            raise ValueError(f"DMD{dmd} source map and reference shapes differ")
        _atomic_npz(
            target / f"dmd{dmd}_images.npz",
            mean_stack=stack,
            valid_trial_counts=valid_trial_counts,
            reference=reference,
            support_mask=support.astype(bool),
            support_components=components.astype(np.int32),
            source_labels=source_labels,
        )
        metadata["dmds"][str(dmd)] = {
            "image_shape": [int(v) for v in reference.shape],
            "pixel_size_um": _pixel_size_um(trial_table, dmd),
            "support_pixel_count": int(support.sum()),
            "support_component_count": int(n_components),
            "n_registered_trials": int(registered.sum()),
            "unregistered_trials": unregistered_trials,
            "chunks": chunks,
        }
        del per_trial, stack, valid_trial_counts

        if overlay_ok:
            try:
                act_reference = np.asarray(
                    spy.xsum.get_actIM(exsum)[dmd], dtype=np.float32
                )
                act_per_trial = _conform_aligned_trials(
                    _aligned_per_trial_images(
                        exsum,
                        key="perTrialActIMsAligned",
                        dmd=dmd,
                        channel_index=None,
                    ),
                    registered,
                )
                act_stack, _ = _binned_aligned_means(
                    act_per_trial,
                    chunks,
                    minimum_valid_trial_fraction=minimum_valid_trial_fraction,
                )
                if act_reference.shape != reference.shape:
                    raise ValueError(
                        f"actIM shape {act_reference.shape} != {reference.shape}"
                    )
                _atomic_npz(
                    target / f"dmd{dmd}_overlays.npz",
                    act_stack=act_stack,
                    act_reference=act_reference,
                )
                del act_per_trial, act_stack
            except (KeyError, ValueError, OSError) as exc:
                print(
                    f"[spinelab] {subject}/{exp}/{loc}--{acq} DMD{dmd}: "
                    f"skipping overlays ({exc})"
                )

    _atomic_json(metadata, target / "metadata.json")


def _self_check(dataset_dir: Path) -> None:
    """Open the finished folder with the labeller's loader, if installed."""
    try:
        from spine_labeller.data import available_dmds, load_dmd_images
    except ImportError:
        return
    dmds = available_dmds(dataset_dir)
    if not dmds:
        raise RuntimeError(f"Self-check failed: {dataset_dir} lists no DMDs")
    for dmd in dmds:
        load_dmd_images(dataset_dir, dmd)
    print(f"[spinelab] dataset verified by spine_labeller loader: DMDs {dmds}")
