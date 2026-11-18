"""Inactive-frame spine-size measurement for labelled datasets.

Runs the Wright-style measurement against a spine-labelling dataset folder
(see :mod:`wisco_slap.spinelab.datasets`) once it contains annotations saved
by the ``spine-labeller`` app. This is the acquisition-machine side of the
loop: it needs the raw registered TIFFs and the ExperimentSummary that the
dataset was generated from, neither of which travels with the folder.

The aligned frame-cache machinery itself is shared project infrastructure in
:mod:`wisco_slap.pns.aligned_frames`; this module wraps it with the dataset
folder's identity/paths and adds the spine measurement on top:

1. an aligned frame cache is built (or reused) over every annotation-relevant
   pixel (heads, backgrounds, shaft disks, plus a safety margin), and new
   caches are automatically validated against the ExperimentSummary's own
   aligned per-trial images;
2. for each spine and time bin, a head-minus-background trace is formed,
   active frames are detected (:func:`wisco_slap.spinelab.activity.
   detect_active_frames`) and excluded, and the Wright optical-area
   estimators are evaluated on the inactive-frame mean projection — in two
   modes: the literal first-10,000-acquired-frames rule and the full bin;
3. tidy Parquet/CSV tables, per-spine QC montage figures, and a provenance
   JSON are written into ``<data_dir>/measurements/``.

The dataset's ``bundle_id`` is re-derived from the ExperimentSummary file
before any measurement, so a dataset whose ExSum has since been regenerated
is refused rather than silently measured against the wrong alignment.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import zarr
from scipy.ndimage import binary_dilation
from spine_labeller.core import (
    CompiledSpine,
    add_relative_measurements,
    load_annotations,
    measure_projection,
)
from spine_labeller.core import compile_spines as _compile_spines
from spine_labeller.data import (
    annotations_path as default_annotations_path,
)
from spine_labeller.data import (
    load_dataset_metadata,
    load_dmd_images,
    validate_annotations_for_dataset,
)

from wisco_slap.pns import aligned_frames as _frames
from wisco_slap.pns.aligned_frames import (  # noqa: F401  (re-exported API)
    FRAME_CACHE_SCHEMA_VERSION,
    _finite_mean,
    context_from_dataset,
)
from wisco_slap.pns.aligned_frames import (
    frame_count as _frame_count,  # noqa: F401  (re-exported API)
)
from wisco_slap.pns.aligned_frames import (
    load_discard_mask as _load_discard_mask,  # noqa: F401  (re-exported API)
)
from wisco_slap.pns.aligned_frames import (
    registration_failed as _registration_failed,  # noqa: F401  (re-exported)
)
from wisco_slap.spinelab.activity import detect_active_frames
from wisco_slap.spinelab.datasets import _atomic_json, _jsonable


def _verified_exsum(metadata: Mapping[str, Any]) -> Path:
    """Return the dataset's ExSum path after re-deriving its bundle_id."""

    exsum = Path(metadata["exsum_path"])
    if not exsum.exists():
        raise FileNotFoundError(
            f"ExperimentSummary behind this dataset not found: {exsum}"
        )
    derived = _frames.file_identity(
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
    return exsum


def default_cache_path(
    data_dir: Path | str,
    dmd: int,
    *,
    max_trials: int | None = None,
    signal: str = "glutamate",
) -> Path:
    base = "frame_cache" if signal == "glutamate" else f"{signal}_cache"
    suffix = "" if max_trials is None else f"_smoke_{int(max_trials)}trials"
    return Path(data_dir) / f"dmd{int(dmd)}_{base}{suffix}.zarr"


def build_frame_cache(
    data_dir: Path | str,
    dmd: int,
    *,
    pixel_mask: np.ndarray | None = None,
    cache_path: Path | str | None = None,
    raw_dir: Path | str | None = None,
    max_trials: int | None = None,
    overwrite: bool = False,
    frame_block: int = 64,
    signal: str = "glutamate",
    workers: int | None = None,
) -> Path:
    """Build (or resume) an aligned frame cache for one dataset DMD/channel.

    Thin dataset-folder wrapper around
    :func:`wisco_slap.pns.aligned_frames.build_frame_cache`; see there for
    parameter semantics.
    """

    directory = Path(data_dir)
    context = context_from_dataset(directory)
    target = Path(
        cache_path
        if cache_path is not None
        else default_cache_path(directory, dmd, max_trials=max_trials, signal=signal)
    )
    return _frames.build_frame_cache(
        context,
        int(dmd),
        target,
        pixel_mask=pixel_mask,
        raw_dir=raw_dir,
        max_trials=max_trials,
        overwrite=overwrite,
        frame_block=frame_block,
        signal=signal,
        workers=workers,
    )


def validate_frame_cache(
    data_dir: Path | str,
    dmd: int,
    cache_path: Path | str,
) -> dict[str, Any]:
    """Validate a dataset DMD's frame cache against the ExperimentSummary."""

    context = context_from_dataset(Path(data_dir))
    return _frames.validate_frame_cache(context, int(dmd), cache_path)


def _relevant_pixel_mask(
    spines: list[CompiledSpine],
    support: np.ndarray,
    *,
    margin_px: int,
) -> np.ndarray:
    """Union of all annotation geometry, dilated, intersected with support."""

    union = np.zeros_like(support, dtype=bool)
    for spine in spines:
        union |= spine.head_mask
        union |= spine.background_mask
        for mask in spine.shaft_disk_masks:
            union |= mask
    if margin_px > 0:
        union = binary_dilation(union, iterations=int(margin_px))
    return union & support


def _mask_cache_columns(mask: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    columns = lookup[np.flatnonzero(mask.ravel())]
    return np.unique(columns[columns >= 0])


def _assert_cache_covers(
    spines: list[CompiledSpine], support: np.ndarray, lookup_2d: np.ndarray
) -> None:
    """Every in-support annotation pixel must be present in the cache."""

    required = _relevant_pixel_mask(spines, support, margin_px=0)
    missing = int(np.count_nonzero(required & (lookup_2d < 0)))
    if missing:
        raise ValueError(
            f"Frame cache is missing {missing} annotation pixels (annotations "
            "changed since the cache was built); rerun with rebuild_cache=True"
        )


def measure_dmd(
    data_dir: Path | str,
    dmd: int,
    *,
    annotations_path: Path | str | None = None,
    cache_path: Path | str | None = None,
    include_10000_frame_variant: bool = True,
    detector_kwargs: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Measure one DMD's annotated spines from its frame cache.

    Returns tidy measurement rows (two projection modes per spine and bin by
    default) with relative-to-baseline columns added. The cache must already
    exist and cover every annotation pixel; :func:`measure_dataset` is the
    orchestrator that builds caches on demand.
    """

    directory = Path(data_dir)
    dmd = int(dmd)
    metadata = load_dataset_metadata(directory)
    annotations = Path(
        annotations_path
        if annotations_path is not None
        else default_annotations_path(directory, dmd)
    )
    document = load_annotations(annotations)
    if (
        document.get("bundle_id") != metadata["bundle_id"]
        or int(document["dmd"]) != dmd
    ):
        raise ValueError("Annotations do not belong to this dataset/DMD")
    validation = validate_annotations_for_dataset(directory, dmd=dmd, document=document)
    if validation["errors"]:
        raise ValueError("Invalid annotations: " + "; ".join(validation["errors"]))
    compiled = _compile_spines(document)
    if not compiled:
        raise ValueError(f"{annotations} contains no spine heads")

    target = Path(
        cache_path if cache_path is not None else default_cache_path(directory, dmd)
    )
    cache = zarr.open_group(str(target), mode="r")
    if cache.attrs.get("schema_version") != FRAME_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Frame cache predates the corrected dropped-trial clock; "
            "rerun with rebuild_cache=True"
        )
    if cache.attrs.get("bundle_id") != metadata["bundle_id"]:
        raise ValueError("Frame cache does not belong to this dataset")
    if int(cache.attrs.get("dmd", -1)) != dmd:
        raise ValueError("Frame cache DMD does not match requested DMD")

    coords = np.asarray(cache["coords_rc"])
    image_shape = tuple(int(value) for value in document["image_shape"])
    lookup = np.full(image_shape[0] * image_shape[1], -1, dtype=np.int64)
    lookup[coords[:, 0] * image_shape[1] + coords[:, 1]] = np.arange(len(coords))
    support = load_dmd_images(directory, dmd)["support_mask"].astype(bool)
    _assert_cache_covers(compiled, support, lookup.reshape(image_shape))

    geometry: dict[str, dict[str, Any]] = {}
    union_columns: set[int] = set()
    for spine in compiled:
        head = _mask_cache_columns(spine.head_mask, lookup)
        background = _mask_cache_columns(spine.background_mask, lookup)
        shaft = [_mask_cache_columns(mask, lookup) for mask in spine.shaft_disk_masks]
        relevant = np.unique(
            np.concatenate([head, background, *shaft])
            if shaft
            else np.concatenate([head, background])
        )
        geometry[spine.spine_id] = {
            "head": head,
            "background": background,
            "relevant": relevant,
        }
        union_columns.update(int(value) for value in relevant)
    union = np.asarray(sorted(union_columns), dtype=np.int64)
    union_lookup = {int(column): index for index, column in enumerate(union)}
    for item in geometry.values():
        for key in ("head", "background", "relevant"):
            item[f"{key}_local"] = np.asarray(
                [union_lookup[int(value)] for value in item[key]], dtype=np.int64
            )

    cached_trials = np.asarray(cache["trial_index"])
    quality = np.asarray(cache["quality_valid"], dtype=bool)
    sample_rate = float(cache.attrs["sample_rate_hz"])
    detector_kwargs = dict(detector_kwargs or {})
    rows: list[dict[str, Any]] = []
    pixel_size = float(metadata["dmds"][str(dmd)]["pixel_size_um"])
    acquisition = {
        str(k): str(v) for k, v in metadata.get("acquisition", {}).items()
    }
    for chunk in metadata["dmds"][str(dmd)]["chunks"]:
        frame_indices = np.flatnonzero(
            (cached_trials >= int(chunk["first_trial"]))
            & (cached_trials <= int(chunk["last_trial"]))
        )
        if frame_indices.size == 0:
            continue
        block = np.asarray(cache["values"].oindex[frame_indices, union])
        block_quality = quality[frame_indices]
        for spine in compiled:
            item = geometry[spine.spine_id]
            head_local = item["head_local"]
            background_local = item["background_local"]
            if head_local.size == 0 or background_local.size == 0:
                trace = np.full(len(frame_indices), np.nan)
            else:
                head_trace = _finite_mean(block[:, head_local], axis=1)
                background_trace = _finite_mean(block[:, background_local], axis=1)
                trace = head_trace - background_trace
            trace_valid = block_quality & np.isfinite(trace)
            activity = detect_active_frames(
                trace,
                sample_rate_hz=sample_rate,
                valid_mask=trace_valid,
                **detector_kwargs,
            )
            modes: list[tuple[str, np.ndarray]] = [
                ("inactive_full_bin", trace_valid & ~activity.active)
            ]
            if include_10000_frame_variant:
                first_10000 = np.zeros(len(frame_indices), dtype=bool)
                first_10000[: min(10000, len(frame_indices))] = True
                modes.append((
                    "inactive_first_10000_acquired",
                    trace_valid & ~activity.active & first_10000,
                ))
            for mode, retained in modes:
                projection = np.full(image_shape, np.nan, dtype=np.float32)
                relevant_local = item["relevant_local"]
                relevant_columns = item["relevant"]
                if retained.any() and relevant_local.size:
                    projection_values = _finite_mean(
                        block[retained][:, relevant_local], axis=0
                    )
                    relevant_coords = coords[relevant_columns]
                    projection[relevant_coords[:, 0], relevant_coords[:, 1]] = (
                        projection_values
                    )
                row = measure_projection(projection, spine, pixel_size_um=pixel_size)
                acquired = len(frame_indices)
                valid_quality_count = int(block_quality.sum())
                row.update({
                    **acquisition,
                    "dmd": dmd,
                    "bin_index": int(chunk["bin_index"]),
                    "projection_mode": mode,
                    "first_trial": int(chunk["first_trial"]),
                    "last_trial": int(chunk["last_trial"]),
                    "n_trials": int(chunk["n_trials"]),
                    "n_acquired_frames": acquired,
                    "n_quality_valid_frames": valid_quality_count,
                    "n_active_frames": int((activity.active & trace_valid).sum()),
                    "n_inactive_retained_frames": int(retained.sum()),
                    "active_frame_fraction": float(
                        (activity.active & trace_valid).sum()
                        / max(1, trace_valid.sum())
                    ),
                    "activity_baseline": float(activity.baseline),
                    "activity_noise": float(activity.noise),
                    "activity_high_threshold": float(activity.high_threshold),
                    "start_seconds": float(chunk["start_seconds"]),
                    "end_seconds": float(chunk["end_seconds"]),
                    "duration_seconds": float(chunk["duration_seconds"]),
                    "partial_bin": bool(chunk["partial_bin"]),
                })
                extra_reasons = []
                if retained.sum() < 1000:
                    extra_reasons.append("fewer_than_1000_inactive_frames")
                if valid_quality_count / max(1, acquired) < 0.5:
                    extra_reasons.append("fewer_than_50pct_quality_frames")
                if extra_reasons:
                    prior = str(row["qc_reasons"])
                    row["qc_reasons"] = ";".join([
                        value for value in [prior, *extra_reasons] if value
                    ])
                    row["qc_valid"] = False
                rows.append(row)
    return add_relative_measurements(rows)


def measure_dataset(
    data_dir: Path | str,
    *,
    dmds: list[int] | None = None,
    annotations: Mapping[int, Path | str] | None = None,
    raw_dir: Path | str | None = None,
    cache_margin_um: float = 2.0,
    include_10000_frame_variant: bool = True,
    rebuild_cache: bool = False,
    output_dir: Path | str | None = None,
    max_trials: int | None = None,
    detector_kwargs: Mapping[str, Any] | None = None,
    write_qc: bool = True,
    workers: int | None = None,
) -> dict[str, Any]:
    """Measure every annotated DMD of a labelled spine dataset.

    Parameters
    ----------
    data_dir : Path | str
        A spine-labelling dataset folder containing saved annotations
        (``dmd{d}_spine_annotations.json``) for at least one DMD.
    dmds : list[int], optional
        Restrict to these DMDs. Default: every DMD with annotations.
    annotations : Mapping[int, Path | str], optional
        Per-DMD annotation-file overrides (default: the canonical files in
        the data directory).
    raw_dir : Path | str, optional
        Raw acquisition directory holding the registered TIFFs. Default:
        resolved from the raw mirror, then the NAS data root.
    cache_margin_um : float
        Safety margin dilated around the annotation pixels when building a
        frame cache, so small annotation edits do not force a rebuild.
    rebuild_cache : bool
        Rebuild an existing cache that no longer covers the annotations
        (rebuilding rereads every registered TIFF — slow).
    output_dir : Path | str, optional
        Where to write results. Default ``<data_dir>/measurements``.
    max_trials : int, optional
        Smoke-test mode: cache and measure only the first N trials.

    Returns
    -------
    dict
        Per-DMD row counts and output paths.
    """

    directory = Path(data_dir)
    metadata = load_dataset_metadata(directory)
    exsum = _verified_exsum(metadata)
    annotations = dict(annotations or {})

    candidate_dmds = (
        [int(d) for d in dmds]
        if dmds is not None
        else sorted(int(k) for k in metadata["dmds"])
    )
    annotated: dict[int, Path] = {}
    for dmd in candidate_dmds:
        path = Path(annotations.get(dmd, default_annotations_path(directory, dmd)))
        if path.exists():
            annotated[dmd] = path
    if not annotated:
        raise FileNotFoundError(
            f"No annotations found in {directory} for any of DMDs "
            f"{candidate_dmds}; label at least one DMD in the spine-labeller "
            "app first"
        )

    out_root = (
        Path(output_dir) if output_dir is not None else directory / "measurements"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"data_dir": str(directory), "dmds": {}}
    for dmd, annotations_path in annotated.items():
        document = load_annotations(annotations_path)
        validation = validate_annotations_for_dataset(
            directory, dmd=dmd, document=document
        )
        if validation["errors"]:
            raise ValueError(
                f"DMD{dmd} annotations are invalid: "
                + "; ".join(validation["errors"])
            )
        compiled = _compile_spines(document)
        if not compiled:
            raise ValueError(f"{annotations_path} contains no spine heads")

        arrays = load_dmd_images(directory, dmd)
        support = arrays["support_mask"].astype(bool)
        pixel_size = float(metadata["dmds"][str(dmd)]["pixel_size_um"])
        margin_px = int(np.ceil(cache_margin_um / pixel_size))
        required = _relevant_pixel_mask(compiled, support, margin_px=margin_px)

        cache_path = default_cache_path(directory, dmd, max_trials=max_trials)
        cache_validation: dict[str, Any] | None = None
        needs_build = True
        if cache_path.exists() and not rebuild_cache:
            cache = zarr.open_group(str(cache_path), mode="r")
            if cache.attrs.get("schema_version") != FRAME_CACHE_SCHEMA_VERSION:
                raise ValueError(
                    "Frame cache predates the corrected dropped-trial clock; "
                    "rerun with rebuild_cache=True"
                )
            compatible = (
                cache.attrs.get("bundle_id") == metadata["bundle_id"]
                and int(cache.attrs.get("dmd", -1)) == dmd
            )
            complete = bool(cache.attrs.get("complete", False))
            if compatible and complete:
                coords = np.asarray(cache["coords_rc"])
                lookup_2d = np.full(support.shape, -1, dtype=np.int64)
                lookup_2d[coords[:, 0], coords[:, 1]] = np.arange(len(coords))
                bare_required = _relevant_pixel_mask(compiled, support, margin_px=0)
                if not np.any(bare_required & (lookup_2d < 0)):
                    needs_build = False
                    print(
                        f"[spinelab] DMD{dmd}: reusing frame cache "
                        f"{cache_path.name}"
                    )
                else:
                    raise ValueError(
                        f"Existing cache {cache_path} is missing annotation "
                        "pixels (annotations changed since it was built); "
                        "rerun with rebuild_cache=True"
                    )
            elif compatible:
                print(
                    f"[spinelab] DMD{dmd}: resuming interrupted frame cache "
                    f"{cache_path.name}"
                )
            else:
                raise ValueError(
                    f"Existing cache {cache_path} belongs to a different "
                    "dataset/DMD; rerun with rebuild_cache=True"
                )
        if needs_build:
            print(
                f"[spinelab] DMD{dmd}: building frame cache "
                f"({int(required.sum())} pixels)..."
            )
            try:
                build_frame_cache(
                    directory,
                    dmd,
                    pixel_mask=required,
                    cache_path=cache_path,
                    raw_dir=raw_dir,
                    max_trials=max_trials,
                    overwrite=rebuild_cache,
                    workers=workers,
                )
            except ValueError as exc:
                raise ValueError(
                    f"DMD{dmd} frame cache could not be built or resumed "
                    f"({exc}); rerun with rebuild_cache=True to start it fresh"
                ) from exc
            cache_validation = validate_frame_cache(directory, dmd, cache_path)
            if not cache_validation["passed"]:
                raise RuntimeError(
                    f"DMD{dmd} frame cache failed validation against the "
                    f"ExperimentSummary: {json.dumps(cache_validation)}"
                )

        rows = measure_dmd(
            directory,
            dmd,
            annotations_path=annotations_path,
            cache_path=cache_path,
            include_10000_frame_variant=include_10000_frame_variant,
            detector_kwargs=detector_kwargs,
        )
        stem = out_root / f"dmd{dmd}_inactive_measurements"
        table = pl.DataFrame(rows)
        parquet_path = stem.with_suffix(".parquet")
        csv_path = stem.with_suffix(".csv")
        table.write_parquet(parquet_path)
        table.write_csv(csv_path)
        qc_paths: list[str] = []
        if write_qc:
            from wisco_slap.spinelab.qc import write_spine_qc_figures

            qc_paths = [
                str(p)
                for p in write_spine_qc_figures(
                    directory,
                    dmd=dmd,
                    annotations_path=annotations_path,
                    rows=rows,
                    output_dir=out_root / f"dmd{dmd}_inactive_measurements_qc",
                )
            ]
        _atomic_json(
            _jsonable({
                "measured_at": datetime.now(UTC).isoformat(),
                "exsum_path": str(exsum),
                "bundle_id": metadata["bundle_id"],
                "annotations_path": str(annotations_path),
                "cache_path": str(cache_path),
                "cache_validation": cache_validation,
                "detector_kwargs": dict(detector_kwargs or {}),
                "include_10000_frame_variant": include_10000_frame_variant,
                "n_rows": len(rows),
                "n_spines": len(compiled),
            }),
            out_root / f"dmd{dmd}_measurement_info.json",
        )
        summary["dmds"][dmd] = {
            "n_spines": len(compiled),
            "n_rows": len(rows),
            "parquet": str(parquet_path),
            "csv": str(csv_path),
            "qc_figures": qc_paths,
        }
        print(
            f"[spinelab] DMD{dmd}: measured {len(compiled)} spines "
            f"({len(rows)} rows) -> {parquet_path}"
        )
    return summary
