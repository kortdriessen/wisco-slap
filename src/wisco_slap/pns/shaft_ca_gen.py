"""Generate shaft-calcium disc traces along proximal-to-distal dendrite lines.

For each dendrite's WISynaptic ``prox_lines`` polyline (drawn on
``roi_locations/roi_locs_dmd{d}.tif``, which is pixel-identical to the ExSum
mean image in display orientation, so coordinates transfer with no
transform), discs are placed at fixed arc-length spacing and a jRGECO1a
(calcium-channel) trace is extracted per disc from the aligned registered
TIFFs via :mod:`wisco_slap.pns.aligned_frames`.

Prox lines are drawn starting at the proximal end, so each disc carries an
``arc_um`` coordinate (0 = proximal endpoint, increasing distally) — the axis
for propagation/attenuation analyses. The output time coordinate follows the
scopex trial boundaries exactly, including NaN-placeholder duration for
dropped trials and multi-epoch offsets. TIFF frames are spaced within each
trial's analyzed duration (~80 Hz); the time coordinate is authoritative
when per-trial frame rounding makes the interval slightly nonuniform.

Outputs per acquisition, in
``analysis_materials/<subject>/<exp>/scopex/<loc>--<acq>/shaft_ca/``:

- ``shaft_ca_dmd{d}.zarr``: xarray Dataset with ``F`` and ``dff``
  (disc x time), per-disc coords (dendrite_id, arc_um, center, n_pixels, F0)
  and per-frame coords (trial, quality). ``dendrite_id`` is the WISynaptic
  dend-ID assigned to the prox line (``A-1``, ``B-2``, ...), so it matches
  the canvas/synapse-label naming; prox lines without unique dend-IDs on
  every line — or saved before :data:`PROX_LINES_MIN_SAVE_DATE` (the
  channel-fix trust cutoff) — are skipped until labeled/re-saved;
- ``discs_dmd{d}.parquet/.csv``: the disc geometry table;
- ``map_dmd{d}.png``: discs over the mean image, colored by arc position;
- ``shaft_ca_info.json``: provenance including cache validation;
- ``shaft_ca__<ExSum-version>.txt`` and
  ``shaft_ca__prox_lines_dmd{d}.txt`` staleness markers (ExSum version and
  a hash of each prox_lines CSV) consumed by
  :mod:`wisco_slap.pns.shaft_ca_mon`.

Interpretation reminder: jRGECO1a kinetics and the 80 Hz frame rate cannot
time-resolve backpropagating-AP direction; use these traces for occurrence
logic (local vs global events), amplitude-vs-arc attenuation profiles, slow
spread, and event-averaged lead/lag versus the soma.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# NOTE: do NOT call matplotlib.use("Agg") here — a module-level backend switch
# hijacks the inline backend of any notebook that imports wisco_slap. The QC
# figure below is created and closed around savefig, which works on any backend.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import polars as pl
import tifffile
import xarray as xr
import zarr
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix
from scipy.stats import gaussian_kde
from spine_labeller.core import disk_mask, sample_polyline

import slap2_py as spy
import wisco_slap as wis
from wisco_slap.defs import anmat_root, annotation_root
from wisco_slap.pns import aligned_frames as frames
from wisco_slap.pns._frame_timing import (
    SHAFT_TIME_AXIS_VERSION,
    SHAFT_TIME_CONVENTION,
    fill_dropped_trial_counts,
    scopex_frame_times,
)

# Prox lines saved before this date were drawn against pre-channel-fix
# reference images (2026-08 reprocessing) and are not trusted: shaft-calcium
# extraction refuses them until they are re-verified against the current
# reference image and re-saved in WISynaptic (which refreshes the file mtime).
PROX_LINES_MIN_SAVE_DATE = datetime(2026, 8, 14)

# ---------------------------------------------------------------------------
# Paths and markers
# ---------------------------------------------------------------------------


def shaft_ca_dir(subject: str, exp: str, loc: str, acq: str) -> Path:
    return Path(anmat_root) / subject / exp / "scopex" / f"{loc}--{acq}" / "shaft_ca"


def prox_lines_csv(subject: str, exp: str, loc: str, acq: str, dmd: int) -> Path:
    return (
        Path(annotation_root)
        / subject
        / exp
        / loc
        / acq
        / "source_sorting"
        / f"prox_lines_dmd{dmd}.csv"
    )


def prox_lines_fingerprint(csv_path: Path | str) -> str:
    return hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()[:20]


def _write_markers(
    out_dir: Path, esum_version: str, fingerprints: dict[int, str]
) -> None:
    for old in out_dir.glob("shaft_ca__*.txt"):
        old.unlink()
    (out_dir / f"shaft_ca__{esum_version}.txt").write_text(f"{esum_version}\n")
    for dmd, fingerprint in fingerprints.items():
        (out_dir / f"shaft_ca__prox_lines_dmd{dmd}.txt").write_text(f"{fingerprint}\n")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


PROX_LINES_ID_COLUMN = "dend-ID"
"""Column WISynaptic writes the per-line dendrite ID (e.g. ``A-1``) into."""


def _read_prox_lines(csv_path: Path | str) -> pl.DataFrame:
    """Read a prox_lines CSV with the dend-ID column normalized to clean text."""

    df = pl.read_csv(csv_path)
    if PROX_LINES_ID_COLUMN in df.columns:
        df = df.with_columns(
            pl.col(PROX_LINES_ID_COLUMN).cast(pl.Utf8).fill_null("").str.strip_chars()
        )
    return df


def _line_dendrite_ids(df: pl.DataFrame) -> dict[int, str]:
    """Map each polyline ``index`` to its dend-ID ("" when unlabeled)."""

    ids: dict[int, str] = {}
    for (index,), sub in df.group_by("index"):
        labels = [v for v in sub[PROX_LINES_ID_COLUMN].to_list() if v]
        ids[int(index)] = labels[0] if labels else ""
    return dict(sorted(ids.items()))


def _prox_lines_problem(df: pl.DataFrame) -> str | None:
    if PROX_LINES_ID_COLUMN not in df.columns:
        return (
            f"no '{PROX_LINES_ID_COLUMN}' column - assign dend-IDs to the prox "
            "lines in WISynaptic"
        )
    ids = _line_dendrite_ids(df)
    unlabeled = [index for index, label in ids.items() if not label]
    if unlabeled:
        return f"prox lines {unlabeled} have no dend-ID - assign them in WISynaptic"
    counts = Counter(ids.values())
    duplicates = sorted(label for label, n in counts.items() if n > 1)
    if duplicates:
        return f"duplicate dend-IDs {duplicates} - each prox line needs a unique ID"
    return None


def prox_lines_problem(csv_path: Path | str) -> str | None:
    """Return why a prox_lines CSV cannot be used, or ``None`` if it is usable.

    Every polyline must carry a WISynaptic dend-ID (``A-1``, ``B-2``, ...)
    and IDs must be unique within the file: they become the ``dendrite_id``
    coordinate of the shaft-calcium outputs, so unlabeled lines are a
    manual-annotation gap rather than something to paper over with
    placeholder names.

    The file must also have been saved on or after
    :data:`PROX_LINES_MIN_SAVE_DATE`: earlier saves predate the 2026-08
    channel-fix reprocessing and must be re-verified (and re-saved) in
    WISynaptic before shaft-calcium extraction will use them.
    """

    saved = datetime.fromtimestamp(Path(csv_path).stat().st_mtime)
    if saved < PROX_LINES_MIN_SAVE_DATE:
        return (
            f"last saved {saved:%Y-%m-%d}, before the "
            f"{PROX_LINES_MIN_SAVE_DATE:%Y-%m-%d} channel-fix cutoff - "
            "re-verify the lines against the current reference image and "
            "re-save them in WISynaptic"
        )
    return _prox_lines_problem(_read_prox_lines(csv_path))


def load_prox_lines(csv_path: Path | str) -> list[dict[str, Any]]:
    """Load WISynaptic prox_lines: one proximal-first polyline per dendrite.

    ``dendrite_id`` is the WISynaptic dend-ID drawn on the line (e.g. ``A-1``).
    Raises ``ValueError`` when the CSV has no usable IDs (see
    :func:`prox_lines_problem`).
    """

    df = _read_prox_lines(csv_path)
    problem = _prox_lines_problem(df)
    if problem is not None:
        raise ValueError(f"{csv_path}: {problem}")
    ids = _line_dendrite_ids(df)
    lines: list[dict[str, Any]] = []
    for (index,), sub in sorted(df.group_by("index"), key=lambda item: int(item[0][0])):
        sub = sub.sort("vertex-index")
        vertices = np.stack(
            [sub["axis-0"].to_numpy(), sub["axis-1"].to_numpy()], axis=1
        ).astype(float)
        if len(vertices) < 2:
            continue
        lines.append({"dendrite_id": ids[int(index)], "vertices": vertices})
    return lines


def build_discs(
    lines: list[dict[str, Any]],
    *,
    image_shape: tuple[int, int],
    pixel_size_um: float,
    disc_diameter_um: float,
    disc_spacing_um: float,
    valid_mask: np.ndarray,
) -> list[dict[str, Any]]:
    """Place discs along each polyline; ``arc_um`` is 0 at the proximal end.

    Discs keep only pixels inside ``valid_mask`` (scan support minus any
    exclusions); discs left with no pixels are dropped.
    """

    radius_px = disc_diameter_um / (2.0 * pixel_size_um)
    discs: list[dict[str, Any]] = []
    for line in lines:
        centers, arcs_px = sample_polyline(
            line["vertices"], spacing_px=disc_spacing_um / pixel_size_um
        )
        for index, (center, arc_px) in enumerate(zip(centers, arcs_px, strict=True)):
            mask = disk_mask(center, radius_px, image_shape)
            mask &= valid_mask
            flat_pixels = np.flatnonzero(mask.ravel())
            if flat_pixels.size == 0:
                continue
            discs.append({
                "disc_id": f"{line['dendrite_id']}:{index:03d}",
                "dendrite_id": str(line["dendrite_id"]),
                "disc_index": int(index),
                "arc_um": float(arc_px * pixel_size_um),
                "center_row": float(center[0]),
                "center_col": float(center[1]),
                "flat_pixels": flat_pixels,
            })
    return discs


def disc_means_over_blocks(
    values: Any,
    column_lists: list[np.ndarray],
    *,
    block_frames: int = 4096,
) -> np.ndarray:
    """NaN-aware per-disc mean traces from a (frames x pixels) array.

    ``values`` is any sliceable 2-D array (zarr dataset or ndarray);
    ``column_lists`` gives each disc's column indices. Streams frame blocks
    through one sparse matrix multiply.
    """

    n_frames, n_columns = values.shape
    rows: list[int] = []
    cols: list[int] = []
    for disc_index, columns in enumerate(column_lists):
        rows.extend([disc_index] * len(columns))
        cols.extend(int(c) for c in columns)
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(column_lists), n_columns),
    )
    out = np.full((len(column_lists), n_frames), np.nan, dtype=np.float32)
    for start in range(0, n_frames, block_frames):
        stop = min(n_frames, start + block_frames)
        block = np.asarray(values[start:stop, :], dtype=np.float32)
        finite = np.isfinite(block)
        sums = matrix @ np.where(finite, block, 0.0).T
        counts = matrix @ finite.astype(np.float32).T
        with np.errstate(invalid="ignore", divide="ignore"):
            means = sums / counts
        means[counts == 0] = np.nan
        out[:, start:stop] = means.astype(np.float32)
    return out


def _kde_mode_f0(trace: np.ndarray, max_samples: int = 20000) -> float:
    """Baseline as the KDE mode of the (decimated) finite trace values."""

    finite = trace[np.isfinite(trace)]
    if finite.size == 0:
        return float("nan")
    if finite.size > max_samples:
        finite = finite[:: max(1, finite.size // max_samples)]
    if finite.size < 8 or np.ptp(finite) <= np.finfo(float).eps:
        return float(np.median(finite))
    low, high = np.quantile(finite, [0.005, 0.995])
    if not high > low:
        return float(np.median(finite))
    try:
        density = gaussian_kde(finite)
        grid = np.linspace(low, high, 512)
        return float(grid[int(np.argmax(density(grid)))])
    except (np.linalg.LinAlgError, ValueError):
        return float(np.median(finite))


def _geometry_guard(
    context: frames.AcqContext, dmd: int, subject: str, exp: str, loc: str, acq: str
) -> None:
    """The prox lines' pixel grid must match this ExSum's image grid."""

    roi_dir = Path(annotation_root) / subject / exp / loc / acq / "roi_locations"
    roi_tif = roi_dir / f"roi_locs_dmd{dmd}.tif"
    expected = context.image_shape(dmd)
    if roi_tif.exists():
        with tifffile.TiffFile(roi_tif) as tif:
            shape = tuple(tif.pages[0].shape)
        if shape != expected:
            raise ValueError(
                f"prox lines were drawn on {roi_tif.name} with shape {shape}, "
                f"but the current ExSum's DMD{dmd} image is {expected}; the "
                "image geometry changed between ExSum versions - regenerate "
                "the annotation materials and re-check the prox lines"
            )
    marker = roi_dir / ".generated_from_exsum.txt"
    if marker.exists():
        drawn_on = marker.read_text().strip()
        current = context.exsum_path.name.split(".mat")[0]
        if drawn_on != current:
            print(
                f"[shaft_ca] note: DMD{dmd} prox lines were drawn on ExSum "
                f"{drawn_on}, current mirror is {current}; geometry matches, "
                "so coordinates remain valid"
            )


def _epoch_offsets(
    subject: str, exp: str, loc: str, acq: str, epochs: np.ndarray
) -> dict[int, float] | None:
    """Scopex-convention epoch offsets (None for single-epoch acqs)."""

    if int(epochs.max()) <= 1:
        return None
    acq_id = f"{loc}--{acq}"
    info = wis.meta.get.epoch_info()
    try:
        acq_info = info[subject][exp][acq_id]
    except KeyError as exc:
        raise KeyError(
            f"[{subject} {exp} {acq_id}] multi-epoch acq but no entry in "
            "epoch_info.yaml; run wis.meta.update() first"
        ) from exc
    return {int(k): float(v["duration_since_epoch_1_s"]) for k, v in acq_info.items()}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def acq_data(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    *,
    disc_diameter_um: float = 1.0,
    disc_spacing_um: float = 1.0,
    cache_margin_um: float = 1.0,
    keep_cache: bool = False,
    workers: int | None = None,
    raw_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Generate shaft-calcium disc traces for one acquisition.

    Requires the current ExperimentSummary mirror, the raw registered TIFFs,
    and WISynaptic ``prox_lines`` for at least one DMD (DMDs without prox
    lines are skipped with a message). Always regenerates; freshness checking
    lives in :mod:`wisco_slap.pns.shaft_ca_mon`.

    Parameters
    ----------
    disc_diameter_um, disc_spacing_um : float
        Disc geometry along each prox line (defaults: 1 um discs every 1 um).
    cache_margin_um : float
        Safety margin dilated around disc pixels in the temporary calcium
        frame cache.
    keep_cache : bool
        Keep the calcium frame cache (``shaft_ca/dmd{d}_calcium_cache.zarr``)
        instead of deleting it after trace extraction. Rebuilds are cheap
        (parallel), so the default is to delete.
    workers : int, optional
        Worker processes for the cache build (default ``min(16, cpus)``).

    Returns
    -------
    dict
        Per-DMD disc counts and output paths.
    """

    tag = f"{subject} {exp} {loc} {acq}"
    esum_path = wis.meta.get.esum_mirror_path(subject, exp, loc, acq)
    if esum_path in (0, "NO_ESUM_MIRROR"):
        raise FileNotFoundError(
            f"[{tag}] no ExperimentSummary mirror; run wis.meta.update() first"
        )
    context = frames.context_from_exsum(esum_path)
    esum_version = context.exsum_path.name.split(".mat")[0]

    csv_paths: dict[int, Path] = {}
    for dmd in context.dmds:
        path = prox_lines_csv(subject, exp, loc, acq, dmd)
        if not path.exists():
            print(f"[{tag}] DMD{dmd}: no prox_lines CSV - skipping")
            continue
        problem = prox_lines_problem(path)
        if problem is not None:
            print(f"[{tag}] DMD{dmd}: prox_lines unusable ({problem}) - skipping")
            continue
        csv_paths[dmd] = path
    if not csv_paths:
        raise FileNotFoundError(
            f"[{tag}] no usable prox_lines for any DMD; draw them and assign "
            "dend-IDs in WISynaptic first"
        )

    # Lazy import: spinelab itself imports pns.aligned_frames, so importing
    # it at module scope would cycle during package initialization.
    from wisco_slap.spinelab.datasets import _atomic_json, _jsonable

    out_dir = shaft_ca_dir(subject, exp, loc, acq)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"acquisition": tag, "dmds": {}}
    info: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "exsum_path": str(context.exsum_path),
        "esum_version": esum_version,
        "disc_diameter_um": disc_diameter_um,
        "disc_spacing_um": disc_spacing_um,
        "dmds": {},
    }
    fingerprints: dict[int, str] = {}

    for dmd, csv_path in csv_paths.items():
        _geometry_guard(context, dmd, subject, exp, loc, acq)
        image_shape = context.image_shape(dmd)
        support = context.support[dmd]
        pixel_size = float(context.pixel_size_um[dmd])
        lines = load_prox_lines(csv_path)
        discs = build_discs(
            lines,
            image_shape=image_shape,
            pixel_size_um=pixel_size,
            disc_diameter_um=disc_diameter_um,
            disc_spacing_um=disc_spacing_um,
            valid_mask=support,
        )
        if not discs:
            raise ValueError(
                f"[{tag}] DMD{dmd}: no discs landed on scan support from {csv_path}"
            )

        required = np.zeros(image_shape, dtype=bool)
        for disc in discs:
            required.ravel()[disc["flat_pixels"]] = True
        margin_px = max(1, int(np.ceil(cache_margin_um / pixel_size)))
        cache_mask = binary_dilation(required, iterations=margin_px) & support

        kept_cache_path = out_dir / f"dmd{dmd}_calcium_cache.zarr"
        if keep_cache:
            cache_path = kept_cache_path
            temp_root = None
        else:
            temp_root = Path(tempfile.mkdtemp(prefix=f"shaft_ca_dmd{dmd}_"))
            cache_path = temp_root / "calcium_cache.zarr"
        try:
            print(
                f"[{tag}] DMD{dmd}: building calcium cache "
                f"({int(cache_mask.sum())} pixels, {len(discs)} discs)..."
            )
            frames.build_frame_cache(
                context,
                dmd,
                cache_path,
                pixel_mask=cache_mask,
                raw_dir=raw_dir,
                signal="calcium",
                workers=workers,
                overwrite=not keep_cache,
            )
            validation = frames.validate_frame_cache(context, dmd, cache_path)
            if not validation["passed"]:
                raise RuntimeError(
                    f"[{tag}] DMD{dmd} calcium cache failed validation "
                    f"against the ExperimentSummary: {validation}"
                )

            cache = zarr.open_group(str(cache_path), mode="r")
            coords = np.asarray(cache["coords_rc"])
            lookup = np.full(image_shape[0] * image_shape[1], -1, dtype=np.int64)
            lookup[coords[:, 0] * image_shape[1] + coords[:, 1]] = np.arange(
                len(coords)
            )
            kept_discs: list[dict[str, Any]] = []
            column_lists: list[np.ndarray] = []
            for disc in discs:
                columns = lookup[disc["flat_pixels"]]
                columns = columns[columns >= 0]
                if columns.size == 0:
                    continue
                kept_discs.append(dict(disc, n_pixels=int(columns.size)))
                column_lists.append(columns.astype(np.int64))

            F = disc_means_over_blocks(cache["values"], column_lists)
            quality = np.asarray(cache["quality_valid"], dtype=bool)
            trial_index = np.asarray(cache["trial_index"], dtype=np.int32)
        finally:
            if temp_root is not None:
                shutil.rmtree(temp_root, ignore_errors=True)

        # Scopex-convention time coordinate at the TIFF rate.
        frame_counts = np.bincount(trial_index, minlength=context.n_trials + 1)[1:]
        epochs = frames.trial_epochs(context.exsum_path, context.n_trials)
        offsets = _epoch_offsets(subject, exp, loc, acq, epochs)
        time_coord, sample_rate = scopex_frame_times(
            [int(c) for c in frame_counts],
            frames.per_trial_sample_counts(context.exsum_path, dmd, context.n_trials),
            context.analyze_hz,
            epochs,
            offsets,
        )

        F0 = np.array([_kde_mode_f0(F[d][quality]) for d in range(len(kept_discs))])
        with np.errstate(invalid="ignore", divide="ignore"):
            dff = (F - F0[:, None]) / F0[:, None]
        bad_f0 = ~np.isfinite(F0) | (np.nan_to_num(F0) <= 0)
        dff[bad_f0, :] = np.nan

        dataset = xr.Dataset(
            {
                "F": (("disc", "time"), F),
                "dff": (("disc", "time"), dff.astype(np.float32)),
            },
            coords={
                "disc": [d["disc_id"] for d in kept_discs],
                "dendrite_id": ("disc", [d["dendrite_id"] for d in kept_discs]),
                "arc_um": ("disc", [d["arc_um"] for d in kept_discs]),
                "center_row": ("disc", [d["center_row"] for d in kept_discs]),
                "center_col": ("disc", [d["center_col"] for d in kept_discs]),
                "n_pixels": ("disc", [d["n_pixels"] for d in kept_discs]),
                "F0": ("disc", F0),
                "time": time_coord,
                "trial": ("time", trial_index),
                "quality": ("time", quality),
            },
            attrs={
                "esum_version": esum_version,
                "dmd": dmd,
                "signal": "calcium",
                "sample_rate_hz": sample_rate,
                "disc_diameter_um": disc_diameter_um,
                "disc_spacing_um": disc_spacing_um,
                "prox_lines_csv": str(csv_path),
                "arc_convention": "arc_um=0 at the proximal end of each line",
                "time_convention": SHAFT_TIME_CONVENTION,
                "time_axis_version": SHAFT_TIME_AXIS_VERSION,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        zarr_path = out_dir / f"shaft_ca_dmd{dmd}.zarr"
        if zarr_path.exists():
            shutil.rmtree(zarr_path)
        chunk_time = min(4096, F.shape[1])
        dataset.to_zarr(
            zarr_path,
            mode="w",
            encoding={
                name: {"chunks": (len(kept_discs), chunk_time)} for name in ("F", "dff")
            },
        )
        table = pl.DataFrame([
            {key: value for key, value in disc.items() if key != "flat_pixels"}
            for disc in kept_discs
        ])
        table.write_parquet(out_dir / f"discs_dmd{dmd}.parquet")
        table.write_csv(out_dir / f"discs_dmd{dmd}.csv")
        mean_image = spy_mean_image(context, dmd)
        _write_disc_map(mean_image, kept_discs, out_dir / f"map_dmd{dmd}.png")

        fingerprints[dmd] = prox_lines_fingerprint(csv_path)
        info["dmds"][str(dmd)] = _jsonable({
            "prox_lines_csv": str(csv_path),
            "prox_lines_fingerprint": fingerprints[dmd],
            "n_lines": len(lines),
            "n_discs": len(kept_discs),
            "cache_validation": validation,
            "cache_kept": bool(keep_cache),
        })
        summary["dmds"][dmd] = {
            "n_lines": len(lines),
            "n_discs": len(kept_discs),
            "zarr": str(zarr_path),
        }
        print(
            f"[{tag}] DMD{dmd}: {len(kept_discs)} shaft-calcium discs on "
            f"{len(lines)} dendrites -> {zarr_path}"
        )

    _atomic_json(info, out_dir / "shaft_ca_info.json")
    _write_markers(out_dir, esum_version, fingerprints)
    return summary


def repair_time_coordinates(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    *,
    dmd: int | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Repair legacy shaft clocks without reading TIFFs or changing fluorescence.

    Requires current ExSum/prox-line markers and verifies trial order and Scopex
    length before writing. Only ``time`` and clock metadata change. Original
    time chunks and metadata are backed up under ``shaft_ca/timing_backups``;
    a failed write restores them. Current-version outputs are left untouched.
    """
    esum = Path(wis.meta.get.esum_mirror_path(subject, exp, loc, acq))
    version = esum.stem
    directory = shaft_ca_dir(subject, exp, loc, acq)
    _, n_trials, analyze_hz = spy.xsum.get_exsum_basics(esum)
    epochs = frames.trial_epochs(esum, n_trials)
    offsets = _epoch_offsets(subject, exp, loc, acq, epochs)
    records = []
    # Validate every requested DMD before mutating any output.
    plans = []
    for d in [dmd] if dmd is not None else [1, 2]:
        path = directory / f"shaft_ca_dmd{d}.zarr"
        if not path.is_dir():
            if dmd is not None:
                raise FileNotFoundError(path)
            continue
        cache = zarr.open_group(str(path), mode="r")
        record = {"dmd": d, "path": str(path)}
        if cache.attrs.get("time_axis_version") == SHAFT_TIME_AXIS_VERSION:
            records.append(record | {"status": "current"})
            continue
        if cache.attrs.get("time_axis_version", 1) != 1:
            raise ValueError(f"DMD{d}: unsupported shaft time-axis version")
        if (
            cache.attrs.get("esum_version") != version
            or not (directory / f"shaft_ca__{version}.txt").is_file()
        ):
            raise ValueError(
                f"DMD{d}: stale ExSum; regenerate traces before clock repair"
            )
        marker = directory / f"shaft_ca__prox_lines_dmd{d}.txt"
        csv = prox_lines_csv(subject, exp, loc, acq, d)
        if not marker.is_file() or marker.read_text().strip() != prox_lines_fingerprint(
            csv
        ):
            raise ValueError(
                f"DMD{d}: stale prox lines; regenerate traces before clock repair"
            )
        trials = np.asarray(cache["trial"])
        if trials.ndim != 1 or np.any(trials != np.floor(trials)):
            raise ValueError(f"DMD{d}: invalid trial identities")
        if not len(trials) or trials.min() < 1 or trials.max() > n_trials:
            raise ValueError(f"DMD{d}: trial identities outside current ExSum")
        counts = np.bincount(trials.astype(int), minlength=n_trials + 1)[1:]
        if not np.array_equal(trials, np.repeat(np.arange(1, n_trials + 1), counts)):
            raise ValueError(f"DMD{d}: shaft samples are not in original trial order")
        analyzed = frames.per_trial_sample_counts(esum, d, n_trials)
        new_time, rate = scopex_frame_times(
            counts, analyzed, analyze_hz, epochs, offsets
        )
        expected_samples = int(fill_dropped_trial_counts(analyzed, epochs).sum())
        scopex_path = directory.parent / "syn_F0.zarr"
        if scopex_path.is_dir():
            with xr.open_zarr(scopex_path, group=f"dmd_{d}") as scopex:
                if scopex.sizes["time"] != expected_samples:
                    raise ValueError(
                        f"DMD{d}: reconstructed trial lengths differ from Scopex"
                    )
        old_time = np.asarray(cache["time"])
        if old_time.shape != new_time.shape or cache["F"].shape[-1] != len(new_time):
            raise ValueError(f"DMD{d}: incompatible shaft/time shapes")
        record |= {
            "status": "would_repair" if dry_run else "repaired",
            "old_rate_hz": float(cache.attrs["sample_rate_hz"]),
            "new_rate_hz": rate,
            "max_time_change_s": float(np.max(np.abs(new_time - old_time))),
        }
        records.append(record)
        plans.append((path, new_time, rate, record))
    if dry_run:
        return records
    for path, new_time, rate, record in plans:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        backup = directory / "timing_backups" / f"dmd{record['dmd']}_{stamp}"
        backup.mkdir(parents=True)
        shutil.copytree(path / "time", backup / "time")
        for name in (".zattrs", ".zmetadata"):
            if (path / name).is_file():
                shutil.copy2(path / name, backup / name)
        try:
            cache = zarr.open_group(str(path), mode="a")
            cache["time"][:] = new_time
            cache.attrs.update({
                "sample_rate_hz": rate,
                "time_axis_version": SHAFT_TIME_AXIS_VERSION,
                "time_convention": SHAFT_TIME_CONVENTION,
                "time_axis_repaired_at": datetime.now(UTC).isoformat(),
                "previous_sample_rate_hz": record["old_rate_hz"],
            })
            zarr.consolidate_metadata(str(path))
            with xr.open_zarr(path) as verified:
                if not np.array_equal(verified.time.values, new_time):
                    raise RuntimeError(
                        f"DMD{record['dmd']}: clock repair failed readback"
                    )
            record["backup"] = str(backup)
            (backup / "repair.json").write_text(json.dumps(record, indent=2))
        except Exception:
            shutil.rmtree(path / "time")
            shutil.copytree(backup / "time", path / "time")
            for name in (".zattrs", ".zmetadata"):
                if (backup / name).is_file():
                    shutil.copy2(backup / name, path / name)
                elif (path / name).exists():
                    (path / name).unlink()
            raise
    return records


def spy_mean_image(context: frames.AcqContext, dmd: int) -> np.ndarray:
    """Glutamate-channel mean image for the QC disc map."""

    import slap2_py as spy

    channel = frames.signal_channel(context.channel_schema, "glutamate")
    images = spy.xsum.get_meanIM(context.exsum_path)
    return np.asarray(images[dmd][int(channel["array_index"])], dtype=np.float32)


def _write_disc_map(
    reference: np.ndarray, discs: list[dict[str, Any]], path: Path
) -> None:
    """Discs over the mean image, colored by arc (dark = proximal)."""

    finite = reference[np.isfinite(reference)]
    vmin, vmax = np.percentile(finite, [1, 99.7]) if finite.size else (0.0, 1.0)
    height, width = reference.shape
    figure, axis = plt.subplots(figsize=(14, max(3.0, 14 * height / width)))
    axis.imshow(reference, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    scatter = None
    for dendrite in sorted({d["dendrite_id"] for d in discs}):
        own = [d for d in discs if d["dendrite_id"] == dendrite]
        arcs = np.array([d["arc_um"] for d in own])
        scatter = axis.scatter(
            [d["center_col"] for d in own],
            [d["center_row"] for d in own],
            c=arcs,
            cmap="viridis",
            s=10,
            linewidths=0,
        )
        proximal = own[int(np.argmin(arcs))]
        axis.plot(
            proximal["center_col"],
            proximal["center_row"],
            "o",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor="cyan",
        )
        axis.annotate(
            f"{dendrite} (prox)",
            (proximal["center_col"], proximal["center_row"]),
            xytext=(0, 9),
            textcoords="offset points",
            color="cyan",
            fontsize=8,
            ha="center",
        )
    if scatter is not None:
        colorbar = figure.colorbar(scatter, ax=axis, fraction=0.03, pad=0.01)
        colorbar.set_label("arc from proximal end (um)")
    axis.set_axis_off()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
