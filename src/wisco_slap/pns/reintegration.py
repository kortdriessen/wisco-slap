"""Verification and migration helpers for the 2026-08 channel-fix
reprocessing reintegration.

This module is transitional tooling: it exists to validate that acquisitions
reprocessed into ``/data/raw_mirror`` have been cleanly reintegrated into
``analysis_materials`` (mirror swapped, annotation materials regenerated,
human annotations preserved / restored / migrated), and to record that fact
in the master tracking table at
``/data/slap_analysis/master_reprocessing_tracking.csv``. Once the
reintegration effort is finished this module can be retired.

Class A acquisitions (``identical_footprints`` true in the tracking table)
have identical image geometry and 1:1 same-index footprints. Recipe:

1. ``wis.meta.exsum_mirror.mirror_one_acq(...)`` — swap the mirror file
2. ``wis.pns.annotation_mat_mon.acq_data(...)`` — regenerate materials
   (archives ``synapse_labels.csv`` into ``attachments/stale_backups/``)
3. ``wis.pns.annotation_mat_mon.restore_synID_labels(...)`` — restore labels
4. :func:`verify_classA_acquisition` — validate everything and stamp
   ``annotation_materials_verified`` in the tracking table

Class B acquisitions changed geometry (a uniform border trim from the
multiRoiReg change) and changed footprint counts, so human annotations must
be migrated rather than restored. Recipe (all state that depends on the OLD
ExSum is captured to ``attachments/classB_staging/`` before the swap,
because the swap deletes the old mirror file and regeneration overwrites the
old canvas PNGs):

1. :func:`prepare_classB_acquisition` — measure the image shift, match old
   footprints to new ones, stage shift-corrected artifacts + a match report.
   Read-only apart from the staging folder. REVIEW THE MATCH REPORT before
   moving on.
2. :func:`apply_classB_swap` — move the human spatial artifacts into staging
   (releasing annotation_mat_mon's manual-review interlock), swap the
   mirror, regenerate annotation materials.
3. :func:`install_classB_artifacts` — write the migrated artifacts to their
   live locations (strokes remapped into the new canvas PNG space, shifted
   prox lines, cropped masks, matched labels).
4. :func:`verify_classB_acquisition` — validate, regenerate visual overlay
   previews for eyeballing, and stamp the tracking table.

After Class B verification, finish the human pass in WISynaptic's synapse
labeller: label the unmatched new sources (tracked by the status monitor's
Label Completeness column) and spot-check carried labels flagged in the
match report.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import tifffile as tiff
from PIL import Image, ImageDraw

import slap2_py as spy
import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.pns.annotation_mat_mon import _acq_dir, _load_existing_key

MASTER_TRACKING_CSV = Path("/data/slap_analysis/master_reprocessing_tracking.csv")
VERIFIED_COLUMN = "annotation_materials_verified"
STAGING_DIRNAME = "classB_staging"
FP_THRESHOLD = 0.02  # same threshold slap2_py get_fp_info uses to binarize
LABELS_HEADER = [
    "source-ID", "synapse-type", "soma-ID", "soma-depth",
    "dend-type", "dend-ID", "notes",
]


# ---------------------------------------------------------------------------
# Tracking-table helpers
# ---------------------------------------------------------------------------


def _tracking_row(subject: str, exp: str, loc: str, acq: str) -> dict | None:
    if not MASTER_TRACKING_CSV.exists():
        return None
    df = pl.read_csv(MASTER_TRACKING_CSV)
    match = df.filter(
        (pl.col("subject") == subject)
        & (pl.col("exp") == exp)
        & (pl.col("loc") == loc)
        & (pl.col("acq") == acq)
    )
    if match.height != 1:
        return None
    return match.row(0, named=True)


def _set_tracking_verified(
    subject: str, exp: str, loc: str, acq: str, value: bool
) -> None:
    df = pl.read_csv(MASTER_TRACKING_CSV)
    if VERIFIED_COLUMN not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Boolean).alias(VERIFIED_COLUMN))
    match = (
        (pl.col("subject") == subject)
        & (pl.col("exp") == exp)
        & (pl.col("loc") == loc)
        & (pl.col("acq") == acq)
    )
    n_match = df.filter(match).height
    assert n_match == 1, (
        f"{subject}/{exp}/{loc}/{acq} matched {n_match} rows in "
        f"{MASTER_TRACKING_CSV} — expected exactly 1"
    )
    df = df.with_columns(
        pl.when(match)
        .then(pl.lit(value))
        .otherwise(pl.col(VERIFIED_COLUMN))
        .alias(VERIFIED_COLUMN)
    )
    df.write_csv(MASTER_TRACKING_CSV)


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------


def _staging_dir(acq_root: Path) -> Path:
    return acq_root / "attachments" / STAGING_DIRNAME


def _labels_source_ids(csv_path: Path) -> list[int]:
    """Numeric source-IDs in a synapse_labels.csv (the roladex also writes a
    non-numeric ``master_image`` row, which is ignored)."""
    raw = pl.read_csv(
        csv_path,
        columns=["source-ID"],
        schema_overrides={"source-ID": pl.String},
    )["source-ID"].to_list()
    return sorted(int(s) for s in raw if s is not None and s.isdigit())


def _load_dmd_footprint_masks(mat_path: Path) -> dict[int, dict]:
    """Per DMD: display-oriented image shape and per-source pixel-index sets.

    Masks are sets of flat indices into the display-oriented (H, W) image —
    the same space as ``roi_locs_dmd#.tif``, prox lines, and meanIM.
    """
    out: dict[int, dict] = {}
    with h5py.File(mat_path, "r") as f:
        root = f["exptSummary"]
        E = root["E"][()]
        if E.shape[0] != 2 and E.shape[1] == 2:
            E = E.T
        sel_refs = root["selPix"][()]
        for dmd_idx in range(2):
            sel = np.asarray(f[sel_refs.flat[dmd_idx]][()]).astype(bool)
            fp = None
            for t in range(E.shape[1]):
                obj = f[E[dmd_idx][t]]
                if isinstance(obj, h5py.Group) and "footprints" in obj:
                    fp = np.asarray(obj["footprints"][()])
                    break
            if fp is None:
                raise RuntimeError(f"{mat_path}: no valid trial for DMD {dmd_idx + 1}")
            masks: list[set[int]] = []
            for i in range(fp.shape[0]):
                m = np.zeros(sel.shape, dtype=bool)
                m[sel] = fp[i] >= FP_THRESHOLD
                masks.append(set(np.flatnonzero(m.T.ravel()).tolist()))
            out[dmd_idx + 1] = {"shape": sel.shape[::-1], "masks": masks}
    return out


def _phase_corr_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """(dy, dx) estimate of the translation between images a and b."""
    H = max(a.shape[0], b.shape[0])
    W = max(a.shape[1], b.shape[1])
    pa = np.zeros((H, W))
    pb = np.zeros((H, W))
    a = np.nan_to_num(np.asarray(a, dtype=float))
    b = np.nan_to_num(np.asarray(b, dtype=float))
    pa[: a.shape[0], : a.shape[1]] = (a - a.mean()) / (a.std() + 1e-9)
    pb[: b.shape[0], : b.shape[1]] = (b - b.mean()) / (b.std() + 1e-9)
    R = np.fft.fft2(pa) * np.conj(np.fft.fft2(pb))
    R /= np.abs(R) + 1e-9
    corr = np.abs(np.fft.ifft2(R))
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    dy = int(dy) - (H if dy > H // 2 else 0)
    dx = int(dx) - (W if dx > W // 2 else 0)
    return dy, dx


def _shift_mask_sets(
    masks: list[set[int]],
    shift: tuple[int, int],
    old_shape: tuple[int, int],
    new_shape: tuple[int, int],
) -> list[set[int]]:
    """Apply (dy, dx) to mask pixel sets, re-raveled into the new shape."""
    sy, sx = shift
    H2, W2 = new_shape
    out: list[set[int]] = []
    for m in masks:
        if not m:
            out.append(set())
            continue
        idx = np.fromiter(m, dtype=np.int64)
        r, c = np.unravel_index(idx, old_shape)
        r = r + sy
        c = c + sx
        keep = (r >= 0) & (r < H2) & (c >= 0) & (c < W2)
        out.append(set((r[keep] * W2 + c[keep]).tolist()))
    return out


def _iou(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _iou_matrix(old_masks: list[set[int]], new_masks: list[set[int]]) -> np.ndarray:
    mat = np.zeros((len(old_masks), len(new_masks)))
    for i, om in enumerate(old_masks):
        for j, nm in enumerate(new_masks):
            mat[i, j] = _iou(om, nm)
    return mat


def _detect_axes_box(img: Image.Image) -> tuple[float, float, float, float]:
    """Locate the imshow axes box (left, top, width, height) in a canvas PNG.

    The dark_background style draws white spines around the axes; with
    aspect='equal' the spine rectangle coincides with the image data extent.
    """
    arr = np.asarray(img.convert("L"))
    white = arr > 160
    # Horizontal spines are full-width bright rows; the vertical spines can
    # be antialiased below threshold, so left/right are taken from the
    # column extent of the top/bottom spine rows instead.
    rows = np.flatnonzero(white.sum(axis=1) > 0.5 * arr.shape[1])
    if len(rows) < 2:
        raise RuntimeError(
            "could not locate the canvas axes box (white spine rows) — "
            "cannot map stroke coordinates"
        )
    top, bottom = int(rows.min()), int(rows.max())
    line_cols = np.flatnonzero(white[top] | white[bottom])
    left, right = int(line_cols.min()), int(line_cols.max())
    return (float(left), float(top), float(right - left), float(bottom - top))


# ---------------------------------------------------------------------------
# Class B step 1 — prepare (BEFORE the mirror swap)
# ---------------------------------------------------------------------------


def prepare_classB_acquisition(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    iou_threshold: float = 0.4,
    overwrite: bool = False,
) -> Path:
    """Stage everything a Class B migration needs while the OLD state exists.

    Must run BEFORE ``mirror_one_acq`` (the swap deletes the old mirror
    ExSum) and before ``acq_data`` (regeneration overwrites the old canvas
    PNGs). Writes only into ``attachments/classB_staging/``:

    - ``staging_meta.json`` — per-DMD measured shift, shapes, versions
    - ``match_dmd{1,2}.csv`` + ``match_report.md`` — footprint matching
    - ``synapse_labels_dmd{1,2}.csv`` — labels carried to new indices
      (blank rows for unmatched new sources)
    - ``prox_lines_dmd{1,2}.csv`` — shift-corrected prox lines
    - ``roi_locs_dmd{1,2}_mask.tif`` — masks resampled into the new geometry
    - ``annotations_raw_dmd{1,2}.json`` — canvas strokes in raw-image coords
      (converted to the new PNG's pixel space at install time)

    Review ``match_report.md`` before running :func:`apply_classB_swap`.

    Returns the staging directory path.
    """
    tag = f"{subject} {exp} {loc} {acq}"
    acq_root = _acq_dir(subject, exp, loc, acq)
    staging = _staging_dir(acq_root)
    if staging.exists() and not overwrite:
        raise FileExistsError(
            f"{staging} already exists — pass overwrite=True to re-stage "
            "(any prior review of the match report is then void)"
        )

    mirror_dir = Path(DEFS.exsum_mirror_root) / subject / exp / f"{loc}--{acq}"
    source_dir = (
        Path(DEFS.exsum_source_root) / subject / exp / loc / acq / "ExperimentSummary"
    )
    old_mats = sorted(mirror_dir.glob("*.mat"))
    new_mats = sorted(source_dir.glob("*.mat"))
    assert len(old_mats) == 1, f"expected 1 mirror .mat, found {len(old_mats)}"
    assert len(new_mats) == 1, f"expected 1 source .mat, found {len(new_mats)}"
    old_mat, new_mat = old_mats[0], new_mats[0]
    if old_mat.name == new_mat.name:
        raise RuntimeError(
            f"mirror already holds {new_mat.name} — the old ExSum is gone, "
            "prepare must run before the mirror swap"
        )

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(f"[{tag}] loading footprints from old ({old_mat.stem}) "
          f"and new ({new_mat.stem}) ExSums...")
    old_fp = _load_dmd_footprint_masks(old_mat)
    new_fp = _load_dmd_footprint_masks(new_mat)
    old_mim = spy.xsum.get_meanIM(str(old_mat))
    new_mim = spy.xsum.get_meanIM(str(new_mat))

    meta: dict = {
        "subject": subject, "exp": exp, "loc": loc, "acq": acq,
        "old_exsum": old_mat.stem, "new_exsum": new_mat.stem,
        "iou_threshold": iou_threshold,
        "created": datetime.now().isoformat(timespec="seconds"),
        "dmd": {},
    }
    report_lines = [
        f"# Class B match report — {tag}",
        f"old ExSum: {old_mat.stem}",
        f"new ExSum: {new_mat.stem}",
        f"IoU threshold for a confident match: {iou_threshold}",
        "",
    ]

    for dmd in (1, 2):
        old_shape = old_fp[dmd]["shape"]
        new_shape = new_fp[dmd]["shape"]
        old_masks = old_fp[dmd]["masks"]
        new_masks = new_fp[dmd]["masks"]

        # --- measure the shift; resolve its sign empirically -------------
        raw_shift = _phase_corr_shift(
            np.asarray(old_mim[dmd][1], dtype=float),
            np.asarray(new_mim[dmd][1], dtype=float),
        )
        candidates = {raw_shift, (-raw_shift[0], -raw_shift[1]), (0, 0)}
        best_shift, best_mat, best_score = None, None, -1.0
        for cand in candidates:
            shifted = _shift_mask_sets(old_masks, cand, old_shape, new_shape)
            mat = _iou_matrix(shifted, new_masks)
            score = float(mat.max(axis=1).sum()) if mat.size else 0.0
            if score > best_score:
                best_shift, best_mat, best_score = cand, mat, score
        assert best_shift is not None and best_mat is not None
        print(
            f"[{tag}] dmd{dmd}: shape {old_shape} -> {new_shape}, "
            f"applied shift (dy, dx) = {best_shift} "
            f"(phase-corr raw estimate {raw_shift})"
        )

        # --- mutual-best matching -----------------------------------------
        n_old, n_new = len(old_masks), len(new_masks)
        matches: dict[int, tuple[int, float]] = {}
        near_misses: list[tuple[int, int, float]] = []
        if n_old and n_new:
            best_new = best_mat.argmax(axis=1)
            best_old = best_mat.argmax(axis=0)
            for i in range(n_old):
                j = int(best_new[i])
                iou_ij = float(best_mat[i, j])
                if int(best_old[j]) != i or iou_ij <= 0:
                    continue
                if iou_ij >= iou_threshold:
                    matches[i] = (j, iou_ij)
                else:
                    near_misses.append((i, j, iou_ij))
        matched_new = {j for j, _ in matches.values()}
        unmatched_new = sorted(set(range(n_new)) - matched_new)
        dropped_old = sorted(set(range(n_old)) - set(matches))
        low_conf = sorted(
            (i, j, v) for i, (j, v) in matches.items() if v < 0.5
        )

        # --- match CSV ------------------------------------------------------
        with (staging / f"match_dmd{dmd}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["old_id", "new_id", "iou", "confident"])
            for i in sorted(matches):
                j, v = matches[i]
                w.writerow([i, j, f"{v:.4f}", True])
            for i, j, v in near_misses:
                w.writerow([i, j, f"{v:.4f}", False])

        # --- carry labels ---------------------------------------------------
        old_labels_path = (
            acq_root / "synapse_ids" / f"dmd-{dmd}" / "synapse_labels.csv"
        )
        dropped_labelled: list[str] = []
        if old_labels_path.exists():
            with old_labels_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                header = reader.fieldnames or LABELS_HEADER
                old_rows = {
                    int(r["source-ID"]): r
                    for r in reader
                    if (r.get("source-ID") or "").strip().isdigit()
                }
            new_to_old = {j: i for i, (j, _) in matches.items()}
            with (staging / f"synapse_labels_dmd{dmd}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as fh:
                w = csv.DictWriter(fh, fieldnames=header)
                w.writeheader()
                for j in range(n_new):
                    row = {k: "" for k in header}
                    row["source-ID"] = str(j)
                    if j in new_to_old and new_to_old[j] in old_rows:
                        src = old_rows[new_to_old[j]]
                        for k in header:
                            if k != "source-ID":
                                row[k] = src.get(k, "") or ""
                    w.writerow(row)
                master_row = {k: "" for k in header}
                master_row["source-ID"] = "master_image"
                w.writerow(master_row)
            for i in dropped_old:
                src = old_rows.get(i)
                if src and any(
                    (src.get(k) or "").strip()
                    for k in header
                    if k != "source-ID"
                ):
                    vals = ", ".join(
                        f"{k}={src[k]}" for k in header
                        if k != "source-ID" and (src.get(k) or "").strip()
                    )
                    dropped_labelled.append(f"old {i}: {vals}")
        else:
            print(f"[{tag}] dmd{dmd}: no old synapse_labels.csv — nothing to carry")

        # --- shift prox lines -------------------------------------------------
        prox_path = acq_root / "source_sorting" / f"prox_lines_dmd{dmd}.csv"
        if prox_path.exists():
            prox = pl.read_csv(prox_path)
            prox = prox.with_columns(
                (pl.col("axis-0") + best_shift[0]).clip(0, new_shape[0] - 1),
                (pl.col("axis-1") + best_shift[1]).clip(0, new_shape[1] - 1),
            )
            prox.write_csv(staging / f"prox_lines_dmd{dmd}.csv")

        # --- resample ROI mask --------------------------------------------------
        mask_path = acq_root / "roi_locations" / f"roi_locs_dmd{dmd}_mask.tif"
        if mask_path.exists():
            old_mask = tiff.imread(mask_path)
            new_mask = np.zeros(new_shape, dtype=old_mask.dtype)
            sy, sx = best_shift
            rr, cc = np.nonzero(old_mask)
            rr = rr + sy
            cc = cc + sx
            keep = (rr >= 0) & (rr < new_shape[0]) & (cc >= 0) & (cc < new_shape[1])
            src_r, src_c = np.nonzero(old_mask)
            new_mask[rr[keep], cc[keep]] = old_mask[src_r[keep], src_c[keep]]
            tiff.imwrite(staging / f"roi_locs_dmd{dmd}_mask.tif", new_mask)

        # --- canvas strokes -> raw coordinates ------------------------------------
        png_path = acq_root / "canvas" / f"DMD-{dmd}.png"
        ann_path = acq_root / "canvas" / f"DMD-{dmd}.png.annotations.json"
        png_meta = None
        if ann_path.exists():
            assert png_path.exists(), f"{ann_path} exists but {png_path} is missing"
            with Image.open(png_path) as im:
                box = _detect_axes_box(im)
                png_meta = {"size": list(im.size), "axes_box": list(box)}
            left, top, bw, bh = box
            H, W = old_shape
            sy, sx = best_shift
            strokes_raw = []
            for s in json.loads(ann_path.read_text(encoding="utf-8"))["strokes"]:
                pts = [
                    [
                        (x - left) / bw * W - 0.5 + sx,
                        (y - top) / bh * H - 0.5 + sy,
                    ]
                    for x, y in s["points"]
                ]
                strokes_raw.append(
                    {
                        "color": s.get("color", "#ff0000"),
                        "width_raw": float(s.get("width", 3.0)) * W / bw,
                        "points_raw": pts,
                        "eraser": bool(s.get("eraser", False)),
                    }
                )
            (staging / f"annotations_raw_dmd{dmd}.json").write_text(
                json.dumps({"strokes": strokes_raw}, indent=2), encoding="utf-8"
            )

        meta["dmd"][str(dmd)] = {
            "shift": list(best_shift),
            "phase_corr_raw": list(raw_shift),
            "old_shape": list(old_shape),
            "new_shape": list(new_shape),
            "n_old": n_old,
            "n_new": n_new,
            "n_matched": len(matches),
            "old_png": png_meta,
        }

        report_lines += [
            f"## DMD {dmd}",
            f"- shape {tuple(old_shape)} -> {tuple(new_shape)}, "
            f"applied shift (dy, dx) = {best_shift}",
            f"- {len(matches)}/{n_old} old sources matched to new "
            f"({n_new} new sources total)",
            f"- unmatched NEW sources (need labelling in the roladex): "
            f"{unmatched_new or 'none'}",
            f"- dropped OLD sources (no confident new match): "
            f"{dropped_old or 'none'}",
        ]
        if dropped_labelled:
            report_lines.append("- dropped old sources that HAD labels:")
            report_lines += [f"    - {t}" for t in dropped_labelled]
        if near_misses:
            report_lines.append(
                "- near-misses (mutual best match below threshold, NOT carried):"
            )
            report_lines += [
                f"    - old {i} ~ new {j} (IoU {v:.3f})" for i, j, v in near_misses
            ]
        if low_conf:
            report_lines.append(
                "- carried but low-confidence (IoU < 0.5) — spot-check these "
                "in the roladex:"
            )
            report_lines += [
                f"    - old {i} -> new {j} (IoU {v:.3f})" for i, j, v in low_conf
            ]
        report_lines.append("")

    (staging / "staging_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    report_path = staging / "match_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[{tag}] staged into {staging}")
    print(f"[{tag}] REVIEW the match report before proceeding: {report_path}")
    return staging


# ---------------------------------------------------------------------------
# Class B step 2 — swap + regenerate
# ---------------------------------------------------------------------------


def apply_classB_swap(subject: str, exp: str, loc: str, acq: str) -> None:
    """Move human spatial artifacts into staging, swap the mirror, regenerate.

    Moving ``canvas/*.annotations.json``, ``source_sorting/prox_lines_*.csv``
    and ``roi_locations/*mask*.tif`` into ``classB_staging/originals/``
    releases annotation_mat_mon's ``manual_review_required`` interlock, so
    ``acq_data`` can regenerate everything from the new ExSum (still
    archiving the old ``synapse_labels.csv`` into ``stale_backups/``).
    """
    tag = f"{subject} {exp} {loc} {acq}"
    acq_root = _acq_dir(subject, exp, loc, acq)
    staging = _staging_dir(acq_root)
    meta_path = staging / "staging_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found — run prepare_classB_acquisition first"
        )

    originals = staging / "originals"
    if originals.exists():
        # A previous apply already stashed the true originals. Re-stashing
        # would overwrite them with whatever is live now (possibly already-
        # migrated artifacts from a prior install) — never do that.
        print(f"[{tag}] originals already stashed — skipping stash")
    else:
        patterns = (
            "canvas/*.annotations.json",
            "source_sorting/prox_lines_dmd*.csv",
            "roi_locations/*mask*.tif",
        )
        for pattern in patterns:
            for src in sorted(acq_root.glob(pattern)):
                dest = originals / src.relative_to(acq_root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
                print(f"[{tag}] stashed {src.relative_to(acq_root)}")

    status = wis.meta.exsum_mirror.mirror_one_acq(subject, exp, loc, acq)
    print(f"[{tag}] mirror_one_acq -> {status}")

    audit = wis.pns.annotation_mat_mon.acq_data(subject, exp, loc, acq)
    if audit.status != "fresh":
        raise RuntimeError(
            f"annotation materials did not come back fresh after regeneration "
            f"(status {audit.status}: {'; '.join(audit.reasons)})"
        )
    print(f"[{tag}] annotation materials regenerated (audit fresh)")


# ---------------------------------------------------------------------------
# Class B step 3 — install migrated artifacts
# ---------------------------------------------------------------------------


def install_classB_artifacts(subject: str, exp: str, loc: str, acq: str) -> None:
    """Write the staged, shift-corrected artifacts to their live locations."""
    tag = f"{subject} {exp} {loc} {acq}"
    acq_root = _acq_dir(subject, exp, loc, acq)
    staging = _staging_dir(acq_root)
    meta = json.loads((staging / "staging_meta.json").read_text(encoding="utf-8"))

    mirror_dir = Path(DEFS.exsum_mirror_root) / subject / exp / f"{loc}--{acq}"
    mirror_mats = sorted(mirror_dir.glob("*.mat"))
    if len(mirror_mats) != 1 or mirror_mats[0].stem != meta["new_exsum"]:
        raise RuntimeError(
            "mirror does not hold the staged new ExSum — run apply_classB_swap "
            "before installing"
        )

    for dmd in (1, 2):
        dmeta = meta["dmd"][str(dmd)]
        H2, W2 = dmeta["new_shape"]

        # canvas strokes: raw coords -> new PNG pixel space
        raw_json = staging / f"annotations_raw_dmd{dmd}.json"
        if raw_json.exists():
            png_path = acq_root / "canvas" / f"DMD-{dmd}.png"
            with Image.open(png_path) as im:
                left, top, bw, bh = _detect_axes_box(im)
                png_w, png_h = im.size
            strokes = []
            n_clamped = 0
            for s in json.loads(raw_json.read_text(encoding="utf-8"))["strokes"]:
                pts = []
                for x, y in s["points_raw"]:
                    px = (x + 0.5) / W2 * bw + left
                    py = (y + 0.5) / H2 * bh + top
                    # Points near the trimmed border (or drawn in the old
                    # PNG's margins) can land outside the new, smaller PNG —
                    # clamp them to its bounds, as prepare does for prox lines.
                    cx = min(max(px, 0.0), float(png_w))
                    cy = min(max(py, 0.0), float(png_h))
                    if cx != px or cy != py:
                        n_clamped += 1
                    pts.append([cx, cy])
                strokes.append(
                    {
                        "color": s["color"],
                        "width": s["width_raw"] * bw / W2,
                        "points": pts,
                        "eraser": s["eraser"],
                    }
                )
            if n_clamped:
                print(
                    f"[{tag}] dmd{dmd}: clamped {n_clamped} stroke point(s) "
                    "to the new PNG bounds"
                )
            ann_path = acq_root / "canvas" / f"DMD-{dmd}.png.annotations.json"
            ann_path.write_text(
                json.dumps({"strokes": strokes}, indent=2), encoding="utf-8"
            )
            print(f"[{tag}] installed {ann_path.relative_to(acq_root)}")

        # prox lines
        staged_prox = staging / f"prox_lines_dmd{dmd}.csv"
        if staged_prox.exists():
            dest = acq_root / "source_sorting" / f"prox_lines_dmd{dmd}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staged_prox, dest)
            print(f"[{tag}] installed {dest.relative_to(acq_root)}")

        # ROI mask
        staged_mask = staging / f"roi_locs_dmd{dmd}_mask.tif"
        if staged_mask.exists():
            dest = acq_root / "roi_locations" / f"roi_locs_dmd{dmd}_mask.tif"
            shutil.copyfile(staged_mask, dest)
            print(f"[{tag}] installed {dest.relative_to(acq_root)}")

        # matched labels
        staged_labels = staging / f"synapse_labels_dmd{dmd}.csv"
        if staged_labels.exists():
            dest = acq_root / "synapse_ids" / f"dmd-{dmd}" / "synapse_labels.csv"
            if dest.exists():
                if dest.read_bytes() == staged_labels.read_bytes():
                    print(
                        f"[{tag}] {dest.relative_to(acq_root)} already "
                        "installed — skipping"
                    )
                    continue
                raise FileExistsError(
                    f"{dest} already exists and differs from the staged "
                    "labels (edited in the roladex since install?) — "
                    "refusing to overwrite live labels"
                )
            shutil.copyfile(staged_labels, dest)
            print(f"[{tag}] installed {dest.relative_to(acq_root)}")


# ---------------------------------------------------------------------------
# Verification (shared core + per-class entry points)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReintegrationVerification:
    """Outcome of one reintegration verification pass."""

    subject: str
    exp: str
    loc: str
    acq: str
    passed: bool = False
    checks_passed: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    tracking_updated: bool = False


# Backwards-compatible alias (pre-Class-B name).
ClassAVerification = ReintegrationVerification


def _verify_acquisition(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    expected_class: str,
    update_tracking: bool,
) -> ReintegrationVerification:
    result = ReintegrationVerification(subject=subject, exp=exp, loc=loc, acq=acq)
    acq_root = _acq_dir(subject, exp, loc, acq)
    staging = _staging_dir(acq_root)

    ok = result.checks_passed.append
    fail = result.failures.append
    skip = result.skipped.append

    # -- 1. Class membership ------------------------------------------------
    row = _tracking_row(subject, exp, loc, acq)
    if row is None:
        fail(
            f"no (unique) row for this acq in {MASTER_TRACKING_CSV} — "
            "regenerate it with tooling/update_master_reprocessing_tracking.py"
        )
    else:
        if row.get("reprocessed") is True:
            ok("tracking table: reprocessed")
        else:
            fail("tracking table says this acq has not been reprocessed")
        if expected_class == "A":
            if row.get("identical_footprints") is True:
                ok("tracking table: identical_footprints")
            else:
                fail(
                    "tracking table identical_footprints is not true — this "
                    "is not a Class A acquisition"
                )
        elif expected_class == "B":  # membership proven by the staging metadata
            if (staging / "staging_meta.json").exists():
                ok("classB staging present (staging_meta.json)")
            else:
                fail(
                    "no classB_staging/staging_meta.json — this acq did not "
                    "go through the Class B pipeline"
                )
        else:  # fresh onboard: no prior ExSum, so no comparison columns
            if row.get("identical_footprints") is None:
                ok("tracking table: no prior ExSum to compare (fresh onboard)")
            else:
                fail(
                    "tracking table has an identical_footprints value — this "
                    "acq HAD a prior ExSum and belongs to Class A or B, not "
                    "the fresh-onboarding path"
                )

    # -- 2. Mirror holds the source-of-truth file ---------------------------
    mirror_dir = Path(DEFS.exsum_mirror_root) / subject / exp / f"{loc}--{acq}"
    source_dir = (
        Path(DEFS.exsum_source_root) / subject / exp / loc / acq / "ExperimentSummary"
    )
    mirror_mats = sorted(mirror_dir.glob("*.mat")) if mirror_dir.is_dir() else []
    source_mats = sorted(source_dir.glob("*.mat")) if source_dir.is_dir() else []
    if len(mirror_mats) != 1:
        fail(f"expected exactly 1 mirror .mat in {mirror_dir}, found {len(mirror_mats)}")
    elif len(source_mats) != 1:
        fail(f"expected exactly 1 source .mat in {source_dir}, found {len(source_mats)}")
    elif mirror_mats[0].name != source_mats[0].name:
        fail(
            f"mirror file {mirror_mats[0].name} != source file "
            f"{source_mats[0].name} — run the mirror swap first"
        )
    else:
        ok(f"mirror matches source ({mirror_mats[0].stem})")

    # -- 3. Annotation materials audit is fresh -----------------------------
    audit = wis.pns.annotation_mat_mon.audit_acq(subject, exp, loc, acq)
    if audit.status == "fresh":
        ok(f"annotation audit fresh (esum {audit.esum_version})")
    else:
        reasons = "; ".join(audit.reasons) if audit.reasons else "no reasons given"
        fail(f"annotation audit status is '{audit.status}': {reasons}")

    # -- 4. Labels present + count-valid per DMD ----------------------------
    for dmd in (1, 2):
        dmd_dir = acq_root / "synapse_ids" / f"dmd-{dmd}"
        key_path = dmd_dir / "source_location_key.npz"
        labels_path = dmd_dir / "synapse_labels.csv"
        if not key_path.exists():
            fail(f"dmd-{dmd}: source_location_key.npz missing")
            continue
        _, id_list = _load_existing_key(key_path)
        n_sources = max(len(id_list) - 1, 0)
        if n_sources == 0:
            skip(f"dmd-{dmd}: no sources, labels not expected")
            continue
        if not labels_path.exists():
            archived = list(
                acq_root.glob(
                    f"attachments/stale_backups/*/synapse_ids/dmd-{dmd}/"
                    "synapse_labels.csv"
                )
            )
            if archived:
                fail(
                    f"dmd-{dmd}: synapse_labels.csv missing but archived "
                    "copies exist — restore/install the labels"
                )
            else:
                skip(f"dmd-{dmd}: never labeled (no live or archived labels)")
            continue
        source_ids = _labels_source_ids(labels_path)
        if source_ids == list(range(n_sources)):
            ok(f"dmd-{dmd}: labels present for all {n_sources} sources")
        else:
            fail(
                f"dmd-{dmd}: label source-IDs do not match the current "
                f"{n_sources} sources"
            )

    # -- 5. Canvas strokes within regenerated PNG bounds --------------------
    for dmd in (1, 2):
        png_path = acq_root / "canvas" / f"DMD-{dmd}.png"
        ann_path = acq_root / "canvas" / f"DMD-{dmd}.png.annotations.json"
        if not ann_path.exists():
            skip(f"dmd-{dmd}: no canvas annotations")
            continue
        if not png_path.exists():
            fail(f"dmd-{dmd}: canvas PNG missing but annotations exist")
            continue
        with Image.open(png_path) as im:
            w, h = im.size
        strokes = json.loads(ann_path.read_text(encoding="utf-8")).get("strokes", [])
        pts = [p for s in strokes for p in s.get("points", [])]
        out_of_bounds = [
            p for p in pts if not (-0.5 <= p[0] <= w + 0.5 and -0.5 <= p[1] <= h + 0.5)
        ]
        if out_of_bounds:
            fail(
                f"dmd-{dmd}: {len(out_of_bounds)}/{len(pts)} canvas stroke "
                f"points fall outside the regenerated PNG ({w}x{h})"
            )
        else:
            ok(f"dmd-{dmd}: {len(pts)} canvas stroke points within PNG bounds")

    # -- 6/7. Prox lines and masks against roi_locs geometry ----------------
    for dmd in (1, 2):
        roi_path = acq_root / "roi_locations" / f"roi_locs_dmd{dmd}.tif"
        if not roi_path.exists():
            fail(f"dmd-{dmd}: roi_locs_dmd{dmd}.tif missing")
            continue
        shape = tiff.imread(roi_path).shape[:2]

        prox_path = acq_root / "source_sorting" / f"prox_lines_dmd{dmd}.csv"
        if prox_path.exists():
            prox = pl.read_csv(prox_path)
            bad = prox.filter(
                (pl.col("axis-0") < -0.5)
                | (pl.col("axis-0") > shape[0] - 0.5)
                | (pl.col("axis-1") < -0.5)
                | (pl.col("axis-1") > shape[1] - 0.5)
            ).height
            if bad:
                fail(
                    f"dmd-{dmd}: {bad}/{prox.height} prox-line vertices fall "
                    f"outside roi_locs image {shape}"
                )
            else:
                ok(f"dmd-{dmd}: prox lines within roi_locs bounds")
        else:
            skip(f"dmd-{dmd}: no prox lines")

        mask_path = acq_root / "roi_locations" / f"roi_locs_dmd{dmd}_mask.tif"
        if mask_path.exists():
            mask_shape = tiff.imread(mask_path).shape[:2]
            if mask_shape == shape:
                ok(f"dmd-{dmd}: ROI mask shape matches roi_locs")
            else:
                fail(
                    f"dmd-{dmd}: ROI mask shape {mask_shape} != roi_locs "
                    f"shape {shape}"
                )
        else:
            skip(f"dmd-{dmd}: no ROI mask")

    # -- Class B extra: regenerate visual previews for eyeballing -----------
    if expected_class == "B" and not result.failures:
        try:
            preview_dir = _write_classB_previews(acq_root, staging)
            ok(f"overlay previews written to {preview_dir}")
        except Exception as exc:  # noqa: BLE001
            fail(f"could not generate overlay previews: {exc}")

    # -- Summary + tracking update ------------------------------------------
    result.passed = not result.failures
    tag = f"{subject} {exp} {loc} {acq}"
    for msg in result.checks_passed:
        print(f"[{tag}] PASS: {msg}")
    for msg in result.skipped:
        print(f"[{tag}] skip: {msg}")
    for msg in result.failures:
        print(f"[{tag}] FAIL: {msg}")

    if result.passed and update_tracking:
        _set_tracking_verified(subject, exp, loc, acq, value=True)
        result.tracking_updated = True
        print(f"[{tag}] VERIFIED — {VERIFIED_COLUMN} set to true in tracking table")
    elif result.passed:
        print(f"[{tag}] VERIFIED (tracking table not updated, update_tracking=False)")
    else:
        print(
            f"[{tag}] NOT verified — {len(result.failures)} failure(s); "
            "tracking table left untouched"
        )
    return result


def _write_classB_previews(acq_root: Path, staging: Path) -> Path:
    """Render the migrated annotations over the new reference images."""
    import matplotlib.pyplot as plt

    preview_dir = staging / "previews"
    preview_dir.mkdir(exist_ok=True)

    for dmd in (1, 2):
        # canvas strokes over the new canvas PNG
        png_path = acq_root / "canvas" / f"DMD-{dmd}.png"
        ann_path = acq_root / "canvas" / f"DMD-{dmd}.png.annotations.json"
        if png_path.exists() and ann_path.exists():
            base = Image.open(png_path).convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            strokes = json.loads(ann_path.read_text(encoding="utf-8"))["strokes"]
            for s in strokes:
                pts = [(x, y) for x, y in s["points"]]
                if len(pts) < 2:
                    pts = pts * 2
                w = max(1, round(float(s.get("width", 3.0))))
                mask = Image.new("L", base.size, 0)
                d = ImageDraw.Draw(mask)
                d.line(pts, fill=255, width=w, joint="curve")
                if s.get("eraser", False):
                    arr = np.array(overlay)
                    arr[np.array(mask) > 0] = 0
                    overlay = Image.fromarray(arr)
                else:
                    c = s.get("color", "#ff0000").lstrip("#")
                    rgba = (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 255)
                    colored = Image.new("RGBA", base.size, rgba)
                    overlay.paste(colored, (0, 0), mask)
            base.alpha_composite(overlay)
            base.convert("RGB").save(preview_dir / f"canvas_dmd{dmd}.png")

        # prox lines over the new roi_locs image
        roi_path = acq_root / "roi_locations" / f"roi_locs_dmd{dmd}.tif"
        prox_path = acq_root / "source_sorting" / f"prox_lines_dmd{dmd}.csv"
        if roi_path.exists() and prox_path.exists():
            img = np.nan_to_num(tiff.imread(roi_path).astype(float))
            v0, v1 = np.percentile(img, [1, 99])
            img = np.clip((img - v0) / max(v1 - v0, 1e-9), 0, 1)
            prox = pl.read_csv(prox_path)
            f, ax = plt.subplots(
                figsize=(max(8, img.shape[1] / 60), max(5, img.shape[0] / 60))
            )
            ax.imshow(img, cmap="gray", interpolation="nearest")
            cmap = plt.get_cmap("tab10")
            # older prox_lines CSVs predate the optional dend-ID column
            has_dend = "dend-ID" in prox.columns
            for k, (_, g) in enumerate(prox.group_by("index", maintain_order=True)):
                g = g.sort("vertex-index")
                dend = g["dend-ID"].drop_nulls() if has_dend else []
                name = dend[0] if len(dend) else f"line{k}"
                ax.plot(
                    g["axis-1"], g["axis-0"], "-o",
                    color=cmap(k % 10), linewidth=1.5, markersize=3,
                )
                ax.annotate(
                    str(name), (g["axis-1"][0], g["axis-0"][0]),
                    color=cmap(k % 10), fontsize=9, fontweight="bold",
                    xytext=(4, -4), textcoords="offset points",
                )
            ax.set_xticks([]), ax.set_yticks([])
            f.tight_layout()
            f.savefig(preview_dir / f"prox_lines_dmd{dmd}.png", dpi=200,
                      bbox_inches="tight")
            plt.close(f)

    return preview_dir


SCOPEX_ARCHIVE_DIRNAME = "_pre_channel_fix_reintegration"


def _zarr_store_readable(store_path: Path) -> None:
    """Open a zarr store and touch one element of every array in it.

    Raises on any unreadable store/group/array — a light integrity check that
    catches half-written zarrs, which scopex_mon's existence+flag audit
    cannot see.
    """
    import zarr

    root = zarr.open_group(str(store_path), mode="r")
    for _, arr in root.arrays(recurse=True):
        if arr.size:
            arr[(0,) * arr.ndim]


def reintegrate_scopex_acq(subject: str, exp: str, loc: str, acq: str) -> str:
    """Archive old scopex outputs and regenerate them from the current mirror.

    For the channel-fix reintegration. Same treatment for every reprocessed
    acquisition (no Class A/B split):

    1. Guard: the ExSum mirror must already hold the source-of-truth
       (raw_mirror) file — i.e. the mirror swap happened.
    2. Delete ``event_detection/`` outright (retired pipeline; never
       regenerated).
    3. Move every other entry of the scopex acq directory (zarrs + version
       flags) into ``_pre_channel_fix_reintegration/``.
    4. Run the normal, unmodified scopex generation
       (:func:`wis.pns.scopex_mon.acq_data`).
    5. Verify the audit reports fresh AND every regenerated zarr is actually
       readable (guards against the half-written-zarr blind spot).

    Idempotent: returns ``"already_reintegrated"`` without touching anything
    when the archive exists and outputs are fresh. If a previous run crashed
    mid-generation (archive exists, outputs not fresh), the partial NEW
    outputs are deleted — never mixed into the archive — and generation
    reruns.

    Returns one of ``"reintegrated"``, ``"already_reintegrated"``,
    ``"already_fresh_no_archive"``.
    """
    tag = f"{subject} {exp} {loc} {acq}"

    # -- 1. mirror must hold the source-of-truth file ------------------------
    mirror_dir = Path(DEFS.exsum_mirror_root) / subject / exp / f"{loc}--{acq}"
    source_dir = (
        Path(DEFS.exsum_source_root) / subject / exp / loc / acq / "ExperimentSummary"
    )
    mirror_mats = sorted(mirror_dir.glob("*.mat")) if mirror_dir.is_dir() else []
    source_mats = sorted(source_dir.glob("*.mat")) if source_dir.is_dir() else []
    if len(mirror_mats) != 1 or len(source_mats) != 1:
        raise RuntimeError(
            f"[{tag}] expected exactly one mirror and one source .mat "
            f"(found {len(mirror_mats)} / {len(source_mats)})"
        )
    if mirror_mats[0].name != source_mats[0].name:
        raise RuntimeError(
            f"[{tag}] mirror ({mirror_mats[0].name}) != source "
            f"({source_mats[0].name}) — run the mirror swap (annotation "
            "reintegration) before regenerating scopex"
        )

    scopex_dir = Path(wis.pns.scopex_mon._scopex_dir(subject, exp, loc, acq))
    archive = scopex_dir / SCOPEX_ARCHIVE_DIRNAME

    # -- 2. event_detection is deleted outright ------------------------------
    ev_dir = scopex_dir / "event_detection"
    if ev_dir.exists():
        shutil.rmtree(ev_dir)
        print(f"[{tag}] deleted {ev_dir}")

    # -- 3. archive old outputs (or clean up a crashed rerun) ----------------
    pre_status = wis.pns.scopex_mon.check_outputs_status(subject, exp, loc, acq)
    live_entries = (
        [p for p in sorted(scopex_dir.iterdir()) if p.name != SCOPEX_ARCHIVE_DIRNAME]
        if scopex_dir.is_dir()
        else []
    )

    if pre_status.status == "fresh":
        if archive.exists():
            print(f"[{tag}] already reintegrated (archive present, outputs fresh)")
            return "already_reintegrated"
        print(
            f"[{tag}] outputs already fresh against {pre_status.esum_version} "
            "but no archive exists — nothing to do (old outputs were already "
            "replaced without archiving)"
        )
        return "already_fresh_no_archive"

    if archive.exists():
        # Crashed previous run: the old data is in the archive; whatever is
        # live is partial regeneration output. Delete it, never archive it.
        for p in live_entries:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            print(f"[{tag}] removed partial output {p.name}")
    elif live_entries:
        archive.mkdir(parents=True)
        for p in live_entries:
            shutil.move(str(p), str(archive / p.name))
        print(
            f"[{tag}] archived {len(live_entries)} entries into "
            f"{SCOPEX_ARCHIVE_DIRNAME}/"
        )
    else:
        print(f"[{tag}] no existing scopex outputs to archive")

    # -- 4. normal scopex generation, unmodified ------------------------------
    wis.pns.scopex_mon.acq_data(subject, exp, loc, acq)

    # -- 5. verify -------------------------------------------------------------
    post = wis.pns.scopex_mon.check_outputs_status(subject, exp, loc, acq)
    if post.status != "fresh":
        raise RuntimeError(
            f"[{tag}] scopex outputs not fresh after regeneration "
            f"(status {post.status}; missing {post.missing_zarrs}, "
            f"stale {post.stale_zarrs})"
        )
    for name in post.expected_zarrs:
        _zarr_store_readable(scopex_dir / f"{name}.zarr")
    print(
        f"[{tag}] REINTEGRATED — {len(post.expected_zarrs)} zarrs fresh against "
        f"{post.esum_version} and readable"
    )
    return "reintegrated"


def verify_classA_acquisition(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    update_tracking: bool = True,
) -> ReintegrationVerification:
    """Validate a fully-reintegrated Class A acquisition.

    Runs every check, printing a checklist. Only if ALL checks pass (and
    ``update_tracking`` is True) is the acquisition's
    ``annotation_materials_verified`` column in the master tracking table set
    to True. On any failure the tracking table is left untouched.

    Checks: Class A membership per the tracking table; mirror holds the
    source-of-truth file; annotation audit fresh; labels present and
    count-valid per DMD (never-labeled acqs are skipped); canvas strokes,
    prox lines, and ROI masks consistent with the regenerated geometry.
    """
    return _verify_acquisition(
        subject, exp, loc, acq, expected_class="A", update_tracking=update_tracking
    )


def verify_classB_acquisition(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    update_tracking: bool = True,
) -> ReintegrationVerification:
    """Validate a fully-migrated Class B acquisition.

    Same structural checks as Class A (membership is proven by the presence
    of ``classB_staging/staging_meta.json`` rather than the volatile
    ``identical_footprints`` column), plus overlay previews of the migrated
    annotations are rendered into ``classB_staging/previews/`` for visual
    review. On full pass, stamps ``annotation_materials_verified``.

    Verification does NOT require complete labels — unmatched new sources
    stay blank until labeled in the roladex (tracked by the WISynaptic
    status monitor's Label Completeness column).
    """
    return _verify_acquisition(
        subject, exp, loc, acq, expected_class="B", update_tracking=update_tracking
    )


def verify_onboarded_acquisition(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    update_tracking: bool = True,
) -> ReintegrationVerification:
    """Validate a freshly-onboarded acquisition (no prior ExSum existed).

    For acquisitions whose preprocessing succeeded for the FIRST time during
    the channel-fix rerun: there is nothing to restore or migrate, so the
    pipeline is plain onboarding (mirror copy + annotation generation), and
    membership means the tracking row has NO ``identical_footprints`` value
    (no old ExSum to compare against).

    Runs the same structural checks as Class A; the human-annotation checks
    all resolve to "skip" on a fresh acq (no labels, strokes, prox lines, or
    masks exist yet — that work is tracked by the WISynaptic status monitor
    as for any new acquisition). On full pass, stamps
    ``annotation_materials_verified`` so the scopex regeneration loop picks
    the acq up.
    """
    return _verify_acquisition(
        subject, exp, loc, acq, expected_class="onboard", update_tracking=update_tracking
    )
