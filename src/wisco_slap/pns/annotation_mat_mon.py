"""Audit, repair, and orchestrate annotation materials generation.

This module is the decision-maker for annotation materials. It distinguishes:

- missing outputs
- present but stale outputs built from an old ExperimentSummary mirror
- unsupported mirrors that cannot expose the current synapse map
- manual-review cases where the current geometry conflicts with existing
  human spatial annotations

The pure writer functions in ``annotation_materials.py`` should not contain
orchestration logic. They simply write the generator-owned outputs when called.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import slap2_py as spy
import tifffile as tiff

import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.pns import annotation_materials

GENERATED_FROM_ESUM = ".generated_from_exsum.txt"
NUMERIC_PNG_RE = re.compile(r"^(\d+)\.png$")
SPATIAL_HUMAN_PATTERNS = (
    "canvas/*.annotations.json",
    "source_sorting/prox_lines_dmd*.csv",
    "roi_locations/*mask*.tif",
)
AUDIT_DF_SCHEMA = {
    "subject": pl.String,
    "exp": pl.String,
    "loc": pl.String,
    "acq": pl.String,
    "status": pl.String,
    "esum_version": pl.String,
    "dmd1_expected_sources": pl.Int64,
    "dmd1_found_images": pl.Int64,
    "dmd2_expected_sources": pl.Int64,
    "dmd2_found_images": pl.Int64,
    "can_auto_repair": pl.Boolean,
    "basics_missing": pl.Boolean,
    "canvas_status": pl.String,
    "synapse_ids_status": pl.String,
    "roi_locations_status": pl.String,
    "stale_components": pl.String,
    "missing_components": pl.String,
    "planned_actions": pl.String,
    "reasons": pl.String,
    "spatial_artifacts": pl.String,
    "geometry_mismatches": pl.String,
    "legacy_unversioned": pl.Boolean,
}


@dataclass(slots=True)
class ComponentAudit:
    """Audit result for one generator-owned component."""

    name: str
    status: str
    has_marker: bool
    marker_version: str | None = None
    missing_files: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(slots=True)
class AnnotationAcqAudit:
    """Notebook-friendly per-acquisition audit status."""

    subject: str
    exp: str
    loc: str
    acq: str
    status: str
    esum_path: str | None
    esum_version: str | None
    can_auto_repair: bool
    dmd1_expected_sources: int = -1
    dmd1_found_images: int = -1
    dmd2_expected_sources: int = -1
    dmd2_found_images: int = -1
    basics_missing: bool = False
    canvas_status: str = "unknown"
    synapse_ids_status: str = "unknown"
    roi_locations_status: str = "unknown"
    stale_components: tuple[str, ...] = ()
    missing_components: tuple[str, ...] = ()
    planned_actions: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    spatial_artifacts: tuple[str, ...] = ()
    geometry_mismatches: tuple[str, ...] = ()
    legacy_unversioned: bool = False

    def to_record(self) -> dict[str, object]:
        """Convert to a flat record suitable for a polars DataFrame."""
        return {
            "subject": self.subject,
            "exp": self.exp,
            "loc": self.loc,
            "acq": self.acq,
            "status": self.status,
            "esum_version": self.esum_version,
            "dmd1_expected_sources": self.dmd1_expected_sources,
            "dmd1_found_images": self.dmd1_found_images,
            "dmd2_expected_sources": self.dmd2_expected_sources,
            "dmd2_found_images": self.dmd2_found_images,
            "can_auto_repair": self.can_auto_repair,
            "basics_missing": self.basics_missing,
            "canvas_status": self.canvas_status,
            "synapse_ids_status": self.synapse_ids_status,
            "roi_locations_status": self.roi_locations_status,
            "stale_components": ",".join(self.stale_components),
            "missing_components": ",".join(self.missing_components),
            "planned_actions": " | ".join(self.planned_actions),
            "reasons": " | ".join(self.reasons),
            "spatial_artifacts": " | ".join(self.spatial_artifacts),
            "geometry_mismatches": " | ".join(self.geometry_mismatches),
            "legacy_unversioned": self.legacy_unversioned,
        }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _acq_dir(subject: str, exp: str, loc: str, acq: str) -> Path:
    """Return the annotation materials directory for a given acquisition."""
    return Path(DEFS.annotation_root) / subject / exp / loc / acq


def _component_dir(acq_root: Path, component: str) -> Path:
    return acq_root / component


def _marker_path(component_dir: Path) -> Path:
    return component_dir / GENERATED_FROM_ESUM


def _attachments_backup_root(acq_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return acq_root / "attachments" / "stale_backups" / timestamp


def _relpath(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_esum_mirror(subject: str, exp: str, loc: str, acq: str) -> str | None:
    """Check whether a valid ExperimentSummary mirror exists."""
    try:
        esum_path = wis.meta.get.esum_mirror_path(subject, exp, loc, acq)
    except (FileNotFoundError, ValueError):
        return None
    if esum_path == "NO_ESUM_MIRROR":
        return None
    return esum_path


def _get_esum_version(esum_path: str) -> str:
    """Extract the ExperimentSummary version string from a mirror path."""
    return os.path.basename(esum_path).split(".mat")[0]


def _read_marker(component_dir: Path) -> str | None:
    marker = _marker_path(component_dir)
    if not marker.exists():
        return None
    return marker.read_text(encoding="utf-8").strip() or None


def _write_marker(component_dir: Path, esum_version: str) -> None:
    wis.util.check_dir(str(component_dir))
    _marker_path(component_dir).write_text(f"{esum_version}\n", encoding="utf-8")


def _expected_source_count(synmap: np.ndarray) -> int:
    return max(int(np.max(synmap)), 0)


def _numeric_png_ids(dmd_dir: Path) -> list[int]:
    if not dmd_dir.exists():
        return []

    ids: list[int] = []
    for fname in os.listdir(dmd_dir):
        match = NUMERIC_PNG_RE.match(fname)
        if match is not None:
            ids.append(int(match.group(1)))
    return sorted(ids)


def _expected_numeric_png_ids(n_sources: int) -> list[int]:
    return list(range(n_sources))


def _expected_numeric_png_label(n_sources: int) -> str:
    if n_sources == 0:
        return "[no numbered PNGs expected]"
    return f"[0..{n_sources - 1}].png"


def _load_existing_key(key_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(key_path) as data:
        label_map = data["label_map"]
        id_list = data["id_list"]
    return label_map, id_list.astype(np.str_)


def _expected_id_list(n_sources: int) -> np.ndarray:
    return annotation_materials.build_synapse_id_list(n_sources)


def _missing_annotation_basics(acq_root: Path) -> tuple[str, ...]:
    missing: list[str] = []
    for fname in ("materials.txt", "notes.md"):
        path = acq_root / fname
        if not path.exists():
            missing.append(_relpath(path, acq_root))
    return tuple(missing)


def _required_canvas_paths(acq_root: Path) -> tuple[Path, ...]:
    return (
        acq_root / "canvas" / "DMD-1.png",
        acq_root / "canvas" / "DMD-2.png",
        acq_root / "canvas" / "syn_overlays" / "DMD-1.png",
        acq_root / "canvas" / "syn_overlays" / "DMD-2.png",
    )


def _required_roi_paths(acq_root: Path) -> tuple[Path, ...]:
    return (
        acq_root / "roi_locations" / "roi_locs_dmd1.tif",
        acq_root / "roi_locations" / "roi_locs_dmd2.tif",
    )


def _missing_paths(paths: tuple[Path, ...], root: Path) -> tuple[str, ...]:
    return tuple(_relpath(path, root) for path in paths if not path.exists())


def _human_spatial_artifacts(acq_root: Path) -> tuple[str, ...]:
    artifacts: list[str] = []
    for pattern in SPATIAL_HUMAN_PATTERNS:
        for path in sorted(acq_root.glob(pattern)):
            artifacts.append(_relpath(path, acq_root))
    return tuple(artifacts)


def _geometry_mismatches(
    acq_root: Path,
    expected_synmaps: dict[int, np.ndarray],
) -> tuple[str, ...]:
    mismatches: list[str] = []

    for dmd in [1, 2]:
        expected_shape = tuple(expected_synmaps[dmd].shape)

        key_path = acq_root / "synapse_ids" / f"dmd-{dmd}" / "source_location_key.npz"
        if key_path.exists():
            try:
                label_map, _ = _load_existing_key(key_path)
            except Exception as exc:
                mismatches.append(
                    f"{_relpath(key_path, acq_root)} could not be read ({exc})"
                )
            else:
                if tuple(label_map.shape) != expected_shape:
                    mismatches.append(
                        f"{_relpath(key_path, acq_root)} shape "
                        f"{tuple(label_map.shape)} "
                        f"!= current footprint map shape {expected_shape}"
                    )

        roi_path = acq_root / "roi_locations" / f"roi_locs_dmd{dmd}.tif"
        if roi_path.exists():
            try:
                roi_img = tiff.imread(roi_path)
            except Exception as exc:
                mismatches.append(
                    f"{_relpath(roi_path, acq_root)} could not be read ({exc})"
                )
            else:
                if tuple(roi_img.shape[:2]) != expected_shape:
                    mismatches.append(
                        f"{_relpath(roi_path, acq_root)} shape "
                        f"{tuple(roi_img.shape[:2])} "
                        f"!= current footprint map shape {expected_shape}"
                    )

    return tuple(mismatches)


def _audit_synapse_ids_component(
    acq_root: Path,
    expected_synmaps: dict[int, np.ndarray],
    esum_version: str,
) -> tuple[ComponentAudit, dict[int, int]]:
    syn_dir = _component_dir(acq_root, "synapse_ids")
    marker_version = _read_marker(syn_dir)
    expected_counts = {
        dmd: _expected_source_count(expected_synmaps[dmd]) for dmd in [1, 2]
    }
    found_counts = {dmd: 0 for dmd in [1, 2]}

    if marker_version is not None:
        missing_files: list[str] = []
        for dmd in [1, 2]:
            dmd_dir = syn_dir / f"dmd-{dmd}"
            found_counts[dmd] = len(_numeric_png_ids(dmd_dir))
            if not (dmd_dir / "master_image.png").exists():
                missing_files.append(_relpath(dmd_dir / "master_image.png", acq_root))
            if not (dmd_dir / "source_location_key.npz").exists():
                missing_files.append(
                    _relpath(dmd_dir / "source_location_key.npz", acq_root)
                )
            expected_pngs = _expected_numeric_png_ids(expected_counts[dmd])
            if _numeric_png_ids(dmd_dir) != expected_pngs:
                missing_files.append(
                    f"{_relpath(dmd_dir, acq_root)}/"
                    f"{_expected_numeric_png_label(expected_counts[dmd])}"
                )

        if marker_version == esum_version and not missing_files:
            status = "fresh"
            reasons: tuple[str, ...] = ()
        elif marker_version != esum_version:
            status = "stale"
            reasons = (
                "marker version "
                f"{marker_version} != current mirror version {esum_version}",
            )
        else:
            status = "missing"
            reasons = ("marker matches current mirror but generated files are missing",)

        return (
            ComponentAudit(
                name="synapse_ids",
                status=status,
                has_marker=True,
                marker_version=marker_version,
                missing_files=tuple(missing_files),
                reasons=reasons,
            ),
            found_counts,
        )

    missing_files: list[str] = []
    stale_reasons: list[str] = []

    for dmd in [1, 2]:
        dmd_dir = syn_dir / f"dmd-{dmd}"
        key_path = dmd_dir / "source_location_key.npz"
        master_path = dmd_dir / "master_image.png"
        numeric_ids = _numeric_png_ids(dmd_dir)
        found_counts[dmd] = len(numeric_ids)
        expected_count = expected_counts[dmd]

        if not master_path.exists():
            missing_files.append(_relpath(master_path, acq_root))
        if not key_path.exists():
            missing_files.append(_relpath(key_path, acq_root))

        expected_pngs = _expected_numeric_png_ids(expected_count)
        has_legacy_synapse_content = (
            bool(numeric_ids) or key_path.exists() or master_path.exists()
        )
        if numeric_ids != expected_pngs and has_legacy_synapse_content:
            found_preview = numeric_ids[:5]
            expected_preview = expected_pngs[:5]
            found_suffix = "..." if len(numeric_ids) > 5 else ""
            expected_suffix = "..." if len(expected_pngs) > 5 else ""
            stale_reasons.append(
                f"dmd-{dmd} numbered PNGs {found_preview}{found_suffix} "
                f"!= expected source IDs {expected_preview}{expected_suffix}"
            )

        if not key_path.exists():
            continue

        try:
            label_map, id_list = _load_existing_key(key_path)
        except Exception as exc:
            stale_reasons.append(
                f"{_relpath(key_path, acq_root)} could not be read ({exc})"
            )
            continue

        expected_synmap = expected_synmaps[dmd]
        if tuple(label_map.shape) != tuple(expected_synmap.shape):
            stale_reasons.append(
                f"dmd-{dmd} label map shape {tuple(label_map.shape)} "
                f"!= current footprint map shape {tuple(expected_synmap.shape)}"
            )
            continue

        if not np.array_equal(label_map, expected_synmap):
            stale_reasons.append(
                f"dmd-{dmd} stored label map does not match current footprint map"
            )

        expected_ids = _expected_id_list(expected_count)
        if not np.array_equal(id_list, expected_ids):
            stale_reasons.append(
                f"dmd-{dmd} stored ID list does not match current source IDs"
            )

    if stale_reasons:
        status = "stale"
    elif missing_files:
        status = "missing"
    else:
        status = "fresh"

    return (
        ComponentAudit(
            name="synapse_ids",
            status=status,
            has_marker=False,
            marker_version=None,
            missing_files=tuple(missing_files),
            reasons=tuple(stale_reasons),
        ),
        found_counts,
    )


def _audit_simple_component(
    component: str,
    acq_root: Path,
    required_paths: tuple[Path, ...],
    esum_version: str,
    legacy_synapse_status: str,
) -> ComponentAudit:
    component_dir = _component_dir(acq_root, component)
    marker_version = _read_marker(component_dir)
    missing_files = _missing_paths(required_paths, acq_root)

    if marker_version is not None:
        if marker_version == esum_version and not missing_files:
            status = "fresh"
            reasons: tuple[str, ...] = ()
        elif marker_version != esum_version:
            status = "stale"
            reasons = (
                "marker version "
                f"{marker_version} != current mirror version {esum_version}",
            )
        else:
            status = "missing"
            reasons = ("marker matches current mirror but generated files are missing",)

        return ComponentAudit(
            name=component,
            status=status,
            has_marker=True,
            marker_version=marker_version,
            missing_files=missing_files,
            reasons=reasons,
        )

    if missing_files:
        status = "missing"
        reasons = ("unversioned legacy output is incomplete",)
    elif legacy_synapse_status == "stale":
        status = "stale"
        reasons = (
            "unversioned legacy output is treated as stale because "
            "synapse_ids are stale",
        )
    else:
        status = "fresh"
        reasons = ()

    return ComponentAudit(
        name=component,
        status=status,
        has_marker=False,
        marker_version=None,
        missing_files=missing_files,
        reasons=reasons,
    )


def _build_audit_df(audits: list[AnnotationAcqAudit]) -> pl.DataFrame:
    if not audits:
        return pl.DataFrame(schema=AUDIT_DF_SCHEMA)
    return pl.DataFrame([audit.to_record() for audit in audits], schema=AUDIT_DF_SCHEMA)


def _component_needs_rebuild(component_status: str, overwrite: bool) -> bool:
    return overwrite or component_status in {"missing", "stale"}


def _archive_file(src: Path, backup_root: Path, acq_root: Path) -> str:
    dest = backup_root / src.relative_to(acq_root)
    wis.util.check_dir(str(dest.parent))
    shutil.move(str(src), str(dest))
    return f"archived {src.relative_to(acq_root)} -> {dest.relative_to(acq_root)}"


def _archive_stale_synapse_sidecars(acq_root: Path) -> list[str]:
    backup_root = _attachments_backup_root(acq_root)
    actions: list[str] = []

    for path in sorted(acq_root.glob("synapse_ids/dmd-*/synapse_labels.csv")):
        actions.append(_archive_file(path, backup_root, acq_root))

    for path in sorted(acq_root.glob("source_sorting/syn_topo_dmd*.json")):
        actions.append(_archive_file(path, backup_root, acq_root))

    return actions


def _clear_generated_synapse_outputs(acq_root: Path) -> list[str]:
    actions: list[str] = []
    for dmd in [1, 2]:
        dmd_dir = acq_root / "synapse_ids" / f"dmd-{dmd}"
        if not dmd_dir.exists():
            continue

        for path in sorted(dmd_dir.iterdir()):
            if path.name in {"master_image.png", "source_location_key.npz"}:
                path.unlink()
                actions.append(f"removed {_relpath(path, acq_root)}")
            elif path.is_file() and NUMERIC_PNG_RE.match(path.name):
                path.unlink()
                actions.append(f"removed {_relpath(path, acq_root)}")

    marker = _marker_path(acq_root / "synapse_ids")
    if marker.exists():
        marker.unlink()
        actions.append(f"removed {_relpath(marker, acq_root)}")

    return actions


def _render_actions(audit: AnnotationAcqAudit, overwrite: bool) -> tuple[str, ...]:
    actions: list[str] = []

    if audit.status in {
        "missing_mirror",
        "unsupported_exsum",
        "manual_review_required",
    }:
        return tuple(actions)

    if audit.basics_missing:
        actions.append("create missing materials.txt / notes.md")

    if _component_needs_rebuild(audit.canvas_status, overwrite):
        actions.append("regenerate canvas images and synapse overlays")

    if _component_needs_rebuild(audit.synapse_ids_status, overwrite):
        if audit.synapse_ids_status == "stale":
            actions.append("archive stale synapse_labels.csv files")
            actions.append("archive stale syn_topo_dmd*.json files")
        actions.append(
            "rebuild synapse_ids numeric PNGs, master_image.png, "
            "and source_location_key.npz"
        )

    if _component_needs_rebuild(audit.roi_locations_status, overwrite):
        actions.append("regenerate roi_locs_dmd*.tif")

    return tuple(actions)


# ---------------------------------------------------------------------------
# Audit API
# ---------------------------------------------------------------------------


def audit_acq(subject: str, exp: str, loc: str, acq: str) -> AnnotationAcqAudit:
    """Audit a single acquisition and classify annotation material freshness."""
    acq_root = _acq_dir(subject, exp, loc, acq)
    basics_missing = bool(_missing_annotation_basics(acq_root))

    esum_path = _check_esum_mirror(subject, exp, loc, acq)
    if esum_path is None:
        audit = AnnotationAcqAudit(
            subject=subject,
            exp=exp,
            loc=loc,
            acq=acq,
            status="missing_mirror",
            esum_path=None,
            esum_version=None,
            can_auto_repair=False,
            basics_missing=basics_missing,
            reasons=("no ExperimentSummary mirror found",),
        )
        audit.planned_actions = _render_actions(audit, overwrite=False)
        return audit

    esum_version = _get_esum_version(esum_path)

    try:
        expected_synmaps, _ = spy.xsum.get_fp_info(esum_path)
    except Exception as exc:
        audit = AnnotationAcqAudit(
            subject=subject,
            exp=exp,
            loc=loc,
            acq=acq,
            status="unsupported_exsum",
            esum_path=esum_path,
            esum_version=esum_version,
            can_auto_repair=False,
            basics_missing=basics_missing,
            reasons=(f"current mirror cannot expose footprint info ({exc})",),
        )
        audit.planned_actions = _render_actions(audit, overwrite=False)
        return audit

    synapse_audit, found_counts = _audit_synapse_ids_component(
        acq_root, expected_synmaps, esum_version
    )
    canvas_audit = _audit_simple_component(
        "canvas",
        acq_root,
        _required_canvas_paths(acq_root),
        esum_version,
        synapse_audit.status,
    )
    roi_audit = _audit_simple_component(
        "roi_locations",
        acq_root,
        _required_roi_paths(acq_root),
        esum_version,
        synapse_audit.status,
    )

    stale_components = tuple(
        comp.name
        for comp in (canvas_audit, synapse_audit, roi_audit)
        if comp.status == "stale"
    )
    missing_components = list(
        comp.name
        for comp in (canvas_audit, synapse_audit, roi_audit)
        if comp.status == "missing"
    )
    if basics_missing:
        missing_components.append("annotation_basics")
    generated_components_need_repair = bool(stale_components) or any(
        comp.name in {"canvas", "synapse_ids", "roi_locations"}
        for comp in (canvas_audit, synapse_audit, roi_audit)
        if comp.status == "missing"
    )

    reasons: list[str] = []
    for comp in (canvas_audit, synapse_audit, roi_audit):
        reasons.extend(comp.reasons)

    spatial_artifacts = _human_spatial_artifacts(acq_root)
    geometry_mismatches = _geometry_mismatches(acq_root, expected_synmaps)
    legacy_unversioned = any(
        comp.status == "fresh" and not comp.has_marker
        for comp in (canvas_audit, synapse_audit, roi_audit)
    )

    if geometry_mismatches and spatial_artifacts and generated_components_need_repair:
        status = "manual_review_required"
        can_auto_repair = False
        reasons.append(
            "current image geometry conflicts with existing human spatial annotations"
        )
    elif stale_components:
        status = "stale"
        can_auto_repair = True
    elif missing_components:
        status = "missing"
        can_auto_repair = True
    else:
        status = "fresh"
        can_auto_repair = False

    audit = AnnotationAcqAudit(
        subject=subject,
        exp=exp,
        loc=loc,
        acq=acq,
        status=status,
        esum_path=esum_path,
        esum_version=esum_version,
        can_auto_repair=can_auto_repair,
        dmd1_expected_sources=_expected_source_count(expected_synmaps[1]),
        dmd1_found_images=found_counts[1],
        dmd2_expected_sources=_expected_source_count(expected_synmaps[2]),
        dmd2_found_images=found_counts[2],
        basics_missing=basics_missing,
        canvas_status=canvas_audit.status,
        synapse_ids_status=synapse_audit.status,
        roi_locations_status=roi_audit.status,
        stale_components=stale_components,
        missing_components=tuple(missing_components),
        reasons=tuple(reasons),
        spatial_artifacts=spatial_artifacts,
        geometry_mismatches=geometry_mismatches,
        legacy_unversioned=legacy_unversioned,
    )
    audit.planned_actions = _render_actions(audit, overwrite=False)
    return audit


def audit_exp(subject: str, exp: str) -> pl.DataFrame:
    """Audit all acquisitions in one experiment and return a polars DataFrame."""
    audits = [
        audit_acq(subject, exp, *la.split("--"))
        for la in wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
    ]
    return _build_audit_df(audits)


def audit_all_subjects() -> pl.DataFrame:
    """Audit every acquisition in the project and return a polars DataFrame."""
    audits: list[AnnotationAcqAudit] = []
    si = wis.meta.get.sync_info()
    for subject in si:
        for exp in si[subject]:
            try:
                locacqs = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
            except (KeyError, FileNotFoundError):
                continue
            for la in locacqs:
                loc, acq = la.split("--")
                audits.append(audit_acq(subject, exp, loc, acq))
    return _build_audit_df(audits)


# ---------------------------------------------------------------------------
# Repair orchestration
# ---------------------------------------------------------------------------


def _print_action_block(tag: str, actions: tuple[str, ...], dry_run: bool) -> None:
    prefix = "DRY RUN" if dry_run else "RUN"
    if not actions:
        print(f"[{tag}] {prefix}: no generator-owned changes needed")
        return
    print(f"[{tag}] {prefix}:")
    for action in actions:
        print(f"  - {action}")


def acq_data(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    overwrite: bool = False,
    dry_run: bool = False,
) -> AnnotationAcqAudit:
    """Audit and, when safe, regenerate stale or missing annotation materials."""
    tag = f"{subject} {exp} {loc} {acq}"
    audit = audit_acq(subject, exp, loc, acq)
    audit.planned_actions = _render_actions(audit, overwrite=overwrite)

    if audit.status in {
        "missing_mirror",
        "unsupported_exsum",
        "manual_review_required",
    }:
        reason_text = "; ".join(audit.reasons) if audit.reasons else audit.status
        print(f"[{tag}] Skipping - {audit.status}: {reason_text}")
        return audit

    if not overwrite and audit.status == "fresh":
        return audit

    _print_action_block(tag, audit.planned_actions, dry_run=dry_run)
    if dry_run:
        return audit

    assert audit.esum_path is not None
    assert audit.esum_version is not None

    if audit.basics_missing:
        annotation_materials.create_annotation_basics(subject, exp, loc, acq)

    acq_root = _acq_dir(subject, exp, loc, acq)

    if _component_needs_rebuild(audit.canvas_status, overwrite):
        annotation_materials.save_acq_mean_images(
            subject,
            exp,
            loc,
            acq,
            audit.esum_path,
        )
        _write_marker(acq_root / "canvas", audit.esum_version)

    if _component_needs_rebuild(audit.synapse_ids_status, overwrite):
        if audit.synapse_ids_status == "stale":
            for action in _archive_stale_synapse_sidecars(acq_root):
                print(f"[{tag}] {action}")

        for action in _clear_generated_synapse_outputs(acq_root):
            print(f"[{tag}] {action}")

        annotation_materials.save_synapse_id_plots_and_key(
            subject,
            exp,
            loc,
            acq,
            audit.esum_path,
        )
        _write_marker(acq_root / "synapse_ids", audit.esum_version)

    if _component_needs_rebuild(audit.roi_locations_status, overwrite):
        annotation_materials.save_roi_location_tiffs(
            subject,
            exp,
            loc,
            acq,
            audit.esum_path,
        )
        _write_marker(acq_root / "roi_locations", audit.esum_version)

    return audit_acq(subject, exp, loc, acq)


def exp_data(
    subject: str,
    exp: str,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[AnnotationAcqAudit]:
    """Run stale-aware annotation repair across one experiment."""
    audits: list[AnnotationAcqAudit] = []
    for la in wis.meta.get.unique_acquisitions_per_experiment(subject, exp):
        loc, acq = la.split("--")
        audits.append(
            acq_data(
                subject,
                exp,
                loc,
                acq,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )
    return audits


def all_subjects(
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[AnnotationAcqAudit]:
    """Run stale-aware annotation repair across all acquisitions."""
    audits: list[AnnotationAcqAudit] = []
    si = wis.meta.get.sync_info()
    for subject in si:
        for exp in si[subject]:
            try:
                locacqs = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
            except (KeyError, FileNotFoundError):
                print(f"[{subject} {exp}] No acquisitions found, skipping")
                continue
            for la in locacqs:
                loc, acq = la.split("--")
                try:
                    audits.append(
                        acq_data(
                            subject,
                            exp,
                            loc,
                            acq,
                            overwrite=overwrite,
                            dry_run=dry_run,
                        )
                    )
                except Exception as exc:
                    print(f"[{subject} {exp} {loc} {acq}] Error: {exc}")
                    continue
    return audits
