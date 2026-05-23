"""Unified ExSum-derivative freshness audit across scopex + event detection
+ annotation materials.

This module is the single place that produces a combined per-acquisition
staleness row, and it is responsible for maintaining the canonical audit
CSV at ``analysis_materials/ExSum_audits.csv`` (next to
``acquisition_master.yaml``). The WISynaptic status monitor dashboard reads
this CSV to display ExSum freshness columns without having to itself
import the three underlying auditors.

Truth always comes from re-running :func:`refresh_exsum_audit`; the CSV is
a cache. Individual auditors (``scopex_mon``, ``glu_ev_basic_mon``,
``annotation_mat_mon``) remain the authoritative sources for their own
components and are called by the helpers here.

Component statuses are mapped into a shared 5-value vocabulary:

- ``"fresh"``
- ``"stale"``
- ``"missing"``
- ``"missing_mirror"``
- ``"not_applicable"``

The annotation auditor's additional ``"manual_review_required"`` state is
folded into ``"stale"`` with a clear note in the details field, since from
the consumer's perspective manual-review rows ARE stale — they just
happen to block auto-repair.

The table also includes lightweight sleep-scoring completion columns. These
are reported separately and intentionally do not affect ``overall_status``,
which remains an ExSum-derivative freshness rollup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.pns import (
    annotation_mat_mon,
    glu_ev_basic_mon,
    scopex_mon,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIT_CSV_PATH: Path = Path(DEFS.anmat_root) / "ExSum_audits.csv"

# Severity order for rolling up a single overall_status per acquisition.
# Higher values are more severe (take precedence when summarising).
_SEVERITY = {
    "fresh": 0,
    "not_applicable": 0,
    "missing": 2,
    "stale": 3,
    "missing_mirror": 4,
}


@dataclass(slots=True)
class ComponentStatus:
    """Per-component ExSum-freshness status, normalized across auditors."""

    status: str
    esum_version: str | None
    details: str = ""


@dataclass(slots=True)
class ScoringStatus:
    """Per-acquisition view of sync-block model-scoring outputs."""

    complete: bool
    status: str
    model_name: str | None
    sync_block: int | None
    details: str = ""


# ---------------------------------------------------------------------------
# Per-component status helpers
# ---------------------------------------------------------------------------


def scopex_status(subject: str, exp: str, loc: str, acq: str) -> ComponentStatus:
    """Freshness of the scopex zarrs for one acquisition.

    Thin wrapper around :func:`scopex_mon.check_outputs_status` that
    returns a normalized :class:`ComponentStatus`.
    """
    result = scopex_mon.check_outputs_status(subject, exp, loc, acq)

    if result.status == "missing_mirror":
        return ComponentStatus(
            status="missing_mirror",
            esum_version=result.esum_version,
            details="no ExperimentSummary mirror",
        )

    if result.status == "fresh":
        return ComponentStatus(
            status="fresh", esum_version=result.esum_version, details=""
        )

    pieces: list[str] = []
    if result.stale_zarrs:
        pieces.append("stale: " + ", ".join(result.stale_zarrs))
    if result.missing_zarrs:
        pieces.append("missing: " + ", ".join(result.missing_zarrs))
    return ComponentStatus(
        status=result.status,
        esum_version=result.esum_version,
        details="; ".join(pieces),
    )


def event_detection_status(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    mode: str = "matchfilt",
) -> ComponentStatus:
    """Freshness of event-detection outputs for one acquisition.

    Parameters
    ----------
    mode : {"matchfilt", "denoised"}
        Which pipeline variant to assess. Both modes can coexist.
    """
    result = glu_ev_basic_mon.check_outputs_status(
        subject, exp, loc, acq, mode=mode
    )

    if result.status == "missing_mirror":
        return ComponentStatus(
            status="missing_mirror",
            esum_version=result.esum_version,
            details="no ExperimentSummary mirror",
        )

    if result.status == "scopex_unavailable":
        # Treat as not_applicable: we can't meaningfully audit event
        # detection when the upstream scopex trace isn't available. The
        # WISynaptic dashboard already surfaces scopex status separately,
        # so we don't double-count here.
        return ComponentStatus(
            status="not_applicable",
            esum_version=result.esum_version,
            details=f"{mode}: upstream scopex zarr not available",
        )

    if result.status == "fresh":
        return ComponentStatus(
            status="fresh", esum_version=result.esum_version, details=""
        )

    pieces: list[str] = []
    if result.stale_files:
        pieces.append("stale: " + ", ".join(result.stale_files))
    if result.missing_files:
        pieces.append("missing: " + ", ".join(result.missing_files))
    detail_body = "; ".join(pieces)
    return ComponentStatus(
        status=result.status,
        esum_version=result.esum_version,
        details=f"{mode}: {detail_body}" if detail_body else f"{mode}",
    )


def annotation_status(
    subject: str, exp: str, loc: str, acq: str
) -> tuple[ComponentStatus, annotation_mat_mon.AnnotationAcqAudit]:
    """Freshness of annotation materials for one acquisition.

    Also returns the full :class:`~annotation_mat_mon.AnnotationAcqAudit`
    so callers can extract per-component detail (canvas / synapse_ids /
    roi_locations status, geometry mismatches, spatial artifacts, etc.).

    Mapping:

    - ``missing_mirror`` / ``unsupported_exsum`` → ``"missing_mirror"``
    - ``manual_review_required`` → ``"stale"`` (details call out the
      review requirement so the consumer knows not to trust auto-repair)
    - ``stale`` → ``"stale"``
    - ``missing`` → ``"missing"``
    - ``fresh`` → ``"fresh"``
    """
    audit = annotation_mat_mon.audit_acq(subject, exp, loc, acq)

    if audit.status in ("missing_mirror", "unsupported_exsum"):
        return (
            ComponentStatus(
                status="missing_mirror",
                esum_version=audit.esum_version,
                details="; ".join(audit.reasons) if audit.reasons else audit.status,
            ),
            audit,
        )

    if audit.status == "fresh":
        return (
            ComponentStatus(
                status="fresh", esum_version=audit.esum_version, details=""
            ),
            audit,
        )

    detail_pieces: list[str] = []
    if audit.status == "manual_review_required":
        detail_pieces.append("MANUAL REVIEW: geometry conflicts with human artifacts")
        mapped = "stale"
    elif audit.status == "stale":
        mapped = "stale"
    elif audit.status == "missing":
        mapped = "missing"
    else:
        mapped = audit.status  # pass through unknown statuses

    if audit.stale_components:
        detail_pieces.append("stale: " + ",".join(audit.stale_components))
    if audit.missing_components:
        detail_pieces.append("missing: " + ",".join(audit.missing_components))
    if audit.geometry_mismatches:
        detail_pieces.append(
            "geometry: "
            + "; ".join(audit.geometry_mismatches[:2])
            + ("..." if len(audit.geometry_mismatches) > 2 else "")
        )

    return (
        ComponentStatus(
            status=mapped,
            esum_version=audit.esum_version,
            details=" | ".join(detail_pieces),
        ),
        audit,
    )


def _model_scored_dir(subject: str, exp: str, sync_block: int) -> Path:
    return (
        Path(DEFS.anmat_root)
        / subject
        / exp
        / "sync_block_data"
        / f"sync_block-{sync_block}"
        / "hypnograms"
        / "model_scored"
    )


def _read_model_version_metadata(path: Path) -> dict[str, str]:
    """Read key-value model metadata, accepting legacy one-line files too."""
    text = path.read_text().strip()
    if not text:
        return {}

    metadata: dict[str, str] = {}
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        if "=" not in line:
            metadata.setdefault("model_name", line)
            continue
        key, value = line.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def scoring_status(subject: str, exp: str, loc: str, acq: str) -> ScoringStatus:
    """Check whether the sync block containing this acquisition has model scores."""
    try:
        sync_block = int(wis.meta.sync.get_acq_sync_block(subject, exp, loc, acq))
    except Exception as exc:  # noqa: BLE001 - status monitor should not crash
        return ScoringStatus(
            complete=False,
            status="missing",
            model_name=None,
            sync_block=None,
            details=f"could not resolve sync block: {type(exc).__name__}: {exc}",
        )

    scored_dir = _model_scored_dir(subject, exp, sync_block)
    if not scored_dir.is_dir():
        return ScoringStatus(
            complete=False,
            status="missing",
            model_name=None,
            sync_block=sync_block,
            details=f"missing model_scored directory: {scored_dir}",
        )

    required_files = [
        "bout_df.parquet",
        "epoch_df.parquet",
        "bout_hypno.csv",
        "epoch_hypno.csv",
        "model_version.txt",
    ]
    missing_files = [
        name for name in required_files if not (scored_dir / name).is_file()
    ]
    if missing_files:
        return ScoringStatus(
            complete=False,
            status="missing",
            model_name=None,
            sync_block=sync_block,
            details="missing: " + ",".join(missing_files),
        )

    metadata = _read_model_version_metadata(scored_dir / "model_version.txt")
    model_name = metadata.get("model_name")
    details = f"sync_block-{sync_block}"
    if "add_wake_epochs" in metadata:
        details += f"; add_wake_epochs={metadata['add_wake_epochs']}"
    if not model_name:
        details += "; model_version.txt does not include model_name"

    return ScoringStatus(
        complete=True,
        status="fresh",
        model_name=model_name or "UNKNOWN_MODEL",
        sync_block=sync_block,
        details=details,
    )


# ---------------------------------------------------------------------------
# Combined per-acquisition row
# ---------------------------------------------------------------------------


_ROW_SCHEMA: dict[str, pl.PolarsDataType] = {
    "subject": pl.String,
    "exp": pl.String,
    "loc": pl.String,
    "acq": pl.String,
    "esum_version": pl.String,
    "overall_status": pl.String,
    "scopex_status": pl.String,
    "scopex_details": pl.String,
    "events_matchfilt_status": pl.String,
    "events_matchfilt_details": pl.String,
    "events_denoised_status": pl.String,
    "events_denoised_details": pl.String,
    "annotation_status": pl.String,
    "annotation_details": pl.String,
    "annotation_stale_components": pl.String,
    "annotation_missing_components": pl.String,
    "annotation_geometry_mismatches": pl.String,
    "annotation_spatial_artifacts": pl.String,
    "annotation_can_auto_repair": pl.Boolean,
    "scoring_complete": pl.Boolean,
    "scoring_status": pl.String,
    "scoring_model_name": pl.String,
    "scoring_sync_block": pl.Int64,
    "scoring_details": pl.String,
    "audited_at": pl.String,
}


def _overall_status(statuses: list[str]) -> str:
    """Roll multiple component statuses up into a single overall status."""
    if not statuses:
        return "fresh"
    severities = [_SEVERITY.get(s, 0) for s in statuses]
    max_sev = max(severities)
    for s in statuses:
        if _SEVERITY.get(s, 0) == max_sev:
            return s
    return "fresh"


def acq_status_row(subject: str, exp: str, loc: str, acq: str) -> dict:
    """Build a single flat dict row summarizing all ExSum-derivative
    freshness for one acquisition. Suitable for a polars DataFrame."""
    sx = scopex_status(subject, exp, loc, acq)
    ev_m = event_detection_status(subject, exp, loc, acq, mode="matchfilt")
    ev_d = event_detection_status(subject, exp, loc, acq, mode="denoised")
    ann, ann_audit = annotation_status(subject, exp, loc, acq)
    score = scoring_status(subject, exp, loc, acq)

    # Pick the version string from whichever component found one.
    esum_version = (
        sx.esum_version
        or ev_m.esum_version
        or ev_d.esum_version
        or ann.esum_version
    )

    overall = _overall_status(
        [sx.status, ev_m.status, ev_d.status, ann.status]
    )

    return {
        "subject": subject,
        "exp": exp,
        "loc": loc,
        "acq": acq,
        "esum_version": esum_version,
        "overall_status": overall,
        "scopex_status": sx.status,
        "scopex_details": sx.details,
        "events_matchfilt_status": ev_m.status,
        "events_matchfilt_details": ev_m.details,
        "events_denoised_status": ev_d.status,
        "events_denoised_details": ev_d.details,
        "annotation_status": ann.status,
        "annotation_details": ann.details,
        "annotation_stale_components": ",".join(ann_audit.stale_components),
        "annotation_missing_components": ",".join(ann_audit.missing_components),
        "annotation_geometry_mismatches": " | ".join(
            ann_audit.geometry_mismatches
        ),
        "annotation_spatial_artifacts": " | ".join(ann_audit.spatial_artifacts),
        "annotation_can_auto_repair": ann_audit.can_auto_repair,
        "scoring_complete": score.complete,
        "scoring_status": score.status,
        "scoring_model_name": score.model_name,
        "scoring_sync_block": score.sync_block,
        "scoring_details": score.details,
        "audited_at": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Full audit + CSV I/O
# ---------------------------------------------------------------------------


def refresh_exsum_audit(csv_path: Path | str | None = None) -> pl.DataFrame:
    """Audit every acquisition in ``acquisition_master.yaml`` and write the
    combined staleness table to ``ExSum_audits.csv``.

    The CSV is written atomically (write to ``.tmp``, then ``os.replace``)
    so that a crashed refresh leaves the previous CSV intact.

    Parameters
    ----------
    csv_path : str or Path, optional
        Destination CSV path. Defaults to :data:`AUDIT_CSV_PATH`, which is
        ``analysis_materials/ExSum_audits.csv`` next to
        ``acquisition_master.yaml``.

    Returns
    -------
    pl.DataFrame
        The audit table, with one row per acquisition.
    """
    target = Path(csv_path) if csv_path is not None else AUDIT_CSV_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    master = wis.meta.get.acq_master()
    for subject in master:
        for exp in master[subject]:
            for acq_id in master[subject][exp]:
                loc, acq = acq_id.split("--")
                try:
                    rows.append(acq_status_row(subject, exp, loc, acq))
                except Exception as exc:  # noqa: BLE001 - intentionally broad
                    print(
                        f"[{subject} {exp} {loc} {acq}] audit error: {exc}"
                    )
                    rows.append(
                        {
                            "subject": subject,
                            "exp": exp,
                            "loc": loc,
                            "acq": acq,
                            "esum_version": None,
                            "overall_status": "error",
                            "scopex_status": "error",
                            "scopex_details": str(exc),
                            "events_matchfilt_status": "error",
                            "events_matchfilt_details": str(exc),
                            "events_denoised_status": "error",
                            "events_denoised_details": str(exc),
                            "annotation_status": "error",
                            "annotation_details": str(exc),
                            "annotation_stale_components": "",
                            "annotation_missing_components": "",
                            "annotation_geometry_mismatches": "",
                            "annotation_spatial_artifacts": "",
                            "annotation_can_auto_repair": False,
                            "scoring_complete": False,
                            "scoring_status": "error",
                            "scoring_model_name": None,
                            "scoring_sync_block": None,
                            "scoring_details": str(exc),
                            "audited_at": datetime.now().isoformat(
                                timespec="seconds"
                            ),
                        }
                    )

    df = pl.DataFrame(rows, schema=_ROW_SCHEMA)

    tmp = target.with_suffix(target.suffix + ".tmp")
    df.write_csv(tmp)
    os.replace(tmp, target)
    print(f"Wrote ExSum audit for {df.height} acquisitions to {target}")
    return df


def read_exsum_audit(csv_path: Path | str | None = None) -> pl.DataFrame:
    """Read the cached ExSum audit CSV.

    Raises
    ------
    FileNotFoundError
        If the CSV does not yet exist. Call :func:`refresh_exsum_audit`
        first.
    """
    target = Path(csv_path) if csv_path is not None else AUDIT_CSV_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"No ExSum audit CSV at {target}. "
            f"Run wis.meta.status.refresh_exsum_audit() to create it."
        )
    return pl.read_csv(target, schema_overrides=_ROW_SCHEMA)
