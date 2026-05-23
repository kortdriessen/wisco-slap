"""Unified automatic metadata updater.

Refreshes every automatically-maintained metadata artifact under
``analysis_materials/`` in the correct dependency order, with per-scope
error isolation. A single call — ``wis.meta.update()`` — iterates
``acquisition_master.yaml`` and brings the following into sync:

1. ``ExSum_mirrors/`` (and ``prepro_info.yaml`` as a side effect)
2. ``sync_info.yaml`` (per subject+experiment)
3. ``epoch_info.yaml`` (per subject+experiment; multi-epoch acqs only)
4. ``dmd_info.yaml`` (per acquisition)
5. ``ExSum_audits.csv`` (staleness table; optional via ``refresh_audit``)

Not touched by this module (intentionally):

- ``acquisition_master.yaml`` — source of truth, maintained by hand.
- ``sb_scoring_times.yaml`` — user-curated sleep-scoring time windows.
- Annotation materials, scopex zarrs, event detection, sync-block data
  — these are content, not meta. Each has its own monitor/orchestrator
  (``annotation_mat_mon``, ``scopex_mon``, ``glu_ev_basic_mon``,
  ``sync_block_mon``); they are deliberately NOT bundled here so that
  a meta refresh stays cheap.

Error handling
--------------

Any exception raised by a per-acq or per-exp sub-step is caught, its
traceback written to
``analysis_materials/meta_issues/<subject>/<exp>/<loc>/<acq>/<stage>__<ts>.txt``
(or ``.../__exp__/...`` for experiment-level failures), and the sweep
continues with the next scope. At the start of each stage, any prior
error files for that scope + stage are deleted first — so a successful
rerun automatically cleans up the ``meta_issues/`` trail. Empty
directories in ``meta_issues/`` are pruned back toward the root.

The presence of a file in ``meta_issues/`` always means "still broken."
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import wisco_slap as wis
import wisco_slap.defs as DEFS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

META_ISSUES_ROOT: Path = Path(DEFS.anmat_root) / "meta_issues"
EXP_SCOPE_NAME = "__exp__"

STAGE_MIRROR = "mirror_exsum"
STAGE_SYNC_INFO = "sync_info"
STAGE_EPOCH_INFO = "epoch_info"
STAGE_DMD_INFO = "dmd_info"
STAGE_AUDIT = "exsum_audit"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ErrorEntry:
    """One recorded error from a ``wis.meta.update()`` run."""

    stage: str
    subject: str
    exp: str
    loc: str | None  # None for experiment-scoped errors
    acq: str | None
    error_file: Path
    exc_type: str
    exc_message: str


@dataclass(slots=True)
class StageSummary:
    """Per-stage attempt/error totals."""

    stage: str
    attempted: int = 0
    errors: int = 0


@dataclass(slots=True)
class MetaUpdateResult:
    """Return value of :func:`update`.

    Attributes
    ----------
    stages : dict[str, StageSummary]
        Per-stage counts.
    errors : list[ErrorEntry]
        Every error captured this run.
    error_dirs : list[Path]
        Unique ``meta_issues/`` scope directories with remaining errors.
    audit_csv_path : Path | None
        Path to ``ExSum_audits.csv``, or None if ``refresh_audit=False``.
    """

    stages: dict[str, StageSummary] = field(default_factory=dict)
    errors: list[ErrorEntry] = field(default_factory=list)
    error_dirs: list[Path] = field(default_factory=list)
    audit_csv_path: Path | None = None


# ---------------------------------------------------------------------------
# meta_issues/ helpers
# ---------------------------------------------------------------------------


def _scope_dir(
    subject: str, exp: str, loc: str | None, acq: str | None
) -> Path:
    """Return the ``meta_issues/`` directory for a given scope.

    Per-acq scope → ``meta_issues/<subject>/<exp>/<loc>/<acq>/``
    Experiment-level scope → ``meta_issues/<subject>/<exp>/__exp__/``
    """
    if loc is not None and acq is not None:
        return META_ISSUES_ROOT / subject / exp / loc / acq
    return META_ISSUES_ROOT / subject / exp / EXP_SCOPE_NAME


def _clear_stage_errors(
    subject: str, exp: str, loc: str | None, acq: str | None, stage: str
) -> None:
    """Remove any existing ``<stage>__*.txt`` files for the given scope.

    Called at the start of each stage-scope attempt so a successful
    rerun leaves no stale error files behind. Prunes empty directories
    back toward ``meta_issues/``.
    """
    d = _scope_dir(subject, exp, loc, acq)
    if not d.exists():
        return
    for p in d.glob(f"{stage}__*.txt"):
        p.unlink()
    _prune_empty(d)


def _prune_empty(path: Path) -> None:
    """Remove ``path`` and any empty parents up to (but not including)
    ``META_ISSUES_ROOT``."""
    try:
        while (
            path != META_ISSUES_ROOT
            and path.is_dir()
            and not any(path.iterdir())
        ):
            path.rmdir()
            path = path.parent
    except FileNotFoundError:
        pass


def _write_error(
    stage: str,
    subject: str,
    exp: str,
    loc: str | None,
    acq: str | None,
    exc: BaseException,
) -> Path:
    """Write an error traceback to ``meta_issues/.../<stage>__<ts>.txt``.

    Returns the path to the file that was written.
    """
    d = _scope_dir(subject, exp, loc, acq)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = d / f"{stage}__{ts}.txt"
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    scope_desc = (
        f"{loc}/{acq}" if (loc and acq) else "(experiment-level)"
    )
    path.write_text(
        f"Stage: {stage}\n"
        f"Subject: {subject}\n"
        f"Exp: {exp}\n"
        f"Scope: {scope_desc}\n"
        f"Time: {datetime.now().isoformat(timespec='seconds')}\n"
        f"\n"
        f"Exception: {type(exc).__name__}: {exc}\n"
        f"\n"
        f"Traceback:\n{tb}",
        encoding="utf-8",
    )
    return path


def _record_error(
    result: MetaUpdateResult,
    stage: str,
    subject: str,
    exp: str,
    loc: str | None,
    acq: str | None,
    exc: BaseException,
) -> None:
    """Write error file, append ErrorEntry, bump stage counter."""
    path = _write_error(stage, subject, exp, loc, acq, exc)
    result.errors.append(
        ErrorEntry(
            stage=stage,
            subject=subject,
            exp=exp,
            loc=loc,
            acq=acq,
            error_file=path,
            exc_type=type(exc).__name__,
            exc_message=str(exc),
        )
    )
    result.stages[stage].errors += 1


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _iter_acqs(master: dict) -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = []
    for subject in master:
        for exp in master[subject]:
            for acq_id in master[subject][exp]:
                loc, acq = acq_id.split("--")
                out.append((subject, exp, loc, acq))
    return out


def _iter_exps(master: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for subject in master:
        for exp in master[subject]:
            out.append((subject, exp))
    return out


def _stage_mirror(result: MetaUpdateResult, acq_tuples) -> None:
    summary = result.stages[STAGE_MIRROR]
    for subject, exp, loc, acq in acq_tuples:
        summary.attempted += 1
        _clear_stage_errors(subject, exp, loc, acq, STAGE_MIRROR)
        try:
            wis.meta.exsum_mirror.mirror_one_acq(subject, exp, loc, acq)
        except BaseException as exc:  # noqa: BLE001 - we want EVERYTHING caught
            _record_error(result, STAGE_MIRROR, subject, exp, loc, acq, exc)


def _stage_sync_info(result: MetaUpdateResult, exp_tuples) -> None:
    summary = result.stages[STAGE_SYNC_INFO]
    for subject, exp in exp_tuples:
        summary.attempted += 1
        _clear_stage_errors(subject, exp, None, None, STAGE_SYNC_INFO)
        try:
            wis.meta.sync.update_sync_info(subject, exp)
        except BaseException as exc:  # noqa: BLE001
            _record_error(result, STAGE_SYNC_INFO, subject, exp, None, None, exc)


def _stage_epoch_info(result: MetaUpdateResult, exp_tuples) -> None:
    summary = result.stages[STAGE_EPOCH_INFO]
    for subject, exp in exp_tuples:
        summary.attempted += 1
        _clear_stage_errors(subject, exp, None, None, STAGE_EPOCH_INFO)
        try:
            wis.meta.epoch.update_epoch_info(subject, exp)
        except BaseException as exc:  # noqa: BLE001
            _record_error(
                result, STAGE_EPOCH_INFO, subject, exp, None, None, exc
            )


def _stage_dmd_info(result: MetaUpdateResult, acq_tuples) -> None:
    summary = result.stages[STAGE_DMD_INFO]
    for subject, exp, loc, acq in acq_tuples:
        summary.attempted += 1
        _clear_stage_errors(subject, exp, loc, acq, STAGE_DMD_INFO)
        try:
            wis.meta.dmd_info._update_dmd_info(subject, exp, loc, acq)
        except BaseException as exc:  # noqa: BLE001
            _record_error(result, STAGE_DMD_INFO, subject, exp, loc, acq, exc)


def _stage_audit(result: MetaUpdateResult) -> None:
    summary = result.stages[STAGE_AUDIT]
    summary.attempted += 1
    try:
        df = wis.meta.status.refresh_exsum_audit()
        result.audit_csv_path = wis.meta.status.AUDIT_CSV_PATH
        # refresh_exsum_audit writes error-rows internally rather than
        # raising. Expose them as ErrorEntries (at exp scope) so the
        # summary is unified.
        error_rows = df.filter(df["overall_status"] == "error") if df.height else df
        for row in error_rows.iter_rows(named=True):
            summary.errors += 1
            # The audit has no persistent per-row error file; synthesize
            # one so the meta_issues trail is still complete.
            scope_dir = _scope_dir(row["subject"], row["exp"], None, None)
            scope_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            path = scope_dir / f"{STAGE_AUDIT}__{ts}.txt"
            detail = (
                row.get("scopex_details")
                or row.get("annotation_details")
                or row.get("events_matchfilt_details")
                or row.get("events_denoised_details")
                or "(no detail)"
            )
            path.write_text(
                f"Stage: {STAGE_AUDIT}\n"
                f"Subject: {row['subject']}\n"
                f"Exp: {row['exp']}\n"
                f"Scope: (experiment-level)\n"
                f"Time: {datetime.now().isoformat(timespec='seconds')}\n"
                f"\n"
                f"refresh_exsum_audit() flagged this experiment as errored.\n"
                f"Detail: {detail}\n",
                encoding="utf-8",
            )
            result.errors.append(
                ErrorEntry(
                    stage=STAGE_AUDIT,
                    subject=row["subject"],
                    exp=row["exp"],
                    loc=None,
                    acq=None,
                    error_file=path,
                    exc_type="AuditError",
                    exc_message=detail,
                )
            )
    except BaseException as exc:  # noqa: BLE001
        _record_error(result, STAGE_AUDIT, "(all)", "(all)", None, None, exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def update(
    refresh_audit: bool = True,
    verbose: bool = True,
) -> MetaUpdateResult:
    """Refresh every automatically-maintained metadata artifact.

    Iterates all acquisitions in ``acquisition_master.yaml`` and runs:

    1. ExSum mirror (per acq) — copies raw ``Summary*.mat`` into
       ``ExSum_mirrors/`` if out-of-date; updates ``prepro_info.yaml``
       as a side effect.
    2. ``sync_info.yaml`` (per subject+exp) — detects sync blocks and
       ephys offsets.
    3. ``epoch_info.yaml`` (per subject+exp) — per-epoch info for
       multi-epoch acqs (``n_epochs > 1`` in sync_info). Skipped for
       single-epoch acqs.
    4. ``dmd_info.yaml`` (per acq) — populates soma lists from each
       acq's mirrored ExSum; preserves existing ``depth`` values.
    5. ``ExSum_audits.csv`` (global) — only when
       ``refresh_audit=True`` (default).

    Per-scope exceptions are caught and written to
    ``analysis_materials/meta_issues/.../<stage>__<timestamp>.txt``
    rather than aborting the sweep. Prior error files in the same scope
    + stage are cleared at the start of each attempt, so a successful
    rerun leaves no stale trail.

    Parameters
    ----------
    refresh_audit : bool, optional
        If True (default), rebuild ``ExSum_audits.csv`` at the end. Set
        False for a faster meta-only pass.
    verbose : bool, optional
        Print a one-screen summary at the end.

    Returns
    -------
    MetaUpdateResult
        Per-stage counts, error list, paths to error directories that
        still contain files, and the path to the audit CSV if one was
        produced.
    """
    result = MetaUpdateResult()
    for stage in (
        STAGE_MIRROR,
        STAGE_SYNC_INFO,
        STAGE_EPOCH_INFO,
        STAGE_DMD_INFO,
        STAGE_AUDIT,
    ):
        result.stages[stage] = StageSummary(stage=stage)

    master = wis.meta.get.acq_master()
    acq_tuples = _iter_acqs(master)
    exp_tuples = _iter_exps(master)

    _stage_mirror(result, acq_tuples)
    _stage_sync_info(result, exp_tuples)
    _stage_epoch_info(result, exp_tuples)
    _stage_dmd_info(result, acq_tuples)
    if refresh_audit:
        _stage_audit(result)

    # Unique meta_issues directories that still contain error files.
    dirs_seen: set[Path] = set()
    for err in result.errors:
        dirs_seen.add(err.error_file.parent)
    result.error_dirs = sorted(dirs_seen)

    if verbose:
        _print_summary(result)

    return result


def _print_summary(result: MetaUpdateResult) -> None:
    print("wis.meta.update() done.")
    label_for = {
        STAGE_MIRROR: "ExSum mirror",
        STAGE_SYNC_INFO: "sync_info   ",
        STAGE_EPOCH_INFO: "epoch_info  ",
        STAGE_DMD_INFO: "dmd_info    ",
        STAGE_AUDIT: "audit CSV   ",
    }
    for stage_name in (
        STAGE_MIRROR,
        STAGE_SYNC_INFO,
        STAGE_EPOCH_INFO,
        STAGE_DMD_INFO,
        STAGE_AUDIT,
    ):
        s = result.stages[stage_name]
        if s.attempted == 0:
            continue
        ok = s.attempted - s.errors
        tag = label_for[stage_name]
        if s.errors:
            print(f"  {tag}: {ok}/{s.attempted} ok, {s.errors} errors")
        else:
            print(f"  {tag}: {ok}/{s.attempted} ok")
    if result.audit_csv_path is not None:
        print(f"  audit CSV path: {result.audit_csv_path}")
    if result.error_dirs:
        print(f"See meta_issues/ for {len(result.error_dirs)} scope(s):")
        # Show up to 10, then summarise
        show = result.error_dirs[:10]
        for d in show:
            try:
                rel = d.relative_to(META_ISSUES_ROOT)
                print(f"  - {rel}/")
            except ValueError:
                print(f"  - {d}/")
        if len(result.error_dirs) > 10:
            print(f"  ...and {len(result.error_dirs) - 10} more")
    else:
        # No errors: make sure we don't leave a lingering empty root.
        if META_ISSUES_ROOT.is_dir() and not any(META_ISSUES_ROOT.iterdir()):
            try:
                META_ISSUES_ROOT.rmdir()
            except OSError:
                pass
