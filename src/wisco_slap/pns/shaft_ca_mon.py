"""Freshness monitor for shaft-calcium disc traces.

Shaft-calcium outputs (see :mod:`wisco_slap.pns.shaft_ca_gen`) live in
``scopex/<loc>--<acq>/shaft_ca/`` and have TWO upstreams: the
ExperimentSummary mirror (alignment, channel schema, discard masks) and the
WISynaptic ``prox_lines`` CSVs, which can be redrawn at any time. Freshness
therefore requires, for every DMD that has a usable prox_lines CSV (every
line labeled with a unique dend-ID, and the CSV saved on or after the
channel-fix trust cutoff
:data:`wisco_slap.pns.shaft_ca_gen.PROX_LINES_MIN_SAVE_DATE`; see
:func:`wisco_slap.pns.shaft_ca_gen.prox_lines_problem`):

- ``shaft_ca_dmd{d}.zarr`` exists;
- the ``shaft_ca__<ExSum-version>.txt`` marker matches the current mirror;
- the ``shaft_ca__prox_lines_dmd{d}.txt`` marker matches the current CSV's
  content hash;
- the dataset's ``time_axis_version`` includes the dropped-trial clock fix.

This is deliberately NOT part of scopex_mon's all-or-nothing group: the
scopex outputs share one expensive ExSum trial-data load, while shaft-ca
reads the raw registered TIFFs instead, so the two are independently
checkable and repairable.

Public API mirrors the other monitors: :func:`check_status` (read-only),
:func:`acq_data`, :func:`exp_data`, :func:`all_subjects`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import wisco_slap as wis
from wisco_slap.pns import shaft_ca_gen
from wisco_slap.pns._frame_timing import SHAFT_TIME_AXIS_VERSION


@dataclass(frozen=True)
class ShaftCaCheckResult:
    """Read-only freshness status of shaft-calcium outputs for one acq.

    ``status`` is one of:

    - ``"fresh"`` — every DMD with prox lines has a current output;
    - ``"stale"`` — outputs exist but the ExSum version, a prox_lines
      fingerprint, or the timestamp schema is outdated;
    - ``"missing"`` — at least one DMD with prox lines has no output;
    - ``"no_prox_lines"`` — no DMD has a usable prox_lines CSV (absent,
      present but without unique dend-IDs on every line, or saved before
      the channel-fix trust cutoff; nothing to do);
    - ``"missing_mirror"`` — no ExperimentSummary mirror.
    """

    status: str
    esum_version: str | None = None
    dmds_with_prox: tuple[int, ...] = ()
    missing_dmds: tuple[int, ...] = ()
    stale_dmds: tuple[int, ...] = ()
    details: tuple[str, ...] = field(default=())


def _esum_mirror(subject: str, exp: str, loc: str, acq: str) -> str | None:
    try:
        path = wis.meta.get.esum_mirror_path(subject, exp, loc, acq)
    except (FileNotFoundError, ValueError):
        return None
    if path in (0, "NO_ESUM_MIRROR"):
        return None
    return str(path)


def check_status(subject: str, exp: str, loc: str, acq: str) -> ShaftCaCheckResult:
    """Return the read-only freshness status; regenerates nothing."""

    esum_path = _esum_mirror(subject, exp, loc, acq)
    if esum_path is None:
        return ShaftCaCheckResult(status="missing_mirror")
    version = os.path.basename(esum_path).split(".mat")[0]
    out_dir = shaft_ca_gen.shaft_ca_dir(subject, exp, loc, acq)

    usable: list[int] = []
    details: list[str] = []
    for dmd in (1, 2):
        csv_path = shaft_ca_gen.prox_lines_csv(subject, exp, loc, acq, dmd)
        if not csv_path.exists():
            continue
        problem = shaft_ca_gen.prox_lines_problem(csv_path)
        if problem is not None:
            details.append(f"dmd{dmd}: prox_lines unusable ({problem})")
            continue
        usable.append(dmd)
    dmds_with_prox = tuple(usable)
    if not dmds_with_prox:
        return ShaftCaCheckResult(
            status="no_prox_lines", esum_version=version, details=tuple(details)
        )

    missing: list[int] = []
    stale: list[int] = []
    exsum_marker_ok = (out_dir / f"shaft_ca__{version}.txt").is_file()
    for dmd in dmds_with_prox:
        zarr_path = out_dir / f"shaft_ca_dmd{dmd}.zarr"
        if not zarr_path.is_dir():
            missing.append(dmd)
            details.append(f"dmd{dmd}: output missing")
            continue
        if not exsum_marker_ok:
            stale.append(dmd)
            details.append(f"dmd{dmd}: ExSum version marker != {version}")
            continue
        marker = out_dir / f"shaft_ca__prox_lines_dmd{dmd}.txt"
        current = shaft_ca_gen.prox_lines_fingerprint(
            shaft_ca_gen.prox_lines_csv(subject, exp, loc, acq, dmd)
        )
        if not marker.is_file() or marker.read_text().strip() != current:
            stale.append(dmd)
            details.append(f"dmd{dmd}: prox_lines changed since generation")
            continue
        try:
            attrs = json.loads((zarr_path / ".zattrs").read_text())
        except (OSError, ValueError):
            attrs = {}
        if attrs.get("time_axis_version") != SHAFT_TIME_AXIS_VERSION:
            stale.append(dmd)
            details.append(
                f"dmd{dmd}: outdated shaft clock; run "
                "wis.pns.shaft_ca_gen.repair_time_coordinates(...) or regenerate"
            )

    if stale:
        status = "stale"
    elif missing:
        status = "missing"
    else:
        status = "fresh"
    return ShaftCaCheckResult(
        status=status,
        esum_version=version,
        dmds_with_prox=dmds_with_prox,
        missing_dmds=tuple(missing),
        stale_dmds=tuple(stale),
        details=tuple(details),
    )


def acq_data(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    **gen_kwargs: Any,
) -> None:
    """Check and (re)generate shaft-calcium outputs for one acquisition.

    Regenerates when any DMD with prox lines is missing or stale (or always,
    with ``overwrite=True``). ``dry_run=True`` only prints the status.
    Extra keyword arguments are forwarded to
    :func:`wisco_slap.pns.shaft_ca_gen.acq_data`.
    """

    tag = f"{subject} {exp} {loc} {acq}"
    result = check_status(subject, exp, loc, acq)
    if result.status == "missing_mirror":
        print(f"[{tag}] Skipping shaft_ca — no ExperimentSummary mirror")
        return
    if result.status == "no_prox_lines":
        why = "; ".join(result.details) or "no prox_lines drawn yet"
        print(f"[{tag}] Skipping shaft_ca — {why}")
        return
    if result.status == "fresh" and not overwrite:
        return
    if dry_run:
        print(
            f"[{tag}] shaft_ca would regenerate ({result.status}): "
            + "; ".join(result.details)
        )
        return
    shaft_ca_gen.acq_data(subject, exp, loc, acq, **gen_kwargs)


def exp_data(
    subject: str,
    exp: str,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    **gen_kwargs: Any,
) -> None:
    """Run :func:`acq_data` for every acquisition of one experiment."""

    for loc, acq in wis.meta.get.locacq(subject, exp):
        acq_data(
            subject,
            exp,
            loc,
            acq,
            overwrite=overwrite,
            dry_run=dry_run,
            **gen_kwargs,
        )


def all_subjects(
    *, overwrite: bool = False, dry_run: bool = False, **gen_kwargs: Any
) -> None:
    """Run :func:`exp_data` for every subject/experiment in the master list."""

    master = wis.meta.get.acq_master()
    for subject, experiments in master.items():
        for exp in experiments:
            exp_data(
                subject,
                exp,
                overwrite=overwrite,
                dry_run=dry_run,
                **gen_kwargs,
            )
