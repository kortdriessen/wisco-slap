"""One-off: standardize event_detection artifacts to trial-clock convention.

Some `glu_events_basic.parquet` (and the corresponding mode's sibling
zarrs) were written with `apply_ephys_offset=True` already applied at
write time, leaving them on the TDT clock. The reader's
`apply_ephys_offset=True` default then double-shifts these. The
canonical convention going forward is **trial-clock on disk**; this
script brings any existing TDT-clock artifacts into line by subtracting
`ephys_offset` from their time-bearing values.

Per acq×mode the writer always produces a self-consistent set:
  matchfilt mode: glu_events_basic.parquet, filtered.zarr, noise_std.zarr
  denoised mode:  glu_events_basic_denoised.parquet, noise_std_denoised.zarr
so the classification (trial/TDT) is determined per-mode and all sibling
artifacts of that mode are patched together.

No backups (per plan): if classification is wrong for a given acq, the
recovery path is to re-run
``wis.pns.glu_ev_basic_mon.exp_data(subject, exp, redo=True)`` on that
experiment to regenerate everything from the ExSum.

Idempotent: re-running classifies already-patched parquets as
TRIAL_CLOCK and is a no-op.

Usage:
    python wisco-slap/scripts/patch_glu_event_clocks.py --dry-run
    python wisco-slap/scripts/patch_glu_event_clocks.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr
import zarr

import wisco_slap as wis

ANMAT = Path("/data/slap_analysis/analysis_materials")
TOL_S = 1.0  # seconds tolerance on bound checks

MATCHFILT_MODE = {
    "label": "matchfilt",
    "parquet": "glu_events_basic.parquet",
    "zarrs": ("filtered.zarr", "noise_std.zarr"),
}
DENOISED_MODE = {
    "label": "denoised",
    "parquet": "glu_events_basic_denoised.parquet",
    "zarrs": ("noise_std_denoised.zarr",),
}
MODES = (MATCHFILT_MODE, DENOISED_MODE)

AUDIT_FIELDS = [
    "subject", "exp", "locacq", "mode", "classification",
    "min_peak_time_before", "max_peak_time_before",
    "ephys_offset", "trial_max", "action_taken", "notes",
]


def discover_acqs():
    """Yield (subject, exp, loc, acq, ev_dir) for every existing event_detection/."""
    skip = {"annotation_materials", "ExSum_mirrors", "PUPIL_MODEL",
            "autoscore_model", "plots", "temp", "meta_issues"}
    for subj_dir in sorted(p for p in ANMAT.iterdir() if p.is_dir() and p.name not in skip):
        for exp_dir in sorted(
            p for p in subj_dir.iterdir()
            if p.is_dir() and p.name.startswith("exp_")
        ):
            scopex_dir = exp_dir / "scopex"
            if not scopex_dir.is_dir():
                continue
            for la_dir in sorted(
                p for p in scopex_dir.iterdir()
                if p.is_dir() and "--" in p.name
            ):
                ev_dir = la_dir / "event_detection"
                if not ev_dir.is_dir():
                    continue
                loc, acq = la_dir.name.split("--", 1)
                yield subj_dir.name, exp_dir.name, loc, acq, ev_dir


def get_trial_max(subject: str, exp: str, loc: str, acq: str) -> float | None:
    """Return the maximum trial-clock time (last sample's end) from the
    scopex zarr, or None if the zarr is missing.

    Works for both single- and multi-epoch acqs (scopex time coord is
    always on the trial clock by construction)."""
    zarr_path = ANMAT / subject / exp / "scopex" / f"{loc}--{acq}" / "syn_dF-ls.zarr"
    if not zarr_path.is_dir():
        return None
    for grp in ("dmd_1", "dmd_2"):
        if (zarr_path / grp).is_dir():
            ds = xr.open_zarr(str(zarr_path), group=grp)
            t = ds.time.values
            if len(t) < 2:
                return None
            dt = float(np.median(np.diff(t)))
            return float(t[-1]) + dt
    return None


def classify(
    min_pt: float, max_pt: float,
    ephys_offset: float | None, trial_max: float | None,
) -> tuple[str, str]:
    """Return (classification, note) for one parquet's peak_time range."""
    if ephys_offset is None:
        return "AMBIGUOUS", "ephys_offset is unresolved"
    if trial_max is None:
        return "AMBIGUOUS", "scopex zarr unavailable; cannot bound trial-clock max"
    if ephys_offset <= TOL_S:
        return "TRIAL_CLOCK", f"ephys_offset={ephys_offset:.3f}s ≤ tolerance; clocks equivalent"

    if min_pt < ephys_offset / 2:
        if (-TOL_S <= min_pt) and (max_pt <= trial_max + TOL_S):
            return "TRIAL_CLOCK", ""
        return "AMBIGUOUS", (
            f"min_pt={min_pt:.3f} suggests trial-clock but bounds violated "
            f"(max_pt={max_pt:.3f}, trial_max={trial_max:.3f})"
        )
    else:
        if (min_pt >= ephys_offset - TOL_S) and (max_pt <= ephys_offset + trial_max + TOL_S):
            return "TDT_CLOCK", ""
        return "AMBIGUOUS", (
            f"min_pt={min_pt:.3f} suggests TDT-clock but bounds violated "
            f"(offset={ephys_offset:.3f}, max_pt={max_pt:.3f}, trial_max={trial_max:.3f})"
        )


def patch_parquet(path: Path, offset: float) -> None:
    df = pl.read_parquet(path)
    df = df.with_columns(
        (pl.col("time") - offset).alias("time"),
        (pl.col("peak_time") - offset).alias("peak_time"),
    )
    df.write_parquet(path)


def patch_zarr_time(path: Path, offset: float) -> None:
    """Subtract `offset` from every dmd_*/time array in the zarr group."""
    zg = zarr.open_group(str(path), mode="r+")
    for grp_name in zg.group_keys():
        time_arr = zg[grp_name]["time"]
        time_arr[:] = time_arr[:] - offset


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Standardize event_detection artifacts to trial-clock on disk."
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print classifications and planned actions but do not modify files.",
    )
    args = ap.parse_args(argv)

    rows: list[dict] = []
    for subject, exp, loc, acq, ev_dir in discover_acqs():
        # Acq-wide lookups (shared across modes)
        try:
            offset: float | None = wis.meta.get.ephys_offset(subject, exp, loc, acq)
        except (ValueError, KeyError):
            offset = None
        trial_max = get_trial_max(subject, exp, loc, acq)

        for mode in MODES:
            pq_path = ev_dir / mode["parquet"]
            if not pq_path.is_file():
                continue

            df = pl.read_parquet(pq_path)
            if df.is_empty():
                rows.append({
                    "subject": subject, "exp": exp, "locacq": f"{loc}--{acq}",
                    "mode": mode["label"], "classification": "EMPTY",
                    "min_peak_time_before": None, "max_peak_time_before": None,
                    "ephys_offset": offset, "trial_max": trial_max,
                    "action_taken": "skipped", "notes": "events parquet has 0 rows",
                })
                continue

            min_pt = float(df["peak_time"].min())
            max_pt = float(df["peak_time"].max())
            cls, note = classify(min_pt, max_pt, offset, trial_max)

            action = "skipped"
            if cls == "TDT_CLOCK":
                if args.dry_run:
                    action = "would_patch"
                else:
                    patch_parquet(pq_path, offset)  # type: ignore[arg-type]
                    for zname in mode["zarrs"]:
                        zpath = ev_dir / zname
                        if zpath.is_dir():
                            patch_zarr_time(zpath, offset)  # type: ignore[arg-type]
                    action = "patched"

            rows.append({
                "subject": subject, "exp": exp, "locacq": f"{loc}--{acq}",
                "mode": mode["label"], "classification": cls,
                "min_peak_time_before": min_pt, "max_peak_time_before": max_pt,
                "ephys_offset": offset, "trial_max": trial_max,
                "action_taken": action, "notes": note,
            })

    audit_path = ANMAT / "event_clock_patch_audit.csv"
    with audit_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    n_total = len(rows)
    n_trial = sum(1 for r in rows if r["classification"] == "TRIAL_CLOCK")
    n_tdt = sum(1 for r in rows if r["classification"] == "TDT_CLOCK")
    n_ambig = sum(1 for r in rows if r["classification"] == "AMBIGUOUS")
    n_empty = sum(1 for r in rows if r["classification"] == "EMPTY")
    n_patched = sum(1 for r in rows if r["action_taken"] == "patched")
    n_would = sum(1 for r in rows if r["action_taken"] == "would_patch")

    print()
    print("=" * 78)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print(f"Total parquets scanned: {n_total}")
    print(f"  TRIAL_CLOCK (no-op):              {n_trial}")
    print(f"  TDT_CLOCK ({'would patch' if args.dry_run else 'patched'}):       {n_would if args.dry_run else n_patched}  (of {n_tdt} classified)")
    print(f"  AMBIGUOUS (skipped):              {n_ambig}")
    print(f"  EMPTY (skipped):                  {n_empty}")
    print(f"Audit CSV written to: {audit_path}")
    print()

    ambig = [r for r in rows if r["classification"] == "AMBIGUOUS"]
    if ambig:
        print("AMBIGUOUS rows requiring inspection:")
        for r in ambig:
            print(
                f"  {r['subject']}/{r['exp']}/{r['locacq']} ({r['mode']}): "
                f"min_pt={r['min_peak_time_before']}, max_pt={r['max_peak_time_before']}, "
                f"offset={r['ephys_offset']}, trial_max={r['trial_max']}"
            )
            print(f"    notes: {r['notes']}")
        print()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
