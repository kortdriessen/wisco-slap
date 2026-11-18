"""Generate slow-wave plots for the somas represented in existing overview folders."""

from __future__ import annotations

import contextlib
import gc
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

import wisco_slap as wis
from wisco_slap.pns.overview_plot_gen import _generate_one
from wisco_slap.viz.overview_and_summary_plots.single_soma._single_soma_sw_locking import PLOT_SPEC


def existing_single_soma_members(root: Path | str | None = None) -> list[tuple[str, ...]]:
    """Union of PNG member IDs in existing plot folders, excluding this output.

    Use the requested on-disk coverage, without applying today's subgroup or
    baseline-validity filters to older figures. Multiple plot types deduplicate.
    """
    root = Path(root) if root is not None else (
        Path(wis.defs.annotation_root) / "OVERVIEW_FIGURES" / "single_soma"
    )
    pattern = re.compile(r"[^/]+--exp_\d+--loc_[^/]+--acq_\d+--soma\d+")
    return sorted({tuple(p.stem.split("--")) for p in root.glob("*/*.png")
                   if p.parent.name != PLOT_SPEC.name and pattern.fullmatch(p.stem)})


def generate_single_soma_sw_locking(
    *, overwrite: bool = False, members: list[tuple[str, ...]] | None = None,
) -> pl.DataFrame:
    """Build only the SW plot; checkpoint every member and keep per-soma logs.

    By default, include every soma in the existing single_soma plot folders.
    Errors are recorded and later members still run. Reruns retry errors;
    ``overwrite=True`` refreshes all results when data or settings change.
    """
    members = existing_single_soma_members() if members is None else sorted(set(members))
    if not members:
        raise ValueError("No single-soma overview PNGs found")
    outdir = PLOT_SPEC.path(*members[0]).parent
    outdir.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(wis.defs.annotation_root) / "_CLASSIFICATIONS" / "baseline_validity.json"
    baseline = {}
    if baseline_path.exists():
        baseline = {tuple(r[k] for k in ("subject", "exp", "loc", "acq")): r["value"]
                    for r in json.loads(baseline_path.read_text())["labels"]}
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    selection="existing single_soma overview PNG member IDs",
                    stage="NREM SWS", align="trough", expected_members=["--".join(m) for m in members],
                    complete=False, rows=[])

    def checkpoint() -> None:
        pending = outdir / "run_manifest.pending.json"
        pending.write_text(json.dumps(manifest, indent=2) + "\n")
        pending.replace(outdir / "run_manifest.json")
        if manifest["rows"]:
            pl.DataFrame(manifest["rows"]).write_csv(outdir / "generation_summary.csv")

    checkpoint()
    print(f"Generating NREM SWS locking for {len(members)} somas → {outdir}", flush=True)
    with plt.ioff():
        for i, ids in enumerate(members, 1):
            start = time.monotonic()
            path = PLOT_SPEC.path(*ids)
            with path.with_suffix(".log").open("a") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    row = _generate_one(PLOT_SPEC, ids, overwrite)
            row.update(n_detected_stage_waves=None, n_plotted_waves=None, n_synapses=None, reason="")
            row["baseline_validity"] = baseline.get(ids[:4], "unlabelled")
            if row["status"] != "error" and path.with_suffix(".json").exists():
                info = json.loads(path.with_suffix(".json").read_text())
                for key in ("n_detected_stage_waves", "n_plotted_waves", "n_synapses", "reason"):
                    row[key] = info[key]
                if info["status"] == "no_waves":
                    row["status"] = "no_waves"
            row["elapsed_s"] = round(time.monotonic() - start, 2)
            manifest["rows"].append(row)
            checkpoint()
            print(f"[{i}/{len(members)}] {'--'.join(ids)}: {row['status']} "
                  f"({row['n_plotted_waves']} waves; {row['elapsed_s']} s) {row['error']}", flush=True)
            gc.collect()
    manifest["complete"] = True
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    result = pl.DataFrame(manifest["rows"])
    print(result.group_by("status").len(), flush=True)
    return result
