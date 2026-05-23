"""Clean up /data/slap_analysis/analysis_materials/ by moving deprecated and
out-of-master content into /data/slap_analysis/ZWISCO_SLAP_GRAVEYARD/bad_analysis_materials/.

See /home/driessen2/.claude/plans/claude-claude-md-ok-i-need-twinkling-bee.md for
the full plan and rationale.

Dry-run by default; --execute required to perform moves.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import polars as pl
import yaml

ANMAT = Path("/data/slap_analysis/analysis_materials")
ANNMAT = ANMAT / "annotation_materials"
GRAVE = Path("/data/slap_analysis/ZWISCO_SLAP_GRAVEYARD/bad_analysis_materials")
GRAVE_AM = GRAVE / "old_analysis_materials"
GRAVE_ANN = GRAVE / "old_annotation_materials"

DEPRECATED_EXP_SUBDIRS = ("activity_data", "mean_IMs", "scoring_data", "synapse_ids")
UNTOUCHABLE_EXP_SUBDIRS = ("sync_block_data",)


def load_master() -> dict[str, dict[str, list[str]]]:
    with open(ANMAT / "acquisition_master.yaml") as f:
        return yaml.safe_load(f)


def _on_disk_subjects(root: Path) -> list[str]:
    if not root.exists():
        return []
    skip = {"annotation_materials", "ExSum_mirrors", "PUPIL_MODEL", "autoscore_model", "plots", "temp"}
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in skip)


def _on_disk_exps(subject_dir: Path) -> list[str]:
    return sorted(p.name for p in subject_dir.iterdir() if p.is_dir() and p.name.startswith("exp_"))


def planned_moves_phase_a(master: dict) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for subject in sorted(master.keys()):
        for exp in sorted(master[subject].keys()):
            exp_dir = ANMAT / subject / exp
            if not exp_dir.exists():
                continue
            for dep in DEPRECATED_EXP_SUBDIRS:
                src = exp_dir / dep
                if src.exists():
                    dst = GRAVE_AM / subject / exp / dep
                    moves.append((src, dst))
    return moves


def planned_moves_phase_b(master: dict) -> list[tuple[Path, Path]]:
    """Subject/exp combos present on disk but absent from the master.

    Moves every child of the orphan exp_ EXCEPT `sync_block_data/` (user decision:
    sync_block_data is indexed by sync_block, not acq, and is out of scope).
    Mirrored for annotation_materials (no sync_block_data there, so the filter is a no-op).
    """
    moves: list[tuple[Path, Path]] = []

    # analysis_materials side
    for subject in _on_disk_subjects(ANMAT):
        subj_dir = ANMAT / subject
        for exp in _on_disk_exps(subj_dir):
            if subject in master and exp in master[subject]:
                continue
            exp_dir = subj_dir / exp
            for child in sorted(exp_dir.iterdir()):
                if child.name in UNTOUCHABLE_EXP_SUBDIRS:
                    continue
                dst = GRAVE_AM / subject / exp / child.name
                moves.append((child, dst))

    # annotation_materials side: move whole orphan exp_ subtree
    if ANNMAT.exists():
        for subject in sorted(p.name for p in ANNMAT.iterdir() if p.is_dir()):
            subj_dir = ANNMAT / subject
            for exp in _on_disk_exps(subj_dir):
                if subject in master and exp in master[subject]:
                    continue
                src = subj_dir / exp
                dst = GRAVE_ANN / subject / exp
                moves.append((src, dst))

    return moves


def planned_moves_phase_c_scopex(master: dict) -> list[tuple[Path, Path]]:
    """scopex subdirs in combined `loc_X--acq_N` format."""
    moves: list[tuple[Path, Path]] = []
    for subject in sorted(master.keys()):
        for exp in sorted(master[subject].keys()):
            scopex_dir = ANMAT / subject / exp / "scopex"
            if not scopex_dir.exists():
                continue
            allowed = set(master[subject][exp])
            for child in sorted(scopex_dir.iterdir()):
                if child.is_dir() and child.name not in allowed:
                    dst = GRAVE_AM / subject / exp / "scopex" / child.name
                    moves.append((child, dst))
    return moves


def planned_moves_phase_c_annotation(master: dict) -> list[tuple[Path, Path]]:
    """annotation_materials per-acq dirs in split `<loc_X>/<acq_N>/` format."""
    moves: list[tuple[Path, Path]] = []
    if not ANNMAT.exists():
        return moves
    for subject in sorted(master.keys()):
        subj_dir = ANNMAT / subject
        if not subj_dir.exists():
            continue
        for exp in sorted(master[subject].keys()):
            exp_dir = subj_dir / exp
            if not exp_dir.exists():
                continue
            allowed = set(master[subject][exp])
            for loc_dir in sorted(exp_dir.iterdir()):
                if not loc_dir.is_dir() or not loc_dir.name.startswith("loc_"):
                    continue
                for acq_dir in sorted(loc_dir.iterdir()):
                    if not acq_dir.is_dir() or not acq_dir.name.startswith("acq_"):
                        continue
                    combined = f"{loc_dir.name}--{acq_dir.name}"
                    if combined not in allowed:
                        dst = GRAVE_ANN / subject / exp / loc_dir.name / acq_dir.name
                        moves.append((acq_dir, dst))
    return moves


def report_phase_d() -> pl.DataFrame:
    rows = []
    for subject in _on_disk_subjects(ANMAT):
        subj_dir = ANMAT / subject
        for child in subj_dir.iterdir():
            if child.is_file() and child.name in ("Thumbs.db", "mean_im_ppt.pptx"):
                rows.append({"subject": subject, "file": child.name, "size_mb": child.stat().st_size / 1e6})
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"subject": pl.Utf8, "file": pl.Utf8, "size_mb": pl.Float64})


def _dir_size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1e6
    total = 0
    for sub in p.rglob("*"):
        if sub.is_file():
            try:
                total += sub.stat().st_size
            except OSError:
                pass
    return total / 1e6


def _preflight(moves: list[tuple[Path, Path]]) -> list[str]:
    """Return a list of human-readable problem strings; empty list = OK."""
    problems: list[str] = []
    grave_resolved = GRAVE.resolve()
    seen_dsts: set[Path] = set()
    for src, dst in moves:
        if not src.exists():
            problems.append(f"SRC_MISSING: {src}")
        if dst.exists():
            problems.append(f"DST_COLLISION: {dst}")
        try:
            if grave_resolved not in dst.resolve().parents and dst.resolve() != grave_resolved:
                problems.append(f"DST_NOT_IN_GRAVE: {dst}")
        except OSError:
            # dst doesn't exist yet; resolve its parent instead
            if grave_resolved not in dst.parent.resolve().parents and dst.parent.resolve() != grave_resolved:
                problems.append(f"DST_NOT_IN_GRAVE: {dst}")
        if dst in seen_dsts:
            problems.append(f"DST_DUPLICATE_IN_PLAN: {dst}")
        seen_dsts.add(dst)
    return problems


def _moves_df(moves: list[tuple[Path, Path]], phase: str) -> pl.DataFrame:
    rows = [
        {
            "phase": phase,
            "src": str(src),
            "dst": str(dst),
            "src_exists": src.exists(),
            "dst_exists": dst.exists(),
            "src_size_mb": round(_dir_size_mb(src), 2) if src.exists() else 0.0,
        }
        for src, dst in moves
    ]
    schema = {"phase": pl.Utf8, "src": pl.Utf8, "dst": pl.Utf8, "src_exists": pl.Boolean, "dst_exists": pl.Boolean, "src_size_mb": pl.Float64}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def execute_moves(moves: list[tuple[Path, Path]], *, dry_run: bool, phase: str) -> None:
    if not moves:
        print(f"[{phase}] No moves planned. Nothing to do.")
        return

    with pl.Config(tbl_rows=len(moves) + 5, tbl_cols=6, fmt_str_lengths=200, tbl_width_chars=500):
        print(f"\n[{phase}] Planned moves ({len(moves)} total):")
        print(_moves_df(moves, phase))

    problems = _preflight(moves)
    if problems:
        print(f"\n[{phase}] PREFLIGHT PROBLEMS ({len(problems)}):")
        for p in problems:
            print(f"  {p}")
        print(f"\n[{phase}] Aborting due to preflight problems.")
        sys.exit(2)

    total_mb = sum(_dir_size_mb(src) for src, _ in moves)
    print(f"\n[{phase}] Preflight: OK. Total size to move: {total_mb:.1f} MB across {len(moves)} targets.")

    if dry_run:
        print(f"[{phase}] DRY-RUN — no changes made. Rerun with --execute to perform moves.")
        return

    print(f"[{phase}] EXECUTING...")
    for i, (src, dst) in enumerate(moves, 1):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        assert dst.exists(), f"post-move assertion failed: {dst} missing"
        assert not src.exists(), f"post-move assertion failed: {src} still present"
        print(f"  [{i}/{len(moves)}] moved: {src} -> {dst}")
    print(f"[{phase}] DONE.")


def verify(master: dict) -> None:
    print("\n=== VERIFY ===")

    # 1. No deprecated subtrees in in-master exps
    stray_deprecated: list[str] = []
    for subject in sorted(master.keys()):
        for exp in sorted(master[subject].keys()):
            exp_dir = ANMAT / subject / exp
            if not exp_dir.exists():
                continue
            for dep in DEPRECATED_EXP_SUBDIRS:
                if (exp_dir / dep).exists():
                    stray_deprecated.append(str(exp_dir / dep))
    print(f"[1] stray deprecated subtrees (activity_data/mean_IMs/scoring_data/synapse_ids): {len(stray_deprecated)}")
    for s in stray_deprecated:
        print(f"    {s}")

    # 2. Out-of-master exps still on disk (non-sync_block_data children)
    orphan_children: list[str] = []
    for subject in _on_disk_subjects(ANMAT):
        for exp in _on_disk_exps(ANMAT / subject):
            if subject in master and exp in master[subject]:
                continue
            exp_dir = ANMAT / subject / exp
            for child in exp_dir.iterdir():
                if child.name in UNTOUCHABLE_EXP_SUBDIRS:
                    continue
                orphan_children.append(str(child))
    print(f"[2] children of out-of-master exps (excl. sync_block_data): {len(orphan_children)}")
    for s in orphan_children:
        print(f"    {s}")

    # 3. scopex loc--acq dirs not in master
    scopex_stray: list[str] = []
    for subject in sorted(master.keys()):
        for exp in sorted(master[subject].keys()):
            scopex_dir = ANMAT / subject / exp / "scopex"
            if not scopex_dir.exists():
                continue
            allowed = set(master[subject][exp])
            for child in scopex_dir.iterdir():
                if child.is_dir() and child.name not in allowed:
                    scopex_stray.append(str(child))
    print(f"[3] stray scopex loc--acq dirs: {len(scopex_stray)}")
    for s in scopex_stray:
        print(f"    {s}")

    # 4. annotation_materials per-acq dirs not in master
    ann_stray: list[str] = []
    if ANNMAT.exists():
        for subject in sorted(master.keys()):
            subj_dir = ANNMAT / subject
            if not subj_dir.exists():
                continue
            for exp in sorted(master[subject].keys()):
                exp_dir = subj_dir / exp
                if not exp_dir.exists():
                    continue
                allowed = set(master[subject][exp])
                for loc_dir in exp_dir.iterdir():
                    if not loc_dir.is_dir() or not loc_dir.name.startswith("loc_"):
                        continue
                    for acq_dir in loc_dir.iterdir():
                        if not acq_dir.is_dir() or not acq_dir.name.startswith("acq_"):
                            continue
                        if f"{loc_dir.name}--{acq_dir.name}" not in allowed:
                            ann_stray.append(str(acq_dir))
    print(f"[4] stray annotation_materials acq dirs: {len(ann_stray)}")
    for s in ann_stray:
        print(f"    {s}")

    # 5. out-of-master exps still present in annotation_materials
    ann_orphan_exps: list[str] = []
    if ANNMAT.exists():
        for subject in sorted(p.name for p in ANNMAT.iterdir() if p.is_dir()):
            subj_dir = ANNMAT / subject
            for exp in _on_disk_exps(subj_dir):
                if subject not in master or exp not in master[subject]:
                    ann_orphan_exps.append(str(subj_dir / exp))
    print(f"[5] orphan exps in annotation_materials: {len(ann_orphan_exps)}")
    for s in ann_orphan_exps:
        print(f"    {s}")

    all_clean = not (stray_deprecated or orphan_children or scopex_stray or ann_stray or ann_orphan_exps)
    print("\n" + ("ALL CLEAN ✓" if all_clean else "VIOLATIONS PRESENT ✗"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["A", "B", "C", "C1", "C2", "D", "verify", "all"])
    ap.add_argument("--execute", action="store_true", help="Actually move files. Default is dry-run.")
    args = ap.parse_args()

    master = load_master()
    dry = not args.execute

    if args.phase == "A" or args.phase == "all":
        execute_moves(planned_moves_phase_a(master), dry_run=dry, phase="A")
    if args.phase == "B" or args.phase == "all":
        execute_moves(planned_moves_phase_b(master), dry_run=dry, phase="B")
    if args.phase == "C" or args.phase == "C1" or args.phase == "all":
        execute_moves(planned_moves_phase_c_scopex(master), dry_run=dry, phase="C1-scopex")
    if args.phase == "C" or args.phase == "C2" or args.phase == "all":
        execute_moves(planned_moves_phase_c_annotation(master), dry_run=dry, phase="C2-annotation")
    if args.phase == "D" or args.phase == "all":
        df = report_phase_d()
        print("\n[D] stray per-subject files (report-only):")
        with pl.Config(tbl_rows=len(df) + 5, fmt_str_lengths=200):
            print(df)
    if args.phase == "verify" or args.phase == "all":
        verify(master)

    return 0


if __name__ == "__main__":
    sys.exit(main())
