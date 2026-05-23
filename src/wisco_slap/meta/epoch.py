"""Per-epoch metadata for multi-epoch acquisitions.

A multi-epoch acquisition is one where scanning was paused and restarted
during imaging, producing multiple ``*_DMD1-CYCLE-000000.dat`` files in
the same acq dir. The MATLAB preprocessing pipeline re-assembles these
into a single concatenated trial stream, but the individual epochs have
gaps between them on the wall clock and may even span different TDT
(sync-block) recordings. Aligning concatenated trial data against ephys
therefore requires knowing both:

1. where each epoch sits **relative to epoch 1** (microscope-only, for a
   unified scopex time axis), and
2. which TDT sync block each epoch falls in, and **that epoch's
   ephys_offset** within that block (for per-epoch ephys alignment).

This module maintains ``analysis_materials/epoch_info.yaml``:

.. code-block:: yaml

    <subject>:
      <exp>:
        <loc--acq>:
          1:
            filename: acq-1_YYYYMMDD_HHMMSS_DMD1-CYCLE-000000.dat
            duration_since_epoch_1_s: 0.0       # FPGA-derived
            fpga_timestamp_s: 3964.629799       # raw FPGA seconds
            sync_block: 1                       # which TDT block covers this epoch
            ephys_offset_s: 80.9214             # offset from that block's ephys_start
          2:
            ...

**Time axis precision:** ``duration_since_epoch_1_s`` comes from each
``.dat`` file's first line-header uint64 timestamp (FPGA clock, typically
200 MHz → 5 ns resolution). The microscope's FPGA clock is continuous
across scan stop/restarts, so these durations are exact regardless of how
long the user paused between epochs, and regardless of TDT sync-block
boundaries.

**Per-epoch ephys offsets:** computed by running the same scope-UP
matcher used for single-epoch acqs (``wis.meta.sync.match_scope_up_window``),
but parameterized by each epoch's filename wall-clock and the epoch's
assigned sync block's ``ephys_start``. Each epoch's assigned sync block
is determined by wall-clock membership in the block's ephys coverage
window.

**Relationship to sync_info.yaml:** ``sync_info.yaml``'s acq-level
``sync_block`` and ``ephys_offset`` always refer to **epoch 1**. This
preserves the single-scalar mental model for consumers who don't care
about epochs, and matches the single-epoch case. For per-epoch values on
a multi-epoch acq, read ``epoch_info.yaml``.

**Not handled yet:** epochs falling in corrupt sync blocks. Raises with
a clear message in that case.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

import wisco_slap as wis
import wisco_slap.defs as DEFS
from slap2_py.utils.datfile import read_dat_header
from wisco_slap.meta import sync as _sync

EPOCH_INFO_PATH = Path(DEFS.anmat_root) / "epoch_info.yaml"

# Fields every per-epoch record must have; used by the idempotency check to
# trigger a re-compute when the schema has grown.
_REQUIRED_EPOCH_FIELDS: frozenset[str] = frozenset(
    (
        "filename",
        "duration_since_epoch_1_s",
        "fpga_timestamp_s",
        "sync_block",
        "ephys_offset_s",
    )
)


def _load_epoch_info() -> dict:
    """Load ``epoch_info.yaml`` from disk; returns an empty dict if missing."""
    if not EPOCH_INFO_PATH.exists():
        return {}
    with open(EPOCH_INFO_PATH) as f:
        d = yaml.safe_load(f)
    return d or {}


def _save_epoch_info(d: dict) -> None:
    """Write ``d`` to ``epoch_info.yaml`` (overwrites wholesale)."""
    with open(EPOCH_INFO_PATH, "w") as f:
        yaml.safe_dump(d, f, sort_keys=True)


def _find_sync_block_for_timestamp(
    wall_ts: pd.Timestamp, si_exp: dict
) -> int:
    """Return the sync_block whose ephys coverage contains ``wall_ts``.

    Raises ``ValueError`` if the timestamp falls in no block or in a
    corrupt one (the latter is a deferred case: would require extending
    ``find_corrupted_sync_offsets`` to per-epoch).
    """
    for sb_key, sbd in si_exp["sync_blocks"].items():
        start = pd.Timestamp(sbd["ephys_start"])
        end = pd.Timestamp(sbd["ephys_end"])
        if start <= wall_ts <= end:
            if sbd.get("corrupt") is not False:
                raise ValueError(
                    f"Epoch at {wall_ts} falls in sync_block {sb_key}, "
                    f"which is marked corrupt ({sbd.get('corrupt')!r}). "
                    f"Per-epoch corrupt-block offset interpolation is not "
                    f"yet implemented."
                )
            return int(sb_key)
    raise ValueError(
        f"Epoch at {wall_ts} does not fall within any sync block's ephys "
        f"coverage window. Check that ephys recording was running during "
        f"this epoch."
    )


def _compute_epochs_for_acq(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    epochs: list[tuple[str, pd.Timestamp]],
    si_exp: dict,
) -> dict:
    """Build the per-epoch info dict for one multi-epoch acq.

    Algorithm:

    1. For each epoch, read the .dat first-line FPGA timestamp (unified
       scopex time axis) and assign the epoch to a sync block by
       wall-clock membership in that block's ephys coverage.
    2. Within each sync block, pick the **earliest** epoch as the anchor;
       run :func:`wisco_slap.meta.sync.match_scope_up_window` once on
       that block's SYNC file to get its ephys_offset_s.
    3. All other epochs in the same block inherit an offset derived by
       adding their FPGA-based time delta from the anchor:
       ``ephys_offset_s[k] = ephys_offset_s[anchor] +
       (fpga_ts[k] - fpga_ts[anchor])``.

    Why extrapolate rather than match each epoch's window independently?
    Some SYNC files merge consecutive epochs into a single continuous
    scope-UP window (the microscope's trigger line doesn't toggle between
    quick stops/restarts). Matching per-epoch in that case silently maps
    every epoch in the block onto the same scope-UP start, producing the
    wrong offset for all epochs after the first. Anchor-plus-FPGA is
    robust to that, and is also more precise (5 ns FPGA vs 200 μs SYNC).

    Caches loaded SYNC data per block so a cross-block acq loads each
    SYNC file at most once.
    """
    acq_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}"

    # Pass 1: collect per-epoch FPGA timestamps and sync-block assignments.
    per_epoch: list[dict] = []
    for fname, wall_ts in epochs:
        hdr = read_dat_header(os.path.join(acq_dir, fname))
        block = _find_sync_block_for_timestamp(wall_ts, si_exp)
        per_epoch.append({
            "filename": fname,
            "wall_ts": wall_ts,
            "fpga_s": hdr.first_line_timestamp_s,
            "sync_block": int(block),
        })

    # Anchor epoch (earliest across all epochs) for the unified scopex
    # time axis; epochs were already sorted by wall_ts in the caller.
    acq_anchor_fpga_s = per_epoch[0]["fpga_s"]

    # Pass 2: for each sync block, identify its earliest epoch (block
    # anchor) and match once. Other epochs in the block get their offsets
    # by FPGA delta from the block anchor.
    block_anchor_offset_s: dict[int, float] = {}
    block_anchor_fpga_s: dict[int, float] = {}
    seen_blocks: set[int] = set()
    for e in per_epoch:
        b = e["sync_block"]
        if b in seen_blocks:
            continue
        seen_blocks.add(b)
        scope, ephys = _sync.load_sync_block(subject, exp, b)
        if isinstance(scope, str) or isinstance(ephys, str):
            raise ValueError(
                f"SYNC file for {subject}/{exp} sync_block {b} is corrupt; "
                f"cannot anchor per-epoch offsets for '{e['filename']}'."
            )
        ephys_start_ts = pd.Timestamp(si_exp["sync_blocks"][b]["ephys_start"])
        offset_s = _sync.match_scope_up_window(
            scope, ephys, e["wall_ts"], ephys_start_ts,
        )
        block_anchor_offset_s[b] = float(offset_s)
        block_anchor_fpga_s[b] = e["fpga_s"]

    # Pass 3: assemble the final dict.
    out: dict = {}
    for i, e in enumerate(per_epoch, start=1):
        b = e["sync_block"]
        fpga_delta_from_block_anchor = e["fpga_s"] - block_anchor_fpga_s[b]
        ephys_offset_s = block_anchor_offset_s[b] + fpga_delta_from_block_anchor
        out[i] = {
            "filename": e["filename"],
            "duration_since_epoch_1_s": float(e["fpga_s"] - acq_anchor_fpga_s),
            "fpga_timestamp_s": float(e["fpga_s"]),
            "sync_block": b,
            "ephys_offset_s": float(ephys_offset_s),
        }
    return out


def _entry_is_complete(acq_entry: dict, n_epochs_expected: int) -> bool:
    """Return True iff the stored acq entry has all expected epochs AND
    every per-epoch record carries every required field.

    Called by the idempotency check — if the schema grows (as it did here
    with the addition of ``sync_block`` and ``ephys_offset_s``), existing
    entries lacking the new fields auto-refresh on the next
    ``wis.meta.update()``.
    """
    if len(acq_entry) != n_epochs_expected:
        return False
    for rec in acq_entry.values():
        if not _REQUIRED_EPOCH_FIELDS.issubset(rec.keys()):
            return False
    return True


def update_epoch_info(subject: str, exp: str) -> bool:
    """Populate ``epoch_info.yaml`` entries for multi-epoch acqs in this exp.

    For every acq in ``sync_info.yaml`` with ``n_epochs > 1`` whose
    ``epoch_info`` entry is missing, has a stale epoch count, or lacks any
    per-epoch field required by the current schema, this parses each
    epoch's ``.dat`` header and writes per-epoch durations, sync-block
    assignments, and ephys offsets. Idempotent: a second call with no
    raw-data or schema changes is a no-op.

    Single-epoch acqs are skipped (never written to epoch_info).

    Returns
    -------
    bool
        True if any changes were written, else False.
    """
    si = wis.meta.get.sync_info()
    si_exp = si.get(subject, {}).get(exp)
    if si_exp is None:
        return False
    acqs = si_exp.get("acquisitions", {})
    ei = _load_epoch_info()
    changed = False
    for acq_id, entry in acqs.items():
        n_epochs = entry.get("n_epochs")
        if n_epochs is None or n_epochs <= 1:
            continue
        loc, acq = acq_id.split("--")
        acq_entry = ei.get(subject, {}).get(exp, {}).get(acq_id)
        if acq_entry is not None and _entry_is_complete(acq_entry, n_epochs):
            continue
        epochs = _sync._list_epoch_start_times(subject, exp, loc, acq)
        ei.setdefault(subject, {}).setdefault(exp, {})[acq_id] = (
            _compute_epochs_for_acq(subject, exp, loc, acq, epochs, si_exp)
        )
        changed = True
    if changed:
        _save_epoch_info(ei)
    return changed
