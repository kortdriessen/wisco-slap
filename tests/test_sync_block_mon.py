"""Tests for sync_block_mon's pure bookkeeping helpers (synthetic inputs only)."""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Whisker-mask borrowing for acquisition-less sync blocks
# ---------------------------------------------------------------------------


def _fake_sync_info(acq_blocks: dict[str, int], n_blocks: int) -> dict:
    return {
        "subj": {
            "exp_1": {
                "sync_blocks": {b: {} for b in range(1, n_blocks + 1)},
                "acquisitions": {k: {"sync_block": v} for k, v in acq_blocks.items()},
            }
        }
    }


@pytest.fixture
def whisk_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import wisco_slap as wis
    from wisco_slap import defs

    monkeypatch.setattr(defs, "anmat_root", str(tmp_path))

    def install(acq_blocks: dict[str, int], n_blocks: int, masks: list[int]):
        monkeypatch.setattr(
            wis.meta.get, "sync_info", lambda: _fake_sync_info(acq_blocks, n_blocks)
        )
        for b in masks:
            d = tmp_path / "subj" / "exp_1" / "sync_block_data" / f"sync_block-{b}"
            (d / "whisking").mkdir(parents=True, exist_ok=True)
            (d / "whisking" / "mask.tif").write_bytes(f"mask-{b}".encode())
        return tmp_path / "subj" / "exp_1" / "sync_block_data"

    return install


def test_borrow_applies_only_to_acquisition_less_blocks(whisk_env) -> None:
    from wisco_slap.pns.sync_block_mon import _borrowable_whisk_mask

    root = whisk_env({"loc_A--acq_1": 1}, n_blocks=2, masks=[1])
    donor = _borrowable_whisk_mask("subj", "exp_1", 2)
    assert donor == str(root / "sync_block-1" / "whisking" / "mask.tif")
    # A block that has its own mask never borrows.
    assert _borrowable_whisk_mask("subj", "exp_1", 1) is None


def test_no_borrow_when_block_has_acquisitions(whisk_env) -> None:
    from wisco_slap.pns.sync_block_mon import _borrowable_whisk_mask

    whisk_env({"loc_A--acq_1": 1, "loc_B--acq_1": 2}, n_blocks=2, masks=[1])
    assert _borrowable_whisk_mask("subj", "exp_1", 2) is None


def test_no_borrow_from_donor_without_acquisitions_or_mask(whisk_env) -> None:
    from wisco_slap.pns.sync_block_mon import _borrowable_whisk_mask

    # Block 1 has a mask but no acquisitions; block 2 has acquisitions but
    # no mask. Neither qualifies as a donor for block 3.
    whisk_env({"loc_A--acq_1": 2}, n_blocks=3, masks=[1])
    assert _borrowable_whisk_mask("subj", "exp_1", 3) is None


def test_nearest_donor_wins_with_lower_block_on_ties(whisk_env) -> None:
    from wisco_slap.pns.sync_block_mon import _borrowable_whisk_mask

    root = whisk_env({"a": 1, "b": 3, "c": 5}, n_blocks=5, masks=[1, 3, 5])
    assert _borrowable_whisk_mask("subj", "exp_1", 4).startswith(
        str(root / "sync_block-3")
    )
    assert _borrowable_whisk_mask("subj", "exp_1", 2).startswith(
        str(root / "sync_block-1")
    )


def test_borrow_copies_mask_and_records_provenance(whisk_env) -> None:
    from wisco_slap.pns.sync_block_mon import (
        _borrow_whisk_mask,
        _borrowable_whisk_mask,
        _check_whisk_mask,
    )

    root = whisk_env({"loc_A--acq_1": 1}, n_blocks=2, masks=[1])
    donor = _borrowable_whisk_mask("subj", "exp_1", 2)
    target = Path(_borrow_whisk_mask("subj", "exp_1", 2, donor))
    assert target == root / "sync_block-2" / "whisking" / "mask.tif"
    assert target.read_bytes() == b"mask-1"
    assert _check_whisk_mask("subj", "exp_1", 2)
    source_note = (root / "sync_block-2" / "whisking" / "mask_source.txt").read_text()
    assert "sync_block-1" in source_note
    # Once the copy exists the block has its own mask and no longer borrows.
    assert _borrowable_whisk_mask("subj", "exp_1", 2) is None
