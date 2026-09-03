"""Tests for shaft-calcium disc geometry and trace extraction helpers."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from wisco_slap.pns.aligned_frames import signal_channel
from wisco_slap.pns.shaft_ca_gen import (
    PROX_LINES_MIN_SAVE_DATE,
    build_discs,
    disc_means_over_blocks,
    load_prox_lines,
    prox_lines_problem,
)


def _write_prox_csv(
    path: Path, dend_ids: tuple[str | None, str | None] | None = ("A-1", "B-1")
) -> None:
    """Write a WISynaptic-format prox_lines CSV with two polylines.

    ``dend_ids`` gives the per-line ``dend-ID`` values (``None`` = blank);
    passing ``None`` for the whole tuple omits the column entirely.
    """

    header = "index,shape-type,vertex-index,axis-0,axis-1"
    if dend_ids is not None:
        header += ",dend-ID"
    lines = [header]

    def _row(index: int, v: int, r: float, c: float) -> str:
        row = f"{index},path,{v},{r},{c}"
        if dend_ids is not None:
            row += f",{dend_ids[index] or ''}"
        return row

    # dendrite 0: proximal at (10, 5), running right to (10, 45)
    for v, (r, c) in enumerate([(10.0, 5.0), (10.0, 25.0), (10.0, 45.0)]):
        lines.append(_row(0, v, r, c))
    # dendrite 1: proximal at (30, 40), running LEFT to (30, 20)
    for v, (r, c) in enumerate([(30.0, 40.0), (30.0, 20.0)]):
        lines.append(_row(1, v, r, c))
    path.write_text("\n".join(lines) + "\n")


def test_load_prox_lines_uses_wisynaptic_dend_ids(tmp_path: Path) -> None:
    csv = tmp_path / "prox_lines_dmd1.csv"
    _write_prox_csv(csv, dend_ids=("A-1", "B-2"))
    lines = load_prox_lines(csv)
    assert [line["dendrite_id"] for line in lines] == ["A-1", "B-2"]
    assert prox_lines_problem(csv) is None


def test_load_prox_lines_tolerates_padded_ids(tmp_path: Path) -> None:
    csv = tmp_path / "prox_lines_dmd1.csv"
    _write_prox_csv(csv, dend_ids=(" A-1 ", "B-1"))
    assert [line["dendrite_id"] for line in load_prox_lines(csv)] == ["A-1", "B-1"]


@pytest.mark.parametrize(
    ("dend_ids", "fragment"),
    [
        (None, "no 'dend-ID' column"),
        (("A-1", None), "have no dend-ID"),
        (("A-1", "A-1"), "duplicate dend-IDs"),
    ],
)
def test_unusable_prox_lines_are_reported_and_rejected(
    tmp_path: Path, dend_ids, fragment: str
) -> None:
    csv = tmp_path / "prox_lines_dmd1.csv"
    _write_prox_csv(csv, dend_ids=dend_ids)
    problem = prox_lines_problem(csv)
    assert problem is not None and fragment in problem
    with pytest.raises(ValueError, match=fragment.replace("'", "'")):
        load_prox_lines(csv)


def test_prox_lines_saved_before_cutoff_are_rejected(tmp_path: Path) -> None:
    csv = tmp_path / "prox_lines_dmd1.csv"
    _write_prox_csv(csv, dend_ids=("A-1", "B-2"))
    stale = (PROX_LINES_MIN_SAVE_DATE - timedelta(days=1)).timestamp()
    os.utime(csv, (stale, stale))
    problem = prox_lines_problem(csv)
    assert problem is not None and "channel-fix cutoff" in problem


def test_load_prox_lines_preserves_vertex_order(tmp_path: Path) -> None:
    csv = tmp_path / "prox_lines_dmd1.csv"
    _write_prox_csv(csv)
    lines = load_prox_lines(csv)
    assert [line["dendrite_id"] for line in lines] == ["A-1", "B-1"]
    # Vertex 0 is the proximal end, exactly as drawn.
    assert lines[0]["vertices"][0].tolist() == [10.0, 5.0]
    assert lines[1]["vertices"][0].tolist() == [30.0, 40.0]
    assert lines[1]["vertices"][-1].tolist() == [30.0, 20.0]


def test_build_discs_arc_starts_at_proximal_end(tmp_path: Path) -> None:
    csv = tmp_path / "prox_lines_dmd1.csv"
    _write_prox_csv(csv)
    lines = load_prox_lines(csv)
    support = np.ones((64, 64), dtype=bool)
    discs = build_discs(
        lines,
        image_shape=(64, 64),
        pixel_size_um=0.25,
        disc_diameter_um=1.0,
        disc_spacing_um=1.0,
        valid_mask=support,
    )
    d0 = [d for d in discs if d["dendrite_id"] == "A-1"]
    assert d0[0]["disc_id"] == "A-1:000"
    # 40 px = 10 um at 1 um spacing -> 11 grid samples (plus none clipped).
    assert len(d0) == 11
    assert d0[0]["arc_um"] == 0.0
    assert d0[0]["center_col"] == pytest.approx(5.0)  # proximal vertex
    assert d0[-1]["arc_um"] == pytest.approx(10.0)
    assert d0[-1]["center_col"] == pytest.approx(45.0)
    # dendrite 1 runs right-to-left: arc 0 must still be its FIRST vertex.
    d1 = [d for d in discs if d["dendrite_id"] == "B-1"]
    assert d1[0]["arc_um"] == 0.0
    assert d1[0]["center_col"] == pytest.approx(40.0)
    assert d1[-1]["center_col"] == pytest.approx(20.0)
    assert all(d["flat_pixels"].size > 0 for d in discs)


def test_build_discs_respects_valid_mask() -> None:
    lines = [{"dendrite_id": "d", "vertices": np.array([[10.0, 2.0], [10.0, 30.0]])}]
    valid = np.zeros((32, 32), dtype=bool)
    valid[:, :16] = True  # right half unscanned
    discs = build_discs(
        lines,
        image_shape=(32, 32),
        pixel_size_um=0.25,
        disc_diameter_um=1.0,
        disc_spacing_um=1.0,
        valid_mask=valid,
    )
    assert discs, "discs on the valid half must survive"
    assert all(d["center_col"] < 18 for d in discs)
    for disc in discs:
        rows, cols = np.unravel_index(disc["flat_pixels"], (32, 32))
        assert (cols < 16).all()


def test_disc_means_are_nan_aware() -> None:
    values = np.full((6, 5), np.nan, dtype=np.float32)
    values[:, 0] = [1, 2, 3, 4, 5, 6]
    values[:3, 1] = 10.0
    values[:, 2] = 100.0
    traces = disc_means_over_blocks(
        values,
        [np.array([0, 1]), np.array([2]), np.array([3, 4])],
        block_frames=4,
    )
    assert traces.shape == (3, 6)
    assert traces[0, 0] == pytest.approx((1 + 10) / 2)  # both finite
    assert traces[0, 4] == pytest.approx(5.0)  # column 1 NaN -> mean of rest
    assert np.array_equal(traces[1], np.full(6, 100.0))
    assert np.isnan(traces[2]).all()  # all-NaN columns


def test_signal_channel_resolves_calcium_and_glutamate() -> None:
    schema = {
        "signalByRawChannel": [["calcium"], ["glutamate"]],
        "rawChannelNumbers": [[1.0], [2.0]],
    }
    calcium = signal_channel(schema, "calcium")
    assert calcium["raw_channel"] == 1
    assert calcium["array_index"] == 0
    glutamate = signal_channel(schema, "glutamate")
    assert glutamate["raw_channel"] == 2
    assert glutamate["array_index"] == 1
    with pytest.raises(ValueError, match="no 'voltage' channel"):
        signal_channel(schema, "voltage")
