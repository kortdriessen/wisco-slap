"""Tests for ``wisco_slap.util.validity``.

The key correctness target is the cursor-walk tiling in
:func:`valid_state_epochs` — every row of the worked-examples table in
the design plan is locked here as a parametrized case. The mask /
duration / interval primitives get smaller targeted tests.

All synthetic inputs are built in the test bodies (no fixture files) so
the test reads as documentation of expected behavior.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import xarray as xr
from electro_py.hypno.hypno import Hypnogram

import wisco_slap as wis


# ----------------------------------------------------------------- helpers


def _grid(t_max: float, dt: float = 0.005) -> np.ndarray:
    """200 Hz uniform time grid covering [0, t_max)."""
    n = int(round(t_max / dt))
    return np.arange(n, dtype=float) * dt


def _mask(time: np.ndarray, nan_intervals: list[tuple[float, float]]) -> xr.DataArray:
    """Build a 1-D boolean mask: True everywhere except in given intervals."""
    valid = np.ones(time.shape, dtype=bool)
    for lo, hi in nan_intervals:
        valid[(time >= lo) & (time < hi)] = False
    return xr.DataArray(valid, dims=("time",), coords={"time": time}, name="is_valid")


def _hypno(rows: list[tuple[str, float, float]]) -> Hypnogram:
    """Build a Hypnogram from a list of (state, start, end) tuples."""
    return Hypnogram(
        pl.DataFrame(
            {
                "state": [r[0] for r in rows],
                "start_time": [r[1] for r in rows],
                "end_time": [r[2] for r in rows],
            }
        )
    )


def _scopex(time: np.ndarray, n_syn: int, nan_intervals: list[tuple[float, float]]) -> xr.DataArray:
    """Build a synthetic scopex-shaped DataArray with NaN intervals shared across syns."""
    data = np.ones((n_syn, time.size), dtype=float)
    for lo, hi in nan_intervals:
        data[:, (time >= lo) & (time < hi)] = np.nan
    return xr.DataArray(
        data,
        dims=("syn_id", "time"),
        coords={"syn_id": np.arange(n_syn), "time": time},
    )


# ---------------------------------------------------------- validity_mask


def test_validity_mask_all_collapses_when_any_syn_nan() -> None:
    time = _grid(1.0)
    da = _scopex(time, n_syn=3, nan_intervals=[])
    da.values[0, 100:120] = np.nan  # only syn 0 NaN at samples 100..119
    mask = wis.validity.validity_mask(da, mode="all")
    assert mask.dims == ("time",)
    expected = np.ones(time.size, dtype=bool)
    expected[100:120] = False
    np.testing.assert_array_equal(mask.values, expected)


def test_validity_mask_any_keeps_timepoint_when_one_syn_valid() -> None:
    time = _grid(1.0)
    da = _scopex(time, n_syn=3, nan_intervals=[])
    da.values[0:2, 100:120] = np.nan  # two of three syns NaN, one still valid
    mask = wis.validity.validity_mask(da, mode="any")
    np.testing.assert_array_equal(mask.values, np.ones(time.size, dtype=bool))


def test_validity_mask_per_syn_returns_2d() -> None:
    time = _grid(1.0)
    da = _scopex(time, n_syn=3, nan_intervals=[])
    da.values[0, 100:120] = np.nan
    mask = wis.validity.validity_mask(da, mode="per_syn")
    assert mask.dims == ("syn_id", "time")
    assert mask.values[0, 100:120].sum() == 0  # syn 0 invalid here
    assert mask.values[1, 100:120].all()       # syn 1 still valid


def test_validity_mask_rejects_multichannel() -> None:
    time = _grid(1.0)
    da = xr.DataArray(
        np.ones((2, 3, time.size), dtype=float),
        dims=("channel", "syn_id", "time"),
        coords={"channel": [0, 1], "syn_id": np.arange(3), "time": time},
    )
    with pytest.raises(ValueError, match="select one"):
        wis.validity.validity_mask(da)


def test_validity_mask_squeezes_singleton_channel() -> None:
    time = _grid(1.0)
    da = xr.DataArray(
        np.ones((1, 3, time.size), dtype=float),
        dims=("channel", "syn_id", "time"),
        coords={"channel": [0], "syn_id": np.arange(3), "time": time},
    )
    mask = wis.validity.validity_mask(da)
    assert mask.dims == ("time",)
    assert mask.values.all()


# ------------------------------------------------------------ resolve_mask


def test_resolve_mask_explicit_wins() -> None:
    time = _grid(1.0)
    da = _scopex(time, n_syn=2, nan_intervals=[(0.0, 0.5)])
    explicit = xr.DataArray(np.ones(time.size, dtype=bool), dims=("time",), coords={"time": time})
    out = wis.validity.resolve_mask(da, mask=explicit)
    assert out.values.all()


def test_resolve_mask_uses_attached_is_valid() -> None:
    time = _grid(1.0)
    da = _scopex(time, n_syn=2, nan_intervals=[])
    custom = np.zeros(time.size, dtype=bool)
    custom[:50] = True
    da_attached = da.assign_coords(is_valid=("time", custom))
    out = wis.validity.resolve_mask(da_attached, mask=None)
    assert out.values[:50].all()
    assert not out.values[50:].any()


def test_resolve_mask_derives_from_float_array() -> None:
    time = _grid(1.0)
    da = _scopex(time, n_syn=2, nan_intervals=[(0.2, 0.3)])
    out = wis.validity.resolve_mask(da, mask=None)
    expected = np.ones(time.size, dtype=bool)
    expected[(time >= 0.2) & (time < 0.3)] = False
    np.testing.assert_array_equal(out.values, expected)


def test_resolve_mask_raises_on_bool_array_without_coord() -> None:
    time = _grid(1.0)
    bool_da = xr.DataArray(
        np.ones((2, time.size), dtype=bool),
        dims=("syn_id", "time"),
        coords={"syn_id": [0, 1], "time": time},
    )
    with pytest.raises(ValueError, match="explicit"):
        wis.validity.resolve_mask(bool_da, mask=None)


# ----------------------------------------------------------- valid_duration


def test_valid_duration_full_extent() -> None:
    time = _grid(2.0)  # 400 samples
    mask = _mask(time, nan_intervals=[(0.5, 1.0)])
    # 100 samples NaN, 300 samples valid → 1.5s.
    assert wis.validity.valid_duration(mask) == pytest.approx(1.5, abs=1e-9)


def test_valid_duration_with_window() -> None:
    time = _grid(2.0)
    mask = _mask(time, nan_intervals=[(0.5, 1.0)])
    # In [0, 0.5) all 100 samples valid → 0.5s.
    # The half-open semantic of the time-mask helper means 0.5 is NaN so include t=0.5? With
    # >=lo & <=hi inclusive bounds, sample at 0.5 is NaN → 100 valid samples = 0.5s.
    assert wis.validity.valid_duration(mask, t1=0.0, t2=0.499) == pytest.approx(0.5, abs=1e-3)


def test_validity_intervals_basic() -> None:
    time = _grid(2.0)
    mask = _mask(time, nan_intervals=[(0.5, 1.0), (1.5, 1.7)])
    df = wis.validity.validity_intervals(mask)
    assert df.height == 3
    # Run 1: [0, 0.5) — 100 samples → ends at sample 99 = 0.495s, duration = 100*dt = 0.5s.
    # Run 2: [1.0, 1.5) — 100 samples → starts at 1.0, ends at 1.495.
    # Run 3: [1.7, 2.0) — 60 samples → starts at 1.7, ends at 1.995.
    assert df.row(0, named=True)["start_time"] == pytest.approx(0.0, abs=1e-9)
    assert df.row(0, named=True)["duration"] == pytest.approx(0.5, abs=1e-9)
    assert df.row(1, named=True)["start_time"] == pytest.approx(1.0, abs=1e-9)
    assert df.row(2, named=True)["start_time"] == pytest.approx(1.7, abs=1e-9)


def test_validity_intervals_empty_mask() -> None:
    time = _grid(1.0)
    mask = _mask(time, nan_intervals=[(0.0, 1.0)])  # all NaN
    df = wis.validity.validity_intervals(mask)
    assert df.height == 0
    assert df.columns == ["start_time", "end_time", "duration"]


# ----------------------------------------------- valid_state_epochs (span)
#
# Each parametrize row corresponds to a row of the worked-examples table
# in /home/driessen2/.claude/plans/warm-doodling-horizon.md. expected is a
# list of (start_time, end_time_approx, valid_duration, wall_duration)
# tuples; end_time is "approximate" because it lands on a sample, so the
# actual value is one dt less than the human-readable bound. We allow
# 2*dt tolerance.


@pytest.mark.parametrize(
    "hypno_rows, nan_intervals, mode, max_nan_span, expected",
    [
        # NREM 0–24, NaN 10–12, ep=10s, span → (0,10), (12,22)
        (
            [("NREM", 0.0, 24.0)],
            [(10.0, 12.0)],
            "span", None,
            [(0.0, 10.0, 10.0, 10.0), (12.0, 22.0, 10.0, 10.0)],
        ),
        # NREM 0–24, NaN 10–12, ep=10s, strict → same
        (
            [("NREM", 0.0, 24.0)],
            [(10.0, 12.0)],
            "strict", None,
            [(0.0, 10.0, 10.0, 10.0), (12.0, 22.0, 10.0, 10.0)],
        ),
        # NREM 0–13, NaN 6–8, ep=10s, span → (0, 12, v=10, w=12)
        (
            [("NREM", 0.0, 13.0)],
            [(6.0, 8.0)],
            "span", None,
            [(0.0, 12.0, 10.0, 12.0)],
        ),
        # NREM 0–13, NaN 6–8, ep=10s, strict → ∅
        (
            [("NREM", 0.0, 13.0)],
            [(6.0, 8.0)],
            "strict", None,
            [],
        ),
        # NREM 0–30, NaN 8–12, ep=10s, span → (0,14,v=10,w=14), (14,24,v=10,w=10)
        (
            [("NREM", 0.0, 30.0)],
            [(8.0, 12.0)],
            "span", None,
            [(0.0, 14.0, 10.0, 14.0), (14.0, 24.0, 10.0, 10.0)],
        ),
        # NREM 0–30, NaN 8–12, ep=10s, strict → (12,22) only
        (
            [("NREM", 0.0, 30.0)],
            [(8.0, 12.0)],
            "strict", None,
            [(12.0, 22.0, 10.0, 10.0)],
        ),
        # NREM 0–30, NaN 0–3, ep=10s, span → (3,13), (13,23)
        (
            [("NREM", 0.0, 30.0)],
            [(0.0, 3.0)],
            "span", None,
            [(3.0, 13.0, 10.0, 10.0), (13.0, 23.0, 10.0, 10.0)],
        ),
        # NREM 0–30, NaN 0–30, ep=10s, span → ∅
        (
            [("NREM", 0.0, 30.0)],
            [(0.0, 30.0)],
            "span", None,
            [],
        ),
        # NREM/Wake/NREM split: ep=10s, span → (0,10), (20,30)
        (
            [("NREM", 0.0, 15.0), ("Wake", 15.0, 20.0), ("NREM", 20.0, 30.0)],
            [],
            "span", None,
            [(0.0, 10.0, 10.0, 10.0), (20.0, 30.0, 10.0, 10.0)],
        ),
        # max_nan_span allows small NaN: NREM 0–11, NaN 5.5–5.6, ep=10s, max=1.0 → (0,10.1)
        (
            [("NREM", 0.0, 11.0)],
            [(5.5, 5.6)],
            "span", 1.0,
            [(0.0, 10.1, 10.0, 10.1)],
        ),
        # max_nan_span rejects bigger NaN: NREM 0–50, NaN 8–38, ep=10s, max=1.0 → (38,48)
        (
            [("NREM", 0.0, 50.0)],
            [(8.0, 38.0)],
            "span", 1.0,
            [(38.0, 48.0, 10.0, 10.0)],
        ),
        # No max_nan_span → unconstrained span: NREM 0–50, NaN 8–38.
        # First epoch spans the gap (wall=40, valid=10). Cursor lands at
        # t=40; the remaining 10s of NREM cleanly fits a second 10s epoch.
        # (The plan's worked-examples table missed the second epoch.)
        (
            [("NREM", 0.0, 50.0)],
            [(8.0, 38.0)],
            "span", None,
            [(0.0, 40.0, 10.0, 40.0), (40.0, 50.0, 10.0, 10.0)],
        ),
    ],
)
def test_valid_state_epochs_worked_examples(hypno_rows, nan_intervals, mode, max_nan_span, expected) -> None:
    dt = 0.005
    # Make the grid extend a bit past the largest end_time so we have room.
    t_max = max(r[2] for r in hypno_rows) + 1.0
    time = _grid(t_max, dt=dt)
    mask = _mask(time, nan_intervals)
    hypno = _hypno(hypno_rows)

    df = wis.validity.valid_state_epochs(
        hypno, mask, "NREM", epoch_length=10.0,
        mode=mode, max_nan_span=max_nan_span,
    )

    assert df.height == len(expected), (
        f"Expected {len(expected)} epochs, got {df.height}: {df}"
    )
    for i, (exp_start, exp_end, exp_valid, exp_wall) in enumerate(expected):
        row = df.row(i, named=True)
        # start_time should land within one dt of the expected boundary.
        assert row["start_time"] == pytest.approx(exp_start, abs=1.5 * dt), (
            f"row {i} start_time mismatch: got {row['start_time']}, expected {exp_start}"
        )
        # end_time is the time-coord of the last valid sample. Allow ±dt
        # because end_time = exp_end - dt (the last valid sample), so the
        # natural tolerance is dt, plus boundary-rounding slack.
        assert row["end_time"] == pytest.approx(exp_end - dt, abs=1.5 * dt), (
            f"row {i} end_time mismatch: got {row['end_time']}, expected ~{exp_end - dt}"
        )
        assert row["valid_duration"] == pytest.approx(exp_valid, abs=1.5 * dt)
        assert row["wall_duration"] == pytest.approx(exp_wall, abs=1.5 * dt)


def test_valid_state_epochs_min_bout_length_filters() -> None:
    time = _grid(60.0)
    mask = _mask(time, nan_intervals=[])
    hypno = _hypno([
        ("NREM", 0.0, 5.0),    # short bout (5s) — should be excluded
        ("Wake", 5.0, 10.0),
        ("NREM", 10.0, 30.0),  # long bout (20s) — keeps two 10s epochs
    ])
    df = wis.validity.valid_state_epochs(
        hypno, mask, "NREM", epoch_length=10.0, min_bout_length=15.0,
    )
    assert df.height == 2
    assert df.row(0, named=True)["start_time"] == pytest.approx(10.0, abs=0.01)


def test_valid_state_epochs_n_epochs_subsample() -> None:
    time = _grid(120.0)
    mask = _mask(time, nan_intervals=[])
    hypno = _hypno([("NREM", 0.0, 100.0)])  # fits 10 ten-second epochs
    df_all = wis.validity.valid_state_epochs(hypno, mask, "NREM", epoch_length=10.0)
    assert df_all.height == 10

    df_sub = wis.validity.valid_state_epochs(
        hypno, mask, "NREM", epoch_length=10.0, n_epochs=3, seed=42,
    )
    assert df_sub.height == 3


def test_valid_state_epochs_n_epochs_int_requires_seed() -> None:
    time = _grid(120.0)
    mask = _mask(time, nan_intervals=[])
    hypno = _hypno([("NREM", 0.0, 100.0)])
    with pytest.raises(ValueError, match="seed"):
        wis.validity.valid_state_epochs(
            hypno, mask, "NREM", epoch_length=10.0, n_epochs=3,
        )


def test_valid_state_intervals_unions_states() -> None:
    time = _grid(30.0)
    mask = _mask(time, nan_intervals=[(10.0, 12.0)])
    hypno = _hypno([
        ("NREM", 0.0, 10.0),
        ("Wake", 10.0, 20.0),
        ("REM",  20.0, 30.0),
    ])
    # NREM ∪ Wake = [0, 20). NaN at [10, 12) splits into [0, 10) and [12, 20).
    df = wis.validity.valid_state_intervals(hypno, mask, ["NREM", "Wake"])
    assert df.height == 2


# ------------------------------------------------------ event_rate_per_state


def test_event_rate_per_state_uses_valid_denominator() -> None:
    time = _grid(20.0)
    # NaN from 5–10s: 5s of dead time inside the 10s NREM bout.
    mask_nan = [(5.0, 10.0)]
    mask = _mask(time, mask_nan)

    # 2 syns, 4 events on syn0 placed at t = 0.5, 1.5, 2.5, 3.5 (all within
    # NREM and within valid time). syn1 has 2 events at t = 11, 12 (within
    # NREM, valid).
    n_syn = 2
    events = np.zeros((n_syn, time.size), dtype=bool)

    def _set(s, t):
        events[s, int(round(t / 0.005))] = True

    for t in [0.5, 1.5, 2.5, 3.5]:
        _set(0, t)
    for t in [11.0, 12.0]:
        _set(1, t)

    ev = xr.DataArray(
        events,
        dims=("syn_id", "time"),
        coords={"syn_id": [0, 1], "time": time},
    )
    hypno = _hypno([("NREM", 0.0, 15.0), ("Wake", 15.0, 20.0)])

    df = wis.validity.event_rate_per_state(ev, hypno, mask=mask, states=["NREM"])

    # Valid duration in NREM = 15s wall - 5s NaN = 10s.
    syn0 = df.filter(pl.col("syn_id") == 0).row(0, named=True)
    syn1 = df.filter(pl.col("syn_id") == 1).row(0, named=True)
    assert syn0["valid_duration_s"] == pytest.approx(10.0, abs=0.05)
    assert syn0["n_events"] == 4
    assert syn0["rate_hz"] == pytest.approx(0.4, abs=1e-3)
    assert syn1["n_events"] == 2
    assert syn1["rate_hz"] == pytest.approx(0.2, abs=1e-3)


def test_event_rate_per_state_requires_mask_for_bool_array() -> None:
    time = _grid(20.0)
    ev = xr.DataArray(
        np.zeros((2, time.size), dtype=bool),
        dims=("syn_id", "time"),
        coords={"syn_id": [0, 1], "time": time},
    )
    hypno = _hypno([("NREM", 0.0, 10.0)])
    with pytest.raises(ValueError, match="explicit"):
        wis.validity.event_rate_per_state(ev, hypno, mask=None)


def test_add_valid_duration_per_bout() -> None:
    time = _grid(30.0)
    # NaN 0-5 sits inside bout 1; NaN 22-23 sits inside bout 3.
    mask = _mask(time, nan_intervals=[(0.0, 5.0), (22.0, 23.0)])
    hypno = _hypno([
        ("NREM", 0.0, 10.0),   # 5s valid (5s NaN at start)
        ("Wake", 10.0, 20.0),  # 10s valid
        ("REM",  20.0, 30.0),  # 9s valid (1s NaN inside)
    ])
    out = wis.validity.add_valid_duration(hypno, mask)
    assert "valid_duration" in out.df.columns
    vd = out.df["valid_duration"].to_list()
    assert vd[0] == pytest.approx(5.0, abs=0.05)
    assert vd[1] == pytest.approx(10.0, abs=0.05)
    assert vd[2] == pytest.approx(9.0, abs=0.05)
    # Original state ordering / count preserved.
    assert out.df["state"].to_list() == ["NREM", "Wake", "REM"]


def test_add_valid_duration_bout_outside_mask_coverage() -> None:
    # Bout 1 lives entirely after the mask ends → 0s valid.
    time = _grid(10.0)
    mask = _mask(time, nan_intervals=[])
    hypno = _hypno([
        ("NREM", 0.0, 5.0),
        ("Wake", 20.0, 30.0),
    ])
    out = wis.validity.add_valid_duration(hypno, mask)
    vd = out.df["valid_duration"].to_list()
    assert vd[0] == pytest.approx(5.0, abs=0.05)
    assert vd[1] == 0.0


def test_valid_duration_per_state() -> None:
    time = _grid(30.0)
    mask = _mask(time, nan_intervals=[(0.0, 5.0)])
    hypno = _hypno([
        ("NREM", 0.0, 10.0),  # NaN 0-5 inside → 5s valid
        ("Wake", 10.0, 20.0), # all valid → 10s valid
        ("REM",  20.0, 30.0), # all valid → 10s valid
    ])
    df = wis.validity.valid_duration_per_state(mask, hypno)
    by_state = {row["state"]: row["valid_duration_s"] for row in df.iter_rows(named=True)}
    assert by_state["NREM"] == pytest.approx(5.0, abs=0.05)
    assert by_state["Wake"] == pytest.approx(10.0, abs=0.05)
    assert by_state["REM"] == pytest.approx(10.0, abs=0.05)
