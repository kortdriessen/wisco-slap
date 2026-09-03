"""Tests for ``wisco_slap.scope.rates``.

The correctness targets are (1) the denominator is always *valid* time, not
wall-clock time; (2) an event on a NaN sample never counts; (3) the complete
synapse universe (including zero-event synapses) is represented; and (4) each
partitioning (state, bout, fraction, epoch) sums events and valid time over
the right sample set.

All synthetic inputs are built in the test bodies so the tests read as
documentation of expected behaviour. Inputs mirror the real pipeline: a
``final_peaks``-shaped boolean DataArray carrying an ``is_valid`` time coord
and per-synapse metadata coords, paired with a ``peaks_df`` whose ``peak_idx``
indexes the same grid.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import xarray as xr
from electro_py.hypno.hypno import Hypnogram

import wisco_slap as wis

DT = 0.005


def _grid(t_max: float, dt: float = DT) -> np.ndarray:
    n = int(round(t_max / dt))
    return np.arange(n, dtype=float) * dt


def _hypno(rows: list[tuple[str, float, float]]) -> Hypnogram:
    return Hypnogram(
        pl.DataFrame(
            {
                "state": [r[0] for r in rows],
                "start_time": [r[1] for r in rows],
                "end_time": [r[2] for r in rows],
            }
        )
    )


def _make_inputs(
    time: np.ndarray,
    nan_intervals: list[tuple[float, float]],
    events: dict[str, list[float]],
    dend_ids: dict[str, str] | None = None,
) -> tuple[xr.DataArray, pl.DataFrame, xr.DataArray]:
    """Build a (final_peaks, peaks_df, mask) triple from an event spec.

    ``events`` maps syn_id -> event times (seconds). A syn with an empty list
    is a zero-event synapse (present in final_peaks, absent from peaks_df).
    """
    valid = np.ones(time.size, dtype=bool)
    for lo, hi in nan_intervals:
        valid[(time >= lo) & (time < hi)] = False

    syn_ids = list(events.keys())
    ev = np.zeros((len(syn_ids), time.size), dtype=bool)
    rows: list[dict] = []
    for si, sid in enumerate(syn_ids):
        for t in events[sid]:
            idx = int(round(t / DT))
            ev[si, idx] = True
            rows.append(
                {
                    "syn_id": sid,
                    "peak_idx": idx,
                    "peak_time": float(time[idx]),
                    "amplitude": 1.0,
                    "sigma": 0.2,
                    "snr": 5.0,
                    "passed_snr": True,
                    "final": True,
                }
            )

    coords: dict = {"syn_id": syn_ids, "time": time, "is_valid": ("time", valid)}
    if dend_ids is not None:
        coords["dend-ID"] = ("syn_id", [dend_ids[s] for s in syn_ids])
    final_peaks = xr.DataArray(ev, dims=("syn_id", "time"), coords=coords)

    peaks_df = pl.DataFrame(
        rows,
        schema_overrides={"syn_id": pl.String, "peak_idx": pl.Int64},
    )
    if dend_ids is not None:
        meta = pl.DataFrame(
            {"syn_id": syn_ids, "dend-ID": [dend_ids[s] for s in syn_ids]}
        )
        peaks_df = peaks_df.join(meta, on="syn_id", how="left")

    mask = xr.DataArray(valid, dims=("time",), coords={"time": time}, name="is_valid")
    return final_peaks, peaks_df, mask


# NaN 5-10 sits inside the NREM bout; events at 7.0 land on a NaN sample.
def _standard():
    time = _grid(20.0)
    events = {
        "1-0": [0.5, 1.5, 2.5, 3.5, 7.0],  # 7.0 is on a NaN sample -> excluded
        "1-1": [11.0, 12.0],
        "1-2": [],  # zero-event synapse
    }
    dend_ids = {"1-0": "d1", "1-1": "d2", "1-2": "d3"}
    final_peaks, peaks_df, mask = _make_inputs(
        time, [(5.0, 10.0)], events, dend_ids
    )
    hypno = _hypno([("NREM", 0.0, 10.0), ("Wake", 10.0, 20.0)])
    return final_peaks, peaks_df, hypno


# --------------------------------------------------------------- compute_rate


def test_by_state_uses_valid_denominator_and_excludes_nan_events() -> None:
    final_peaks, peaks_df, hypno = _standard()
    df = wis.scope.rates.compute_rate(peaks_df, hypno, final_peaks, by="state")

    # 3 synapses x 2 states = 6 rows, all metadata present.
    assert df.height == 6
    assert "dend-ID" in df.columns
    row = {(r["syn_id"], r["state"]): r for r in df.iter_rows(named=True)}

    # NREM valid = 10s wall - 5s NaN = 5s. The 7.0 event is on NaN -> not counted.
    assert row[("1-0", "NREM")]["valid_duration_s"] == pytest.approx(5.0, abs=0.05)
    assert row[("1-0", "NREM")]["n_events"] == 4
    assert row[("1-0", "NREM")]["rate_hz"] == pytest.approx(0.8, abs=1e-3)
    # Wake fully valid = 10s.
    assert row[("1-1", "Wake")]["valid_duration_s"] == pytest.approx(10.0, abs=0.05)
    assert row[("1-1", "Wake")]["n_events"] == 2
    assert row[("1-1", "Wake")]["rate_hz"] == pytest.approx(0.2, abs=1e-3)
    # Zero-event synapse present everywhere with rate 0.
    assert row[("1-2", "NREM")]["n_events"] == 0
    assert row[("1-2", "NREM")]["rate_hz"] == 0.0


def test_by_state_carries_state_summary_columns() -> None:
    final_peaks, peaks_df, hypno = _standard()
    df = wis.scope.rates.compute_rate_by_state(peaks_df, hypno, final_peaks)
    nrem = df.filter(pl.col("state") == "NREM").row(0, named=True)
    assert nrem["n_bouts"] == 1
    assert nrem["wall_duration_s"] == pytest.approx(10.0, abs=1e-6)


def test_by_bout_matches_by_state_for_one_bout_each() -> None:
    final_peaks, peaks_df, hypno = _standard()
    df = wis.scope.rates.compute_rate_by_bout(peaks_df, hypno, final_peaks)
    # One NREM bout, one Wake bout -> same numbers as by_state.
    assert df.height == 6
    assert set(df.columns) >= {"bout_idx", "start_time", "end_time", "duration"}
    r = df.filter((pl.col("syn_id") == "1-0") & (pl.col("state") == "NREM")).row(
        0, named=True
    )
    assert r["bout_idx"] == 0
    assert r["n_events"] == 4
    assert r["rate_hz"] == pytest.approx(0.8, abs=1e-3)


# ------------------------------------------------- compute_rate: hypno context


def _standard_with_context(extra: dict[str, list] | None = None):
    """``_standard()`` but the hypnogram carries extra per-bout columns.

    Mirrors a real diced hypnogram that has had bandpower / elapsed-time /
    valid-duration columns attached before the rate call.
    """
    final_peaks, peaks_df, _ = _standard()
    cols = {
        "state": ["NREM", "Wake"],
        "start_time": [0.0, 10.0],
        "end_time": [10.0, 20.0],
        "delta": [3.5, 0.5],
        "valid_duration": [5.0, 10.0],
        **(extra or {}),
    }
    return final_peaks, peaks_df, Hypnogram(pl.DataFrame(cols))


def test_by_bout_joins_hypno_context_by_default() -> None:
    final_peaks, peaks_df, hypno = _standard_with_context()
    df = wis.scope.rates.compute_rate_by_bout(peaks_df, hypno, final_peaks)

    assert {"delta", "valid_duration"} <= set(df.columns)
    # Every row carries its own bout's values, and no bout is left unmatched.
    assert df["delta"].null_count() == 0
    for row in df.iter_rows(named=True):
        assert row["delta"] == (3.5 if row["state"] == "NREM" else 0.5)
        assert row["valid_duration"] == (5.0 if row["state"] == "NREM" else 10.0)


def test_hypno_context_does_not_duplicate_descriptor_columns() -> None:
    final_peaks, peaks_df, hypno = _standard_with_context()
    df = wis.scope.rates.compute_rate_by_bout(peaks_df, hypno, final_peaks)
    # state/start_time/end_time/duration come from the descriptor; the
    # hypnogram's copies must not reappear under a suffixed name.
    assert not [c for c in df.columns if c.endswith("_right")]
    assert df.columns.count("duration") == 1
    # The hypnogram's `valid_duration` and the computed `valid_duration_s` are
    # different quantities and both survive under their own names.
    assert {"valid_duration", "valid_duration_s"} <= set(df.columns)


def test_hypno_context_can_be_disabled() -> None:
    final_peaks, peaks_df, hypno = _standard_with_context()
    df = wis.scope.rates.compute_rate(
        peaks_df, hypno, final_peaks, by="bout", join_hypno_context=False
    )
    assert "delta" not in df.columns
    assert "valid_duration" not in df.columns


def test_hypno_context_ignored_for_by_state() -> None:
    final_peaks, peaks_df, hypno = _standard_with_context()
    df = wis.scope.rates.compute_rate(peaks_df, hypno, final_peaks, by="state")
    # A state row pools bouts, so per-bout values have no single meaning.
    assert "delta" not in df.columns


def test_hypno_context_warns_and_skips_shadowing_column() -> None:
    # 'dend-ID' is a per-synapse metadata column; a hypnogram column of the
    # same name must not clobber it.
    final_peaks, peaks_df, hypno = _standard_with_context(
        {"dend-ID": ["bout-level", "bout-level"]}
    )
    with pytest.warns(UserWarning, match="dend-ID"):
        df = wis.scope.rates.compute_rate_by_bout(peaks_df, hypno, final_peaks)

    assert set(df["dend-ID"].unique()) == {"d1", "d2", "d3"}
    assert "delta" in df.columns  # the non-clashing columns still come through


def test_hypno_context_survives_bout_filtering() -> None:
    # Bouts dropped by `states` / `min_valid_duration_s` must not shift the
    # context join off by a row.
    final_peaks, peaks_df, hypno = _standard_with_context()
    df = wis.scope.rates.compute_rate(
        peaks_df, hypno, final_peaks, by="bout", states="Wake"
    )
    assert set(df["bout_idx"].unique()) == {1}
    assert set(df["delta"].unique()) == {0.5}


def test_states_filter_and_complete_grid_false() -> None:
    final_peaks, peaks_df, hypno = _standard()
    # Only NREM, and drop zero-event rows.
    df = wis.scope.rates.compute_rate(
        peaks_df, hypno, final_peaks, by="state",
        states="NREM", complete_grid=False,
    )
    assert set(df["state"].unique()) == {"NREM"}
    # Only syn 1-0 has events in NREM.
    assert set(df["syn_id"].to_list()) == {"1-0"}


def test_filtered_peaks_df_reduces_numerator() -> None:
    final_peaks, peaks_df, hypno = _standard()
    # Keep only 2 of syn 1-0's NREM events -> rate halves.
    sub = peaks_df.filter(
        ~((pl.col("syn_id") == "1-0") & (pl.col("peak_time").is_in([2.5, 3.5])))
    )
    df = wis.scope.rates.compute_rate(sub, hypno, final_peaks, by="state", states="NREM")
    r = df.filter(pl.col("syn_id") == "1-0").row(0, named=True)
    assert r["n_events"] == 2
    assert r["rate_hz"] == pytest.approx(0.4, abs=1e-3)


# --------------------------------------------------------------- rolling


def test_rolling_windows_valid_denominator_and_state_label() -> None:
    final_peaks, peaks_df, hypno = _standard()
    df = wis.scope.rates.compute_rate_rolling(
        peaks_df, hypno, final_peaks, window_s=5.0, step_s=5.0
    )
    # 4 windows x 3 syns = 12 rows.
    assert df.height == 12
    s0 = df.filter(pl.col("syn_id") == "1-0").sort("window_idx")
    # Window 0 [0,5): 4 events / 5s valid.
    w0 = s0.row(0, named=True)
    assert w0["n_events"] == 4
    assert w0["valid_duration_s"] == pytest.approx(5.0, abs=0.05)
    assert w0["rate_hz"] == pytest.approx(0.8, abs=1e-3)
    assert w0["state"] == "NREM"
    # Window 1 [5,10): all NaN -> zero valid time -> rate NaN, event at 7 excluded.
    w1 = s0.row(1, named=True)
    assert w1["valid_duration_s"] == pytest.approx(0.0, abs=1e-9)
    assert w1["n_events"] == 0
    assert np.isnan(w1["rate_hz"])
    # syn 1-1 in window 2 [10,15): 2 events / 5s.
    s1w2 = (
        df.filter((pl.col("syn_id") == "1-1") & (pl.col("window_idx") == 2))
        .row(0, named=True)
    )
    assert s1w2["n_events"] == 2
    assert s1w2["state"] == "Wake"
    assert s1w2["rate_hz"] == pytest.approx(0.4, abs=1e-3)


def test_rolling_min_valid_frac_drops_empty_windows() -> None:
    final_peaks, peaks_df, hypno = _standard()
    df = wis.scope.rates.compute_rate_rolling(
        peaks_df, hypno, final_peaks, window_s=5.0, step_s=5.0, min_valid_frac=0.5
    )
    # Window 1 (all NaN) is dropped for every syn -> 3 windows x 3 syns.
    assert df.height == 9
    assert 1 not in df["window_idx"].unique().to_list()


# --------------------------------------------------------------- fraction


def test_fraction_splits_state_by_equal_valid_time() -> None:
    final_peaks, peaks_df, hypno = _standard()
    df = wis.scope.rates.compute_rate_by_fraction(
        peaks_df, hypno, final_peaks, n_bins=2, within="state", states="NREM"
    )
    s0 = df.filter(pl.col("syn_id") == "1-0").sort("frac_bin")
    # NREM valid timeline [0,5) split in half: [0,2.5) and [2.5,5).
    early = s0.row(0, named=True)
    late = s0.row(1, named=True)
    assert early["frac_label"] == "early"
    assert late["frac_label"] == "late"
    assert early["valid_duration_s"] == pytest.approx(2.5, abs=0.05)
    assert late["valid_duration_s"] == pytest.approx(2.5, abs=0.05)
    # events 0.5,1.5 in early; 2.5,3.5 in late.
    assert early["n_events"] == 2
    assert late["n_events"] == 2
    assert early["rate_hz"] == pytest.approx(0.8, abs=1e-3)


def test_fraction_within_bout_has_bout_columns() -> None:
    final_peaks, peaks_df, hypno = _standard()
    df = wis.scope.rates.compute_rate_by_fraction(
        peaks_df, hypno, final_peaks, n_bins=3, within="bout", states="NREM"
    )
    assert {"bout_idx", "frac_bin", "frac_label"} <= set(df.columns)
    # 3 bins x 3 syns for the single NREM bout.
    assert df.height == 9
    assert set(df["frac_label"].unique()) == {"early", "middle", "late"}


# --------------------------------------------------------------- epoch


def test_epoch_fixed_valid_duration() -> None:
    final_peaks, peaks_df, hypno = _standard()
    df = wis.scope.rates.compute_rate_by_epoch(
        peaks_df, hypno, final_peaks, epoch_length_s=2.5, mode="strict", states="NREM"
    )
    # NREM valid [0,5) -> two 2.5s epochs, x3 syns = 6 rows.
    assert df.height == 6
    s0 = df.filter(pl.col("syn_id") == "1-0").sort("epoch_idx")
    for r in s0.iter_rows(named=True):
        assert r["valid_duration_s"] == pytest.approx(2.5, abs=0.05)
        assert r["n_events"] == 2
        assert r["rate_hz"] == pytest.approx(0.8, abs=1e-3)


# --------------------------------------------------------------- guards


def test_mismatched_peak_idx_raises() -> None:
    final_peaks, peaks_df, hypno = _standard()
    bad = peaks_df.with_columns((pl.col("peak_idx") + 10_000).alias("peak_idx"))
    with pytest.raises(ValueError, match="same master_event_detector call"):
        wis.scope.rates.compute_rate(bad, hypno, final_peaks, by="state")


def test_datetime_hypno_rejected() -> None:
    final_peaks, peaks_df, _ = _standard()
    dt_hypno = Hypnogram(
        pl.DataFrame(
            {
                "state": ["NREM"],
                "start_time": ["2026-01-01 00:00:00"],
                "end_time": ["2026-01-01 00:10:00"],
            }
        )
    )
    with pytest.raises(ValueError, match="numeric"):
        wis.scope.rates.compute_rate(peaks_df, dt_hypno, final_peaks, by="state")


def test_add_valid_duration_agrees_with_by_bout_denominator() -> None:
    # Cross-check: the per-bout valid seconds used by compute_rate must match
    # the canonical add_valid_duration primitive.
    final_peaks, peaks_df, hypno = _standard()
    _, _, mask = _make_inputs(_grid(20.0), [(5.0, 10.0)], {"1-0": []})
    canonical = wis.validity.add_valid_duration(hypno, mask).df
    df = wis.scope.rates.compute_rate_by_bout(peaks_df, hypno, final_peaks)
    by_bout = (
        df.select("bout_idx", "valid_duration_s").unique().sort("bout_idx")
    )
    np.testing.assert_allclose(
        by_bout["valid_duration_s"].to_numpy(),
        canonical["valid_duration"].to_numpy(),
        atol=0.02,
    )
