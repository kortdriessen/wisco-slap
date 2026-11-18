"""Contracts for frozen anchors, quiet preservation and bounded bout merging."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from wisco_slap.peri._hypno_substates.wake_merge import (
    WakeMergeConfig,
    merge_wake_substates,
)


def context(active, slow=0.0, motion=4.0):
    n = len(active)
    return pl.DataFrame(
        dict(
            start_s=np.arange(n, dtype=float),
            end_s=np.arange(1, n + 1, dtype=float),
            time=np.arange(n) + 0.5,
            duration_s=np.ones(n),
            run_id=np.ones(n, dtype=int),
            original_state=["Wake"] * n,
            eeg_valid=np.ones(n, bool),
            fast_peak_fraction=np.full(n, 0.1),
            stage1_active=active,
            active_selected=active,
            stage1_label=np.where(active, "Clear Active wake", "Wake"),
            quiet_slow_index=np.broadcast_to(slow, n),
            whisk_z=np.broadcast_to(motion, n),
            pupil_motion_z=np.full(n, np.nan),
            whisk_valid_frac=np.ones(n),
            pupil_valid_frac=np.zeros(n),
        )
    )


def result(frame, **kwargs):
    return merge_wake_substates(
        frame, replace(WakeMergeConfig(smooth_seconds=1), **kwargs)
    )["epochs"]


def test_short_gap_bridges_but_unanchored_wake_stays_quiet():
    assert result(context([True, False, False, True]))["active_selected"].all()
    assert set(result(context([False] * 4))["stage2_label"]) == {"Quiet wake"}


def test_slow_and_still_core_survives_even_enormous_budgets():
    f = context([True, False, False, True], slow=1.0, motion=0.0)
    r = result(f, bridge_budget_s=1000, edge_budget_s=1000)
    assert r["protected_quiet"].to_list() == [False, True, True, False]
    assert r["active_selected"].to_list() == [True, False, False, True]
    assert r["stage1_label"].equals(f["stage1_label"])


def test_quiet_evidence_resists_merging_before_hard_protection():
    f = context([True, False, True], slow=1.0, motion=0.0)
    assert not result(f, bridge_budget_s=4.0, edge_budget_s=0.0)["active_selected"][1]
    assert result(
        context([True, False, True], slow=0.0), bridge_budget_s=4.0, edge_budget_s=0.0
    )["active_selected"][1]


def test_edge_growth_is_bounded_and_does_not_create_new_anchors():
    f = context([True] + [False] * 20)
    r = result(f, edge_budget_s=2.0)
    assert np.flatnonzero(r["active_selected"]).tolist() == [0, 1, 2]
    assert result(r, edge_budget_s=2.0)["active_selected"].equals(r["active_selected"])


def test_long_gap_preserves_middle_and_trims_both_ends():
    r = result(
        context([True] + [False] * 20 + [True]), edge_budget_s=2.0, bridge_budget_s=8.0
    )
    assert np.flatnonzero(r["active_selected"]).tolist() == [0, 1, 2, 19, 20, 21]


def test_sleep_boundaries_and_time_gaps_block_merging_even_with_bad_run_ids():
    f = context([True, False, False, False]).with_columns(
        pl.Series("original_state", ["Wake", "NREM", "Wake", "Wake"]),
        pl.Series("stage1_label", ["Clear Active wake", "NREM", "Wake", "Wake"]),
    )
    r = result(f)
    assert r["stage2_label"].to_list() == [
        "Active wake",
        "NREM",
        "Quiet wake",
        "Quiet wake",
    ]
    f = context([True, False, False]).with_columns(
        pl.Series("start_s", [0.0, 5.0, 6.0]), pl.Series("end_s", [1.0, 6.0, 7.0])
    )
    assert result(f)["active_selected"].to_list() == [True, False, False]


def test_invalid_eeg_blocks_growth_and_is_retained():
    f = context([True, False, False, True]).with_columns(
        pl.Series("eeg_valid", [True, False, True, True])
    )
    r = result(f, edge_budget_s=0.0)
    assert r["active_selected"].to_list() == [True, False, False, True]
    assert r.height == 4 and r["duration_s"].sum() == 4
    assert r["stage2_label"][1] == "Quiet wake" and not r["step2_eeg_valid"][1]


def test_missing_tracking_is_not_stillness_and_cannot_borrow_local_motion():
    f = context([True, False, False, True], slow=1.0, motion=0.0).with_columns(
        pl.lit(0.0).alias("whisk_valid_frac")
    )
    r = result(f)
    assert not r["protected_quiet"].any()
    assert not r["step2_motion_observed"].any()
    assert np.isnan(r["stillness_evidence"].to_numpy()).all()


def test_budgets_are_monotonic_and_quiet_dials_protect_more():
    rng = np.random.default_rng(24)
    a = rng.random(300) < 0.2
    a[:2] = True
    f = context(a, rng.uniform(0, 1.2, 300), rng.uniform(-1, 4, 300))
    base = result(f)["active_selected"].to_numpy()
    for params in [dict(bridge_budget_s=20), dict(edge_budget_s=5)]:
        assert np.all(~base | result(f, **params)["active_selected"].to_numpy())
    for params in [dict(quiet_slow_min=0.2), dict(still_motion_max=3.0)]:
        assert np.all(base | ~result(f, **params)["active_selected"].to_numpy())


def test_durations_drive_budget_and_all_states_are_retained():
    f = context([True, False, False, True]).with_columns(
        pl.Series("start_s", [0.0, 1.0, 1.2, 1.4]),
        pl.Series("end_s", [1.0, 1.2, 1.4, 2.4]),
        pl.Series("duration_s", [1.0, 0.2, 0.2, 1.0]),
    )
    assert result(f, bridge_budget_s=0.5, edge_budget_s=0.0)["active_selected"].all()
    with pytest.raises(ValueError):
        result(f, bridge_budget_s=-1)


def test_new_band_includes_four_to_five_hz_and_respects_eeg_gaps():
    import xarray as xr

    from wisco_slap.peri._hypno_substates.active_wake import (
        build_active_wake_context,
        select_active_wake,
    )
    from wisco_slap.peri._hypno_substates.wake_merge import build_wake_merge_context

    fs = 400.0
    t = np.arange(int(fs * 180)) / fs
    rng = np.random.default_rng(7)
    amplitude = np.where(t < 70, 30.0, 8.0)
    y = amplitude * np.sin(2 * np.pi * 4.7 * t) + 4 * rng.normal(size=len(t))
    y[(t >= 100) & (t < 104)] = np.nan
    eeg = xr.DataArray(y, dims=["time"], coords={"time": t}, attrs={"fs": fs})
    states = pl.DataFrame(
        dict(start_s=[0.0, 70.0], end_s=[70.0, 180.0], label=["NREM", "Wake"])
    )
    eye = pl.DataFrame(schema={"time": pl.Float64})
    whisk = pl.DataFrame(schema={"time": pl.Float64, "whis": pl.Float64})
    c, m, _ = build_active_wake_context(eeg, eye, whisk, states)
    s1 = select_active_wake(c)["epochs"]
    s2, meta = build_wake_merge_context(s1, eeg, m)
    assert s2["stage1_label"].equals(s1["stage1_label"])
    ref = s2.filter(pl.col("time").is_between(10, 60))
    assert ref["quiet_slow_power_0p5_5"].median() > 10 * ref["raw_delta_power"].median()
    assert meta["step2_slow_calibration"]["available"]
    gap = s2.filter(pl.col("time").is_between(100, 104))
    assert gap["quiet_slow_power_0p5_5"].is_nan().all()
