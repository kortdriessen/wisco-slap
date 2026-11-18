"""Scientific and boundary invariants of the final NREM pass."""

from dataclasses import replace

import numpy as np
import polars as pl
import xarray as xr

from wisco_slap.peri._hypno_substates.nrem_sigma import (
    NremSigmaConfig,
    build_nrem_sigma_context,
    label_nrem_substates,
)


def context(ratios, fractions=None, states=None, times=None):
    ratios = np.asarray(ratios, float)
    n = len(ratios)
    fractions = np.full(n, 0.35) if fractions is None else np.asarray(fractions, float)
    states = np.full(n, "NREM") if states is None else np.asarray(states)
    times = np.arange(n, dtype=float) if times is None else np.asarray(times, float)
    sigma = np.full(n, 10.0)
    return pl.DataFrame(
        dict(
            start_s=times,
            end_s=times + 1,
            time=times + 0.5,
            duration_s=np.ones(n),
            original_state=states,
            stage2_label=np.where(states == "Wake", "Quiet wake", states),
            eeg_valid=np.ones(n, bool),
            fast_peak_fraction=np.full(n, 0.1),
            nrem_slow_power=sigma / ratios,
            nrem_sigma_power=sigma,
            nrem_total_power=sigma / fractions,
            nrem_beta_power=np.full(n, 2.0),
        )
    )


CFG = NremSigmaConfig(smooth_seconds=1)


def test_seeds_then_supported_edges_and_default_sws():
    r = label_nrem_substates(context([0.2, 0.8, 1.2, 1.2, 0.8, 0.2]), CFG)["epochs"]
    assert r["sigma_seed"].to_list() == [False, False, True, True, False, False]
    assert r["sigma_selected"].to_list() == [False, True, True, True, True, False]
    assert r["final_label"].to_list()[0] == "NREM SWS"


def test_short_sigma_candidate_needs_persistent_anchor():
    r = label_nrem_substates(context([0.8, 0.8, 1.2, 0.8, 0.8]), CFG)
    assert not r["epochs"]["sigma_selected"].any()


def test_bridge_requires_support_and_cannot_erase_slow_barrier():
    for middle, expected in [(0.6, True), (0.4, False)]:
        r = label_nrem_substates(context([1.2, 1.2, middle, middle, 1.2, 1.2]), CFG)[
            "epochs"
        ]
        assert r["sigma_bridged"].to_list()[2:4] == [expected, expected]
    r = label_nrem_substates(
        context([1.2, 1.2, 0.6, 0.6, 1.2, 1.2]), replace(CFG, bridge_max_s=1)
    )
    assert not r["epochs"]["sigma_bridged"].any()


def test_non_nrem_frozen_and_no_state_boundary_crossing():
    c = context(
        [2, 2, 2, 2, 2, 2], states=["Wake", "Wake", "NREM", "REM", "NREM", "NREM"]
    )
    r = label_nrem_substates(c, CFG)["epochs"]
    assert r["final_label"].to_list() == [
        "Quiet wake",
        "Quiet wake",
        "NREM SWS",
        "REM",
        "NREM sigma",
        "NREM sigma",
    ]
    assert r["stage2_label"].equals(c["stage2_label"])


def test_time_gap_stops_persistence_and_real_gaps_split_bouts():
    r = label_nrem_substates(context([2, 2], times=[0, 10]), CFG)
    assert not r["epochs"]["sigma_selected"].any()
    assert r["bouts"].height == 2


def test_invalid_eeg_stops_growth_and_is_explicit_default():
    c = context([2, 2, 0.8, 0.8]).with_columns(
        pl.Series("eeg_valid", [True, True, False, True])
    )
    r = label_nrem_substates(c, CFG)["epochs"]
    assert r["sigma_selected"].to_list() == [True, True, False, False]
    assert "unavailable" in r["nrem_reason"][2]


def test_broadband_flattening_is_not_sigma_and_both_seed_thresholds_required():
    c = context([2, 2, 2, 2], fractions=[0.1, 0.1, 0.35, 0.35]).with_columns(
        pl.Series("nrem_beta_power", [2.0, 2.0, 20.0, 20.0])
    )
    assert not label_nrem_substates(c, CFG)["epochs"]["sigma_selected"].any()


def test_power_scaling_does_not_change_labels():
    c = context([0.2, 0.8, 1.2, 1.2, 0.8, 0.6, 0.6, 1.2, 1.2, 0.2])
    powers = [
        "nrem_slow_power",
        "nrem_sigma_power",
        "nrem_beta_power",
        "nrem_total_power",
    ]
    scaled = c.with_columns([pl.col(k) * 1e6 for k in powers])
    assert label_nrem_substates(c, CFG)["epochs"]["final_label"].equals(
        label_nrem_substates(scaled, CFG)["epochs"]["final_label"]
    )


def test_idempotence_and_monotonic_tuners():
    c = context([0.2, 0.8, 1.2, 1.2, 0.8, 0.6, 0.6, 1.2, 1.2, 0.2])
    base = label_nrem_substates(c, CFG)["epochs"]
    assert base["final_label"].equals(
        label_nrem_substates(base, CFG)["epochs"]["final_label"]
    )
    for changes in [
        dict(sigma_slow_min=0.8),
        dict(sigma_fraction_min=0.22),
        dict(support_fraction=0.55),
        dict(bridge_max_s=8.0),
    ]:
        more = label_nrem_substates(c, replace(CFG, **changes))["epochs"][
            "sigma_selected"
        ].to_numpy()
        assert np.all(more[base["sigma_selected"].to_numpy()])


def test_duration_is_elapsed_time_not_row_count():
    c = context([2, 2]).with_columns(
        pl.Series("start_s", [0.0, 0.25]),
        pl.Series("end_s", [0.25, 0.5]),
        pl.Series("duration_s", [0.25, 0.25]),
        pl.Series("time", [0.125, 0.375]),
    )
    assert not label_nrem_substates(c, CFG)["epochs"]["sigma_selected"].any()


def test_measurement_respects_states_and_missing_samples():
    fs = 200.0
    t = np.arange(0, 12, 1 / fs)
    y = np.where(t < 6, 30 * np.sin(2 * np.pi * 12 * t), 30 * np.sin(2 * np.pi * 2 * t))
    c = context(np.ones(12), states=["Wake"] * 6 + ["NREM"] * 6)
    eeg = xr.DataArray(y, dims=["time"], coords={"time": t}, attrs={"fs": fs})
    measured, _ = build_nrem_sigma_context(c, eeg, {})
    assert measured["nrem_sigma_power"][6] < 0.001 * measured["nrem_slow_power"][6]
    y[(t >= 8) & (t < 10)] = np.nan
    measured, _ = build_nrem_sigma_context(c, eeg, {})
    assert np.isnan(measured["nrem_sigma_power"][8])
    assert np.isnan(measured["nrem_sigma_power"][9])
    assert measured["stage2_label"].equals(c["stage2_label"])


def test_no_nrem_and_invalid_nrem_have_complete_hypnograms():
    c = context([2, 2], states=["Wake", "REM"])
    assert label_nrem_substates(c, CFG)["epochs"]["final_label"].to_list() == [
        "Quiet wake",
        "REM",
    ]
    c = context([2, 2]).with_columns(pl.lit(False).alias("eeg_valid"))
    r = label_nrem_substates(c, CFG)
    assert r["epochs"]["final_label"].to_list() == ["NREM SWS", "NREM SWS"]
    assert r["bouts"].height == 1
