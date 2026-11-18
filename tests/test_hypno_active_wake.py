"""Behavioral contracts for a scientific selection mask, independent of labels."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
import xarray as xr

from wisco_slap.peri._hypno_substates.active_wake import (
    ActiveWakeConfig,
    build_active_wake_context,
    select_active_wake,
)


def context(n=24):
    frame = pl.DataFrame(
        dict(
            start_s=np.arange(n, dtype=float),
            end_s=np.arange(1, n + 1, dtype=float),
            time=np.arange(n) + 0.5,
            duration_s=np.ones(n),
            run_id=np.ones(n, dtype=int),
            original_state=["Wake"] * n,
            eeg_valid=np.ones(n, bool),
            fast_peak_fraction=np.full(n, 0.15),
            whisk_valid_frac=np.zeros(n),
            pupil_valid_frac=np.zeros(n),
        )
    )
    return frame.with_columns(
        *[
            pl.lit(0.0).alias(c)
            for c in [
                "tilt_log2",
                "flatness_log2",
                "gamma_35_45_log2",
                "gamma_45_55_log2",
                "high_gamma_65_85_log2",
                "high_gamma_85_110_log2",
                "muscle_130_150_log2",
                "muscle_150_170_log2",
            ]
        ],
        pl.lit(float("nan")).alias("whisk_z"),
        pl.lit(float("nan")).alias("pupil_motion_z"),
    )


def selected(frame, **kwargs):
    return select_active_wake(
        frame,
        replace(
            ActiveWakeConfig(
                smooth_seconds=1, assisted_seed_fold=1.3, slow_activity_max=0.25
            ),
            **kwargs,
        ),
    )["epochs"]["active_selected"].to_numpy()


@pytest.mark.parametrize(
    "columns",
    [
        ["tilt_log2", "flatness_log2"],
        ["gamma_35_45_log2", "gamma_45_55_log2"],
        ["high_gamma_65_85_log2", "high_gamma_85_110_log2"],
        ["muscle_130_150_log2", "muscle_150_170_log2"],
    ],
)
def test_each_eeg_route_can_seed_without_video(columns):
    frame = context().with_columns(*[pl.lit(3.5).alias(c) for c in columns])
    assert selected(frame).all()
    # A narrow peak in one half of a band is insufficient.
    assert not selected(frame.with_columns(pl.lit(0.0).alias(columns[0]))).any()


def test_motion_requires_positive_eeg_and_valid_tracking():
    f = context().with_columns(
        pl.lit(4.0).alias("whisk_z"), pl.lit(1.0).alias("whisk_valid_frac")
    )
    assert not selected(f).any()
    f = f.with_columns(
        pl.lit(0.5).alias("tilt_log2"), pl.lit(0.5).alias("flatness_log2")
    )
    assert selected(f).all()
    assert not selected(f.with_columns(pl.lit(0.0).alias("whisk_valid_frac"))).any()


def test_growth_never_bridges_unsupported_gap_or_state_boundary():
    strength = np.r_[0.0, 0.0, 2.0, 2.0, 0.95, 0.0, 0.95, 0.95, 0.0, 0.0]
    f = context(10).with_columns(
        pl.Series("tilt_log2", strength), pl.Series("flatness_log2", strength)
    )
    assert np.flatnonzero(selected(f)).tolist() == [2, 3, 4]
    # A state change splits runs even when the neighboring EEG is supportive.
    f = f.with_columns(
        pl.Series("run_id", [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
        pl.Series("original_state", ["Wake"] * 4 + ["NREM"] * 6),
    )
    assert np.flatnonzero(selected(f)).tolist() == [2, 3]


def test_isolated_spike_and_short_seed_are_rejected():
    f = context().with_columns(
        pl.lit(5.0).alias("tilt_log2"),
        pl.lit(5.0).alias("flatness_log2"),
        pl.lit(0.9).alias("fast_peak_fraction"),
    )
    assert not selected(f).any()
    values = np.zeros(24)
    values[5] = 5
    f = context().with_columns(
        pl.Series("tilt_log2", values), pl.Series("flatness_log2", values)
    )
    assert not selected(f).any()


def test_legacy_raw_columns_do_not_override_precomputed_evidence():
    f = context().with_columns(
        pl.lit(4.0).alias("tilt_log2"), pl.lit(4.0).alias("flatness_log2")
    )
    g = f.with_columns(
        pl.lit(1e20).alias("delta_power"),
        pl.lit(0.0).alias("eeg_activation"),
        pl.lit(1e30).alias("load_cell"),
    )
    np.testing.assert_array_equal(selected(f), selected(g))


def test_low_slow_seeds_without_gamma_enrichment_or_motion():
    f = context().with_columns(
        pl.lit(0.1).alias("slow_activity_index"),
        pl.lit(0.2).alias("raw_delta_fraction"),
    )
    result = select_active_wake(f)["epochs"]
    assert result["active_selected"].all()
    assert set(result["selection_reason"]) == {"Low-slow seed"}
    assert not selected(f.with_columns(pl.lit(0.8).alias("raw_delta_fraction"))).any()
    assert not selected(
        f.with_columns(pl.lit(float("nan")).alias("slow_activity_index"))
    ).any()


def test_high_delta_cannot_veto_existing_hf_seeds():
    f = context().with_columns(
        pl.lit(2.0).alias("muscle_130_150_log2"),
        pl.lit(2.0).alias("muscle_150_170_log2"),
        pl.lit(30.0).alias("slow_activity_index"),
        pl.lit(0.99).alias("raw_delta_fraction"),
    )
    assert selected(f).all()


def test_low_slow_dial_and_missing_local_evidence():
    strength = np.r_[1.0, 0.1, 0.1, 0.3, 0.5, 1.0, 0.1, 0.1, 1.0]
    f = context(9).with_columns(
        pl.Series("slow_activity_index", strength),
        pl.lit(0.2).alias("raw_delta_fraction"),
    )
    base = selected(f)
    assert np.flatnonzero(base).tolist() == [1, 2, 3, 6, 7]
    assert np.all(~base | selected(f, slow_activity_max=0.4))
    assert not selected(f, slow_activity_max=0.0).any()
    broken = strength.copy()
    broken[3] = np.nan
    f = f.with_columns(pl.Series("slow_activity_index", broken))
    assert not select_active_wake(f)["epochs"]["active_selected"][3]


def test_unselected_and_invalid_rows_keep_original_labels():
    f = context(8).with_columns(
        pl.Series("original_state", ["Wake"] * 4 + ["NREM", "NREM", "REM", "Unscored"]),
        pl.Series("run_id", [1, 1, 1, 1, 2, 2, 3, 4]),
        pl.Series("eeg_valid", [True, True, True, False, True, False, True, True]),
    )
    result = select_active_wake(f)["epochs"]
    assert result["stage1_label"].to_list() == f["original_state"].to_list()
    assert result["duration_s"].sum() == 8


def test_dials_are_monotonic_and_expansion_cannot_create_seeds():
    rng = np.random.default_rng(24)
    vals = rng.uniform(0.5, 4, 200)
    f = context(200).with_columns(
        pl.Series("tilt_log2", vals),
        pl.Series("flatness_log2", vals),
        pl.lit(4.0).alias("whisk_z"),
        pl.lit(1.0).alias("whisk_valid_frac"),
    )
    base = selected(f)
    for changes in [
        dict(eeg_seed_fold=1.5),
        dict(assisted_seed_fold=1.1),
        dict(edge_relax_fraction=0.4),
    ]:
        assert np.all(~base | selected(f, **changes))
    result = select_active_wake(f, ActiveWakeConfig(edge_relax_fraction=0.4))["epochs"]
    old = select_active_wake(f, ActiveWakeConfig())["epochs"]
    assert result["active_seed"].equals(old["active_seed"])


def test_raw_slow_wave_rejection_and_missing_reference_preserves_rows():
    fs = 400.0
    t = np.arange(int(fs * 100)) / fs
    rng = np.random.default_rng(42)
    y = 10 * rng.normal(size=len(t)) + 20 * np.sin(2 * np.pi * 11 * t)
    y[t >= 70] += 25 * rng.normal(size=(t >= 70).sum())
    eeg = xr.DataArray(y, dims=["time"], coords={"time": t}, attrs={"fs": fs})
    states = pl.DataFrame(
        dict(start_s=[0.0, 70.0], end_s=[70.0, 100.0], label=["NREM", "Wake"])
    )
    eye = pl.DataFrame(schema={"time": pl.Float64})
    whisk = pl.DataFrame(schema={"time": pl.Float64, "whis": pl.Float64})
    a, meta, _ = build_active_wake_context(eeg, eye, whisk, states)
    b, _, _ = build_active_wake_context(
        eeg + 1000 * np.sin(2 * np.pi * 1.3 * t), eye, whisk, states
    )
    assert meta["reference_available"] and meta["delta_used"]
    # Ignore the high-pass filter's recording-edge transients.
    for col in ["fast_tilt", "fast_flatness", "muscle_65_110", "muscle_130_170"]:
        np.testing.assert_allclose(
            a[col].to_numpy()[5:-5], b[col].to_numpy()[5:-5], rtol=1e-5, atol=1e-6
        )
    # Slow waves may suppress the new route, never the pre-existing routes.
    np.testing.assert_array_equal(
        selected(a.with_columns(pl.lit(float("nan")).alias("slow_activity_index")))[
            5:-5
        ],
        selected(b.with_columns(pl.lit(float("nan")).alias("slow_activity_index")))[
            5:-5
        ],
    )
    no_ref = states.with_columns(pl.lit("Wake").alias("label"))
    c, meta, _ = build_active_wake_context(eeg, eye, whisk, no_ref)
    result = select_active_wake(c)["epochs"]
    assert not meta["reference_available"] and result.height == a.height
    assert not result["active_selected"].any() and result["eeg_valid"].all()


def test_amplitude_scaling_alone_does_not_look_like_relative_activation():
    fs = 400.0
    t = np.arange(int(fs * 200)) / fs
    rng = np.random.default_rng(84)
    template = rng.normal(size=int(fs * 4)) + 4 * np.sin(
        2 * np.pi * 2 * t[: int(fs * 4)]
    )
    y = np.tile(template, 50)
    y[t >= 70] *= 0.01  # Same spectrum at much lower amplitude.
    eeg = xr.DataArray(y, dims=["time"], coords={"time": t}, attrs={"fs": fs})
    states = pl.DataFrame(
        dict(start_s=[0.0, 70.0], end_s=[70.0, 200.0], label=["NREM", "Wake"])
    )
    eye = pl.DataFrame(schema={"time": pl.Float64})
    whisk = pl.DataFrame(schema={"time": pl.Float64, "whis": pl.Float64})
    frame, meta, _ = build_active_wake_context(eeg, eye, whisk, states)
    assert meta["slow_calibration"]["available"]
    result = select_active_wake(frame)["epochs"].filter(
        pl.col("time").is_between(75, 95)
    )
    assert (result["delta_suppression_z"] > 2).all()
    assert not result["low_delta_candidate"].any()


def test_whole_recording_gain_invariance_and_reference_quality():
    fs = 400.0
    t = np.arange(int(fs * 180)) / fs
    rng = np.random.default_rng(482)
    slow_amp = np.where(t < 70, 25.0, 4.0)
    y = slow_amp * np.sin(2 * np.pi * 2 * t) + 8 * rng.normal(size=len(t))
    eeg = xr.DataArray(y, dims=["time"], coords={"time": t}, attrs={"fs": fs})
    states = pl.DataFrame(
        dict(start_s=[0.0, 70.0], end_s=[70.0, 180.0], label=["NREM", "Wake"])
    )
    eye = pl.DataFrame(schema={"time": pl.Float64})
    whisk = pl.DataFrame(schema={"time": pl.Float64, "whis": pl.Float64})
    a, ma, _ = build_active_wake_context(eeg, eye, whisk, states)
    b, mb, _ = build_active_wake_context(7 * eeg, eye, whisk, states)
    assert ma["slow_calibration"]["available"]
    np.testing.assert_allclose(
        a["slow_activity_index"], b["slow_activity_index"], atol=1e-10
    )
    np.testing.assert_array_equal(selected(a), selected(b))
    assert selected(a)[80:170].all()  # Quiet, low-slow EEG, no camera evidence.
    # A recording without a separated slow/activated range must not force a
    # fixed fraction of Wake active just because a Wake quantile exists.
    from wisco_slap.peri._hypno_substates.active_wake import (
        calibrate_active_wake_context,
    )

    flat = a.with_columns(pl.lit(100.0).alias("raw_delta_power"))
    flat, meta = calibrate_active_wake_context(flat, ma)
    assert not meta["slow_calibration"]["available"]
    assert not select_active_wake(flat)["epochs"]["low_delta_candidate"].any()


def test_eeg_enrichment_is_independent_of_nrem_dispersion():
    from wisco_slap.peri._hypno_substates.active_wake import (
        calibrate_active_wake_context,
    )

    n = 120
    f = context(n).with_columns(
        pl.Series("original_state", ["NREM"] * 60 + ["Wake"] * 60),
        pl.Series("run_id", [0] * 60 + [1] * 60),
        pl.Series("reference_epoch", [True] * 60 + [False] * 60),
        pl.Series("raw_delta_power", [400.0] * 60 + [40.0] * 60),
        pl.lit(1000.0).alias("raw_total_0p5_55_power"),
    )
    refs = {}
    a = []
    b = []
    for name, old in [("fast_tilt", "tilt_z"), ("flatness_logit", "flatness_z")] + [
        (k, k + "_z")
        for k in [
            "gamma_35_45",
            "gamma_45_55",
            "high_gamma_65_85",
            "high_gamma_85_110",
            "muscle_130_150",
            "muscle_150_170",
        ]
    ]:
        refs[name] = dict(center=0.0, scale=0.2, n=60)
        a.append(pl.lit(2.0).alias(old))
        b.append(pl.lit(0.2).alias(old))
    ca, _ = calibrate_active_wake_context(f.with_columns(a), dict(references=refs))
    refs2 = {k: {**v, "scale": 2.0} for k, v in refs.items()}
    cb, _ = calibrate_active_wake_context(f.with_columns(b), dict(references=refs2))
    np.testing.assert_allclose(ca["tilt_log2"], cb["tilt_log2"])
    np.testing.assert_array_equal(selected(ca), selected(cb))
