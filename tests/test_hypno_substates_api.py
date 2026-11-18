"""Production loader contracts and full three-stage scientific integration."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

import wisco_slap as wis
from electro_py.hypno import Hypnogram
from wisco_slap.get import _get_hypnos as gh
from wisco_slap.get import _get_sync_block_dat as gd
from wisco_slap.peri import anno


def eeg(duration=200.0, fs=200.0):
    t = np.arange(0, duration, 1 / fs)
    y = (
        3 * np.sin(2 * np.pi * 2 * t)
        + 4 * np.sin(2 * np.pi * 12 * t)
        + np.sin(2 * np.pi * 40 * t)
    )
    return xr.DataArray(y, dims=["time"], coords={"time": t}, attrs={"fs": fs})


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(wis.defs, "anmat_root", str(tmp_path))
    monkeypatch.setattr(gh, "anmat_root", str(tmp_path))
    root = (
        tmp_path
        / "mouse"
        / "exp_1"
        / "sync_block_data"
        / "sync_block-1"
        / "hypnograms"
        / "model_scored"
    )
    root.mkdir(parents=True)
    original = pl.DataFrame(
        dict(
            start_s=[0.0, 80.0, 160.0],
            end_s=[80.0, 160.0, 200.0],
            label=["Wake", "NREM", "REM"],
        )
    )
    original.write_csv(root / "bout_hypno.csv")
    pl.DataFrame(
        dict(
            start_s=np.arange(200, dtype=float),
            end_s=np.arange(1, 201, dtype=float),
            pred_label=["Wake"] * 200,
            P_Wake=np.full(200, 0.8),
            P_NREM=np.full(200, 0.1),
            P_REM=np.full(200, 0.1),
        )
    ).write_parquet(root / "epoch_df.parquet")
    signal = eeg()

    def ephys(subject, exp, sb, stores):
        assert stores == ("EEG_",)  # No load-cell dependency.
        return {"EEG_": signal}

    monkeypatch.setattr(gd, "ephys", ephys)

    def absent(*args):
        raise FileNotFoundError("optional video unavailable")

    monkeypatch.setattr(wis.peri.vid, "load_eye_metric_df", absent)
    monkeypatch.setattr(wis.peri.vid, "load_whisking_df", absent)
    monkeypatch.setattr(
        gh,
        "_get_sync_info",
        lambda: {
            "mouse": {"exp_1": {"acquisitions": {"loc_A--acq_1": {"sync_block": 1}}}}
        },
    )
    return root, original, signal


def test_approved_defaults_are_not_old_library_defaults():
    c = asdict(anno.ActiveWakeConfig())
    assert c["slow_activity_max"] == 0.15 and c["assisted_seed_fold"] == 2.0
    assert anno.WakeMergeConfig().bridge_budget_s == 8.0
    assert anno.NremSigmaConfig().sigma_fraction_min == 0.28


def test_pure_function_without_camera_and_all_parameters(source):
    _, original, signal = source
    r = anno.label_hypnogram_substates(
        original,
        signal,
        step1_config={"eeg_seed_fold": 2.2},
        step2_config={"edge_budget_s": 0},
        step3_config={"sigma_slow_min": 1.2},
        return_details=True,
    )
    assert isinstance(r["hypnogram"], Hypnogram)
    assert r["metadata"]["step1_config"]["eeg_seed_fold"] == 2.2
    assert r["metadata"]["step1_config"]["slow_activity_max"] == 0.15
    assert r["metadata"]["step2_config"]["edge_budget_s"] == 0
    assert r["metadata"]["step3_config"]["sigma_slow_min"] == 1.2
    assert r["metadata"]["missing_eye"] and r["metadata"]["missing_whisk"]
    assert not r["epochs"]["protected_quiet"].any()
    assert r["epochs"].filter(pl.col("original_state") == "REM")[
        "final_label"
    ].unique().to_list() == ["REM"]
    assert set(r["hypnogram"].states) <= set([
        "Active wake",
        "Quiet wake",
        "NREM SWS",
        "NREM sigma",
        "REM",
    ])


def test_all_public_loaders_default_to_same_substates(source):
    r = anno.load_auto_hypno("mouse", "exp_1", 1, return_details=True)
    assert isinstance(r["table"], pd.DataFrame)
    assert r["table"].columns.tolist() == [
        "start_time",
        "end_time",
        "state",
        "duration",
    ]
    assert set(r["table"]["state"]).isdisjoint({"Wake", "NREM"})
    assert wis.peri.load_auto_hypno is anno.load_auto_hypno
    for fn, args in [
        (wis.get.sb_bout_hypno, ("mouse", "exp_1", 1)),
        (wis.get.bout_hypno, ("mouse", "exp_1", "loc_A", "acq_1")),
    ]:
        h = fn(*args)
        assert h.df.equals(r["hypnogram"].df)
        raw = fn(*args, reformat=False)
        assert raw.columns == ["start_s", "end_s", "label"]
        assert raw["label"].to_list() == r["table"]["state"].tolist()
    tuned = wis.get.sb_bout_hypno(
        "mouse", "exp_1", 1, step3_config={"sigma_slow_min": 3.0}
    )
    assert (
        tuned.df.filter(pl.col("state") == "NREM sigma")["duration"].sum()
        <= r["hypnogram"].df.filter(pl.col("state") == "NREM sigma")["duration"].sum()
    )


def test_broad_opt_out_never_reads_peripherals_and_uses_bouts_not_argmax(
    source, monkeypatch
):
    _, original, _ = source

    def forbidden(*args, **kwargs):
        raise AssertionError("broad path must not read EEG/video")

    monkeypatch.setattr(gd, "ephys", forbidden)
    for fn, args in [
        (anno.load_auto_hypno, ("mouse", "exp_1", 1)),
        (wis.get.sb_bout_hypno, ("mouse", "exp_1", 1)),
    ]:
        h = fn(*args, apply_substates=False)
        states = (
            h["state"].tolist()
            if isinstance(h, pd.DataFrame)
            else h.df["state"].to_list()
        )
        assert states == original["label"].to_list()


def test_validated_is_unchanged_even_with_default_substates(source, monkeypatch):
    root, original, _ = source
    original.write_csv(root.parent / "bout_hypno_VALIDATED.csv")

    def forbidden(*args, **kwargs):
        raise AssertionError("validated must not read EEG")

    monkeypatch.setattr(gd, "ephys", forbidden)
    h = wis.get.sb_bout_hypno("mouse", "exp_1", 1, version="validated", reformat=False)
    assert h.equals(original)


def test_missing_required_eeg_is_not_silently_broad(source, monkeypatch):
    def absent(*args, **kwargs):
        raise FileNotFoundError("EEG_ missing")

    monkeypatch.setattr(gd, "ephys", absent)
    with pytest.raises(FileNotFoundError, match="EEG_ missing"):
        wis.get.sb_bout_hypno("mouse", "exp_1", 1)
    assert (
        "NREM"
        in wis.get.sb_bout_hypno("mouse", "exp_1", 1, apply_substates=False).states
    )


def test_legacy_layout_confidence_and_gaps(tmp_path, monkeypatch):
    monkeypatch.setattr(wis.defs, "anmat_root", str(tmp_path))
    root = (
        tmp_path
        / "old"
        / "exp_1"
        / "scoring_data"
        / "sync_block-1"
        / "hypnograms"
        / "model_labelled"
    )
    root.mkdir(parents=True)
    pl.DataFrame(
        dict(
            start_s=[0.0, 1.0, 4.0],
            end_s=[1.0, 2.0, 5.0],
            label=["Wake"] * 3,
            NREM_smooth=[0.1, 0.1, 0.1],
            REM_smooth=[0.1, 0.1, 0.1],
            Wake_smooth=[0.9, 0.2, 0.9],
        )
    ).write_csv(root / "raw_epochs.csv")
    raw = anno.load_auto_hypno("old", "exp_1", 1, apply_substates=False, rename=False)
    assert raw["start_s"].tolist() == [0.0, 4.0] and raw["end_s"].tolist() == [2.0, 5.0]
    filtered = anno.load_auto_hypno(
        "old", "exp_1", 1, apply_substates=False, filter_unclear=0.7, rename=False
    )
    assert filtered["label"].tolist() == ["Wake", "unclear", "Wake"]


def test_modern_confidence_uses_raw_scores_without_replacing_hsmm(source):
    root, _, _ = source
    ep = pl.read_parquet(root / "epoch_df.parquet").with_columns(
        pl
        .when(pl.col("start_s") == 90)
        .then(0.2)
        .otherwise(pl.col("P_Wake"))
        .alias("P_Wake")
    )
    ep.write_parquet(root / "epoch_df.parquet")
    h = anno.load_auto_hypno(
        "mouse", "exp_1", 1, apply_substates=False, filter_unclear=0.7, filter_on="raw"
    )
    assert h["state"].tolist() == ["Wake", "NREM", "unclear", "NREM", "REM"]
    with pytest.raises(ValueError, match="filter_on='raw'"):
        anno.load_auto_hypno("mouse", "exp_1", 1, filter_unclear=0.7)


def test_off_grid_boundaries_and_annotation_gaps_are_preserved():
    original = pl.DataFrame(
        dict(
            start_s=[0.2, 7.7, 12.8],
            end_s=[7.7, 11.2, 20.0],
            label=["Wake", "REM", "NREM"],
        )
    )
    r = anno.label_hypnogram_substates(original, eeg(20), return_details=True)
    ep = r["epochs"]
    assert ep.filter(pl.col("original_state") == "REM")[
        "duration_s"
    ].sum() == pytest.approx(3.5)
    gap = ep.filter((pl.col("start_s") >= 11.2) & (pl.col("end_s") <= 12.8))
    assert gap["duration_s"].sum() == pytest.approx(1.6) and gap[
        "final_label"
    ].unique().to_list() == ["Unscored"]


def test_repeated_subsetting_is_rejected(source):
    _, original, signal = source
    h = anno.label_hypnogram_substates(original, signal)
    with pytest.raises(ValueError, match="already subsetted"):
        anno.label_hypnogram_substates(h, signal)


def test_coverage_aggregates_substate_families(monkeypatch):
    h = Hypnogram(
        pl.DataFrame(
            dict(
                start_time=[0.0, 10.0, 20.0, 30.0, 40.0],
                end_time=[10.0, 20.0, 30.0, 40.0, 50.0],
                state=["Active wake", "Quiet wake", "NREM SWS", "NREM sigma", "REM"],
            )
        )
    )
    monkeypatch.setattr(gh, "bout_hypno", lambda *a, **k: h)
    monkeypatch.setattr(gh, "get_acq_timing", lambda *a: (0.0, 50.0, 50.0, 1))
    _, _, fw, fn, fr = gh.acq_sleep_coverage("m", "e", "l", "a")
    assert (fw, fn, fr) == pytest.approx((0.4, 0.4, 0.2))


def test_nrem_normalization_and_swa_include_both_substates():
    h = Hypnogram(
        pl.DataFrame(
            dict(
                start_time=[0.0, 50.0, 100.0, 150.0],
                end_time=[50.0, 100.0, 150.0, 200.0],
                state=["NREM SWS", "NREM sigma", "Quiet wake", "Active wake"],
            )
        )
    )
    t = np.arange(200, dtype=float)
    eye = pl.DataFrame(
        dict(time=t, diameter=np.ones(200), motion=np.zeros(200), lid_norm=np.ones(200))
    )
    whisk = pl.DataFrame(dict(time=t, whis=np.zeros(200)))
    enriched = anno.add_peripheral_context_to_hypno(h, eeg(), eye, whisk)
    assert enriched.df["delta_norm"].is_finite().all()
    sw = enriched.df["swa_cum"].to_numpy()
    assert sw[2] > sw[1] > sw[0]  # Both NREM substates contributed before Wake.
    assert sw[3] == pytest.approx(sw[2])


def test_overview_summary_recognizes_substate_families():
    from wisco_slap.viz.overview_and_summary_plots.single_soma._single_soma_overview import (
        _state_fractions,
    )

    h = Hypnogram(
        pl.DataFrame(
            dict(
                start_time=[0.0, 10.0, 20.0, 30.0, 40.0],
                end_time=[10.0, 20.0, 30.0, 40.0, 50.0],
                state=["Active wake", "Quiet wake", "NREM SWS", "NREM sigma", "REM"],
            )
        )
    )
    fr = _state_fractions(h, 0.0, 50.0)
    assert (fr["Wake"], fr["NREM"], fr["REM"], fr["other"]) == pytest.approx((
        0.4,
        0.4,
        0.2,
        0.0,
    ))
