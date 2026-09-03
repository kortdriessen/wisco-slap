from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import polars as pl
import pytest
import tifffile as tiff
import xarray as xr

matplotlib.use("Agg")

import wisco_slap as wis  # noqa: E402
from wisco_slap.viz.overview_and_summary_plots.single_soma._single_soma_overview import (  # noqa: E402
    _bin_max_uniform,
    _decimate_keep_nan,
    _display_image,
    _footprint_classes,
    _footprint_cores,
    _soma_polygon,
    _state_fractions,
    _time_gaps,
    single_soma_overview_path,
    single_soma_overview_plot,
)

# Synthetic experiment: two 100 s sync blocks, one acquisition in each; the
# acquisition under test sits in block 1 and runs 10 s -> 70 s of block time.
_EPHYS_FS = 200.0
_BLOCK_S = 100.0
_FS = 50.0
_ACQ_T0, _ACQ_T1 = 10.0, 70.0
_BLOCKS = {
    1: {"ephys_start": "2026-01-01 10:00:00", "ephys_end": "2026-01-01 10:01:40", "corrupt": False},
    2: {"ephys_start": "2026-01-01 10:05:00", "ephys_end": "2026-01-01 10:06:40", "corrupt": False},
}
_ACQS = {
    "loc_A--acq_1": {"sync_block": 1, "dat_start": "2026-01-01 10:00:10", "ephys_offset": _ACQ_T0},
    "loc_B--acq_1": {"sync_block": 2, "dat_start": "2026-01-01 10:05:20", "ephys_offset": 20.0},
}
_DMD_INFO = {
    "dmd-1": {"depth": 100, "somas": ["soma1"]},
    "dmd-2": {"depth": 50, "somas": []},
}


def _ephys(subject, exp, sb, stores=("EEG_",)):
    t = np.arange(int(_BLOCK_S * _EPHYS_FS)) / _EPHYS_FS
    rng = np.random.default_rng(sb)
    out = {}
    for store in stores:
        da = xr.DataArray(rng.normal(0, 1, t.size), coords={"time": t}, dims=["time"])
        da.attrs["fs"] = _EPHYS_FS
        out[store] = da
    return out


def _whisk(subject, exp, sb):
    t = np.arange(0, _BLOCK_S, 0.02)
    return pl.DataFrame({"time": t, "whis": np.abs(np.sin(t))})


def _hypno(subject, exp, sb, version="model"):
    df = pl.DataFrame({
        "state": ["Wake", "NREM", "REM"],
        "start_time": [0.0, 30.0, 70.0],
        "end_time": [30.0, 70.0, 100.0],
    })
    return SimpleNamespace(df=df)


def _timing(subject, exp, loc, acq):
    entry = _ACQS[f"{loc}--{acq}"]
    start = float(entry["ephys_offset"])
    return SimpleNamespace(
        start_time=start, end_time=start + 60.0, duration=60.0, sync_block=entry["sync_block"]
    )


def _synthetic_data() -> dict:
    """A tiny but complete ``load_single_soma_data`` result for soma1."""
    rng = np.random.default_rng(0)
    t = _ACQ_T0 + np.arange(int((_ACQ_T1 - _ACQ_T0) * _FS)) / _FS
    n_syn = 6
    heat = xr.DataArray(
        np.abs(rng.normal(0, 1, (n_syn, t.size))),
        dims=("syn_id", "time"),
        coords={
            "syn_id": [f"1-{i}" for i in range(3)] + [f"2-{i}" for i in range(3)],
            "time": t,
            "dmd": ("syn_id", [1, 1, 1, 2, 2, 2]),
            "soma-ID": ("syn_id", ["soma1"] * n_syn),
            "dend-ID": ("syn_id", ["A-1", "A-1", "B-1", "C-1", "C-1", "C-1"]),
            "pos": ("syn_id", [1.0, 2.0, 1.0, 1.0, 2.0, 3.0]),
        },
    )
    heat[:, 100:150] = np.nan  # a short invalid gap
    soma_ca = xr.DataArray(
        rng.normal(0, 0.1, t.size), dims=("time",), coords={"time": t}
    )
    spikes = pl.DataFrame({
        "time": t[::200], "soma_id": ["soma1"] * len(t[::200]), "spike_prob": 0.5
    })
    h, w1, w2 = 20, 30, 25
    meanIM = {1: rng.random((2, h, w1)), 2: rng.random((2, h, w2))}
    syn_map = {1: np.full((h, w1), -1), 2: np.full((h, w2), -1)}
    fp_vals = {1: np.full((h, w1), np.nan), 2: np.full((h, w2), np.nan)}
    for dmd in (1, 2):
        for label in (1, 2, 3, 4):  # four 3x3 footprints per DMD, peak in the centre
            r, c = 5, 5 * label
            syn_map[dmd][r - 1 : r + 2, c - 1 : c + 2] = label
            fp_vals[dmd][r - 1 : r + 2, c - 1 : c + 2] = 0.02
            fp_vals[dmd][r, c] = 0.1
    # footprints 0-2 of each DMD are the heatmap synapses; footprint 3 is a
    # dendritic synapse of another soma (DMD-1) / a synapse without dend-ID (DMD-2)
    syn_labels = pl.DataFrame({
        "syn_id": [0, 1, 2, 3, 0, 1, 2, 3],
        "dmd": [1, 1, 1, 1, 2, 2, 2, 2],
        "soma-ID": ["soma1", "soma1", "soma1", "soma2", "soma1", "soma1", "soma1", "soma1"],
        "dend-ID": ["A-1", "A-1", "B-1", "Z-1", "C-1", "C-1", "C-1", None],
    })
    roi_polys = {
        1: [{"Label": "soma1", "Position": np.array([[10.0, 18.0, 18.0, 10.0], [8.0, 8.0, 14.0, 14.0]])}],
        2: [],
    }
    return dict(
        heat=heat, heat_label="test signal", soma_ca=soma_ca, spikes=spikes,
        meanIM=meanIM, syn_map=syn_map, fp_vals=fp_vals, roi_polys=roi_polys,
        syn_labels=syn_labels,
    )


def _stub_experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the plot at a synthetic two-block experiment rooted in tmp_path."""
    si = {"mouse": {"exp_1": {"sync_blocks": _BLOCKS, "acquisitions": _ACQS}}}
    monkeypatch.setattr(wis.meta.get, "sync_info", lambda: si)
    monkeypatch.setattr(
        wis.meta.get, "dmd_info", lambda: {"mouse": {"exp_1": {"loc_A": {"acq_1": _DMD_INFO}}}}
    )
    monkeypatch.setattr(wis.get, "ephys", _ephys)
    monkeypatch.setattr(wis.get, "whisk_df", _whisk)
    monkeypatch.setattr(wis.get, "sb_bout_hypno", _hypno)
    monkeypatch.setattr(wis.meta, "get_acq_timing", _timing)
    monkeypatch.setattr(wis.defs, "anmat_root", str(tmp_path))
    monkeypatch.setattr(wis.defs, "annotation_root", str(tmp_path / "annotation_materials"))


def _write_mask(tmp_path: Path) -> None:
    mask_dir = tmp_path / "annotation_materials" / "mouse" / "exp_1" / "loc_A" / "acq_1" / "roi_locations"
    mask_dir.mkdir(parents=True)
    mask = np.zeros((20, 30), dtype=np.uint32)
    mask[8:14, 10:18] = 1
    tiff.imwrite(mask_dir / "roi_locs_dmd1_mask.tif", mask)


def test_bin_max_uniform_pools_max_and_keeps_nan_only_bins() -> None:
    t = np.arange(10, dtype=float)
    values = np.array([[0, 5, 1, 1, np.nan, np.nan, 2, 3, 9, 9]], dtype=float)
    da = xr.DataArray(
        values, dims=("syn_id", "time"),
        coords={"syn_id": ["1-0"], "time": t, "dmd": ("syn_id", [1])},
    )
    out, edges = _bin_max_uniform(da, n_cols=5)  # bins of 1.8 s over [0, 9]
    assert out.sizes == {"syn_id": 1, "time": 5}
    np.testing.assert_allclose(edges, np.linspace(0, 9, 6))
    # bins: {0,1} {2,3} {4,5} {6,7} {8,9} -> the all-NaN bin stays NaN
    np.testing.assert_array_equal(out.values[0], [5, 1, np.nan, 3, 9])
    np.testing.assert_allclose(out.time.values, (edges[:-1] + edges[1:]) / 2)
    assert out["dmd"].values.tolist() == [1]
    # more columns than samples -> one bin per sample, nothing lost
    out, _ = _bin_max_uniform(da, n_cols=50)
    assert out.sizes["time"] == 10
    np.testing.assert_array_equal(out.values[0], values[0])


def test_bin_max_uniform_leaves_time_gaps_empty() -> None:
    # two epochs: 0-4 s and 10-14 s, nothing in between
    t = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14], dtype=float)
    da = xr.DataArray(
        np.ones((2, 10)), dims=("syn_id", "time"),
        coords={"syn_id": ["1-0", "1-1"], "time": t},
    )
    out, edges = _bin_max_uniform(da, n_cols=5)  # bins of 2.8 s over [0, 14]
    np.testing.assert_allclose(edges, np.linspace(0, 14, 6))
    # [5.6, 8.4) holds no sample -> NaN column; every other bin has data
    np.testing.assert_array_equal(out.values, [[1, 1, np.nan, 1, 1]] * 2)
    assert _time_gaps(t) == (1, 6.0)
    assert _time_gaps(np.arange(5.0)) == (0, 0.0)


def test_footprint_cores_keeps_peak_and_trims_weak_pixels() -> None:
    syn_map = np.array([[1, 1, 1, -1], [2, 2, -1, -1]])
    fp_vals = np.array([[0.1, 0.04, 0.06, np.nan], [0.02, 0.02, np.nan, np.nan]])
    core = _footprint_cores(syn_map, fp_vals, rel_cut=0.5)
    # synapse 1: peak 0.1 -> keeps 0.1 and 0.06, drops 0.04; synapse 2: flat -> all kept
    assert core.tolist() == [[True, False, True, False], [True, True, False, False]]
    np.testing.assert_array_equal(_footprint_cores(syn_map, None, 0.5), syn_map > 0)
    np.testing.assert_array_equal(_footprint_cores(syn_map, fp_vals, 0.0), syn_map > 0)


def test_decimate_keep_nan_preserves_gaps() -> None:
    t = np.arange(10, dtype=float)
    v = np.array([0, 1, 2, np.nan, 4, 5, 6, 7, 8, 9], dtype=float)
    td, vd = _decimate_keep_nan(t, v, 2)
    np.testing.assert_array_equal(td, [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(vd, [0, np.nan, 4, 6, 8])  # the gap survives
    td, vd = _decimate_keep_nan(t, v, 1)
    np.testing.assert_array_equal(td, t)
    np.testing.assert_array_equal(vd, v)
    # a jump in time (missing trial) breaks the line at the sample after it
    t_jump = np.array([0, 1, 2, 3, 40, 41, 42], dtype=float)
    _, vj = _decimate_keep_nan(t_jump, np.ones(7), 1)
    np.testing.assert_array_equal(vj, [1, 1, 1, 1, np.nan, 1, 1])


def test_footprint_classes_follow_heatmap_membership_and_dend_id() -> None:
    data = _synthetic_data()
    for dmd in (1, 2):
        classes = _footprint_classes(data["syn_map"][dmd], dmd, data["heat"], data["syn_labels"])
        labels = {k: sorted(np.unique(data["syn_map"][dmd][m]).tolist()) for k, m in classes.items()}
        assert labels["this soma"] == [1, 2, 3]
        if dmd == 1:
            assert labels["other dendritic"] == [4] and labels["no dend-ID"] == []
        else:
            assert labels["other dendritic"] == [] and labels["no dend-ID"] == [4]
    # without a label table nothing can be called 'no dend-ID'
    classes = _footprint_classes(data["syn_map"][2], 2, data["heat"], None)
    assert sorted(np.unique(data["syn_map"][2][classes["other dendritic"]]).tolist()) == [4]
    assert not classes["no dend-ID"].any()


def test_display_image_lifts_dim_structure_and_saturates_soma() -> None:
    img = np.full((50, 50), 1.0)
    img[10:40, 10:40] = 1.0 + np.random.default_rng(0).random((30, 30)) * 0.01  # noise bg
    img[20, 5:45] = 3.0  # a dim dendrite
    img[30:34, 30:34] = 60.0  # a very bright soma
    shown, black, white = _display_image(img)
    assert shown.min() == 0.0 and shown.max() == 1.0
    assert shown[30, 30] == 1.0  # soma saturates
    assert shown[20, 8] > 0.15  # dendrite still clearly above the background
    assert shown[5, 5] == 0.0  # background is black
    flat, b, w = _display_image(np.full((4, 4), 2.0))
    assert flat.max() == 0.0 and b == w == 2.0


def test_soma_polygon_lookup() -> None:
    polys = _synthetic_data()["roi_polys"]
    assert _soma_polygon(polys, 1, "soma1").shape == (2, 4)
    assert _soma_polygon(polys, 2, "soma1") is None
    assert _soma_polygon(polys, 1, "soma9") is None
    assert _soma_polygon(None, 1, "soma1") is None


def test_state_fractions_clips_to_window_and_reports_unscored() -> None:
    hypno = _hypno("m", "exp_1", 1)  # Wake 0-30, NREM 30-70, REM 70-100
    fr = _state_fractions(hypno, 20.0, 80.0)
    assert fr["Wake"] == pytest.approx(10 / 60)
    assert fr["NREM"] == pytest.approx(40 / 60)
    assert fr["REM"] == pytest.approx(10 / 60)
    assert fr["other"] == 0.0 and fr["unscored"] == pytest.approx(0.0)
    fr = _state_fractions(hypno, 90.0, 110.0)  # scoring stops at 100
    assert fr["REM"] == pytest.approx(0.5) and fr["unscored"] == pytest.approx(0.5)


def test_single_soma_overview_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(wis.defs, "annotation_root", str(tmp_path))
    assert single_soma_overview_path("m", "exp_1", "loc_A", "acq_1", "soma1") == (
        tmp_path / "OVERVIEW_FIGURES" / "single_soma" / "single_soma_overview"
        / "m--exp_1--loc_A--acq_1--soma1.png"
    )


def test_single_soma_overview_plot_builds_and_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    _stub_experiment(monkeypatch, tmp_path)
    data = _synthetic_data()
    out = single_soma_overview_path("mouse", "exp_1", "loc_A", "acq_1", "soma1")
    assert not out.exists()

    fig, axes = single_soma_overview_plot(
        "mouse", "exp_1", "loc_A", "acq_1", "soma1", data=data, dpi=20
    )
    assert set(axes) == {
        "blocks", "acquisitions", "EEG_", "EEG_ spectrogram", "whisking",
        "spikes", "soma", "heatmap", "dmd1", "dmd2", "stats",
    }
    assert out.exists()
    first_mtime = out.stat().st_mtime_ns
    # title carries the soma and the per-DMD synapse counts
    assert "soma1 (DMD-1): 6 dendritic synapses (3 on DMD-1, 3 on DMD-2)" in axes["blocks"].get_title()
    # the ROI polygon is drawn (as a line) on the soma's DMD only
    assert len(axes["dmd1"].lines) == 1 and not axes["dmd2"].lines
    assert "3 syn of soma1  |  1 other dendritic  |  0 no dend-ID" in axes["dmd1"].get_title()
    assert "3 syn of soma1  |  0 other dendritic  |  1 no dend-ID" in axes["dmd2"].get_title()
    assert "ExSum ROI polygon" in axes["stats"].get_legend().get_texts()[3].get_text()
    stats_text = axes["stats"].texts[0].get_text()
    assert "acquisition duration: 1.0 min (60 s)" in stats_text
    assert "Wake 33.3%, NREM 66.7%, REM 0.0%" in stats_text
    assert "DMD-1 98.3% (3 syn), DMD-2 98.3% (3 syn)" in stats_text
    report = capsys.readouterr().out
    assert "ROI mask not found" in report and "drew the ExSum ROI polygon" in report
    fig.clear()

    # Existing file + overwrite=False -> untouched; overwrite=True -> rewritten.
    fig, _ = single_soma_overview_plot(
        "mouse", "exp_1", "loc_A", "acq_1", "soma1", data=data, dpi=20
    )
    assert out.stat().st_mtime_ns == first_mtime
    fig.clear()
    fig, _ = single_soma_overview_plot(
        "mouse", "exp_1", "loc_A", "acq_1", "soma1", data=data, dpi=20, overwrite=True
    )
    assert out.stat().st_mtime_ns > first_mtime
    fig.clear()


def test_single_soma_overview_plot_prefers_mask_then_polygon_then_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    _stub_experiment(monkeypatch, tmp_path)
    _write_mask(tmp_path)
    data = _synthetic_data()  # has an ExSum polygon too: the mask must win
    fig, axes = single_soma_overview_plot(
        "mouse", "exp_1", "loc_A", "acq_1", "soma1", data=data, dpi=20
    )
    assert axes["dmd1"].collections and not axes["dmd1"].lines  # contour, no polygon
    assert "WISynaptic mask" in axes["stats"].get_legend().get_texts()[3].get_text()
    # the view stays pinned to the image (plus the small margin)
    assert axes["dmd1"].get_xlim() == pytest.approx((-0.5 - 0.6, 29.5 + 0.6))
    fig.clear()
    capsys.readouterr()

    (tmp_path / "annotation_materials" / "mouse" / "exp_1" / "loc_A" / "acq_1" / "roi_locations" / "roi_locs_dmd1_mask.tif").unlink()
    data["roi_polys"] = {1: [], 2: []}
    fig, axes = single_soma_overview_plot(
        "mouse", "exp_1", "loc_A", "acq_1", "soma1", data=data, dpi=20, overwrite=True
    )
    report = capsys.readouterr().out
    assert "no ExSum ROI polygon either — NOT drawn" in report
    assert "MISSING" in axes["stats"].get_legend().get_texts()[3].get_text()
    fig.clear()


def test_single_soma_overview_plot_rejects_unknown_soma_and_acq(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_experiment(monkeypatch, tmp_path)
    data = _synthetic_data()
    with pytest.raises(ValueError, match="soma9"):
        single_soma_overview_plot("mouse", "exp_1", "loc_A", "acq_1", "soma9", data=data)
    with pytest.raises(KeyError, match="loc_Z--acq_9"):
        single_soma_overview_plot("mouse", "exp_1", "loc_Z", "acq_9", "soma1", data=data)


def test_single_soma_overview_is_registered_in_the_single_soma_set() -> None:
    specs = wis.viz.overview_and_summary_plots.plot_specs("single_soma")
    assert [s.name for s in specs] == [
        "single_soma_corr_overview",
        "single_soma_overview",
    ]
    spec = next(s for s in specs if s.name == "single_soma_overview")
    assert spec.plot is single_soma_overview_plot
    assert spec.path is single_soma_overview_path
    assert wis.viz.single_soma_overview_plot is single_soma_overview_plot
