"""Tests for the single-soma correlation overview module (registration, figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xarray as xr  # noqa: E402

import wisco_slap as wis  # noqa: E402
from electro_py.hypno.hypno import Hypnogram  # noqa: E402
from wisco_slap.viz.overview_and_summary_plots import plot_specs  # noqa: E402
from wisco_slap.viz.overview_and_summary_plots.single_soma._single_soma_corr_overview import (  # noqa: E402, E501
    PLOT_SPEC,
    SIGNALS,
    compute_single_soma_corr_differences,
    compute_single_soma_corr_matrices,
    corr_difference_figure,
    corr_overview_figure,
    single_soma_corr_overview_path,
)


def test_path_and_registration(monkeypatch) -> None:
    monkeypatch.setattr(wis.defs, "annotation_root", "/tmp/ann_root")
    p = single_soma_corr_overview_path("alnair", "exp_1", "loc_D", "acq_1", "soma1")
    assert p == Path(
        "/tmp/ann_root/OVERVIEW_FIGURES/single_soma/single_soma_corr_overview/"
        "alnair--exp_1--loc_D--acq_1--soma1.png"
    )
    names = {s.name for s in plot_specs("single_soma")}
    assert PLOT_SPEC.name in names
    assert PLOT_SPEC.path is single_soma_corr_overview_path
    assert SIGNALS == ("mf", "ls", "denoised")


def _fake_data():
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 200.0, 0.01)
    X = rng.normal(size=(6, t.size)).astype(np.float32)
    X[:, 5000:5020] = np.nan
    coords = {
        "syn_id": [
            f"{d}-{i}" for d, i in zip([1, 1, 1, 2, 2, 2], range(6), strict=True)
        ],
        "time": t,
        "dmd": ("syn_id", np.array([1, 1, 1, 2, 2, 2])),
        "dend-ID": (
            "syn_id",
            np.array(["A-1", "A-1", "B-1", "C-1", "C-1", "C-1"], dtype=object),
        ),
        "pos": ("syn_id", np.array([0.0, 1.0, 0.0, 0.0, 1.0, 2.0])),
        "dmd-depth": ("syn_id", np.array([235, 235, 235, 160, 160, 160])),
        "soma-ID": ("syn_id", np.array(["soma1"] * 6, dtype=object)),
        "soma-depth": ("syn_id", np.array([235.0] * 6)),
    }
    base = xr.DataArray(X, dims=("syn_id", "time"), coords=coords)
    signals = {"mf": base * 2, "ls": base, "denoised": base.copy(data=np.abs(X))}
    hypno = Hypnogram(
        pl.DataFrame({
            "state": ["Wake", "NREM", "REM", "Wake"],
            "start_time": [0.0, 60.0, 140.0, 170.0],
            "end_time": [60.0, 140.0, 170.0, 200.0],
        })
    )
    return {"signals": signals, "hypno": hypno, "acq_duration": 200.0}


def test_compute_and_figures() -> None:
    data = _fake_data()
    mats, tiles = compute_single_soma_corr_matrices(data, epoch_length=10.0)
    assert tiles.columns == ["All", "Wake", "NREM", "REM"]
    # default keeps every tile: REM (30 s) gives 3, the other columns more
    assert tiles.equalised is False
    assert tiles.n_tiles["REM"] == 3
    assert tiles.n_tiles["Wake"] >= 8 and tiles.n_tiles["NREM"] >= 7
    assert list(mats) == ["mf", "ls", "denoised"]
    # mf is ls times a constant -> identical Pearson matrices
    for col in tiles.columns:
        assert np.allclose(
            mats["mf"].r[col].values, mats["ls"].r[col].values, atol=1e-6
        )
    fig, axes = corr_overview_figure(mats, title="test")
    assert axes.shape == (3, 4)
    for ax in axes.ravel():
        assert len(ax.images) == 1
    matplotlib.pyplot.close(fig)

    diffs = compute_single_soma_corr_differences(mats)
    assert list(diffs["ls"].d) == ["Wake − NREM", "Wake − REM", "NREM − REM"]
    fig2, axes2 = corr_difference_figure(diffs, title="diff")
    assert axes2.shape == (3, 3)
    for ax in axes2.ravel():
        assert len(ax.images) == 1
    matplotlib.pyplot.close(fig2)

    matched, tiles_m = compute_single_soma_corr_matrices(
        data, epoch_length=10.0, equalise=True
    )
    assert set(tiles_m.n_tiles.values()) == {3}
