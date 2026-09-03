"""Tests for ``wisco_slap.scope.corr.state_matrices`` and ``corr_matrix_order``.

Synthetic recordings only: a handful of synapses at 100 Hz with a planted
shared component in one state, NaN gaps, and a two-state hypnogram. The
checks pin the accounting (equal tile sample counts, duration matching,
symmetric matrices with unit diagonal, mirrored coords) and the science
(planted correlation recovered in the right state only, noise floor small,
Spearman equals Pearson on ranks, partial correlation removes an indirect
path, STTC recovers coincident events, differences propagate the floors).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import xarray as xr

from electro_py.hypno.hypno import Hypnogram
from wisco_slap.scope.corr import (
    corr_matrix_order,
    matrix_difference,
    pair_class_summary,
    state_corr_matrices,
    state_corr_matrices_multi,
    state_differences,
    state_sttc_matrices,
    state_tiles,
    timescale_sweep,
)
from wisco_slap.scope.corr.state_matrices import (
    ALL_LABEL,
    _circular_shift_surrogate,
    _corr_dense,
    _dense_tile,
    common_validity_mask,
)

_FS = 100.0
_DT = 1.0 / _FS


def _hypno(bouts: list[tuple[str, float, float]]) -> Hypnogram:
    return Hypnogram(
        pl.DataFrame({
            "state": [b[0] for b in bouts],
            "start_time": [float(b[1]) for b in bouts],
            "end_time": [float(b[2]) for b in bouts],
        })
    )


def _scopex(X: np.ndarray, time: np.ndarray, **coords) -> xr.DataArray:
    n = X.shape[0]
    c = {"syn_id": [f"1-{i}" for i in range(n)], "time": time}
    c["dmd"] = ("syn_id", np.asarray(coords.pop("dmd", np.ones(n, dtype=int))))
    for k, v in coords.items():
        c[k] = ("syn_id", np.asarray(v))
    return xr.DataArray(X, dims=("syn_id", "time"), coords=c)


def _synthetic(seed: int = 0):
    """4 synapses, 300 s. Synapses 0 & 1 share a component during NREM only."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 300.0, _DT)
    n = t.size
    X = rng.normal(size=(4, n))
    # bouts: Wake 0-100, NREM 100-220, Wake 220-300
    nrem = (t >= 100.0) & (t < 220.0)
    shared = rng.normal(size=n) * 2.0
    X[0, nrem] += shared[nrem]
    X[1, nrem] += shared[nrem]
    # slow drift common to all (must be removed by per-tile centring)
    X += np.sin(t / 60.0)[None, :] * 3.0
    # NaN gaps: 0.5 s inside NREM, 5 s inside first Wake bout
    X[:, (t >= 150.0) & (t < 150.5)] = np.nan
    X[:, (t >= 40.0) & (t < 45.0)] = np.nan
    da = _scopex(
        X,
        t,
        dmd=[1, 1, 2, 2],
        **{"dend-ID": ["A-1", "A-1", "B-1", "C-1"], "pos": [0.0, 1.0, 0.0, 0.0]},
        **{
            "dmd-depth": [200, 200, 100, 100],
            "soma-ID": ["s1"] * 4,
            "soma-depth": [200.0] * 4,
        },
    )
    hypno = _hypno([
        ("Wake", 0.0, 100.0),
        ("NREM", 100.0, 220.0),
        ("Wake", 220.0, 300.0),
    ])
    return da, hypno


def test_common_validity_mask_is_and_over_signals() -> None:
    da, _ = _synthetic()
    other = da.copy()
    other[2, 1000:1100] = np.nan
    m = common_validity_mask({"a": da, "b": other})
    assert m.dims == ("time",)
    assert not m.values[1000:1100].any()
    assert m.values.sum() == (~np.isnan(other.values).any(axis=0)).sum()


def test_state_tiles_equal_sample_counts_and_duration_matching() -> None:
    da, hypno = _synthetic()
    mask = common_validity_mask(da)
    tiles = state_tiles(
        hypno,
        mask,
        states=("Wake", "NREM", "REM"),
        epoch_length=10.0,
        equalise=True,
        seed=1,
    )
    # REM absent from the hypnogram -> not a column; All first.
    assert tiles.columns == [ALL_LABEL, "Wake", "NREM"]
    assert tiles.n_available["NREM"] in (11, 12)
    n_state = min(tiles.n_available["Wake"], tiles.n_available["NREM"])
    assert all(v == n_state for v in tiles.n_tiles.values())
    times = np.asarray(da.time.values)
    valid = np.asarray(mask.values)
    for col, b in tiles.tiles.items():
        counts = {
            _dense_tile(da.values, times, valid, r["start_time"], r["end_time"]).shape[
                1
            ]
            for r in b.iter_rows(named=True)
        }
        assert counts == {1000}, (col, counts)
        # tiles never leave their state
        for r in b.iter_rows(named=True):
            if col == ALL_LABEL:
                continue
            inside = hypno.mask_times_by_state(
                np.array([r["start_time"], r["end_time"]]), [col]
            )
            assert inside.all()


def test_state_tiles_default_keeps_everything() -> None:
    da, hypno = _synthetic()
    mask = common_validity_mask(da)
    tiles = state_tiles(hypno, mask, epoch_length=10.0)
    assert tiles.equalised is False
    assert tiles.n_tiles == tiles.n_available
    assert tiles.n_tiles["NREM"] >= 11


def test_state_corr_matrices_recovers_planted_structure() -> None:
    da, hypno = _synthetic()
    mask = common_validity_mask(da)
    tiles = state_tiles(hypno, mask, epoch_length=10.0)
    res = state_corr_matrices(da, tiles)
    assert list(res.r) == tiles.columns
    for col, r in res.r.items():
        M = r.values
        assert M.shape == (4, 4)
        assert np.allclose(np.diag(M), 1.0)
        assert np.allclose(M, M.T, equal_nan=True)
        # coords mirrored on both dims, including depths
        assert "dmd-depth_1" in r.coords and "dmd-depth_2" in r.coords
        assert list(r.coords["dend-ID_1"].values) == ["A-1", "A-1", "B-1", "C-1"]
        # noise floor is small and finite
        assert np.isfinite(res.null_sd[col]) and res.null_sd[col] < 0.06
    # shared component -> strong r in NREM, ~0 in Wake; other pairs ~0 everywhere
    assert res.r["NREM"].values[0, 1] > 0.6
    assert abs(res.r["Wake"].values[0, 1]) < 0.1
    assert abs(res.r["NREM"].values[2, 3]) < 0.1
    assert abs(res.r["Wake"].values[0, 2]) < 0.1
    # the slow common drift did not leak in (per-tile centring)
    assert abs(res.r["Wake"].values[2, 3]) < 0.1
    assert res.n_tiles["NREM"] == tiles.n_tiles["NREM"]
    assert res.valid_seconds["NREM"] == pytest.approx(10.0 * tiles.n_tiles["NREM"])


def test_spearman_equals_pearson_on_ranks_and_multi_shares_tiles() -> None:
    da, hypno = _synthetic()
    mask = common_validity_mask(da)
    tiles = state_tiles(hypno, mask, epoch_length=10.0)
    sp = state_corr_matrices(da, tiles, method="spearman")
    # a monotone transform must not change Spearman
    transformed = da.copy(data=np.exp(da.values / 4.0))
    sp2 = state_corr_matrices(transformed, tiles, method="spearman")
    for col in sp.r:
        assert np.allclose(sp.r[col].values, sp2.r[col].values, atol=1e-6)
    multi = state_corr_matrices_multi({"a": da, "b": transformed}, tiles)
    assert set(multi) == {"a", "b"}
    assert multi["a"].tiles is tiles
    # Pearson on the transformed trace differs from the original (the test is not
    # vacuous)
    assert not np.allclose(
        multi["a"].r["NREM"].values, multi["b"].r["NREM"].values, atol=1e-3
    )


def test_partial_correlation_removes_indirect_path() -> None:
    """Chain A -> B -> C: Pearson(A, C) > 0 but partial(A, C | B) ~ 0."""
    rng = np.random.default_rng(5)
    t = np.arange(0.0, 200.0, _DT)
    n = t.size
    a = rng.normal(size=n)
    b = a + 0.7 * rng.normal(size=n)
    c = b + 0.7 * rng.normal(size=n)
    d = rng.normal(size=n)
    da = _scopex(np.vstack([a, b, c, d]), t)
    hypno = _hypno([("Wake", 0.0, 200.0)])
    tiles = state_tiles(hypno, da.notnull().all("syn_id"), epoch_length=10.0)
    pear = state_corr_matrices(da, tiles).r["Wake"].values
    part = state_corr_matrices(da, tiles, method="partial").r["Wake"].values
    assert pear[0, 2] > 0.4
    assert abs(part[0, 2]) < 0.08
    assert part[0, 1] > 0.5 and part[1, 2] > 0.5
    assert np.allclose(np.diag(part), 1.0)
    assert np.allclose(part, part.T)
    assert abs(part[0, 3]) < 0.08


def test_bin_and_remove_common_run_and_change_values() -> None:
    da, hypno = _synthetic()
    mask = common_validity_mask(da)
    tiles = state_tiles(hypno, mask, epoch_length=10.0)
    base = state_corr_matrices(da, tiles)
    binned = state_corr_matrices(da, tiles, bin_s=0.1)
    resid = state_corr_matrices(da, tiles, remove_common=True)
    assert binned.params["bin_s"] == 0.1
    assert resid.params["remove_common"] is True
    # planted pair still recovered after binning (shared white noise averages
    # coherently)
    assert binned.r["NREM"].values[0, 1] > 0.5
    # regressing out the population mean lowers the planted pair's r (it dominates
    # the mean)
    assert resid.r["NREM"].values[0, 1] < base.r["NREM"].values[0, 1]
    with pytest.raises(ValueError):
        state_corr_matrices(da, tiles, method="kendall")


def test_circular_shift_surrogate_preserves_rows_and_breaks_sync() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 500))
    X[1] = X[0]  # perfectly synchronous pair
    S = _circular_shift_surrogate(X, np.random.default_rng(1), min_shift=50)
    for i in range(3):
        assert np.allclose(np.sort(S[i]), np.sort(X[i]))
    assert abs(_corr_dense(S)[0, 1]) < 0.3


def test_corr_dense_matches_numpy() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5, 400))
    assert np.allclose(_corr_dense(X), np.corrcoef(X), atol=1e-12)
    Y = rng.normal(size=(5, 400))
    ref = np.corrcoef(X, Y)[:5, 5:]
    assert np.allclose(_corr_dense(X, Y), ref, atol=1e-12)


def test_sttc_recovers_coincident_events() -> None:
    rng = np.random.default_rng(7)
    t = np.arange(0.0, 300.0, _DT)
    n = t.size
    E = np.zeros((4, n))
    # synapse 0: 1 Hz Poisson-ish events; synapse 1: the same events jittered +-1 sample
    ev0 = np.sort(rng.choice(n, size=300, replace=False))
    E[0, ev0] = 1.0
    E[1, np.clip(ev0 + rng.integers(-1, 2, size=ev0.size), 0, n - 1)] = 1.0
    # synapse 2: independent, 5x the rate; synapse 3: independent, low rate
    E[2, rng.choice(n, size=1500, replace=False)] = 1.0
    E[3, rng.choice(n, size=60, replace=False)] = 1.0
    da = _scopex(E, t)
    hypno = _hypno([("Wake", 0.0, 300.0)])
    tiles = state_tiles(hypno, da.notnull().all("syn_id"), epoch_length=10.0)
    res = state_sttc_matrices(da, tiles, window_s=0.02)
    M = res.r["Wake"].values
    assert res.params["method"] == "sttc"
    assert M[0, 1] > 0.9
    assert abs(M[0, 2]) < 0.1 and abs(M[2, 3]) < 0.1 and abs(M[0, 3]) < 0.1
    assert np.allclose(M, M.T) and np.allclose(np.diag(M), 1.0)
    assert res.null_sd["Wake"] < 0.1


def test_state_differences_propagate_floors() -> None:
    da, hypno = _synthetic()
    mask = common_validity_mask(da)
    tiles = state_tiles(hypno, mask, epoch_length=10.0)
    res = state_corr_matrices(da, tiles)
    diffs = state_differences(res, [("Wake", "NREM"), ("NREM", "REM")])
    assert diffs.columns == ["Wake − NREM"]  # REM absent -> skipped
    D = diffs.d["Wake − NREM"].values
    assert np.isnan(D[0, 0])
    assert D[0, 1] == pytest.approx(
        res.r["Wake"].values[0, 1] - res.r["NREM"].values[0, 1]
    )
    assert D[0, 1] < -0.5
    assert diffs.null_sd["Wake − NREM"] == pytest.approx(
        np.hypot(res.null_sd["Wake"], res.null_sd["NREM"])
    )
    assert diffs.pairs["Wake − NREM"] == ("Wake", "NREM")


def test_timescale_sweep_schema() -> None:
    da, hypno = _synthetic()
    mask = common_validity_mask(da)
    tiles = state_tiles(hypno, mask, epoch_length=10.0)
    df = timescale_sweep(da, tiles, bins_s=(None, 0.1), signal="x")
    assert set(df.columns) >= {"signal", "bin_s", "column", "pair_class", "mean_r"}
    assert df.height == 2 * 3 * 4  # bins x columns x classes
    native = df.filter(pl.col("bin_s") < 0.05)
    assert native["bin_s"].to_list()[0] == pytest.approx(_DT)


def test_pair_class_summary_and_matrix_difference() -> None:
    da, hypno = _synthetic()
    mask = common_validity_mask(da)
    tiles = state_tiles(hypno, mask, epoch_length=10.0)
    res = state_corr_matrices(da, tiles)
    s = pair_class_summary(res.r["NREM"])
    # within-dend = pair (0,1) only; between-dend = (2,3); cross-DMD = the other four
    assert s["within_dend"] == pytest.approx(res.r["NREM"].values[0, 1])
    assert s["between_dend"] == pytest.approx(res.r["NREM"].values[2, 3])
    assert abs(s["cross_dmd"]) < 0.1
    d = matrix_difference(res.r["NREM"], res.r["NREM"])
    assert d["rms_diff"] == 0.0 and d["max_abs_diff"] == 0.0
    assert d["corr"] == pytest.approx(1.0)


def test_corr_matrix_order_depth_puts_superficial_first() -> None:
    da, hypno = _synthetic()
    mask = common_validity_mask(da)
    tiles = state_tiles(hypno, mask, epoch_length=10.0)
    r = state_corr_matrices(da, tiles).r["NREM"]
    legacy = corr_matrix_order(r)  # DMD 1 first
    assert [int(v) for v in r.coords["dmd_1"].values[legacy.order_idx]] == [1, 1, 2, 2]
    by_depth = corr_matrix_order(r, dmd_order="depth")  # DMD 2 is at 100 um -> first
    assert [int(v) for v in r.coords["dmd_1"].values[by_depth.order_idx]] == [
        2,
        2,
        1,
        1,
    ]
    assert by_depth.middle_blocks[0][0] == 2
    assert by_depth.dmd_depth == {1: 200.0, 2: 100.0}
    assert by_depth.soma_depth == {"s1": 200.0}
    explicit = corr_matrix_order(r, dmd_order=[2, 1])
    assert explicit.order_idx == by_depth.order_idx
    # within a DMD, dendrite then pos
    assert by_depth.dend_blocks == [("B-1", 0, 0), ("C-1", 1, 1), ("A-1", 2, 3)]
    with pytest.raises(ValueError):
        corr_matrix_order(r, dmd_order="sideways")
