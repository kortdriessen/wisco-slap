"""Tests for wisco_slap.scope.gecopur (matched-pursuit calcium event detection)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import xarray as xr

import wisco_slap as wis
from wisco_slap.scope._GECO_PURSUIT_PIPELINE import (
    _bridge_gaps,
    _forward_all,
    _template,
    baseline_dff,
    detect_events,
    estimate_kinetics,
    event_train,
    rates_at,
)

FS = 200.0


def _synthetic(
    seed: int = 0,
    dur_s: float = 600.0,
    rate: float = 0.25,
    tau_r: float = 0.06,
    tau_d: float = 0.6,
    amp_mu: float = 0.6,
    min_gap_s: float = 0.4,
    noise: float = 0.6,
    bleach_tau: float = 900.0,
    gaps: bool = True,
    n_soma: int = 1,
) -> tuple[xr.DataArray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Planted rise/decay calcium events on a bleaching baseline with
    shot-like noise. Returns (F, planted onsets per soma, amps per soma, h)."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    tr, td = tau_r * FS, tau_d * FS
    k = np.arange(int(6 * td))
    h = (1 - np.exp(-k / tr)) * np.exp(-k / td)
    h /= h.max()
    F_rows, onsets_all, amps_all = [], [], []
    for _ in range(n_soma):
        n_ev = int(rate * dur_s)
        onsets = np.sort(
            rng.choice(np.arange(int(2 * FS), n - int(5 * FS)), n_ev, replace=False)
        )
        if onsets.size:
            onsets = onsets[np.r_[True, np.diff(onsets) > int(min_gap_s * FS)]]
        amps = rng.lognormal(np.log(amp_mu), 0.5, onsets.size)
        clean = np.zeros(n)
        for o, a in zip(onsets, amps, strict=True):
            clean[o : o + h.size] += a * h[: n - o]
        F0 = 100 * np.exp(-t / bleach_tau) + 5
        Ftrue = F0 * (1 + clean)
        F = Ftrue + rng.normal(0, 1, n) * np.sqrt(Ftrue) * noise
        if gaps:
            F[5000:5010] = np.nan  # 50 ms: bridged
            F[60000:60400] = np.nan  # 2 s: not bridged
        F_rows.append(F)
        onsets_all.append(onsets)
        amps_all.append(amps)
    da = xr.DataArray(
        np.array(F_rows),
        dims=["soma_id", "time"],
        coords={"soma_id": [f"soma{i + 1}" for i in range(n_soma)], "time": t},
    )
    return da, onsets_all, amps_all, h


def _match(det_idx: np.ndarray, planted: np.ndarray, tol_s: float = 0.15):
    """(n hits within tol, n detections without a planted partner, n planted
    without a detection)."""
    tol = tol_s * FS
    d_det = np.array([np.min(np.abs(planted - o)) for o in det_idx])
    d_pl = np.array([np.min(np.abs(det_idx - o)) for o in planted])
    return int((d_pl <= tol).sum()), int((d_det > tol).sum()), int((d_pl > tol).sum())


def test_exposed_on_wis_scope() -> None:
    assert wis.scope.gecopur.detect_events is detect_events


def test_template_footprint_and_norms() -> None:
    T = _template(0.08, 0.5, FS)
    assert T.h_shape.max() == pytest.approx(1.0)
    assert np.linalg.norm(T.k_unit) == pytest.approx(1.0)
    # a unit event filtered with its own kernel peaks at e_norm, at zero lag
    assert T.footprint.max() == pytest.approx(1.0)
    assert T.peak_off == T.klen
    assert T.e_norm == pytest.approx(np.linalg.norm(T.h_shape))
    Tz = _template(0.08, 0.5, FS, zero_mean=True)
    assert abs(Tz.k_unit.sum()) < 1e-9
    assert Tz.snr_factor < 1.0


def test_bridge_gaps_and_forward_all() -> None:
    x = np.arange(10, dtype=float)
    x[3:5] = np.nan  # 2-sample gap: bridged
    x[7:10] = np.nan  # touches the end: never bridged
    out, bridged = _bridge_gaps(x, max_gap=2)
    assert np.allclose(out[3:5], [3.0, 4.0])
    assert bridged.tolist() == [False] * 3 + [True] * 2 + [False] * 5
    assert np.isnan(out[7:]).all()
    mask = np.array([True, True, False, True, True, True])
    # window of 2: valid where t and t+1 are True (last sample: window runs off)
    assert _forward_all(mask, 2).tolist() == [True, False, False, True, True, False]


def test_baseline_tracks_bleaching() -> None:
    F, _, _, _ = _synthetic(gaps=False, rate=0.05)
    dff, f0 = baseline_dff(F)
    t = F["time"].values
    true_f0 = 100 * np.exp(-t / 900) + 5
    ratio = f0.values[0] / true_f0
    print("f0/true ratio quantiles", np.nanquantile(ratio, [0.0, 0.02, 0.5, 0.98, 1.0]))
    # a low percentile sits a few % below the noisy quiet level by design;
    # what matters is that it tracks the ~2x decline of these 600 s
    assert np.nanmedian(ratio) == pytest.approx(1.0, abs=0.05)
    assert np.nanquantile(ratio, 0.02) > 0.93 and np.nanquantile(ratio, 0.98) < 1.07
    # quiet dF/F sits near zero
    assert abs(np.nanmedian(dff.values[0])) < 0.05


def test_detect_recovers_planted_events_with_timing() -> None:
    F, onsets, amps, h = _synthetic()
    res = detect_events(F, verbose=False)
    ev = res.events.filter(pl.col("z_shot") >= res.params["z_star"])
    planted = onsets[0]
    hits, extra, missed = _match(ev["idx_onset"].to_numpy(), planted)
    assert hits >= 0.95 * planted.size, (hits, planted.size)
    assert extra <= 0.03 * planted.size, extra
    # onset within one template rise step, peak within ~60 ms of the true peak
    on = ev["idx_onset"].to_numpy()
    pk = ev["idx_peak"].to_numpy()
    j = np.array([np.argmin(np.abs(planted - o)) for o in on])
    ok = np.abs(on - planted[j]) <= 0.15 * FS
    on_err = (on[ok] - planted[j][ok]) / FS
    pk_err = (pk[ok] - (planted[j][ok] + np.argmax(h))) / FS
    assert abs(np.median(on_err)) <= 0.01
    assert np.quantile(np.abs(on_err), 0.95) <= 0.03
    assert np.quantile(np.abs(pk_err), 0.95) <= 0.08
    # amplitudes: the data rise and the pursuit amplitude both track the truth
    a_true = amps[0][j][ok]
    assert np.median(ev["rise_dff"].to_numpy()[ok] / a_true) == pytest.approx(
        1.0, abs=0.15
    )
    assert np.median(ev["amp"].to_numpy()[ok] / a_true) == pytest.approx(1.0, abs=0.15)
    # auto decay lands near the planted 0.6 s
    assert 0.45 <= res.params["tau_decay_by_soma"]["soma1"] <= 0.85
    # a noise-only null should be nearly empty at the operating point
    row = res.fdr.filter(pl.col("z") == res.params["z_star"]).row(0, named=True)
    assert row["fdr"] <= 0.05


def test_detection_is_bleach_invariant() -> None:
    """The same planted events must be found equally well early (bright) and
    late (bleached): the noise scale is local."""
    F, onsets, _, _ = _synthetic(dur_s=1200.0, bleach_tau=600.0, gaps=False)
    res = detect_events(F, verbose=False)
    ev = res.events.filter(pl.col("z_shot") >= res.params["z_star"])
    planted = onsets[0]
    on = ev["idx_onset"].to_numpy()
    half = F.sizes["time"] // 2
    early = np.array([
        np.min(np.abs(on - o)) <= 0.15 * FS for o in planted[planted < half]
    ])
    late = np.array([
        np.min(np.abs(on - o)) <= 0.15 * FS for o in planted[planted >= half]
    ])
    assert early.mean() >= 0.9 and late.mean() >= 0.9
    assert abs(early.mean() - late.mean()) <= 0.08
    # dF/F amplitudes are comparable early and late (F0-normalised)
    e_amp = ev.filter(pl.col("idx_onset") < half)["rise_dff"].median()
    l_amp = ev.filter(pl.col("idx_onset") >= half)["rise_dff"].median()
    assert abs(np.log(e_amp / l_amp)) < 0.25


def test_gap_handling() -> None:
    F, onsets, _, _ = _synthetic()
    res = detect_events(F, verbose=False)
    ev = res.events
    # the 2-s gap blinds the detector for one template length before it and
    # is never bridged; the 50-ms gap is bridged and flagged on events
    # spanning it
    snr = res.snr.values[0]
    assert np.isnan(snr[60000:60400]).all()
    assert np.isnan(snr[60000 - 5])
    assert "gap_bridged" in ev.columns
    assert ev["gap_bridged"].dtype == pl.Boolean


def test_stacked_events_are_resolved() -> None:
    F, onsets, _, _ = _synthetic(rate=0.6, min_gap_s=0.25, dur_s=600.0)
    res = detect_events(F, verbose=False)
    ev = res.events.filter(pl.col("z_shot") >= res.params["z_star"])
    hits, extra, missed = _match(ev["idx_onset"].to_numpy(), onsets[0])
    assert hits >= 0.85 * onsets[0].size, (hits, onsets[0].size)
    assert extra <= 0.03 * onsets[0].size


def test_no_events_when_only_noise() -> None:
    F, _, _, _ = _synthetic(rate=0.0, gaps=False)
    res = detect_events(F, verbose=False)
    ev = res.events.filter(pl.col("z_shot") >= res.params["z_star"])
    assert ev.height <= 3
    assert res.per_soma["n"][0] == ev.height


def test_multi_soma_and_1d_input() -> None:
    F, onsets, _, _ = _synthetic(n_soma=2, seed=3)
    res = detect_events(F, verbose=False)
    assert set(res.events["soma_id"].unique().to_list()) <= {"soma1", "soma2"}
    assert res.per_soma.height == 2
    assert res.dff.dims == ("soma_id", "time") and res.snr.shape == F.shape
    for i, sid in enumerate(["soma1", "soma2"]):
        ev = res.events.filter(
            (pl.col("soma_id") == sid) & (pl.col("z_shot") >= res.params["z_star"])
        )
        hits, _, _ = _match(ev["idx_onset"].to_numpy(), onsets[i])
        assert hits >= 0.9 * onsets[i].size
    # 1-D input with a scalar soma_id coord is promoted to one row
    one = F.isel(soma_id=0)
    res1 = detect_events(one, verbose=False)
    assert res1.events["soma_id"].unique().to_list() == ["soma1"]
    assert res1.dff.dims == ("soma_id", "time")


def test_input_is_dff_skips_baseline() -> None:
    F, onsets, _, _ = _synthetic(gaps=False, bleach_tau=1e9)
    dff, _ = baseline_dff(F)
    res = detect_events(dff, input_is_dff=True, verbose=False)
    assert float(res.f0.values.min()) == 1.0
    ev = res.events.filter(pl.col("z_shot") >= res.params["z_star"])
    hits, _, _ = _match(ev["idx_onset"].to_numpy(), onsets[0])
    assert hits >= 0.9 * onsets[0].size


def test_rates_at_and_event_train() -> None:
    F, _, _, _ = _synthetic()
    res = detect_events(F, verbose=False)
    loose = rates_at(res, 3.0)
    strict = rates_at(res, 20.0)
    assert loose["n"][0] >= strict["n"][0]
    # rate is per detectable second
    n = loose["n"][0]
    assert loose["rate"][0] == pytest.approx(
        n / (res.per_soma["detectable_frac"][0] * res.params["duration_s"])
    )
    tr = event_train(res, res.dff, value="ones")
    assert tr.shape == res.dff.shape
    assert int(tr.sum()) == res.per_soma["n"][0]
    tr_amp = event_train(res, res.dff, at="peak", value="rise_dff")
    ev = res.events.filter(pl.col("z_shot") >= res.params["z_star"])
    assert float(tr_amp.sum()) == pytest.approx(float(ev["rise_dff"].sum()), rel=1e-5)


def test_fixed_tau_and_fdr_alpha_paths() -> None:
    F, onsets, _, _ = _synthetic()
    res = detect_events(F, tau_decay_s=0.6, z_threshold=6.0, verbose=False)
    assert res.per_soma["tau_source"][0] == "fixed"
    assert res.params["z_star"] == 6.0
    ev = res.events.filter(pl.col("z_shot") >= 6.0)
    hits, _, _ = _match(ev["idx_onset"].to_numpy(), onsets[0])
    assert hits >= 0.9 * onsets[0].size
    with pytest.raises(ValueError):
        detect_events(F, tau_decay_s="nonsense", verbose=False)


def test_estimate_kinetics_recovers_decay() -> None:
    F, _, _, _ = _synthetic(tau_d=1.0, rate=0.15)
    dff, _ = baseline_dff(F)
    kin = estimate_kinetics(dff)
    row = kin.row(0, named=True)
    assert row["n_used"] >= 5
    # half-decay after the peak of a rise/decay shape overestimates tau_d a
    # little (the rise term is still growing): allow [0.8, 1.6] for 1.0
    assert 0.8 <= row["tau_decay_s"] <= 1.6
    assert 0.0 < row["rise_time_s"] < 0.4


def test_event_columns_are_consistent() -> None:
    F, _, _, _ = _synthetic()
    res = detect_events(F, verbose=False)
    ev = res.events
    assert (ev["peak_time"] >= ev["onset_time"]).all()
    assert (ev["rise_dff"] > 0).all()
    assert np.allclose(
        ev["rise_dff"].to_numpy(), (ev["peak_dff"] - ev["pre_dff"]).to_numpy()
    )
    assert (ev["z_shot"] <= ev["z"] + 1e-9).all()
    assert (ev["rise_sigma"] >= res.params["min_rise_sigma"]).all()
    assert (ev["n_merged"] >= 1).all()
    assert (ev["f0_frac"] > 0).all()
    assert ev.sort(["soma_id", "onset_time"]).equals(ev)
