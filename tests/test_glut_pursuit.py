"""Tests for wisco_slap.scope.pursuit (matched-pursuit event detection)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import wisco_slap as wis
from wisco_slap.scope._GLUT_PURSUIT_PIPELINE import (
    denoised_sigma_units,
    detect_events,
    matchfilt_snr,
    rates_at,
    spike_train,
)

FS = 200.0
N = 60_000  # 5 min
TRUE_RATES = {"a": 0.5, "b": 1.0, "c": 0.2}  # ev/s
AMPS = {"a": 8.0, "b": 12.0, "c": 10.0}  # in noise-sigma units


def _synthetic(
    seed: int = 0,
) -> tuple[xr.DataArray, xr.DataArray, dict[str, int]]:
    """Planted exponential events + white noise; dn = clean, ls = clean + noise.

    Also returns the REALIZED planted-event count per synapse (spikes closer
    than the 10 ms refractory merged), which is the proper ground truth.
    """
    rng = np.random.default_rng(seed)
    tau = 0.026 * FS
    kern = np.exp(-np.arange(26) / tau)
    dn_rows, ls_rows, planted = [], [], {}
    for sid in TRUE_RATES:
        spikes = (rng.random(N) < TRUE_RATES[sid] / FS) * AMPS[sid]
        idx = np.flatnonzero(spikes)
        planted[sid] = int(1 + np.sum(np.diff(idx) >= 2)) if idx.size else 0
        clean = np.convolve(spikes, kern)[:N]
        dn_rows.append(clean)
        ls_rows.append(clean + rng.normal(0.0, 1.0, N))
    coords = {
        "syn_id": list(TRUE_RATES),
        "time": np.arange(N) / FS,
        "dend-ID": ("syn_id", ["A-1", "A-1", "B-1"]),
        "pos": ("syn_id", [1.0, 2.0, 1.0]),
    }
    dn = xr.DataArray(np.array(dn_rows), dims=["syn_id", "time"], coords=coords)
    ls = xr.DataArray(np.array(ls_rows), dims=["syn_id", "time"], coords=coords)
    return dn, ls, planted


def test_pursuit_exposed_on_wis_scope() -> None:
    assert wis.scope.pursuit.detect_events is detect_events
    assert wis.scope.pursuit.matchfilt_snr is matchfilt_snr


def test_detect_events_recovers_planted_events() -> None:
    dn, ls, planted = _synthetic()
    res = detect_events(dn, ls, verbose=False)
    got = dict(
        zip(res.rates["syn_id"].to_list(), res.rates["n"].to_list(), strict=True)
    )
    for sid, n_true in planted.items():
        # large planted events on white noise: counts recovered within 15%
        assert got[sid] == pytest.approx(n_true, rel=0.15), (sid, got[sid], n_true)
    # white synthetic noise -> noise-color constant ~1
    assert all(0.85 < c < 1.2 for c in res.params["c_by_group"].values())


def test_events_carry_synapse_metadata() -> None:
    dn, ls, _ = _synthetic()
    res = detect_events(dn, ls, verbose=False)
    for col in ("dend-ID", "pos"):
        assert col in res.events.columns
        assert col in res.rates.columns
        assert col in rates_at(res, 8.0).columns
    assert set(res.params["meta_cols"]) == {"dend-ID", "pos"}
    c_events = res.events.filter(res.events["syn_id"] == "c")
    assert c_events["dend-ID"].unique().to_list() == ["B-1"]


def test_rates_at_reslices_without_recompute() -> None:
    dn, ls, _ = _synthetic()
    res = detect_events(dn, ls, verbose=False)
    loose = rates_at(res, 4.0)["rate"].to_numpy()
    strict = rates_at(res, 12.0)["rate"].to_numpy()
    assert np.all(strict <= loose)


def test_matchfilt_snr_shape_coords_and_invariant() -> None:
    dn, ls, _ = _synthetic()
    snr = matchfilt_snr(dn, ls)
    assert snr.dims == dn.dims and snr.shape == dn.shape
    assert list(snr["syn_id"].values) == list(dn["syn_id"].values)
    assert "dend-ID" in snr.coords
    # catalog z can never exceed the raw SNR trace at the event sample
    res = detect_events(dn, ls, verbose=False)
    rows = {s: i for i, s in enumerate(dn["syn_id"].values)}
    ev = res.events
    raw = snr.values[[rows[s] for s in ev["syn_id"].to_list()], ev["idx"].to_numpy()]
    assert np.nanmax(ev["z"].to_numpy() - raw) < 1e-3
    # shot-corrected variant never exceeds the plain one
    snr_s = matchfilt_snr(dn, ls, shot_correction=True)
    assert np.nanmax(snr_s.values - snr.values) < 1e-6


def test_matchfilt_snr_nan_near_gaps() -> None:
    dn, ls, _ = _synthetic()
    ls = ls.copy()
    ls[:, 1000:1200] = np.nan
    snr = matchfilt_snr(dn, ls)
    assert np.isnan(snr.values[:, 1000:1200]).all()
    assert np.isnan(snr.values[:, 990]).all()  # within one template length
    assert np.isfinite(snr.values[:, 5000]).all()


def test_spike_train_rasterizes_catalog() -> None:
    dn, ls, _ = _synthetic()
    res = detect_events(dn, ls, verbose=False)
    st = spike_train(res, dn)
    assert st.dims == dn.dims and st.shape == dn.shape
    n_events = res.events.filter(res.events["z_shot"] >= res.params["z_star"]).height
    assert int((st.values > 0).sum()) == n_events
    clean = spike_train(res, dn, reconvolve=True)
    assert clean.shape == dn.shape and float(clean.max()) > 0


def test_detect_events_rejects_wrong_dims() -> None:
    dn, ls, _ = _synthetic()
    with pytest.raises(ValueError):
        detect_events(dn.expand_dims(channel=[0]), ls.expand_dims(channel=[0]))


def test_denoised_sigma_units_is_dn_over_sigma0() -> None:
    dn, ls, _ = _synthetic()
    out = denoised_sigma_units(dn, ls, sigma_window_s=None)
    assert out.dims == dn.dims and out.shape == dn.shape
    assert "dend-ID" in out.coords
    sigma0 = wis.util.lower_half_mad_sigma(dn, ls).values
    expected = dn.values / sigma0[:, None]
    assert np.allclose(out.values, expected, rtol=1e-5)
    # synthetic noise sd is 1, so the yardstick itself should be ~1
    assert np.allclose(sigma0, 1.0, atol=0.15)
    # default is now ROLLING sigma0 (SIGMA_WINDOW_S_DEFAULT)
    rolling = denoised_sigma_units(dn, ls)
    assert rolling.shape == dn.shape
    assert rolling.attrs["sigma_window_s"] == 300.0
    assert (
        denoised_sigma_units(dn, ls, sigma_window_s=None).attrs["sigma_window_s"]
        == "scalar"
    )


def test_detectable_mask_matches_detector_valid_and_dilates_gaps():
    """``detectable_mask`` is the detector's own exclusion zone: one template
    length (3 tau, causal filter) BEFORE each NaN gap and nothing after."""
    import numpy as np
    import xarray as xr

    from wisco_slap.scope import _GLUT_PURSUIT_PIPELINE as gp

    rng = np.random.default_rng(0)
    fs, n = 200.0, 4000
    t = np.arange(n) / fs
    ls = xr.DataArray(
        rng.standard_normal((3, n)).astype(np.float32),
        dims=["syn_id", "time"],
        coords={"syn_id": ["a", "b", "c"], "time": t},
    )
    ls[:, 1000:1100] = np.nan
    ls[2, 3000] = np.nan  # a single bad sample
    dn = ls.copy()
    tau_s = 0.026
    klen = gp._template_len(tau_s, fs)
    assert klen == int(np.ceil(3 * tau_s * fs))

    mask = gp.detectable_mask(ls, tau_s=tau_s)
    assert mask.dims == ls.dims and mask.dtype == bool
    P = gp._prepare(
        dn,
        ls,
        tau_s=tau_s,
        baseline_quantile=0.5,
        quiet_syn_frac=0.3,
        group_coord=None,
        time_dim="time",
        verbose=False,
    )
    assert np.array_equal(mask.values, P.valid)
    bad = ~mask.values
    # exclusion = [g0 - klen + 1, g1]: klen-1 samples before, none after
    assert bad[0, 1000 - klen + 1 : 1100].all()
    assert not bad[0, 1000 - klen] and not bad[0, 1100]
    assert bad[2, 3000 - klen + 1 : 3001].all() and not bad[2, 3001]
    assert not bad[1, 3000]
    # transposed input keeps its own dim order
    assert gp.detectable_mask(ls.transpose("time", "syn_id"), tau_s=tau_s).dims == (
        "time",
        "syn_id",
    )


def test_matchfilt_snr_nan_floor_masks_only_raw_nans():
    import numpy as np
    import xarray as xr

    from wisco_slap.scope import _GLUT_PURSUIT_PIPELINE as gp

    rng = np.random.default_rng(1)
    fs, n = 200.0, 6000
    t = np.arange(n) / fs
    ls = xr.DataArray(
        rng.standard_normal((4, n)).astype(np.float32),
        dims=["syn_id", "time"],
        coords={"syn_id": list("abcd"), "time": t},
    )
    ls[:, 2000:2010] = np.nan
    dn = xr.zeros_like(ls)
    strict = gp.matchfilt_snr(dn, ls, tau_s=0.026)
    floor = gp.matchfilt_snr(dn, ls, tau_s=0.026, nan_floor=True)
    klen = gp._template_len(0.026, fs)
    n_raw = int(ls.isnull().sum())
    assert int(floor.isnull().sum()) == n_raw
    assert int(strict.isnull().sum()) == 4 * (10 + klen - 1)
    assert floor.attrs["nan_floor"] is True and strict.attrs["nan_floor"] is False
    # identical wherever the strict version keeps data
    keep = np.isfinite(strict.values)
    assert np.array_equal(strict.values[keep], floor.values[keep])
    # the extra retained samples in the leading window are finite (attenuated)
    lead = floor.values[:, 2000 - klen + 1 : 2000]
    assert np.isfinite(lead).all()


def test_sigma_window_s_stabilises_rates_under_gain_drift():
    """A noise ramp (bleaching stand-in) skews the z_floor candidate rate over
    time with a scalar sigma0; rolling sigma0 balances it."""
    import numpy as np
    import xarray as xr

    from wisco_slap.scope import _GLUT_PURSUIT_PIPELINE as gp

    rng = np.random.default_rng(7)
    fs, n = 200.0, 120_000  # 10 min
    t = np.arange(n) / fs
    ramp = np.linspace(1.0, 2.0, n)  # noise doubles over the recording
    noise = rng.standard_normal((3, n)) * ramp
    ls = xr.DataArray(
        noise.astype(np.float32),
        dims=["syn_id", "time"],
        coords={"syn_id": ["a", "b", "c"], "time": t},
    )
    dn = xr.zeros_like(ls)  # all-quiet: every candidate is a false positive

    def half_ratio(res):
        ev = res.events
        first = ev.filter(ev["time"] < t[n // 2]).height
        last = ev.height - first
        return last / max(first, 1)

    scalar = gp.detect_events(dn, ls, tau_s=0.026, sigma_window_s=None, verbose=False)
    rolling = gp.detect_events(dn, ls, tau_s=0.026, sigma_window_s=60.0, verbose=False)
    assert scalar.params["sigma_window_s"] is None
    assert rolling.params["sigma_window_s"] == 60.0
    r_scalar, r_rolling = half_ratio(scalar), half_ratio(rolling)
    assert r_scalar > 3.0  # scalar sigma0: noisy half dominates the catalog
    assert 0.5 < r_rolling < 2.0  # rolling sigma0: balanced null

    # matchfilt_snr inherits: its dispersion becomes stationary
    snr_s = gp.matchfilt_snr(dn, ls, tau_s=0.026, sigma_window_s=None).values
    snr_r = gp.matchfilt_snr(dn, ls, tau_s=0.026, sigma_window_s=60.0).values

    def mad(x):
        x = x[np.isfinite(x)]
        return np.median(np.abs(x - np.median(x)))

    disp_s = mad(snr_s[:, -n // 4 :]) / mad(snr_s[:, : n // 4])
    disp_r = mad(snr_r[:, -n // 4 :]) / mad(snr_r[:, : n // 4])
    assert disp_s > 1.5 and 0.8 < disp_r < 1.25


def test_denoised_sigma_units_rolling_has_no_nan_strips():
    import numpy as np
    import xarray as xr

    from wisco_slap.scope import _GLUT_PURSUIT_PIPELINE as gp

    rng = np.random.default_rng(3)
    fs, n = 100.0, 60_000
    t = np.arange(n) / fs
    level = np.where(t < 180.0, 10.0, 1.0) + rng.random((2, n))
    ls = xr.DataArray(
        (level + rng.standard_normal((2, n))).astype(np.float32),
        dims=["syn_id", "time"],
        coords={"syn_id": ["a", "b"], "time": t},
    )
    dn = ls.copy(data=level.astype(np.float32))
    dn[0, 5000:5100] = np.nan
    out = gp.denoised_sigma_units(dn, ls, sigma_window_s=60.0)
    assert out.attrs["sigma_window_s"] == 60.0
    # NaN exactly where dn is NaN -- never from the rolling normalisation
    assert np.array_equal(np.isnan(out.values), np.isnan(dn.values))
