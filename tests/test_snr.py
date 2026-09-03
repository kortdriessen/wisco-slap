"""Tests for wisco_slap.util.snr rolling scale estimators."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from wisco_slap.util.snr import lower_half_mad_sigma, rolling_mad

# Shared synthetic geometry: long enough for stable medians, window sized so
# the central region is unaffected by edge padding.
N = 6000
WINDOW = 1001
CORE = slice(WINDOW, N - WINDOW)


def _noise(sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma, N)


def _positive_events(n_events: int, seed: int = 0) -> np.ndarray:
    """Strictly positive exponential-decay transients (glutamate-event-like)."""
    rng = np.random.default_rng(seed)
    tau, klen = 30, 150
    kern = np.exp(-np.arange(klen) / tau)
    events = np.zeros(N)
    starts = rng.choice(np.arange(0, N - klen), size=n_events, replace=False)
    for s in starts:
        events[s : s + klen] += rng.uniform(5, 15) * kern
    return events


def test_rolling_mad_recovers_sigma_on_gaussian_noise() -> None:
    da = xr.DataArray(_noise(), dims=["time"])
    std_est = float(rolling_mad(da, WINDOW)[CORE].median())
    low_est = float(rolling_mad(da, WINDOW, lower_envelope=True)[CORE].median())
    # On symmetric noise the lower-envelope MAD equals the two-sided MAD, and
    # both (x1.4826) estimate Gaussian sigma.
    assert abs(std_est - 1.0) < 0.05
    assert abs(low_est - 1.0) < 0.05


def test_rolling_mad_lower_envelope_resists_positive_events() -> None:
    # Heavy positive-transient load (~40% occupancy, true sigma = 1): the
    # two-sided MAD inflates badly; the lower envelope inflates less (only
    # via the median shift, its deviation pool stays event-free).
    trace = _noise() + _positive_events(n_events=40)
    da = xr.DataArray(trace, dims=["time"])
    std_est = float(rolling_mad(da, WINDOW)[CORE].median())
    low_est = float(rolling_mad(da, WINDOW, lower_envelope=True)[CORE].median())
    assert std_est > 1.5  # two-sided MAD is badly event-inflated
    assert abs(low_est - 1.0) < abs(std_est - 1.0)  # lower envelope closer
    assert low_est < std_est - 0.1


def test_rolling_mad_lower_envelope_collapses_on_rectified_trace() -> None:
    # Documents the caveat: on a non-negative floored trace (denoised-like),
    # the below-median half hugs the floor and the estimate is ~0, i.e. the
    # lower envelope is the WRONG tool there.
    rect = np.maximum(_noise() + _positive_events(n_events=20) - 1.0, 0.0)
    da = xr.DataArray(rect, dims=["time"])
    low_est = float(rolling_mad(da, WINDOW, lower_envelope=True)[CORE].median())
    assert low_est < 0.05


def test_rolling_mad_lower_envelope_nan_gap() -> None:
    trace = _noise()
    trace[3000 : 3000 + 2 * WINDOW] = np.nan
    da = xr.DataArray(trace, dims=["time"])
    out = rolling_mad(da, WINDOW, lower_envelope=True)
    assert np.isnan(out[3000 + WINDOW].item())  # center of long gap
    assert np.isfinite(out[2990].item())  # just before the gap


def _ls_dn_pair(n_events: int, seed: int = 0) -> tuple[xr.DataArray, xr.DataArray]:
    """Synthetic (denoised, ls) pair: dn = clean events, ls = events + noise."""
    events = _positive_events(n_events=n_events, seed=seed)
    time = np.arange(N) / 200.0
    dn = xr.DataArray(events[None, :], dims=["syn_id", "time"], coords={"time": time})
    ls = xr.DataArray(
        (events + _noise(seed=seed))[None, :],
        dims=["syn_id", "time"],
        coords={"time": time},
    )
    return dn, ls


def test_lower_half_mad_sigma_recovers_sigma() -> None:
    dn, ls = _ls_dn_pair(n_events=20)
    sigma = lower_half_mad_sigma(dn, ls, baseline_quantile=0.5)
    assert sigma.dims == ("syn_id",)
    assert abs(float(sigma.item()) - 1.0) < 0.1


def test_lower_half_mad_sigma_conditioning_beats_unconditioned() -> None:
    # Heavy sustained activity: conditioning on the denoised-quiet half keeps
    # the estimate near true sigma where the unconditioned lower envelope of
    # ls inflates via the median shift.
    dn, ls = _ls_dn_pair(n_events=40)
    cond = float(lower_half_mad_sigma(dn, ls, baseline_quantile=0.5).item())
    uncond = float(lower_half_mad_sigma(dn, ls, baseline_quantile=None).item())
    assert abs(cond - 1.0) < 0.1
    assert abs(cond - 1.0) < abs(uncond - 1.0)


def test_lower_half_mad_sigma_rolling_mode() -> None:
    dn, ls = _ls_dn_pair(n_events=20)
    sigma = lower_half_mad_sigma(dn, ls, baseline_quantile=0.5, window_s=20.0)
    assert sigma.dims == ls.dims
    core_vals = sigma.isel(time=CORE).values
    assert np.nanmedian(core_vals) == pytest.approx(1.0, abs=0.15)


def test_lower_half_mad_sigma_multidim_axis_order() -> None:
    # 3D input with time NOT last: result must match per-slice computation.
    rng = np.random.default_rng(7)
    time = np.arange(N) / 200.0
    dn3 = xr.DataArray(
        rng.exponential(1.0, (2, N, 3)),
        dims=["channel", "time", "syn_id"],
        coords={"time": time},
    )
    ls3 = dn3 + xr.DataArray(
        rng.normal(0, 1.0, (2, N, 3)),
        dims=["channel", "time", "syn_id"],
        coords={"time": time},
    )
    full = lower_half_mad_sigma(dn3, ls3, baseline_quantile=0.5)
    assert full.dims == ("channel", "syn_id")
    for ch in range(2):
        for s in range(3):
            single = lower_half_mad_sigma(
                dn3.isel(channel=ch, syn_id=s),
                ls3.isel(channel=ch, syn_id=s),
                baseline_quantile=0.5,
            )
            assert float(full.isel(channel=ch, syn_id=s)) == pytest.approx(
                float(single), rel=1e-12
            )


def test_rolling_mad_lower_envelope_multidim_and_uncentered() -> None:
    arr = np.stack([_noise(seed=0), _noise(seed=1)])
    da = xr.DataArray(arr, dims=["syn", "time"])
    out = rolling_mad(da, WINDOW, lower_envelope=True, center=False)
    assert out.dims == da.dims
    assert out.shape == da.shape
    # trailing window: first window-1 samples lack min_count real samples only
    # partially; core region must be finite and near sigma for both rows
    core_vals = out.isel(time=CORE)
    assert np.isfinite(core_vals).all()
    med = core_vals.median(dim="time").values
    assert np.all(np.abs(med - 1.0) < 0.05)


def test_rolling_sigma_uses_local_baseline_conditioning():
    """A trace whose denoised level sits above its recording-wide median for a
    long early stretch (slow calcium before bleaching) must still get a rolling
    sigma there — the quiet half is defined within the window, not globally."""
    import numpy as np
    import xarray as xr

    from wisco_slap.util.snr import lower_half_mad_sigma

    rng = np.random.default_rng(0)
    fs, n = 100.0, 60_000  # 10 min
    t = np.arange(n) / fs
    noise = rng.standard_normal(n) * 2.0  # true sigma = 2 throughout
    # denoised level: high plateau for the first 3 minutes, low afterwards
    dn_level = np.where(t < 180.0, 10.0, 1.0) + rng.random(n)
    ls = xr.DataArray(
        (dn_level + noise)[None, :].astype(np.float32),
        dims=["syn_id", "time"],
        coords={"syn_id": ["ca"], "time": t},
    )
    dn = xr.DataArray(
        dn_level[None, :].astype(np.float32), dims=ls.dims, coords=ls.coords
    )
    sig = lower_half_mad_sigma(dn, ls, window_s=60.0).values[0]
    early, late = sig[(t > 30) & (t < 150)], sig[(t > 240) & (t < 570)]
    assert np.isfinite(early).all() and np.isfinite(late).all()
    assert np.allclose(np.median(early), 2.0, rtol=0.15)
    assert np.allclose(np.median(late), 2.0, rtol=0.15)
    # scalar mode unchanged: global conditioning, one value
    scalar = lower_half_mad_sigma(dn, ls).values
    assert scalar.shape == (1,) and np.isfinite(scalar[0])
