"""Tests for the canonical matched filter in ``wisco_slap.scope.pro``.

Locks in the properties that make ``ls_to_matched_filter`` the correct,
optimal-SNR base: a unit-energy template, onset-aligned output (no centering
shift), the ``sqrt(E)`` SNR gain of a matched filter under white noise, and
NaN-safe behaviour.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from wisco_slap.scope.pro import (
    IGLUSNFR_TAU_S,
    _matched_filter_kernel,
    ls_to_matched_filter,
)


def _exp_event(length: int, tau_frames: float, amp: float) -> np.ndarray:
    """Raw (un-normalised) instantaneous-rise + exp-decay event shape."""
    return amp * np.exp(-np.arange(length) / tau_frames)


def test_kernel_is_unit_energy() -> None:
    kernel = _matched_filter_kernel(5.2)
    assert np.isclose(np.linalg.norm(kernel), 1.0)
    # onset at index 0 == max for the pure-decay template
    assert int(np.argmax(kernel)) == 0


def test_kernel_finite_rise_starts_at_zero() -> None:
    kernel = _matched_filter_kernel(5.2, tau_rise_frames=1.0)
    assert np.isclose(np.linalg.norm(kernel), 1.0)
    assert kernel[0] == 0.0  # finite rise => onset value is zero
    assert int(np.argmax(kernel)) > 0  # peak after onset


def test_onset_alignment_no_shift() -> None:
    """A clean event whose onset is at n0 must produce an MF peak at n0."""
    fs = 200.0
    tau_frames = IGLUSNFR_TAU_S * fs
    n = 4000
    time = np.arange(n) / fs
    klen = len(_matched_filter_kernel(tau_frames))

    x = np.zeros(n)
    onsets = [800, 1900, 3100]
    for o in onsets:
        x[o : o + klen] += _exp_event(klen, tau_frames, amp=10.0)

    da = xr.DataArray(x[None, :], dims=["syn_id", "time"],
                      coords={"syn_id": [0], "time": time})
    mf = ls_to_matched_filter(da, tau_s=IGLUSNFR_TAU_S).values[0]

    for o in onsets:
        local_peak = int(np.argmax(mf[o - 15 : o + 15])) - 15
        assert local_peak == 0, f"onset {o} shifted by {local_peak}"


def test_unit_norm_noise_and_snr_gain() -> None:
    """Output noise std ~= input sigma, and peak SNR gain ~= sqrt(E)."""
    rng = np.random.default_rng(20260622)
    fs = 200.0
    tau_frames = IGLUSNFR_TAU_S * fs
    n = 60000
    time = np.arange(n) / fs
    klen = len(_matched_filter_kernel(tau_frames))
    raw_shape = _exp_event(klen, tau_frames, amp=1.0)
    energy = float(np.sum(raw_shape**2))

    amp, sigma = 6.0, 1.0
    onsets = np.arange(600, n - 600, 1200)
    clean = np.zeros(n)
    for o in onsets:
        clean[o : o + klen] += amp * raw_shape
    noisy = clean + rng.normal(0.0, sigma, n)

    da = xr.DataArray(noisy[None, :], dims=["syn_id", "time"],
                      coords={"syn_id": [0], "time": time})
    mf = ls_to_matched_filter(da, tau_s=IGLUSNFR_TAU_S).values[0]

    # noise std in an event-free region (before first onset)
    mf_noise = float(np.nanstd(mf[100:500]))
    assert 0.8 < mf_noise / sigma < 1.2

    mf_peaks = np.array([np.nanmax(mf[o - 5 : o + klen]) for o in onsets])
    # peak height ~= amp * sqrt(E)
    assert np.isclose(mf_peaks.mean(), amp * np.sqrt(energy), rtol=0.1)
    # SNR gain ~= sqrt(E)
    gain = (mf_peaks.mean() / mf_noise) / (amp / sigma)
    assert np.isclose(gain, np.sqrt(energy), rtol=0.15)


def test_nan_safe_gaps() -> None:
    fs = 200.0
    n = 3000
    time = np.arange(n) / fs
    x = np.zeros(n)
    x[1000:1010] += _exp_event(10, IGLUSNFR_TAU_S * fs, amp=8.0)
    x[1500:1700] = np.nan  # a gap

    da = xr.DataArray(x[None, :], dims=["syn_id", "time"],
                      coords={"syn_id": [0], "time": time})
    mf = ls_to_matched_filter(da, tau_s=IGLUSNFR_TAU_S).values[0]

    # interior of the gap must be NaN; far-from-gap region must be finite
    assert np.isnan(mf[1590])
    assert np.isfinite(mf[1005])
    assert np.isfinite(mf[100])


def test_shape_dims_coords_attrs_preserved() -> None:
    fs = 200.0
    n = 500
    da = xr.DataArray(
        np.random.default_rng(0).normal(size=(2, 3, n)).astype(np.float32),
        dims=["channel", "syn_id", "time"],
        coords={"channel": [0, 1], "syn_id": [10, 11, 12],
                "time": np.arange(n) / fs},
    )
    mf = ls_to_matched_filter(da, tau_s=IGLUSNFR_TAU_S)
    assert mf.dims == da.dims
    assert mf.shape == da.shape
    assert list(mf.coords["syn_id"].values) == [10, 11, 12]
    assert mf.attrs["trace_type"] == "matched_filter"
    assert mf.attrs["tau_s"] == IGLUSNFR_TAU_S
