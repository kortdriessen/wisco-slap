"""Tests for wisco_slap.viz._ephys_locking (slow-wave-locked single-soma overview)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest
import xarray as xr

from wisco_slap.viz._ephys_locking import (
    superficial_dmd,
    wave_locked_overview,
    wave_locked_peths,
)

FS = 200.0
FS_EEG = 610.35
DUR_S = 120.0
BUMP_AMP = 2.0
BUMP_LAG = 0.10
EEG_AMP = -40.0
OFFSETS = {"start": -0.30, "mid": 0.15, "up_peak": 0.35, "end": 0.70}


def _waves() -> pl.DataFrame:
    trough = np.arange(5.0, DUR_S - 4.0, 5.0)
    stage = np.where(np.arange(trough.size) % 2 == 0, "NREM", "Wake")
    return pl.DataFrame(
        {
            "trough_time": trough,
            "start_time": trough + OFFSETS["start"],
            "mid_crossing_time": trough + OFFSETS["mid"],
            "up_peak_time": trough + OFFSETS["up_peak"],
            "end_time": trough + OFFSETS["end"],
            "stage": stage,
        }
    )


def _synapses(waves: pl.DataFrame, nan_gap: tuple[float, float] | None = (50.0, 52.0)) -> xr.DataArray:
    """Six synapses: three on DMD 1 (deep, flat) and three on DMD 2 (superficial, bump after every trough)."""
    t = np.arange(int(DUR_S * FS)) / FS
    x = np.zeros((6, t.size))
    for trough in waves["trough_time"].to_numpy():
        x[3:] += BUMP_AMP * np.exp(-((t - trough - BUMP_LAG) ** 2) / (2 * 0.02**2))
    if nan_gap is not None:
        x[:, (t >= nan_gap[0]) & (t < nan_gap[1])] = np.nan
    da = xr.DataArray(
        x,
        dims=("syn_id", "time"),
        coords={
            "syn_id": [f"1-{i}" for i in range(3)] + [f"2-{i}" for i in range(3)],
            "time": t,
            "dmd": ("syn_id", [1, 1, 1, 2, 2, 2]),
            "dmd-depth": ("syn_id", [235, 235, 235, 160, 160, 160]),
            "soma-ID": ("syn_id", ["soma1"] * 6),
            "dend-ID": ("syn_id", ["A-1", "A-1", "A-1", "B-1", "B-1", "B-1"]),
            "pos": ("syn_id", [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
        },
        attrs={"units": "SNR"},
    )
    return da


def _eeg(waves: pl.DataFrame) -> xr.DataArray:
    t = np.arange(int(DUR_S * FS_EEG)) / FS_EEG
    x = np.zeros(t.size)
    for trough in waves["trough_time"].to_numpy():
        x += EEG_AMP * np.exp(-((t - trough) ** 2) / (2 * 0.1**2))
    return xr.DataArray(x, dims=("time",), coords={"time": t}, attrs={"fs": FS_EEG, "units": "uV"})


def _soma(waves: pl.DataFrame, lag: float, amp: float) -> xr.DataArray:
    t = np.arange(int(DUR_S * 80.0)) / 80.0
    x = np.zeros(t.size)
    for trough in waves["trough_time"].to_numpy():
        x += amp * np.exp(-((t - trough - lag) ** 2) / (2 * 0.05**2))
    return xr.DataArray(x[None, :], dims=("soma_id", "time"), coords={"soma_id": ["soma1"], "time": t})


class _FakeSlowWaves:
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df


def test_superficial_dmd_uses_dmd_depth() -> None:
    da = _synapses(_waves())
    assert superficial_dmd(da) == 2
    assert superficial_dmd(da.drop_vars("dmd-depth")) == 1


def test_peths_recover_bumps_landmarks_and_stage_filter() -> None:
    waves = _waves()
    da = _synapses(waves)
    P = wave_locked_peths(
        da, _FakeSlowWaves(waves), _eeg(waves), _soma(waves, 0.3, 1.0), _soma(waves, 0.0, 0.5),
        align="trough", stages="NREM",
    )
    n_nrem = waves.filter(pl.col("stage") == "NREM").height
    assert P["n_waves"] == n_nrem
    assert P["n_by_stage"] == {"NREM": n_nrem}
    assert P["align"] == "trough" and P["landmarks"]["trough"] == 0.0
    for lm, off in OFFSETS.items():
        assert P["landmarks"][lm] == pytest.approx(off, abs=1e-9)
    assert P["landmark_summary"].height == 5
    assert P["eeg_sem"].shape == P["eeg"].shape and np.all(np.isfinite(P["eeg_sem"]))
    assert P["soma_ca_sem"].shape == P["soma_ca"].shape
    i_bump = int(np.argmin(np.abs(P["lags"] - BUMP_LAG)))
    assert np.allclose(P["syn"][3:, i_bump], BUMP_AMP, atol=0.05)
    assert np.allclose(P["syn"][:3, i_bump], 0.0, atol=1e-9)
    assert P["population"][i_bump] == pytest.approx(3 * BUMP_AMP, abs=0.15)
    i0 = int(np.argmin(np.abs(P["lags"])))
    assert P["eeg"][i0] == pytest.approx(EEG_AMP, abs=1.0)
    assert P["soma_ca"][int(np.argmin(np.abs(P["lags"] - 0.3)))] == pytest.approx(1.0, abs=0.05)
    assert P["spike_prob"][i0] == pytest.approx(0.5, abs=0.05)
    assert np.all(np.isfinite(P["syn"]))          # the NaN gap only removes samples, never poisons a lag
    assert P["dmd_first"] == 2
    # recording mean is per synapse over observed samples: the bump synapses carry a small positive mean
    assert np.all(P["syn_mean"][3:] > 0) and np.allclose(P["syn_mean"][:3], 0.0)


def test_peths_start_alignment_shifts_everything() -> None:
    waves = _waves()
    da = _synapses(waves, nan_gap=None)
    # the default is the trough ...
    assert wave_locked_peths(da, waves, _eeg(waves))["align"] == "trough"
    # ... and align="start" re-centres every trace on the start zero-crossing
    P = wave_locked_peths(da, waves, _eeg(waves), _soma(waves, 0.3, 1.0), align="start")
    assert P["align"] == "start" and P["landmarks"]["start"] == 0.0
    # every landmark offset is now measured from the start zero-crossing
    assert P["landmarks"]["trough"] == pytest.approx(-OFFSETS["start"], abs=1e-9)
    for lm in ("mid", "up_peak", "end"):
        assert P["landmarks"][lm] == pytest.approx(OFFSETS[lm] - OFFSETS["start"], abs=1e-9)
    # ... and so are the synaptic / EEG / soma PETHs
    i_bump = int(np.argmin(np.abs(P["lags"] - (BUMP_LAG - OFFSETS["start"]))))
    assert np.allclose(P["syn"][3:, i_bump], BUMP_AMP, atol=0.05)
    i_tr = int(np.argmin(np.abs(P["lags"] + OFFSETS["start"])))
    assert P["eeg"][i_tr] == pytest.approx(EEG_AMP, abs=1.0)
    assert P["soma_ca"][int(np.argmin(np.abs(P["lags"] - (0.3 - OFFSETS["start"]))))] == pytest.approx(1.0, abs=0.05)
    with pytest.raises(ValueError):
        wave_locked_peths(da, waves, _eeg(waves), align="mid")


def test_peths_mean_population_and_no_stage_column() -> None:
    waves = _waves().drop("stage")
    da = _synapses(waves, nan_gap=None)
    P = wave_locked_peths(da, waves, _eeg(waves), align="trough", population="mean")
    i_bump = int(np.argmin(np.abs(P["lags"] - BUMP_LAG)))
    assert P["population"][i_bump] == pytest.approx(BUMP_AMP / 2, abs=0.05)
    assert P["n_by_stage"] == {}
    assert P["soma_ca"] is None and P["spike_prob"] is None
    with pytest.raises(ValueError):
        wave_locked_peths(da, waves, _eeg(waves), stages="NREM")


def test_overview_figure_builds_with_and_without_soma() -> None:
    waves = _waves()
    da = _synapses(waves)
    fig, axes = wave_locked_overview(da, waves, _eeg(waves), _soma(waves, 0.3, 1.0), _soma(waves, 0.0, 0.5), stages="NREM")
    assert set(axes) == {"eeg", "heatmap", "population", "soma_ca", "spike_prob"}
    assert axes["spike_prob"].get_xlabel() == "s from wave trough"
    assert len(axes["heatmap"].images) == 1
    assert axes["heatmap"].images[0].get_array().shape == (6, P_LAGS)
    assert "DMD 2 on top" in axes["heatmap"].get_ylabel()
    # the EEG panel labels the zero-crossing landmarks with their phase
    assert {t.get_text() for t in axes["eeg"].texts} >= {"0", "π", "2π"}
    # every panel carries the 5 landmark lines; the trace panels one data line on top of them
    assert len(axes["soma_ca"].lines) == 6 and len(axes["spike_prob"].lines) == 6 and len(axes["population"].lines) == 6
    matplotlib.pyplot.close(fig)
    fig, axes = wave_locked_overview(da, waves, _eeg(waves), align="start", heatmap_units="z", title="custom")
    assert axes["eeg"].get_title() == "custom"
    assert axes["spike_prob"].get_xlabel() == "s from wave start"
    assert len(axes["soma_ca"].lines) == 5 and len(axes["spike_prob"].lines) == 5      # landmark lines only
    matplotlib.pyplot.close(fig)


P_LAGS = np.arange(-1.5, 1.5001, 0.01).size
