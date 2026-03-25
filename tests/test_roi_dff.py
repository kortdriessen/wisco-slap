import numpy as np
import xarray as xr
import xarray.testing as xrt
from slap2_py.core.xarr_summ import save_xr_to_zarr

import wisco_slap.get._get_scopex as get_scopex
from wisco_slap.scope.pro import roi_to_dff


def _make_synthetic_roi_dataarray(
    *,
    n_channels: int = 2,
    n_rois: int = 3,
    n_time: int = 600,
    fs: float = 50.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Create synthetic ROI fluorescence with slow drift, sparse transients, and NaNs."""
    rng = np.random.default_rng(20260321)
    time = np.arange(n_time, dtype=np.float64) / fs
    data = np.empty((n_channels, n_rois, n_time), dtype=np.float32)
    baseline = np.empty_like(data)

    for channel in range(n_channels):
        for roi in range(n_rois):
            scale = 80.0 + 8.0 * channel + 5.0 * roi
            drift = 4.0 * np.sin(2.0 * np.pi * time / (18.0 + roi)) + 0.25 * time
            base = scale + drift
            transients = (
                (10.0 + 2.0 * channel + roi)
                * np.exp(-0.5 * ((time - (2.0 + 0.08 * roi)) / 0.06) ** 2)
                + (16.0 + channel)
                * np.exp(-0.5 * ((time - (6.3 - 0.1 * roi)) / 0.08) ** 2)
                + (9.0 + 0.5 * roi)
                * np.exp(-0.5 * ((time - (9.4 + 0.05 * channel)) / 0.05) ** 2)
            )
            noise = rng.normal(loc=0.0, scale=0.5 + 0.1 * channel, size=n_time)
            data[channel, roi] = base + transients + noise
            baseline[channel, roi] = base

    if n_rois > 1 and n_time > 140:
        data[:, 1, 140 : min(155, n_time)] = np.nan
    if n_channels > 1 and n_rois > 2 and n_time > 330:
        data[1, 2, 330 : min(338, n_time)] = np.nan

    coords = {
        "channel": np.arange(n_channels),
        "soma_id": np.array([f"roi_{roi}" for roi in range(n_rois)]),
        "time": time,
    }
    roi_da = xr.DataArray(
        data,
        dims=["channel", "soma_id", "time"],
        coords=coords,
    )
    baseline_da = xr.DataArray(
        baseline,
        dims=["channel", "soma_id", "time"],
        coords=coords,
    )
    return roi_da, baseline_da


def test_roi_to_dff_recovers_slow_baseline_and_preserves_nans() -> None:
    roi_da, baseline_da = _make_synthetic_roi_dataarray()

    dff_da, f0_da = roi_to_dff(roi_da, trace="Fsvd", return_f0=True)

    assert dff_da.dims == roi_da.dims
    assert f0_da.dims == roi_da.dims
    assert list(dff_da.coords) == list(roi_da.coords)
    assert np.array_equal(np.isnan(dff_da.values), np.isnan(roi_da.values))
    assert np.array_equal(np.isnan(f0_da.values), np.isnan(roi_da.values))

    time = roi_da.coords["time"].values
    quiet = (
        (time < 1.5)
        | ((time > 4.0) & (time < 5.5))
        | ((time > 10.0) & (time < 11.5))
    )

    baseline_err = np.nanmedian(
        np.abs(
            f0_da.sel(channel=0, soma_id="roi_0").values[quiet]
            - baseline_da.sel(channel=0, soma_id="roi_0").values[quiet]
        )
    )
    assert baseline_err < 5.0

    quiet_dff = dff_da.sel(channel=0, soma_id="roi_0").values[quiet]
    assert abs(np.nanmedian(quiet_dff)) < 0.05
    assert float(dff_da.sel(channel=0, soma_id="roi_0").max(skipna=True)) > 0.12


def test_roi_to_dff_matches_per_trace_application() -> None:
    roi_da, _ = _make_synthetic_roi_dataarray(n_rois=2, n_time=400)

    all_at_once = roi_to_dff(roi_da, trace="Fsvd")

    assert all_at_once.dims == roi_da.dims
    assert list(all_at_once.coords["channel"].values) == list(
        roi_da.coords["channel"].values
    )
    assert list(all_at_once.coords["soma_id"].values) == list(
        roi_da.coords["soma_id"].values
    )

    for channel in roi_da.coords["channel"].values:
        for soma_id in roi_da.coords["soma_id"].values:
            per_trace = roi_to_dff(
                roi_da.sel(channel=channel, soma_id=soma_id),
                trace="Fsvd",
            )
            xrt.assert_allclose(
                all_at_once.sel(channel=channel, soma_id=soma_id),
                per_trace,
            )


def test_roi_F_return_dff_matches_manual_conversion(tmp_path, monkeypatch) -> None:
    subject = "mouse_a"
    exp = "exp_a"
    loc = "loc_i"
    acq = "acq_1"
    scopex_dir = tmp_path / subject / exp / "scopex" / f"{loc}--{acq}"
    scopex_dir.mkdir(parents=True)

    roi_fsvd, _ = _make_synthetic_roi_dataarray(n_rois=2, n_time=320)
    roi_fraw = (roi_fsvd * 1.6 + 40.0).astype(np.float32)

    save_xr_to_zarr(
        {
            "dmd_1": roi_fsvd,
            "dmd_2": roi_fsvd.isel(soma_id=slice(0, 1)),
        },
        str(scopex_dir / "ROI_Fsvd.zarr"),
    )
    save_xr_to_zarr(
        {
            "dmd_1": roi_fraw,
            "dmd_2": roi_fraw.isel(soma_id=slice(0, 1)),
        },
        str(scopex_dir / "ROI_Fraw.zarr"),
    )

    monkeypatch.setattr(get_scopex, "anmat_root", str(tmp_path))

    loaded_fsvd = get_scopex.roi_F(
        subject,
        exp,
        loc,
        acq,
        trace="Fsvd",
        channels=1,
        apply_ephys_offset=False,
    )
    loaded_fsvd_dff = get_scopex.roi_F(
        subject,
        exp,
        loc,
        acq,
        trace="Fsvd",
        channels=1,
        apply_ephys_offset=False,
        return_dFF=True,
    )
    loaded_fraw = get_scopex.roi_F(
        subject,
        exp,
        loc,
        acq,
        trace="Fraw",
        channels=1,
        apply_ephys_offset=False,
    )
    loaded_fraw_dff = get_scopex.roi_F(
        subject,
        exp,
        loc,
        acq,
        trace="Fraw",
        channels=1,
        apply_ephys_offset=False,
        return_dFF=True,
    )

    assert set(loaded_fsvd_dff) == set(loaded_fsvd)
    assert set(loaded_fraw_dff) == set(loaded_fraw)

    for key in loaded_fsvd:
        assert loaded_fsvd_dff[key].dims == loaded_fsvd[key].dims
        assert loaded_fraw_dff[key].dims == loaded_fraw[key].dims
        xrt.assert_allclose(
            loaded_fsvd_dff[key],
            roi_to_dff(loaded_fsvd[key], trace="Fsvd"),
        )
        xrt.assert_allclose(
            loaded_fraw_dff[key],
            roi_to_dff(loaded_fraw[key], trace="Fraw"),
        )
