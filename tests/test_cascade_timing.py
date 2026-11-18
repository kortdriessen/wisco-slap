"""Regression coverage for CASCADE's physical clock and discrete-bin mapping."""

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from wisco_slap.scope.cascade import (
    _discrete_indices,
    _prepare_for_cascade,
    run_cascade_detection,
)

MODEL = "Global_EXC_30Hz_smoothing25ms"
MODEL_FOLDER = "/home/driessen2/CascadeTorch/Pretrained_models"


def test_resampling_preserves_noninteger_source_clock():
    fs = 199.999206
    t = 1500.0 + np.arange(720001) / fs
    y, td, _, actual_fs = _prepare_for_cascade(t, np.zeros(t.size))
    assert actual_fs == pytest.approx(fs, rel=1e-9)
    assert td[-1] <= t[-1] + 1e-9
    assert td[-1] == pytest.approx(t[0] + (len(y) - 1) / (fs * 3 / 20), abs=1e-6)
    # Exactly 30-Hz timestamps would finish more than 14 ms early here.
    assert td[-1] - (t[0] + (len(y) - 1) / 30) > 0.014


def test_resampling_rejects_unsegmented_timestamp_gaps():
    t = np.r_[np.arange(1000) / 200, 20 + np.arange(1000) / 200]
    with pytest.raises(ValueError, match="contiguous"):
        _prepare_for_cascade(t, np.zeros(t.size))


def test_resampling_all_missing_retains_missing_data():
    y, td, mask, _ = _prepare_for_cascade(np.arange(2000) / 200, np.full(2000, np.nan))
    assert np.isnan(y).all()
    assert mask.all()
    assert y.size == td.size


def test_discrete_supports_do_not_join_or_shift():
    p = np.full(300, np.nan)
    p[32:100] = 0
    p[140:268] = 0
    p[60], p[200] = 1, 1
    seen = []

    def legacy_infer(values):
        assert np.isfinite(values).all()
        seen.append(values.shape[1])
        return values, [np.array([np.argmax(values) - 1])]

    np.testing.assert_array_equal(_discrete_indices(p, legacy_infer), [60, 200])
    assert seen == [68, 128]


def test_installed_discrete_fitter_aligns_known_spike():
    fitter = pytest.importorskip(
        "cascade2p.utils_discrete_spikes"
    ).infer_discrete_spikes
    from pathlib import Path

    if not (Path(MODEL_FOLDER) / MODEL / "config.yaml").exists():
        pytest.skip("Local CASCADE model configuration unavailable")
    p = np.zeros(240)
    p[120] = 1
    p = gaussian_filter1d(p, sigma=0.75)
    p[:32] = p[-32:] = np.nan
    p[60:90] = np.nan
    state = np.random.get_state()
    try:
        np.random.seed(345)
        idx = _discrete_indices(
            p, fitter, model_name=MODEL, model_folder=MODEL_FOLDER, verbosity=0
        )
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(idx, [120])


def test_wrapper_preserves_two_timestamp_runs(monkeypatch):
    cascade = pytest.importorskip("cascade2p.cascade")
    discrete = pytest.importorskip("cascade2p.utils_discrete_spikes")
    config = pytest.importorskip("cascade2p.config")
    fs = 199.999206
    t = np.r_[500 + np.arange(2000) / fs, 530 + np.arange(2000) / fs]
    da = xr.DataArray(
        np.zeros(t.size), dims="time", coords={"time": t, "soma_id": "soma1"}
    )
    calls = []

    def predict(model_name, values, **kwargs):
        calls.append(values.shape[1])
        p = np.zeros_like(values)
        p[0, values.shape[1] // 2] = 1
        p[:, :32] = p[:, -32:] = np.nan
        return p

    def infer(values, **kwargs):
        return values, [np.array([np.argmax(values) - 1])]

    monkeypatch.setattr(cascade, "predict", predict)
    monkeypatch.setattr(discrete, "infer_discrete_spikes", infer)
    monkeypatch.setattr(config, "read_config", lambda _: {"windowsize": 64})
    pred, spikes = run_cascade_detection(
        da, model_folder="/tmp", device="cpu", download_if_missing=False, verbosity=0
    )
    assert len(calls) == 2
    assert pred.attrs["n_timestamp_runs"] == 2
    assert pred.attrs["fs"] == pytest.approx(fs * 3 / 20, rel=1e-9)
    assert pred.attrs["model_fs"] == 30
    assert spikes.height == 2
    np.testing.assert_allclose(spikes["time"], [505.00001985, 535.00001985], atol=1e-7)
    assert not ((pred.time.values > t[1999]) & (pred.time.values < 530)).any()
