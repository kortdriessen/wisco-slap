from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import polars as pl
import pytest
import xarray as xr

matplotlib.use("Agg")

import wisco_slap as wis  # noqa: E402
from wisco_slap.viz.overview_and_summary_plots.acq_level._acq_within_exp import (  # noqa: E402
    _robust_norm01,
    _yscale_line,
    acq_within_exp_path,
    acq_within_exp_plot,
)

# Synthetic experiment: two 100 s sync blocks, one acquisition in each.
_EPHYS_FS = 200.0
_BLOCK_S = 100.0
_BLOCKS = {
    1: {
        "ephys_start": "2026-01-01 10:00:00",
        "ephys_end": "2026-01-01 10:01:40",
        "corrupt": False,
    },
    2: {
        "ephys_start": "2026-01-01 10:05:00",
        "ephys_end": "2026-01-01 10:06:40",
        "corrupt": False,
    },
}
_ACQS = {
    "loc_A--acq_1": {
        "sync_block": 1,
        "dat_start": "2026-01-01 10:00:10",
        "ephys_offset": 10.0,
    },
    "loc_B--acq_1": {
        "sync_block": 2,
        "dat_start": "2026-01-01 10:05:20",
        "ephys_offset": 20.0,
    },
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


def _eye(subject, exp, sb):
    t = np.arange(0, _BLOCK_S, 0.02)
    return pl.DataFrame({
        "time": t,
        "diameter": 50 + 20 * np.sin(t / 5),
        "motion": np.abs(np.cos(t)),
        "pup_likelihood": np.where(t < 1.0, 0.1, 0.99),  # first second is blanked
    })


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
        start_time=start,
        end_time=start + 60.0,
        duration=60.0,
        sync_block=entry["sync_block"],
    )


def _stub_experiment(monkeypatch: pytest.MonkeyPatch, tmp_path, **overrides) -> None:
    """Point the plot at a fully synthetic two-block experiment rooted in tmp_path."""
    si = {"mouse": {"exp_1": {"sync_blocks": _BLOCKS, "acquisitions": _ACQS}}}
    monkeypatch.setattr(wis.meta.get, "sync_info", lambda: si)
    monkeypatch.setattr(wis.get, "ephys", overrides.get("ephys", _ephys))
    monkeypatch.setattr(wis.get, "eye_metrics", overrides.get("eye_metrics", _eye))
    monkeypatch.setattr(wis.get, "whisk_df", overrides.get("whisk_df", _whisk))
    monkeypatch.setattr(
        wis.get, "sb_bout_hypno", overrides.get("sb_bout_hypno", _hypno)
    )
    monkeypatch.setattr(
        wis.meta, "get_acq_timing", overrides.get("get_acq_timing", _timing)
    )
    monkeypatch.setattr(wis.defs, "annotation_root", str(tmp_path))


def _plot(**kwargs):
    return acq_within_exp_plot("mouse", "exp_1", "loc_A", "acq_1", dpi=20, **kwargs)


def test_robust_norm01_maps_percentiles_to_unit_interval_and_keeps_nans() -> None:
    v = np.concatenate([np.linspace(0, 10, 101), [1000.0, np.nan]])  # one huge outlier
    norm, lo, hi = _robust_norm01(v, 1, 99)
    assert 0 <= lo < hi <= 10.2
    assert np.isnan(norm[-1])
    assert norm[-2] == 1.0  # outlier clipped, not stretching the scale
    finite = norm[np.isfinite(norm)]
    assert finite.min() == 0.0 and finite.max() == 1.0


@pytest.mark.parametrize("values", [np.full(10, np.nan), np.full(10, 3.0)])
def test_robust_norm01_rejects_degenerate_input(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="cannot normalise"):
        _robust_norm01(values, 1, 99)


def test_yscale_line_flags_spiky_data_only() -> None:
    quiet = np.linspace(-1, 1, 1000)
    assert "CHECK" not in _yscale_line("q", quiet, -1, 1, "test")
    spiky = np.concatenate([quiet, [500.0]])  # raw extent >> shown range
    assert "CHECK" in _yscale_line("s", spiky, -1, 1, "test")
    assert "no finite data" in _yscale_line("e", np.full(5, np.nan), 0, 1, "test")


def test_acq_within_exp_plot_layout_highlight_and_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    _stub_experiment(monkeypatch, tmp_path)
    out = acq_within_exp_path("mouse", "exp_1", "loc_A", "acq_1")
    assert not out.exists()

    fig, ax = _plot()
    # blocks, acquisitions, EEG_, EEG_ spectrogram, loal, pupil, pupil motion, whisking
    assert ax.shape == (8,)
    assert out.exists()
    first_mtime = out.stat().st_mtime_ns

    # The requested acquisition is filled, the other one is an outline only.
    bars = [p for p in ax[1].patches if p.get_height() == pytest.approx(0.8)]
    assert len(bars) == 2
    filled = [b for b in bars if b.get_facecolor()[3] > 0]
    assert len(filled) == 1
    labels = {t.get_text() for t in ax[1].texts}
    assert any(s.startswith("loc_A--acq_1") for s in labels)
    assert any(s.startswith("loc_B--acq_1") for s in labels)
    # Behavioural panels are on the fixed normalised axis.
    for a in ax[5:]:
        assert a.get_ylim() == pytest.approx((-0.05, 1.05))
    report = capsys.readouterr().out
    assert "every component was found and plotted" in report
    assert "y-scale report" in report
    fig.clear()

    # Existing file + overwrite=False -> file untouched; overwrite=True -> rewritten.
    fig, _ = _plot()
    assert out.stat().st_mtime_ns == first_mtime
    fig.clear()
    fig, _ = _plot(overwrite=True)
    assert out.stat().st_mtime_ns > first_mtime
    fig.clear()


def test_acq_within_exp_plot_skips_missing_components_and_reports_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    def no_whisk(subject, exp, sb):
        raise FileNotFoundError("whisk_df.parquet")

    def no_hypno_sb2(subject, exp, sb, version="model"):
        return None if sb == 2 else _hypno(subject, exp, sb, version)

    _stub_experiment(
        monkeypatch, tmp_path, whisk_df=no_whisk, sb_bout_hypno=no_hypno_sb2
    )
    fig, ax = _plot()
    report = capsys.readouterr().out
    assert "sync_block-1: whisk_df — FileNotFoundError" in report
    assert "sync_block-2: whisk_df — FileNotFoundError" in report
    assert "sync_block-2: hypnogram — FileNotFoundError" in report
    assert "sync_block-1: hypnogram" not in report
    # Whisking panel still exists, just empty (no lines drawn).
    assert len(ax[-1].lines) == 0
    fig.clear()


def test_acq_within_exp_plot_rejects_unknown_acq(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _stub_experiment(monkeypatch, tmp_path)
    with pytest.raises(KeyError, match="loc_Z--acq_9"):
        acq_within_exp_plot("mouse", "exp_1", "loc_Z", "acq_9")


def test_acq_within_exp_is_registered_in_the_acq_level_set() -> None:
    names = [s.name for s in wis.viz.overview_and_summary_plots.plot_specs("acq_level")]
    assert "acq_within_exp" in names
    assert wis.viz.acq_within_exp_plot is acq_within_exp_plot
