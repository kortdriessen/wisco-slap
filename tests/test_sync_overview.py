from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import wisco_slap as wis  # noqa: E402
from wisco_slap.viz.overview_and_summary_plots.acq_level._sync_overview import (  # noqa: E402
    SYNC_FS,
    _has_edge_near,
    _zoom_window,
    basic_sync_summary_path,
    basic_sync_summary_plot,
)

# Synthetic SYNC layout (60 s at 5 kHz): ephys line steps up at 2.1 s, the
# scope trigger goes high at 3.6 s and stays high (> 10 s, so it is a valid
# acquisition segment). The "true" ephys_offset is therefore 1.5 s.
_N = 60 * SYNC_FS
_EPHYS_START = 2 * SYNC_FS + 500
_SCOPE_UP = _EPHYS_START + int(1.5 * SYNC_FS)
_EPHYS_START_WALL = "2026-01-01 10:00:00"
_DAT_START_WALL = "2026-01-01 10:00:01.5"  # wall-clock gap 1.5 s


@pytest.mark.parametrize(
    ("center", "half", "n", "expected"),
    [
        (1000, 300, 5000, (700, 1300, 300)),  # interior: symmetric window
        (100, 300, 5000, (0, 400, 100)),  # clipped at left: marker shifts left
        (4900, 300, 5000, (4600, 5000, 300)),  # clipped at right: marker unchanged
        (0, 300, 5000, (0, 300, 0)),  # center at first sample
        (-2000, 300, 5000, (0, 0, -2000)),  # entirely before the array: empty
        (9000, 300, 5000, (5000, 5000, 4000)),  # entirely after the array: empty
    ],
)
def test_zoom_window_clips_to_array_and_tracks_center(
    center: int, half: int, n: int, expected: tuple[int, int, int]
) -> None:
    assert _zoom_window(center, half, n) == expected


def test_has_edge_near_detects_small_step_before_large_activity() -> None:
    # SYNC-style ephys copy: quiet ~0.004 V baseline, a ~0.1 V step at 10_000,
    # then large physiological signal that dominates the channel's range.
    rng = np.random.default_rng(0)
    sig = np.full(50_000, 0.004) + rng.normal(0, 0.0005, 50_000)
    sig[10_000:] += 0.1
    sig[20_000:] += rng.normal(0, 0.3, 30_000)
    assert _has_edge_near(sig, 10_000, tol=3)
    assert _has_edge_near(sig, 10_002, tol=3)  # within tolerance
    assert not _has_edge_near(sig, 5_000, tol=3)  # flat baseline
    assert not _has_edge_near(sig, 30_000, tol=3)  # inside noisy activity
    assert not _has_edge_near(sig, -5, tol=3)  # out of range
    assert not _has_edge_near(np.ones(1000), 500, tol=3)  # constant line


def _preview_then_acquisition(preview_s: float) -> np.ndarray:
    """Trigger toggling (duty 0.6) for ``preview_s`` after _SCOPE_UP, then solid."""
    scope = np.zeros(_N)
    scope[_SCOPE_UP:] = 1.0
    n_prev = int(preview_s * SYNC_FS)
    preview = scope[_SCOPE_UP : _SCOPE_UP + n_prev].reshape(-1, 10)
    preview[:, 6:] = 0.0
    return scope


def _stub_sync_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    ephys_offset: float = 1.5,
    ephys: np.ndarray | None = None,
    scope: np.ndarray | None = None,
    stored_ephys_start: int | None = None,
    with_ratio: bool = False,
) -> None:
    """Point the plot at synthetic SYNC channels + metadata rooted in tmp_path."""
    if ephys is None:
        ephys = np.zeros(_N)
        ephys[_EPHYS_START:] = 1.0
    if scope is None:
        scope = np.zeros(_N)
        scope[_SCOPE_UP:] = 1.0
    block = {"ephys_start": _EPHYS_START_WALL}
    if stored_ephys_start is not None:
        block["ephys_start_sample"] = stored_ephys_start
    acq = {
        "sync_block": 1,
        "dat_start": _DAT_START_WALL,
        "ephys_offset": ephys_offset,
        "n_epochs": 1,
    }
    if with_ratio:  # ephys_offset is TDT seconds; the raw vDaq value is separate
        ratio = 0.9999
        acq["ephys_offset"] = ephys_offset / ratio
        acq["ephys_offset_uncorrected"] = ephys_offset
        acq["clock_ratio_applied"] = ratio
    si = {
        "mouse": {
            "exp_1": {"sync_blocks": {1: block}, "acquisitions": {"loc_A--acq_1": acq}}
        }
    }
    monkeypatch.setattr(wis.meta.get, "sync_info", lambda: si)
    monkeypatch.setattr(
        wis.meta.sync, "load_sync_block", lambda subject, exp, sb: (scope, ephys)
    )
    monkeypatch.setattr(wis.defs, "annotation_root", str(tmp_path))


def _plot(**kwargs):
    return basic_sync_summary_plot("mouse", "exp_1", "loc_A", "acq_1", dpi=30, **kwargs)


def test_basic_sync_summary_plot_saves_then_respects_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _stub_sync_data(monkeypatch, tmp_path)
    out = basic_sync_summary_path("mouse", "exp_1", "loc_A", "acq_1")
    assert not out.exists()

    fig, ax = _plot()
    assert ax.shape == (7,)
    assert out.exists()
    first_mtime = out.stat().st_mtime_ns
    fig.clear()

    # Existing file + overwrite=False -> figure still returned, file untouched.
    fig, _ = _plot()
    assert out.stat().st_mtime_ns == first_mtime
    fig.clear()

    # overwrite=True -> rewritten.
    fig, _ = _plot(overwrite=True)
    assert out.stat().st_mtime_ns > first_mtime
    fig.clear()


def test_basic_sync_summary_plot_recovers_starts_from_sync_info(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    _stub_sync_data(monkeypatch, tmp_path)
    fig, ax = _plot()
    # Worked backwards from the acquisition onset: acq start = onset,
    # ephys start = onset - offset, which coincides with the ephys step.
    assert f"acq_start_sample = {_SCOPE_UP} " in ax[3].get_title()
    assert f"ephys_start_sample = {_EPHYS_START} " in ax[0].get_title()
    assert "ephys edge at sample: yes" in ax[0].get_title()
    assert f"auto-detector: {_EPHYS_START}" in ax[0].get_title()
    assert "onset: +0.000 s" in ax[3].get_title()
    assert "(Δ +0.0 s)" in ax[3].get_title()
    assert "duty in 10 s after acq start = 1.00" in ax[6].get_title()
    # Wide views: ephys 0 -> ephys_start + 10 s, scope +/- 60 s (absolute samples).
    assert "0 → 12.1 s" in ax[1].get_title()
    assert ax[1].lines[0].get_xdata()[-1] == _EPHYS_START + 10 * SYNC_FS - 1
    assert "± 60 s" in ax[4].get_title()
    assert ax[4].lines[0].get_xdata()[0] == 0  # clipped at the file start
    assert ax[4].lines[0].get_xdata()[-1] == min(_N, _SCOPE_UP + 60 * SYNC_FS) - 1
    assert len(ax[4].lines) == 3  # trace + green + orange markers
    assert "WARNING" not in capsys.readouterr().out
    fig.clear()


def test_basic_sync_summary_plot_uses_stored_ephys_start_and_uncorrected_offset(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    _stub_sync_data(
        monkeypatch, tmp_path, stored_ephys_start=_EPHYS_START, with_ratio=True
    )
    fig, ax = _plot()
    assert "stored in sync_info" in ax[0].get_title()
    assert f"ephys_start_sample = {_EPHYS_START} " in ax[0].get_title()
    # Samples come from the raw vDaq offset (1.5 s), not the TDT-corrected one.
    assert f"acq_start_sample = {_SCOPE_UP} " in ax[3].get_title()
    assert "vDaq 1.5000 s" in ax[3].get_title()
    assert "WARNING" not in capsys.readouterr().out
    fig.clear()


def test_basic_sync_summary_plot_infers_manual_override_when_detector_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    # Like alnilam/exp_1: the ephys line is already high at sample 0 (so the
    # 3 x max(first 2 s) detector cannot fire), drops, then rises again at the
    # sample that was passed to update_sync_info as a manual override.
    ephys = np.zeros(_N)
    ephys[: 1 * SYNC_FS] = 1.0
    ephys[_EPHYS_START:] = 1.0
    _stub_sync_data(monkeypatch, tmp_path, ephys=ephys)
    fig, ax = _plot()
    assert f"ephys_start_sample = {_EPHYS_START} " in ax[0].get_title()
    assert "inferred" in ax[0].get_title()
    assert "auto-detector: not detected" in ax[0].get_title()
    assert "ephys edge at sample: yes" in ax[0].get_title()
    assert f"acq_start_sample = {_SCOPE_UP} " in ax[3].get_title()
    assert "WARNING" not in capsys.readouterr().out
    fig.clear()


def test_basic_sync_summary_plot_flags_bad_offset(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    # Offset wildly inconsistent with the data: the inferred ephys start lands
    # far outside the file, and the offset disagrees with the wall-clock gap.
    _stub_sync_data(monkeypatch, tmp_path, ephys_offset=-100.0)
    fig, ax = _plot()
    assert "outside the SYNC file" in ax[2].get_title()
    assert len(ax[2].lines) == 0
    # ephys start beyond the file end: the wide view just shows the whole file
    assert "(0 → 60.0 s)" in ax[1].get_title()
    assert "ephys edge at sample: NO" in ax[0].get_title()
    assert "(Δ -101.5 s)" in ax[3].get_title()
    out = capsys.readouterr().out
    assert "WARNING" in out and "wall-clock gap" in out and "DISAGREES" in out
    fig.clear()


def test_basic_sync_summary_plot_flags_stored_offset_pointing_at_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    # The old matcher's failure mode: sync_info's offset points at the start of
    # a preview episode (duty 0.6) 15 s before the acquisition actually starts.
    _stub_sync_data(
        monkeypatch,
        tmp_path,
        scope=_preview_then_acquisition(15.0),
        stored_ephys_start=_EPHYS_START,
    )
    fig, ax = _plot()
    assert f"acq_start_sample = {_SCOPE_UP} " in ax[3].get_title()
    assert "onset: +15.000 s" in ax[3].get_title()
    assert "duty in 10 s after acq start = 0.60" in ax[6].get_title()
    assert len(ax[6].lines) >= 3  # duty trace + two markers
    out = capsys.readouterr().out
    assert "preview scanning" in out and "acquisition-signature onset" in out
    fig.clear()


def test_basic_sync_summary_plot_back_calculation_uses_acquisition_onset(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    # No stored ephys start + a stale (preview-start) offset: the acq start is
    # taken from the acquisition onset, so the back-calculated ephys start no
    # longer lands on the ephys edge -> flagged via the edge/detector checks.
    _stub_sync_data(monkeypatch, tmp_path, scope=_preview_then_acquisition(15.0))
    fig, ax = _plot()
    assert f"acq_start_sample = {_SCOPE_UP + 15 * SYNC_FS} " in ax[3].get_title()
    assert "ephys edge at sample: NO" in ax[0].get_title()
    assert "DISAGREES" in capsys.readouterr().out
    fig.clear()


def test_basic_sync_summary_plot_falls_back_without_scope_up_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    _stub_sync_data(monkeypatch, tmp_path, scope=np.zeros(_N))
    fig, ax = _plot()
    assert "no scope-UP window" in ax[3].get_title()
    # Falls back to detector + offset.
    assert f"ephys_start_sample = {_EPHYS_START} " in ax[0].get_title()
    assert f"acq_start_sample = {_SCOPE_UP} " in ax[3].get_title()
    assert "no scope-UP window" in capsys.readouterr().out
    fig.clear()


def test_basic_sync_summary_plot_raises_on_corrupt_sync(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _stub_sync_data(monkeypatch, tmp_path)
    monkeypatch.setattr(
        wis.meta.sync,
        "load_sync_block",
        lambda subject, exp, sb: ("Corrupt", "Corrupt"),
    )
    with pytest.raises(RuntimeError, match="corrupt"):
        _plot()


def test_basic_sync_summary_plot_back_calculation_anchors_candidates_on_ephys_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    # alnitak/exp_2-like layout: ephys starts late (20 s); a 12 s
    # acquisition-like burst at 1-13 s sits closer to the wall-clock gap
    # (1.5 s) in raw file time than the real acquisition at 21.5 s — but not
    # once the detected ephys start is subtracted, which is how the matcher
    # selects. Without anchoring the burst wins and the back-calculated ephys
    # start would be negative.
    ephys_start = 20 * SYNC_FS
    acq_onset = ephys_start + int(1.5 * SYNC_FS)
    ephys = np.zeros(_N)
    ephys[ephys_start:] = 1.0
    scope = np.zeros(_N)
    scope[1 * SYNC_FS : 13 * SYNC_FS] = 1.0
    scope[acq_onset:] = 1.0
    _stub_sync_data(monkeypatch, tmp_path, ephys=ephys, scope=scope)
    fig, ax = _plot()
    assert f"acq_start_sample = {acq_onset} " in ax[3].get_title()
    assert f"ephys_start_sample = {ephys_start} " in ax[0].get_title()
    assert "ephys edge at sample: yes" in ax[0].get_title()
    assert "WARNING" not in capsys.readouterr().out
    fig.clear()
