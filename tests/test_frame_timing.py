"""Regression checks for dropped-trial rate inflation and exact trial boundaries."""

from pathlib import Path

import numpy as np
import pytest
import zarr

import wisco_slap as wis
from wisco_slap.pns import aligned_frames as frames
from wisco_slap.pns._frame_timing import (
    estimate_frame_rate,
    fill_dropped_trial_counts,
    scopex_frame_times,
)


def test_dropped_trials_do_not_inflate_frame_rate() -> None:
    assert estimate_frame_rate(200, [800, 800, 800], [2000, 2000, 2000]) == 80
    assert estimate_frame_rate(200, [800, 800, 800], [2000, 0, 2000]) == 80
    assert estimate_frame_rate(200, [800, 800], [0, 2000]) == 80


def test_rate_uses_corresponding_trial_lengths() -> None:
    # The dropped trial's TIFF length is irrelevant to the rate estimate.
    assert estimate_frame_rate(200, [800, 1200, 400], [2000, 0, 1000]) == 80


def test_all_dropped_rate_estimate_fails() -> None:
    with pytest.raises(ValueError, match="surviving"):
        estimate_frame_rate(200, [800, 800], [0, 0])


def test_scopex_clock_retains_leading_middle_and_trailing_dropped_trials() -> None:
    time, rate = scopex_frame_times([4] * 5, [0, 10, 0, 10, 0], 10, [1] * 5)
    np.testing.assert_array_equal(time, np.arange(20) / 4)
    assert rate == 4


def test_rounding_and_partial_trials_do_not_accumulate_boundary_drift() -> None:
    time, _ = scopex_frame_times([799, 801, 400], [2000, 2000, 1000], 200, [1] * 3)
    assert time[0] == 0
    assert time[799] == 10
    assert time[1600] == 20
    assert time[-1] == pytest.approx(24.9875)


def test_multiepoch_placeholders_use_their_own_template_and_offset() -> None:
    counts = [0, 2000, 0, 1000]
    epochs = [1, 1, 2, 2]
    np.testing.assert_array_equal(
        fill_dropped_trial_counts(counts, epochs), [2000, 2000, 1000, 1000]
    )
    time, _ = scopex_frame_times(
        [800, 800, 400, 400], counts, 200, epochs, {1: 0, 2: 60}
    )
    np.testing.assert_array_equal(time[[0, 800, 1600, 2000]], [0, 10, 60, 65])
    assert time[-1] == pytest.approx(69.9875)


@pytest.mark.parametrize(
    ("counts", "epochs", "offsets", "message"),
    [
        ([10, 0], [1, 2], {1: 0, 2: 5}, "no analyzed trial"),
        ([10, 10], [1, 2], None, "explicit epoch offsets"),
        ([10, 10], [1, 2], {1: 0, 2: 0.1}, "nonmonotonic"),
    ],
)
def test_ambiguous_or_overlapping_epochs_fail(counts, epochs, offsets, message) -> None:
    with pytest.raises(ValueError, match=message):
        scopex_frame_times([4, 4], counts, 10, epochs, offsets)


def test_missing_tiff_trial_still_advances_scopex_time() -> None:
    time, _ = scopex_frame_times([4, 0, 4], [10, 0, 10], 10, [1] * 3)
    np.testing.assert_array_equal(time, [0, 0.25, 0.5, 0.75, 2, 2.25, 2.5, 2.75])


@pytest.mark.parametrize("max_trials", [None, 2])
def test_cache_build_uses_correct_rate_and_refuses_old_clock(
    tmp_path: Path, monkeypatch, max_trials
) -> None:
    context = frames.AcqContext(
        exsum_path=tmp_path / "fake.mat",
        identity="clock-test",
        acquisition={},
        channel_schema={
            "signalByRawChannel": [["calcium"]],
            "rawChannelNumbers": [[1.0]],
        },
        number_of_channels=1,
        n_trials=3,
        analyze_hz=20,
        dmds=(1,),
        support={1: np.ones((1, 1), dtype=bool)},
    )
    tt = {
        "fnRegDS": np.array([["a"], ["b"], ["c"]]),
        "fnAdata": np.array([["a.mat"], ["b.mat"], ["c.mat"]]),
        "trialStartTimeInferred": np.arange(3) / 86400,
    }
    monkeypatch.setattr(
        frames.spy.xsum,
        "deref_h5_any",
        lambda path, *_: tt if path.endswith("trialTable") else np.zeros((3, 2)),
    )
    monkeypatch.setattr(frames, "trial_epochs", lambda *_: np.ones(3, dtype=int))
    monkeypatch.setattr(wis.meta.get, "epoch_offsets", lambda *_: None)
    monkeypatch.setattr(frames, "resolve_raw_dir", lambda *_: tmp_path)
    monkeypatch.setattr(frames, "frame_count", lambda *_: 8)
    monkeypatch.setattr(frames, "per_trial_sample_counts", lambda *_: [20, 0, 20])
    monkeypatch.setattr(
        frames, "_sample_trial_frames", lambda *_: np.ones((8, 1), dtype=np.float32)
    )
    monkeypatch.setattr(
        frames, "load_discard_mask", lambda _, d, t, nf: np.full(nf, t == 1)
    )
    monkeypatch.setattr(frames, "registration_failed", lambda *_: False)
    path = tmp_path / "cache.zarr"
    frames.build_frame_cache(
        context, 1, path, signal="calcium", workers=1, max_trials=max_trials
    )
    cache = zarr.open_group(str(path), mode="a")
    assert cache.attrs["sample_rate_hz"] == 8
    np.testing.assert_allclose(
        cache["time_seconds"][:], np.arange(cache["time_seconds"].size) / 8
    )
    assert not cache["quality_valid"][8:16].any()
    cache.attrs["schema_version"] = 1
    with pytest.raises(ValueError, match="corrected dropped-trial clock"):
        frames.build_frame_cache(
            context, 1, path, signal="calcium", workers=1, max_trials=max_trials
        )
    with pytest.raises(ValueError, match="corrected dropped-trial clock"):
        frames.validate_frame_cache(context, 1, path)
