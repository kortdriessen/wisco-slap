from __future__ import annotations
import numpy as np
import pytest
from wisco_slap.scope.event_timing import contiguous_support, reflected_jitter, count_other_events, count_partners


def test_support_splits_nan_masks_and_time_discontinuities() -> None:
    t = np.array([0., .005, .01, .015, .1, .105])
    starts, ends = contiguous_support(t, np.array([1,1,0,1,1,1], bool), fs=200)
    assert starts.tolist() == [0,3,4]
    assert ends.tolist() == [2,4,6]


def test_jitter_respects_grid_phase_and_half_open_single_frame_support() -> None:
    t = np.tile([.003, .008, .013, .103], 1000)
    lo = np.tile([.002, .002, .002, .102], 1000)
    hi = np.tile([.015, .015, .015, .104], 1000)
    out = reflected_jitter(t, lo, hi, 200., 1., np.random.default_rng(1))
    assert np.all((out >= lo) & (out < hi))
    assert np.allclose((out - t) * 200, np.round((out-t)*200))
    assert np.allclose(out[3::4], .103)
    assert len(np.unique(np.round(out[:3], 6))) > 1
    with pytest.raises(ValueError):
        reflected_jitter(np.array([1.]), np.array([0.]), np.array([1.]), 200, 1, np.random.default_rng(1))


def test_partner_counts_match_brute_force_at_boundaries_and_long_bouts() -> None:
    rng = np.random.default_rng(4)
    t = np.r_[0., .01, .01, 100001., 100001.01, rng.integers(0, 100, 100) / 200]
    b = np.r_[0,0,0,0,0,rng.integers(0,3,100)]
    s = np.r_[0,1,0,0,1,rng.integers(0,5,100)]
    expected = np.array([np.sum((b==b[i]) & (s!=s[i]) & (abs(t-t[i]) <= .01+1e-6)) for i in range(t.size)])
    assert np.array_equal(count_other_events(t,b,s,.01), expected)
    q = np.arange(t.size)[::3]
    exp = np.array([np.sum((b==b[i]) & (abs(t-t[i]) <= .01+1e-6)) for i in q])
    assert np.array_equal(count_partners(t[q],b[q],t,b,.01), exp)
    assert count_other_events(np.array([]),np.array([]),np.array([]),.01).size == 0
