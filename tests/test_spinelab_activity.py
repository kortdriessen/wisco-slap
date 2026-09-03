"""Tests for Wright-style activity detection (spine-size measurement)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wisco_slap.spinelab.activity import detect_active_frames


def test_detector_marks_large_transient_and_not_baseline() -> None:
    rng = np.random.default_rng(5)
    trace = 100 + rng.normal(0, 1, 1600)
    trace[600:680] += 40
    detected = detect_active_frames(trace, sample_rate_hz=80)
    assert detected.baseline == pytest.approx(100, abs=2)
    assert detected.active[620:660].mean() > 0.95
    assert detected.active[:400].mean() < 0.02
    assert detected.high_threshold >= 0.2


def test_detector_catches_brief_iglusnfr4f_like_transients() -> None:
    """Brief 4f events (~125 ms at 80 Hz) must not be smoothed away.

    Regression test for an earlier double-smoothing (0.5-s boxcar under the
    1-s Savitzky-Golay), which attenuated these transients below the 0.2
    minimum-amplitude floor; the single Savitzky-Golay pass keeps them above.
    """

    rng = np.random.default_rng(11)
    trace = 100 + rng.normal(0, 1, 4000)
    for start in (500, 1500, 2500):
        trace[start : start + 10] += 55
    detected = detect_active_frames(trace, sample_rate_hz=80)
    for start in (500, 1500, 2500):
        assert detected.active[start : start + 10].any()
    assert detected.active[:400].mean() < 0.05


def test_zero_baseline_trace_fails_closed() -> None:
    result = detect_active_frames(np.zeros(100))
    assert not result.active.any()
    assert math.isnan(result.noise)
