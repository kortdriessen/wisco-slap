"""Tests for the shared aligned-frames ExperimentSummary readers.

Exercises the dropped-trial handling: preprocessing stores trials it dropped
as MATLAB-empty ``[0, 0]`` placeholder datasets in ``exptSummary/E`` instead
of per-trial struct groups (see slap2-py's ``read_full_trial_data_dict``,
which maps the same pattern to ``"BAD_TRIAL"``).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from wisco_slap.pns.aligned_frames import (
    load_discard_mask as _load_discard_mask,
)
from wisco_slap.pns.aligned_frames import (
    registration_failed as _registration_failed,
)


def _write_synthetic_exsum(path: Path) -> None:
    """One DMD, two trials: a normal trial and a MATLAB-empty placeholder."""

    with h5py.File(path, "w") as handle:
        summary = handle.create_group("exptSummary")
        good = handle.create_group("trial_good")
        # 20 analyze-rate samples, frames 5..9 discarded.
        discard = np.zeros((20, 1))
        discard[5:10] = 1
        good.create_dataset("discardFrames", data=discard)
        empty = handle.create_dataset(
            "trial_empty", data=np.array([0, 0], dtype=np.uint64)
        )
        references = summary.create_dataset("E", (1, 2), dtype=h5py.ref_dtype)
        references[0, 0] = good.ref
        references[0, 1] = empty.ref


def test_discard_mask_reads_and_resamples_normal_trials(tmp_path: Path) -> None:
    exsum = tmp_path / "exsum.mat"
    _write_synthetic_exsum(exsum)
    same_rate = _load_discard_mask(exsum, 1, 0, target_frames=20)
    assert same_rate.sum() == 5
    # Downsampled 20 -> 10: the discarded block survives proportionally.
    downsampled = _load_discard_mask(exsum, 1, 0, target_frames=10)
    assert downsampled.shape == (10,)
    assert 2 <= downsampled.sum() <= 3


def test_discard_mask_marks_dropped_trials_fully_discarded(tmp_path: Path) -> None:
    exsum = tmp_path / "exsum.mat"
    _write_synthetic_exsum(exsum)
    mask = _load_discard_mask(exsum, 1, 1, target_frames=16)
    assert mask.shape == (16,)
    assert mask.all()


def test_registration_flag_is_conservative_when_unreadable(tmp_path: Path) -> None:
    assert _registration_failed(tmp_path / "missing.h5") is True
    sidecar = tmp_path / "sidecar.h5"
    with h5py.File(sidecar, "w") as handle:
        handle.create_group("aData").create_dataset(
            "registrationFailed", data=np.array([[0.0]])
        )
    assert _registration_failed(sidecar) is False
    with h5py.File(tmp_path / "no_key.h5", "w") as handle:
        handle.create_group("aData")
    assert _registration_failed(tmp_path / "no_key.h5") is True
