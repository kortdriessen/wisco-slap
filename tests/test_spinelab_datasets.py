"""Tests for the pure helpers behind spine-labelling dataset generation.

The ExperimentSummary-reading orchestration is exercised end-to-end against
real acquisitions outside the test suite (per test conventions, tests here
use only synthetic inputs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wisco_slap.spinelab.datasets import (
    _binned_aligned_means,
    _bundle_id,
    _chunk_records,
)


def _chunks_for(n_trials: int, bin_trials: int) -> list[dict]:
    # 10-second trials starting every 12 seconds, single epoch, 100
    # frames per trial.
    starts = np.arange(n_trials) * 12.0 + 500.0
    return _chunk_records(
        start_seconds=starts,
        end_seconds=starts + 10.0,
        epochs=np.ones(n_trials, dtype=int),
        frame_counts=[100] * n_trials,
        bin_trials=bin_trials,
    )


def test_chunk_records_bins_trials_with_partial_tail() -> None:
    chunks = _chunks_for(n_trials=7, bin_trials=3)
    assert [c["bin_index"] for c in chunks] == [1, 2, 3]
    assert [(c["first_trial"], c["last_trial"]) for c in chunks] == [
        (1, 3),
        (4, 6),
        (7, 7),
    ]
    assert [c["n_trials"] for c in chunks] == [3, 3, 1]
    assert [c["partial_bin"] for c in chunks] == [False, False, True]
    assert [c["n_frames"] for c in chunks] == [300, 300, 100]
    # Times are re-referenced to the first trial's start.
    assert chunks[0]["start_seconds"] == 0.0
    assert chunks[0]["end_seconds"] == pytest.approx(2 * 12.0 + 10.0)
    assert chunks[1]["start_seconds"] == pytest.approx(3 * 12.0)
    assert chunks[2]["duration_seconds"] == pytest.approx(10.0)


def test_chunk_records_propagates_first_trial_epoch() -> None:
    starts = np.arange(4) * 12.0
    chunks = _chunk_records(
        start_seconds=starts,
        end_seconds=starts + 10.0,
        epochs=np.array([1, 1, 2, 2]),
        frame_counts=[100] * 4,
        bin_trials=2,
    )
    assert [c["epoch"] for c in chunks] == [1, 2]


def test_chunk_records_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        _chunk_records(
            start_seconds=np.zeros(3),
            end_seconds=np.zeros(3),
            epochs=np.ones(3, dtype=int),
            frame_counts=[100] * 2,
            bin_trials=2,
        )


def test_binned_means_apply_minimum_valid_trial_fraction() -> None:
    # 4 trials of a 1x2 image. Pixel 0 is finite in all trials; pixel 1
    # only in the first trial, below the 50% threshold of a 4-trial bin.
    per_trial = np.full((4, 1, 2), np.nan, dtype=np.float32)
    per_trial[:, 0, 0] = [1.0, 2.0, 3.0, 4.0]
    per_trial[0, 0, 1] = 7.0
    chunks = _chunks_for(n_trials=4, bin_trials=4)
    stack, counts = _binned_aligned_means(
        per_trial, chunks, minimum_valid_trial_fraction=0.5
    )
    assert stack.shape == (1, 1, 2)
    assert stack[0, 0, 0] == pytest.approx(2.5)
    assert np.isnan(stack[0, 0, 1])
    assert counts[0, 0, 0] == 4
    assert counts[0, 0, 1] == 1


def test_binned_means_keep_pixels_at_exact_threshold() -> None:
    per_trial = np.full((4, 1, 1), np.nan, dtype=np.float32)
    per_trial[:2, 0, 0] = [1.0, 3.0]  # 2 of 4 trials = exactly 50%
    chunks = _chunks_for(n_trials=4, bin_trials=4)
    stack, counts = _binned_aligned_means(
        per_trial, chunks, minimum_valid_trial_fraction=0.5
    )
    assert stack[0, 0, 0] == pytest.approx(2.0)
    assert counts[0, 0, 0] == 2


def test_bundle_id_is_stable_and_sensitive_to_bin_trials(tmp_path: Path) -> None:
    exsum = tmp_path / "SummaryLoCo-260101-000000.mat"
    exsum.write_bytes(b"not a real mat file")
    provenance = {"runIdentifier": "run-1", "ophysGit": {"commit": "abc123"}}
    first = _bundle_id(exsum, provenance, bin_trials=30)
    assert first == _bundle_id(exsum, provenance, bin_trials=30)
    assert len(first) == 20
    assert first != _bundle_id(exsum, provenance, bin_trials=20)
    # The recipe must stay byte-compatible with existing bundles: same
    # identity fields, sorted-key JSON, sha256 truncated to 20 hex chars.
    import hashlib

    stat = exsum.stat()
    expected = hashlib.sha256(
        json.dumps(
            {
                "path": str(exsum.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "run": "run-1",
                "ophys_commit": "abc123",
                "bin_trials": 30,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:20]
    assert first == expected


# ---------------------------------------------------------------------------
# Target-folder handling
# ---------------------------------------------------------------------------


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_spinelab_dataset_dir_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from wisco_slap import defs
    from wisco_slap.spinelab.datasets import spinelab_dataset_dir

    monkeypatch.setattr(defs, "anmat_root", str(tmp_path))
    assert spinelab_dataset_dir("kaus", "exp_1", "loc_E", "acq_1") == (
        tmp_path / "kaus" / "exp_1" / "spinelab" / "loc_E--acq_1"
    )


def test_prepare_target_dir_creates_missing(tmp_path: Path):
    from wisco_slap.spinelab.datasets import _prepare_target_dir

    target = tmp_path / "a" / "b" / "loc_E--acq_1"
    _prepare_target_dir(target, overwrite=False)
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_prepare_target_dir_uses_existing_empty_dir(tmp_path: Path):
    from wisco_slap.spinelab.datasets import _prepare_target_dir

    target = tmp_path / "loc_E--acq_1"
    target.mkdir()
    _prepare_target_dir(target, overwrite=False)
    assert target.is_dir()


def test_prepare_target_dir_refuses_non_empty(tmp_path: Path):
    from wisco_slap.spinelab.datasets import _prepare_target_dir

    target = tmp_path / "loc_E--acq_1"
    annotations = _touch(target / "dmd1_spine_annotations.json", "{}")
    with pytest.raises(FileExistsError, match="not empty"):
        _prepare_target_dir(target, overwrite=False)
    assert annotations.read_text() == "{}"
    # Hidden files count as content too.
    hidden_only = tmp_path / "hidden"
    _touch(hidden_only / ".DS_Store")
    with pytest.raises(FileExistsError):
        _prepare_target_dir(hidden_only, overwrite=False)


def test_prepare_target_dir_overwrite_keeps_foreign_files(tmp_path: Path):
    from wisco_slap.spinelab.datasets import _prepare_target_dir

    target = tmp_path / "loc_E--acq_1"
    generated = [
        _touch(target / "metadata.json"),
        _touch(target / "dmd1_images.npz"),
        _touch(target / "dmd2_images.npz"),
        _touch(target / "dmd1_overlays.npz"),
    ]
    kept = [
        _touch(target / "dmd1_spine_annotations.json", "{}"),
        _touch(target / "measurements" / "dmd1_spines.parquet"),
        _touch(target / "dmd1_frame_cache.zarr" / ".zarray"),
    ]
    _prepare_target_dir(target, overwrite=True)
    assert not any(p.exists() for p in generated)
    assert all(p.exists() for p in kept)


def test_prepare_target_dir_rejects_file_target(tmp_path: Path):
    from wisco_slap.spinelab.datasets import _prepare_target_dir

    target = _touch(tmp_path / "not_a_dir")
    with pytest.raises(NotADirectoryError):
        _prepare_target_dir(target, overwrite=True)


# ---------------------------------------------------------------------------
# Conforming ExSum aligned per-trial arrays to the trial table
# ---------------------------------------------------------------------------


def test_conform_aligned_trials_pads_trailing_unregistered_trials() -> None:
    from wisco_slap.spinelab.datasets import _conform_aligned_trials

    # 5 trials in the table; the summarizer skipped the last two, so the
    # saved array only holds trials 1-3.
    images = np.arange(3 * 2 * 2, dtype=np.float64).reshape(3, 2, 2) + 1.0
    registered = np.array([True, True, True, False, False])
    out = _conform_aligned_trials(images, registered)
    assert out.shape == (5, 2, 2)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[:3], images.astype(np.float32))
    assert np.isnan(out[3:]).all()


def test_conform_aligned_trials_blanks_interior_zero_fill() -> None:
    from wisco_slap.spinelab.datasets import _conform_aligned_trials

    # Trial 2 was skipped mid-acquisition: MATLAB left zeros there, which
    # must become NaN so it does not count as valid data in the bin means.
    images = np.ones((4, 1, 3), dtype=np.float64)
    images[1] = 0.0
    registered = np.array([True, False, True, True])
    out = _conform_aligned_trials(images, registered)
    assert np.isnan(out[1]).all()
    assert (out[[0, 2, 3]] == 1.0).all()


def test_conform_aligned_trials_accepts_nan_filled_unregistered_slices() -> None:
    from wisco_slap.spinelab.datasets import _conform_aligned_trials

    # perTrialActIMsAligned is preallocated NaN, so skipped trials are NaN
    # there rather than zero; both are fine.
    images = np.ones((3, 2, 2), dtype=np.float64)
    images[0] = np.nan
    out = _conform_aligned_trials(images, np.array([False, True, True]))
    assert np.isnan(out[0]).all()
    assert (out[1:] == 1.0).all()


def test_conform_aligned_trials_rejects_missing_registered_trials() -> None:
    from wisco_slap.spinelab.datasets import _conform_aligned_trials

    images = np.ones((3, 2, 2))
    with pytest.raises(ValueError, match=r"stop at trial 3 but trials \[5\]"):
        _conform_aligned_trials(images, np.array([True, True, True, False, True]))


def test_conform_aligned_trials_rejects_too_many_images() -> None:
    from wisco_slap.spinelab.datasets import _conform_aligned_trials

    with pytest.raises(ValueError, match="4 aligned trial images for 3 trials"):
        _conform_aligned_trials(np.ones((4, 2, 2)), np.array([True, True, True]))


def test_conform_aligned_trials_rejects_data_in_unregistered_slot() -> None:
    from wisco_slap.spinelab.datasets import _conform_aligned_trials

    # A finite non-zero image where the offsets say "not aligned" means the
    # mask does not describe this array; refuse rather than silently blank it.
    images = np.ones((3, 2, 2))
    with pytest.raises(ValueError, match="trial 2 has no alignment offsets"):
        _conform_aligned_trials(images, np.array([True, False, True]))


# ---------------------------------------------------------------------------
# Re-binning an annotated folder
# ---------------------------------------------------------------------------


def test_guard_annotated_rebin_refuses_different_bin_trials(tmp_path: Path) -> None:
    from wisco_slap.spinelab.datasets import _guard_annotated_rebin

    target = tmp_path / "loc_E--acq_1"
    _touch(target / "metadata.json", json.dumps({"bin_trials": 18}))
    _touch(target / "dmd1_spine_annotations.json", "{}")
    with pytest.raises(ValueError, match="bin_trials=18"):
        _guard_annotated_rebin(target, 30)
    # Same value, or no annotations, or no metadata: nothing to object to.
    _guard_annotated_rebin(target, 18)
    (target / "dmd1_spine_annotations.json").unlink()
    _guard_annotated_rebin(target, 30)
    _touch(target / "dmd1_spine_annotations.json", "{}")
    (target / "metadata.json").unlink()
    _guard_annotated_rebin(target, 30)
