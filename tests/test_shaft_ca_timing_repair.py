"""Legacy clock repair preserves fluorescence and refuses ambiguous provenance."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

import wisco_slap as wis
from wisco_slap.pns import shaft_ca_gen as gen
from wisco_slap.pns import shaft_ca_mon
from wisco_slap.pns._frame_timing import SHAFT_TIME_AXIS_VERSION

KEYS = ("subj", "exp_1", "loc_A", "acq_1")


@pytest.fixture
def legacy_shaft(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gen, "anmat_root", str(tmp_path))
    monkeypatch.setattr(gen, "annotation_root", str(tmp_path / "ann"))
    exsum = tmp_path / "SummaryLoCo-test.mat"
    monkeypatch.setattr(wis.meta.get, "esum_mirror_path", lambda *_: str(exsum))
    monkeypatch.setattr(gen.spy.xsum, "get_exsum_basics", lambda *_: ({}, 3, 20))
    monkeypatch.setattr(gen.frames, "trial_epochs", lambda *_: np.ones(3, dtype=int))
    monkeypatch.setattr(gen.frames, "per_trial_sample_counts", lambda *_: [20, 0, 20])
    csv = gen.prox_lines_csv(*KEYS, 1)
    csv.parent.mkdir(parents=True)
    csv.write_text(
        "index,shape-type,vertex-index,axis-0,axis-1,dend-ID\n"
        "0,path,0,0,0,A-1\n0,path,1,0,20,A-1\n"
    )
    out = gen.shaft_ca_dir(*KEYS)
    out.mkdir(parents=True)
    path = out / "shaft_ca_dmd1.zarr"
    values = np.arange(48, dtype=np.float32).reshape(2, 24)
    values[:, 8:16] = np.nan
    xr.Dataset(
        {"F": (("disc", "time"), values), "dff": (("disc", "time"), values / 10)},
        coords={
            "disc": ["A-1:000", "A-1:001"],
            "time": np.arange(24) / 12,
            "trial": ("time", np.repeat([1, 2, 3], 8)),
            "quality": ("time", np.repeat([True, False, True], 8)),
        },
        attrs={"esum_version": exsum.stem, "sample_rate_hz": 12, "dmd": 1},
    ).to_zarr(path, mode="w", consolidated=True)
    gen._write_markers(out, exsum.stem, {1: gen.prox_lines_fingerprint(csv)})
    return path


def test_loader_refuses_legacy_clock(legacy_shaft):
    with pytest.raises(ValueError, match="repair_time_coordinates"):
        wis.get.shaft_ca(*KEYS, dmd=1, apply_ephys_offset=False)


def test_repair_is_dry_runnable_idempotent_and_preserves_values(legacy_shaft):
    path = legacy_shaft
    with xr.open_zarr(path) as ds:
        before = ds.load()
    assert shaft_ca_mon.check_status(*KEYS).status == "stale"
    record = gen.repair_time_coordinates(*KEYS, dry_run=True)[0]
    assert record["status"] == "would_repair"
    assert not (path.parent / "timing_backups").exists()
    np.testing.assert_array_equal(xr.open_zarr(path).time, before.time)
    record = gen.repair_time_coordinates(*KEYS)[0]
    repaired = wis.get.shaft_ca(*KEYS, dmd=1, apply_ephys_offset=False)["dmd_1"]
    np.testing.assert_array_equal(repaired.time, np.arange(24) / 8)
    for name in ("F", "dff", "trial", "quality", "disc"):
        np.testing.assert_array_equal(repaired[name].values, before[name].values)
    assert repaired.attrs["time_axis_version"] == SHAFT_TIME_AXIS_VERSION
    assert repaired.attrs["sample_rate_hz"] == 8
    backup = Path(record["backup"])
    np.testing.assert_array_equal(zarr.open_array(str(backup / "time"))[:], before.time)
    assert shaft_ca_mon.check_status(*KEYS).status == "fresh"
    assert gen.repair_time_coordinates(*KEYS)[0]["status"] == "current"
    assert len(list((path.parent / "timing_backups").iterdir())) == 1


def test_stale_exsum_is_not_used_to_repair(legacy_shaft):
    path = legacy_shaft
    zarr.open_group(str(path), mode="a").attrs["esum_version"] = "old"
    with pytest.raises(ValueError, match="stale ExSum"):
        gen.repair_time_coordinates(*KEYS)
    assert not (path.parent / "timing_backups").exists()


def test_reordered_trials_are_rejected(legacy_shaft):
    path = legacy_shaft
    cache = zarr.open_group(str(path), mode="a")
    cache["trial"][:] = np.repeat([2, 1, 3], 8)
    with pytest.raises(ValueError, match="original trial order"):
        gen.repair_time_coordinates(*KEYS)
    assert not (path.parent / "timing_backups").exists()


def test_failed_repair_restores_original_time_and_metadata(legacy_shaft, monkeypatch):
    path = legacy_shaft
    before_attrs = (path / ".zattrs").read_bytes()
    before_consolidated = (path / ".zmetadata").read_bytes()

    def fail_consolidation(*args, **kwargs):
        raise OSError("simulated metadata failure")

    monkeypatch.setattr(zarr, "consolidate_metadata", fail_consolidation)
    with pytest.raises(OSError, match="simulated"):
        gen.repair_time_coordinates(*KEYS)
    assert (path / ".zattrs").read_bytes() == before_attrs
    assert (path / ".zmetadata").read_bytes() == before_consolidated
    np.testing.assert_array_equal(xr.open_zarr(path).time, np.arange(24) / 12)
