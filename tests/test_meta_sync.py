from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

import wisco_slap.defs as DEFS
from slap2_py.utils.drec import generate_acq_segment_df, generate_scope_index_df
from wisco_slap.meta.sync import (
    back_calculate_ephys_start,
    detect_ephys_start_sample,
    match_scope_up_window,
    resolve_ephys_start_sample,
)

FS = 5000


def _synthetic_scope(
    preview: tuple[float, float] = (20.0, 50.0),
    acquisition: tuple[float, float] = (50.0, 110.0),
    total_s: float = 120.0,
) -> np.ndarray:
    """0/5 V trigger: slow toggling (duty 0.6) during ``preview``, then an
    acquisition with a 1 ms dip every 50 ms (duty 0.98)."""
    x = np.zeros(int(total_s * FS))
    p0, p1 = int(preview[0] * FS), int(preview[1] * FS)
    a0, a1 = int(acquisition[0] * FS), int(acquisition[1] * FS)
    prev = x[p0:p1].reshape(-1, 10)
    prev[:, :6] = 5.0
    acq = x[a0:a1].reshape(-1, 250)
    acq[:, :] = 5.0
    acq[:, 100:105] = 0.0
    return x


def test_detect_ephys_start_sample_finds_first_step() -> None:
    ephys = np.zeros(4 * FS)
    ephys[3 * FS + 7 :] = 1.0
    assert detect_ephys_start_sample(ephys, fs=FS) == 3 * FS + 7


def test_detect_ephys_start_sample_raises_when_line_already_high() -> None:
    ephys = np.ones(4 * FS)  # never exceeds 3 x max(first 2 s)
    with pytest.raises(ValueError, match="ephys_start_sample"):
        detect_ephys_start_sample(ephys, fs=FS)


def test_resolve_ephys_start_sample_precedence() -> None:
    ephys = np.zeros(4 * FS)
    ephys[3 * FS :] = 1.0
    assert resolve_ephys_start_sample(ephys, override=123, stored=456, fs=FS) == (
        123,
        "override",
    )
    assert resolve_ephys_start_sample(ephys, override=None, stored=456, fs=FS) == (
        456,
        "stored",
    )
    assert resolve_ephys_start_sample(ephys, fs=FS) == (3 * FS, "detector")


def test_generate_acq_segment_df_ignores_preview_and_refines_onset() -> None:
    scope = _synthetic_scope()
    seg = generate_acq_segment_df(scope, fs=FS)
    assert len(seg) == 1
    assert abs(int(seg.loc[0, "start_idx"]) - 50 * FS) <= 2
    assert abs(int(seg.loc[0, "end_idx"]) - 110 * FS) <= 5
    assert seg.loc[0, "duration_s"] == pytest.approx(60.0, abs=0.01)
    # The hysteresis detector merges the preview into the same window, so its
    # start is the *preview* start — the failure this detector exists to fix.
    win = generate_scope_index_df(scope, fs=FS)
    assert int(win.loc[0, "start_idx"]) < 25 * FS


def test_generate_acq_segment_df_drops_short_bursts_and_flat_signals() -> None:
    burst = np.zeros(60 * FS)
    burst[10 * FS : 15 * FS] = 5.0  # 5 s < min_duration_s
    assert generate_acq_segment_df(burst, fs=FS).empty
    assert generate_acq_segment_df(np.zeros(60 * FS), fs=FS).empty
    assert generate_acq_segment_df(np.full(60 * FS, 5.0), fs=FS).empty


def test_match_scope_up_window_anchors_on_acquisition_onset() -> None:
    scope = _synthetic_scope()
    ephys = np.zeros(scope.size)
    ephys[2 * FS + 500 :] = 1.0  # ephys ON at 2.1 s
    ephys_start_ts = pd.Timestamp("2026-01-01 10:00:00")
    dat_start = ephys_start_ts + pd.Timedelta(seconds=48)  # wall clock ~ 47.9 s
    offset = match_scope_up_window(scope, ephys, dat_start, ephys_start_ts, fs=FS)
    expected_vdaq_s = (50 * FS - (2 * FS + 500)) / FS  # 47.9 s
    assert offset == pytest.approx(
        expected_vdaq_s / DEFS.VDAQ_TDT_CLOCK_RATIO, abs=2 / FS
    )


def test_match_scope_up_window_falls_back_to_scope_up_windows(capsys) -> None:
    # A 40 s preview-style episode (duty 0.6): the hysteresis detector sees one
    # scope-UP window (gaps of 0.8 ms are merged) but no bin reaches duty 0.9,
    # so the matcher must fall back to the window start.
    scope = np.zeros(80 * FS)
    prev = scope[20 * FS : 60 * FS].reshape(-1, 10)
    prev[:, :6] = 5.0
    assert generate_acq_segment_df(scope, fs=FS).empty
    ephys = np.zeros(scope.size)
    ephys[3 * FS :] = 1.0
    ephys_start_ts = pd.Timestamp("2026-01-01 10:00:00")
    offset = match_scope_up_window(
        scope, ephys, ephys_start_ts + pd.Timedelta(seconds=17), ephys_start_ts, fs=FS
    )
    assert offset == pytest.approx(17.0 / DEFS.VDAQ_TDT_CLOCK_RATIO, abs=2 / FS)
    assert "scope-UP window" in capsys.readouterr().out


def test_update_sync_info_redo_refits_existing_acqs_and_persists_ephys_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """redo=True must re-run the matcher for acquisitions that already have an
    entry (it used to touch only new ones), keep their dat_start, and persist
    the ephys_start_sample used for the block."""
    import yaml

    import wisco_slap.meta.sync as sync_mod

    yaml_path = tmp_path / "sync_info.yaml"
    existing = {
        "mouse": {
            "exp_1": {
                "sync_blocks": {
                    1: {
                        "ephys_start": "2026-01-01 10:00:00",
                        "ephys_end": "2026-01-01 11:00:00",
                        "corrupt": False,
                    }
                },
                "acquisitions": {
                    "loc_A--acq_1": {
                        "sync_block": 1,
                        "dat_start": "2026-01-01 10:00:20.123456",
                        "ephys_offset": 111.0,
                        "n_epochs": 1,
                    }
                },
            }
        }
    }
    yaml_path.write_text(yaml.safe_dump(existing))
    ephys = np.zeros(60 * FS)
    ephys[3 * FS :] = 1.0
    monkeypatch.setattr(DEFS, "anmat_root", str(tmp_path))
    monkeypatch.setattr(
        sync_mod, "sync_info", lambda: yaml.safe_load(yaml_path.read_text())
    )
    monkeypatch.setattr(
        sync_mod.wis.meta.get,
        "unique_acquisitions_per_experiment",
        lambda s, e: ["loc_A--acq_1"],
    )
    monkeypatch.setattr(sync_mod, "_has_first_cycle_dat_file", lambda *a: True)
    monkeypatch.setattr(
        sync_mod,
        "_list_epoch_start_times",
        lambda *a: [("f.dat", pd.Timestamp("2026-01-01 10:00:20"))],
    )
    monkeypatch.setattr(
        sync_mod, "get_all_sync_paths", lambda s, e: ([1], ["/x/SYNC_00001.h5"])
    )
    monkeypatch.setattr(
        sync_mod,
        "get_ephys_coverage",
        lambda s, e: {
            1: (
                pd.Timestamp("2026-01-01 10:00:00"),
                pd.Timestamp("2026-01-01 11:00:00"),
            )
        },
    )
    monkeypatch.setattr(
        sync_mod,
        "load_all_sync_blocks",
        lambda s, e: ({1: np.zeros(60 * FS)}, {1: ephys}),
    )
    calls: list[int | None] = []

    def fake_offset(
        subject,
        exp,
        loc,
        acq,
        si,
        scope,
        ephys,
        fs=5000,
        ephys_start_sample=None,
        clock_ratio=None,
    ):
        calls.append(ephys_start_sample)
        return 17.5

    monkeypatch.setattr(sync_mod, "compute_ephys_offset", fake_offset)

    sync_mod.update_sync_info("mouse", "exp_1", redo=True)
    out = yaml.safe_load(yaml_path.read_text())["mouse"]["exp_1"]
    acq = out["acquisitions"]["loc_A--acq_1"]
    assert calls == [3 * FS]  # matcher re-run once, with the detected ephys start
    assert acq["ephys_offset"] == pytest.approx(17.5)
    assert acq["dat_start"] == "2026-01-01 10:00:20.123456"  # preserved, not re-parsed
    assert out["sync_blocks"][1]["ephys_start_sample"] == 3 * FS

    # Second redo reuses the persisted sample (no re-detection) and an explicit
    # override wins over it.
    calls.clear()
    sync_mod.update_sync_info("mouse", "exp_1", redo=True)
    assert calls == [3 * FS]
    calls.clear()
    sync_mod.update_sync_info(
        "mouse", "exp_1", redo=True, ephys_start_sample={"loc_A--acq_1": 4242}
    )
    assert calls == [4242]
    assert (
        yaml.safe_load(yaml_path.read_text())["mouse"]["exp_1"]["sync_blocks"][1][
            "ephys_start_sample"
        ]
        == 4242
    )


def test_load_sync_block_falls_back_to_recovered_copy(monkeypatch, tmp_path) -> None:
    import h5py

    import wisco_slap.meta.sync as sync_mod

    raw = tmp_path / "raw" / "SYNC_00001.h5"
    raw.parent.mkdir()
    raw.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 4096)  # unopenable
    rec_dir = tmp_path / "recovery" / "mouse" / "exp_1"
    rec_dir.mkdir(parents=True)
    with h5py.File(rec_dir / "SYNC_00001.h5", "w") as h:
        h.create_dataset("slap2_acquiring_trigger", data=np.arange(10, dtype="<f4"))
        h.create_dataset("electrophysiology", data=np.arange(10, 20, dtype="<f4"))
    monkeypatch.setattr(sync_mod, "get_all_sync_paths", lambda s, e: ([1], [str(raw)]))
    monkeypatch.setattr(DEFS, "sync_recovery_root", str(tmp_path / "recovery"))

    scope, ephys = sync_mod.load_sync_block("mouse", "exp_1", 1)
    assert scope[3] == 3 and ephys[3] == 13
    assert sync_mod.sync_source("mouse", "exp_1", 1) == "recovered"
    # without a recovered copy the block is reported corrupt
    monkeypatch.setattr(DEFS, "sync_recovery_root", str(tmp_path / "nowhere"))
    assert sync_mod.load_sync_block("mouse", "exp_1", 1) == ("Corrupt", "Corrupt")
    assert sync_mod.sync_source("mouse", "exp_1", 1) == "corrupt"


def _block_fixture(n_s: float = 120.0):
    """Trigger with two acquisitions (onsets 40 s and 80 s, 30 s long) and an
    ephys line with an early short pulse (4-5 s) and the real rise at 10 s."""
    n = int(n_s * FS)
    scope = np.zeros(n)
    scope[40 * FS : 70 * FS] = 5.0
    scope[80 * FS : 110 * FS] = 5.0
    ephys = np.full(n, 0.004)
    ephys[4 * FS : 5 * FS] = 0.15
    ephys[10 * FS :] = 0.15
    return scope, ephys


def test_back_calculate_ephys_start_recovers_override_and_flags_bad_offsets() -> None:
    scope, ephys = _block_fixture()
    ts = pd.Timestamp("2026-01-01 10:00:00")
    real = 10 * FS  # the value a manual override would have supplied
    acq = {
        "dat_start": str(ts + pd.Timedelta(seconds=30)),
        "ephys_offset_uncorrected": (40 * FS - real) / FS,
    }
    r = back_calculate_ephys_start(scope, ephys, acq, ts, fs=FS)
    assert (
        r["value"] == real
        and r["edge"]
        and r["detector"] == 4 * FS
        and r["onset"] == 40 * FS
    )
    # offset computed with the (wrong) detector value -> back-calculates to the
    # pulse, which is still an edge
    acq2 = dict(acq, ephys_offset_uncorrected=(40 * FS - 4 * FS) / FS)
    assert back_calculate_ephys_start(scope, ephys, acq2, ts, fs=FS)["value"] == 4 * FS
    # a nonsense offset lands nowhere near an edge
    acq3 = dict(acq, ephys_offset_uncorrected=(40 * FS - 7 * FS) / FS)
    r3 = back_calculate_ephys_start(scope, ephys, acq3, ts, fs=FS)
    assert r3["value"] == 7 * FS and not r3["edge"] and "no ephys edge" in r3["reason"]


def test_backfill_ephys_start_samples_writes_only_verified_blocks(
    monkeypatch, tmp_path
) -> None:
    import yaml

    import wisco_slap.meta.sync as sync_mod

    scope, ephys = _block_fixture()
    ts = "2026-01-01 10:00:00"
    real = 10 * FS

    def acq(offset_from: int, gap_s: float):
        return {
            "sync_block": 1,
            "dat_start": str(pd.Timestamp(ts) + pd.Timedelta(seconds=gap_s)),
            "ephys_offset_uncorrected": (int(gap_s * FS) + 10 * FS - offset_from) / FS,
            "ephys_offset": 0.0,
        }

    si = {
        "mouse": {
            # block 1: two acqs, both encode the override (10 s) -> write, override
            "exp_1": {
                "sync_blocks": {1: {"ephys_start": ts, "corrupt": False}},
                "acquisitions": {
                    "loc_A--acq_1": acq(real, 30),
                    "loc_A--acq_2": acq(real, 70),
                },
            },
            # block 1: acqs disagree -> manual_review
            "exp_2": {
                "sync_blocks": {1: {"ephys_start": ts, "corrupt": False}},
                "acquisitions": {
                    "loc_A--acq_1": acq(real, 30),
                    "loc_A--acq_2": acq(4 * FS, 70),
                },
            },
            # already stored (mismatch) -> reported, untouched; plus a block w/o acqs
            "exp_3": {
                "sync_blocks": {
                    1: {
                        "ephys_start": ts,
                        "corrupt": False,
                        "ephys_start_sample": 12345,
                    },
                    2: {"ephys_start": ts, "corrupt": False},
                },
                "acquisitions": {"loc_A--acq_1": acq(real, 30)},
            },
        }
    }
    yaml_path = tmp_path / "sync_info.yaml"
    yaml_path.write_text(yaml.safe_dump(si))
    monkeypatch.setattr(DEFS, "anmat_root", str(tmp_path))
    monkeypatch.setattr(
        sync_mod, "sync_info", lambda: yaml.safe_load(yaml_path.read_text())
    )
    monkeypatch.setattr(sync_mod, "load_sync_block", lambda s, e, b: (scope, ephys))

    df = sync_mod.backfill_ephys_start_samples(dry_run=True, verbose=False)
    st = {(r["exp"], r["block"]): r for r in df.to_dicts()}
    assert st[("exp_1", 1)]["status"] == "write" and st[("exp_1", 1)]["value"] == real
    assert st[("exp_1", 1)]["source"] == f"backfill (override; detector={4 * FS})"
    assert st[("exp_2", 1)]["status"] == "manual_review"
    assert (
        st[("exp_3", 1)]["status"] == "already_stored"
        and "MISMATCH" in st[("exp_3", 1)]["note"]
    )
    assert st[("exp_3", 2)]["status"] == "no_acquisitions"
    assert yaml.safe_load(yaml_path.read_text()) == si  # dry run wrote nothing

    df = sync_mod.backfill_ephys_start_samples(dry_run=False, verbose=False)
    out = yaml.safe_load(yaml_path.read_text())["mouse"]
    assert out["exp_1"]["sync_blocks"][1]["ephys_start_sample"] == real
    assert out["exp_1"]["sync_blocks"][1]["ephys_start_source"].startswith(
        "backfill (override"
    )
    assert "ephys_start_sample" not in out["exp_2"]["sync_blocks"][1]
    assert (
        out["exp_3"]["sync_blocks"][1]["ephys_start_sample"] == 12345
    )  # never overwritten
    assert "ephys_start_sample" not in out["exp_3"]["sync_blocks"][2]
    assert df.filter(pl.col("status") == "written").height == 1
    assert list(tmp_path.glob("sync_info.yaml.bak-*-pre-ephys-start-backfill"))
