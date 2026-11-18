"""Tests for shaft-calcium freshness checking (two-upstream staleness)."""

from __future__ import annotations

from pathlib import Path

import zarr

from wisco_slap.pns import shaft_ca_gen, shaft_ca_mon
from wisco_slap.pns._frame_timing import SHAFT_TIME_AXIS_VERSION


def _setup(monkeypatch, tmp_path: Path, esum_name: str = "SummaryLoCo-1.mat"):
    monkeypatch.setattr(shaft_ca_gen, "anmat_root", str(tmp_path))
    monkeypatch.setattr(shaft_ca_gen, "annotation_root", str(tmp_path / "ann"))
    import wisco_slap as wis

    monkeypatch.setattr(
        wis.meta.get,
        "esum_mirror_path",
        lambda *args: str(tmp_path / esum_name),
    )


def _prox_csv(dend_ids: list[str | None], *, id_column: bool = True) -> str:
    """Minimal WISynaptic-format prox_lines CSV, one 2-vertex line per ID."""

    rows = [
        "index,shape-type,vertex-index,axis-0,axis-1"
        + (",dend-ID" if id_column else "")
    ]
    for index, dend_id in enumerate(dend_ids):
        for v in range(2):
            row = f"{index},path,{v},{10.0 * index},{5.0 + 20.0 * v}"
            if id_column:
                row += f",{dend_id or ''}"
            rows.append(row)
    return "\n".join(rows) + "\n"


def _write_prox(tmp_path: Path, dmd: int, content: str = _prox_csv(["A-1"])) -> Path:
    path = shaft_ca_gen.prox_lines_csv("subj", "exp_1", "loc_A", "acq_1", dmd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _write_outputs(tmp_path: Path, dmds: list[int], version: str = "SummaryLoCo-1"):
    out = shaft_ca_gen.shaft_ca_dir("subj", "exp_1", "loc_A", "acq_1")
    out.mkdir(parents=True, exist_ok=True)
    fingerprints = {}
    for dmd in dmds:
        zarr.open_group(str(out / f"shaft_ca_dmd{dmd}.zarr"), mode="a").attrs[
            "time_axis_version"
        ] = SHAFT_TIME_AXIS_VERSION
        csv = shaft_ca_gen.prox_lines_csv("subj", "exp_1", "loc_A", "acq_1", dmd)
        fingerprints[dmd] = shaft_ca_gen.prox_lines_fingerprint(csv)
    shaft_ca_gen._write_markers(out, version, fingerprints)
    return out


def test_status_no_prox_lines(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    result = shaft_ca_mon.check_status("subj", "exp_1", "loc_A", "acq_1")
    assert result.status == "no_prox_lines"


def test_status_unlabeled_prox_lines_count_as_absent(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _setup(monkeypatch, tmp_path)
    _write_prox(tmp_path, 1, content=_prox_csv(["A-1", None]))  # one blank ID
    _write_prox(tmp_path, 2, content=_prox_csv(["C-1"], id_column=False))
    result = shaft_ca_mon.check_status("subj", "exp_1", "loc_A", "acq_1")
    assert result.status == "no_prox_lines"
    assert result.dmds_with_prox == ()
    assert any("dmd1" in d and "no dend-ID" in d for d in result.details)
    assert any("dmd2" in d and "no 'dend-ID' column" in d for d in result.details)

    # A labeled DMD is checked normally; the unlabeled one is reported, not fatal.
    _write_prox(tmp_path, 2, content=_prox_csv(["C-1", "D-1"]))
    result = shaft_ca_mon.check_status("subj", "exp_1", "loc_A", "acq_1")
    assert result.status == "missing"
    assert result.dmds_with_prox == (2,)
    assert result.missing_dmds == (2,)
    assert any("dmd1: prox_lines unusable" in d for d in result.details)

    # acq_data reports the reason when nothing is usable.
    _write_prox(tmp_path, 2, content=_prox_csv(["C-1"], id_column=False))
    shaft_ca_mon.acq_data("subj", "exp_1", "loc_A", "acq_1")
    assert "no 'dend-ID' column" in capsys.readouterr().out


def test_status_missing_then_fresh(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    _write_prox(tmp_path, 1)
    result = shaft_ca_mon.check_status("subj", "exp_1", "loc_A", "acq_1")
    assert result.status == "missing"
    assert result.missing_dmds == (1,)

    _write_outputs(tmp_path, [1])
    result = shaft_ca_mon.check_status("subj", "exp_1", "loc_A", "acq_1")
    assert result.status == "fresh"


def test_status_stale_when_prox_lines_change(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    _write_prox(tmp_path, 1)
    _write_prox(tmp_path, 2)
    _write_outputs(tmp_path, [1, 2])
    assert shaft_ca_mon.check_status("subj", "exp_1", "loc_A", "acq_1").status == (
        "fresh"
    )
    # Redrawing one DMD's prox lines must flag only that DMD stale.
    _write_prox(tmp_path, 2, content=_prox_csv(["A-1", "B-1"]))
    result = shaft_ca_mon.check_status("subj", "exp_1", "loc_A", "acq_1")
    assert result.status == "stale"
    assert result.stale_dmds == (2,)


def test_status_stale_when_exsum_version_changes(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    _write_prox(tmp_path, 1)
    _write_outputs(tmp_path, [1], version="SummaryLoCo-0")  # old version marker
    result = shaft_ca_mon.check_status("subj", "exp_1", "loc_A", "acq_1")
    assert result.status == "stale"
    assert "ExSum version" in result.details[0]


def test_old_clock_is_stale_even_when_exsum_and_prox_lines_match(
    monkeypatch, tmp_path: Path
) -> None:
    _setup(monkeypatch, tmp_path)
    _write_prox(tmp_path, 1)
    out = _write_outputs(tmp_path, [1])
    cache = zarr.open_group(str(out / "shaft_ca_dmd1.zarr"), mode="a")
    del cache.attrs["time_axis_version"]
    result = shaft_ca_mon.check_status("subj", "exp_1", "loc_A", "acq_1")
    assert result.status == "stale"
    assert result.stale_dmds == (1,)
    assert "outdated shaft clock" in result.details[0]


def test_acq_data_skips_fresh(monkeypatch, tmp_path: Path, capsys) -> None:
    _setup(monkeypatch, tmp_path)
    _write_prox(tmp_path, 1)
    _write_outputs(tmp_path, [1])
    calls: list[tuple] = []
    monkeypatch.setattr(
        shaft_ca_gen, "acq_data", lambda *args, **kwargs: calls.append(args)
    )
    shaft_ca_mon.acq_data("subj", "exp_1", "loc_A", "acq_1")
    assert calls == []  # fresh -> no regeneration
    shaft_ca_mon.acq_data("subj", "exp_1", "loc_A", "acq_1", overwrite=True)
    assert len(calls) == 1
