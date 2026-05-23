from __future__ import annotations

import wisco_slap.meta.status as status_mod


def test_scoring_status_reports_complete_model_name(monkeypatch, tmp_path) -> None:
    anmat_root = tmp_path / "analysis_materials"
    monkeypatch.setattr(status_mod.DEFS, "anmat_root", str(anmat_root))
    monkeypatch.setattr(
        status_mod.wis.meta.sync,
        "get_acq_sync_block",
        lambda subject, exp, loc, acq: 1,
    )

    scored_dir = (
        anmat_root
        / "alnilam"
        / "exp_3"
        / "sync_block_data"
        / "sync_block-1"
        / "hypnograms"
        / "model_scored"
    )
    scored_dir.mkdir(parents=True)
    for filename in (
        "bout_df.parquet",
        "epoch_df.parquet",
        "bout_hypno.csv",
        "epoch_hypno.csv",
    ):
        (scored_dir / filename).write_text("exists\n")
    (scored_dir / "model_version.txt").write_text(
        "model_name=MODEL_V2.pkl\nadd_wake_epochs=0.8\n"
    )

    result = status_mod.scoring_status("alnilam", "exp_3", "loc_F", "acq_1")

    assert result.complete is True
    assert result.status == "fresh"
    assert result.model_name == "MODEL_V2.pkl"
    assert result.sync_block == 1
    assert "add_wake_epochs=0.8" in result.details


def test_scoring_status_reports_missing_outputs(monkeypatch, tmp_path) -> None:
    anmat_root = tmp_path / "analysis_materials"
    monkeypatch.setattr(status_mod.DEFS, "anmat_root", str(anmat_root))
    monkeypatch.setattr(
        status_mod.wis.meta.sync,
        "get_acq_sync_block",
        lambda subject, exp, loc, acq: 2,
    )

    result = status_mod.scoring_status("avior", "exp_1", "loc_I", "acq_1")

    assert result.complete is False
    assert result.status == "missing"
    assert result.model_name is None
    assert result.sync_block == 2
    assert "model_scored" in result.details
