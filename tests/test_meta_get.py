from __future__ import annotations

import wisco_slap.meta.get as meta_get


def _write_required_yaml(root) -> None:
    (root / "acquisition_master.yaml").write_text(
        """
subj:
  exp_1:
    - loc_A--acq_1
    - loc_B--acq_1
    - loc_C--acq_1
""".lstrip(),
        encoding="utf-8",
    )
    (root / "dmd_info.yaml").write_text(
        """
subj:
  exp_1:
    loc_A:
      acq_1:
        dmd-1: {depth: 100}
        dmd-2: {depth: 110}
    loc_B:
      acq_1:
        dmd-1: {depth: 100}
        dmd-2: {depth: 110}
    loc_C:
      acq_1:
        dmd-1: {depth: 100}
        dmd-2: {depth: 110}
""".lstrip(),
        encoding="utf-8",
    )
    (root / "sync_info.yaml").write_text(
        """
subj:
  exp_1:
    acquisitions:
      loc_A--acq_1:
        sync_block: 1
      loc_B--acq_1:
        sync_block: 2
      loc_C--acq_1:
        sync_block: 3
""".lstrip(),
        encoding="utf-8",
    )


def _write_audit_csv(root) -> None:
    (root / "ExSum_audits.csv").write_text(
        "\n".join(
            [
                "subject,exp,loc,acq,scopex_status,events_matchfilt_status,"
                "events_denoised_status,annotation_status,scoring_complete",
                "subj,exp_1,loc_A,acq_1,fresh,not_applicable,fresh,fresh,true",
                "subj,exp_1,loc_B,acq_1,fresh,missing,fresh,fresh,true",
                "subj,exp_1,loc_C,acq_1,fresh,fresh,fresh,fresh,true",
            ]
        ),
        encoding="utf-8",
    )


def _write_annotation_files(root, loc: str, complete_labels: bool = True) -> None:
    acq_dir = root / "annotation_materials" / "subj" / "exp_1" / loc / "acq_1"
    for dmd in ("dmd-1", "dmd-2"):
        dmd_dir = acq_dir / "synapse_ids" / dmd
        dmd_dir.mkdir(parents=True, exist_ok=True)
        (dmd_dir / "1.png").touch()
        label = "soma_a" if complete_labels or dmd == "dmd-1" else ""
        (dmd_dir / "synapse_labels.csv").write_text(
            f"source-ID,soma-ID\n1,{label}\n",
            encoding="utf-8",
        )

    sorting_dir = acq_dir / "source_sorting"
    sorting_dir.mkdir(parents=True, exist_ok=True)
    (sorting_dir / "prox_lines_dmd1.csv").touch()
    (sorting_dir / "prox_lines_dmd2.csv").touch()


def _write_sync_block_files(root, sync_block: int) -> None:
    sb_dir = (
        root
        / "subj"
        / "exp_1"
        / "sync_block_data"
        / f"sync_block-{sync_block}"
    )
    (sb_dir / "whisking").mkdir(parents=True, exist_ok=True)
    (sb_dir / "whisking" / "whisk_df.parquet").touch()
    (sb_dir / "pupil" / "eye_metrics").mkdir(parents=True, exist_ok=True)
    (sb_dir / "pupil" / "eye_metrics" / "eye_metrics.parquet").touch()
    (sb_dir / "ephys").mkdir(parents=True, exist_ok=True)
    (sb_dir / "ephys" / "EEGr.nc").touch()


def test_valid_acquisitions_filters_to_complete_red_arrow_items(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(meta_get, "anmat_root", str(tmp_path))
    _write_required_yaml(tmp_path)
    _write_audit_csv(tmp_path)

    _write_annotation_files(tmp_path, "loc_A", complete_labels=True)
    _write_annotation_files(tmp_path, "loc_B", complete_labels=True)
    _write_annotation_files(tmp_path, "loc_C", complete_labels=False)
    for sync_block in (1, 2, 3):
        _write_sync_block_files(tmp_path, sync_block)

    assert meta_get.valid_acquisitions() == ["subj--exp_1--loc_A--acq_1"]
