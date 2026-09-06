from __future__ import annotations

import json

import pytest

import wisco_slap.meta.acq_types._classifications as meta_classifications
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
                "subject,exp,loc,acq,scopex_status,annotation_status,"
                "scoring_complete",
                "subj,exp_1,loc_A,acq_1,fresh,fresh,true",
                "subj,exp_1,loc_B,acq_1,stale,fresh,true",
                "subj,exp_1,loc_C,acq_1,fresh,fresh,true",
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


def _write_baseline_validity(root, labels: dict[str, str]) -> None:
    """Write a baseline_validity classification; keys are ``loc`` names."""
    cdir = root / "annotation_materials" / "_CLASSIFICATIONS"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "baseline_validity.json").write_text(
        json.dumps(
            {
                "name": "baseline_validity",
                "values": [
                    {"value": "valid", "hotkey": "v"},
                    {"value": "invalid", "hotkey": "x"},
                ],
                "labels": [
                    {
                        "subject": "subj",
                        "exp": "exp_1",
                        "loc": loc,
                        "acq": "acq_1",
                        "value": value,
                    }
                    for loc, value in labels.items()
                ],
            }
        ),
        encoding="utf-8",
    )


def _setup_all_material_checks_pass(monkeypatch, tmp_path) -> None:
    """Everything green for loc_A only; loc_B is stale, loc_C mislabeled."""
    monkeypatch.setattr(meta_get, "anmat_root", str(tmp_path))
    monkeypatch.setattr(meta_classifications, "anmat_root", str(tmp_path))
    _write_required_yaml(tmp_path)
    _write_audit_csv(tmp_path)

    _write_annotation_files(tmp_path, "loc_A", complete_labels=True)
    _write_annotation_files(tmp_path, "loc_B", complete_labels=True)
    _write_annotation_files(tmp_path, "loc_C", complete_labels=False)
    for sync_block in (1, 2, 3):
        _write_sync_block_files(tmp_path, sync_block)


def test_valid_acquisitions_filters_to_complete_red_arrow_items(
    monkeypatch, tmp_path
) -> None:
    _setup_all_material_checks_pass(monkeypatch, tmp_path)
    _write_baseline_validity(tmp_path, {"loc_A": "valid"})

    assert meta_get.valid_acquisitions() == ["subj--exp_1--loc_A--acq_1"]


def test_valid_acquisitions_excludes_non_valid_baseline_label(
    monkeypatch, tmp_path
) -> None:
    _setup_all_material_checks_pass(monkeypatch, tmp_path)
    _write_baseline_validity(tmp_path, {"loc_A": "invalid"})

    assert meta_get.valid_acquisitions() == []


def test_valid_acquisitions_excludes_unlabeled_with_warning(
    monkeypatch, tmp_path
) -> None:
    _setup_all_material_checks_pass(monkeypatch, tmp_path)
    # loc_A is absent from the file entirely; loc_B carries the seen-but-unlabeled
    # placeholder (it fails a material check anyway, so it must not be named).
    _write_baseline_validity(tmp_path, {"loc_B": meta_classifications.NO_VALUE})

    with pytest.warns(UserWarning, match="no 'baseline_validity' label"):
        assert meta_get.valid_acquisitions() == []


def test_valid_acquisitions_placeholder_label_counts_as_unlabeled(
    monkeypatch, tmp_path
) -> None:
    _setup_all_material_checks_pass(monkeypatch, tmp_path)
    _write_baseline_validity(tmp_path, {"loc_A": meta_classifications.NO_VALUE})

    with pytest.warns(UserWarning, match="subj--exp_1--loc_A--acq_1"):
        assert meta_get.valid_acquisitions() == []


def test_valid_acquisitions_can_skip_the_baseline_check(monkeypatch, tmp_path) -> None:
    _setup_all_material_checks_pass(monkeypatch, tmp_path)
    # No classification file at all: the check would raise, the escape hatch must not.
    assert meta_get.valid_acquisitions(require_baseline_valid=False) == [
        "subj--exp_1--loc_A--acq_1"
    ]


def test_valid_acquisitions_requires_the_classification_file(
    monkeypatch, tmp_path
) -> None:
    _setup_all_material_checks_pass(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError, match="require_baseline_valid=False"):
        meta_get.valid_acquisitions()
