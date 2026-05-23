from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import wisco_slap.pns.sleepscore._features as features_mod
import wisco_slap.pns.sleepscore._model as model_mod
import wisco_slap.pns.sleepscore.score_all as score_all_mod
from wisco_slap.pns.sleepscore._config import (
    VIDEO_QUALITY_FEATURES,
    ClassifierConfig,
    HSMMConfig,
    ScoreConfig,
)
from wisco_slap.pns.sleepscore._data import validate_and_standardize_labels
from wisco_slap.pns.sleepscore._evaluate import evaluate_loso, evaluate_scored_session
from wisco_slap.pns.sleepscore._hsmm import build_hsmm_components


def _make_synthetic_feature_bundle(
    session_ids: list[str],
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, float | int | str]] = []
    edges_by_session: dict[str, np.ndarray] = {}
    label_order = ["NREM", "NREM", "REM", "REM", "Wake", "Wake"]
    feature_lookup = {
        "NREM": (-4.0, -1.0),
        "REM": (1.0, 3.0),
        "Wake": (5.0, 7.0),
    }

    for session_index, session_id in enumerate(session_ids):
        edges_by_session[session_id] = np.arange(7, dtype=float)
        for epoch_idx, label in enumerate(label_order):
            feat_a, feat_b = feature_lookup[label]
            rows.append(
                {
                    "session_id": session_id,
                    "epoch_idx": epoch_idx,
                    "start_s": float(epoch_idx),
                    "end_s": float(epoch_idx + 1),
                    "feat_a": feat_a + 0.05 * session_index,
                    "feat_b": (
                        None
                        if session_index == 0 and epoch_idx == len(label_order) - 1
                        else feat_b + 0.05 * epoch_idx
                    ),
                    "feat_constant": 1.0,
                    "pu_valid_frac": 1.0 if epoch_idx < 5 else 0.5,
                    "eyelid_valid_frac": 1.0 if epoch_idx != 3 else 0.25,
                    "whisk_valid_frac": 1.0 if epoch_idx < 4 else 0.0,
                    "camera_off_frac": 0.0 if epoch_idx < 4 else 1.0,
                    "camera_off_epoch": 0.0 if epoch_idx < 4 else 1.0,
                }
            )

    return pl.DataFrame(rows), edges_by_session


def _make_labels(wake_label: str = "Artifact") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "start_s": [0.0, 2.0, 4.0],
            "end_s": [2.0, 4.0, 6.0],
            "label": ["NREM_SWS", "REM", wake_label],
        }
    )


def _make_labels_with_artifact_and_wake() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "start_s": [0.0, 2.0, 4.0, 5.0],
            "end_s": [2.0, 4.0, 5.0, 6.0],
            "label": ["NREM_SWS", "REM", "Artifact", "Wake"],
        }
    )


def _make_training_config(calibration_cv: int = 3) -> ScoreConfig:
    return ScoreConfig(
        classifier=ClassifierConfig(
            lgbm_params={
                "n_estimators": 30,
                "max_depth": 3,
                "learning_rate": 0.1,
                "num_leaves": 7,
                "min_child_samples": 1,
                "reg_lambda": 0.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "verbose": -1,
            },
            calibration_cv=calibration_cv,
            random_state=123,
        )
    )


def _make_epoch_and_bout_outputs_with_single_error() -> tuple[pl.DataFrame, pl.DataFrame]:
    epoch_df = pl.DataFrame(
        {
            "epoch_idx": [0, 1, 2, 3, 4, 5],
            "start_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "end_s": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "pred_label": ["NREM", "NREM", "REM", "Wake", "Wake", "Wake"],
            "P_NREM": [0.93, 0.91, 0.04, 0.03, 0.05, 0.04],
            "P_REM": [0.03, 0.05, 0.89, 0.12, 0.03, 0.02],
            "P_Wake": [0.04, 0.04, 0.07, 0.85, 0.92, 0.94],
        }
    )
    bout_df = pl.DataFrame(
        {
            "start_s": [0.0, 2.0, 3.0],
            "end_s": [2.0, 3.0, 6.0],
            "label": ["NREM", "REM", "Wake"],
            "mean_P_NREM": [0.92, 0.04, 0.04],
            "mean_P_REM": [0.04, 0.89, 0.06],
            "mean_P_Wake": [0.04, 0.07, 0.90],
        }
    )
    return epoch_df, bout_df


def _make_video_mask_session() -> tuple[dict, np.ndarray]:
    fs_eeg = 64.0
    eeg_timestamps = np.arange(0.0, 4.0, 1.0 / fs_eeg)
    eeg_signal = np.sin(2.0 * np.pi * 8.0 * eeg_timestamps)

    fs_pu = 10.0
    pupil_timestamps = np.arange(0.0, 4.0, 1.0 / fs_pu)
    n_frames = len(pupil_timestamps)

    diameter = np.linspace(10.0, 14.0, n_frames)
    motion = np.full(n_frames, 0.5)
    eyelid = np.full(n_frames, 2.0)
    eyelid_norm = np.full(n_frames, 0.6)
    whisking = np.full(n_frames, 5.0)
    pup_likelihood = np.full(n_frames, 0.95)
    lid_likelihood = np.full(n_frames, 0.95)

    pup_likelihood[:10] = 0.10
    lid_likelihood[10:20] = 0.10
    pup_likelihood[20:30] = 0.05
    lid_likelihood[20:30] = 0.05
    whisking[20:30] = 0.0
    pup_likelihood[35:40] = 0.10

    session = {
        "session_id": "kaus_exp_2_sb-2",
        "eeg": {
            "signal": eeg_signal,
            "timestamps": eeg_timestamps,
            "fs": fs_eeg,
        },
        "pupil": {
            "diameter": diameter,
            "motion": motion,
            "eyelid": eyelid,
            "eyelid_norm": eyelid_norm,
            "pup_likelihood": pup_likelihood,
            "lid_likelihood": lid_likelihood,
            "whisking": whisking,
            "timestamps": pupil_timestamps,
            "fs": fs_pu,
        },
    }
    edges = np.arange(5, dtype=float)
    return session, edges


def test_validate_and_standardize_labels_maps_nrem_alias_and_artifact() -> None:
    df = pl.DataFrame(
        {
            "start_s": [0.0, 1.0, 2.0],
            "end_s": [1.0, 2.0, 3.0],
            "label": [" NREM_SWS ", "artifact", "wake_quiet"],
        }
    )

    standardized = validate_and_standardize_labels(df)

    assert standardized["label"].to_list() == ["NREM", "Wake", "Wake"]
    assert standardized["raw_label"].to_list() == [
        "NREM_SWS",
        "artifact",
        "wake_quiet",
    ]

    restandardized = validate_and_standardize_labels(standardized)
    assert restandardized["raw_label"].to_list() == standardized["raw_label"].to_list()


def test_default_hsmm_transition_rules_disable_self_and_penalize_unlikely_paths() -> None:
    _, logA, _, _ = build_hsmm_components(HSMMConfig(), 1.0, ["NREM", "REM", "Wake"])
    idx = {"NREM": 0, "REM": 1, "Wake": 2}

    assert np.all(np.isneginf(np.diag(logA)))
    assert np.exp(logA).sum(axis=1) == pytest.approx(np.ones(3))
    assert (
        logA[idx["Wake"], idx["REM"]]
        < logA[idx["Wake"], idx["NREM"]] - 7.5
    )
    assert (
        logA[idx["REM"], idx["NREM"]]
        < logA[idx["REM"], idx["Wake"]] - 7.5
    )


def test_validate_and_standardize_labels_rejects_is_and_unknown_labels() -> None:
    df = pl.DataFrame(
        {
            "start_s": [0.0, 1.0],
            "end_s": [1.0, 2.0],
            "label": ["IS", "mystery_state"],
        }
    )

    with pytest.raises(ValueError, match="obsolete labels are not supported anymore"):
        validate_and_standardize_labels(df)


def test_train_model_requires_all_three_states(monkeypatch: pytest.MonkeyPatch) -> None:
    session_ids = ["alioth_exp_1_sb-1", "avior_exp_1_sb-1", "kaus_exp_1_sb-1"]
    features_df, edges_by_session = _make_synthetic_feature_bundle(session_ids)
    monkeypatch.setattr(
        model_mod,
        "build_features",
        lambda sessions, config: (features_df, edges_by_session),
    )

    sessions = [{"session_id": session_id} for session_id in session_ids]
    labels_by_session = {
        session_id: validate_and_standardize_labels(_make_labels(wake_label="REM"))
        for session_id in session_ids
    }

    with pytest.raises(ValueError, match="Training labels must include all states"):
        model_mod.train_model(
            sessions,
            labels_by_session,
            config=_make_training_config(calibration_cv=2),
        )


def test_train_model_excludes_artifact_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_ids = ["alioth_exp_1_sb-1", "avior_exp_1_sb-1", "kaus_exp_1_sb-1"]
    features_df, edges_by_session = _make_synthetic_feature_bundle(session_ids)
    monkeypatch.setattr(
        model_mod,
        "build_features",
        lambda sessions, config: (features_df, edges_by_session),
    )

    captured: dict[str, np.ndarray] = {}
    original_compute_weights = model_mod._compute_class_balanced_sample_weights

    def _capture_y(y: np.ndarray, n_classes: int) -> np.ndarray:
        captured["y"] = y.copy()
        return original_compute_weights(y, n_classes)

    monkeypatch.setattr(
        model_mod,
        "_compute_class_balanced_sample_weights",
        _capture_y,
    )

    sessions = [{"session_id": session_id} for session_id in session_ids]
    labels_by_session = {
        session_id: validate_and_standardize_labels(_make_labels_with_artifact_and_wake())
        for session_id in session_ids
    }

    model = model_mod.train_model(
        sessions,
        labels_by_session,
        config=_make_training_config(calibration_cv=2),
        exclude_artifact=True,
    )

    assert model["exclude_artifact"] is True
    assert len(captured["y"]) == 15


def test_train_model_excluding_artifact_can_remove_all_wake_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_ids = ["alioth_exp_1_sb-1", "avior_exp_1_sb-1", "kaus_exp_1_sb-1"]
    features_df, edges_by_session = _make_synthetic_feature_bundle(session_ids)
    monkeypatch.setattr(
        model_mod,
        "build_features",
        lambda sessions, config: (features_df, edges_by_session),
    )

    sessions = [{"session_id": session_id} for session_id in session_ids]
    labels_by_session = {
        session_id: validate_and_standardize_labels(_make_labels())
        for session_id in session_ids
    }

    with pytest.raises(ValueError, match="Training labels must include all states"):
        model_mod.train_model(
            sessions,
            labels_by_session,
            config=_make_training_config(calibration_cv=2),
            exclude_artifact=True,
        )


def test_prepare_matrix_preserves_missing_values() -> None:
    df = pl.DataFrame(
        {
            "session_id": ["kaus_exp_2_sb-2"] * 3,
            "epoch_idx": [0, 1, 2],
            "start_s": [0.0, 1.0, 2.0],
            "end_s": [1.0, 2.0, 3.0],
            "feat_a": [1.0, None, 3.0],
            "feat_b": [2.0, 4.0, 6.0],
        }
    )

    prepared = model_mod._prepare_matrix(df, ["feat_a", "feat_b"])

    assert prepared["feat_a"].null_count() == 1
    assert prepared["feat_a"][1] is None


def test_prepare_matrix_handles_all_null_feature_columns() -> None:
    df = pl.DataFrame(
        {
            "session_id": ["sargas_exp_3_sb-1"] * 3,
            "epoch_idx": [0, 1, 2],
            "start_s": [0.0, 1.0, 2.0],
            "end_s": [1.0, 2.0, 3.0],
            "feat_all_null": pl.Series([None, None, None], dtype=pl.Null),
            "feat_b": [2.0, 4.0, 6.0],
        }
    )

    prepared = model_mod._prepare_matrix(df, ["feat_all_null", "feat_b"])

    assert prepared["feat_all_null"].dtype == pl.Float64
    assert prepared["feat_all_null"].null_count() == 3


def test_profile_train_model_reports_stage_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_ids = ["alioth_exp_1_sb-1", "avior_exp_1_sb-1", "kaus_exp_1_sb-1"]
    features_df, edges_by_session = _make_synthetic_feature_bundle(session_ids)
    monkeypatch.setattr(
        model_mod,
        "build_features",
        lambda sessions, config: (features_df, edges_by_session),
    )

    sessions = [{"session_id": session_id} for session_id in session_ids]
    labels_by_session = {
        session_id: validate_and_standardize_labels(_make_labels())
        for session_id in session_ids
    }

    profile = model_mod.profile_train_model(
        sessions,
        labels_by_session,
        config=_make_training_config(calibration_cv=2),
        verbose=False,
    )

    stages = set(profile["timings_df"]["stage"].to_list())
    assert stages == {
        "build_features",
        "label_to_epoch_labels",
        "join_features_labels",
        "prepare_matrix",
        "drop_zero_variance_features",
        "fit_calibrator",
        "compute_feature_importance",
    }
    assert profile["metadata"]["n_sessions"] == 3
    assert profile["metadata"]["n_labeled_epochs"] == 18
    assert profile["metadata"]["calibration_cv"] == 2
    assert profile["metadata"]["fit_n_jobs"] == 4
    assert profile["metadata"]["calibration_outer_n_jobs"] == 1
    assert profile["metadata"]["n_features_dropped"] >= 1


def test_profile_train_model_reports_reduced_labeled_epochs_when_excluding_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_ids = ["alioth_exp_1_sb-1", "avior_exp_1_sb-1", "kaus_exp_1_sb-1"]
    features_df, edges_by_session = _make_synthetic_feature_bundle(session_ids)
    monkeypatch.setattr(
        model_mod,
        "build_features",
        lambda sessions, config: (features_df, edges_by_session),
    )

    sessions = [{"session_id": session_id} for session_id in session_ids]
    labels_by_session = {
        session_id: validate_and_standardize_labels(_make_labels_with_artifact_and_wake())
        for session_id in session_ids
    }

    profile = model_mod.profile_train_model(
        sessions,
        labels_by_session,
        config=_make_training_config(calibration_cv=2),
        exclude_artifact=True,
        verbose=False,
    )

    assert profile["metadata"]["n_labeled_epochs"] == 15
    assert profile["metadata"]["exclude_artifact"] is True


def test_train_model_drops_zero_variance_columns_and_keeps_importance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_ids = ["alioth_exp_1_sb-1", "avior_exp_1_sb-1", "kaus_exp_1_sb-1"]
    features_df, edges_by_session = _make_synthetic_feature_bundle(session_ids)
    monkeypatch.setattr(
        model_mod,
        "build_features",
        lambda sessions, config: (features_df, edges_by_session),
    )

    sessions = [{"session_id": session_id} for session_id in session_ids]
    labels_by_session = {
        session_id: validate_and_standardize_labels(_make_labels())
        for session_id in session_ids
    }

    model = model_mod.train_model(
        sessions,
        labels_by_session,
        config=_make_training_config(calibration_cv=2),
    )

    assert "feat_constant" not in model["feature_cols"]
    assert "feat_constant" in model["dropped_feature_cols"]
    assert set(model["feature_importance"]) == set(model["feature_cols"])


def test_score_session_emits_only_three_state_probability_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_ids = ["alioth_exp_1_sb-1", "avior_exp_1_sb-1", "kaus_exp_1_sb-1"]
    features_df, edges_by_session = _make_synthetic_feature_bundle(session_ids)
    monkeypatch.setattr(
        model_mod,
        "build_features",
        lambda sessions, config: (features_df, edges_by_session),
    )
    monkeypatch.setattr(
        model_mod,
        "decode_session",
        lambda log_probs, hsmm_cfg, epoch_len_s, state_names: (
            np.array([0, 1, 2], dtype=int),
            np.array([2, 2, 2], dtype=int),
            np.array([0, 2, 4], dtype=int),
        ),
    )

    sessions = [{"session_id": session_id} for session_id in session_ids]
    labels_by_session = {
        session_id: validate_and_standardize_labels(_make_labels())
        for session_id in session_ids
    }
    model = model_mod.train_model(
        sessions,
        labels_by_session,
        config=_make_training_config(),
    )

    bout_df, epoch_df = model_mod.score_session(
        model,
        {"session_id": session_ids[0]},
    )

    assert "P_IS" not in epoch_df.columns
    assert "mean_P_IS" not in bout_df.columns
    assert {"P_NREM", "P_REM", "P_Wake"}.issubset(epoch_df.columns)
    assert {"mean_P_NREM", "mean_P_REM", "mean_P_Wake"}.issubset(bout_df.columns)
    assert set(VIDEO_QUALITY_FEATURES).issubset(epoch_df.columns)
    assert {f"mean_{name}" for name in VIDEO_QUALITY_FEATURES}.issubset(bout_df.columns)
    assert epoch_df["pred_label"].to_list() == ["NREM", "NREM", "REM", "REM", "Wake", "Wake"]


def test_score_sync_block_skips_when_same_model_already_scored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    production_dir = tmp_path / "production_model"
    production_dir.mkdir()
    model_path = production_dir / "SLAP_SCORE_V1.pkl"
    model_path.write_bytes(b"model-v1")

    anmat_root = tmp_path / "analysis_materials"
    monkeypatch.setattr(score_all_mod.DEFS, "anmat_root", str(anmat_root))

    hypnogram_dir = score_all_mod._sync_block_hypnogram_dir("avior", "exp_1", 1)
    model_scored_dir = hypnogram_dir / score_all_mod.MODEL_SCORED_DIRNAME
    model_scored_dir.mkdir(parents=True)
    pl.DataFrame({"epoch_idx": [0], "pred_label": ["Wake"]}).write_parquet(
        model_scored_dir / score_all_mod.EPOCH_DF_FILENAME
    )
    pl.DataFrame({"label": ["Wake"], "start_s": [0.0], "end_s": [30.0]}).write_parquet(
        model_scored_dir / score_all_mod.BOUT_DF_FILENAME
    )
    pl.DataFrame({"start_s": [0.0], "end_s": [30.0], "label": ["Wake"]}).write_csv(
        model_scored_dir / score_all_mod.EPOCH_HYPNO_FILENAME
    )
    pl.DataFrame({"start_s": [0.0], "end_s": [30.0], "label": ["Wake"]}).write_csv(
        model_scored_dir / score_all_mod.BOUT_HYPNO_FILENAME
    )
    score_all_mod._write_model_version(
        model_scored_dir,
        score_all_mod._model_signature(model_path),
        score_all_mod._scoring_signature(0.8),
    )

    def _unexpected_load_model(path):
        raise AssertionError("score_sync_block should skip without loading the model")

    monkeypatch.setattr(score_all_mod, "load_model", _unexpected_load_model)

    result = score_all_mod.score_sync_block(
        "avior",
        "exp_1",
        1,
        production_model_dir=production_dir,
        verbose=False,
    )

    assert result.status == "skipped"
    assert result.reason == "same_model"
    assert result.model_name == "SLAP_SCORE_V1.pkl"


def test_score_sync_block_archives_old_model_outputs_and_saves_new(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    production_dir = tmp_path / "production_model"
    production_dir.mkdir()
    model_path = production_dir / "SLAP_SCORE_V2.pkl"
    model_path.write_bytes(b"model-v2")

    anmat_root = tmp_path / "analysis_materials"
    monkeypatch.setattr(score_all_mod.DEFS, "anmat_root", str(anmat_root))

    hypnogram_dir = score_all_mod._sync_block_hypnogram_dir("avior", "exp_1", 1)
    old_scored_dir = hypnogram_dir / score_all_mod.MODEL_SCORED_DIRNAME
    old_scored_dir.mkdir(parents=True)
    (old_scored_dir / "old_marker.txt").write_text("old outputs\n")
    (old_scored_dir / score_all_mod.MODEL_VERSION_FILENAME).write_text(
        "model_name=SLAP_SCORE_V1.pkl\nmodel_sha256=old-hash\n"
    )

    monkeypatch.setattr(
        score_all_mod,
        "load_scoring_times",
        lambda subject, exp, sync_block: ([0.0], [60.0]),
    )
    monkeypatch.setattr(
        score_all_mod,
        "create_session",
        lambda subject, exp, sync_block, t1, t2, store_chans: {
            "session_id": f"{subject}_{exp}_sb-{sync_block}",
            "t1": t1,
            "t2": t2,
        },
    )
    monkeypatch.setattr(
        score_all_mod,
        "load_model",
        lambda path: {"model_path": str(path)},
    )
    monkeypatch.setattr(
        score_all_mod,
        "score_session",
        lambda model, session: (
            pl.DataFrame({"label": ["NREM"], "start_s": [0.0], "end_s": [4.0]}),
            pl.DataFrame(
                {
                    "epoch_idx": [0, 1, 2, 3],
                    "session_id": ["avior_exp_1_sb-1"] * 4,
                    "start_s": [0.0, 1.0, 2.0, 3.0],
                    "end_s": [1.0, 2.0, 3.0, 4.0],
                    "pred_label": ["NREM", "NREM", "NREM", "NREM"],
                    "P_NREM": [0.8, 0.7, 0.1, 0.2],
                    "P_REM": [0.1, 0.1, 0.1, 0.1],
                    "P_Wake": [0.1, 0.2, 0.8, 0.85],
                }
            ),
        ),
    )

    result = score_all_mod.score_sync_block(
        "avior",
        "exp_1",
        1,
        production_model_dir=production_dir,
        verbose=False,
    )

    assert result.status == "scored"
    assert result.archived_dir is not None
    assert (result.archived_dir / "old_marker.txt").is_file()
    assert pl.read_parquet(result.epoch_path)["pred_label"].to_list() == [
        "NREM",
        "NREM",
        "NREM",
        "NREM",
    ]
    saved_bout_df = pl.read_parquet(result.bout_path)
    assert saved_bout_df["label"].to_list() == ["NREM", "Wake"]
    assert saved_bout_df["start_s"].to_list() == [0.0, 2.0]
    assert saved_bout_df["end_s"].to_list() == [2.0, 4.0]
    assert {"mean_P_NREM", "mean_P_REM", "mean_P_Wake"}.issubset(
        saved_bout_df.columns
    )
    assert pl.read_csv(result.epoch_hypno_path).to_dict(as_series=False) == {
        "start_s": [0.0, 2.0],
        "end_s": [2.0, 4.0],
        "label": ["NREM", "Wake"],
    }
    assert pl.read_csv(result.bout_hypno_path).to_dict(as_series=False) == {
        "start_s": [0.0, 2.0],
        "end_s": [2.0, 4.0],
        "label": ["NREM", "Wake"],
    }
    model_version_text = (
        result.model_scored_dir / score_all_mod.MODEL_VERSION_FILENAME
    ).read_text()
    assert "model_name=SLAP_SCORE_V2.pkl" in model_version_text
    assert "add_wake_epochs=0.8" in model_version_text


def test_wake_epoch_insertion_can_be_disabled() -> None:
    bout_df = pl.DataFrame(
        {
            "session_id": ["avior_exp_1_sb-1"],
            "start_s": [0.0],
            "end_s": [3.0],
            "label": ["NREM"],
            "n_epochs": [3],
        }
    )
    epoch_df = pl.DataFrame(
        {
            "session_id": ["avior_exp_1_sb-1"] * 3,
            "start_s": [0.0, 1.0, 2.0],
            "end_s": [1.0, 2.0, 3.0],
            "pred_label": ["NREM", "NREM", "NREM"],
            "P_Wake": [0.1, 0.95, 0.96],
        }
    )

    patched = score_all_mod._insert_high_probability_wake_epochs(
        bout_df,
        epoch_df,
        add_wake_epochs=False,
    )

    assert patched.to_dict(as_series=False) == bout_df.to_dict(as_series=False)


def test_score_all_subjects_uses_acquisition_master_and_skips_missing_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    anmat_root = tmp_path / "analysis_materials"
    data_root = tmp_path / "raw_data"
    monkeypatch.setattr(score_all_mod.DEFS, "anmat_root", str(anmat_root))
    monkeypatch.setattr(score_all_mod.DEFS, "data_root", str(data_root))

    monkeypatch.setattr(
        score_all_mod.wis.meta.get,
        "acq_master",
        lambda: {
            "avior": {
                "exp_1": ["loc_Z--acq_1", "loc_I--acq_1"],
                "exp_2": ["loc_C--acq_2"],
            }
        },
    )
    monkeypatch.setattr(
        score_all_mod.wis.meta.get,
        "sync_info",
        lambda: {
            "avior": {
                "exp_1": {
                    "acquisitions": {
                        "loc_Z--acq_1": {"sync_block": 1},
                        "loc_I--acq_1": {"sync_block": 1},
                    },
                    "sync_blocks": {1: {"corrupt": False}},
                },
                "exp_2": {
                    "acquisitions": {"loc_C--acq_2": {"sync_block": 2}},
                    "sync_blocks": {2: {"corrupt": False}},
                },
            }
        },
    )

    raw_ephys_dir = data_root / "avior" / "exp_1" / "ephys" / "ephys-1"
    eye_dir = (
        anmat_root
        / "avior"
        / "exp_1"
        / "sync_block_data"
        / "sync_block-1"
        / "pupil"
        / "eye_metrics"
    )
    whisk_dir = (
        anmat_root
        / "avior"
        / "exp_1"
        / "sync_block_data"
        / "sync_block-1"
        / "whisking"
    )
    raw_ephys_dir.mkdir(parents=True)
    eye_dir.mkdir(parents=True)
    whisk_dir.mkdir(parents=True)
    (eye_dir / "eye_metrics.parquet").write_bytes(b"exists")
    (whisk_dir / "whisk_df.parquet").write_bytes(b"exists")

    calls = []

    def _score_sync_block(
        subject,
        exp,
        sync_block,
        *,
        production_model_dir,
        store_chans,
        add_wake_epochs,
        force,
        verbose,
    ):
        calls.append((subject, exp, sync_block, add_wake_epochs, force))
        scored_dir = (
            anmat_root
            / subject
            / exp
            / "sync_block_data"
            / f"sync_block-{sync_block}"
            / "hypnograms"
            / "model_scored"
        )
        return score_all_mod.ScoreSyncBlockResult(
            subject=subject,
            exp=exp,
            sync_block=sync_block,
            status="scored",
            model_name="SLAP_SCORE_TEST.pkl",
            model_path=tmp_path / "SLAP_SCORE_TEST.pkl",
            hypnogram_dir=scored_dir.parent,
            model_scored_dir=scored_dir,
            epoch_path=scored_dir / score_all_mod.EPOCH_DF_FILENAME,
            bout_path=scored_dir / score_all_mod.BOUT_DF_FILENAME,
            epoch_hypno_path=scored_dir / score_all_mod.EPOCH_HYPNO_FILENAME,
            bout_hypno_path=scored_dir / score_all_mod.BOUT_HYPNO_FILENAME,
        )

    monkeypatch.setattr(score_all_mod, "score_sync_block", _score_sync_block)

    summary = score_all_mod.score_all_subjects(verbose=False)

    assert calls == [("avior", "exp_1", 1, 0.8, False)]
    rows = {
        (row["exp"], row["sync_block"]): row
        for row in summary.to_dicts()
    }
    assert rows[("exp_1", 1)]["status"] == "scored"
    assert rows[("exp_1", 1)]["source_acq_ids"] == "loc_Z--acq_1,loc_I--acq_1"
    assert rows[("exp_2", 2)]["status"] == "skipped_missing_data"
    assert "raw_ephys" in rows[("exp_2", 2)]["missing_data"]


def test_extract_features_for_session_masks_unreliable_video_frames() -> None:
    session, edges = _make_video_mask_session()
    config = ScoreConfig(feature_windows_s=[1.0])

    feats = features_mod.extract_features_for_session(session, edges, config)

    epoch0 = feats.row(0, named=True)
    epoch1 = feats.row(1, named=True)
    epoch2 = feats.row(2, named=True)
    epoch3 = feats.row(3, named=True)

    assert epoch0["pu_diam_mean"] is None
    assert epoch0["pu_motion_mean"] is None
    assert epoch0["pu_eyelid_mean"] is not None
    assert epoch0["whisk_mean"] is not None
    assert epoch0["pu_valid_frac"] == pytest.approx(0.0)
    assert epoch0["eyelid_valid_frac"] == pytest.approx(1.0)
    assert epoch0["camera_off_epoch"] == pytest.approx(0.0)

    assert epoch1["pu_diam_mean"] is not None
    assert epoch1["pu_eyelid_mean"] is None
    assert epoch1["pu_valid_frac"] == pytest.approx(1.0)
    assert epoch1["eyelid_valid_frac"] == pytest.approx(0.0)
    assert epoch1["whisk_valid_frac"] == pytest.approx(1.0)

    assert epoch2["pu_diam_mean"] is None
    assert epoch2["pu_eyelid_mean"] is None
    assert epoch2["whisk_mean"] is None
    assert epoch2["whisk_valid_frac"] == pytest.approx(0.0)
    assert epoch2["camera_off_frac"] == pytest.approx(1.0)
    assert epoch2["camera_off_epoch"] == pytest.approx(1.0)

    assert epoch3["pu_diam_mean"] is not None
    assert epoch3["pu_valid_frac"] == pytest.approx(0.5)
    assert epoch3["eyelid_valid_frac"] == pytest.approx(1.0)
    assert epoch3["camera_off_frac"] == pytest.approx(0.0)


def test_evaluate_loso_uses_three_state_confusion_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions_by_session = {
        "alioth_exp_1_sb-1": np.array([0, 0, 1, 1, 2, 2], dtype=int),
        "avior_exp_1_sb-1": np.array([0, 0, 1, 1, 2, 2], dtype=int),
    }

    def _score_session(model: dict, session: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
        return (
            pl.DataFrame(),
            pl.DataFrame(
                {
                    "pred_state_idx": predictions_by_session[session["session_id"]].tolist()
                }
            ),
        )

    monkeypatch.setattr("wisco_slap.pns.sleepscore._model.train_model", lambda *args, **kwargs: {})
    monkeypatch.setattr("wisco_slap.pns.sleepscore._model.score_session", _score_session)

    timestamps = np.arange(7, dtype=float)
    sessions = [
        {
            "session_id": "alioth_exp_1_sb-1",
            "eeg": {"timestamps": timestamps},
            "pupil": {"timestamps": timestamps},
        },
        {
            "session_id": "avior_exp_1_sb-1",
            "eeg": {"timestamps": timestamps},
            "pupil": {"timestamps": timestamps},
        },
    ]
    labels_by_session = {
        session["session_id"]: validate_and_standardize_labels(_make_labels())
        for session in sessions
    }

    results = evaluate_loso(sessions, labels_by_session, verbose=False)

    assert results["overall"]["confusion_matrix"].shape == (3, 3)
    assert set(results["overall"]["per_class"]) == {"NREM", "REM", "Wake"}


def test_evaluate_scored_session_returns_metrics_and_figures() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    ground_truth_labels = pl.DataFrame(
        {
            "start_s": [0.0, 2.0, 4.0],
            "end_s": [2.0, 4.0, 6.0],
            "label": ["NREM_SWS", "REM", "Artifact"],
        }
    )
    epoch_df, bout_df = _make_epoch_and_bout_outputs_with_single_error()

    results = evaluate_scored_session(
        ground_truth_labels,
        epoch_df,
        bout_df,
        make_plots=True,
        show_plots=False,
        print_summary=False,
    )

    assert results["epoch"]["metrics"]["accuracy"] == pytest.approx(5 / 6)
    assert results["bout"]["metrics"]["accuracy"] == pytest.approx(5 / 6)
    assert results["epoch"]["mismatches_df"].height == 1
    assert results["bout"]["mismatches_df"]["duration_s"].sum() == pytest.approx(1.0)
    assert results["ground_truth_bouts_df"]["label"].to_list() == ["NREM", "REM", "Wake"]
    assert set(results["figures"]) == {"confusion_matrices", "timeline"}
