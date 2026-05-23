"""LightGBM classifier training, calibration, and persistence."""

from __future__ import annotations

import cProfile
import io
import pickle
import pstats
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from ._config import ScoreConfig, VIDEO_QUALITY_FEATURES
from ._data import build_epoch_grid, intervals_to_epoch_labels, prepare_training_labels
from ._features import add_context_features, build_features, extract_features_for_session
from ._hsmm import decode_session
from ._utils import robust_scale_per_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Return the list of numeric feature columns (everything except metadata)."""
    meta = {"session_id", "epoch_idx", "start_s", "end_s", "y"}
    return [c for c in df.columns if c not in meta]


def _prepare_matrix(
    df: pl.DataFrame,
    feature_cols: list[str],
) -> pl.DataFrame:
    """Robust-scale features per session while preserving missing values."""
    return robust_scale_per_session(df, feature_cols)


def _compute_class_balanced_sample_weights(
    y: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Return class-balanced sample weights for an integer label array."""
    class_counts = np.bincount(y, minlength=n_classes).astype(float)
    inv = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    sample_weight = inv[y]
    return sample_weight * (len(y) / (np.sum(sample_weight) + 1e-12))


def _make_lgbm_classifier(clf_cfg: Any) -> LGBMClassifier:
    """Return a LightGBM classifier with the configured thread cap."""
    return LGBMClassifier(
        **clf_cfg.lgbm_params,
        random_state=clf_cfg.random_state,
        n_jobs=clf_cfg.fit_n_jobs,
    )


def _resolve_calibration_cv(y: np.ndarray, requested_cv: int) -> int:
    """Clamp calibration CV to the smallest class count and require at least 2 folds."""
    class_counts = np.bincount(y).astype(int)
    min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
    resolved_cv = min(int(requested_cv), min_class_count)
    if resolved_cv < 2:
        raise ValueError(
            "Calibration requires at least 2 labeled epochs in each state; "
            f"smallest class has {min_class_count} labeled epochs."
        )
    return resolved_cv


def _drop_zero_variance_features(
    X: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Drop columns with zero or non-finite variance from a design matrix."""
    with np.errstate(invalid="ignore"):
        col_var = np.nanvar(X, axis=0)
    keep_mask = np.isfinite(col_var) & (~np.isclose(col_var, 0.0))

    kept_feature_cols = [
        col for col, keep in zip(feature_cols, keep_mask, strict=False) if keep
    ]
    dropped_feature_cols = [
        col for col, keep in zip(feature_cols, keep_mask, strict=False) if not keep
    ]
    if not kept_feature_cols:
        raise ValueError("All training features had zero variance after preprocessing.")
    return X[:, keep_mask], kept_feature_cols, dropped_feature_cols


def _compute_feature_importance_from_calibrator(
    calibrator: CalibratedClassifierCV,
    feature_cols: list[str],
) -> dict[str, float]:
    """Average feature importances across the estimators already fit for calibration."""
    importance_rows: list[np.ndarray] = []
    for calibrated in getattr(calibrator, "calibrated_classifiers_", []):
        estimator = getattr(calibrated, "estimator", None)
        if estimator is None or not hasattr(estimator, "feature_importances_"):
            continue
        values = np.asarray(estimator.feature_importances_, dtype=float)
        if values.shape[0] != len(feature_cols):
            continue
        importance_rows.append(values)

    if not importance_rows:
        return {}

    mean_importance = np.mean(np.vstack(importance_rows), axis=0)
    return dict(zip(feature_cols, mean_importance.tolist(), strict=False))


def _make_timing_df(stage_timings: list[tuple[str, float]]) -> pl.DataFrame:
    """Convert a stage-timing list into a summary DataFrame."""
    total_s = float(sum(seconds for _, seconds in stage_timings))
    return pl.DataFrame(
        {
            "stage": [name for name, _ in stage_timings],
            "seconds": [seconds for _, seconds in stage_timings],
            "pct_total": [
                (100.0 * seconds / total_s) if total_s > 0 else 0.0
                for _, seconds in stage_timings
            ],
        }
    )


def _print_timing_df(timing_df: pl.DataFrame, *, title: str) -> None:
    """Pretty-print a timing summary DataFrame."""
    print(f"\n{title}")
    print("-" * len(title))
    print(
        timing_df.with_columns(
            pl.col("seconds").round(3),
            pl.col("pct_total").round(1),
        )
    )


def _profile_feature_pipeline(
    sessions: list[dict],
    config: ScoreConfig,
) -> tuple[pl.DataFrame, dict[str, np.ndarray], pl.DataFrame, pl.DataFrame]:
    """Build features while timing each internal feature-generation stage."""
    breakdown: list[tuple[str, float]] = []

    t0 = time.perf_counter()
    edges_by_session = build_epoch_grid(sessions, config.epoch_len_s)
    breakdown.append(("build_epoch_grid", time.perf_counter() - t0))

    feat_frames: list[pl.DataFrame] = []
    session_rows: list[dict[str, float | str]] = []
    for sess in sessions:
        sid = sess["session_id"]
        t0 = time.perf_counter()
        feats = extract_features_for_session(sess, edges_by_session[sid], config)
        elapsed = time.perf_counter() - t0
        session_rows.append(
            {
                "session_id": sid,
                "seconds": elapsed,
                "n_epochs": float(feats.height),
            }
        )
        feat_frames.append(feats)

    per_session_df = pl.DataFrame(session_rows) if session_rows else pl.DataFrame()
    breakdown.append(
        (
            "extract_features_for_session",
            float(per_session_df["seconds"].sum()) if not per_session_df.is_empty() else 0.0,
        )
    )

    t0 = time.perf_counter()
    feats_df = pl.concat(feat_frames) if feat_frames else pl.DataFrame()
    breakdown.append(("concat_feature_frames", time.perf_counter() - t0))

    t0 = time.perf_counter()
    feats_df = add_context_features(feats_df)
    breakdown.append(("add_context_features", time.perf_counter() - t0))

    return feats_df, edges_by_session, _make_timing_df(breakdown), per_session_df


def _run_build_features_cprofile(
    sessions: list[dict],
    config: ScoreConfig,
    *,
    top_n: int = 30,
) -> str:
    """Return cProfile stats text for ``build_features``."""
    profiler = cProfile.Profile()
    profiler.enable()
    build_features(sessions, config)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(top_n)
    return stream.getvalue()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    sessions: list[dict],
    labels_by_session: dict[str, pl.DataFrame],
    config: ScoreConfig | None = None,
    *,
    exclude_artifact: bool = False,
) -> dict[str, Any]:
    """Train a sleep scoring model (LightGBM + calibration + HSMM config).

    Parameters
    ----------
    sessions : list of session dicts (from ``create_session``).
    labels_by_session : dict mapping session_id to polars DataFrame with
        columns ``start_s``, ``end_s``, ``label`` (already validated).
    config : scoring configuration; uses defaults if None.
    exclude_artifact : if True, exclude Artifact-labeled intervals from training.

    Returns
    -------
    Model dict containing the calibrated classifier, feature columns,
    HSMM configuration, and all metadata needed for inference.
    """
    if config is None:
        config = ScoreConfig()

    clf_cfg = config.classifier
    state_names = config.states

    # --- Features ---
    feats_df, edges_by_session = build_features(sessions, config)

    # --- Labels → epoch-level targets ---
    y_parts: list[pl.DataFrame] = []
    for sess in sessions:
        sid = sess["session_id"]
        labels_df = prepare_training_labels(
            labels_by_session.get(sid, None),
            exclude_artifact=exclude_artifact,
        )
        y = intervals_to_epoch_labels(
            edges_by_session[sid], labels_df, config.min_overlap_frac,
        )
        y_parts.append(pl.DataFrame({
            "session_id": [sid] * len(y),
            "epoch_idx": list(range(len(y))),
            "y": y.tolist(),
        }))
    y_all = pl.concat(y_parts)

    # --- Merge features + labels ---
    df = feats_df.join(y_all, on=["session_id", "epoch_idx"], how="left")

    feature_cols = _get_feature_cols(df)
    df = _prepare_matrix(df, feature_cols)

    # --- Extract labeled subset ---
    y = df["y"].to_numpy()
    labeled_mask = y >= 0
    if not np.any(labeled_mask):
        raise ValueError("No labeled epochs found for training.")

    df_labeled = df.filter(pl.col("y") >= 0)
    X_lab = df_labeled.select(feature_cols).to_numpy().astype(float)
    y_lab = df_labeled["y"].to_numpy()

    # Check all required states are present
    present = set(np.unique(y_lab).tolist())
    required = set(range(len(state_names)))
    if present != required:
        present_names = [state_names[i] for i in sorted(present)]
        required_names = state_names
        raise ValueError(
            f"Training labels must include all states {required_names}. "
            f"Present: {present_names}"
        )

    n_classes = len(state_names)
    sample_weight = _compute_class_balanced_sample_weights(y_lab, n_classes)
    calibration_cv = _resolve_calibration_cv(y_lab, clf_cfg.calibration_cv)
    X_lab, feature_cols, dropped_feature_cols = _drop_zero_variance_features(
        X_lab, feature_cols
    )

    # --- Calibration ---
    calibrator = CalibratedClassifierCV(
        estimator=_make_lgbm_classifier(clf_cfg),
        method=clf_cfg.calibration_method,
        cv=calibration_cv,
        n_jobs=clf_cfg.calibration_outer_n_jobs,
    )
    calibrator.fit(X_lab, y_lab, sample_weight=sample_weight)

    # --- Feature importance ---
    importance = _compute_feature_importance_from_calibrator(
        calibrator, feature_cols
    )

    # --- Assemble model dict ---
    model: dict[str, Any] = {
        "epoch_len_s": config.epoch_len_s,
        "feature_cols": feature_cols,
        "hsmm_cfg": config.hsmm,
        "calibrated_clf": calibrator,
        "state_names": state_names,
        "feature_importance": importance,
        "calibration_cv": calibration_cv,
        "dropped_feature_cols": dropped_feature_cols,
        "exclude_artifact": exclude_artifact,
        "config": config,
    }
    return model


def profile_train_model(
    sessions: list[dict],
    labels_by_session: dict[str, pl.DataFrame],
    config: ScoreConfig | None = None,
    *,
    exclude_artifact: bool = False,
    return_model: bool = False,
    feature_breakdown: bool = False,
    cprofile_build_features: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Profile the main stages of ``train_model``.

    This uses the same internal helpers as ``train_model`` and returns a compact
    timing summary so notebook debugging can quickly identify the dominant cost.
    """
    if config is None:
        config = ScoreConfig()

    clf_cfg = config.classifier
    state_names = config.states
    stage_timings: list[tuple[str, float]] = []
    feature_breakdown_df: pl.DataFrame | None = None
    feature_session_df: pl.DataFrame | None = None

    # --- Features ---
    t0 = time.perf_counter()
    if feature_breakdown:
        feats_df, edges_by_session, feature_breakdown_df, feature_session_df = (
            _profile_feature_pipeline(sessions, config)
        )
    else:
        feats_df, edges_by_session = build_features(sessions, config)
    stage_timings.append(("build_features", time.perf_counter() - t0))

    # --- Labels → epoch-level targets ---
    t0 = time.perf_counter()
    y_parts: list[pl.DataFrame] = []
    for sess in sessions:
        sid = sess["session_id"]
        labels_df = prepare_training_labels(
            labels_by_session.get(sid, None),
            exclude_artifact=exclude_artifact,
        )
        y = intervals_to_epoch_labels(
            edges_by_session[sid], labels_df, config.min_overlap_frac,
        )
        y_parts.append(
            pl.DataFrame(
                {
                    "session_id": [sid] * len(y),
                    "epoch_idx": list(range(len(y))),
                    "y": y.tolist(),
                }
            )
        )
    y_all = pl.concat(y_parts)
    stage_timings.append(("label_to_epoch_labels", time.perf_counter() - t0))

    # --- Merge features + labels ---
    t0 = time.perf_counter()
    df = feats_df.join(y_all, on=["session_id", "epoch_idx"], how="left")
    stage_timings.append(("join_features_labels", time.perf_counter() - t0))

    feature_cols = _get_feature_cols(df)

    # --- Scaling / matrix prep ---
    t0 = time.perf_counter()
    df = _prepare_matrix(df, feature_cols)
    stage_timings.append(("prepare_matrix", time.perf_counter() - t0))

    # --- Extract labeled subset ---
    y = df["y"].to_numpy()
    labeled_mask = y >= 0
    if not np.any(labeled_mask):
        raise ValueError("No labeled epochs found for training.")

    df_labeled = df.filter(pl.col("y") >= 0)
    X_lab = df_labeled.select(feature_cols).to_numpy().astype(float)
    y_lab = df_labeled["y"].to_numpy()

    present = set(np.unique(y_lab).tolist())
    required = set(range(len(state_names)))
    if present != required:
        present_names = [state_names[i] for i in sorted(present)]
        raise ValueError(
            f"Training labels must include all states {state_names}. "
            f"Present: {present_names}"
        )

    n_classes = len(state_names)
    sample_weight = _compute_class_balanced_sample_weights(y_lab, n_classes)
    calibration_cv = _resolve_calibration_cv(y_lab, clf_cfg.calibration_cv)
    t0 = time.perf_counter()
    X_lab, feature_cols, dropped_feature_cols = _drop_zero_variance_features(
        X_lab, feature_cols
    )
    stage_timings.append(("drop_zero_variance_features", time.perf_counter() - t0))

    # --- Calibrated fit ---
    calibrator = CalibratedClassifierCV(
        estimator=_make_lgbm_classifier(clf_cfg),
        method=clf_cfg.calibration_method,
        cv=calibration_cv,
        n_jobs=clf_cfg.calibration_outer_n_jobs,
    )
    t0 = time.perf_counter()
    calibrator.fit(X_lab, y_lab, sample_weight=sample_weight)
    stage_timings.append(("fit_calibrator", time.perf_counter() - t0))

    t0 = time.perf_counter()
    importance = _compute_feature_importance_from_calibrator(
        calibrator, feature_cols
    )
    stage_timings.append(("compute_feature_importance", time.perf_counter() - t0))
    model: dict[str, Any] = {
        "epoch_len_s": config.epoch_len_s,
        "feature_cols": feature_cols,
        "hsmm_cfg": config.hsmm,
        "calibrated_clf": calibrator,
        "state_names": state_names,
        "feature_importance": importance,
        "calibration_cv": calibration_cv,
        "dropped_feature_cols": dropped_feature_cols,
        "exclude_artifact": exclude_artifact,
        "config": config,
    }

    timings_df = _make_timing_df(stage_timings)
    metadata = {
        "n_sessions": len(sessions),
        "n_epochs_total": int(df.height),
        "n_labeled_epochs": int(df_labeled.height),
        "n_features": len(feature_cols),
        "n_features_dropped": len(dropped_feature_cols),
        "calibration_cv": calibration_cv,
        "fit_n_jobs": clf_cfg.fit_n_jobs,
        "calibration_outer_n_jobs": clf_cfg.calibration_outer_n_jobs,
        "exclude_artifact": exclude_artifact,
    }
    build_features_cprofile = (
        _run_build_features_cprofile(sessions, config)
        if cprofile_build_features
        else None
    )

    if verbose:
        _print_timing_df(timings_df, title="train_model stage timings")
        print(
            "\nmetadata\n--------\n"
            f"n_sessions={metadata['n_sessions']}, "
            f"n_epochs_total={metadata['n_epochs_total']}, "
            f"n_labeled_epochs={metadata['n_labeled_epochs']}, "
            f"n_features={metadata['n_features']}, "
            f"calibration_cv={metadata['calibration_cv']}"
        )
        if feature_breakdown and feature_breakdown_df is not None:
            _print_timing_df(feature_breakdown_df, title="build_features breakdown")
            if feature_session_df is not None and not feature_session_df.is_empty():
                print("\nper-session feature extraction")
                print("----------------------------")
                print(feature_session_df.with_columns(pl.col("seconds").round(3)))
        if build_features_cprofile is not None:
            print("\nbuild_features cProfile (top 30 by cumulative time)")
            print("---------------------------------------------------")
            print(build_features_cprofile)

    result: dict[str, Any] = {
        "timings_df": timings_df,
        "metadata": metadata,
        "feature_breakdown_df": feature_breakdown_df,
        "feature_session_df": feature_session_df,
        "build_features_cprofile": build_features_cprofile,
    }
    if return_model:
        result["model"] = model
    return result


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def score_session(
    model: dict[str, Any],
    session: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score a single session using a trained model.

    Parameters
    ----------
    model : trained model dict (from ``train_model``).
    session : session dict (from ``create_session``).

    Returns
    -------
    bout_df : polars DataFrame with one row per state bout.
    epoch_df : polars DataFrame with one row per epoch, including per-state probabilities.
    """
    config = model["config"]
    clf = model["calibrated_clf"]
    feature_cols = model["feature_cols"]
    state_names = model["state_names"]
    hsmm_cfg = model["hsmm_cfg"]
    epoch_len_s = model["epoch_len_s"]

    # --- Features ---
    feats_df, edges_by_session = build_features([session], config)

    sid = session["session_id"]
    raw_df = feats_df.filter(pl.col("session_id") == sid)
    df = _prepare_matrix(raw_df, feature_cols)

    # Ensure column order matches training
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in inference: {missing}")

    X = df.select(feature_cols).to_numpy().astype(float)

    # --- Predict probabilities ---
    P = clf.predict_proba(X)
    # Ensure column order matches state index ordering
    clf_classes = list(clf.classes_)
    n_states = len(state_names)
    if clf_classes != list(range(n_states)):
        order = [clf_classes.index(c) for c in range(n_states)]
        P = P[:, order]

    T = P.shape[0]
    eps = 1e-12
    log_probs = np.log(np.clip(P, eps, 1.0))

    # --- HSMM decoding ---
    states, durs, starts = decode_session(log_probs, hsmm_cfg, epoch_len_s, state_names)

    # Build per-epoch decoded state array
    dec = np.repeat(states, durs)[:T]
    if len(dec) < T:
        dec = np.concatenate([dec, np.full(T - len(dec), dec[-1])])

    # --- Epoch DataFrame ---
    epoch_data: dict[str, list] = {
        "session_id": df["session_id"].to_list(),
        "epoch_idx": df["epoch_idx"].to_list(),
        "start_s": df["start_s"].to_list(),
        "end_s": df["end_s"].to_list(),
        "pred_state_idx": dec.tolist(),
        "pred_label": [state_names[i] for i in dec],
    }
    for si, name in enumerate(state_names):
        epoch_data[f"P_{name}"] = P[:, si].tolist()
    for name in VIDEO_QUALITY_FEATURES:
        if name in raw_df.columns:
            epoch_data[name] = raw_df[name].to_list()

    epoch_df = pl.DataFrame(epoch_data)

    # --- Bout DataFrame ---
    bout_rows: list[dict] = []
    if T > 0:
        cur_state = int(dec[0])
        cur_start = float(df["start_s"][0])
        ps_list = [P[0]]

        for t in range(1, T):
            if int(dec[t]) != cur_state:
                cur_end = float(df["start_s"][t])
                Ps = np.vstack(ps_list)
                bout = {
                    "session_id": sid,
                    "start_s": cur_start,
                    "end_s": cur_end,
                    "label": state_names[cur_state],
                    "n_epochs": len(ps_list),
                }
                for si, name in enumerate(state_names):
                    bout[f"mean_P_{name}"] = float(Ps[:, si].mean())
                for name in VIDEO_QUALITY_FEATURES:
                    if name in raw_df.columns:
                        bout[f"mean_{name}"] = float(raw_df[name][t - len(ps_list):t].mean())
                bout_rows.append(bout)

                cur_state = int(dec[t])
                cur_start = float(df["start_s"][t])
                ps_list = [P[t]]
            else:
                ps_list.append(P[t])

        # Last bout
        cur_end = float(df["end_s"][T - 1])
        Ps = np.vstack(ps_list)
        bout = {
            "session_id": sid,
            "start_s": cur_start,
            "end_s": cur_end,
            "label": state_names[cur_state],
            "n_epochs": len(ps_list),
        }
        for si, name in enumerate(state_names):
            bout[f"mean_P_{name}"] = float(Ps[:, si].mean())
        for name in VIDEO_QUALITY_FEATURES:
            if name in raw_df.columns:
                bout[f"mean_{name}"] = float(raw_df[name][T - len(ps_list):T].mean())
        bout_rows.append(bout)

    bout_df = pl.DataFrame(bout_rows) if bout_rows else pl.DataFrame()

    return bout_df, epoch_df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: dict[str, Any], path: str | Path) -> None:
    """Save a trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a trained model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def print_feature_importance(model: dict[str, Any], top_n: int = 30) -> None:
    """Print the top-N most important features from a trained model."""
    importance = model.get("feature_importance", {})
    if not importance:
        print("No feature importance data in model.")
        return
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop {top_n} features by importance:")
    print("-" * 50)
    for i, (feat, imp) in enumerate(sorted_feats[:top_n], 1):
        print(f"  {i:3d}. {feat:<40s} {imp:>8.1f}")
