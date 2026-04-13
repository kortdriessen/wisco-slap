"""LightGBM classifier training, calibration, and persistence."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from ._config import N_STATES, STATE_NAMES, STATE_TO_IDX, ScoreConfig
from ._data import build_epoch_grid, intervals_to_epoch_labels
from ._features import build_features
from ._hsmm import decode_session
from ._utils import impute_per_session, robust_scale_per_session


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
    """Impute and robust-scale features per session."""
    df = impute_per_session(df, feature_cols)
    df = robust_scale_per_session(df, feature_cols)
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    sessions: list[dict],
    labels_by_session: dict[str, pl.DataFrame],
    config: ScoreConfig | None = None,
) -> dict[str, Any]:
    """Train a sleep scoring model (LightGBM + calibration + HSMM config).

    Parameters
    ----------
    sessions : list of session dicts (from ``create_session``).
    labels_by_session : dict mapping session_id to polars DataFrame with
        columns ``start_s``, ``end_s``, ``label`` (already validated).
    config : scoring configuration; uses defaults if None.

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
        labels_df = labels_by_session.get(sid, None)
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

    # --- Train/test split ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_lab, y_lab,
        test_size=clf_cfg.test_size,
        random_state=clf_cfg.random_state,
        stratify=y_lab,
    )

    # Class-balanced sample weights
    n_classes = len(state_names)
    class_counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    inv = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    sw = inv[y_tr] * (len(y_tr) / (np.sum(inv[y_tr]) + 1e-12))

    # --- Base classifier: LightGBM ---
    base = LGBMClassifier(
        **clf_cfg.lgbm_params,
        random_state=clf_cfg.random_state,
        n_jobs=-1,
    )
    base.fit(X_tr, y_tr, sample_weight=sw)

    # --- Calibration ---
    calibrator = CalibratedClassifierCV(
        base,
        method=clf_cfg.calibration_method,
        cv=clf_cfg.calibration_cv,
    )
    calibrator.fit(X_te, y_te)

    # --- Feature importance ---
    importance = dict(zip(feature_cols, base.feature_importances_.tolist()))

    # --- Assemble model dict ---
    model: dict[str, Any] = {
        "epoch_len_s": config.epoch_len_s,
        "feature_cols": feature_cols,
        "hsmm_cfg": config.hsmm,
        "calibrated_clf": calibrator,
        "state_names": state_names,
        "feature_importance": importance,
        "config": config,
    }
    return model


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
    df = feats_df.filter(pl.col("session_id") == sid)
    df = _prepare_matrix(df, feature_cols)

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
