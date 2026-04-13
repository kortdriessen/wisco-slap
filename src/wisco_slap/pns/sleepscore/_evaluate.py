"""Evaluation pipeline: leave-one-subject-out cross-validation and metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ._config import ScoreConfig, STATE_NAMES


def _extract_subject_from_session_id(session_id: str) -> str:
    """Extract subject name from session_id format: '{subject}_{exp}_sb-{sb}'."""
    parts = session_id.split("_")
    # Subject name is everything before the first 'exp_' token
    exp_idx = None
    for i, p in enumerate(parts):
        if p == "exp":
            exp_idx = i
            break
    if exp_idx is not None:
        return "_".join(parts[:exp_idx])
    return parts[0]


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_names: list[str],
) -> dict[str, Any]:
    """Compute per-class precision, recall, F1, and overall accuracy.

    Parameters
    ----------
    y_true, y_pred : integer state indices.
    state_names : list of state names in index order.

    Returns
    -------
    dict with keys: ``accuracy``, ``per_class`` (dict per state), ``confusion_matrix``.
    """
    n_states = len(state_names)
    cm = np.zeros((n_states, n_states), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_states and 0 <= p < n_states:
            cm[t, p] += 1

    accuracy = float(np.trace(cm)) / max(cm.sum(), 1)

    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(state_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[i, :].sum()),
        }

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def _print_metrics(metrics: dict[str, Any], state_names: list[str], title: str = "") -> None:
    """Pretty-print evaluation metrics."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    print(f"\n  Overall accuracy: {metrics['accuracy']:.3f}")
    print(f"\n  {'State':<12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print(f"  {'-' * 52}")
    for name in state_names:
        m = metrics["per_class"][name]
        print(f"  {name:<12s} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10d}")

    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = "  " + " " * 10 + "".join(f"{n:>8s}" for n in state_names)
    print(header)
    for i, name in enumerate(state_names):
        row = "  " + f"{name:<10s}" + "".join(f"{cm[i, j]:>8d}" for j in range(len(state_names)))
        print(row)


def evaluate_loso(
    sessions: list[dict],
    labels_by_session: dict[str, pl.DataFrame],
    config: ScoreConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Leave-one-subject-out cross-validation.

    For each unique subject, trains on all other subjects' sessions and
    evaluates on the held-out subject's sessions.

    Parameters
    ----------
    sessions : list of all session dicts.
    labels_by_session : dict mapping session_id to validated label DataFrames.
    config : scoring configuration.
    verbose : whether to print results.

    Returns
    -------
    dict with ``overall`` metrics and ``per_fold`` results.
    """
    from ._model import train_model, score_session

    if config is None:
        config = ScoreConfig()

    state_names = config.states

    # Group sessions by subject
    subject_sessions: dict[str, list[dict]] = {}
    for sess in sessions:
        subj = _extract_subject_from_session_id(sess["session_id"])
        subject_sessions.setdefault(subj, []).append(sess)

    subjects = sorted(subject_sessions.keys())
    if len(subjects) < 2:
        raise ValueError(
            f"LOSO requires at least 2 subjects, got {len(subjects)}: {subjects}"
        )

    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    per_fold: dict[str, dict] = {}

    for held_out in subjects:
        if verbose:
            print(f"\n--- Fold: held out subject = {held_out} ---")

        train_sessions = [s for subj, ss in subject_sessions.items() if subj != held_out for s in ss]
        test_sessions = subject_sessions[held_out]

        # Check that train set has all classes
        from ._data import build_epoch_grid, intervals_to_epoch_labels

        train_edges = build_epoch_grid(train_sessions, config.epoch_len_s)
        train_labels_present = set()
        for sess in train_sessions:
            sid = sess["session_id"]
            ldf = labels_by_session.get(sid)
            if ldf is not None:
                y = intervals_to_epoch_labels(train_edges[sid], ldf, config.min_overlap_frac)
                train_labels_present.update(set(y[y >= 0].tolist()))

        required = set(range(len(state_names)))
        if train_labels_present != required:
            if verbose:
                missing = required - train_labels_present
                missing_names = [state_names[i] for i in sorted(missing)]
                print(f"  Skipping fold: training set missing classes {missing_names}")
            continue

        try:
            model = train_model(train_sessions, labels_by_session, config)
        except ValueError as e:
            if verbose:
                print(f"  Skipping fold due to training error: {e}")
            continue

        fold_y_true: list[int] = []
        fold_y_pred: list[int] = []

        for sess in test_sessions:
            sid = sess["session_id"]
            ldf = labels_by_session.get(sid)
            if ldf is None or ldf.is_empty():
                continue

            _, epoch_df = score_session(model, sess)

            # Get true labels for this session
            test_edges = build_epoch_grid([sess], config.epoch_len_s)
            y_true = intervals_to_epoch_labels(test_edges[sid], ldf, config.min_overlap_frac)

            # Only evaluate on labeled epochs
            y_pred = epoch_df["pred_state_idx"].to_numpy()
            labeled = y_true >= 0

            if np.any(labeled):
                fold_y_true.extend(y_true[labeled].tolist())
                fold_y_pred.extend(y_pred[labeled].tolist())

        if fold_y_true:
            fold_metrics = _compute_metrics(
                np.array(fold_y_true), np.array(fold_y_pred), state_names,
            )
            per_fold[held_out] = fold_metrics
            all_y_true.extend(fold_y_true)
            all_y_pred.extend(fold_y_pred)
            if verbose:
                _print_metrics(fold_metrics, state_names, f"Fold: {held_out}")
        elif verbose:
            print(f"  No labeled test epochs for subject {held_out}")

    # Overall
    overall = {}
    if all_y_true:
        overall = _compute_metrics(np.array(all_y_true), np.array(all_y_pred), state_names)
        if verbose:
            _print_metrics(overall, state_names, "OVERALL (all folds)")

    return {
        "overall": overall,
        "per_fold": per_fold,
    }
