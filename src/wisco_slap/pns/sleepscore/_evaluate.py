"""Evaluation pipeline: leave-one-subject-out cross-validation and metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ._config import STATE_NAMES, ScoreConfig
from ._data import (
    intervals_to_epoch_labels,
    prepare_training_labels,
    validate_and_standardize_labels,
)


_STATE_COLORS: dict[str, str] = {
    "NREM": "#4b71e3",
    "REM": "#e34bde",
    "Wake": "#4be350",
    "Unlabeled": "#d9d9d9",
}


def _coerce_polars_df(df: Any, *, name: str) -> pl.DataFrame:
    """Return *df* as a polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df.clone()

    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is available in project env
        pd = None

    if pd is not None and isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)

    raise TypeError(f"{name} must be a polars or pandas DataFrame, got {type(df)!r}")


def _merge_consecutive_label_rows(
    df: pl.DataFrame,
    *,
    label_col: str = "label",
    start_col: str = "start_s",
    end_col: str = "end_s",
) -> pl.DataFrame:
    """Merge consecutive rows with identical labels."""
    if df.is_empty():
        return pl.DataFrame(
            schema={start_col: pl.Float64, end_col: pl.Float64, label_col: pl.String}
        )

    return (
        df.sort(start_col)
        .with_columns(
            pl.col(label_col)
            .ne(pl.col(label_col).shift(1))
            .fill_null(True)
            .cum_sum()
            .alias("_group")
        )
        .group_by("_group", maintain_order=True)
        .agg(
            pl.col(start_col).first().alias(start_col),
            pl.col(end_col).last().alias(end_col),
            pl.col(label_col).first().alias(label_col),
        )
        .select([start_col, end_col, label_col])
    )


def _validate_labels(labels: list[str], state_names: list[str], *, name: str) -> None:
    """Raise if *labels* contains values outside *state_names*."""
    invalid = sorted({label for label in labels if label not in state_names})
    if invalid:
        raise ValueError(
            f"{name} contains labels outside the supported state set {state_names}: {invalid}"
        )


def _compute_weighted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,
    state_names: list[str],
) -> dict[str, Any]:
    """Compute per-class precision, recall, F1, and accuracy with sample weights."""
    n_states = len(state_names)
    cm = np.zeros((n_states, n_states), dtype=float)
    for t, p, w in zip(y_true, y_pred, sample_weight, strict=False):
        if 0 <= t < n_states and 0 <= p < n_states:
            cm[t, p] += float(w)

    accuracy = float(np.trace(cm)) / max(float(cm.sum()), 1e-12)

    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(state_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support_s": float(cm[i, :].sum()),
        }

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": cm,
        "total_duration_s": float(cm.sum()),
    }


def _print_weighted_metrics(
    metrics: dict[str, Any],
    state_names: list[str],
    *,
    title: str = "",
) -> None:
    """Pretty-print duration-weighted metrics."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    print(f"\n  Duration-weighted accuracy: {metrics['accuracy']:.3f}")
    print(
        f"\n  {'State':<12s} {'Precision':>10s} {'Recall':>10s} "
        f"{'F1':>10s} {'Support (s)':>12s}"
    )
    print(f"  {'-' * 58}")
    for name in state_names:
        m = metrics["per_class"][name]
        print(
            f"  {name:<12s} {m['precision']:>10.3f} {m['recall']:>10.3f} "
            f"{m['f1']:>10.3f} {m['support_s']:>12.1f}"
        )


def _max_prob_labels(epoch_df: pl.DataFrame, state_names: list[str]) -> list[str]:
    """Return the argmax-of-probabilities label per epoch row.

    Requires a ``P_<state>`` column for every state in *state_names*. This is the
    raw per-epoch classifier prediction, before HSMM smoothing.
    """
    prob_cols = [f"P_{state}" for state in state_names]
    missing = [c for c in prob_cols if c not in epoch_df.columns]
    if missing:
        raise ValueError(
            f"epoch_df must contain probability columns for every state. Missing: {missing}"
        )
    P = epoch_df.select(prob_cols).to_numpy().astype(float)
    idx = np.argmax(P, axis=1)
    return [state_names[i] for i in idx.tolist()]


def _resolve_bout_label_col(bout_df: pl.DataFrame) -> str:
    """Return the bout label column name."""
    if "label" in bout_df.columns:
        return "label"
    if "pred_label" in bout_df.columns:
        return "pred_label"
    raise ValueError("bout_df must contain either 'label' or 'pred_label'.")


def _build_atomic_segment_alignment(
    truth_df: pl.DataFrame,
    pred_df: pl.DataFrame,
    *,
    truth_label_col: str = "label",
    pred_label_col: str = "label",
) -> pl.DataFrame:
    """Split time by the union of truth/pred boundaries and assign one label to each segment."""
    truth_sorted = truth_df.sort("start_s")
    pred_sorted = pred_df.sort("start_s")

    boundaries = np.unique(
        np.concatenate(
            [
                truth_sorted["start_s"].to_numpy(),
                truth_sorted["end_s"].to_numpy(),
                pred_sorted["start_s"].to_numpy(),
                pred_sorted["end_s"].to_numpy(),
            ]
        ).astype(float)
    )
    if len(boundaries) < 2:
        return pl.DataFrame(
            schema={
                "start_s": pl.Float64,
                "end_s": pl.Float64,
                "duration_s": pl.Float64,
                "true_label": pl.String,
                "pred_label": pl.String,
                "true_bout_idx": pl.Int64,
                "pred_bout_idx": pl.Int64,
            }
        )

    def _label_segments(
        intervals: pl.DataFrame,
        label_col: str,
    ) -> tuple[list[str | None], list[int | None]]:
        starts = intervals["start_s"].to_numpy().astype(float)
        ends = intervals["end_s"].to_numpy().astype(float)
        labels = intervals[label_col].to_list()
        out_labels: list[str | None] = []
        out_ids: list[int | None] = []
        idx = 0
        for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:], strict=False):
            midpoint = float(seg_start + (seg_end - seg_start) / 2.0)
            while idx < len(ends) and ends[idx] <= midpoint:
                idx += 1
            if idx < len(starts) and starts[idx] <= midpoint < ends[idx]:
                out_labels.append(labels[idx])
                out_ids.append(idx)
            else:
                out_labels.append(None)
                out_ids.append(None)
        return out_labels, out_ids

    true_labels, true_ids = _label_segments(truth_sorted, truth_label_col)
    pred_labels, pred_ids = _label_segments(pred_sorted, pred_label_col)

    segment_df = pl.DataFrame(
        {
            "start_s": boundaries[:-1].tolist(),
            "end_s": boundaries[1:].tolist(),
            "duration_s": np.diff(boundaries).tolist(),
            "true_label": true_labels,
            "pred_label": pred_labels,
            "true_bout_idx": true_ids,
            "pred_bout_idx": pred_ids,
        }
    )

    return segment_df.filter(
        pl.col("duration_s") > 0,
        pl.col("true_label").is_not_null(),
        pl.col("pred_label").is_not_null(),
    )


def _summarize_predicted_bout_purity(segment_df: pl.DataFrame) -> pl.DataFrame:
    """Summarize how pure each predicted bout is with respect to the truth labels."""
    if segment_df.is_empty():
        return pl.DataFrame(
            schema={
                "pred_bout_idx": pl.Int64,
                "start_s": pl.Float64,
                "end_s": pl.Float64,
                "pred_label": pl.String,
                "duration_s": pl.Float64,
                "dominant_true_label": pl.String,
                "dominant_overlap_s": pl.Float64,
                "purity": pl.Float64,
            }
        )

    overlap_by_truth = (
        segment_df.group_by(["pred_bout_idx", "pred_label", "true_label"], maintain_order=True)
        .agg(pl.col("duration_s").sum().alias("overlap_s"))
        .sort(["pred_bout_idx", "overlap_s"], descending=[False, True])
    )
    dominant_truth = (
        overlap_by_truth.group_by("pred_bout_idx", maintain_order=True)
        .agg(
            pl.col("true_label").first().alias("dominant_true_label"),
            pl.col("overlap_s").first().alias("dominant_overlap_s"),
        )
    )
    bout_totals = (
        segment_df.group_by(["pred_bout_idx", "pred_label"], maintain_order=True)
        .agg(
            pl.col("start_s").min().alias("start_s"),
            pl.col("end_s").max().alias("end_s"),
            pl.col("duration_s").sum().alias("duration_s"),
        )
    )

    return (
        bout_totals.join(dominant_truth, on="pred_bout_idx", how="left")
        .with_columns(
            (pl.col("dominant_overlap_s") / pl.col("duration_s")).alias("purity")
        )
        .sort("start_s")
    )


def _plot_confusion_matrix(
    ax: Any,
    cm: np.ndarray,
    state_names: list[str],
    *,
    title: str,
    normalize: bool,
    duration_weighted: bool,
) -> None:
    """Render a confusion matrix heatmap on *ax*."""
    row_sums = cm.sum(axis=1, keepdims=True)
    if normalize:
        display = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    else:
        display = cm

    im = ax.imshow(display, cmap="Blues", vmin=0.0)
    ax.set_xticks(range(len(state_names)), state_names, rotation=45, ha="right")
    ax.set_yticks(range(len(state_names)), state_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    for row_idx in range(display.shape[0]):
        for col_idx in range(display.shape[1]):
            if normalize:
                text = f"{display[row_idx, col_idx]:.0%}"
            elif duration_weighted:
                text = f"{display[row_idx, col_idx]:.1f}s"
            else:
                text = str(int(display[row_idx, col_idx]))
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                color="white" if display[row_idx, col_idx] > 0.5 else "black",
                fontsize=9,
            )

    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_interval_track(
    ax: Any,
    intervals_df: pl.DataFrame,
    *,
    label_col: str,
    title: str,
    xlim: tuple[float, float],
) -> None:
    """Plot label intervals as a colored track."""
    for row in intervals_df.iter_rows(named=True):
        label = row[label_col]
        color = _STATE_COLORS.get(str(label), _STATE_COLORS["Unlabeled"])
        ax.broken_barh(
            [(float(row["start_s"]), float(row["end_s"] - row["start_s"]))],
            (0, 1),
            facecolors=color,
            edgecolors="none",
        )

    ax.set_ylim(0, 1)
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.set_title(title, loc="left")


def evaluate_scored_session(
    ground_truth_labels: Any,
    epoch_df: Any,
    bout_df: Any,
    *,
    state_names: list[str] | None = None,
    min_overlap_frac: float = 0.0,
    make_plots: bool = True,
    show_plots: bool = True,
    print_summary: bool = True,
) -> dict[str, Any]:
    """Evaluate one scored session against manually labeled ground truth.

    Parameters
    ----------
    ground_truth_labels : polars or pandas DataFrame
        Ground-truth interval labels with at least ``start_s``, ``end_s``, and ``label``.
        Raw labels such as ``NREM_SWS`` or ``Artifact`` are standardized automatically.
    epoch_df : polars or pandas DataFrame
        Per-epoch output from ``score_session``. Must contain ``start_s``, ``end_s``,
        and one ``P_<state>`` column for every state in *state_names* (``P_NREM``,
        ``P_REM``, ``P_Wake`` by default). The per-epoch predicted label used for
        evaluation is the argmax of these probability columns — i.e. the raw
        per-epoch classifier output, before HSMM smoothing. The HSMM-smoothed
        per-epoch ``pred_label`` column, if present, is ignored.
    bout_df : polars or pandas DataFrame
        Bout-level output from ``score_session``. Must contain ``start_s``, ``end_s``,
        and either ``label`` or ``pred_label``.
    state_names : list[str], optional
        Canonical state order for metrics and plots. Defaults to the package state order.
    min_overlap_frac : float, optional
        Minimum fraction of an epoch that must be covered by a ground-truth interval for
        epoch-level evaluation.
    make_plots : bool, optional
        Whether to create matplotlib figures.
    show_plots : bool, optional
        Whether to immediately display created plots.
    print_summary : bool, optional
        Whether to print a concise metric summary.

    Returns
    -------
    dict
        A nested result dict with epoch metrics, bout metrics, mismatch tables, and
        optionally figure objects under ``results["figures"]``.
    """
    if state_names is None:
        state_names = list(STATE_NAMES)
    state_to_idx = {state: idx for idx, state in enumerate(state_names)}

    truth_df = validate_and_standardize_labels(
        _coerce_polars_df(ground_truth_labels, name="ground_truth_labels")
    )
    epoch_pl = _coerce_polars_df(epoch_df, name="epoch_df").sort("start_s")
    bout_pl = _coerce_polars_df(bout_df, name="bout_df").sort("start_s")

    if truth_df.is_empty():
        raise ValueError("ground_truth_labels is empty after label standardization.")
    if epoch_pl.is_empty():
        raise ValueError("epoch_df is empty.")
    if bout_pl.is_empty():
        raise ValueError("bout_df is empty.")

    bout_label_col = _resolve_bout_label_col(bout_pl)

    max_epoch_labels = _max_prob_labels(epoch_pl, state_names)
    _validate_labels(max_epoch_labels, state_names, name="epoch_df (max_label)")
    _validate_labels(bout_pl[bout_label_col].to_list(), state_names, name="bout_df")

    # Merge consecutive truth intervals after canonical remapping so the plots are cleaner.
    truth_bouts = _merge_consecutive_label_rows(truth_df.select(["start_s", "end_s", "label"]))

    # ------------------------------------------------------------------
    # Epoch-level evaluation
    # ------------------------------------------------------------------
    epoch_edges = np.concatenate(
        [
            epoch_pl["start_s"].to_numpy().astype(float),
            np.array([float(epoch_pl["end_s"][-1])], dtype=float),
        ]
    )
    y_true_epoch = intervals_to_epoch_labels(epoch_edges, truth_df, min_overlap_frac)
    y_pred_epoch = np.array([state_to_idx[label] for label in max_epoch_labels], dtype=int)
    labeled_epoch_mask = y_true_epoch >= 0
    if not np.any(labeled_epoch_mask):
        raise ValueError("No epoch_df rows overlap labeled ground truth.")

    true_epoch_labels = [
        state_names[idx] if idx >= 0 else None for idx in y_true_epoch.tolist()
    ]
    epoch_alignment_df = (
        epoch_pl.with_columns(
            pl.Series("max_label", max_epoch_labels),
            pl.Series("true_state_idx", y_true_epoch.tolist()),
            pl.Series("true_label", true_epoch_labels),
        )
        .filter(pl.col("true_state_idx") >= 0)
        .with_columns(
            pl.col("max_label").eq(pl.col("true_label")).alias("correct")
        )
    )
    epoch_metrics = _compute_metrics(
        y_true_epoch[labeled_epoch_mask],
        y_pred_epoch[labeled_epoch_mask],
        state_names,
    )

    # ------------------------------------------------------------------
    # Bout-level evaluation
    # ------------------------------------------------------------------
    pred_bouts = bout_pl.select(["start_s", "end_s", bout_label_col]).rename(
        {bout_label_col: "label"}
    )
    bout_alignment_df = _build_atomic_segment_alignment(
        truth_bouts,
        pred_bouts,
        truth_label_col="label",
        pred_label_col="label",
    )
    if bout_alignment_df.is_empty():
        raise ValueError("No overlap between bout_df and labeled ground truth.")

    y_true_bout = np.array(
        [state_to_idx[label] for label in bout_alignment_df["true_label"].to_list()],
        dtype=int,
    )
    y_pred_bout = np.array(
        [state_to_idx[label] for label in bout_alignment_df["pred_label"].to_list()],
        dtype=int,
    )
    duration_weight = bout_alignment_df["duration_s"].to_numpy().astype(float)
    bout_metrics = _compute_weighted_metrics(
        y_true_bout,
        y_pred_bout,
        duration_weight,
        state_names,
    )
    bout_alignment_df = bout_alignment_df.with_columns(
        pl.col("pred_label").eq(pl.col("true_label")).alias("correct")
    )
    pred_bout_purity_df = _summarize_predicted_bout_purity(bout_alignment_df)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    epoch_mismatches_df = epoch_alignment_df.filter(~pl.col("correct")).select(
        ["start_s", "end_s", "true_label", "max_label"]
    )
    bout_mismatches_df = bout_alignment_df.filter(~pl.col("correct")).select(
        ["start_s", "end_s", "duration_s", "true_label", "pred_label"]
    )

    if print_summary:
        _print_metrics(epoch_metrics, state_names, "Epoch-Level Evaluation")
        _print_weighted_metrics(
            bout_metrics,
            state_names,
            title="Bout-Level Evaluation (Duration-Weighted)",
        )
        if pred_bout_purity_df.height > 0:
            median_purity = float(pred_bout_purity_df["purity"].median())
            print(f"\n  Median predicted-bout purity: {median_purity:.3f}")
            print(f"  Epoch mismatches: {epoch_mismatches_df.height}")
            print(
                f"  Bout mismatch time: {float(bout_mismatches_df['duration_s'].sum()):.1f}s"
            )

    results: dict[str, Any] = {
        "epoch": {
            "metrics": epoch_metrics,
            "alignment_df": epoch_alignment_df,
            "mismatches_df": epoch_mismatches_df,
        },
        "bout": {
            "metrics": bout_metrics,
            "alignment_df": bout_alignment_df,
            "mismatches_df": bout_mismatches_df,
            "predicted_bout_purity_df": pred_bout_purity_df,
        },
        "ground_truth_bouts_df": truth_bouts,
        "figures": {},
    }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if make_plots:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        cm_fig, cm_axes = plt.subplots(1, 2, figsize=(12, 4.8))
        _plot_confusion_matrix(
            cm_axes[0],
            epoch_metrics["confusion_matrix"],
            state_names,
            title=f"Epoch Confusion\nAccuracy={epoch_metrics['accuracy']:.3f}",
            normalize=True,
            duration_weighted=False,
        )
        _plot_confusion_matrix(
            cm_axes[1],
            bout_metrics["confusion_matrix"],
            state_names,
            title=f"Bout Confusion (Duration-Weighted)\nAccuracy={bout_metrics['accuracy']:.3f}",
            normalize=True,
            duration_weighted=True,
        )
        cm_fig.tight_layout()

        gt_start = float(truth_bouts["start_s"].min())
        gt_end = float(truth_bouts["end_s"].max())

        # Max-probability hypnogram: merge consecutive same-label epochs into bouts.
        max_label_intervals = _merge_consecutive_label_rows(
            pl.DataFrame(
                {
                    "start_s": epoch_pl["start_s"].to_list(),
                    "end_s": epoch_pl["end_s"].to_list(),
                    "label": max_epoch_labels,
                }
            )
        )

        timeline_fig, timeline_axes = plt.subplots(
            4,
            1,
            figsize=(14, 8.4),
            sharex=True,
            height_ratios=[0.7, 0.7, 0.7, 2.2],
        )
        _plot_interval_track(
            timeline_axes[0],
            truth_bouts,
            label_col="label",
            title="Ground Truth",
            xlim=(gt_start, gt_end),
        )
        _plot_interval_track(
            timeline_axes[1],
            pred_bouts,
            label_col="label",
            title="Predicted Bouts",
            xlim=(gt_start, gt_end),
        )
        _plot_interval_track(
            timeline_axes[2],
            max_label_intervals,
            label_col="label",
            title="Max Probability Hypnogram",
            xlim=(gt_start, gt_end),
        )

        probability_ax = timeline_axes[3]
        epoch_centers = (
            (epoch_alignment_df["start_s"] + epoch_alignment_df["end_s"]) / 2.0
        ).to_numpy()
        legend_handles = [
            Patch(facecolor=_STATE_COLORS[state], edgecolor="none", label=state)
            for state in state_names
        ]
        for state in state_names:
            prob_col = f"P_{state}"
            if prob_col in epoch_alignment_df.columns:
                probability_ax.plot(
                    epoch_centers,
                    epoch_alignment_df[prob_col].to_numpy(),
                    label=state,
                    color=_STATE_COLORS[state],
                    linewidth=1.6,
                )
        for row in epoch_mismatches_df.iter_rows(named=True):
            probability_ax.axvspan(
                float(row["start_s"]),
                float(row["end_s"]),
                color="#ff7f7f",
                alpha=0.14,
                ec="none",
            )
        probability_ax.set_xlim(gt_start, gt_end)
        probability_ax.set_ylim(-0.02, 1.02)
        probability_ax.set_ylabel("Probability")
        probability_ax.set_xlabel("Time (s)")
        probability_ax.set_title("Per-Epoch Probabilities (red shading = mismatch)", loc="left")
        probability_ax.legend(
            handles=legend_handles,
            loc="upper right",
            ncol=len(state_names),
            frameon=False,
        )
        timeline_fig.tight_layout()

        results["figures"] = {
            "confusion_matrices": cm_fig,
            "timeline": timeline_fig,
        }

        if show_plots:
            plt.show()

    return results


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
    *,
    exclude_artifact: bool = False,
) -> dict[str, Any]:
    """Leave-one-subject-out cross-validation.

    For each unique subject, trains on all other subjects' sessions and
    evaluates on the held-out subject's sessions.

    Parameters
    ----------
    sessions : list of all session dicts.
    labels_by_session : dict mapping session_id to validated label DataFrames.
    config : scoring configuration.
    exclude_artifact : if True, exclude Artifact intervals from each training fold.
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
            ldf = prepare_training_labels(
                labels_by_session.get(sid),
                exclude_artifact=exclude_artifact,
            )
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
            model = train_model(
                train_sessions,
                labels_by_session,
                config,
                exclude_artifact=exclude_artifact,
            )
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
