"""Evidence-weighted merging of frozen Step 1 Active-wake selections.

Only original Wake is split. Step 1 selections are immutable anchors. Every
remaining Wake second receives Quiet wake unless an adjacent anchor absorbs it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.signal import welch

from .active_wake import _runs


@dataclass(frozen=True)
class WakeMergeConfig:
    """Budgets are equivalent seconds, increased by quiet evidence per second."""

    bridge_budget_s: float = 8.0
    edge_budget_s: float = 2.0
    quiet_slow_min: float = 0.5
    still_motion_max: float = 1.5
    quiet_weight: float = 4.0
    quiet_min_s: float = 2.0
    smooth_seconds: int = 3
    peak_fraction_max: float = 0.65


def build_wake_merge_context(
    step1_epochs: pl.DataFrame, eeg: Any, metadata: dict
) -> tuple[pl.DataFrame, dict]:
    """Add 0.5–5 Hz power on the existing grid without recomputing Step 1 labels.

    Four-second Welch spectra use each continuous finite EEG segment separately.
    Both the immutable Step 1 labels and its validity columns are preserved.
    """
    frame = step1_epochs.sort("start_s")
    t = np.asarray(eeg.time, float)
    y = np.asarray(eeg, float).squeeze()
    fs = float(eeg.attrs.get("fs", 1 / np.median(np.diff(t))))
    if y.shape != t.shape or np.any(np.diff(t) <= 0):
        raise ValueError("Expected one EEG channel and increasing timestamps.")
    finite = np.flatnonzero(np.isfinite(y))
    segment = np.full(len(t), -1, int)
    segments = []
    if len(finite):
        cuts = np.r_[
            0,
            np.flatnonzero((np.diff(finite) > 1) | (np.diff(t[finite]) > 1.5 / fs)) + 1,
            len(finite),
        ]
        segments = [
            (int(finite[a]), int(finite[b - 1] + 1))
            for a, b in zip(cuts[:-1], cuts[1:])
        ]
    for sid, (a, b) in enumerate(segments):
        segment[a:b] = sid
    power = np.full(frame.height, np.nan)
    half_window = float(metadata.get("window_s", 4.0)) / 2
    for i, center in enumerate(frame["time"]):
        j = min(int(np.searchsorted(t, center)), len(t) - 1)
        sid = segment[j]
        if sid < 0 or not frame["eeg_valid"][i]:
            continue
        a = max(segments[sid][0], int(np.searchsorted(t, center - half_window)))
        b = min(segments[sid][1], int(np.searchsorted(t, center + half_window)))
        if b - a < 1.9 * fs:
            continue
        f, p = welch(y[a:b], fs=fs, nperseg=b - a, detrend="linear")
        df = f[1] - f[0]
        weights = np.maximum(
            0, np.minimum(f + df / 2, 5.0) - np.maximum(f - df / 2, 0.5)
        )
        power[i] = float(np.sum(p * weights))
    frame = frame.with_columns(pl.Series("quiet_slow_power_0p5_5", power))
    sm = frame.select(
        pl
        .col("quiet_slow_power_0p5_5")
        .fill_nan(None)
        .rolling_median(3, center=True, min_samples=1)
        .over("run_id")
    )["quiet_slow_power_0p5_5"].to_numpy()
    valid = (
        frame["eeg_valid"].to_numpy()
        & np.isfinite(sm)
        & (frame["fast_peak_fraction"].to_numpy() <= 0.65)
        & (frame["duration_s"].to_numpy() >= 0.999)
    )
    wake = valid & (frame["original_state"].to_numpy() == "Wake")
    ref = valid & frame["reference_epoch"].to_numpy()
    enough = wake.sum() >= 60 and ref.sum() >= 30
    low = float(np.quantile(sm[wake], 0.1)) if enough else None
    high = float(np.quantile(sm[ref], 0.75)) if enough else None
    available = bool(enough and low > 0 and high / low >= 2.5)
    slow = (power - low) / (high - low) if available else np.full(len(power), np.nan)
    frame = frame.with_columns(
        pl.Series("quiet_slow_index", slow),
        pl.col("active_selected").alias("stage1_active"),
    )
    meta = {
        **metadata,
        "step2_algorithm": "wake-substates-v1-weighted-merging",
        "step2_slow_calibration": dict(
            available=available,
            band_hz=[0.5, 5.0],
            low_wake_power=low,
            slow_nrem_power=high,
            contrast_fold=high / low if enough and low > 0 else None,
            reference_seconds=int(ref.sum()),
            wake_seconds=int(wake.sum()),
            reason="available"
            if available
            else "insufficient reference data or <2.5-fold slow contrast",
        ),
        "step2_missing_evidence": "remaining Wake defaults to Quiet wake; missing EEG is a merge barrier, missing video is not stillness",
    }
    return frame, meta


def merge_wake_substates(
    context: pl.DataFrame, config: WakeMergeConfig = WakeMergeConfig()
) -> dict:
    """Bridge short weak-quiet gaps, then trim edges using frozen active anchors.

    Cost per second = 1 + quiet_weight * quiet_evidence (0–1). An entire gap
    between two Step 1 anchors merges when its total cost fits the bridge budget.
    Otherwise each anchored edge may absorb only its own edge budget. No new
    anchor is created, so repeated application cannot creep through long bouts.
    Quiet cores and missing EEG stop growth regardless of the budgets.
    """
    values = asdict(config)
    if (
        not all(np.isfinite(v) for v in values.values())
        or min(config.bridge_budget_s, config.edge_budget_s, config.quiet_weight) < 0
        or config.quiet_slow_min <= 0
        or config.quiet_min_s <= 0
        or config.smooth_seconds < 1
        or int(config.smooth_seconds) != config.smooth_seconds
        or not 0 < config.peak_fraction_max <= 1
    ):
        raise ValueError("Invalid Step 2 budgets, evidence or persistence settings.")
    frame = context.sort("start_s")
    required = {"stage1_active", "stage1_label", "quiet_slow_index"}
    if not required <= set(frame.columns):
        raise ValueError(
            "Use build_wake_merge_context with the frozen Step 1 result first."
        )
    n = frame.height
    if n == 0:
        raise ValueError("Empty Step 1 context.")
    start, end = frame["start_s"].to_numpy(), frame["end_s"].to_numpy()
    if np.any(end <= start) or np.any(start[1:] < end[:-1] - 1e-8):
        raise ValueError("Epochs must be positive-duration and non-overlapping.")
    state = frame["original_state"].to_numpy()
    wake = state == "Wake"
    fixed = frame["stage1_active"].to_numpy().astype(bool)
    if np.any(fixed & ~wake):
        raise ValueError("Step 1 Active wake must be confined to original Wake.")
    valid = (
        frame["eeg_valid"].to_numpy()
        & np.isfinite(frame["quiet_slow_index"].to_numpy())
        & (frame["fast_peak_fraction"].to_numpy() <= config.peak_fraction_max)
    )
    source_run = frame["run_id"].to_numpy()
    # Defend the merger against an incomplete caller-provided run_id too.
    run = np.cumsum(
        np.r_[
            True,
            (source_run[1:] != source_run[:-1])
            | (state[1:] != state[:-1])
            | (start[1:] > end[:-1] + 1e-8)
            | ~valid[1:]
            | ~valid[:-1],
        ]
    )
    frame = frame.with_columns(pl.Series("merge_run_id", run))
    smoothcols = ["quiet_slow_index", "whisk_z", "pupil_motion_z"]
    frame = frame.with_columns([
        pl
        .col(c)
        .fill_nan(None)
        .rolling_median(config.smooth_seconds, min_samples=1, center=True)
        .over("merge_run_id")
        .alias("merge_" + c)
        for c in smoothcols
    ])
    slow = frame["merge_quiet_slow_index"].to_numpy()
    whisk = frame["merge_whisk_z"].to_numpy()
    pupil = frame["merge_pupil_motion_z"].to_numpy()
    whisk = np.where(frame["whisk_valid_frac"].to_numpy() >= 0.8, whisk, np.nan)
    pupil = np.where(frame["pupil_valid_frac"].to_numpy() >= 0.8, pupil, np.nan)
    motion = np.fmax(whisk, pupil)
    tracked = np.isfinite(motion)
    stillness = np.where(
        tracked, np.clip(config.still_motion_max + 1 - motion, 0, 1), np.nan
    )
    slow_strength = np.clip(slow / config.quiet_slow_min, 0, 1)
    # Missing behavior supplies no stillness claim, and retains a moderate
    # slow-only resistance rather than behaving like either immobility or motion.
    quiet = slow_strength * (0.25 + 0.75 * np.where(tracked, stillness, 0.5))
    cost = 1 + config.quiet_weight * quiet
    duration = end - start
    remaining = wake & ~fixed
    clear = (
        remaining
        & valid
        & tracked
        & (motion <= config.still_motion_max)
        & (slow >= config.quiet_slow_min)
    )
    protected = np.zeros(n, bool)
    for rid in np.unique(run):
        idx = np.flatnonzero(run == rid)
        for a, b in _runs(clear[idx]):
            ii = idx[a:b]
            if duration[ii].sum() + 1e-9 >= config.quiet_min_s:
                protected[ii] = True
    selected = fixed.copy()
    merged = np.zeros(n, bool)
    reason = np.where(wake, "Quiet: not absorbed", "Original state retained").astype(
        object
    )
    reason[fixed] = "Step 1 Active wake (locked)"
    reason[remaining & ~valid] = "Quiet: EEG evidence unavailable"
    reason[protected] = "Quiet: protected slow + still core"
    gap_seconds = np.full(n, np.nan)
    gap_cost = np.full(n, np.nan)
    adjacent = np.zeros(n, int)
    audits = []
    for rid in np.unique(run):
        idx = np.flatnonzero(run == rid)
        for a, b in _runs(remaining[idx]):
            ii = idx[a:b]
            left = a > 0 and fixed[idx[a - 1]]
            right = b < len(idx) and fixed[idx[b]]
            total = (
                float(np.sum(cost[ii] * duration[ii])) if valid[ii].all() else np.inf
            )
            gap_seconds[ii] = float(duration[ii].sum())
            gap_cost[ii] = total
            adjacent[ii] = int(left) + int(right)
            if (
                left
                and right
                and not protected[ii].any()
                and total <= config.bridge_budget_s + 1e-9
            ):
                merged[ii] = True
                reason[ii] = "Active: bridged short weak-quiet gap"
            else:
                for order in ([ii] if left else []) + ([ii[::-1]] if right else []):
                    spent = 0.0
                    for j in order:
                        if not valid[j] or protected[j]:
                            break
                        spent += cost[j] * duration[j]
                        if spent > config.edge_budget_s + 1e-9:
                            break
                        merged[j] = True
                        reason[j] = "Active: absorbed edge next to Step 1"
            audits.append(
                dict(
                    start_s=float(start[ii[0]]),
                    end_s=float(end[ii[-1]]),
                    duration_s=float(duration[ii].sum()),
                    merge_cost_s=total,
                    active_neighbors=int(left) + int(right),
                    protected_quiet_s=float(duration[ii][protected[ii]].sum()),
                    merged_s=float(duration[ii][merged[ii]].sum()),
                )
            )
    selected |= merged
    labels = np.where(wake, np.where(selected, "Active wake", "Quiet wake"), state)
    frame = frame.with_columns(
        pl.Series("active_selected", selected),
        pl.Series("stage2_label", labels),
        pl.Series("quiet_slow_activity", slow),
        pl.Series("quiet_motion_z", motion),
        pl.Series("stillness_evidence", stillness),
        pl.Series("quiet_evidence", quiet),
        pl.Series("merge_cost_per_s", cost),
        pl.Series("quiet_core_candidate", clear),
        pl.Series("protected_quiet", protected),
        pl.Series("merged_to_active", merged),
        pl.Series("merge_reason", reason),
        pl.Series("remaining_gap_s", gap_seconds),
        pl.Series("remaining_gap_cost_s", gap_cost),
        pl.Series("active_neighbors", adjacent),
        pl.Series("step2_eeg_valid", valid),
        pl.Series("step2_motion_observed", tracked),
        pl.lit(config.quiet_slow_min).alias("quiet_slow_threshold"),
        pl.lit(config.still_motion_max).alias("still_motion_threshold"),
    )
    # Missing-feature barriers stop merging, but must not fragment an unchanged
    # NREM/Quiet label into one-second bouts. Real timestamp gaps remain breaks.
    boundaries = np.r_[
        True, (labels[1:] != labels[:-1]) | (start[1:] > end[:-1] + 1e-8)
    ]
    bouts = (
        frame
        .with_columns(pl.Series("_bout", np.cumsum(boundaries) - 1))
        .group_by("_bout", maintain_order=True)
        .agg(
            pl.col("start_s").first(),
            pl.col("end_s").last(),
            pl.col("stage2_label").first().alias("label"),
            pl.col("duration_s").sum(),
            pl.col("stage1_active").sum().alias("step1_epochs"),
            pl.col("merged_to_active").sum().alias("merged_epochs"),
        )
        .drop("_bout")
    )
    audit = (
        pl.DataFrame(audits)
        if audits
        else pl.DataFrame(
            schema={
                "start_s": pl.Float64,
                "end_s": pl.Float64,
                "duration_s": pl.Float64,
                "merge_cost_s": pl.Float64,
                "active_neighbors": pl.Int64,
                "protected_quiet_s": pl.Float64,
                "merged_s": pl.Float64,
            }
        )
    )
    return dict(
        epochs=frame,
        bouts=bouts,
        active_bouts=bouts.filter(pl.col("label") == "Active wake"),
        quiet_bouts=bouts.filter(pl.col("label") == "Quiet wake"),
        gap_audit=audit,
        config=values,
    )
