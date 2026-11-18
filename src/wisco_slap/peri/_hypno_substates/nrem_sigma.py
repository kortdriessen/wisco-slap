"""Conservative sigma-rich NREM selection after the two frozen wake passes.

Ratios use actual band power, not subject percentiles or absolute amplitude.
This labels sustained spectral substates, not individual spindle events.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.signal import welch

from .active_wake import _runs


@dataclass(frozen=True)
class NremSigmaConfig:
    """Four main dials: dominance, spectral share, support, and gap duration."""

    sigma_slow_min: float = 1.0
    sigma_fraction_min: float = 0.28
    support_fraction: float = 0.70
    bridge_max_s: float = 3.0
    seed_min_s: float = 2.0
    sigma_beta_density_min: float = 1.5
    bridge_score_min: float = 0.50
    slow_barrier_ratio: float = 0.50
    smooth_seconds: int = 3
    peak_fraction_max: float = 0.65


def _temporal_runs(frame: pl.DataFrame, valid: np.ndarray | None = None) -> np.ndarray:
    start, end = frame["start_s"].to_numpy(), frame["end_s"].to_numpy()
    state = frame["original_state"].to_numpy()
    split = (start[1:] > end[:-1] + 1e-8) | (state[1:] != state[:-1])
    if valid is not None:
        split |= ~valid[1:] | ~valid[:-1]
    return np.cumsum(np.r_[True, split])


def build_nrem_sigma_context(
    step2_epochs: pl.DataFrame, eeg: Any, metadata: dict
) -> tuple[pl.DataFrame, dict]:
    """Add raw-EEG spectral powers on the existing one-second grid.

    Four-second, linearly detrended Hann periodograms (Welch with one segment) are
    clipped at original-state boundaries, timestamp gaps and nonfinite EEG. At least
    1.9 seconds of continuous EEG are needed. No interpolation across missing data.
    """
    frame = step2_epochs.sort("start_s")
    if not frame.height or "stage2_label" not in frame.columns:
        raise ValueError("Provide the frozen Step 2 epoch table.")
    t, y = np.asarray(eeg.time, float), np.asarray(eeg, float).squeeze()
    if (
        y.shape != t.shape
        or len(t) < 2
        or np.any(~np.isfinite(t))
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError(
            "Expected a single EEG channel with finite increasing timestamps."
        )
    fs = float(eeg.attrs.get("fs", 1 / np.median(np.diff(t))))
    if not np.isfinite(fs) or fs <= 60:
        raise ValueError("EEG sampling rate must exceed 60 Hz.")
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
    runs = _temporal_runs(frame)
    state_start, state_end = np.empty(frame.height), np.empty(frame.height)
    for rid in np.unique(runs):
        ii = np.flatnonzero(runs == rid)
        state_start[ii] = frame["start_s"][int(ii[0])]
        state_end[ii] = frame["end_s"][int(ii[-1])]
    bands = {
        "nrem_slow_power": (0.5, 5.0),
        "nrem_sigma_power": (9.0, 16.0),
        "nrem_beta_power": (20.0, 30.0),
        "nrem_total_power": (0.5, 30.0),
    }
    window_s = float(metadata.get("nrem_window_s", 4.0))
    powers = {key: np.full(frame.height, np.nan) for key in bands}
    support = np.zeros(frame.height)
    for i, center in enumerate(frame["time"]):
        j = min(int(np.searchsorted(t, center)), len(t) - 1)
        sid = segment[j]
        if sid < 0 or not frame["eeg_valid"][i]:
            continue
        a = max(
            segments[sid][0],
            int(np.searchsorted(t, max(center - window_s / 2, state_start[i]))),
        )
        b = min(
            segments[sid][1],
            int(np.searchsorted(t, min(center + window_s / 2, state_end[i]))),
        )
        support[i] = max(0, b - a) / fs
        if support[i] < 1.9:
            continue
        f, p = welch(y[a:b], fs=fs, nperseg=b - a, detrend="linear")
        df = f[1] - f[0]
        for key, (lo, hi) in bands.items():
            weights = np.maximum(
                0, np.minimum(f + df / 2, hi) - np.maximum(f - df / 2, lo)
            )
            powers[key][i] = float(np.sum(p * weights))
    frame = frame.with_columns(
        [pl.Series(k, v) for k, v in powers.items()]
        + [pl.Series("nrem_spectral_support_s", support)]
    )
    meta = {
        **metadata,
        "step3_algorithm": "nrem-sigma-v1-seeded-hysteresis",
        "step3_bands_hz": {k: list(v) for k, v in bands.items()},
        "step3_window_s": window_s,
        "step3_state_clipped_spectra": True,
        "step3_missing_evidence": "NREM without usable EEG defaults to SWS; validity retained",
        "step3_interpretation": "Sigma-rich spectral substate, not individual spindle detection; SWS is the residual NREM label.",
    }
    return frame, meta


def label_nrem_substates(
    context: pl.DataFrame, config: NremSigmaConfig = NremSigmaConfig()
) -> dict:
    """Seed clear sigma, grow supported edges, bridge short weak gaps, else SWS.

    Seeds need sigma/slow integrated power >= threshold, sigma share of 0.5–30 Hz
    >= threshold, and sigma power density >=1.5 times beta (20–30 Hz) density.
    This last guard rejects broadband flattening. Seeds must last two seconds.
    Connected edges may use relaxed dominance/share thresholds. A final single
    bridge pass fills only short gaps with weaker but still positive sigma support;
    slow dominance, bad EEG, original-state boundaries and timestamp gaps stop it.
    """
    c = config
    values = asdict(c)
    if (
        not all(np.isfinite(v) for v in values.values())
        or c.sigma_slow_min <= 0
        or not 0 < c.sigma_fraction_min < 1
        or not 0 < c.support_fraction <= 1
        or c.bridge_max_s < 0
        or c.seed_min_s <= 0
        or c.sigma_beta_density_min < 1
        or not 0 <= c.bridge_score_min <= c.support_fraction
        or c.slow_barrier_ratio < 0
        or c.slow_barrier_ratio >= c.sigma_slow_min
        or c.smooth_seconds < 1
        or int(c.smooth_seconds) != c.smooth_seconds
        or not 0 < c.peak_fraction_max <= 1
    ):
        raise ValueError("Invalid sigma thresholds, support or duration settings.")
    frame = context.sort("start_s")
    required = {
        "stage2_label",
        "nrem_slow_power",
        "nrem_sigma_power",
        "nrem_beta_power",
        "nrem_total_power",
    }
    if not frame.height or not required <= set(frame.columns):
        raise ValueError(
            "Use build_nrem_sigma_context on the frozen Step 2 result first."
        )
    start, end = frame["start_s"].to_numpy(), frame["end_s"].to_numpy()
    if np.any(end <= start) or np.any(start[1:] < end[:-1] - 1e-8):
        raise ValueError("Epochs must have positive duration without overlap.")
    duration = end - start
    nrem = frame["original_state"].to_numpy() == "NREM"
    powercols = [
        "nrem_slow_power",
        "nrem_sigma_power",
        "nrem_beta_power",
        "nrem_total_power",
    ]
    valid = frame["eeg_valid"].to_numpy() & (
        frame["fast_peak_fraction"].to_numpy() <= c.peak_fraction_max
    )
    for key in powercols:
        v = frame[key].to_numpy()
        valid &= np.isfinite(v) & (v > 0)
    run = _temporal_runs(frame, valid)
    frame = frame.with_columns(pl.Series("nrem_run_id", run))
    # Smooth band powers first, then form ratios: a tiny denominator in one
    # second cannot dominate the local decision. Never smooth across barriers.
    sm = frame.select([
        pl
        .col(k)
        .fill_nan(None)
        .rolling_median(c.smooth_seconds, center=True, min_samples=1)
        .over("nrem_run_id")
        for k in powercols
    ])
    slow, sigma, beta, total = [sm[k].to_numpy() for k in powercols]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sigma / slow
        fraction = sigma / total
        beta_ratio = (sigma / 7) / (beta / 10)
    score = np.minimum(ratio / c.sigma_slow_min, fraction / c.sigma_fraction_min)
    spectral_ok = valid & (beta_ratio >= c.sigma_beta_density_min)
    slow_barrier = nrem & valid & (ratio <= c.slow_barrier_ratio)
    candidate = nrem & spectral_ok & ~slow_barrier & (score >= 1)
    seeds = np.zeros(frame.height, bool)
    for rid in np.unique(run):
        ii = np.flatnonzero(run == rid)
        for a, b in _runs(candidate[ii]):
            jj = ii[a:b]
            if duration[jj].sum() + 1e-9 >= c.seed_min_s:
                seeds[jj] = True
    supported = nrem & spectral_ok & ~slow_barrier & (score >= c.support_fraction)
    selected = seeds.copy()
    # Grow both ways by taking only support components attached to clear seeds.
    for rid in np.unique(run):
        ii = np.flatnonzero(run == rid)
        for a, b in _runs(supported[ii]):
            jj = ii[a:b]
            if seeds[jj].any():
                selected[jj] = True
    grown = selected & ~seeds
    bridged = np.zeros(frame.height, bool)
    bridge_ok = nrem & spectral_ok & ~slow_barrier & (score >= c.bridge_score_min)
    audits = []
    for rid in np.unique(run):
        ii = np.flatnonzero(run == rid)
        for a, b in _runs((nrem & ~selected)[ii]):
            jj = ii[a:b]
            left = a > 0 and selected[ii[a - 1]]
            right = b < len(ii) and selected[ii[b]]
            if not (left and right):
                continue
            accept = bool(
                duration[jj].sum() <= c.bridge_max_s + 1e-9 and bridge_ok[jj].all()
            )
            if accept:
                bridged[jj] = True
            audits.append(
                dict(
                    start_s=float(start[jj[0]]),
                    end_s=float(end[jj[-1]]),
                    duration_s=float(duration[jj].sum()),
                    min_sigma_score=float(np.min(score[jj])),
                    slow_barrier=bool(slow_barrier[jj].any()),
                    merged=accept,
                )
            )
    selected |= bridged
    reason = np.full(frame.height, "Previous state retained", object)
    reason[nrem] = "SWS: no connected clear sigma evidence"
    reason[nrem & ~valid] = "SWS: EEG evidence unavailable (default)"
    reason[slow_barrier] = "SWS: slow-dominated barrier"
    reason[seeds] = "Sigma: clear sustained seed"
    reason[grown] = "Sigma: supported edge of a clear seed"
    reason[bridged] = "Sigma: short supported gap between sigma bouts"
    previous = frame["stage2_label"].to_numpy()
    labels = np.where(nrem, np.where(selected, "NREM sigma", "NREM SWS"), previous)
    frame = frame.with_columns(
        pl.Series("stage3_label", labels),
        pl.Series("final_label", labels),
        pl.Series("sigma_selected", selected),
        pl.Series("sigma_seed", seeds),
        pl.Series("sigma_grown", grown),
        pl.Series("sigma_bridged", bridged),
        pl.Series("sigma_candidate", candidate),
        pl.Series("sigma_slow_ratio", ratio),
        pl.Series("sigma_fraction", fraction),
        pl.Series("sigma_beta_density_ratio", beta_ratio),
        pl.Series("sigma_score", score),
        pl.Series("sigma_support_candidate", supported),
        pl.Series("sws_barrier", slow_barrier),
        pl.Series("step3_eeg_valid", valid),
        pl.Series("nrem_reason", reason),
        pl.lit(c.sigma_slow_min).alias("sigma_slow_threshold"),
        pl.lit(c.sigma_fraction_min).alias("sigma_fraction_threshold"),
        pl.lit(c.support_fraction).alias("sigma_support_threshold"),
    )
    boundaries = np.r_[
        True, (labels[1:] != labels[:-1]) | (start[1:] > end[:-1] + 1e-8)
    ]
    bouts = (
        frame
        .with_columns(pl.Series("_bout", np.cumsum(boundaries)))
        .group_by("_bout", maintain_order=True)
        .agg(
            pl.col("start_s").first(),
            pl.col("end_s").last(),
            pl.col("final_label").first().alias("label"),
            pl.col("duration_s").sum(),
            pl.col("sigma_seed").sum().alias("sigma_seed_epochs"),
            pl.col("sigma_bridged").sum().alias("sigma_bridged_epochs"),
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
                "min_sigma_score": pl.Float64,
                "slow_barrier": pl.Boolean,
                "merged": pl.Boolean,
            }
        )
    )
    return dict(
        epochs=frame,
        bouts=bouts,
        config=values,
        sigma_gap_audit=audit,
        sigma_bouts=bouts.filter(pl.col("label") == "NREM sigma"),
        sws_bouts=bouts.filter(pl.col("label") == "NREM SWS"),
        active_bouts=bouts.filter(pl.col("label") == "Active wake"),
        quiet_bouts=bouts.filter(pl.col("label") == "Quiet wake"),
    )
