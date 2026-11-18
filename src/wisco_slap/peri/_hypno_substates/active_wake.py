"""Seed-first Active-wake selection. Unselected data retain original states.

Fast/HF routes use frequencies >=8 Hz and fold changes above quiet NREM.
A separate positive-only route uses a within-block anchored slow-power index.
Original states provide a
trusted NREM reference and limit accepted labels to original Wake; candidate
detections outside Wake are retained for review. No complete hypnogram is inferred.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.signal import butter, sosfiltfilt, welch


@dataclass(frozen=True)
class ActiveWakeConfig:
    """Heuristic evidence thresholds, not calibrated confidence probabilities."""

    eeg_seed_fold: float = 2.0
    assisted_seed_fold: float = 2.0
    edge_relax_fraction: float = 0.1
    slow_activity_max: float = 0.15
    slow_fraction_max: float = 0.5
    motion_z: float = 3.0
    seed_min_s: float = 2.0
    smooth_seconds: int = 3
    peak_fraction_max: float = 0.65


COLORS = {
    "Clear Active wake": "#e46b37",
    "Not selected yet": "#494949",
    "No EEG": "#888888",
    "Wake": "#ddb444",
    "NREM": "#547dad",
    "REM": "#d56594",
    "Unscored": "#777777",
}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    e = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return list(zip(e[::2], e[1::2]))


def _robust_reference(values: np.ndarray, floor: float) -> dict | None:
    values = values[np.isfinite(values)]
    if len(values) < 30:
        return None
    center = float(np.median(values))
    scale = max(floor, float(1.4826 * np.median(np.abs(values - center))))
    return dict(center=center, scale=scale, n=int(len(values)))


def _z(values: np.ndarray, reference: dict | None) -> np.ndarray:
    if reference is None:
        return np.full(len(values), np.nan)
    return (values - reference["center"]) / reference["scale"]


def build_active_wake_context(
    eeg: Any,
    eye: pl.DataFrame,
    whisk: pl.DataFrame,
    original: pl.DataFrame,
    *,
    grid_origin_s: float | None = None,
    window_s: float = 4.0,
    eye_likelihood_min: float = 0.75,
) -> tuple[pl.DataFrame, dict, np.ndarray]:
    """Measure fresh fast-EEG and camera features on the original one-second grid.

    High-pass each continuous EEG segment at 8 Hz before its PSD, preventing
    large slow swings from leaking into the fast-spectrum estimate. Gaps are
    never interpolated. A separate raw, linearly detrended four-second PSD
    measures delta and the 30-55 Hz fraction for the low-amplitude route.
    The returned high-pass waveform is for raw-data review.
    """
    t = np.asarray(eeg.time, float)
    y = np.asarray(eeg, float).squeeze()
    if (
        t.ndim != 1
        or y.shape != t.shape
        or len(t) < 10
        or np.any(~np.isfinite(t))
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("Expected one EEG channel with increasing timestamps.")
    fs = float(eeg.attrs.get("fs", 1 / np.median(np.diff(t))))
    if fs / 2 <= 55 or window_s < 2:
        raise ValueError("Fast EEG features require Nyquist >55 Hz and windows >=2 s.")
    original = original.select("start_s", "end_s", "label").sort("start_s")
    if np.any(
        original["end_s"].to_numpy()[:-1] > original["start_s"].to_numpy()[1:] + 1e-8
    ):
        raise ValueError("Overlapping original-state intervals.")
    origin = float(original["start_s"][0] if grid_origin_s is None else grid_origin_s)
    start, end = float(t[0]), float(t[-1] + 1 / fs)
    grid = origin + np.arange(np.floor(start - origin), np.ceil(end - origin) + 1)
    edges = np.unique(np.r_[start, grid[(grid > start) & (grid < end)], end])
    # Preserve off-grid original-state boundaries and annotation gaps too.
    source_edges = np.r_[original["start_s"].to_numpy(), original["end_s"].to_numpy()]
    source_edges = source_edges[(source_edges > start) & (source_edges < end)]
    j = np.searchsorted(edges, source_edges)
    distance = np.minimum(
        np.abs(source_edges - edges[np.maximum(j - 1, 0)]),
        np.abs(source_edges - edges[np.minimum(j, len(edges) - 1)]),
    )
    edges = np.unique(np.r_[edges, source_edges[distance > 1e-8]])
    centers = (edges[:-1] + edges[1:]) / 2
    n = len(centers)
    ot0, ot1, olab = (original[c].to_numpy() for c in ["start_s", "end_s", "label"])
    oi = np.searchsorted(ot0, centers, side="right") - 1
    state = np.full(n, "Unscored", object)
    covered = (oi >= 0) & (centers < ot1[np.maximum(oi, 0)])
    state[covered] = olab[oi[covered]]

    hp = np.full(len(y), np.nan)
    sample_segment = np.full(len(y), -1, int)
    good = np.flatnonzero(np.isfinite(y))
    segments = []
    if len(good):
        cuts = np.r_[
            0,
            np.flatnonzero((np.diff(good) > 1) | (np.diff(t[good]) > 1.5 / fs)) + 1,
            len(good),
        ]
        segments = [
            (int(good[a]), int(good[b - 1] + 1)) for a, b in zip(cuts[:-1], cuts[1:])
        ]
    sos = butter(6, 8.0, btype="highpass", fs=fs, output="sos")
    for sid, (a, b) in enumerate(segments):
        sample_segment[a:b] = sid
        if b - a >= 1.9 * fs:
            hp[a:b] = sosfiltfilt(sos, y[a:b])
    bands = {
        "power_8_16": (8, 16),
        "power_20_45": (20, 45),
        "power_35_55": (35, 55),
        "gamma_35_45": (35, 45),
        "gamma_45_55": (45, 55),
        "muscle_65_110": (65, 110),
        "muscle_130_170": (130, 170),
        "high_gamma_65_85": (65, 85),
        "high_gamma_85_110": (85, 110),
        "muscle_130_150": (130, 150),
        "muscle_150_170": (150, 170),
    }
    raw_bands = {
        "raw_delta_power": (0.5, 4),
        "raw_30_55_power": (30, 55),
        "raw_total_0p5_55_power": (0.5, 55),
    }
    columns = {
        c: np.full(n, np.nan)
        for c in [
            *bands,
            *raw_bands,
            "fast_tilt",
            "fast_flatness",
            "fast_peak_fraction",
            "eeg_coverage",
            "spectral_support_s",
            "whisk_mean",
            "whisk_valid_frac",
            "pupil_motion",
            "pupil_diameter",
            "pupil_valid_frac",
        ]
    }
    valid = np.zeros(n, bool)
    seg = np.full(n, -1, int)
    camera_streams = {}
    for name, frame in [("eye", eye), ("whisk", whisk)]:
        frame = frame.sort("time")
        vt = frame["time"].to_numpy()
        dt = np.diff(vt)
        dt = dt[dt > 0]
        camera_streams[name] = (
            frame,
            vt,
            float(1 / np.median(dt)) if len(dt) else 10.0,
        )
    for i, (a, b, c) in enumerate(zip(edges[:-1], edges[1:], centers)):
        lo, hi = np.searchsorted(t, [a, b])
        coverage = min(1.0, np.isfinite(y[lo:hi]).sum() / max(1.0, (b - a) * fs))
        columns["eeg_coverage"][i] = coverage
        j = min(int(np.searchsorted(t, c)), len(t) - 1)
        sid = int(sample_segment[j])
        seg[i] = sid
        if sid >= 0:
            l = max(segments[sid][0], int(np.searchsorted(t, c - window_s / 2)))
            r = min(segments[sid][1], int(np.searchsorted(t, c + window_s / 2)))
            x = hp[l:r]
            columns["spectral_support_s"][i] = len(x) / fs
            if (
                coverage >= 0.9
                and len(x) >= 1.9 * fs
                and np.all(np.isfinite(x))
                and hi - lo >= 3
                and np.nanstd(y[lo:hi]) > 1e-8
            ):
                f, p = welch(
                    x, fs=fs, nperseg=min(len(x), round(2 * fs)), detrend="constant"
                )
                df = f[1] - f[0]
                for name, (low, high) in bands.items():
                    if high < fs / 2:
                        weights = np.maximum(
                            0,
                            np.minimum(f + df / 2, high) - np.maximum(f - df / 2, low),
                        )
                        columns[name][i] = float(np.sum(p * weights))
                # Keep raw delta completely separate from the established HP features.
                rf, rp = welch(y[l:r], fs=fs, nperseg=r - l, detrend="linear")
                rdf = rf[1] - rf[0]
                for name, (low, high) in raw_bands.items():
                    weights = np.maximum(
                        0,
                        np.minimum(rf + rdf / 2, high) - np.maximum(rf - rdf / 2, low),
                    )
                    columns[name][i] = float(np.sum(rp * weights))
                slow_fast = columns["power_8_16"][i] / 8
                fast = columns["power_20_45"][i] / 25
                columns["fast_tilt"][i] = np.log(
                    max(fast, 1e-12) / max(slow_fast, 1e-12)
                )
                spectrum = p[(f >= 8) & (f <= 45)]
                columns["fast_flatness"][i] = np.exp(
                    np.mean(np.log(spectrum + 1e-15))
                ) / (np.mean(spectrum) + 1e-15)
                # An isolated broadband spike must not create a multi-second seed.
                chunk = max(1, round(0.25 * fs))
                energy = np.add.reduceat(x * x, np.arange(0, len(x), chunk))
                columns["fast_peak_fraction"][i] = float(
                    energy.max() / max(energy.sum(), 1e-15)
                )
                valid[i] = True
        ef, et, ehz = camera_streams["eye"]
        ei, ej = np.searchsorted(et, [a, b])
        cam_off = False
        if ej > ei and {"pup_likelihood", "lid_likelihood"} <= set(ef.columns):
            pup = ef["pup_likelihood"].to_numpy()[ei:ej]
            lid = ef["lid_likelihood"].to_numpy()[ei:ej]
            cam_off = (
                np.mean((np.nan_to_num(pup) < 0.2) & (np.nan_to_num(lid) < 0.2)) > 0.8
            )
        if ej > ei and "pup_likelihood" in ef.columns:
            q = ef["pup_likelihood"].to_numpy()[ei:ej]
            for source, out in [
                ("motion", "pupil_motion"),
                ("diameter", "pupil_diameter"),
            ]:
                if source not in ef.columns:
                    continue
                vals = ef[source].to_numpy()[ei:ej]
                ok = (
                    np.isfinite(vals)
                    & np.isfinite(q)
                    & (q >= eye_likelihood_min)
                    & (not cam_off)
                )
                if source == "motion":
                    columns["pupil_valid_frac"][i] = min(
                        1.0, ok.sum() / max(1.0, ehz * (b - a))
                    )
                if ok.any():
                    columns[out][i] = float(vals[ok].mean())
        wf, wt, whz = camera_streams["whisk"]
        wi, wj = np.searchsorted(wt, [a, b])
        if wj > wi:
            vals = wf["whis"].to_numpy()[wi:wj]
            ok = np.isfinite(vals) & (not cam_off)
            columns["whisk_valid_frac"][i] = min(
                1.0, ok.sum() / max(1.0, whz * (b - a))
            )
            if ok.any():
                columns["whisk_mean"][i] = float(vals[ok].mean())
    run = np.cumsum(
        np.r_[
            True,
            (np.diff(seg) != 0) | ~valid[:-1] | ~valid[1:] | (state[1:] != state[:-1]),
        ]
    )
    frame = pl.DataFrame(
        dict(
            start_s=edges[:-1],
            end_s=edges[1:],
            time=centers,
            duration_s=np.diff(edges),
            original_state=state,
            eeg_valid=valid,
            run_id=run,
            **columns,
        )
    )

    interior = np.zeros(n, bool)
    for r in original.filter(pl.col("label") == "NREM").iter_rows(named=True):
        interior |= (centers >= r["start_s"] + max(3.0, window_s / 2)) & (
            centers <= r["end_s"] - max(3.0, window_s / 2)
        )
    reference_base = valid & interior & (columns["fast_peak_fraction"] <= 0.65)
    references = {}
    motion_z = []
    for name, coverage in [
        ("whisk_mean", "whisk_valid_frac"),
        ("pupil_motion", "pupil_valid_frac"),
    ]:
        values = np.log1p(np.maximum(0, columns[name]))
        values[~(columns[coverage] >= 0.8)] = np.nan
        ref = _robust_reference(values[reference_base], 0.15)
        references[name] = ref
        motion_z.append(_z(values, ref))
    # Calibrate spectral shape in quiet NREM, allowing absent video.
    quiet = reference_base.copy()
    for z in motion_z:
        quiet &= ~np.isfinite(z) | (z < 1.5)
    if np.sum(quiet) < 30:
        quiet = reference_base
    tilt = columns["fast_tilt"]
    flat = np.clip(columns["fast_flatness"], 1e-6, 1 - 1e-6)
    logit_flat = np.log(flat / (1 - flat))
    references["fast_tilt"] = _robust_reference(tilt[quiet], 0.15)
    references["flatness_logit"] = _robust_reference(logit_flat[quiet], 0.15)
    hf_z = {}
    for name in [
        "gamma_35_45",
        "gamma_45_55",
        "muscle_65_110",
        "muscle_130_170",
        "high_gamma_65_85",
        "high_gamma_85_110",
        "muscle_130_150",
        "muscle_150_170",
    ]:
        values = np.log(np.maximum(columns[name], 1e-12))
        references[name] = _robust_reference(values[quiet], 0.15)
        hf_z[name + "_z"] = _z(values, references[name])
    delta_log = np.log(np.maximum(columns["raw_delta_power"], 1e-12))
    gamma_fraction = columns["raw_30_55_power"] / np.maximum(
        columns["raw_total_0p5_55_power"], 1e-12
    )
    clipped_fraction = np.clip(gamma_fraction, 1e-6, 1 - 1e-6)
    gamma_logit = np.log(clipped_fraction / (1 - clipped_fraction))
    references["raw_delta_log"] = _robust_reference(delta_log[quiet], 0.15)
    references["relative_gamma_logit"] = _robust_reference(gamma_logit[quiet], 0.15)
    delta_reference = references["raw_delta_log"]
    fraction_reference = references["relative_gamma_logit"]
    delta_fold = (
        columns["raw_delta_power"] / np.exp(delta_reference["center"])
        if delta_reference
        else np.full(n, np.nan)
    )
    gamma_reference = (
        1 / (1 + np.exp(-fraction_reference["center"]))
        if fraction_reference
        else np.nan
    )
    frame = frame.with_columns(
        pl.Series("reference_epoch", quiet),
        pl.Series("tilt_z", _z(tilt, references["fast_tilt"])),
        pl.Series("flatness_z", _z(logit_flat, references["flatness_logit"])),
        pl.Series("whisk_z", motion_z[0]),
        pl.Series("pupil_motion_z", motion_z[1]),
        pl.Series("delta_suppression_z", -_z(delta_log, delta_reference)),
        pl.Series("relative_gamma_z", _z(gamma_logit, fraction_reference)),
        pl.Series("delta_fold_nrem", delta_fold),
        pl.Series("relative_gamma_fraction", gamma_fraction),
        pl.Series("relative_gamma_fold_nrem", gamma_fraction / gamma_reference),
        *[pl.Series(k, v) for k, v in hf_z.items()],
    )
    metadata = dict(
        algorithm="active-wake-stage1-v2-low-delta",
        references=references,
        reference_seconds=float(np.diff(edges)[quiet].sum()),
        reference_available=all(
            references[k] is not None for k in ["fast_tilt", "flatness_logit"]
        ),
        reference_definition="original NREM interiors, quiet when tracked behavior is available",
        fs_hz=fs,
        window_s=window_s,
        grid_origin_s=origin,
        timebase="TDT sync-block seconds",
        feature_min_frequency_hz=0.5,
        fast_route_min_frequency_hz=8.0,
        delta_used=True,
        load_cell_used=False,
        delta_role="positive-only low-delta AND relative 30-55 Hz route; never gates fast/HF routes",
        low_delta_reference_available=delta_reference is not None
        and fraction_reference is not None,
        eye_likelihood_min=eye_likelihood_min,
        interpretation="Positive Active-wake mask; unselected epochs retain their original state, not Quiet wake.",
    )
    frame, metadata = calibrate_active_wake_context(frame, metadata)
    return frame, metadata, hp


def calibrate_active_wake_context(
    frame: pl.DataFrame, metadata: dict
) -> tuple[pl.DataFrame, dict]:
    """Calibrate measured features without reloading EEG or changing time bins.

    EEG enrichment uses log2 fold change, never NREM dispersion. Slow power
    is placed between the Wake 10th percentile and quiet-NREM 75th percentile.
    These are empirical reference levels, not certified pure states. A minimum
    2.5-fold separation and adequate data are required; otherwise that route is
    disabled, with the reason exposed. No target Active-wake occupancy is set.
    """
    refs = metadata["references"]
    updates = []
    for name, old in [("fast_tilt", "tilt_z"), ("flatness_logit", "flatness_z")] + [
        (k, k + "_z")
        for k in [
            "gamma_35_45",
            "gamma_45_55",
            "high_gamma_65_85",
            "high_gamma_85_110",
            "muscle_130_150",
            "muscle_150_170",
        ]
    ]:
        ref = refs.get(name)
        values = (
            frame[old].to_numpy() * ref["scale"] / np.log(2)
            if ref
            else np.full(frame.height, np.nan)
        )
        updates.append(pl.Series(old.removesuffix("_z") + "_log2", values))
    good = (
        frame["eeg_valid"].to_numpy()
        & (frame["fast_peak_fraction"].to_numpy() <= 0.65)
        & (frame["duration_s"].to_numpy() >= 0.999)
    )
    st = frame["original_state"].to_numpy()
    p = frame["raw_delta_power"].to_numpy()
    smooth = frame.select(
        pl
        .col("raw_delta_power")
        .fill_nan(None)
        .rolling_median(3, center=True, min_samples=1)
        .over("run_id")
    )["raw_delta_power"].to_numpy()
    wake = good & (st == "Wake") & np.isfinite(smooth)
    nrem = good & frame["reference_epoch"].to_numpy() & np.isfinite(smooth)
    enough = wake.sum() >= 60 and nrem.sum() >= 30
    low = float(np.quantile(smooth[wake], 0.1)) if enough else None
    high = float(np.quantile(smooth[nrem], 0.75)) if enough else None
    available = bool(enough and low > 0 and high / low >= 2.5)
    reason = (
        "available"
        if available
        else "insufficient Wake/NREM reference seconds"
        if not enough
        else "slow-rich NREM / low-Wake contrast is below 2.5-fold"
    )
    index = (p - low) / (high - low) if available else np.full(len(p), np.nan)
    updates.extend([
        pl.Series("slow_activity_index", index),
        pl.Series(
            "raw_delta_fraction",
            p / np.maximum(frame["raw_total_0p5_55_power"].to_numpy(), 1e-12),
        ),
    ])
    meta = {
        **metadata,
        "algorithm": "active-wake-stage1-v3-anchored",
        "eeg_evidence_units": "log2 fold change from quiet NREM; not divided by NREM spread",
        "delta_role": "positive-only anchored low-slow route, no gamma enrichment or motion required",
        "low_delta_reference_available": available,
        "slow_calibration": dict(
            available=available,
            reason=reason,
            wake_floor_power=low,
            slow_nrem_power=high,
            contrast_fold=high / low if enough and low > 0 else None,
            wake_reference_s=int(wake.sum()),
            nrem_reference_s=int(nrem.sum()),
            wake_quantile=0.1,
            nrem_quantile=0.75,
            min_contrast_fold=2.5,
            anchor_smoothing_seconds=3,
        ),
    }
    return frame.with_columns(updates), meta


def select_active_wake(
    context: pl.DataFrame, config: ActiveWakeConfig = ActiveWakeConfig()
) -> dict:
    """Lock sustained clear seeds, then greedily grow them through local support.

    Standalone seeds use fast spectral shape, very strong gamma, OR very strong
    broadband muscle evidence. Gamma/muscle each require two bands to reject a
    single narrow spectral peak. The assisted route requires positive EEG
    evidence plus strongly elevated valid pupil or whisker motion. A separate
    low-slow route can seed without movement or gamma enrichment. It only adds candidates/support and cannot veto any fast/HF
    detection. Original NREM/REM are not overwritten.
    """
    if (
        min(config.eeg_seed_fold, config.assisted_seed_fold) <= 1
        or min(config.motion_z, config.seed_min_s) <= 0
        or not 0 <= config.slow_activity_max <= 1
        or not 0 < config.slow_fraction_max <= 1
        or not 0 <= config.edge_relax_fraction < 1
        or config.smooth_seconds < 1
        or int(config.smooth_seconds) != config.smooth_seconds
        or not 0 < config.peak_fraction_max <= 1
        or not all(np.isfinite(v) for v in asdict(config).values())
    ):
        raise ValueError("Positive evidence/persistence thresholds are required.")
    frame = context.sort("start_s")
    if "tilt_log2" not in frame.columns:
        raise ValueError(
            "Rebuild context with the current launcher, or call calibrate_active_wake_context(context, metadata)."
        )
    # Missing slow measurements disable only that route.
    for col in ["slow_activity_index", "raw_delta_fraction"]:
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(float("nan")).alias(col))
    cols = [
        "tilt_log2",
        "flatness_log2",
        "whisk_z",
        "pupil_motion_z",
        "gamma_35_45_log2",
        "gamma_45_55_log2",
        "high_gamma_65_85_log2",
        "high_gamma_85_110_log2",
        "muscle_130_150_log2",
        "muscle_150_170_log2",
        "slow_activity_index",
        "raw_delta_fraction",
    ]
    frame = frame.with_columns([
        pl
        .col(c)
        .fill_nan(None)
        .rolling_median(config.smooth_seconds, min_samples=1, center=True)
        .over("run_id")
        .alias(c + "_smooth")
        for c in cols
    ])
    shape = np.minimum(
        frame["tilt_log2_smooth"].to_numpy(), frame["flatness_log2_smooth"].to_numpy()
    )
    gamma = np.minimum(
        frame["gamma_35_45_log2_smooth"].to_numpy(),
        frame["gamma_45_55_log2_smooth"].to_numpy(),
    )
    high_gamma = np.minimum(
        frame["high_gamma_65_85_log2_smooth"].to_numpy(),
        frame["high_gamma_85_110_log2_smooth"].to_numpy(),
    )
    muscle = np.minimum(
        frame["muscle_130_150_log2_smooth"].to_numpy(),
        frame["muscle_150_170_log2_smooth"].to_numpy(),
    )
    routes = np.column_stack([shape, gamma, high_gamma, muscle])
    eeg = np.fmax(np.fmax(shape, gamma), np.fmax(high_gamma, muscle))
    winner = np.argmax(np.nan_to_num(routes, nan=-np.inf), axis=1)
    wz, pz = (
        frame["whisk_z_smooth"].to_numpy(),
        frame["pupil_motion_z_smooth"].to_numpy(),
    )
    # Missing local tracking cannot borrow a neighbour's motion evidence.
    wz = np.where(frame["whisk_valid_frac"].to_numpy() >= 0.8, wz, np.nan)
    pz = np.where(frame["pupil_valid_frac"].to_numpy() >= 0.8, pz, np.nan)
    motion = np.fmax(wz, pz)
    behavior = np.isfinite(motion) & (motion >= config.motion_z)
    slow = frame["slow_activity_index_smooth"].to_numpy().copy()
    slow[~np.isfinite(frame["slow_activity_index"].to_numpy())] = np.nan
    # Reject tiny-amplitude waveforms that are still delta-dominated. This
    # broad shape guard is NOT a requirement for extra gamma enrichment.
    fraction = frame["raw_delta_fraction_smooth"].to_numpy()
    slow_ok = (
        np.isfinite(slow)
        & np.isfinite(fraction)
        & (fraction <= config.slow_fraction_max)
        & np.isfinite(frame["raw_delta_fraction"].to_numpy())
    )
    feature_valid = (
        frame["eeg_valid"].to_numpy()
        & (np.isfinite(eeg) | slow_ok)
        & (frame["fast_peak_fraction"].to_numpy() <= config.peak_fraction_max)
    )
    eeg_threshold = np.log2(config.eeg_seed_fold)
    assisted_threshold = np.log2(config.assisted_seed_fold)
    standalone = feature_valid & (eeg >= eeg_threshold)
    assisted = feature_valid & (eeg >= assisted_threshold) & behavior
    low_delta_candidate = feature_valid & slow_ok & (slow <= config.slow_activity_max)
    candidate = standalone | assisted | low_delta_candidate
    eligible = feature_valid & (frame["original_state"].to_numpy() == "Wake")
    seed = np.zeros(frame.height, bool)
    duration = frame["duration_s"].to_numpy()
    run = frame["run_id"].to_numpy()
    grow_eeg = np.log2(
        max(1.05, config.eeg_seed_fold * 2 ** (-config.edge_relax_fraction))
    )
    grow_assisted = np.log2(
        max(1.05, config.assisted_seed_fold * 2 ** (-config.edge_relax_fraction))
    )
    grow_slow = min(1.0, config.slow_activity_max + config.edge_relax_fraction)
    support = eligible & (
        (eeg >= grow_eeg)
        | ((eeg >= grow_assisted) & behavior)
        | (slow_ok & (slow <= grow_slow))
    )
    seed_runs = []
    for rid in np.unique(run):
        idx = np.flatnonzero(run == rid)
        for a, b in _runs(candidate[idx] & eligible[idx]):
            ii = idx[a:b]
            if duration[ii].sum() + 1e-9 >= config.seed_min_s:
                seed[ii] = True
                seed_runs.append((
                    float(np.nanmax(np.fmax(eeg[ii], 1 - slow[ii]))),
                    int(ii[0]),
                    int(ii[-1] + 1),
                ))
    selected = seed.copy()
    # Strongest seed first; no nearest-state filling or unsupported gap bridging.
    for _, a, b in sorted(seed_runs, key=lambda r: (-r[0], r[1])):
        i = a - 1
        while i >= 0 and run[i] == run[a] and support[i]:
            selected[i] = True
            i -= 1
        i = b
        while i < len(selected) and run[i] == run[a] and support[i]:
            selected[i] = True
            i += 1
    reason = np.full(frame.height, "Not selected yet", object)
    reason[selected] = "Supported edge"
    reason[seed & low_delta_candidate] = "Low-slow seed"
    reason[seed & assisted] = "EEG + motion seed"
    for k, name in enumerate([
        "Fast-shape seed",
        "Gamma seed",
        "High-gamma seed",
        "Muscle seed",
    ]):
        reason[seed & standalone & (winner == k)] = name
    # The selection is a positive overlay on the immutable original hypnogram.
    # Invalid/unsupported/unselected data keep their ORIGINAL label and validity.
    labels = np.where(selected, "Clear Active wake", frame["original_state"].to_numpy())
    frame = frame.with_columns(
        pl.Series("eeg_enrichment_fold", 2.0**eeg),
        pl.Series("motion_evidence_z", motion),
        pl.Series("fast_shape_fold", 2.0**shape),
        pl.Series("gamma_enrichment_fold", 2.0**gamma),
        pl.Series("high_gamma_enrichment_fold", 2.0**high_gamma),
        pl.Series("muscle_enrichment_fold", 2.0**muscle),
        pl.Series("slow_activity", slow),
        pl.Series("low_delta_candidate", low_delta_candidate),
        pl.Series("eeg_only_candidate", standalone),
        pl.Series("assisted_candidate", assisted),
        pl.Series("active_seed", seed),
        pl.Series("active_selected", selected),
        pl.Series("selection_reason", reason),
        pl.Series("stage1_label", labels),
        pl.Series(
            "outside_wake_candidate",
            candidate & (frame["original_state"].to_numpy() != "Wake"),
        ),
        pl.lit(config.eeg_seed_fold).alias("eeg_seed_threshold"),
        pl.lit(config.assisted_seed_fold).alias("assisted_seed_threshold"),
        pl.lit(config.slow_activity_max).alias("slow_seed_threshold"),
        pl.lit(grow_slow).alias("slow_edge_threshold"),
        pl.lit(2.0**grow_eeg).alias("eeg_edge_threshold"),
        pl.lit(2.0**grow_assisted).alias("assisted_edge_threshold"),
    )
    boundaries = np.r_[True, (labels[1:] != labels[:-1]) | (np.diff(run) != 0)]
    frame = frame.with_columns(pl.Series("_bout", np.cumsum(boundaries) - 1))
    bouts = (
        frame
        .group_by("_bout", maintain_order=True)
        .agg(
            pl.col("start_s").first(),
            pl.col("end_s").last(),
            pl.col("stage1_label").first().alias("label"),
            pl.col("duration_s").sum(),
            pl.col("active_seed").sum().alias("seed_epochs"),
        )
        .drop("_bout")
    )
    return dict(
        epochs=frame.drop("_bout"),
        bouts=bouts,
        active_bouts=bouts.filter(pl.col("label") == "Clear Active wake"),
        config=asdict(config),
    )
