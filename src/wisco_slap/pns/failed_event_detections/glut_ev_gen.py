"""Pure generator functions for glutamate event detection.

Detects glutamate release events from iGluSnFR4f fluorescence traces using
L1-regularized non-negative deconvolution (FISTA) on the **nonneg** trace,
with hierarchical empirical Bayes refinement.

Pipeline:
1. Generate nonneg trace from LS via matched filter + Richardson-Lucy
   (using the same algorithms as the old NoLoCo pipeline, via pro.py)
2. Normalize by time-varying F0 → nonneg dF/F
3. Highpass at 0.05 Hz + noise estimation (MAD of first differences)
4. FISTA deconvolution with L1 sparsity penalty
5. Threshold a_hat > amp_thresh_sigma * sigma, merge contiguous events
6. Optional hierarchical empirical Bayes refinement

The nonneg trace has dramatically better SNR than raw LS because
Richardson-Lucy enforces non-negativity and suppresses noise iteratively.
FISTA on this pre-cleaned signal gives very sparse, clean output.

All existence checking and orchestration is in glut_ev_mon.py.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import polars as pl
import slap2_py as spy

import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.pns.glut_ev_defs import (
    DEFAULT_AMP_THRESH_SIGMA,
    DEFAULT_EB_ALPHA_SCALE_BOUNDS,
    DEFAULT_EB_RATE_SHRINK_BETA,
    DEFAULT_EB_SECOND_PASS,
    DEFAULT_MIN_PEAK_DISTANCE_S,
    FISTA_BASELINE_CUTOFF_HZ,
    FISTA_LAM_MULT,
    FISTA_MAX_ITER,
    FISTA_TOL,
    IGLUSNFR_TAU_S,
    MAD_NORMAL_SCALE,
    OUTPUT_DIR_NAME,
)
from wisco_slap.scope.pro import (
    _conv1d_last,
    _make_exp_kernel,
    _matched_exp_filter,
    _richardson_lucy,
)

try:
    from scipy import signal
except ImportError as e:
    raise ImportError(
        "glut_ev_gen requires scipy. Please `uv add scipy`."
    ) from e


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _scopex_dir(subject: str, exp: str, loc: str, acq: str) -> str:
    return os.path.join(
        DEFS.anmat_root, subject, exp, "scopex", f"{loc}--{acq}"
    )


def load_traces_from_scopex(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    channel: int = 0,
) -> dict[str, dict[str, Any]]:
    """Load ls and F0 traces from scopex zarrs.

    Returns
    -------
    dict mapping DMD name to a dict with keys:
        ``"ls"``, ``"f0_ts"``, ``"f0_median"``, ``"fs"``, ``"time"``,
        ``"syn_ids"``.
    """
    sx_dir = _scopex_dir(subject, exp, loc, acq)
    sel = {"channel": channel}

    ls_xr = spy.core.xarr_summ.load_xr_from_zarr(
        os.path.join(sx_dir, "syn_dF-ls.zarr"), sel=sel
    )
    f0_xr = spy.core.xarr_summ.load_xr_from_zarr(
        os.path.join(sx_dir, "syn_F0.zarr"), sel=sel
    )

    result: dict[str, dict[str, Any]] = {}
    for dmd_key in sorted(ls_xr.keys()):
        ls_da = ls_xr[dmd_key]
        f0_da = f0_xr[dmd_key]
        ls_arr = np.asarray(ls_da.values, dtype=np.float64)
        f0_arr = np.asarray(f0_da.values, dtype=np.float64)
        time_coords = ls_da.coords["time"].values.astype(np.float64)
        syn_ids = ls_da.coords["syn_id"].values
        fs = float(1.0 / np.median(np.diff(time_coords)))
        result[dmd_key] = {
            "ls": ls_arr,
            "f0_ts": f0_arr,
            "f0_median": np.nanmedian(f0_arr, axis=-1),
            "fs": fs,
            "time": time_coords,
            "syn_ids": syn_ids,
        }
    return result


# ---------------------------------------------------------------------------
# Nonneg trace generation (from pro.py internals)
# ---------------------------------------------------------------------------


def _ls_to_nonneg_numpy(
    ls: np.ndarray,
    fs: float,
    tau_s: float = IGLUSNFR_TAU_S,
    rl_iterations: int = 20,
    nan_fill_rounds: int = 3,
) -> np.ndarray:
    """Generate nonneg trace from LS via matched filter + Richardson-Lucy.

    Operates on numpy arrays directly (no xarray). Equivalent to
    ``pro.ls_to_nonneg()`` but for shape (n_syn, n_time) arrays.

    Parameters
    ----------
    ls : np.ndarray, shape (n_syn, n_time)
    fs : float
    tau_s : float
    rl_iterations : int
    nan_fill_rounds : int

    Returns
    -------
    np.ndarray, shape (n_syn, n_time) — nonneg trace.
    """
    tau_frames = tau_s * fs

    # 1. Matched filter (backward exponential)
    match_data = _matched_exp_filter(ls, tau_frames)

    # 2. Build backward (time-reversed) kernel
    kernel = _make_exp_kernel(tau_frames)
    f_kernel = kernel[::-1].copy()

    # 3. Shift for positivity
    nan_mask = np.isnan(match_data)
    hsub = (
        np.nanmedian(match_data, axis=-1, keepdims=True)
        - 2.0 * np.nanpercentile(match_data, 1, axis=-1, keepdims=True)
    )
    observed = np.where(nan_mask, 0.0, match_data) + hsub
    observed = np.maximum(observed, 0.0)

    weights = (~nan_mask).astype(np.float64)

    # 4. Initial RL deconvolution
    estimate = _richardson_lucy(observed, f_kernel, rl_iterations, weights)

    # 5. Iterative NaN-fill rounds
    if nan_mask.any():
        for _ in range(nan_fill_rounds):
            recon = _conv1d_last(estimate, f_kernel)
            observed[nan_mask] = recon[nan_mask]
            estimate = _richardson_lucy(
                observed, f_kernel, rl_iterations, weights,
                initial_estimate=estimate,
            )

    # 6. Remove shift, restore NaNs
    result = estimate - hsub
    result[nan_mask] = np.nan
    return result


# ---------------------------------------------------------------------------
# Noise estimation
# ---------------------------------------------------------------------------


def _robust_sigma_from_diff(x: np.ndarray) -> float:
    """Robust noise sigma from MAD of first differences, ignoring NaNs."""
    d = np.diff(x)
    d = d[np.isfinite(d)]
    if d.size < 10:
        return 0.0
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    return float(mad / MAD_NORMAL_SCALE / np.sqrt(2.0) + 1e-12)


def estimate_noise_array(traces: np.ndarray) -> np.ndarray:
    """Estimate per-synapse GLOBAL noise (MAD of first differences).

    Parameters
    ----------
    traces : np.ndarray, shape (n_syn, n_time)

    Returns
    -------
    np.ndarray, shape (n_syn,)
    """
    n_syn, _ = traces.shape
    sigmas = np.empty(n_syn, dtype=np.float64)
    for s in range(n_syn):
        sigmas[s] = _robust_sigma_from_diff(traces[s])
    return sigmas


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def _next_pow_two(n: int) -> int:
    return 1 << (int(n - 1).bit_length())


def _build_exp_kernel(
    fs: float, tau_s: float, eps: float = 1e-3
) -> np.ndarray:
    """Causal single-exponential decay kernel h(t)=exp(-t/tau), h(0)=1."""
    tau_s = float(tau_s)
    if tau_s <= 0:
        raise ValueError("tau_s must be > 0")
    L = int(np.ceil((-tau_s * np.log(eps)) * fs)) + 1
    t = np.arange(L, dtype=np.float64) / fs
    k = np.exp(-t / tau_s)
    k /= k[0]
    return k


def _butter_highpass(
    y: np.ndarray, fs: float, cutoff_hz: float, order: int = 2
) -> np.ndarray:
    """Zero-phase Butterworth highpass (lowpass baseline removal)."""
    if cutoff_hz is None or cutoff_hz <= 0:
        return y.astype(np.float64, copy=True)
    nyq = fs / 2.0
    w = float(cutoff_hz / nyq)
    w = min(max(w, 1e-6), 0.99)
    sos = signal.butter(order, w, btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, y.astype(np.float64, copy=False))


def _fill_nans_linear(x: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation; edge NaNs forward/backfilled."""
    if not np.isnan(x).any():
        return x
    x = x.copy()
    idx = np.arange(x.size)
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


# ---------------------------------------------------------------------------
# Core solver: FISTA for nonneg L1 with convolution operator
# ---------------------------------------------------------------------------


def _precompute_fista_fft(
    kernel: np.ndarray, n_time: int
) -> dict[str, Any]:
    """Precompute FFT quantities shared across all synapses."""
    k = kernel.astype(np.float64, copy=False)
    Lk = k.size
    full_len = n_time + Lk - 1
    nfft = _next_pow_two(full_len)
    fft_k = np.fft.rfft(k, nfft)
    fft_k_rev = np.fft.rfft(k[::-1], nfft)
    max_power = float(np.max(np.abs(fft_k) ** 2))
    lips = max_power + 1e-12
    step = 1.0 / lips
    return {
        "nfft": nfft, "fft_k": fft_k, "fft_k_rev": fft_k_rev,
        "lips": lips, "step": step, "full_len": full_len, "Lk": Lk,
    }


def _fista_nonneg_l1_conv(
    y: np.ndarray,
    alpha: float,
    precomp: dict[str, Any],
    max_iter: int = 80,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve min_{a>=0} 0.5||(a*k)[:T]-y||^2 + alpha*sum(a) via FISTA.

    Returns (a_hat, recon) both shape (T,).
    """
    y = y.astype(np.float64, copy=False)
    T = y.size
    nfft = precomp["nfft"]
    fft_k = precomp["fft_k"]
    fft_k_rev = precomp["fft_k_rev"]
    step = precomp["step"]
    full_len = precomp["full_len"]
    Lk = precomp["Lk"]
    thr = float(alpha) * step

    a = np.zeros(T, dtype=np.float64)
    z = a.copy()
    t = 1.0
    a_pad = np.zeros(full_len, dtype=np.float64)
    r_full = np.zeros(full_len, dtype=np.float64)

    def conv_full(x_full, fft_ker):
        X = np.fft.rfft(x_full, nfft)
        return np.fft.irfft(X * fft_ker, nfft)[:full_len]

    prev_obj = np.inf
    for it in range(max_iter):
        a_pad[:T] = z
        a_pad[T:] = 0.0
        pred = conv_full(a_pad, fft_k)[:T]
        resid = pred - y
        r_full[:T] = resid
        r_full[T:] = 0.0
        grad = conv_full(r_full, fft_k_rev)[Lk - 1: Lk - 1 + T]
        x = z - step * grad
        a_new = x - thr
        a_new[a_new < 0.0] = 0.0
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = a_new + ((t - 1.0) / t_new) * (a_new - a)
        a = a_new
        t = t_new
        if it % 5 == 0 or it == max_iter - 1:
            a_pad[:T] = a
            a_pad[T:] = 0.0
            pred = conv_full(a_pad, fft_k)[:T]
            resid = pred - y
            obj = 0.5 * float(np.dot(resid, resid)) + alpha * float(
                a.sum()
            )
            if np.isfinite(prev_obj):
                if abs(prev_obj - obj) / (prev_obj + 1e-12) < tol:
                    break
            prev_obj = obj

    a_pad[:T] = a
    a_pad[T:] = 0.0
    recon = conv_full(a_pad, fft_k)[:T]
    return a, recon


# ---------------------------------------------------------------------------
# Event extraction from FISTA a_hat (old bayglutev style)
# ---------------------------------------------------------------------------


def _extract_events_from_a(
    a: np.ndarray,
    fs: float,
    amp_min: float,
    merge_gap_s: float = 0.01,
) -> list[dict[str, Any]]:
    """Threshold a_hat and merge contiguous above-threshold samples.

    Returns list of event dicts (no synapse id — caller adds it).
    """
    idx = np.where(a > amp_min)[0]
    if idx.size == 0:
        return []
    merge_gap = max(0, int(round(merge_gap_s * fs)))
    events: list[dict[str, Any]] = []
    start = int(idx[0])
    prev = int(idx[0])

    def flush(seg_start: int, seg_end: int) -> None:
        seg = np.arange(seg_start, seg_end + 1)
        peak_i = int(seg[np.argmax(a[seg])])
        events.append({
            "peak_idx": peak_i,
            "time": float(peak_i / fs),
            "amp_peak": float(a[peak_i]),
            "amp_sum": float(a[seg].sum()),
            "width_s": float((seg_end - seg_start + 1) / fs),
        })

    for i in idx[1:]:
        i = int(i)
        if i <= prev + merge_gap + 1:
            prev = i
        else:
            flush(start, prev)
            start = i
            prev = i
    flush(start, prev)
    return events


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def detect_events(
    ls: np.ndarray,
    f0_ts: np.ndarray,
    f0_median: np.ndarray,
    fs: float,
    time: np.ndarray,
    tau_s: float = IGLUSNFR_TAU_S,
    lam_mult: float = FISTA_LAM_MULT,
    max_iter: int = FISTA_MAX_ITER,
    tol: float = FISTA_TOL,
    baseline_cutoff_hz: float = FISTA_BASELINE_CUTOFF_HZ,
    amp_thresh_sigma: float = DEFAULT_AMP_THRESH_SIGMA,
    merge_gap_s: float = DEFAULT_MIN_PEAK_DISTANCE_S,
    second_pass: bool = DEFAULT_EB_SECOND_PASS,
    rate_shrink_beta: float = DEFAULT_EB_RATE_SHRINK_BETA,
    alpha_scale_bounds: tuple[float, float] = DEFAULT_EB_ALPHA_SCALE_BOUNDS,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run FISTA deconvolution on nonneg dF/F and extract events.

    Pipeline:
    1. Generate nonneg from LS (matchFilt + Richardson-Lucy)
    2. Normalize by F0(t) → nonneg dF/F
    3. Highpass + noise estimation
    4. FISTA deconvolution
    5. Threshold a_hat + EB refinement

    Returns
    -------
    events : list of event dicts
    rates_hz : np.ndarray, shape (n_syn,)
    a_hat : np.ndarray, shape (n_syn, n_time)
    recon : np.ndarray, shape (n_syn, n_time)
    """
    n_syn, n_time = ls.shape
    duration_s = float(time[-1] - time[0]) + 1.0 / fs

    # --- Step 1: Generate nonneg from LS ---
    print("    Generating nonneg trace (matchFilt + RL)...")
    nonneg = _ls_to_nonneg_numpy(ls, fs, tau_s)

    # --- Step 2: Normalize to dF/F ---
    f0_safe = np.where(
        (f0_ts > 0) & np.isfinite(f0_ts), f0_ts, np.nan
    )
    nonneg_dff = nonneg / f0_safe

    # --- Step 3: NaN fill + highpass + sigma ---
    nonneg_hp = np.empty_like(nonneg_dff)
    for s in range(n_syn):
        y = _fill_nans_linear(nonneg_dff[s])
        nonneg_hp[s] = _butter_highpass(y, fs, baseline_cutoff_hz)

    sigmas = estimate_noise_array(nonneg_hp)

    # --- Step 4: FISTA deconvolution ---
    kernel = _build_exp_kernel(fs=fs, tau_s=tau_s)
    precomp = _precompute_fista_fft(kernel, n_time)

    def alpha_from_sigma(sigma: float, scale: float = 1.0) -> float:
        return float(
            scale * lam_mult * sigma
            * np.sqrt(2.0 * np.log(max(10, n_time)))
        )

    a_hat = np.zeros((n_syn, n_time), dtype=np.float64)
    recon = np.zeros((n_syn, n_time), dtype=np.float64)

    print("    Running FISTA pass 1...")
    for s in range(n_syn):
        alpha_s = alpha_from_sigma(sigmas[s])
        a_s, r_s = _fista_nonneg_l1_conv(
            nonneg_hp[s], alpha_s, precomp, max_iter, tol
        )
        a_hat[s] = a_s
        recon[s] = r_s

    # --- Pass 1 event extraction for EB rate estimation ---
    rates_hz = np.zeros(n_syn, dtype=np.float64)
    for s in range(n_syn):
        amp_min = amp_thresh_sigma * sigmas[s]
        evs = _extract_events_from_a(a_hat[s], fs, amp_min, merge_gap_s)
        rates_hz[s] = len(evs) / max(duration_s, 1e-12)

    # --- Pass 2: hierarchical EB ---
    eb_scales = np.ones(n_syn, dtype=np.float64)
    if second_pass:
        eps = 1e-6
        pop_rate = float(
            np.exp(np.mean(np.log(rates_hz + eps)))
        )
        lo, hi = alpha_scale_bounds
        eb_scales = (pop_rate / (rates_hz + eps)) ** float(
            rate_shrink_beta
        )
        eb_scales = np.clip(eb_scales, lo, hi)

        print("    Running FISTA pass 2 (EB)...")
        for s in range(n_syn):
            alpha_s = alpha_from_sigma(
                sigmas[s], scale=float(eb_scales[s])
            )
            a_s, r_s = _fista_nonneg_l1_conv(
                nonneg_hp[s], alpha_s, precomp, max_iter, tol
            )
            a_hat[s] = a_s
            recon[s] = r_s

    # --- Final event extraction ---
    all_events: list[dict[str, Any]] = []
    rates_hz_final = np.zeros(n_syn, dtype=np.float64)
    for s in range(n_syn):
        amp_min = amp_thresh_sigma * sigmas[s]
        actual_alpha = alpha_from_sigma(
            sigmas[s], scale=float(eb_scales[s])
        )
        f0_ok = np.isfinite(f0_median[s]) and f0_median[s] > 0
        f0_med_s = float(f0_median[s]) if f0_ok else np.nan

        evs = _extract_events_from_a(
            a_hat[s], fs, amp_min, merge_gap_s
        )
        rates_hz_final[s] = len(evs) / max(duration_s, 1e-12)
        for e in evs:
            e["source_id"] = int(s)
            e["sigma"] = float(sigmas[s])
            e["snr"] = float(e["amp_peak"] / (sigmas[s] + 1e-12))
            e["alpha"] = float(actual_alpha)
            e["eb_scale"] = float(eb_scales[s])
            e["amp_dff"] = float(e["amp_peak"])
            e["amp_df"] = (
                float(e["amp_peak"] * f0_med_s)
                if np.isfinite(f0_med_s) else np.nan
            )
        all_events.extend(evs)

    return all_events, rates_hz_final, a_hat, recon, sigmas


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------


def build_events_dataframe(
    events: list[dict[str, Any]], dmd: int
) -> pl.DataFrame:
    """Convert event dicts to a polars DataFrame."""
    if not events:
        return pl.DataFrame(
            schema={
                "dmd": pl.Int32,
                "source_id": pl.Int32,
                "peak_idx": pl.Int64,
                "time": pl.Float64,
                "amp_peak": pl.Float64,
                "amp_sum": pl.Float64,
                "width_s": pl.Float64,
                "snr": pl.Float64,
                "amp_dff": pl.Float64,
                "amp_df": pl.Float64,
                "sigma": pl.Float64,
                "alpha": pl.Float64,
                "eb_scale": pl.Float64,
            }
        )

    df = pl.DataFrame(events)
    df = df.with_columns(
        pl.lit(dmd).cast(pl.Int32).alias("dmd"),
        pl.col("source_id").cast(pl.Int32),
        pl.col("peak_idx").cast(pl.Int64),
    )
    cols = [
        "dmd", "source_id", "peak_idx", "time",
        "amp_peak", "amp_sum", "width_s",
        "snr", "amp_dff", "amp_df",
        "sigma", "alpha", "eb_scale",
    ]
    existing = [c for c in cols if c in df.columns]
    return df.select(existing).sort("time")


def build_synapse_summary(
    events_df: pl.DataFrame,
    n_synapses: int,
    duration_s: float,
    sigmas: np.ndarray,
    f0_median: np.ndarray,
    dmd: int,
) -> pl.DataFrame:
    """Build per-synapse summary statistics."""
    base = pl.DataFrame({
        "dmd": pl.Series([dmd] * n_synapses, dtype=pl.Int32),
        "source_id": pl.Series(
            np.arange(n_synapses), dtype=pl.Int32
        ),
        "sigma": pl.Series(sigmas, dtype=pl.Float64),
        "f0_median": pl.Series(f0_median, dtype=pl.Float64),
    })

    if events_df.height == 0:
        return base.with_columns(
            pl.lit(0).cast(pl.Int64).alias("n_events"),
            pl.lit(0.0).alias("rate_hz"),
            pl.lit(None).cast(pl.Float64).alias("mean_amp_peak"),
            pl.lit(None).cast(pl.Float64).alias("mean_snr"),
            pl.lit(None).cast(pl.Float64).alias("mean_amp_dff"),
        )

    agg_cols = [
        pl.len().cast(pl.Int64).alias("n_events"),
        pl.col("amp_peak").mean().alias("mean_amp_peak"),
        pl.col("snr").mean().alias("mean_snr"),
        pl.col("amp_dff").mean().alias("mean_amp_dff"),
    ]
    stats = events_df.group_by("source_id").agg(agg_cols)
    result = base.join(stats, on="source_id", how="left")
    result = result.with_columns(
        pl.col("n_events").fill_null(0),
        (
            pl.col("n_events").fill_null(0).cast(pl.Float64)
            / max(duration_s, 1e-12)
        ).alias("rate_hz"),
    )
    return result.sort("source_id")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def _output_dir(subject: str, exp: str, loc: str, acq: str) -> str:
    return os.path.join(
        DEFS.anmat_root, subject, exp,
        "activity_data", loc, acq, OUTPUT_DIR_NAME,
    )


def detect_and_save(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    channel: int = 0,
    amp_thresh_sigma: float = DEFAULT_AMP_THRESH_SIGMA,
    min_peak_distance_s: float = DEFAULT_MIN_PEAK_DISTANCE_S,
    second_pass: bool = DEFAULT_EB_SECOND_PASS,
    rate_shrink_beta: float = DEFAULT_EB_RATE_SHRINK_BETA,
    alpha_scale_bounds: tuple[float, float] = DEFAULT_EB_ALPHA_SCALE_BOUNDS,
    tau_s: float = IGLUSNFR_TAU_S,
    lam_mult: float = FISTA_LAM_MULT,
) -> None:
    """Run glutamate event detection and save results."""
    out_dir = _output_dir(subject, exp, loc, acq)
    wis.util.check_dir(out_dir)

    trace_data = load_traces_from_scopex(
        subject, exp, loc, acq, channel=channel
    )

    params = {
        "channel": channel,
        "amp_thresh_sigma": amp_thresh_sigma,
        "min_peak_distance_s": min_peak_distance_s,
        "second_pass": second_pass,
        "rate_shrink_beta": rate_shrink_beta,
        "alpha_scale_bounds": list(alpha_scale_bounds),
        "tau_s": tau_s,
        "lam_mult": lam_mult,
    }

    for dmd_key, data in trace_data.items():
        dmd_num = int(dmd_key.split("_")[1])
        ls = data["ls"]
        f0_ts = data["f0_ts"]
        f0_median = data["f0_median"]
        fs = data["fs"]
        t = data["time"]
        n_syn = ls.shape[0]
        duration_s = float(t[-1] - t[0]) + 1.0 / fs

        events, rates_hz, _, _, sigmas = detect_events(
            ls, f0_ts, f0_median, fs, t,
            tau_s=tau_s, lam_mult=lam_mult,
            amp_thresh_sigma=amp_thresh_sigma,
            merge_gap_s=min_peak_distance_s,
            second_pass=second_pass,
            rate_shrink_beta=rate_shrink_beta,
            alpha_scale_bounds=alpha_scale_bounds,
        )

        ev_df = build_events_dataframe(events, dmd=dmd_num)
        syn_df = build_synapse_summary(
            ev_df, n_syn, duration_s, sigmas, f0_median,
            dmd=dmd_num,
        )

        ev_df.write_parquet(
            os.path.join(out_dir, f"events_dmd{dmd_num}.parquet")
        )
        syn_df.write_parquet(
            os.path.join(
                out_dir, f"synapse_summary_dmd{dmd_num}.parquet"
            )
        )

        n_events = ev_df.height
        mean_rate = (
            float(rates_hz.mean()) if rates_hz.size > 0 else 0.0
        )
        print(
            f"  DMD {dmd_num}: {n_events} events across {n_syn} "
            f"synapses (mean rate {mean_rate:.2f} Hz)"
        )

    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
