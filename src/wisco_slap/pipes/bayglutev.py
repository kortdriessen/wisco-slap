"""
Hierarchical (empirical Bayes) event detection for iGluSnFR nonneg dF/F traces.

Conceptual model (per synapse s):
    y_s(t) ≈ (a_s * h)(t) + noise
where
    - a_s(t) >= 0 is a sparse latent release-event amplitude train
    - h(t) is a causal indicator kernel (here: single exponential decay)
    - noise is handled via robust per-synapse sigma estimates and an L1 sparsity prior

Inference:
    - Per synapse, solve a nonnegative L1-regularized least squares (MAP under Laplace prior):
          min_{a>=0} 0.5 || y - H a ||^2 + alpha * sum(a)
      using FISTA with fast FFT-based convolutions.
    - “Hierarchical” part: do a 2-pass empirical Bayes update:
        Pass 1 estimates per-synapse event rates; Pass 2 adjusts alpha per synapse to
        partially pool rates across synapses (quiet synapses get stronger sparsity; very active
        synapses get weaker sparsity), then re-solves.

Outputs:
    - events: list of dicts (or pandas DataFrame if pandas is installed)
    - per_synapse: dict of per-synapse parameters and summary stats
    - a_hat: inferred event trains (n_synapses, n_timepoints)
    - recon: reconstructed traces (n_synapses, n_timepoints)

You provide:
    y: np.ndarray shape (n_synapses, n_timepoints)  # dF/F of nonneg trace
    fs: sampling rate (Hz)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy import signal
except Exception as e:
    raise ImportError(
        "This pipeline requires scipy (scipy.signal). Please `pip install scipy`."
    ) from e
import os

import polars as pl

import wisco_slap as wis
import wisco_slap.defs as DEFS

# -----------------------------
# Utilities
# -----------------------------


def _next_pow_two(n: int) -> int:
    return 1 << (int(n - 1).bit_length())


def _robust_sigma_from_diff(x: np.ndarray) -> float:
    """
    Robust sigma estimate using MAD of first differences.
    Works well when there is slow drift / baseline wander.
    """
    d = np.diff(x)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    # 0.674489... is median(|N(0,1)|)
    return float(mad / 0.6744897501960817 / np.sqrt(2.0) + 1e-12)


def _butter_lowpass_baseline(
    y: np.ndarray, fs: float, cutoff_hz: float, order: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Low-pass estimate of baseline via zero-phase Butterworth (sosfiltfilt).
    Returns (y_highpassed, baseline).
    """
    if cutoff_hz is None or cutoff_hz <= 0:
        baseline = np.zeros_like(y)
        return y.astype(np.float64, copy=True), baseline

    nyq = fs / 2.0
    w = float(cutoff_hz / nyq)
    w = min(max(w, 1e-6), 0.99)
    sos = signal.butter(order, w, btype="lowpass", output="sos")
    baseline = signal.sosfiltfilt(sos, y.astype(np.float64, copy=False))
    return (y.astype(np.float64, copy=False) - baseline), baseline


def _fill_nans_linear(x: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation; edge NaNs are forward/backfilled."""
    if not np.isnan(x).any():
        return x
    x = x.copy()
    n = x.size
    idx = np.arange(n)
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def _build_exp_kernel(fs: float, tau_s: float, eps: float = 1e-3) -> np.ndarray:
    """
    Causal single-exponential decay kernel h(t)=exp(-t/tau), normalized so h(0)=1.
    Truncated when exp(-t/tau) < eps.
    """
    tau_s = float(tau_s)
    if tau_s <= 0:
        raise ValueError("tau_s must be > 0")

    L = int(np.ceil((-tau_s * np.log(eps)) * fs)) + 1
    t = np.arange(L, dtype=np.float64) / fs
    k = np.exp(-t / tau_s)
    k /= k[0]  # peak=1 at t=0
    return k


def _estimate_tau_decay_from_data(
    Y_hp: np.ndarray,
    fs: float,
    sigmas: np.ndarray,
    max_synapses: int = 60,
    max_peaks_total: int = 2000,
    prominence_sigma: float = 6.0,
    min_peak_distance_s: float = 0.2,
    fit_tail_start_s: float = 0.01,
    fit_tail_end_s: float = 0.25,
    min_tail_frac: float = 0.15,
    default_tau_s: float = 0.10,
) -> float:
    """
    Roughly estimate a global tau_decay by averaging normalized isolated events
    and fitting an exponential to the post-peak tail in log space.

    If insufficient peaks are found, returns default_tau_s.
    """
    S, T = Y_hp.shape
    use_S = min(S, max_synapses)
    min_dist = max(1, int(round(min_peak_distance_s * fs)))

    snippets: list[np.ndarray] = []
    peaks_used = 0

    tail_end = int(round(fit_tail_end_s * fs))
    if tail_end < 5:
        return default_tau_s

    for s in range(use_S):
        y = Y_hp[s]
        sigma = float(sigmas[s])
        if not np.isfinite(sigma) or sigma <= 0:
            continue

        # Find prominent peaks in high-passed trace
        thr = np.median(y) + prominence_sigma * sigma
        peaks, props = signal.find_peaks(y, height=thr, distance=min_dist)
        if peaks.size == 0:
            continue

        # Take up to a fixed number per synapse to avoid domination by one synapse
        per_syn = max(10, max_peaks_total // max(1, use_S))
        peaks = peaks[:per_syn]

        for p in peaks:
            if p < 1:
                continue
            if p + tail_end >= T:
                continue
            peak_val = y[p]
            if not np.isfinite(peak_val) or peak_val <= 0:
                continue

            snip = y[p : p + tail_end].copy()
            snip /= peak_val  # normalize peak to 1
            # enforce nonneg in snippet for log-fit robustness
            snip[snip < 0] = 0.0
            snippets.append(snip)
            peaks_used += 1
            if peaks_used >= max_peaks_total:
                break
        if peaks_used >= max_peaks_total:
            break

    if len(snippets) < 50:
        return default_tau_s

    avg = np.mean(np.stack(snippets, axis=0), axis=0)

    # Fit tail: log(avg) ~ -t/tau + c on region where avg is sufficiently above floor
    t = np.arange(avg.size) / fs
    start = int(round(fit_tail_start_s * fs))
    end = avg.size

    tail = avg[start:end]
    tt = t[start:end]

    # select region above min_tail_frac of peak (peak=1)
    keep = tail > float(min_tail_frac)
    if keep.sum() < 10:
        return default_tau_s

    ylog = np.log(tail[keep] + 1e-12)
    tfit = tt[keep]

    # linear fit ylog = b + m*t ; tau = -1/m
    m, b = np.polyfit(tfit, ylog, deg=1)
    if m >= 0:
        return default_tau_s

    tau = float(-1.0 / m)
    # clamp to sane range
    tau = float(np.clip(tau, 0.01, 1.0))
    return tau


# -----------------------------
# Core solver: FISTA for nonneg L1 with convolution operator
# -----------------------------


def _fista_nonneg_l1_conv(
    y: np.ndarray,
    k: np.ndarray,
    alpha: float,
    max_iter: int = 80,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve: min_{a>=0} 0.5|| (a*k)[:T] - y ||^2 + alpha * sum(a)
    where k is causal kernel, a is length T, and (a*k) is full linear convolution.

    Uses FFT-based convolutions for speed.
    Returns (a_hat, recon), where recon is (a*k)[:T].
    """
    y = y.astype(np.float64, copy=False)
    k = k.astype(np.float64, copy=False)

    T = y.size
    Lk = k.size
    full_len = T + Lk - 1
    nfft = _next_pow_two(full_len)

    fft_k = np.fft.rfft(k, nfft)
    fft_k_rev = np.fft.rfft(k[::-1], nfft)

    max_power = float(np.max(np.abs(fft_k) ** 2))
    lips = max_power + 1e-12
    step = 1.0 / lips
    thr = float(alpha) * step

    a = np.zeros(T, dtype=np.float64)
    z = a.copy()
    t = 1.0

    a_pad = np.zeros(full_len, dtype=np.float64)
    r_full = np.zeros(full_len, dtype=np.float64)

    def conv_full(x_full: np.ndarray, fft_ker: np.ndarray) -> np.ndarray:
        X = np.fft.rfft(x_full, nfft)
        c = np.fft.irfft(X * fft_ker, nfft)
        return c[:full_len]

    prev_obj = np.inf

    for it in range(max_iter):
        # forward
        a_pad[:T] = z
        a_pad[T:] = 0.0
        pred_full = conv_full(a_pad, fft_k)
        pred = pred_full[:T]
        resid = pred - y

        # gradient: H^T resid
        r_full[:T] = resid
        r_full[T:] = 0.0
        grad_full = conv_full(r_full, fft_k_rev)
        grad = grad_full[Lk - 1 : Lk - 1 + T]

        # proximal gradient + nonneg soft-threshold
        x = z - step * grad
        a_new = x - thr
        a_new[a_new < 0.0] = 0.0

        # FISTA momentum
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = a_new + ((t - 1.0) / t_new) * (a_new - a)
        a = a_new
        t = t_new

        # convergence check
        if it % 5 == 0 or it == max_iter - 1:
            a_pad[:T] = a
            a_pad[T:] = 0.0
            pred = conv_full(a_pad, fft_k)[:T]
            resid = pred - y
            obj = 0.5 * float(np.dot(resid, resid)) + float(alpha) * float(a.sum())
            if np.isfinite(prev_obj):
                rel = abs(prev_obj - obj) / (prev_obj + 1e-12)
                if rel < tol:
                    break
            prev_obj = obj

    a_pad[:T] = a
    a_pad[T:] = 0.0
    recon = conv_full(a_pad, fft_k)[:T]
    return a, recon


# -----------------------------
# Event extraction from a_hat
# -----------------------------


def _extract_events_from_a(
    a: np.ndarray,
    fs: float,
    amp_min: float,
    merge_gap_s: float = 0.01,
) -> list[dict[str, Any]]:
    """
    Convert sample-wise latent amplitudes a[t] into a list of discrete events by:
      - thresholding a[t] > amp_min
      - grouping contiguous (or near-contiguous) samples
      - reporting peak time and amplitude

    Returns list of dicts with event fields (no synapse id; caller adds it).
    """
    a = a.astype(np.float64, copy=False)
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
        amp_peak = float(a[peak_i])
        amp_sum = float(a[seg].sum())
        events.append(
            dict(
                start_idx=int(seg_start),
                end_idx=int(seg_end),
                peak_idx=int(peak_i),
                t_sec=float(peak_i / fs),
                amp_peak=amp_peak,
                amp_sum=amp_sum,
                width_s=float((seg_end - seg_start + 1) / fs),
            )
        )

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


# -----------------------------
# Main pipeline
# -----------------------------


def detect_glutamate_release_events(
    Y: np.ndarray,
    fs: float,
    # Baseline / noise
    baseline_cutoff_hz: float = 0.05,
    baseline_filter_order: int = 2,
    # Kernel
    tau_decay_s: float | None = None,
    kernel_eps: float = 1e-3,
    # Sparsity / inference
    lam_mult: float = 1.0,
    max_iter: int = 80,
    tol: float = 1e-4,
    # Event extraction thresholds
    amp_thresh_sigma: float = 2.0,
    merge_gap_s: float = 0.01,
    # Hierarchical (2-pass EB) refinement
    second_pass: bool = True,
    rate_shrink_beta: float = 0.6,
    alpha_scale_bounds: tuple[float, float] = (0.5, 2.0),
    # Tau estimation settings (if tau_decay_s=None)
    tau_est_max_synapses: int = 60,
    tau_est_max_peaks_total: int = 2000,
    tau_est_prominence_sigma: float = 6.0,
    tau_est_min_peak_distance_s: float = 0.2,
    tau_est_default_tau_s: float = 0.10,
    # Output controls
    return_recon: bool = True,
) -> dict[str, Any]:
    """
    Parameters
    ----------
    Y : np.ndarray
        dF/F nonneg trace, shape (n_synapses, n_timepoints)
    fs : float
        sampling rate in Hz

    Returns
    -------
    result : dict with keys
        - "events": list of event dicts (or pandas DataFrame if pandas installed)
        - "per_synapse": dict of arrays (sigma, alpha, rate_hz, etc.)
        - "a_hat": inferred latent event trains, shape (S, T)
        - "recon": reconstruction (optional), shape (S, T)
        - "kernel": kernel used (np.ndarray)
        - "tau_decay_s": tau used (float)
    """
    if not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array")
    if Y.ndim != 2:
        raise ValueError(f"Y must have shape (n_synapses, n_timepoints), got {Y.shape}")
    if fs <= 0:
        raise ValueError("fs must be > 0")

    Y = Y.astype(np.float64, copy=False)
    S, T = Y.shape

    # Fill NaNs per synapse
    if np.isnan(Y).any():
        Y = np.stack([_fill_nans_linear(Y[s]) for s in range(S)], axis=0)

    # Baseline removal + noise estimation
    Y_hp = np.empty_like(Y)
    baselines = np.empty_like(Y)
    sigmas = np.empty(S, dtype=np.float64)

    for s in range(S):
        y_hp, b = _butter_lowpass_baseline(
            Y[s], fs, baseline_cutoff_hz, order=baseline_filter_order
        )
        Y_hp[s] = y_hp
        baselines[s] = b
        sigmas[s] = _robust_sigma_from_diff(y_hp)

    # Choose / estimate tau_decay
    if tau_decay_s is None:
        tau_decay_s = _estimate_tau_decay_from_data(
            Y_hp=Y_hp,
            fs=fs,
            sigmas=sigmas,
            max_synapses=tau_est_max_synapses,
            max_peaks_total=tau_est_max_peaks_total,
            prominence_sigma=tau_est_prominence_sigma,
            min_peak_distance_s=tau_est_min_peak_distance_s,
            default_tau_s=tau_est_default_tau_s,
        )

    kernel = _build_exp_kernel(fs=fs, tau_s=float(tau_decay_s), eps=kernel_eps)

    # Helper: compute alpha per synapse from sigma and length
    # (alpha is in dF/F units; higher alpha => sparser events)
    def alpha_from_sigma(sigma: float, scale: float = 1.0) -> float:
        # universal-threshold-like scaling; tuned by lam_mult and later EB scaling
        return float(scale * lam_mult * sigma * np.sqrt(2.0 * np.log(max(10, T))))

    # Pass 1: per-synapse deconvolution
    a_hat = np.zeros((S, T), dtype=np.float64)
    recon = np.zeros((S, T), dtype=np.float64) if return_recon else None
    alpha_pass1 = np.empty(S, dtype=np.float64)

    for s in range(S):
        alpha_s = alpha_from_sigma(sigmas[s], scale=1.0)
        alpha_pass1[s] = alpha_s
        a_s, recon_s = _fista_nonneg_l1_conv(
            Y_hp[s], kernel, alpha_s, max_iter=max_iter, tol=tol
        )
        a_hat[s] = a_s
        if return_recon:
            recon[s] = recon_s

    # Extract events (pass 1) to estimate rates
    duration_s = T / fs
    rates_hz = np.zeros(S, dtype=np.float64)
    events_pass1: list[dict[str, Any]] = []

    for s in range(S):
        amp_min = float(amp_thresh_sigma * sigmas[s])
        evs = _extract_events_from_a(
            a_hat[s], fs=fs, amp_min=amp_min, merge_gap_s=merge_gap_s
        )
        rates_hz[s] = len(evs) / max(duration_s, 1e-12)
        for e in evs:
            e2 = dict(e)
            e2["synapse"] = int(s)
            e2["sigma"] = float(sigmas[s])
            e2["snr_peak"] = float(e2["amp_peak"] / (sigmas[s] + 1e-12))
            e2["alpha"] = float(alpha_pass1[s])
            events_pass1.append(e2)

    # Pass 2 (hierarchical EB): adjust alpha per synapse based on pooled rate
    # Idea: shrink rates toward population center; use that to scale alpha.
    alpha_pass2 = alpha_pass1.copy()
    if second_pass:
        eps = 1e-6
        # Use robust population rate center (geometric mean) to avoid domination by outliers
        pop_rate = float(np.exp(np.mean(np.log(rates_hz + eps))))
        lo, hi = alpha_scale_bounds

        # alpha scale: quiet synapses => larger alpha; hyper-active => smaller alpha
        # scale_s = (pop_rate / (rate_s+eps))^beta ; clamped
        scales = (pop_rate / (rates_hz + eps)) ** float(rate_shrink_beta)
        scales = np.clip(scales, lo, hi)

        # Re-solve with adjusted alpha
        for s in range(S):
            alpha_s = alpha_from_sigma(sigmas[s], scale=float(scales[s]))
            alpha_pass2[s] = alpha_s
            a_s, recon_s = _fista_nonneg_l1_conv(
                Y_hp[s], kernel, alpha_s, max_iter=max_iter, tol=tol
            )
            a_hat[s] = a_s
            if return_recon:
                recon[s] = recon_s

        # Re-extract events after pass 2
        events: list[dict[str, Any]] = []
        rates_hz2 = np.zeros(S, dtype=np.float64)

        for s in range(S):
            amp_min = float(amp_thresh_sigma * sigmas[s])
            evs = _extract_events_from_a(
                a_hat[s], fs=fs, amp_min=amp_min, merge_gap_s=merge_gap_s
            )
            rates_hz2[s] = len(evs) / max(duration_s, 1e-12)
            for e in evs:
                e2 = dict(e)
                e2["synapse"] = int(s)
                e2["sigma"] = float(sigmas[s])
                e2["snr_peak"] = float(e2["amp_peak"] / (sigmas[s] + 1e-12))
                e2["alpha"] = float(alpha_pass2[s])
                events.append(e2)

        rates_hz = rates_hz2
    else:
        events = events_pass1

    # Optional: return as DataFrame if available
    try:
        import pandas as pd  # type: ignore

        events_out: list[dict[str, Any]] | pd.DataFrame = pd.DataFrame(events)
    except Exception:
        events_out = events

    per_synapse = dict(
        sigma=sigmas,
        alpha=alpha_pass2 if second_pass else alpha_pass1,
        rate_hz=rates_hz,
        baseline_cutoff_hz=float(baseline_cutoff_hz),
    )

    return dict(
        events=events_out,
        per_synapse=per_synapse,
        a_hat=a_hat,
        recon=recon,
        kernel=kernel,
        tau_decay_s=float(tau_decay_s),
        fs=float(fs),
    )


def bayesian_glut_event_detection(nonneg, matchFilt, dmd, tau_decay_s=0.05):
    """Bayesian event detection for glutamate events"""
    mf = matchFilt.filter(pl.col("dmd") == dmd)
    mf = mf.with_columns((pl.col("data") / pl.col("noise")).alias("snr"))
    nn = nonneg.filter(pl.col("dmd") == dmd)
    times = nn.filter(pl.col("source-ID") == 0)["time"][:1000].to_numpy()
    fs = int(1 / np.diff(times).mean())
    if fs != 200:
        raise ValueError("fs is not 200")
    sources = nn["source-ID"].unique().to_numpy()
    sources = np.sort(sources)
    source_nonnegs = []
    for source in sources:
        source_array = nn.filter(pl.col("source-ID") == source)["data"].to_numpy()
        source_nonnegs.append(source_array)
    source_data = np.stack(source_nonnegs)
    print("Running event detection...")
    detres = detect_glutamate_release_events(
        source_data, fs=200.0, tau_decay_s=tau_decay_s
    )

    g = pl.from_pandas(detres["events"])
    g = g.sort("t_sec")
    events_seconds = g["t_sec"].to_numpy()
    event_sources = g["synapse"].to_numpy()

    # Now for each detected event, we add the context of the matchFilt SNR, averaging
    # the SNR over 3 samples centered on the event.

    # 1. Create a temporary DataFrame from the event arrays with an index to preserve order
    qdf = pl.DataFrame(
        {"time": events_seconds, "source-ID": event_sources}
    ).with_row_index("idx")
    # 2. Prepare target: Select columns and sort by source-ID then time
    # This sort is crucial to ensure 'rolling' uses the temporal neighbors correctly.
    target = mf.select(["time", "source-ID", "snr"]).sort(["source-ID", "time"])
    # Calculate 3-sample centered rolling mean of SNR per source
    # min_periods=1 ensures we calculate a mean even if we are at the start/end (nanmean behavior)
    target = target.with_columns(
        pl.col("snr")
        .rolling_mean(window_size=3, center=True, min_periods=1)
        .over("source-ID")
        .alias("snr_mean")
    )
    # Sort by time as required for join_asof
    target = target.sort("time")
    # 3. Perform join_asof with tolerance using the newly calculated averaged SNR
    res = qdf.join_asof(
        target, on="time", by="source-ID", tolerance=1e-5, strategy="nearest"
    )
    # 4. Sort by the index and extract the 'snr_mean' array
    event_snrs = res.sort("idx")["snr_mean"].to_numpy()
    g = g.with_columns(pl.lit(event_snrs).alias("mfsnr"))
    g = g.drop_nans(subset=["mfsnr"])
    g = g.with_columns(pl.lit(dmd).alias("dmd"))
    return g


def save_glut_detections(subject, exp, loc, acq, overwrite=False):
    root_dir = os.path.join(
        DEFS.anmat_root,
        subject,
        exp,
        "activity_data",
        loc,
        acq,
        "glutamate_event_detection",
        "bayes_hm",
    )
    wis.util.gen.check_dir(root_dir)
    dmd1_path = os.path.join(root_dir, "dmd1.parquet")
    dmd2_path = os.path.join(root_dir, "dmd2.parquet")
    if os.path.exists(dmd1_path) and os.path.exists(dmd2_path):
        if not overwrite:
            return
        else:
            os.remove(dmd1_path)
            os.remove(dmd2_path)
    mfilt = wis.scope.io.load_syndf(
        subject, exp, loc, acq, trace_types=["matchFilt"], trace_group="dFF"
    )
    nonneg = wis.scope.io.load_syndf(
        subject, exp, loc, acq, trace_types=["nonneg"], trace_group="dFF"
    )
    for dmd, dpath in zip([1, 2], [dmd1_path, dmd2_path], strict=False):
        if os.path.exists(dpath) and not overwrite:
            continue
        print(f"Processing dmd={dmd}...")
        dets = bayesian_glut_event_detection(nonneg, mfilt, dmd, tau_decay_s=0.05)
        dets.write_parquet(dpath)
    return
