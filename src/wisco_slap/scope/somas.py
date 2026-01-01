import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import fftconvolve, find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import polars as pl


@dataclass
class Event:
    onset_time: float
    peak_time: float
    amp_dff: float  # amplitude (ΔF/F) via local kernel projection
    snr: float  # matched-filter SNR at peak (aligned)
    width_s: float  # half-prominence width (s)
    idx_peak: int  # index in the *original* timestamps array


@dataclass
class DetectionResult:
    events: np.ndarray
    dff: np.ndarray  # length == len(timestamps)
    baseline: np.ndarray  # length == len(timestamps)
    kernel_time: np.ndarray
    kernel: np.ndarray
    mf_output: np.ndarray  # length == len(timestamps), aligned to ΔF/F peak time
    params: Dict[str, Any]


# ------------------ utilities ------------------


def _asls_baseline(
    y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10
) -> np.ndarray:
    """Asymmetric least-squares baseline (Eilers & Boelens)."""
    y = np.asarray(y)
    n = y.size
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    C = lam * (D.T @ D)
    w = np.ones(n)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(n, n))
        z = spsolve(W + C, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def _bi_exp(t, A, tau_r, tau_d):
    t = np.maximum(t, 0.0)
    return A * (1.0 - np.exp(-t / max(tau_r, 1e-6))) * np.exp(-t / max(tau_d, 1e-6))


def _mad(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))


def _resample_to_uniform(
    t: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    If sampling is irregular, resample to uniform. Return (t_uni, y_uni, fs, valid_mask_original).
    valid_mask_original marks non-NaN input samples and is used to restore NaNs later.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    valid = np.isfinite(t) & np.isfinite(y)
    t0, y0 = t[valid], y[valid]

    dt = np.diff(t0)
    med_dt = np.median(dt)
    irregular = np.nanmax(np.abs(dt - med_dt) / med_dt) > 0.02

    if irregular:
        # uniform grid covering the same span
        n_steps = int(np.round((t0[-1] - t0[0]) / med_dt))
        t_uni = t0[0] + np.arange(n_steps + 1) * med_dt
        f = interp1d(
            t0, y0, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        y_uni = f(t_uni)
    else:
        t_uni, y_uni = t0, y0

    fs = 1.0 / np.median(np.diff(t_uni))
    return t_uni, y_uni, fs, valid


def _estimate_kernel(
    dff: np.ndarray,
    fs: float,
    peak_z: float,
    pre_ms: float,
    post_ms: float,
    fallback_taus: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Estimate a bi-exponential kernel from strong events; else fallback."""
    z = (dff - np.nanmedian(dff)) / (_mad(dff) + 1e-9)
    min_dist = int(max(1, 0.15 * fs))
    peaks, _ = find_peaks(z, height=peak_z, distance=min_dist)

    def _fallback():
        tau_r, tau_d = fallback_taus
        tker = np.arange(0, int(post_ms / 1000 * fs) + 1) / fs
        k = _bi_exp(tker, 1.0, tau_r, tau_d)
        k /= np.linalg.norm(k) + 1e-12
        return tker, k, {"tau_r": tau_r, "tau_d": tau_d, "from": "fallback"}

    if peaks.size < 5:
        return _fallback()

    pre = int(pre_ms / 1000 * fs)
    post = int(post_ms / 1000 * fs)
    snips = []
    for p in peaks:
        a, b = p - pre, p + post
        if a < 0 or b > len(dff):
            continue
        seg = dff[a:b].copy()
        seg -= np.nanmedian(seg[: max(1, int(0.02 * fs))])
        if np.any(~np.isfinite(seg)):
            continue
        snips.append(seg)
    if len(snips) < 5:
        return _fallback()

    S = np.vstack(snips)
    m = np.median(S, axis=0)
    peak_idx = np.argmax(m)
    t = (np.arange(len(m)) - peak_idx) / fs
    mask = t >= 0
    t_fit = t[mask]
    y_fit = m[mask]
    try:
        A0 = np.nanmax(y_fit)
        tr0, td0 = 0.05, 0.4
        popt, _ = curve_fit(
            _bi_exp,
            t_fit,
            y_fit,
            p0=[A0, tr0, td0],
            bounds=([0, 1e-3, 0.05], [np.inf, 0.2, 2.5]),
        )
        _, tau_r, tau_d = popt
    except Exception:
        return _fallback()

    tker = np.arange(0, int(post_ms / 1000 * fs) + 1) / fs
    k = _bi_exp(tker, 1.0, tau_r, tau_d)
    k /= np.linalg.norm(k) + 1e-12
    return tker, k, {"tau_r": float(tau_r), "tau_d": float(tau_d), "from": "fit"}


# ------------------ main API ------------------


def detect_calcium_events(
    trace: np.ndarray,
    timestamps: np.ndarray,
    *,
    baseline_method: str = "asls",
    asls_lam: float = 2e6,
    asls_p: float = 0.01,
    z_for_kernel: float = 4.0,
    kernel_pre_ms: float = 50.0,
    kernel_post_ms: float = 1000.0,
    fallback_kernel_taus: Tuple[float, float] = (0.06, 0.45),
    threshold: Optional[float] = 4.0,
    fdr_alpha: Optional[float] = None,
    min_event_separation_s: float = 0.25,
) -> DetectionResult:
    """
    Returns arrays exactly aligned to the input `timestamps` and peak times indexed on that grid.
    """
    t_in = np.asarray(timestamps)
    y_in = np.asarray(trace)
    assert (
        t_in.ndim == 1 and y_in.ndim == 1 and len(t_in) == len(y_in)
    ), "Lengths must match."

    # resample to a uniform grid (internal only)
    t_uni, y_uni, fs, valid_mask = _resample_to_uniform(t_in, y_in)

    # baseline
    if baseline_method != "asls":
        raise ValueError("Only 'asls' baseline is implemented.")
    base_uni = _asls_baseline(y_uni, lam=asls_lam, p=asls_p, niter=10)

    # ΔF/F
    eps = np.percentile(base_uni, 1) * 1e-6 + 1e-12
    dff_uni = (y_uni - base_uni) / (np.maximum(base_uni, eps))

    # kernel
    tker, k, k_info = _estimate_kernel(
        dff_uni,
        fs,
        peak_z=z_for_kernel,
        pre_ms=kernel_pre_ms,
        post_ms=kernel_post_ms,
        fallback_taus=fallback_kernel_taus,
    )
    M = len(k)
    k_peak_idx = int(np.argmax(k))
    # full correlation (matched filter) and explicit lag compensation to align to ΔF/F peak time
    mf_full = fftconvolve(dff_uni, k[::-1], mode="full")
    sigma_mf = _mad(mf_full) + 1e-12
    z_full = mf_full / sigma_mf

    # Align so that index n in the aligned array corresponds to the ΔF/F peak time at n.
    lag = (M - 1) - k_peak_idx  # samples
    # slice of length == len(dff_uni)
    z_aligned_uni = z_full[lag : lag + len(dff_uni)]

    # ----- detection on the uniform, aligned SNR -----
    if fdr_alpha is not None:
        from scipy.stats import norm

        pvals = 1.0 - norm.cdf(z_aligned_uni)
        order = np.argsort(pvals)
        m = len(pvals)
        thr_idx = np.nonzero(pvals[order] <= fdr_alpha * (np.arange(1, m + 1) / m))[0]
        thr = np.inf if len(thr_idx) == 0 else z_aligned_uni[order[thr_idx.max()]]
    else:
        thr = float(threshold) if threshold is not None else 4.0

    min_dist = max(1, int(min_event_separation_s * fs))
    pk_uni, props = find_peaks(z_aligned_uni, height=thr, distance=min_dist)

    # refine peak on ΔF/F near each MF peak
    half_w = max(1, int(0.2 * fs))  # +/- 200 ms search
    refined_pk_uni = []
    for p in pk_uni:
        a = max(0, p - half_w)
        b = min(len(dff_uni), p + half_w + 1)
        local = dff_uni[a:b]
        rp = a + int(np.argmax(local))
        refined_pk_uni.append(rp)
    refined_pk_uni = np.array(sorted(set(refined_pk_uni)))

    # widths from matched-filter SNR (half-prominence)
    if refined_pk_uni.size > 0:
        w_samples, _, _, _ = peak_widths(z_aligned_uni, refined_pk_uni, rel_height=0.5)
    else:
        w_samples = np.array([])

    # amplitudes via local kernel projection (non-negative least squares approx)
    events = []
    for i, pk in enumerate(refined_pk_uni):
        # project kernel starting at pk - k_peak_idx so that kernel peak aligns with signal peak
        start = max(0, pk - k_peak_idx)
        end = min(len(dff_uni), start + M)
        kk = k[: end - start]
        seg = dff_uni[start:end]
        A = max(0.0, float(np.dot(seg, kk) / (np.dot(kk, kk) + 1e-12)))
        width_s = float(w_samples[i] / fs) if i < len(w_samples) else float(M / fs / 3)

        # times on uniform grid
        pk_time = t_uni[pk]
        # map peak time to exact index on the original timestamps
        idx_orig = int(np.clip(np.searchsorted(t_in, pk_time), 0, len(t_in) - 1))
        # adjust to the nearest neighbor (not just ceil)
        if 0 < idx_orig < len(t_in):
            if abs(t_in[idx_orig] - pk_time) > abs(t_in[idx_orig - 1] - pk_time):
                idx_orig -= 1

        events.append(
            (
                float(
                    t_in[idx_orig]
                ),  # onset_time placeholder == peak_time (could add 10%-rise if desired)
                float(t_in[idx_orig]),
                A,
                float(z_aligned_uni[min(pk, len(z_aligned_uni) - 1)]),
                width_s,
                int(idx_orig),
            )
        )

    ev_struct = np.array(
        events,
        dtype=[
            ("onset_time", "f8"),
            ("peak_time", "f8"),
            ("amp_dff", "f8"),
            ("snr", "f8"),
            ("width_s", "f8"),
            ("idx_peak", "i8"),
        ],
    )

    # ----- resample outputs back to ORIGINAL timestamps (exact length) -----
    # Build interpolators on the uniform grid
    to_orig = lambda arr: interp1d(
        t_uni, arr, kind="linear", bounds_error=False, fill_value="extrapolate"
    )(t_in)

    baseline = to_orig(base_uni)
    dff = to_orig(dff_uni)
    mf_output = to_orig(z_aligned_uni)

    # preserve NaNs where original trace had NaNs
    baseline[~valid_mask] = np.nan
    dff[~valid_mask] = np.nan
    mf_output[~valid_mask] = np.nan

    return DetectionResult(
        events=ev_struct,
        dff=dff,
        baseline=baseline,
        kernel_time=tker,
        kernel=k,
        mf_output=mf_output,
        params=dict(
            fs=fs,
            threshold=thr,
            fdr_alpha=fdr_alpha,
            baseline_method=baseline_method,
            asls_lam=asls_lam,
            asls_p=asls_p,
            kernel_info=k_info,
            kernel_peak_index=k_peak_idx,
            alignment_lag_samples=lag,
            min_event_separation_s=min_event_separation_s,
        ),
    )


def detect_CaEvents(roidf):
    evdfs = []
    dffs = []
    for soma_id in roidf["soma-ID"].unique():
        print(f"Detecting CaEvents for soma {soma_id}")
        sdf = roidf.filter(pl.col("soma-ID") == soma_id).sort("time")
        data = sdf["data"].to_numpy()
        time = sdf["time"].to_numpy()
        dres = detect_calcium_events(data, time)
        evdf = pl.DataFrame(dres.events)
        evdf = evdf.with_columns(pl.lit(soma_id).alias("soma-ID"))
        dff = pl.DataFrame({"dff": dres.dff, "soma-ID": soma_id, "time": time})
        evdfs.append(evdf)
        dffs.append(dff)
    evdf = pl.concat(evdfs)
    dff = pl.concat(dffs)
    return dff, evdf
