"""Matched-pursuit glutamate event detection (calibrated spike train from LS).

Exposed as ``wis.scope.pursuit``. Developed in
``slap_mi_2_sleep/principled_glut_events/`` (see the step-by-step walkthrough
notebook ``pursuit_pipeline_spelled_out.ipynb`` there for figures of every
step and the derivation); that folder's ``glut_pursuit.py`` is now a shim
re-exporting this module.

Usage
-----
>>> import wisco_slap as wis
>>> dn = wis.get.combine_scopex_arrays(
...     wis.get.syn_dF(subject, exp, loc, acq, trace="denoised")).sel(channel=1)
>>> ls = wis.get.combine_scopex_arrays(
...     wis.get.syn_dF(subject, exp, loc, acq, trace="ls")).sel(channel=1)
>>> res = wis.scope.pursuit.detect_events(dn, ls)      # catalog + rates + FDR table
>>> res.rates                                          # per-synapse rates at z*
>>> wis.scope.pursuit.rates_at(res, z=8)               # re-slice, no recompute
>>> res.fdr                                            # the full FDR(z) curve
>>> snr = wis.scope.pursuit.matchfilt_snr(dn, ls)      # step-4 trace in SNR units
>>> st = wis.scope.pursuit.spike_train(res, dn)        # impulse-train DataArray
>>> dn_sig = wis.scope.pursuit.denoised_sigma_units(dn, ls)  # dn / sigma0

Method summary
--------------
1. Matched filter of LS with the sensor template (exp decay, ``tau_s``).
2. sigma_null per synapse DERIVED as c * sigma0 — sigma0 from
   ``lower_half_mad_sigma(dn, ls)``; c measured on the quietest
   ``quiet_syn_frac`` of synapses (per ``group_coord`` when present) —
   never measured on the filtered trace itself (it has no signal-free
   samples at hyperactive synapses).
3. Greedy matched pursuit: accept largest peak, subtract its template
   footprint (resolves bursts by explaining-away), ``min_sep_s`` refractory.
4. Shot-noise deflation of each event's z by the PRE-event baseline level
   (never the event's own peak — circular), capped at ``shot_cap``.
5. Threshold by empirical FDR: the identical statistic on sign-flipped
   quiet samples counts false positives by construction (noise symmetric,
   events positive-only).

The denoised trace is used only as a mask (which samples are quiet) and as
the local signal-level estimate for shot noise; every statistic is computed
on the LS trace, which is the data.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import polars as pl
import xarray as xr
from scipy.signal import fftconvolve, find_peaks

from ..util.snr import lower_half_mad_sigma


class PursuitResult(NamedTuple):
    """Everything detect_events computes.

    events : full candidate catalog (one row per accepted pursuit event, down
        to ``z_floor``) with per-event ``z`` (baseline-null units) and
        ``z_shot`` (shot-deflated — use this one), plus every synapse-level
        coord of the input (soma-ID, dend-ID, pos, dend_type, dmd, depths, ...)
        as metadata columns. Filter it yourself or via :func:`rates_at`.
    rates : per-synapse event rates at the chosen operating point ``z_star``.
    fdr : the calibration table — expected false events vs threshold.
    per_syn : per-synapse diagnostics (sigma0, sigma_null, act_frac, shot b/a).
    params : dict of resolved parameters (c per group, z_star, duration, ...).
    """

    events: pl.DataFrame
    rates: pl.DataFrame
    fdr: pl.DataFrame
    per_syn: pl.DataFrame
    params: dict


#: Default rolling-sigma0 window (s) for detect_events / matchfilt_snr /
#: denoised_sigma_units. Gain-stabilises against bleaching so z is measured
#: against the LOCAL noise; pass ``sigma_window_s=None`` for the legacy
#: scalar-per-synapse behaviour.
SIGMA_WINDOW_S_DEFAULT = 300.0


def _template_len(tau_s: float, fs: float) -> int:
    """Template length in samples (``max(4, ceil(3 tau))``) -- one shared
    definition so the detector and :func:`detectable_mask` agree exactly.

    3 tau keeps 99.75%% of the exponential kernel's energy (matched-filter z
    changes ~0.1%%) while shrinking the per-gap exclusion zone by 40%% versus
    the earlier 5 tau template. The product is rounded (1e-6) before the ceil
    so floating-point noise in an estimated ``fs`` cannot flip an
    exact-integer ``3 tau`` up by one."""
    return max(4, int(np.ceil(round(3 * tau_s * fs, 6))))


def _valid_mask(X_ls: np.ndarray, klen: int) -> np.ndarray:
    """Samples where the pursuit may place an event.

    The matched filter is causal -- ``m[t]`` is computed from samples
    ``[t, t + klen - 1]`` only -- so a NaN gap ``[g0, g1]`` contaminates
    exactly ``t in [g0 - klen + 1, g1]``: one template length *before* the
    gap and nothing after it. A sample is valid iff its forward template
    window contains no non-finite LS sample. (The earlier symmetric
    ``+- klen`` dilation wasted the entire trailing window.)"""
    n_syn, n_t = X_ls.shape
    valid = np.zeros(X_ls.shape, dtype=bool)
    kern = np.ones(klen)
    for i in range(n_syn):
        bad = (~np.isfinite(X_ls[i])).astype(np.int8)
        cov = np.convolve(bad, kern, "full")[klen - 1 : klen - 1 + n_t]
        valid[i] = cov == 0
    return valid


def _matched_filter(arr: np.ndarray, k_unit: np.ndarray) -> np.ndarray:
    klen = k_unit.size
    rev = k_unit[::-1][np.newaxis, :]
    full = fftconvolve(np.nan_to_num(arr), rev, mode="full", axes=-1)
    return full[:, klen - 1 : klen - 1 + arr.shape[-1]]


def _shot_ba(dn_1d: np.ndarray, resid_1d: np.ndarray, s0: float) -> float:
    """b/a of the shot-noise law var(resid) ~ a + b*dn, with physical guards."""
    m = np.isfinite(dn_1d) & np.isfinite(resid_1d)
    d, r = dn_1d[m], resid_1d[m]
    if d.size < 5000:
        return np.nan
    qs = np.unique(np.quantile(d, [0.0, 0.5, 0.75, 0.9, 0.97, 0.995, 1.0]))
    levels, variances = [], []
    for lo, hi in zip(qs[:-1], qs[1:], strict=True):
        sel = (d >= lo) & (d < hi)
        if sel.sum() < 200:
            continue
        rb = r[sel]
        sig = 1.4826 * np.median(np.abs(rb - np.median(rb)))
        levels.append(d[sel].mean())
        variances.append(sig**2)
    if len(levels) < 3:
        return np.nan
    b, a = np.polyfit(levels, variances, 1)
    a_eff = max(a, 0.3 * s0**2)
    return max(b, 0.0) / a_eff


def _synapse_meta(da: xr.DataArray, syn_dim: str) -> pl.DataFrame:
    """One row per synapse with every 1-D coord along ``syn_dim`` (soma-ID,
    dend-ID, pos, dend_type, dmd, dmd-depth, soma-depth, ...) as a column."""
    syn_ids = np.asarray(da[syn_dim].values).astype(str)
    cols: dict[str, list] = {"syn_id": syn_ids.tolist()}
    for name, coord in da.coords.items():
        if name == syn_dim or coord.dims != (syn_dim,):
            continue
        vals = np.asarray(coord.values)
        if vals.dtype.kind in "fiub":
            cols[str(name)] = [
                None if (isinstance(v, float) and np.isnan(v)) else v
                for v in vals.tolist()
            ]
        else:
            cols[str(name)] = [
                None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
                for v in vals.tolist()
            ]
    return pl.DataFrame(cols)


def _prepare(
    dn: xr.DataArray,
    ls: xr.DataArray,
    tau_s: float,
    baseline_quantile: float,
    quiet_syn_frac: float,
    group_coord: str | None,
    time_dim: str,
    verbose: bool,
    sigma_window_s: float | None = None,
) -> SimpleNamespace:
    """Shared front end: template, matched filter, sigma0, derived sigma_null,
    quiet/valid masks and shot-noise slopes. Used by both detect_events and
    matchfilt_snr so the two can never disagree about the noise scale."""
    if dn.dims != ls.dims or dn.ndim != 2 or time_dim not in dn.dims:
        raise ValueError(
            "dn and ls must share exactly two dims: (synapse_dim, time). "
            f"Got dn={dn.dims}, ls={ls.dims}. Select the channel first."
        )
    syn_dim = next(d for d in dn.dims if d != time_dim)
    dn = dn.transpose(syn_dim, time_dim)
    ls = ls.transpose(syn_dim, time_dim)
    t = np.asarray(dn[time_dim].values, dtype=float)
    fs = 1.0 / float(np.median(np.diff(t)))
    dur = float(t[-1] - t[0])
    syn_ids = np.asarray(dn[syn_dim].values).astype(str)
    n_syn, n_t = dn.shape
    X_ls = np.asarray(ls.values, dtype=np.float32)
    X_dn = np.asarray(dn.values, dtype=np.float32)

    # ---- optional gain stabilisation (rolling sigma0) ---------------------
    # Bleaching / slow gain drift makes the per-sample noise non-stationary,
    # which a single per-synapse sigma0 cannot absorb (early z inflated, late
    # z deflated -> spurious rate decline). With ``sigma_window_s`` set, each
    # trace is divided by its smooth gain profile sigma_roll(t)/sigma_scalar
    # BEFORE anything else, so every downstream quantity (matched filter,
    # sigma0, noise-colour c, shot law, FDR null) sees stationary noise.
    # Amplitudes are then in gain-stabilised dF units (referenced to the
    # recording-median gain), which keeps event sizes comparable across time.
    gain = None
    if sigma_window_s is not None:
        sig_roll = (
            lower_half_mad_sigma(
                dn,
                ls,
                baseline_quantile=baseline_quantile,
                window_s=sigma_window_s,
                time_dim=time_dim,
            )
            .transpose(syn_dim, time_dim)
            .values.astype(np.float64)
        )
        sig_scal = lower_half_mad_sigma(
            dn, ls, baseline_quantile=baseline_quantile, time_dim=time_dim
        ).values.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            gain = sig_roll / sig_scal[:, None]
        idx_t = np.arange(n_t)
        for i in range(n_syn):
            fin = np.isfinite(gain[i])
            if fin.any():
                gain[i] = np.interp(idx_t, idx_t[fin], gain[i][fin])
            else:
                gain[i] = 1.0
        gain = np.clip(gain, 0.3, 3.0).astype(np.float32)
        X_ls = X_ls / gain
        X_dn = X_dn / gain
        dn = dn.copy(data=X_dn)
        ls = ls.copy(data=X_ls)

    # ---- template ---------------------------------------------------------
    tau = tau_s * fs
    klen = _template_len(tau_s, fs)
    e_shape = np.exp(-np.arange(klen) / tau)
    k_unit = (e_shape / np.linalg.norm(e_shape)).astype(np.float64)
    e_norm = float(np.linalg.norm(e_shape))
    foot_full = np.correlate(
        np.r_[np.zeros(klen), e_shape, np.zeros(klen)], k_unit, mode="valid"
    )
    peak_off = int(np.argmax(foot_full))
    footprint = foot_full / foot_full[peak_off]

    m_all = _matched_filter(X_ls, k_unit)

    # ---- sigma0, quiet/valid masks, act_frac ------------------------------
    sigma0 = lower_half_mad_sigma(
        dn, ls, baseline_quantile=baseline_quantile, time_dim=time_dim
    ).values.astype(float)
    quiet = np.zeros((n_syn, n_t), dtype=bool)
    valid = _valid_mask(X_ls, klen)
    sigma_meas = np.full(n_syn, np.nan)
    act_frac = np.full(n_syn, np.nan)
    for i in range(n_syn):
        v = valid[i]
        fin_dn = np.isfinite(X_dn[i])
        if not fin_dn.any():
            continue
        thr = np.nanquantile(np.where(fin_dn, X_dn[i], np.nan), baseline_quantile)
        quiet[i] = v & fin_dn & (X_dn[i] <= thr)
        if quiet[i].sum() > 1000:
            q = m_all[i][quiet[i]]
            sigma_meas[i] = 1.4826 * np.median(np.abs(q - np.median(q)))
        act_frac[i] = np.nanmean(X_dn[i] > 3 * sigma0[i])

    # ---- derived sigma_null = c * sigma0 ----------------------------------
    if group_coord is not None and group_coord in dn.coords:
        groups = np.asarray(dn[group_coord].values).astype(str)
    else:
        groups = np.array(["all"] * n_syn)
    sigma_null = np.full(n_syn, np.nan)
    c_by_group: dict[str, float] = {}
    for g in np.unique(groups):
        sel = (groups == g) & np.isfinite(sigma_meas) & np.isfinite(sigma0)
        if sel.sum() < 3:
            continue
        qs = sel & (act_frac <= np.nanquantile(act_frac[sel], quiet_syn_frac))
        c_g = float(np.median(sigma_meas[qs] / sigma0[qs]))
        c_by_group[g] = c_g
        sigma_null[sel] = c_g * sigma0[sel]
    if verbose:
        print(
            "noise-color c per group:", {k: round(v, 2) for k, v in c_by_group.items()}
        )

    # ---- shot b/a per synapse ---------------------------------------------
    resid = X_ls - X_dn
    ba = np.array([
        _shot_ba(X_dn[i], resid[i], sigma0[i]) if np.isfinite(sigma0[i]) else np.nan
        for i in range(n_syn)
    ])
    ba = np.where(np.isfinite(ba), ba, np.nanmedian(ba))

    return SimpleNamespace(
        gain=gain,
        dn=dn,
        ls=ls,
        syn_dim=syn_dim,
        t=t,
        fs=fs,
        dur=dur,
        syn_ids=syn_ids,
        n_syn=n_syn,
        n_t=n_t,
        X_ls=X_ls,
        X_dn=X_dn,
        klen=klen,
        e_shape=e_shape,
        k_unit=k_unit,
        e_norm=e_norm,
        peak_off=peak_off,
        footprint=footprint,
        m_all=m_all,
        sigma0=sigma0,
        quiet=quiet,
        valid=valid,
        sigma_meas=sigma_meas,
        act_frac=act_frac,
        groups=groups,
        sigma_null=sigma_null,
        c_by_group=c_by_group,
        ba=ba,
    )


def detect_events(
    dn: xr.DataArray,
    ls: xr.DataArray,
    tau_s: float = 0.026,
    fdr_alpha: float | None = None,
    z_threshold: float = 5.0,
    z_floor: float = 3.0,
    min_sep_s: float = 0.010,
    baseline_quantile: float = 0.5,
    quiet_syn_frac: float = 0.3,
    group_coord: str | None = "dmd",
    shot_correction: bool = True,
    shot_cap: float = 4.0,
    f0: xr.DataArray | None = None,
    time_dim: str = "time",
    sigma_window_s: float | None = SIGMA_WINDOW_S_DEFAULT,
    verbose: bool = True,
) -> PursuitResult:
    """Detect discrete glutamate release events by calibrated matched pursuit.

    Parameters
    ----------
    dn, ls : xr.DataArray
        Denoised and LS traces, dims exactly (synapse_dim, time). Select the
        channel first (e.g. ``.sel(channel=1)``).
    tau_s : float
        Sensor decay constant for the template. 0.026 (iGluSnFR4f). Use
        ~0.15 for jRGECO if you ever try the calcium channel.
    fdr_alpha : float or None
        None (default): use ``z_threshold`` directly. A float: pick the
        smallest z whose event set has estimated FDR <= alpha. CAUTION: the
        FDR curve is flat and non-monotonic (~6-10%) over a wide z range on
        bellatrix data, so this dial is touchy — always inspect ``res.fdr``
        before trusting it.
    z_threshold : float
        Operating point on ``z_shot``. 5.0 (default) = the "balanced" point
        validated on bellatrix (FDR ~ 8%, mean rate ~ 0.5 ev/s, lognormal);
        8.75+ = strict / high purity. Re-slice cheaply later with
        :func:`rates_at` — the returned catalog keeps everything >= z_floor.
    z_floor : float
        Lowest z retained in the candidate catalog (pursuit base threshold).
    min_sep_s : float
        Refractory / minimum event separation (10 ms default).
    baseline_quantile : float
        Denoised-quantile conditioning for sigma0 and the quiet mask.
    sigma_window_s : float or None
        Rolling-sigma0 window in seconds (default
        :data:`SIGMA_WINDOW_S_DEFAULT` = 300): sigma0 is estimated in a
        centered rolling window and each trace is divided by its smooth gain
        profile before detection, so z is measured against the LOCAL noise.
        Without this, a global sigma0 turns slow gain drift (bleaching) into
        spurious rate trends — a mere ~8%% end-of-recording noise drop was
        measured to inflate the apparent first/last rate ratio from 1.23x to
        1.61x (bellatrix loc_Z acq_1). Amplitude columns (``amp``,
        ``amp_dff``) are in gain-stabilised units (dF referenced to the
        recording-median gain). Pass ``None`` for the legacy scalar sigma0.
    quiet_syn_frac : float
        Fraction of least-active synapses used to estimate the noise-color
        constant c (per group when ``group_coord`` exists).
    group_coord : str or None
        Coordinate on the synapse dim to group the c estimate by (e.g.
        ``"dmd"``). Falls back to a single global c when absent.
    shot_correction : bool
        Deflate event z by the pre-event baseline shot factor. Disable to
        see the uncorrected statistic (events in bursts will be inflated).
    shot_cap : float
        Maximum shot deflation factor.
    f0 : xr.DataArray or None
        Optional baseline fluorescence (same dims); adds ``amp_dff``.
    """
    P = _prepare(
        dn,
        ls,
        tau_s=tau_s,
        baseline_quantile=baseline_quantile,
        quiet_syn_frac=quiet_syn_frac,
        group_coord=group_coord,
        time_dim=time_dim,
        verbose=verbose,
        sigma_window_s=sigma_window_s,
    )
    dn, ls = P.dn, P.ls
    syn_dim, t, fs, dur, syn_ids = P.syn_dim, P.t, P.fs, P.dur, P.syn_ids
    n_syn, n_t, X_dn = P.n_syn, P.n_t, P.X_dn
    e_norm, peak_off, footprint, m_all = P.e_norm, P.peak_off, P.footprint, P.m_all
    sigma0, quiet, valid, act_frac, groups = (
        P.sigma0,
        P.quiet,
        P.valid,
        P.act_frac,
        P.groups,
    )
    sigma_null, c_by_group, ba = P.sigma_null, P.c_by_group, P.ba
    min_sep = max(1, int(round(min_sep_s * fs)))

    def pre_infl(i: int, idx: np.ndarray) -> np.ndarray:
        if not shot_correction:
            return np.ones(idx.size)
        pre = np.nanmedian(
            np.stack([X_dn[i][np.maximum(idx - k, 0)] for k in (2, 3, 4)]), axis=0
        )
        return np.minimum(
            np.sqrt(1.0 + ba[i] * np.maximum(np.nan_to_num(pre), 0)), shot_cap
        )

    # ---- greedy pursuit ----------------------------------------------------
    def pursue(i: int) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(sigma_null[i]):
            return np.array([], dtype=int), np.array([])
        m = m_all[i].astype(np.float64)
        m[~valid[i]] = -np.inf
        thr_abs = z_floor * sigma_null[i]
        acc_i: list[int] = []
        acc_a: list[float] = []
        for _sweep in range(3):
            pk, _ = find_peaks(m, height=thr_abs)
            if pk.size == 0:
                break
            changed = False
            for p in pk[np.argsort(-m[pk])]:
                val = m[p]
                if not np.isfinite(val) or val < thr_abs:
                    continue
                lo, hi = max(0, p - 1), min(n_t, p + 2)
                if val < np.max(m[lo:hi]) - 1e-9:
                    continue
                amp = val / e_norm
                a0 = p - peak_off
                b0 = a0 + footprint.size
                fa, fb = max(0, a0), min(n_t, b0)
                m[fa:fb] -= amp * e_norm * footprint[fa - a0 : fb - a0]
                m[max(0, p - min_sep + 1) : p + min_sep] = -np.inf
                acc_i.append(p)
                acc_a.append(amp)
                changed = True
            if not changed:
                break
        idx = np.array(acc_i, dtype=int)
        amp = np.array(acc_a)
        order = np.argsort(idx)
        return idx[order], amp[order]

    rows = []
    if f0 is not None:
        f0 = f0.transpose(syn_dim, time_dim)
        f0_med = np.nanmedian(np.asarray(f0.values, dtype=float), axis=1)
    for i in range(n_syn):
        idx, amp = pursue(i)
        if idx.size == 0:
            continue
        z_base = amp * e_norm / sigma_null[i]
        z_shot = z_base / pre_infl(i, idx)
        for j in range(idx.size):
            row = {
                "syn_id": syn_ids[i],
                "idx": int(idx[j]),
                "time": float(t[idx[j]]),
                "amp": float(amp[j]),
                "amp_sigma0": float(amp[j] / sigma0[i]),
                "z": float(z_base[j]),
                "z_shot": float(z_shot[j]),
            }
            if f0 is not None:
                row["amp_dff"] = float(amp[j] / f0_med[i])
            rows.append(row)
    events = pl.DataFrame(rows)
    meta = _synapse_meta(dn, syn_dim)
    meta_cols = [c for c in meta.columns if c != "syn_id"]
    if events.height:
        events = events.join(meta, on="syn_id", how="left")
    if verbose:
        print(f"catalog: {events.height} candidate events (z >= {z_floor})")

    # ---- FDR calibration on the shot-corrected statistic -------------------
    z_grid = np.arange(z_floor, 14.01, 0.25)
    null_tot = np.zeros(z_grid.size)
    for i in range(n_syn):
        if not np.isfinite(sigma_null[i]):
            continue
        neg = np.where(quiet[i], -m_all[i], -np.inf)
        pk, _ = find_peaks(neg)
        if pk.size == 0 or quiet[i].mean() == 0:
            continue
        zc = (neg[pk] / sigma_null[i]) / pre_infl(i, pk)
        null_tot += np.array([(zc >= zz).sum() for zz in z_grid]) / quiet[i].mean()
    zs_all = events["z_shot"].to_numpy() if events.height else np.array([])
    det_tot = np.array([(zs_all >= zz).sum() for zz in z_grid])
    fdr_curve = null_tot / np.maximum(det_tot, 1)
    fdr_df = pl.DataFrame({
        "z": z_grid,
        "detections": det_tot.astype(int),
        "expected_fp": null_tot,
        "fdr": fdr_curve,
    })

    if fdr_alpha is not None:
        ok = fdr_curve <= fdr_alpha
        z_star = float(z_grid[np.argmax(ok)]) if ok.any() else float(z_grid[-1])
    else:
        z_star = float(z_threshold)
    if verbose:
        j = min(int(np.searchsorted(z_grid, z_star)), z_grid.size - 1)
        mode = f"FDR<={fdr_alpha}" if fdr_alpha is not None else "z_threshold"
        print(
            f"z* = {z_star} ({mode}) | detections {int(det_tot[j])}"
            f" | expected FP {null_tot[j]:.0f} | set FDR {fdr_curve[j]:.3f}"
        )

    per_syn = meta.join(
        pl.DataFrame({
            "syn_id": syn_ids,
            "sigma0": sigma0,
            "sigma_null": sigma_null,
            "act_frac": act_frac,
            "shot_b_over_a": ba,
            "group": groups,
        }),
        on="syn_id",
        how="left",
    )
    params = {
        "sigma_window_s": sigma_window_s,
        "tau_s": tau_s,
        "fs": fs,
        "duration_s": dur,
        "z_star": z_star,
        "c_by_group": c_by_group,
        "fdr_alpha": fdr_alpha,
        "min_sep_s": min_sep_s,
        "shot_correction": shot_correction,
        "meta_cols": meta_cols,
    }
    res = PursuitResult(events, pl.DataFrame(), fdr_df, per_syn, params)
    rates = rates_at(res, z_star)
    res = PursuitResult(events, rates, fdr_df, per_syn, params)
    if verbose and rates.height:
        r = rates["rate"].to_numpy()
        nz = r[r > 0]
        print(
            f"rates at z*: mean {r.mean():.3f} ev/s | median {np.median(r):.3f} "
            f"| log10-sd {np.log10(nz).std() if nz.size > 2 else float('nan'):.2f}"
        )
    return res


def matchfilt_snr(
    dn: xr.DataArray,
    ls: xr.DataArray,
    tau_s: float = 0.026,
    baseline_quantile: float = 0.5,
    quiet_syn_frac: float = 0.3,
    group_coord: str | None = "dmd",
    shot_correction: bool = False,
    shot_cap: float = 4.0,
    time_dim: str = "time",
    sigma_window_s: float | None = SIGMA_WINDOW_S_DEFAULT,
    nan_floor: bool = False,
    verbose: bool = False,
) -> xr.DataArray:
    """Matched-filtered LS trace in SNR units — the trace the pursuit runs on.

    Returns ``matched_filter(ls) / sigma_null`` with exactly the dims/coords of
    ``dn`` (which must match ``ls``). sigma_null is the pipeline's derived
    noise scale (c * sigma0 per synapse, see :func:`detect_events`), so values
    are directly comparable across synapses: a value of 5 means "5x what
    baseline noise produces in the filtered trace at this synapse" — the same
    units as the per-event ``z`` column of the event catalog. By default,
    samples whose forward template window overlaps a NaN gap (one template
    length *before* each gap — the filter is causal, nothing after) are NaN,
    matching :func:`detectable_mask`.

    Parameters
    ----------
    shot_correction : bool
        False (default): the plain baseline-noise SNR, i.e. the step-4 trace.
        True: additionally deflate each sample by the shot-noise inflation
        implied by the denoised level 10-20 ms earlier (the continuous analog
        of the per-event pre-event correction in detect_events) — comparable
        to the catalog's ``z_shot`` rather than ``z``.
    nan_floor : bool
        False (default): NaN out the detector's full exclusion zone
        (:func:`detectable_mask`) — every finite value is a trustworthy z.
        True: NaN out only the samples that are NaN in ``ls`` itself —
        maximum data retention for visualization. Values inside the leading
        template window of a gap are then attenuated (the gap is zero-filled),
        so do NOT threshold or count on them.
    sigma_window_s : float or None
        As in :func:`detect_events` — gain-stabilise against a rolling sigma0
        so the SNR is measured against the local noise (default
        :data:`SIGMA_WINDOW_S_DEFAULT` = 300; ``None`` for the legacy scalar).
    Other parameters are as in :func:`detect_events`.
    """
    orig_dims = dn.dims
    P = _prepare(
        dn,
        ls,
        tau_s=tau_s,
        baseline_quantile=baseline_quantile,
        quiet_syn_frac=quiet_syn_frac,
        group_coord=group_coord,
        time_dim=time_dim,
        verbose=verbose,
        sigma_window_s=sigma_window_s,
    )
    snr = P.m_all / P.sigma_null[:, None]
    if shot_correction:
        X = np.maximum(np.nan_to_num(P.X_dn), 0)
        shifted = [
            np.concatenate([np.repeat(X[:, :1], k, axis=1), X[:, :-k]], axis=1)
            for k in (2, 3, 4)
        ]
        pre = np.median(np.stack(shifted), axis=0)
        infl = np.minimum(np.sqrt(1.0 + P.ba[:, None] * pre), shot_cap)
        snr = snr / infl
    snr = snr.astype(np.float32)
    if nan_floor:
        snr[~np.isfinite(P.X_ls)] = np.nan
    else:
        snr[~P.valid] = np.nan
    out = xr.DataArray(
        snr,
        dims=P.dn.dims,
        coords=P.dn.coords,
        name="matchfilt_snr",
        attrs={
            "units": "sigma_null",
            "tau_s": tau_s,
            "shot_correction": shot_correction,
            "nan_floor": nan_floor,
            "sigma_window_s": sigma_window_s,
            "c_by_group": {k: round(v, 3) for k, v in P.c_by_group.items()},
        },
    )
    return out.transpose(*orig_dims)


def _fill_sigma_gaps(sigma: xr.DataArray, time_dim: str) -> xr.DataArray:
    """Bridge NaN stretches of a rolling sigma along time (per trace) by linear
    interpolation / edge extension. sigma is a smooth nuisance scale, so a
    window that could not be estimated (long NaN gap, no finite samples) is
    far better filled from its neighbours than left as a NaN strip in the
    normalised output (which is NaN wherever the data itself is NaN anyway).
    """
    arr = sigma.transpose(..., time_dim).values.astype(np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    idx = np.arange(flat.shape[1])
    for i in range(flat.shape[0]):
        fin = np.isfinite(flat[i])
        if fin.any() and not fin.all():
            flat[i] = np.interp(idx, idx[fin], flat[i][fin])
    out = sigma.copy(data=flat.reshape(arr.shape).astype(sigma.dtype))
    return out.transpose(*sigma.dims)


def denoised_sigma_units(
    dn: xr.DataArray,
    ls: xr.DataArray,
    baseline_quantile: float = 0.5,
    sigma_window_s: float | None = SIGMA_WINDOW_S_DEFAULT,
    time_dim: str = "time",
) -> xr.DataArray:
    """Denoised trace in units of the LS noise floor (``dn / sigma0``).

    A NORMALIZATION, not an SNR — deliberately not named like
    :func:`matchfilt_snr`. It expresses the fitted signal in units of the
    *data's* per-sample noise, which makes event sizes comparable across
    synapses (like plotting a fitted curve against the data's error bars).
    But the denoised trace's own fluctuations are not noise at unit scale: on
    quiet samples it sits on the rectified reconstruction floor (median ~1,
    robust sd ~0.6 on bellatrix), no sign-flip null exists for it, and its
    peaks relate to ``matchfilt_snr`` by a data-dependent factor (~1.1 vs the
    0.82 a linear filter would give). Use it for display and comparing event
    sizes; use :func:`matchfilt_snr` or the event catalog for anything you
    threshold, count, or attach a false-positive rate to.

    Parameters
    ----------
    dn, ls : xr.DataArray
        Denoised and LS traces with identical dims/coords; any dims are fine
        as long as ``time_dim`` is one of them (no channel pre-selection
        needed — sigma0 is estimated per non-time slice).
    baseline_quantile : float
        Denoised-quantile conditioning for the sigma0 estimate (see
        ``wis.util.lower_half_mad_sigma``).
    sigma_window_s : float or None
        Rolling-sigma0 window in seconds (default
        :data:`SIGMA_WINDOW_S_DEFAULT` = 300) — a centered rolling sigma0, so
        the normalisation tracks slow gain drift (bleaching). ``None``: one
        scalar sigma0 per synapse (legacy; NaN where a rolling window has too
        few quiet samples does not apply then).

    Returns
    -------
    xr.DataArray
        ``dn / sigma0`` with exactly the dims/coords of ``dn``.

    Notes
    -----
    This function never dilates NaNs: the output is NaN exactly where ``dn``
    is NaN (it is already at the "NaN floor" — no ``nan_floor`` flag needed,
    unlike :func:`matchfilt_snr`). In rolling mode, windows whose sigma could
    not be estimated are bridged by interpolating the (smooth) sigma along
    time, so the normalisation itself never introduces NaN strips.
    """
    if dn.dims != ls.dims or time_dim not in dn.dims:
        raise ValueError(
            f"dn and ls must share dims including '{time_dim}'. "
            f"Got dn={dn.dims}, ls={ls.dims}."
        )
    sigma0 = lower_half_mad_sigma(
        dn,
        ls,
        baseline_quantile=baseline_quantile,
        window_s=sigma_window_s,
        time_dim=time_dim,
    )
    if sigma_window_s is not None:
        sigma0 = _fill_sigma_gaps(sigma0, time_dim)
    out = (dn / sigma0).astype(np.float32)
    out.name = "denoised_sigma_units"
    out.attrs = {
        "units": "sigma0 (LS per-sample noise)",
        "note": "normalization, not an SNR — see docstring",
        "baseline_quantile": baseline_quantile,
        "sigma_window_s": sigma_window_s if sigma_window_s is not None else "scalar",
    }
    return out.transpose(*dn.dims)


def spike_train(
    res: PursuitResult,
    like: xr.DataArray,
    z: float | None = None,
    reconvolve: bool = False,
    time_dim: str = "time",
) -> xr.DataArray:
    """Rasterize the event catalog into an impulse-train DataArray.

    The calibrated version of what the pipeline's ``events`` zarr wanted to
    be: amplitude at each detected event sample, zero elsewhere.

    Parameters
    ----------
    like : xr.DataArray
        Any array with the same (synapse, time) axes the detection ran on
        (e.g. the ``dn`` you passed to :func:`detect_events`) — provides
        dims/coords for the output.
    z : float or None
        z_shot threshold; defaults to the result's operating point.
    reconvolve : bool
        If True, convolve the impulse train with the sensor template —
        a "clean denoised" trace containing only significant events.
    """
    if z is None:
        z = res.params["z_star"]
    syn_dim = next(d for d in like.dims if d != time_dim)
    like = like.transpose(syn_dim, time_dim)
    syn_ids = np.asarray(like[syn_dim].values).astype(str)
    row = {s: i for i, s in enumerate(syn_ids)}
    out = np.zeros(like.shape, dtype=np.float32)
    sub = res.events.filter(pl.col("z_shot") >= z)
    for r in sub.iter_rows(named=True):
        i = row.get(r["syn_id"])
        if i is not None:
            out[i, r["idx"]] += r["amp"]
    if reconvolve:
        fs = res.params["fs"]
        tau = res.params["tau_s"] * fs
        k = np.exp(-np.arange(max(4, int(np.ceil(5 * tau)))) / tau)
        out = fftconvolve(out, k[np.newaxis, :], mode="full", axes=-1)[
            :, : out.shape[-1]
        ].astype(np.float32)
    return xr.DataArray(out, dims=like.dims, coords=like.coords, name="spike_train")


def detectable_mask(
    ls: xr.DataArray, tau_s: float = 0.026, time_dim: str = "time"
) -> xr.DataArray:
    """Where :func:`detect_events` is *able* to place an event (bool, like ``ls``).

    This is exactly the detector's internal ``valid`` mask: a sample ``t`` is
    detectable only if the LS trace is finite over its entire forward template
    window ``[t, t + klen - 1]`` (``klen = max(4, ceil(3 tau_s fs))``). The
    filter is causal, so each NaN gap costs one template length *before* it
    and nothing after.
    The pursuit sets the matched-filter trace to ``-inf`` everywhere else, so
    an empty stretch of a raster that falls inside ``~detectable_mask`` means
    "could not be detected", not "no events". Cheap (no noise estimation), so
    it can be called on any subset of synapses, e.g. to shade rasters.
    """
    if ls.ndim != 2 or time_dim not in ls.dims:
        raise ValueError(f"ls must be 2-D with a {time_dim!r} dim; got {ls.dims}")
    syn_dim = next(d for d in ls.dims if d != time_dim)
    ls_t = ls.transpose(syn_dim, time_dim)
    t = np.asarray(ls_t[time_dim].values, dtype=float)
    fs = 1.0 / float(np.median(np.diff(t)))
    valid = _valid_mask(
        np.asarray(ls_t.values, dtype=np.float32), _template_len(tau_s, fs)
    )
    out = xr.DataArray(valid, dims=ls_t.dims, coords=ls_t.coords, name="detectable")
    return out.transpose(*ls.dims)


def rates_at(res: PursuitResult, z: float) -> pl.DataFrame:
    """Per-synapse rates at any z_shot threshold — no recompute needed."""
    dur = res.params["duration_s"]
    got = (
        res.events
        .filter(pl.col("z_shot") >= z)
        .group_by("syn_id")
        .agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / dur).alias("rate"))
    )
    return (
        res.per_syn
        .select(["syn_id", *res.params.get("meta_cols", [])])
        .join(got, on="syn_id", how="left")
        .fill_null(0)
        .with_columns(pl.lit(float(z)).alias("z_threshold"))
    )


__all__ = [
    "detectable_mask",
    "PursuitResult",
    "denoised_sigma_units",
    "detect_events",
    "matchfilt_snr",
    "rates_at",
    "spike_train",
]
