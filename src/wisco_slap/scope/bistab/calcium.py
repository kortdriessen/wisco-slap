"""Synaptic calcium (jRGECO1a) around glutamate events, inside and outside ON periods.

The calcium channel carries a scaled same-frame copy of the iGluSnFR transient
(crosstalk), so every response here removes it per synapse before calling
anything "synaptic calcium".
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .core import GLU_CHANNEL, Detector, _core, _kw, _span_mask, state_at, third_of
from .data import load_ca, load_dff

#: Calcium response windows around a glutamate event (seconds): baseline and response.
CA_PRE = (-0.20, -0.02)
CA_POST = (0.03, 0.30)
#: Frames (from the event) used to estimate glutamate->calcium-channel crosstalk.
XTALK_LAGS = (0, 3)


def _nan_cumsum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(x)
    return (
        np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))]),
        np.concatenate([[0], np.cumsum(ok)]),
    )


def _win_mean(cs, cn, idx, a, b):
    """Mean of x[idx+a : idx+b] from cumulative sums (NaN-aware, NaN if empty)."""
    n = cs.size - 1
    lo, hi = np.clip(idx + a, 0, n), np.clip(idx + b, 0, n)
    cnt = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt >= 0.8 * (b - a), (cs[hi] - cs[lo]) / cnt, np.nan)


def ca_responses(
    det: Detector, n_pseudo: int = 1, rng_seed: int = 0, **params
) -> pl.DataFrame:
    """Calcium response of a synapse to each of its own glutamate events.

    ``ca_raw`` = mean calcium dF/F0 in :data:`CA_POST` minus mean in
    :data:`CA_PRE` (relative to the event). The calcium channel carries a
    scaled copy of the fast glutamate transient (peaking in the very frame of
    the event, far faster than jRGECO1a can rise), most likely optical
    crosstalk. Per synapse, the crosstalk factor ``alpha`` is the calcium /
    glutamate ratio over the first :data:`XTALK_LAGS` frames of the
    event-triggered averages, and ``ca`` = ``ca_raw - alpha * glu`` (``glu`` is
    the same window difference on the glutamate channel). Taking all of the
    early calcium as crosstalk is conservative: any real calcium in those
    frames is removed too.

    One row per event with ``syn_id``, ``kind`` (``event`` / ``pseudo``),
    ``state``, ``third``, ``in_on`` / ``in_off``, ``glu_amp`` (pursuit amplitude),
    ``ca_raw``, ``glu``, ``alpha``, ``ca`` and ``isolated`` (no other event of the
    same synapse within ±0.4 s, so the window holds this one release only).

    Pseudo-events (``kind = "pseudo"``) are random moments, ``n_pseudo`` per real
    event and drawn in the same condition (inside an ON or not), at which the
    synapse had no glutamate event within ±0.4 s. They measure how much calcium
    rises at that synapse just from being in that moment (e.g. dendritic or
    back-propagated calcium during an ON). ``event − pseudo`` isolates the part
    of the response tied to the synapse's own glutamate.
    """
    kw = _kw(params)
    acq = det.acq
    ca, glu = load_ca(acq), load_dff(acq, GLU_CHANNEL)
    fs = 1.0 / float(np.median(np.diff(acq.time)))
    pre = (int(round(CA_PRE[0] * fs)), int(round(CA_PRE[1] * fs)))
    post = (int(round(CA_POST[0] * fs)), int(round(CA_POST[1] * fs)))
    per = det.periods(**kw)
    on_b, off_b = _span_mask(det, per, "ON"), _span_mask(det, per, "OFF")
    ok_b = det.testable(**_core(kw))
    ev = acq.events.filter(pl.col("z_shot") >= kw["z_min"])
    row = {s: i for i, s in enumerate(acq.syn["syn_id"].to_list())}
    rng = np.random.default_rng(rng_seed)
    excl = int(round(0.4 * fs))
    nf = acq.time.size
    ok_f = ok_b[np.minimum(np.arange(nf) // det.fpb, det.n_bins - 1)]
    fr = np.arange(nf)
    frames = []
    for sid, g in ev.group_by("syn_id", maintain_order=True):
        i = row[sid[0]]
        idx = g["idx"].to_numpy()
        keep = ok_f[np.clip(idx, 0, nf - 1)]
        idx, amp = idx[keep], g["amp"].to_numpy()[keep]
        if idx.size < 5:
            continue
        x_ca = np.asarray(ca[i], dtype=np.float64)
        x_gl = np.asarray(glu[i], dtype=np.float64)
        cs_c, cn_c = _nan_cumsum(x_ca)
        cs_g, cn_g = _nan_cumsum(x_gl)
        pre_c = _win_mean(cs_c, cn_c, idx, *pre)
        pre_g = _win_mean(cs_g, cn_g, idx, *pre)
        e0 = np.clip(idx[:, None] + np.arange(*XTALK_LAGS)[None, :], 0, nf - 1)
        num = np.nansum(x_ca[e0] - pre_c[:, None])
        den = np.nansum(x_gl[e0] - pre_g[:, None])
        alpha = float(num / den) if den > 0 else np.nan

        def resp(at, pc, pg, cs_c=cs_c, cn_c=cn_c, cs_g=cs_g, cn_g=cn_g):
            rc = _win_mean(cs_c, cn_c, at, *post) - pc
            rg = _win_mean(cs_g, cn_g, at, *post) - pg
            return rc, rg

        rc, rg = resp(idx, pre_c, pre_g)
        # isolated = no other event of this synapse within ±0.4 s
        srt = np.sort(idx)
        lo_n = np.searchsorted(srt, idx - excl, side="left")
        hi_n = np.searchsorted(srt, idx + excl, side="right")
        iso = (hi_n - lo_n) == 1
        parts = [("event", idx, amp, rc, rg, iso)]
        # pseudo-events: same condition (ON / not ON), no own event within ±0.4 s
        has = np.zeros(nf + 1, dtype=np.int32)
        np.add.at(has, np.clip(idx, 0, nf - 1) + 1, 1)
        ce = np.cumsum(has)
        near = ce[np.minimum(fr + excl + 1, nf)] - ce[np.maximum(fr - excl, 0)] > 0
        cand = np.flatnonzero(~near & ok_f)
        cand_on = on_b[np.minimum(cand // det.fpb, det.n_bins - 1)]
        ev_on = on_b[np.minimum(idx // det.fpb, det.n_bins - 1)]
        p_idx = []
        for want_on in (True, False):
            pool = cand[cand_on == want_on]
            n_want = int((ev_on == want_on).sum()) * n_pseudo
            if pool.size and n_want:
                p_idx.append(rng.choice(pool, size=n_want, replace=pool.size < n_want))
        if p_idx:
            pi = np.sort(np.concatenate(p_idx))
            pc, pg = resp(
                pi, _win_mean(cs_c, cn_c, pi, *pre), _win_mean(cs_g, cn_g, pi, *pre)
            )
            parts.append((
                "pseudo",
                pi,
                np.full(pi.size, np.nan),
                pc,
                pg,
                np.ones(pi.size, dtype=bool),
            ))
        for kind, at, am, rc_, rg_, iso_ in parts:
            b = np.minimum(at // det.fpb, det.n_bins - 1)
            t = acq.time[np.clip(at, 0, nf - 1)]
            frames.append(
                pl.DataFrame({
                    "syn_id": sid[0],
                    "kind": kind,
                    "t": t,
                    "state": state_at(acq, t),
                    "third": third_of(det, t),
                    "in_on": on_b[b],
                    "in_off": off_b[b],
                    "glu_amp": am.astype(np.float64),
                    "ca_raw": rc_,
                    "glu": rg_,
                    "alpha": alpha,
                    "isolated": iso_,
                })
            )
    df = pl.concat(frames).with_columns(
        (pl.col("ca_raw") - pl.col("alpha") * pl.col("glu")).alias("ca")
    )
    return df.filter(pl.col("ca").is_finite())


def ca_trigger_average(
    det: Detector, half_s: float = 1.0, max_events: int = 4000, **params
) -> pl.DataFrame:
    """Event-triggered average of both channels around glutamate events.

    Calcium (channel 0) and glutamate (channel 1) LS dF/F0, baseline-subtracted
    over :data:`CA_PRE`, averaged within each synapse (up to ``max_events``
    events), then over synapses. Shows the calcium channel's crosstalk copy of
    the glutamate transient and the slower calcium-like remainder.
    """
    kw = _kw(params)
    acq = det.acq
    ca, glu = load_ca(acq), load_dff(acq, GLU_CHANNEL)
    fs = 1.0 / float(np.median(np.diff(acq.time)))
    h = int(round(half_s * fs))
    lags = np.arange(-h, h + 1)
    pre = slice(h + int(round(CA_PRE[0] * fs)), h + int(round(CA_PRE[1] * fs)))
    ok_b = det.testable(**_core(kw))
    ev = acq.events.filter(pl.col("z_shot") >= kw["z_min"])
    row = {s: i for i, s in enumerate(acq.syn["syn_id"].to_list())}
    acc: dict[str, list[np.ndarray]] = {"calcium": [], "glutamate": []}
    for sid, g in ev.group_by("syn_id", maintain_order=True):
        i = row[sid[0]]
        idx = g["idx"].to_numpy()
        idx = idx[(idx > h) & (idx < acq.time.size - h)]
        idx = idx[ok_b[np.minimum(idx // det.fpb, det.n_bins - 1)]][:max_events]
        if idx.size < 20:
            continue
        for name, arr in (("calcium", ca), ("glutamate", glu)):
            w = np.asarray(arr[i], dtype=np.float32)[idx[:, None] + lags[None, :]]
            w = w - np.nanmean(w[:, pre], axis=1, keepdims=True)
            acc[name].append(np.nanmean(w, axis=0))
    return pl.DataFrame({
        "lag_s": lags / fs,
        "calcium": np.nanmean(np.stack(acc["calcium"]), 0),
        "glutamate": np.nanmean(np.stack(acc["glutamate"]), 0),
    })
