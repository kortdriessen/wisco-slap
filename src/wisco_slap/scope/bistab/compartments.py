"""Dendrites and the cell's ON/OFF periods: participation, local detection, depression.

Comparisons across dendrites of very different sizes are made either against
the rest of the cell (leave-one-dendrite-out anchors, so a dendrite cannot vote
for its own periods) or at matched evidence (event trains thinned to a common
rate), always with size-matched random synapse groups as the control.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .core import (
    BIN_S,
    DEFAULTS,
    AcqData,
    Detector,
    _box,
    _core,
    _intervals,
    _kw,
    _overlaps,
    state_at,
    third_of,
)


def dendrite_groups(acq: AcqData, min_syn: int = 8) -> dict[str, np.ndarray]:
    """``{dend-ID: synapse mask}`` for labelled dendrites with >= ``min_syn`` synapses."""
    ids = np.array(acq.syn["dend-ID"].to_list(), dtype=object).astype(str)
    out = {}
    for d in sorted(set(ids)):
        if d.lower() in ("nan", "none", "") or (ids == d).sum() < min_syn:
            continue
        out[d] = ids == d
    return out


def participation(
    acq: AcqData, min_syn: int = 8, n_random: int = 0, rng_seed: int = 0, **params
) -> pl.DataFrame:
    """How strongly does each dendrite take part in the cell's ON and OFF periods?

    For every dendrite with >= ``min_syn`` synapses: detect ON/OFF periods on all
    *other* synapses of the cell (leave-one-dendrite-out, so the dendrite cannot
    vote for its own periods), then measure the dendrite's own event count ``N``
    against its own independent-synapse expectation ``E`` (each synapse's local
    rate over ``baseline_s``) inside those periods. ``ratio = N / E``: 1 = the
    dendrite behaves as usual, 0 = completely silent, > 1 = more active than
    usual. Also reported for testable time outside any period (``label = "none"``).
    Rows: dendrite x state x label with ``N``, ``E``, ``seconds`` and ``ratio``.

    ``n_random`` adds, per dendrite, that many *size-matched random* synapse
    sets drawn from the cell's other synapses (``kind = "random"``, ``dend`` =
    the dendrite they are matched to; capped at the number of other synapses
    when a dendrite holds over half the cell), analysed identically: the comparison
    tells whether a real dendrite is more (or less) tied to the cell's periods
    than an anatomy-blind group of the same size.
    """
    kw = _kw(params)
    rows = []
    rng = np.random.default_rng(rng_seed)
    dgroups = dendrite_groups(acq, min_syn)
    groups = []
    for dend, mask in dgroups.items():
        groups.append((dend, "dendrite", mask))
        for _ in range(n_random):
            others = np.flatnonzero(~mask)
            # a dendrite holding over half the cell gets the largest possible group
            pick = rng.choice(
                others, size=min(int(mask.sum()), others.size), replace=False
            )
            rmask = np.zeros_like(mask)
            rmask[pick] = True
            groups.append((dend, "random", rmask))
    for dend, kind, mask in groups:
        anchor = Detector(acq.subset(~mask, f"-without-{dend}"))
        per = anchor.periods(**kw)
        a_ok = anchor.testable(**_core(kw))
        comp = Detector(acq.subset(mask, f"-{dend}"))
        st = comp.stats(kw["z_min"], BIN_S, kw["baseline_s"])
        ok = a_ok & (st["frac_valid"] >= 1.0 - 1e-9)
        lab = np.zeros(comp.n_bins, dtype=np.int8)  # 0 none, 1 ON, -1 OFF
        for s_, e_, l_ in per.select("start", "end", "label").iter_rows():
            b0 = int(np.floor((s_ - comp.bin_t[0]) / comp.dt))
            b1 = int(np.ceil((e_ - comp.bin_t[0]) / comp.dt))
            lab[max(b0, 0) : min(b1, comp.n_bins)] = 1 if l_ == "ON" else -1
        state = state_at(acq, comp.bin_t + comp.dt / 2)
        dtype = acq.syn.filter(pl.Series(dgroups[dend]))["dend_type"][0]
        for s in np.unique(state):
            for name, code in (("ON", 1), ("OFF", -1), ("none", 0)):
                m = ok & (state == s) & (lab == code)
                if not m.any():
                    continue
                N, E = float(st["N"][m].sum()), float(st["E"][m].sum())
                rows.append(
                    dict(
                        dend=dend,
                        kind=kind,
                        dend_type=str(dtype),
                        n_syn=int(mask.sum()),
                        state=s,
                        label=name,
                        N=N,
                        E=E,
                        seconds=float(m.sum() * comp.dt),
                        ratio=N / E if E > 0 else np.nan,
                    )
                )
        del anchor, comp
    return pl.DataFrame(rows)


def group_rate(det: Detector, z_min: float) -> float:
    """Total event rate (Hz) of the detector's synapses in testable time."""
    ok = det.testable(z_min, DEFAULTS["window_s"], DEFAULTS["baseline_s"])
    return float((det.counts(z_min).sum(0) * ok).sum() / (ok.sum() * det.dt))


def local_detection(
    acq: AcqData,
    rate_target: float = 15.0,
    window_s: float = 0.5,
    min_syn: int = 8,
    n_random: int = 3,
    rng_seed: int = 0,
    **params,
) -> pl.DataFrame:
    """Single-dendrite ON/OFF detection at matched evidence.

    Every dendrite with a total event rate >= ``rate_target`` is thinned (events
    kept at random) down to exactly ``rate_target`` events/s, so all dendrites
    and controls give the detector the same amount of evidence, and its periods
    are detected with a ``window_s`` window (long enough for silences to be
    detectable at that rate). Each dendrite period is compared with the periods
    of the *rest of the cell* (full data, default settings): ``shared`` if it
    overlaps a rest period of the same type, ``opposite`` if it overlaps one of
    the other type only, otherwise ``local``.

    Note: ``window_s`` here is the *local* (thinned-dendrite) window; the
    rest-of-cell anchors use the detector settings in ``params``.

    Controls per dendrite: ``random`` = ``n_random`` size-matched random synapse
    groups from the rest of the cell, thinned and analysed identically;
    ``shuffle`` = the dendrite's own thinned synapses with their trains shifted
    independently (false alarms). One row per group x state x label with counts
    of shared / opposite / local periods and testable minutes.
    """
    kw = _kw(params)
    loc_kw = {**kw, "window_s": window_s}
    rng = np.random.default_rng(rng_seed)
    rows = []
    groups = dendrite_groups(acq, min_syn)
    for dend, mask in groups.items():
        sets = [("dendrite", mask)]
        for _ in range(n_random):
            others = np.flatnonzero(~mask)
            pick = rng.choice(
                others, size=min(int(mask.sum()), others.size), replace=False
            )
            rm = np.zeros_like(mask)
            rm[pick] = True
            sets.append(("random", rm))
        for kind, m in sets:
            sub = acq.subset(m, f"-{dend}-{kind}")
            rate = group_rate(Detector(sub), kw["z_min"])
            if kind == "dendrite" and rate < rate_target:
                break  # dendrite too quiet for this evidence level: skip it and its controls
            if rate < rate_target:
                continue
            thin = Detector(sub.thin(rate_target / rate, int(rng.integers(1 << 30))))
            rest = Detector(acq.subset(~m, f"-without-{dend}-{kind}"))
            rper = rest.periods(**kw)
            r_ok = rest.testable(**_core(kw))
            runs = [(0, kind)] + ([(1, "shuffle")] if kind == "dendrite" else [])
            for k_shuf, label_kind in runs:
                lper = thin.periods(**loc_kw, shuffle=k_shuf)
                ok = thin.testable(**_core(loc_kw), shuffle=k_shuf) & r_ok
                mid = thin.bin_t + thin.dt / 2
                st_b = state_at(acq, mid)
                mins = {
                    s: float((ok & (st_b == s)).sum() * thin.dt / 60)
                    for s in np.unique(st_b)
                }
                for lab in ("ON", "OFF"):
                    ls_, le_ = _intervals(lper, lab)
                    same_s, same_e = _intervals(rper, lab)
                    opp_s, opp_e = _intervals(rper, "OFF" if lab == "ON" else "ON")
                    shared = _overlaps(ls_, le_, same_s, same_e)
                    opp = ~shared & _overlaps(ls_, le_, opp_s, opp_e)
                    st = state_at(acq, (ls_ + le_) / 2)
                    for s, m_ in mins.items():
                        sel = st == s
                        rows.append(
                            dict(
                                dend=dend,
                                kind=label_kind,
                                n_syn=int(m.sum()),
                                rate=rate,
                                state=s,
                                label=lab,
                                minutes=m_,
                                n=int(sel.sum()),
                                shared=int((shared & sel).sum()),
                                opposite=int((opp & sel).sum()),
                                local=int((~shared & ~opp & sel).sum()),
                            )
                        )
            del thin, rest
    return pl.DataFrame(rows)


def depression_check(
    det: Detector, gap_s: float = 0.5, ctrl_s: float = 0.3, **params
) -> pl.DataFrame:
    """Are synapses that did *not* fire in an ON also silent in the OFF after it?

    For each OFF that starts within ``gap_s`` after an ON ends, the cell's
    synapses are split into ``participants`` (>= 1 event during that ON) and
    ``non-participants``. Each group's event count during the OFF is compared
    with its own expectation (every synapse's local rate). If post-burst
    silence were vesicle depletion at the synapses that just released, only
    participants would be silent.

    Control (``where = "after ON, no OFF"``): ONs *not* followed by an OFF; the
    same split is measured in the ``ctrl_s`` right after the ON. It shows how
    participants and non-participants differ after a burst simply because
    participants were the ones active at that moment (a selection effect that
    works against finding depression).
    """
    kw = _kw(params)
    per = det.periods(**kw)
    c = det.counts(kw["z_min"])
    kb = max(2, int(round(kw["baseline_s"] / det.dt)))
    v = det.valid.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = _box(c, kb) / _box(v, kb)
    mu = np.nan_to_num(mu) * v
    t0 = det.bin_t[0]

    def bins(s_, e_):
        return (
            max(int(np.floor((s_ - t0) / det.dt)), 0),
            min(int(np.ceil((e_ - t0) / det.dt)), det.n_bins),
        )

    on_s, on_e = _intervals(per, "ON")
    off_s, off_e = _intervals(per, "OFF")
    j = np.searchsorted(off_s, on_e, side="left")
    rows = []
    for k in range(on_s.size):
        a0, a1 = bins(on_s[k], on_e[k])
        part = c[:, a0:a1].sum(1) > 0
        if part.all() or not part.any():
            continue
        if j[k] < off_s.size and off_s[j[k]] - on_e[k] <= gap_s:
            b0, b1 = bins(off_s[j[k]], off_e[j[k]])
            where = "OFF after ON"
        else:
            b0, b1 = bins(on_e[k], on_e[k] + ctrl_s)
            where = "after ON, no OFF"
        if b1 <= b0:
            continue
        n = c[:, b0:b1].sum(1)
        e = mu[:, b0:b1].sum(1)
        tmid = t0 + (b0 + b1) / 2 * det.dt
        rows.append(
            dict(
                where=where,
                state=str(state_at(det.acq, np.array([tmid]))[0]),
                third=str(third_of(det, np.array([tmid]))[0]),
                n_part=int(part.sum()),
                N_part=float(n[part].sum()),
                E_part=float(e[part].sum()),
                N_non=float(n[~part].sum()),
                E_non=float(e[~part].sum()),
            )
        )
    return pl.DataFrame(rows)
