"""Null models: independent-synapse shuffles, ON↔OFF alternation, chains, episode coupling.

Every test here compares the real periods with a null that keeps what the
detector itself constrains (number of periods, durations, testable time, no
same-type overlap) and scrambles only the structure being asked about.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .core import (
    Detector,
    _core,
    _intervals,
    _kw,
    _overlaps,
    _Testable,
    state_at,
)
from .summary import testable_minutes


def shuffle_rates(det: Detector, n_shuffles: int = 5, **params) -> pl.DataFrame:
    """ON/OFF rate per state: real data and ``n_shuffles`` independent-synapse shuffles.

    Long format: ``state``, ``label``, ``kind`` ("real" / "shuffle"), ``k``
    (0 = real, 1..n = shuffle realisation), ``per_min``, ``minutes``.
    """
    kw = _kw(params)
    rows = []
    for k in range(n_shuffles + 1):
        per = det.periods(**kw, shuffle=k)
        mins = testable_minutes(det, shuffle=k, **kw)
        st = state_at(det.acq, ((per["start"] + per["end"]) / 2).to_numpy())
        for state, m in mins.items():
            for label in ("ON", "OFF"):
                n = int(np.sum((st == state) & (per["label"].to_numpy() == label)))
                rows.append(
                    dict(
                        state=state,
                        label=label,
                        kind="real" if k == 0 else "shuffle",
                        k=k,
                        per_min=n / m if m > 0 else np.nan,
                        minutes=m,
                    )
                )
    return pl.DataFrame(rows)


def _jitter(s, e, block_s, block_e, testable, jitter_s, rng, rounds=60):
    """Move each span by U(-jitter_s, jitter_s) under the detector's own constraints.

    A moved span may not overlap a ``block`` span (the other period type), must
    lie in testable time, and may not overlap another moved span of its own type
    (real same-type periods never overlap: the detector merges them). Spans are
    redrawn until they fit; the few that never fit keep their real position.
    """
    s2, e2 = s.copy(), e.copy()
    todo = np.ones(s.size, dtype=bool)
    for _ in range(rounds):
        if todo.any():
            d = rng.uniform(-jitter_s, jitter_s, todo.sum())
            cs, ce = s[todo] + d, e[todo] + d
            good = ~_overlaps(cs, ce, block_s, block_e) & testable(cs, ce)
            idx = np.flatnonzero(todo)[good]
            s2[idx], e2[idx] = cs[good], ce[good]
            todo[idx] = False
        # same-type overlaps among placed spans: redraw the later one of each pair
        o = np.argsort(s2)
        clash = np.flatnonzero(e2[o][:-1] > s2[o][1:]) + 1
        clash = o[clash]
        clash = clash[~todo[clash]]
        if clash.size == 0 and not todo.any():
            break
        todo[clash] = True
    stuck = todo
    s2[stuck], e2[stuck] = s[stuck], e[stuck]
    return s2, e2


def _follow_frac(xe: np.ndarray, ys: np.ndarray, gap_s: float) -> np.ndarray:
    """For each X end, does a Y start within [end, end + gap_s]?

    ``side="left"`` matters: ON and OFF periods that meet are split at a shared
    time point, so the follower can start exactly where the leader ends.
    """
    ys = np.sort(ys)
    i = np.searchsorted(ys, xe, side="left")
    ok = i < ys.size
    out = np.zeros(xe.size, dtype=bool)
    out[ok] = ys[i[ok]] <= xe[ok] + gap_s
    return out


def alternation_test(
    det: Detector,
    gap_s: float = 0.5,
    jitter_s: float = 5.0,
    n_null: int = 200,
    rng_seed: int = 0,
    **params,
) -> pl.DataFrame:
    """Do OFFs start right after ONs end (and vice versa) more often than chance?

    For each ON (OFF) period: does an OFF (ON) period start within ``gap_s``
    after it ends? The observed fraction is compared with ``n_null`` jittered
    versions in which every *following-type* period is moved by a random
    ±``jitter_s`` shift. A moved period may not overlap a period of the other
    type and must stay in testable time, so the null keeps how many periods
    there are, their durations and their slow (± a few s) rate changes, and only
    scrambles their fine placement. Rows: one per state x direction, with the
    number of anchor periods, the observed fraction, the null mean and its
    2.5 / 97.5 % range.
    """
    kw = _kw(params)
    per = det.periods(**kw)
    testable = _Testable(det, kw)
    rng = np.random.default_rng(rng_seed)
    rows = []
    for src, dst in (("ON", "OFF"), ("OFF", "ON")):
        xs, xe = _intervals(per, src)
        ys, ye = _intervals(per, dst)
        x_state = state_at(det.acq, (xs + xe) / 2)
        obs = _follow_frac(xe, ys, gap_s)
        null = np.empty((n_null, xe.size), dtype=bool)
        for k in range(n_null):
            js, _ = _jitter(ys, ye, xs, xe, testable, jitter_s, rng)
            null[k] = _follow_frac(xe, js, gap_s)
        for state in np.unique(x_state):
            m = x_state == state
            nm = null[:, m].mean(1)
            rows.append(
                dict(
                    state=state,
                    direction=f"{src}→{dst}",
                    n=int(m.sum()),
                    obs=float(obs[m].mean()),
                    null=float(nm.mean()),
                    null_lo=float(np.quantile(nm, 0.025)),
                    null_hi=float(np.quantile(nm, 0.975)),
                )
            )
    return pl.DataFrame(rows)


def chains(
    starts: np.ndarray, ends: np.ndarray, labels: np.ndarray, gap_s: float
) -> pl.DataFrame:
    """Runs of strictly alternating periods (ON/OFF/ON/...), each gap <= ``gap_s``.

    Returns one row per chain: ``start``, ``end``, ``n`` (periods in the chain).
    Isolated periods are chains of length 1.
    """
    o = np.argsort(starts, kind="stable")
    s, e, lab = starts[o], ends[o], np.asarray(labels)[o]
    if s.size == 0:
        return pl.DataFrame(schema=dict(start=pl.Float64, end=pl.Float64, n=pl.Int64))
    link = (lab[1:] != lab[:-1]) & (s[1:] - e[:-1] <= gap_s)
    brk = np.flatnonzero(~link) + 1
    first = np.r_[0, brk]
    last = np.r_[brk - 1, s.size - 1]
    return pl.DataFrame({
        "start": s[first],
        "end": np.maximum.reduceat(e, first),
        "n": (last - first + 1).astype(np.int64),
    })


def jitter_periods(
    det: Detector,
    n: int = 50,
    move: str = "OFF",
    jitter_s: float = 5.0,
    rng_seed: int = 0,
    periods: pl.DataFrame | None = None,
    **params,
) -> pl.DataFrame:
    """``n`` null versions of the periods with every ``move``-type period jittered.

    The null of :func:`alternation_test` / :func:`chain_test`, exposed so that any
    sequence statistic can be computed on it: each ``move`` period is shifted by
    U(±``jitter_s``) without overlapping a period of the other type or another
    moved period, and staying in testable time; the other type is copied
    unchanged into every realisation. Number, durations and slow (± a few s)
    placement of the periods are kept; only their fine ordering is scrambled.

    ``periods``: the periods to jitter (default ``det.periods(**params)``).
    Long format: ``rep`` (1..n), ``label``, ``start``, ``end``, ``duration``.
    """
    if move not in ("ON", "OFF"):
        raise ValueError(f"move must be 'ON' or 'OFF', got {move!r}")
    kw = _kw(params)
    per = det.periods(**kw) if periods is None else periods
    testable = _Testable(det, kw)
    rng = np.random.default_rng(rng_seed)
    stay = "ON" if move == "OFF" else "OFF"
    ss, se = _intervals(per, stay)
    ms, me = _intervals(per, move)
    labels = [stay] * ss.size + [move] * ms.size
    parts = []
    for k in range(1, n + 1):
        js, je = _jitter(ms, me, ss, se, testable, jitter_s, rng)
        parts.append(
            pl.DataFrame({
                "rep": np.full(len(labels), k, dtype=np.int64),
                "label": labels,
                "start": np.r_[ss, js].astype(float),
                "end": np.r_[se, je].astype(float),
            })
        )
    if not parts:
        return pl.DataFrame(
            schema=dict(rep=pl.Int64, label=pl.String, start=pl.Float64, end=pl.Float64, duration=pl.Float64)
        )
    return pl.concat(parts).with_columns((pl.col("end") - pl.col("start")).alias("duration"))


def chain_test(
    det: Detector,
    gap_s: float = 0.5,
    min_len: int = 4,
    jitter_s: float = 5.0,
    n_null: int = 100,
    rng_seed: int = 0,
    **params,
) -> pl.DataFrame:
    """Sustained alternation: chains of >= ``min_len`` alternating periods per minute.

    Observed chain rate per state (state at the chain's midpoint, per testable
    minute) versus ``n_null`` versions in which the OFF periods are jittered
    (as in :func:`alternation_test`) while ON periods stay put.
    """
    kw = _kw(params)
    per = det.periods(**kw)
    testable = _Testable(det, kw)
    mins = testable_minutes(det, **kw)
    rng = np.random.default_rng(rng_seed)
    on_s, on_e = _intervals(per, "ON")
    off_s, off_e = _intervals(per, "OFF")
    lab = np.r_[np.full(on_s.size, "ON"), np.full(off_s.size, "OFF")]

    def rate(off_s_, off_e_) -> dict[str, float]:
        ch = chains(np.r_[on_s, off_s_], np.r_[on_e, off_e_], lab, gap_s).filter(
            pl.col("n") >= min_len
        )
        st = state_at(det.acq, ((ch["start"] + ch["end"]) / 2).to_numpy())
        return {s: float(np.sum(st == s)) / m for s, m in mins.items() if m > 0}

    obs = rate(off_s, off_e)
    null = [
        rate(*_jitter(off_s, off_e, on_s, on_e, testable, jitter_s, rng))
        for _ in range(n_null)
    ]
    return pl.DataFrame([
        dict(
            state=s,
            minutes=mins[s],
            obs_per_min=obs[s],
            null_per_min=float(np.mean([n[s] for n in null])),
            null_lo=float(np.quantile([n[s] for n in null], 0.025)),
            null_hi=float(np.quantile([n[s] for n in null], 0.975)),
        )
        for s in obs
    ])


def episode_coupling(
    det: Detector, chunk_s: float = 5.0, n_perm: int = 1000, rng_seed: int = 0, **params
) -> pl.DataFrame:
    """Do ONs and OFFs come in the *same* few-second stretches?

    Testable time is cut into ``chunk_s`` chunks that lie in a single state and
    are >= 90 % testable. Per chunk: ON and OFF rate (periods per testable
    second, by period midpoint). Per state: Spearman correlation across chunks
    between ON and OFF rate, with a permutation null (OFF rates shuffled across
    the state's chunks). Positive = ON-rich stretches are also OFF-rich
    (episodes of bistability); negative = ONs and OFFs live in different
    stretches (busy vs quiet stretches); ~0 = unrelated.

    A chunk with more events overall gives the detector more evidence for both
    kinds of period, so the correlation is computed after regressing each
    chunk's total event rate (ranks) out of both ON and OFF rates (a partial
    Spearman correlation); ``rho_raw`` is the plain one.
    """
    from scipy.stats import rankdata

    kw = _kw(params)
    per = det.periods(**kw)
    ok = det.testable(**_core(kw))
    kb = int(round(chunk_s / det.dt))
    nchunk = det.n_bins // kb
    okc = ok[: nchunk * kb].reshape(nchunk, kb)
    st_bins = state_at(det.acq, det.bin_t[: nchunk * kb] + det.dt / 2).reshape(
        nchunk, kb
    )
    one_state = (st_bins == st_bins[:, :1]).all(1)
    use = one_state & (okc.mean(1) >= 0.9)
    mid = ((per["start"] + per["end"]) / 2).to_numpy()
    ci = np.floor((mid - det.bin_t[0]) / (kb * det.dt)).astype(int)
    # all counted events (z >= z_min) per chunk, testable bins only
    c = det.counts(kw["z_min"]).sum(0) * ok
    ev_chunk = np.repeat(np.arange(nchunk), kb)
    ev_chunk = np.repeat(ev_chunk, c[: nchunk * kb].astype(int))
    rows, rng = [], np.random.default_rng(rng_seed)
    for state in np.unique(st_bins[use, 0]):
        cm = use & (st_bins[:, 0] == state)
        secs = okc[cm].sum(1) * det.dt
        rates = {}
        for lab in ("ON", "OFF"):
            sel = (per["label"].to_numpy() == lab) & (ci >= 0) & (ci < nchunk)
            cnt = np.bincount(ci[sel], minlength=nchunk)[cm]
            rates[lab] = cnt / secs
        if cm.sum() < 10:
            continue
        ev = np.bincount(ev_chunk, minlength=nchunk)[cm] / secs
        r = rankdata(ev) - rankdata(ev).mean()

        def resid(x: np.ndarray, r: np.ndarray = r) -> np.ndarray:
            x = rankdata(x) - rankdata(x).mean()
            return x - r * (x @ r) / (r @ r) if r @ r > 0 else x

        a, b = resid(rates["ON"]), resid(rates["OFF"])
        rho = float(np.corrcoef(a, b)[0, 1])
        rho_raw = float(
            np.corrcoef(rankdata(rates["ON"]), rankdata(rates["OFF"]))[0, 1]
        )
        null = np.array([
            np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n_perm)
        ])
        rows.append(
            dict(
                state=state,
                n_chunks=int(cm.sum()),
                rho=rho,
                rho_raw=rho_raw,
                null_lo=float(np.nanquantile(null, 0.025)),
                null_hi=float(np.nanquantile(null, 0.975)),
            )
        )
    return pl.DataFrame(rows)
