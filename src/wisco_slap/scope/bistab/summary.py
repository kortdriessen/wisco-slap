"""Per-state summaries of ON/OFF periods, per-period measures and time trends."""

from __future__ import annotations

import numpy as np
import polars as pl

from .core import (
    BIN_S,
    DEFAULTS,
    STATE_ORDER,
    THIRDS,
    Detector,
    _core,
    _kw,
    _span_mask,
    refine_edges,
    state_at,
    third_of,
)


def summarize(det: Detector, shuffle: bool | int = False, **params) -> pl.DataFrame:
    """Periods per minute and mean duration by hypnogram state.

    ``params`` are detector settings (``z_min``, ``window_s``, ``baseline_s``,
    ``seed``, ``edge``; defaults from :data:`DEFAULTS`). The denominator is the
    *testable* time in each state (all synapses observable), not wall-clock
    time, so DMD dropouts do not deflate rates.
    """
    kw = {k: params.get(k, DEFAULTS[k]) for k in ("z_min", "window_s", "baseline_s")}
    periods = det.periods(
        **kw,
        seed=params.get("seed", DEFAULTS["seed"]),
        edge=params.get("edge", DEFAULTS["edge"]),
        shuffle=shuffle,
    )
    ok = det.testable(**kw, shuffle=shuffle)
    minutes = (
        pl
        .DataFrame({"state": state_at(det.acq, det.bin_t + det.dt / 2)[ok]})
        .group_by("state")
        .len()
        .with_columns((pl.col("len") * det.dt / 60).alias("testable_min"))
        .drop("len")
    )
    mid = ((periods["start"] + periods["end"]) / 2).to_numpy()
    by = (
        periods
        .with_columns(pl.Series("state", state_at(det.acq, mid)))
        .group_by("state", "label")
        .agg(pl.len().alias("n"), pl.col("duration").mean().alias("mean_dur_s"))
    )
    return (
        by
        .join(minutes, on="state")
        .with_columns((pl.col("n") / pl.col("testable_min")).alias("per_min"))
        .sort("state", "label")
    )


def state_table(
    det: Detector,
    shuffle: bool | int = False,
    refined: bool = False,
    margin_s: float = 0.5,
    **params,
) -> pl.DataFrame:
    """One row per state (+ ``all``): ON/OFF rate, mean duration and % of time.

    Wide companion of :func:`summarize` for the live Loupe table. Rates and
    time fractions use testable time only; a period belongs to the state at its
    midpoint. ``refined`` / ``margin_s``: use event-defined edges (see
    :meth:`Detector.display_periods`).
    """
    kw = {k: params.get(k, DEFAULTS[k]) for k in ("z_min", "window_s", "baseline_s")}
    per = det.display_periods(
        **kw,
        seed=params.get("seed", DEFAULTS["seed"]),
        edge=params.get("edge", DEFAULTS["edge"]),
        shuffle=shuffle,
        refined=refined,
        margin_s=margin_s,
    )
    ok = det.testable(**kw, shuffle=shuffle)
    st_bins = state_at(det.acq, det.bin_t + det.dt / 2)
    mins = {
        k: float(np.sum(ok & (st_bins == k)) * det.dt / 60) for k in np.unique(st_bins)
    }
    mins["all"] = float(ok.sum() * det.dt / 60)
    mid = ((per["start"] + per["end"]) / 2).to_numpy()
    per = per.with_columns(pl.Series("state", state_at(det.acq, mid)))
    rows = []
    states = [x for x in STATE_ORDER if mins.get(x, 0) > 0]
    states += [x for x in mins if x not in states and x != "all" and mins[x] > 0] + [
        "all"
    ]
    for state in states:
        sub = per if state == "all" else per.filter(pl.col("state") == state)
        row = {"state": state, "min": mins[state]}
        for label in ("ON", "OFF"):
            d = sub.filter(pl.col("label") == label)["duration"]
            row[f"{label} /min"] = d.len() / mins[state]
            row[f"{label} dur (s)"] = float(d.mean()) if d.len() else float("nan")
            row[f"{label} % time"] = 100 * float(d.sum()) / 60 / mins[state]
        rows.append(row)
    return pl.DataFrame(rows)


def testable_minutes(
    det: Detector, shuffle: bool | int = False, **params
) -> dict[str, float]:
    """Testable minutes per state (all synapses observable)."""
    ok = det.testable(**_core(_kw(params)), shuffle=shuffle)
    st = state_at(det.acq, det.bin_t + det.dt / 2)
    return {s: float(np.sum(ok & (st == s)) * det.dt / 60) for s in np.unique(st)}


def testable_by_bin(det: Detector, bin_min: float = 5.0, **params) -> pl.DataFrame:
    """Testable minutes per state in consecutive ``bin_min``-minute bins."""
    ok = det.testable(**_core(_kw(params)))
    mid = det.bin_t + det.dt / 2
    b = np.floor((mid - det.acq.time[0]) / 60 / bin_min).astype(int)
    st = state_at(det.acq, mid)
    df = pl.DataFrame({"bin": b[ok], "state": st[ok]})
    return (
        df
        .group_by("bin", "state")
        .len()
        .with_columns((pl.col("len") * det.dt / 60).alias("minutes"))
        .drop("len")
    )


def event_rates(det: Detector, far_s: float = 2.0, **params) -> pl.DataFrame:
    """Per-synapse glutamate event rate by state, third and period condition.

    Conditions (all within testable time):
    ``all``; ``no ON`` (ON periods removed); ``no OFF``; ``no ON/OFF`` (both
    removed); ``far`` (only time more than ``far_s`` from any ON or OFF period).
    Events are the detector's events (``z_shot >= z_min``). One row per
    synapse x state x third (incl. ``all``) x condition with ``events``,
    ``seconds`` and ``rate`` (Hz).
    """
    kw = _kw(params)
    per = det.periods(**kw)
    ok = det.testable(**_core(kw))
    on, off = _span_mask(det, per, "ON"), _span_mask(det, per, "OFF")
    near = _span_mask(det, per, None, pad_s=far_s)
    conds = {
        "all": ok,
        "no ON": ok & ~on,
        "no OFF": ok & ~off,
        "no ON/OFF": ok & ~on & ~off,
        "far": ok & ~near,
    }
    mid = det.bin_t + det.dt / 2
    state, third = state_at(det.acq, mid), third_of(det, mid)
    keys, cols = [], []
    for s in np.unique(state):
        for th in ("all",) + THIRDS:
            base = (state == s) & ((third == th) if th != "all" else True)
            for cname, cm in conds.items():
                keys.append((s, th, cname))
                cols.append(base & cm)
    M = np.stack(cols, axis=1).astype(np.float32)  # n_bins x n_masks
    ev = det.counts(kw["z_min"]) @ M  # n_syn x n_masks
    secs = M.sum(0) * det.dt
    ids = det.acq.syn["syn_id"].to_list()
    rows = []
    for j, (s, th, cname) in enumerate(keys):
        if secs[j] <= 0:
            continue
        for i, sid in enumerate(ids):
            rows.append((sid, s, th, cname, float(ev[i, j]), float(secs[j])))
    df = pl.DataFrame(
        rows,
        schema=["syn_id", "state", "third", "condition", "events", "seconds"],
        orient="row",
    )
    return df.with_columns((pl.col("events") / pl.col("seconds")).alias("rate"))


def period_stats(det: Detector, margin_s: float = 0.5, **params) -> pl.DataFrame:
    """One row per ON/OFF period with timing, size and depth measures.

    Edges are the refined (event-based) ones from :func:`refine_edges`; a
    side whose fit hit the search margin keeps its detector edge.
    Columns: ``label``, ``state``, ``t_min`` (minutes from the start of the
    analysis window, at the period's middle), ``duration`` (s), ``depth``
    (population events ÷ independent-synapse expectation inside the period:
    > 1 for ONs, < 1 for OFFs), ``peak_ratio`` (N/E of the period's most
    surprising detector window: peak for ONs, trough for OFFs), ``peak`` (that
    window's −log10 p), ``n_active`` / ``frac_active`` (synapses with >= 1 event
    inside the period, count and fraction of the cell's synapses).
    """
    kw = _kw(params)
    per = det.periods(**kw).select("start", "label", "peak", "ratio")
    rf = refine_edges(det, margin_s, **kw)
    rf = rf.with_columns(
        pl
        .when(pl.col("left_margin"))
        .then(pl.col("start"))
        .otherwise(pl.col("r_start"))
        .alias("s"),
        pl
        .when(pl.col("right_margin"))
        .then(pl.col("end"))
        .otherwise(pl.col("r_end"))
        .alias("e"),
    ).join(per, on=["start", "label"], how="left")
    t0 = det.bin_t[0]
    b0 = np.clip(
        np.floor((rf["s"].to_numpy() - t0) / det.dt).astype(int), 0, det.n_bins
    )
    b1 = np.clip(np.ceil((rf["e"].to_numpy() - t0) / det.dt).astype(int), 0, det.n_bins)
    c = det.counts(kw["z_min"])
    cs = np.zeros((c.shape[0], c.shape[1] + 1), dtype=np.float32)
    np.cumsum(c, axis=1, out=cs[:, 1:])
    n_active = ((cs[:, b1] - cs[:, b0]) > 0).sum(0)
    st = det.stats(kw["z_min"], BIN_S, kw["baseline_s"])
    cn = np.r_[0.0, np.cumsum(st["N"])]
    ce = np.r_[0.0, np.cumsum(st["E"])]
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = (cn[b1] - cn[b0]) / (ce[b1] - ce[b0])
    mid = (rf["s"].to_numpy() + rf["e"].to_numpy()) / 2
    return pl.DataFrame({
        "label": rf["label"],
        "state": state_at(det.acq, mid),
        "t_min": (mid - det.acq.time[0]) / 60,
        "duration": rf["e"].to_numpy() - rf["s"].to_numpy(),
        "depth": depth,
        "peak_ratio": rf["ratio"],
        "peak": rf["peak"],
        "n_active": n_active,
        "frac_active": n_active / det.acq.n_syn,
    })


#: One-at-a-time detector variations around the defaults, for sensitivity checks.
SENSITIVITY = {
    "z_min": [3.5, 5.0],
    "window_s": [0.10, 0.25],
    "baseline_s": [10.0, 40.0],
    "seed": [2.5, 5.0],
}
TREND_METRICS = ["freq", "duration", "depth", "peak_ratio", "frac_active"]


def time_trends(
    ps: pl.DataFrame,
    tb: pl.DataFrame,
    bin_min: float = 5.0,
    states: tuple[str, ...] = ("NREM SWS", "NREM sigma"),
    min_minutes: float = 0.5,
    min_periods: int = 5,
) -> pl.DataFrame:
    """Per-acquisition linear trend (% change per 10 min) of ON/OFF measures.

    ``ps`` / ``tb`` are cohort results of :func:`period_stats` and
    :func:`testable_by_bin` (same ``bin_min``; both need an ``acq`` column, as
    :func:`~.cohort.cohort` adds). Per acquisition, state, label and bin:
    frequency (periods per testable second) and the median of each per-period
    measure; each acquisition is divided by its own mean across bins, and a
    straight line is fitted against time. Rows: ``state``, ``label``,
    ``metric``, ``acq``, ``pct_per_10min``.
    """
    ps = ps.filter(pl.col("state").is_in(states)).with_columns(
        (pl.col("t_min") // bin_min).cast(pl.Int64).alias("bin")
    )
    per_bin = (
        ps
        .group_by("acq", "state", "label", "bin")
        .agg(
            pl.len().alias("n"),
            pl.col("duration", "depth", "peak_ratio", "frac_active").median(),
        )
        .join(tb, on=["acq", "state", "bin"])
        .filter(pl.col("minutes") >= min_minutes)
        .with_columns((pl.col("n") / pl.col("minutes") / 60).alias("freq"))
    )
    long = pl.concat([
        per_bin.filter((pl.col("n") >= min_periods) | pl.lit(m == "freq")).select(
            "acq",
            "state",
            "label",
            "bin",
            pl.lit(m).alias("metric"),
            pl.col(m).cast(pl.Float64).alias("value"),
        )
        for m in TREND_METRICS
    ]).with_columns(
        (
            pl.col("value")
            / pl.col("value").mean().over("acq", "state", "label", "metric")
        ).alias("rel"),
        ((pl.col("bin") + 0.5) * bin_min).alias("t"),
    )
    rows = []
    for (s, lab, m, a), g in long.group_by("state", "label", "metric", "acq"):
        if g.height >= 3 and np.isfinite(g["rel"].to_numpy()).all():
            slope = np.polyfit(g["t"].to_numpy(), g["rel"].to_numpy(), 1)[0]
            rows.append(
                dict(
                    state=s,
                    label=lab,
                    metric=m,
                    acq=a,
                    pct_per_10min=float(slope * 1000),
                )
            )
    return pl.DataFrame(rows)
