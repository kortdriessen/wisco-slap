"""Core of :mod:`wisco_slap.scope.bistab`: the data container, the detector, edge refinement.

Everything else in the package builds on three objects defined here:

- :class:`AcqData` — one soma's glutamate events, observability mask and
  hypnogram over an analysis window (built by :func:`~.data.load_acq`).
- :class:`Detector` — the population ON/OFF detector on one :class:`AcqData`,
  with a per-parameter result cache so live tuning (Loupe) is cheap.
- :func:`refine_edges` — event-based (Poisson change-point) period edges.

Plus the small shared helpers the analysis modules import (``state_at``,
``third_of``, ``_kw``, ``_intervals`` ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr
from scipy import stats

#: Scopex channel of iGluSnFR4f glutamate (ch0 is jRGECO1a calcium).
GLU_CHANNEL = 1
#: Scopex channel of jRGECO1a calcium.
CA_CHANNEL = 0
#: Pursuit template decay (s) used to build the event catalog.
GLU_TAU_S = 0.026
#: Detector resolution (s): 2 frames at 200 Hz.
BIN_S = 0.01

#: Detector defaults (tune live in Loupe with :func:`~.viz.build_view`).
DEFAULTS: dict[str, float] = dict(
    z_min=4.0,  # which events count (pursuit z_shot)
    window_s=0.15,  # length of the sliding counting window
    baseline_s=20.0,  # how far around each moment "typical" activity is taken from
    seed=3.5,  # -log10 p a window must reach to start a period
    edge=1.5,  # -log10 p a window must keep to extend a period
    # Fraction of the soma's synapses (both DMDs) whose whole window must be
    # observable. 1.0 = every synapse; a DMD dropout or trial gap then makes the
    # moment untestable instead of changing which synapses are being counted.
    min_valid=1.0,
)

#: 5-state hypnogram order used by tables and plots.
STATE_ORDER = ["Quiet wake", "Active wake", "NREM SWS", "NREM sigma", "REM"]
STATE_SHORT = {
    "Quiet wake": "QW",
    "Active wake": "AW",
    "NREM SWS": "SWS",
    "NREM sigma": "Sigma",
    "REM": "REM",
}
THIRDS = ("early", "middle", "late")


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class AcqData:
    """One soma's inputs over the analysis window (all times in ephys seconds).

    Normally built by :func:`wisco_slap.scope.bistab.load_acq`; can also be
    constructed directly (e.g. from synthetic or pre-extracted event trains).
    """

    key: str
    ids: tuple[str, str, str, str, str]  # subject, exp, loc, acq, soma
    time: np.ndarray  # frame times (n_frames,)
    syn: pl.DataFrame  # per-synapse metadata, row order = array row order
    events: (
        pl.DataFrame
    )  # pursuit events of these synapses: syn_id, idx, time, z_shot, amp
    detectable: np.ndarray  # (n_syn, n_frames) bool: an event could be detected
    mf_snr: np.ndarray | None  # (n_syn, n_frames) float16 matched-filter SNR (display)
    dn_sum: np.ndarray  # (n_frames,) summed denoised glutamate, sigma units
    hypno: pl.DataFrame  # start_s / end_s / label, 5-state model hypnogram
    info: dict = field(default_factory=dict)
    #: Where per-acquisition side caches (e.g. dF/F0 arrays) are kept; ``None`` = no caching.
    cache_dir: Path | None = None

    @property
    def n_syn(self) -> int:
        return self.syn.height

    @property
    def span_s(self) -> float:
        """Length of the analysis window (s).

        The requested duration when the acquisition was cut to one (and the
        data reach that far), otherwise the span of the frame times.
        """
        dt = float(np.median(np.diff(self.time))) if self.time.size > 1 else 0.0
        actual = float(self.time[-1] - self.time[0]) + dt
        d = self.info.get("duration_s", self.info.get("window_s"))
        if d is not None and float(d) <= actual + dt:
            return float(d)
        return actual

    def subset(self, keep: np.ndarray, tag: str = "") -> AcqData:
        """The same acquisition restricted to synapses where ``keep`` is True.

        Analysis inputs (``syn``, ``events``, ``detectable``) are restricted; the
        display-only ``mf_snr`` is dropped (``None``) and ``dn_sum`` is kept as is,
        so a subset can be fed to :class:`Detector` but not to
        :func:`~.viz.build_view`. Subsets never write side caches.
        """
        keep = np.asarray(keep, dtype=bool)
        syn = self.syn.filter(pl.Series(keep))
        return AcqData(
            key=f"{self.key}{tag}",
            ids=self.ids,
            time=self.time,
            syn=syn,
            events=self.events.filter(pl.col("syn_id").is_in(syn["syn_id"].to_list())),
            detectable=self.detectable[keep],
            mf_snr=None,
            dn_sum=self.dn_sum,
            hypno=self.hypno,
            info={**self.info, "subset": tag},
        )

    def thin(self, p: float, seed: int) -> AcqData:
        """Copy keeping each event independently with probability ``p``."""
        rng = np.random.default_rng(seed)
        keep = rng.random(self.events.height) < p
        return AcqData(
            key=self.key + f"-thin{p:.3f}",
            ids=self.ids,
            time=self.time,
            syn=self.syn,
            events=self.events.filter(pl.Series(keep)),
            detectable=self.detectable,
            mf_snr=None,
            dn_sum=self.dn_sum,
            hypno=self.hypno,
            info=self.info,
        )

    def syn_array(self, values: np.ndarray, name: str) -> xr.DataArray:
        """Wrap a (n_syn, n_frames) array with time + synapse coordinates."""
        coords = {"syn_id": self.syn["syn_id"].to_list(), "time": self.time}
        for col in ("dend-ID", "dend_type", "pos", "dmd"):
            coords[col] = ("syn_id", self.syn[col].to_numpy())
        return xr.DataArray(values, dims=("syn_id", "time"), coords=coords, name=name)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def _box(x: np.ndarray, k: int) -> np.ndarray:
    """Centered running sum over ``k`` bins along the last axis (edges truncated).

    Window of bin ``t`` = bins ``[t - k//2, t - k//2 + k)``.
    """
    n = x.shape[-1]
    left = k // 2
    cs = np.zeros(x.shape[:-1] + (n + k + 1,), dtype=np.float64)
    np.cumsum(x, axis=-1, out=cs[..., left + 1 : left + 1 + n])
    cs[..., left + 1 + n :] = cs[..., left + n : left + n + 1]  # flat right pad
    return (cs[..., k : k + n] - cs[..., :n]).astype(np.float32)


class Detector:
    """Population ON/OFF detector for one :class:`AcqData`.

    Glutamate events (``z_shot >= z_min``) of every synapse are binned at
    :data:`BIN_S`. In a sliding window of ``window_s`` the population count
    ``N`` is compared with what independent synapses, each firing at its own
    local rate (estimated over ``baseline_s``), would produce: mean ``E`` and
    variance ``V``. The surprise of a window is ``-log10 p`` of a count that
    low (OFF) or that high (ON) under a negative binomial with that mean and
    variance. Runs of windows above ``edge`` that contain one above ``seed``
    become periods spanning the union of their windows. ON and OFF are found
    the same way; nothing prefers alternation.

    Results are cached per parameter combination so the Loupe Tuner (which
    asks for several outputs per slider move) computes each stage once.
    """

    def __init__(self, acq: AcqData) -> None:
        self.acq = acq
        self.fpb = max(1, int(round(BIN_S / float(np.median(np.diff(acq.time))))))
        self.n_bins = acq.time.size // self.fpb
        self.dt = float(acq.time[self.fpb] - acq.time[0])
        self.bin_t = acq.time[: self.n_bins * self.fpb : self.fpb]  # bin start times
        det = acq.detectable[:, : self.n_bins * self.fpb]
        self.valid = det.reshape(acq.n_syn, self.n_bins, self.fpb).all(-1)
        row = {s: i for i, s in enumerate(acq.syn["syn_id"].to_list())}
        ev = acq.events
        self._ev_row = np.array(
            [row[s] for s in ev["syn_id"].to_list()], dtype=np.int64
        )
        self._ev_bin = ev["idx"].to_numpy() // self.fpb
        self._ev_z = ev["z_shot"].to_numpy()
        self._cache: dict = {}

    def _store(self, key: tuple, value, keep: int) -> None:
        """Cache ``value``, keeping only the ``keep`` newest entries of its stage."""
        same = [k for k in self._cache if k[0] == key[0] and k != key]
        for k in same[: max(0, len(same) - keep + 1)]:
            del self._cache[k]
        self._cache[key] = value

    # -- stage 1: binned events --------------------------------------------
    def counts(self, z_min: float, shuffle: bool | int = False) -> np.ndarray:
        """Events per synapse per bin.

        ``shuffle``: ``False``/``0`` = real data; ``True`` or an integer ``k >= 1``
        circularly shifts each synapse's train by its own random offset
        (realisation ``k``; ``True`` is realisation 1). Every ``shuffle`` argument
        of the detector accepts the same values.
        """
        key = ("counts", round(z_min, 4), shuffle)
        if key not in self._cache:
            keep = (self._ev_z >= z_min) & (self._ev_bin < self.n_bins)
            c = np.zeros((self.acq.n_syn, self.n_bins), dtype=np.float32)
            np.add.at(c, (self._ev_row[keep], self._ev_bin[keep]), 1.0)
            if shuffle:  # independence control: same trains, random relative timing
                rng = np.random.default_rng(int(shuffle) - 1)
                lo, hi = int(0.1 * self.n_bins), int(0.9 * self.n_bins)
                for i in range(c.shape[0]):
                    c[i] = np.roll(c[i], rng.integers(lo, hi))
            c *= self.valid
            self._store(key, c, keep=2)
        return self._cache[key]

    # -- stage 2: observed vs independent-synapse expectation ---------------
    def stats(
        self,
        z_min: float,
        window_s: float,
        baseline_s: float,
        shuffle: bool | int = False,
    ) -> dict[str, np.ndarray]:
        """Per-bin population count ``N``, expectation ``E``, null variance ``V``."""
        key = (
            "stats",
            round(z_min, 4),
            round(window_s, 4),
            round(baseline_s, 3),
            shuffle,
        )
        if key in self._cache:
            return self._cache[key]
        c = self.counts(z_min, shuffle)
        k = max(1, int(round(window_s / self.dt)))
        kb = max(k + 1, int(round(baseline_s / self.dt)))
        n_i = _box(c, k)  # each synapse's count in the window
        ok_i = (
            _box(self.valid.astype(np.float32), k) >= k - 0.5
        )  # window fully observed
        n_i *= ok_i
        s0 = _box(ok_i.astype(np.float32), kb)
        s1 = _box(n_i, kb)
        s2 = _box(n_i * n_i, kb)
        enough = s0 >= 0.25 * kb
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(enough, s1 / s0, 0.0)
            var = np.where(enough, np.maximum(s2 / s0 - mu * mu, 0.0), 0.0)
        use = ok_i & enough
        out = dict(
            N=(n_i * use).sum(0),
            E=(mu * use).sum(0),
            V=(var * use).sum(0),
            frac_valid=use.mean(0),
            k=np.int64(k),
        )
        self._store(key, out, keep=4)
        return out

    def surprise(
        self,
        z_min: float,
        window_s: float,
        baseline_s: float,
        shuffle: bool | int = False,
        min_valid: float = DEFAULTS["min_valid"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(s_on, s_off)``: -log10 p of a count this high / this low.

        p comes from a negative binomial with the independent-synapse mean ``E``
        and variance ``V`` (Poisson when ``V <= E``). Bins with too few
        observable synapses get 0.
        """
        key = (
            "surprise",
            round(z_min, 4),
            round(window_s, 4),
            round(baseline_s, 3),
            shuffle,
            min_valid,
        )
        if key in self._cache:
            return self._cache[key]
        st = self.stats(z_min, window_s, baseline_s, shuffle)
        N, E, V = st["N"], st["E"], st["V"]
        ok = (st["frac_valid"] >= min_valid - 1e-9) & (E > 0)
        s_on = np.zeros_like(E)
        s_off = np.zeros_like(E)
        nb = ok & (V > E * 1.001)
        po = ok & ~nb
        if nb.any():
            r = E[nb] ** 2 / (V[nb] - E[nb])
            p = E[nb] / V[nb]
            s_off[nb] = -stats.nbinom.logcdf(N[nb], r, p)
            s_on[nb] = -stats.nbinom.logsf(N[nb] - 1, r, p)
        if po.any():
            s_off[po] = -stats.poisson.logcdf(N[po], E[po])
            s_on[po] = -stats.poisson.logsf(N[po] - 1, E[po])
        s_on = np.minimum(s_on / np.log(10), 30.0)
        s_off = np.minimum(s_off / np.log(10), 30.0)
        self._store(key, (s_on, s_off), keep=4)
        return s_on, s_off

    # -- stage 3: periods --------------------------------------------------
    def periods(
        self,
        z_min: float = DEFAULTS["z_min"],
        window_s: float = DEFAULTS["window_s"],
        baseline_s: float = DEFAULTS["baseline_s"],
        seed: float = DEFAULTS["seed"],
        edge: float = DEFAULTS["edge"],
        shuffle: bool | int = False,
        min_valid: float = DEFAULTS["min_valid"],
    ) -> pl.DataFrame:
        """ON / OFF periods: ``start``, ``end``, ``label``, ``duration``, ``peak``, ``ratio``.

        ``ratio`` is the population count relative to expectation (N / E) at the
        period's most surprising window: ~0 for a deep OFF, >1 for an ON.
        """
        edge = min(edge, seed)
        s_on, s_off = self.surprise(z_min, window_s, baseline_s, shuffle, min_valid)
        st = self.stats(z_min, window_s, baseline_s, shuffle)
        k = int(st["k"])
        rows = []
        for label, s in (("ON", s_on), ("OFF", s_off)):
            above = s >= edge
            edges = np.flatnonzero(np.diff(np.r_[False, above, False]))
            for a, b in zip(edges[::2], edges[1::2], strict=True):
                j = a + int(np.argmax(s[a:b]))
                if s[j] < seed:
                    continue
                lo = max(a - k // 2, 0)
                hi = min(b - 1 - k // 2 + k, self.n_bins)
                rows.append((
                    lo,
                    hi,
                    label,
                    float(s[j]),
                    float(st["N"][j] / st["E"][j]),
                ))
        if not rows:
            return pl.DataFrame(
                schema=dict(
                    start=pl.Float64,
                    end=pl.Float64,
                    label=pl.String,
                    duration=pl.Float64,
                    peak=pl.Float64,
                    ratio=pl.Float64,
                )
            )
        rows.sort()
        merged: list[list] = []
        for lo, hi, label, peak, ratio in rows:
            if merged and lo < merged[-1][1]:
                prev = merged[-1]
                if prev[2] == label:  # same kind: union
                    prev[1] = max(prev[1], hi)
                    if peak > prev[3]:
                        prev[3], prev[4] = peak, ratio
                    continue
                cut = (lo + prev[1]) // 2  # ON meets OFF: split the overlap
                prev[1], lo = cut, cut
            merged.append([lo, hi, label, peak, ratio])
        lo = np.array([m[0] for m in merged])
        hi = np.array([m[1] for m in merged])
        t = np.r_[self.bin_t, self.bin_t[-1] + self.dt]
        return (
            pl
            .DataFrame({
                "start": t[lo],
                "end": t[hi],
                "label": [m[2] for m in merged],
                "peak": [m[3] for m in merged],
                "ratio": [m[4] for m in merged],
            })
            .with_columns((pl.col("end") - pl.col("start")).alias("duration"))
            .filter(pl.col("duration") > 0)
            .select("start", "end", "label", "duration", "peak", "ratio")
        )

    def testable(
        self,
        z_min: float,
        window_s: float,
        baseline_s: float,
        shuffle: bool | int = False,
        min_valid: float = DEFAULTS["min_valid"],
    ) -> np.ndarray:
        """Bins where detection is possible: enough synapses observable for the
        whole window (and with a usable baseline)."""
        st = self.stats(z_min, window_s, baseline_s, shuffle)
        return (st["frac_valid"] >= min_valid - 1e-9) & (st["E"] > 0)

    def no_data_spans(
        self,
        z_min: float,
        window_s: float,
        baseline_s: float,
        shuffle: bool | int = False,
        min_valid: float = DEFAULTS["min_valid"],
    ) -> pl.DataFrame:
        """Stretches where too few synapses are observable, labelled ``"no data"``.

        These are the raw gaps (bins where any synapse, on either DMD, has no
        detectable data when ``min_valid=1``). Detection additionally needs the
        whole window to be observed, so no period ever reaches into a gap.
        """
        ok = self.valid.mean(0) >= min_valid - 1e-9
        edges = np.flatnonzero(np.diff(np.r_[True, ok, True].astype(np.int8)))
        t = np.r_[self.bin_t, self.bin_t[-1] + self.dt]
        lo, hi = edges[::2], edges[1::2]
        return pl.DataFrame({
            "start": t[lo],
            "end": t[hi],
            "label": ["no data"] * lo.size,
        })

    def display_periods(
        self,
        z_min: float,
        window_s: float,
        baseline_s: float,
        seed: float,
        edge: float,
        shuffle: bool | int = False,
        refined: bool = False,
        margin_s: float = 0.5,
    ) -> pl.DataFrame:
        """``start``, ``end``, ``label``, ``duration`` of the periods to show.

        With ``refined=True`` the edges come from :func:`refine_edges` (Poisson
        change points on the events, searched within ``margin_s`` of the detector
        edges, never past the middle of the gap to a neighbouring period). A
        side keeps its detector edge when the fit was impossible (data gap
        nearby) or ran into the search margin.
        """
        per = self.periods(z_min, window_s, baseline_s, seed, edge, shuffle)
        if refined and per.height:
            key = (
                "refined",
                round(z_min, 4),
                round(window_s, 4),
                round(baseline_s, 3),
                round(seed, 3),
                round(edge, 3),
                shuffle,
                round(margin_s, 3),
            )
            if key not in self._cache:
                rf = refine_edges(
                    self,
                    margin_s,
                    shuffle=shuffle,
                    z_min=z_min,
                    window_s=window_s,
                    baseline_s=baseline_s,
                    seed=seed,
                    edge=edge,
                )
                # a side whose fit ran into the search margin keeps its detector edge
                self._store(
                    key,
                    rf.select(
                        "start",
                        "label",
                        pl
                        .when(pl.col("left_margin"))
                        .then(None)
                        .otherwise(pl.col("r_start"))
                        .alias("r_start"),
                        pl
                        .when(pl.col("right_margin"))
                        .then(None)
                        .otherwise(pl.col("r_end"))
                        .alias("r_end"),
                    ),
                    keep=2,
                )
            per = (
                per
                .join(self._cache[key], on=["start", "label"], how="left")
                .with_columns(
                    pl.coalesce("r_start", "start").alias("start"),
                    pl.coalesce("r_end", "end").alias("end"),
                )
                .drop("r_start", "r_end")
            )
        return per.with_columns(
            (pl.col("end") - pl.col("start")).alias("duration")
        ).select("start", "end", "label", "duration")

    def view_spans(
        self,
        z_min: float,
        window_s: float,
        baseline_s: float,
        seed: float,
        edge: float,
        shuffle: bool | int = False,
        refined: bool = False,
        margin_s: float = 0.5,
    ) -> pl.DataFrame:
        """ON / OFF periods plus "no data" spans, for the Loupe shading."""
        per = self.display_periods(
            z_min, window_s, baseline_s, seed, edge, shuffle, refined, margin_s
        )
        gaps = self.no_data_spans(z_min, window_s, baseline_s, shuffle)
        return pl.concat([per.select("start", "end", "label"), gaps])

    # -- display traces ------------------------------------------------------
    def ratio_trace(
        self,
        z_min: float,
        window_s: float,
        baseline_s: float,
        shuffle: bool | int = False,
    ) -> xr.DataArray:
        """Population count relative to expectation, N / E (1 = typical)."""
        st = self.stats(z_min, window_s, baseline_s, shuffle)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(st["E"] > 0, st["N"] / st["E"], np.nan)
        r[~self.testable(z_min, window_s, baseline_s, shuffle)] = np.nan
        return xr.DataArray(r, dims="time", coords={"time": self.bin_t}, name="N / E")

    def surprise_trace(
        self,
        z_min: float,
        window_s: float,
        baseline_s: float,
        shuffle: bool | int = False,
    ) -> xr.DataArray:
        """Signed surprise: +(-log10 p) for excess activity, -(-log10 p) for silence."""
        s_on, s_off = self.surprise(z_min, window_s, baseline_s, shuffle)
        s = np.where(s_on >= s_off, s_on, -s_off)
        return xr.DataArray(
            s, dims="time", coords={"time": self.bin_t}, name="surprise"
        )

    def level_trace(self, level: float, sign: float = 1.0) -> xr.DataArray:
        """Constant line (threshold display)."""
        return xr.DataArray(
            np.full(self.n_bins, sign * level), dims="time", coords={"time": self.bin_t}
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _kw(params: dict) -> dict:
    """Detector settings with defaults filled in (``min_valid`` excluded)."""
    return {
        k: params.get(k, DEFAULTS[k])
        for k in ("z_min", "window_s", "baseline_s", "seed", "edge")
    }


def _core(kw: dict) -> dict:
    return {k: kw[k] for k in ("z_min", "window_s", "baseline_s")}


def state_at(acq: AcqData, t: np.ndarray) -> np.ndarray:
    """Hypnogram state at each time ``t`` (ephys seconds)."""
    hyp = acq.hypno
    lab = np.array(hyp["label"].to_list(), dtype=object)
    idx = (
        np.searchsorted(hyp["start_s"].to_numpy(), np.asarray(t, float), side="right")
        - 1
    )
    return lab[np.clip(idx, 0, None)].astype(str)


def third_of(det: Detector, t: np.ndarray) -> np.ndarray:
    """``early`` / ``middle`` / ``late`` third of the analysis window for times ``t``."""
    span = det.acq.span_s
    i = np.floor((np.asarray(t, float) - det.acq.time[0]) / (span / 3)).astype(int)
    return np.array(THIRDS, dtype=object)[np.clip(i, 0, 2)].astype(str)


def _intervals(per: pl.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray]:
    p = per.filter(pl.col("label") == label).sort("start")
    return p["start"].to_numpy(), p["end"].to_numpy()


def _overlaps(
    s: np.ndarray, e: np.ndarray, bs: np.ndarray, be: np.ndarray
) -> np.ndarray:
    """Does span [s, e) overlap any of the sorted, non-overlapping spans [bs, be)?"""
    if bs.size == 0:
        return np.zeros(s.shape, dtype=bool)
    i = np.searchsorted(bs, e, side="left") - 1
    return (i >= 0) & (be[np.clip(i, 0, None)] > s)


def _span_mask(
    det: Detector, per: pl.DataFrame, label: str | None = None, pad_s: float = 0.0
) -> np.ndarray:
    """Bins covered by periods (optionally of one label), widened by ``pad_s``."""
    m = np.zeros(det.n_bins, dtype=bool)
    p = per if label is None else per.filter(pl.col("label") == label)
    t0 = det.bin_t[0]
    for s_, e_ in p.select("start", "end").iter_rows():
        b0 = int(np.floor((s_ - pad_s - t0) / det.dt))
        b1 = int(np.ceil((e_ + pad_s - t0) / det.dt))
        m[max(b0, 0) : min(b1, det.n_bins)] = True
    return m


class _Testable:
    """Fast "is this whole span testable?" lookups for one detector setting."""

    def __init__(self, det: Detector, kw: dict) -> None:
        ok = det.testable(**_core(kw))
        self.t0, self.dt, self.n = float(det.bin_t[0]), det.dt, ok.size
        self.cu = np.r_[0, np.cumsum(~ok)]

    def __call__(self, s: np.ndarray, e: np.ndarray) -> np.ndarray:
        b0 = np.floor((s - self.t0) / self.dt).astype(int)
        b1 = np.ceil((e - self.t0) / self.dt).astype(int)
        inside = (b0 >= 0) & (b1 <= self.n)
        b0, b1 = np.clip(b0, 0, self.n), np.clip(b1, 0, self.n)
        return inside & (self.cu[b1] - self.cu[b0] == 0)


# ---------------------------------------------------------------------------
# Event-based edges
# ---------------------------------------------------------------------------


def refine_edges(
    det: Detector, margin_s: float = 0.5, shuffle: bool | int = False, **params
) -> pl.DataFrame:
    """Window-independent period edges from a Poisson change-point fit.

    Around each detected period (± ``margin_s``), the population count per
    10-ms bin is modelled as three stretches (before / inside / after), each
    with its own rate relative to the independent-synapse expectation. The two
    change points that maximise the likelihood, constrained to enclose the
    period's centre, are the refined edges; they sit on the events themselves,
    not on the counting window. Also returns the fitted inside/outside rate
    ratios (``depth`` = inside ÷ expected, i.e. ~0 for a deep OFF).

    The search never crosses into a neighbouring period: each side is bounded
    by the midpoint of the gap to the neighbouring period (of either type), so
    refined periods cannot overlap. ``left_margin`` / ``right_margin`` flag a
    side whose edge landed on the ``margin_s`` limit (the fit wanted to go
    further, so that edge is not trustworthy); ``at_margin`` = either side.
    """
    kw = _kw(params)
    per = det.periods(**kw, shuffle=shuffle)
    st = det.stats(kw["z_min"], BIN_S, kw["baseline_s"], shuffle)
    N, E = st["N"].astype(np.float64), st["E"].astype(np.float64)
    ok = det.testable(**_core(kw), shuffle=shuffle)
    m = int(round(margin_s / det.dt))
    t0 = det.bin_t[0]
    rows = []

    def ll(n, e):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                n > 0, n * np.log(np.maximum(n, 1e-12) / np.maximum(e, 1e-12)) - n, 0.0
            )

    per = per.sort("start")
    ps_, pe_ = per["start"].to_numpy(), per["end"].to_numpy()
    prev_end = np.r_[-np.inf, pe_[:-1]]
    next_start = np.r_[ps_[1:], np.inf]
    for i, (s_, e_, lab) in enumerate(per.select("start", "end", "label").iter_rows()):
        lo = int(np.floor((s_ - t0) / det.dt))
        hi = int(np.ceil((e_ - t0) / det.dt))
        L_m, R_m = max(lo - m, 0), min(hi + m, det.n_bins)
        # never search past the middle of the gap to a neighbouring period
        L_n = (
            int(np.ceil(((prev_end[i] + s_) / 2 - t0) / det.dt))
            if np.isfinite(prev_end[i])
            else 0
        )
        R_n = (
            int(np.floor(((e_ + next_start[i]) / 2 - t0) / det.dt))
            if np.isfinite(next_start[i])
            else det.n_bins
        )
        L, R = max(L_m, min(L_n, lo)), min(R_m, max(R_n, hi))
        if not ok[L:R].all():
            continue
        c = (lo + hi) // 2
        cn = np.r_[0.0, np.cumsum(N[L:R])]
        ce = np.r_[0.0, np.cumsum(E[L:R])]
        a = np.arange(0, c - L + 1)  # inside starts at L + a (<= centre)
        b = np.arange(c - L + 1, R - L + 1)  # inside ends at L + b (> centre)
        nin = cn[b][None, :] - cn[a][:, None]
        ein = ce[b][None, :] - ce[a][:, None]
        tot = (
            ll(cn[a], ce[a])[:, None]
            + ll(nin, ein)
            + ll(cn[-1] - cn[b], ce[-1] - ce[b])[None, :]
        )
        ia, ib = np.unravel_index(np.argmax(tot), tot.shape)
        A, B = L + a[ia], L + b[ib]
        ein_ = ce[b[ib]] - ce[a[ia]]
        eout = ce[-1] - ein_
        rows.append(
            dict(
                label=lab,
                start=s_,
                end=e_,
                duration=e_ - s_,
                r_start=float(t0 + A * det.dt),
                r_end=float(t0 + B * det.dt),
                r_duration=float((B - A) * det.dt),
                depth=float(nin[ia, ib] / ein_) if ein_ > 0 else np.nan,
                outside=float((cn[-1] - nin[ia, ib]) / eout) if eout > 0 else np.nan,
                left_margin=bool(A == L and L == L_m and L_m > 0),
                right_margin=bool(B == R and R == R_m and R_m < det.n_bins),
            )
        )
    if not rows:
        f, b_, s = pl.Float64, pl.Boolean, pl.String
        return pl.DataFrame(
            schema=dict(
                label=s,
                start=f,
                end=f,
                duration=f,
                r_start=f,
                r_end=f,
                r_duration=f,
                depth=f,
                outside=f,
                left_margin=b_,
                right_margin=b_,
                at_margin=b_,
                state=s,
                third=s,
            )
        )
    df = pl.DataFrame(rows).with_columns(
        (pl.col("left_margin") | pl.col("right_margin")).alias("at_margin")
    )
    mid = ((df["r_start"] + df["r_end"]) / 2).to_numpy()
    return df.with_columns(
        pl.Series("state", state_at(det.acq, mid)),
        pl.Series("third", third_of(det, mid)),
    )
