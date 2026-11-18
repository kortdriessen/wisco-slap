"""Synthetic contract checks for the population ON/OFF detector (wis.scope.bistab).

A population of independent Poisson synapses gets planted population-wide
silences (OFF) and bursts (ON). The checks are about construction: the planted
periods are found, independent shuffles find far fewer, periods never overlap,
untestable time is respected, and the cohort cache round-trips. They do not
establish detector sensitivity or specificity on recorded data.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from wisco_slap.scope import bistab as bs

FS = 200.0
N_SYN = 60
RATE = 1.0  # events / s / synapse
DUR_S = 300.0
OFF_AT = np.arange(20.0, 280.0, 20.0)  # planted 0.4-s silences
ON_AT = OFF_AT + 10.0  # planted 0.2-s bursts (x8 rate)


def make_acq(seed: int = 0, gap: tuple[float, float] | None = None) -> bs.AcqData:
    """Independent Poisson synapses with planted population OFFs and ONs."""
    rng = np.random.default_rng(seed)
    n = int(DUR_S * FS)
    t = 1000.0 + np.arange(n) / FS
    rate = np.full(n, RATE / FS)
    for a in OFF_AT:
        rate[int(a * FS) : int((a + 0.4) * FS)] = 0.0
    for a in ON_AT:
        rate[int(a * FS) : int((a + 0.2) * FS)] *= 8.0
    hits = rng.random((N_SYN, n)) < rate[None, :]
    row, idx = np.nonzero(hits)
    ids = np.array([f"s{i}" for i in range(N_SYN)])
    detectable = np.ones((N_SYN, n), dtype=bool)
    if gap is not None:
        detectable[: N_SYN // 2, int(gap[0] * FS) : int(gap[1] * FS)] = False
    events = pl.DataFrame({
        "syn_id": ids[row],
        "idx": idx.astype(np.int64),
        "time": t[idx],
        "z_shot": np.full(idx.size, 6.0),
        "amp": np.ones(idx.size),
    })
    syn = pl.DataFrame({
        "syn_id": ids,
        "dend-ID": np.where(np.arange(N_SYN) < 30, "A-1", "B-1"),
        "dend_type": np.where(np.arange(N_SYN) < 30, "apical", "basal"),
        "pos": np.arange(N_SYN, dtype=float),
        "dmd": np.where(np.arange(N_SYN) < 30, 1, 2),
        "dmd-depth": np.where(np.arange(N_SYN) < 30, 100.0, 200.0),
    })
    hypno = pl.DataFrame({
        "start_s": [t[0], t[0] + 150.0],
        "end_s": [t[0] + 150.0, t[-1]],
        "label": ["Quiet wake", "NREM SWS"],
    })
    return bs.AcqData(
        key=f"synthetic-{seed}",
        ids=("syn", "exp_0", "loc_A", "acq_1", "soma1"),
        time=t,
        syn=syn,
        events=events,
        detectable=detectable,
        mf_snr=None,
        dn_sum=np.zeros(n, np.float32),
        hypno=hypno,
    )


def _hits(
    per: pl.DataFrame, label: str, starts: np.ndarray, dur: float, t0: float
) -> float:
    p = per.filter(pl.col("label") == label)
    s, e = p["start"].to_numpy() - t0, p["end"].to_numpy() - t0
    found = [np.any((s < a + dur) & (e > a)) for a in starts]
    return float(np.mean(found))


def test_planted_periods_found_and_shuffle_quiet() -> None:
    acq = make_acq()
    det = bs.Detector(acq)
    per = det.periods()
    t0 = float(acq.time[0])
    assert _hits(per, "OFF", OFF_AT, 0.4, t0) >= 0.8
    assert _hits(per, "ON", ON_AT, 0.2, t0) >= 0.8
    assert det.periods(shuffle=1).height < 0.5 * per.height


def test_periods_never_overlap() -> None:
    det = bs.Detector(make_acq(seed=1))
    for refined in (False, True):
        p = det.display_periods(
            **{
                k: bs.DEFAULTS[k]
                for k in ("z_min", "window_s", "baseline_s", "seed", "edge")
            },
            refined=refined,
        ).sort("start")
        s, e = p["start"].to_numpy(), p["end"].to_numpy()
        assert np.all(s[1:] >= e[:-1] - 1e-9)
        assert np.all(e > s)


def test_untestable_gap_has_no_periods() -> None:
    gap = (100.0, 130.0)
    acq = make_acq(gap=gap)
    det = bs.Detector(acq)
    per = det.periods()
    t0 = float(acq.time[0])
    inside = (per["end"] > t0 + gap[0]) & (per["start"] < t0 + gap[1])
    assert not inside.any()
    nd = det.no_data_spans(**{
        k: bs.DEFAULTS[k] for k in ("z_min", "window_s", "baseline_s")
    })
    assert nd.height == 1
    assert nd["start"][0] == pytest.approx(t0 + gap[0], abs=0.02)


def test_state_table_and_testable_minutes() -> None:
    det = bs.Detector(make_acq())
    tab = bs.state_table(det)
    assert tab["state"].to_list() == ["Quiet wake", "NREM SWS", "all"]
    mins = bs.testable_minutes(det)
    assert sum(mins.values()) == pytest.approx(
        tab.filter(pl.col("state") == "all")["min"][0]
    )
    assert sum(mins.values()) <= DUR_S / 60


def test_seed_is_detector_threshold_and_rng_seed_is_null_seed() -> None:
    det = bs.Detector(make_acq())
    lo = bs.alternation_test(det, n_null=5, seed=2.5, rng_seed=1)
    hi = bs.alternation_test(det, n_null=5, seed=6.0, rng_seed=1)
    assert lo["n"].sum() > hi["n"].sum()


def test_chains() -> None:
    s = np.array([0.0, 1.0, 1.6, 2.4, 10.0])
    e = np.array([0.8, 1.5, 2.2, 3.0, 10.5])
    lab = np.array(["ON", "OFF", "ON", "OFF", "OFF"])
    ch = bs.chains(s, e, lab, gap_s=0.5)
    assert ch["n"].to_list() == [4, 1]


def test_jitter_periods_keeps_constraints() -> None:
    det = bs.Detector(make_acq(gap=(100.0, 130.0)))
    per = det.periods()
    null = bs.jitter_periods(det, n=4, rng_seed=3, periods=per)
    assert sorted(null["rep"].unique().to_list()) == [1, 2, 3, 4]
    on, off = per.filter(pl.col("label") == "ON"), per.filter(pl.col("label") == "OFF")
    t0 = float(det.acq.time[0])
    for (rep,), g in null.group_by("rep"):
        g_on, g_off = g.filter(pl.col("label") == "ON"), g.filter(pl.col("label") == "OFF")
        # the other type is untouched; moved periods keep their number and durations
        assert g_on.sort("start")["start"].to_list() == on.sort("start")["start"].to_list()
        np.testing.assert_allclose(np.sort(g_off["duration"].to_numpy()), np.sort(off["duration"].to_numpy()))
        # no overlap among all periods, and none in the untestable gap
        p = g.sort("start")
        assert np.all(p["start"].to_numpy()[1:] >= p["end"].to_numpy()[:-1] - 1e-9)
        assert not ((g["end"] > t0 + 100.0) & (g["start"] < t0 + 130.0)).any()


def test_participation_and_subset() -> None:
    acq = make_acq()
    sub = acq.subset(acq.syn["dend-ID"].to_numpy() == "A-1", "-A")
    assert sub.n_syn == 30 and sub.mf_snr is None and sub.cache_dir is None
    pa = bs.participation(acq, min_syn=8, window_s=0.4)
    off = pa.filter(pl.col("label") == "OFF")
    # planted OFFs are population-wide, so each dendrite is silent in the other's OFFs
    assert (off["N"].sum() / off["E"].sum()) < 0.5


def test_span_and_thirds() -> None:
    acq = make_acq()
    assert acq.span_s == pytest.approx(DUR_S)
    acq.info["duration_s"] = 240.0
    det = bs.Detector(acq)
    t = acq.time[0] + np.array([10.0, 100.0, 200.0, 290.0])
    assert bs.third_of(det, t).tolist() == ["early", "middle", "late", "late"]


def test_cohort_caches(tmp_path) -> None:
    acqs = {
        "a": ("syn", "exp_0", "loc_A", "acq_1", "soma1"),
        "b": ("syn", "exp_0", "loc_A", "acq_2", "soma1"),
    }
    calls: list[str] = []

    def loader(key: str) -> bs.AcqData:
        calls.append(key)
        return make_acq(seed=0 if key == "a" else 1)

    df = bs.cohort(
        "testable_by_bin", acqs, loader=loader, cache_dir=tmp_path, bin_min=1.0
    )
    assert set(df["acq"].unique()) == {"a", "b"}
    again = bs.cohort(
        "testable_by_bin", acqs, loader=loader, cache_dir=tmp_path, bin_min=1.0
    )
    assert calls == ["a", "b"]
    assert again.equals(df)
    with pytest.raises(KeyError):
        bs.cohort("nope", acqs, loader=loader, cache_dir=tmp_path)
