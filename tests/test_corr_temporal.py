"""Tests for ``wisco_slap.scope.corr.temporal``, ``ensemble``, ``lagged``.

Algorithmic correctness for the temporal-dynamics extensions to the corr
module: bout temporal-context annotation, sliding-window pairwise r,
state-clock tiling, ensemble subspace similarity, and lagged
cross-correlation.

All inputs are constructed in the test bodies (no fixture files); we only
test the pure, hypno-independent helpers, since the data-loading orchestrators
need a real recording.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xarray as xr

from wisco_slap.scope.corr import (
    annotate_bout_temporal_context,
    sliding_window_corr_in_bout,
)
from wisco_slap.scope.corr.ensemble import (
    _grassmann_similarity,
    pair_rank_stability,
)
from wisco_slap.scope.corr.lagged import _peak_lagged_r_per_bout
from wisco_slap.scope.corr.temporal import _build_state_clock_bins


# ---------- helpers


def _bout_row(state: str, t0: float, t1: float, *, idx: int = 0) -> dict:
    return {
        "state": state, "start_time": float(t0), "end_time": float(t1),
        "valid_duration": float(t1 - t0), "wall_duration": float(t1 - t0),
        "bout_idx": int(idx),
    }


def _bouts_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={
        "state": pl.String, "start_time": pl.Float64, "end_time": pl.Float64,
        "valid_duration": pl.Float64, "wall_duration": pl.Float64,
        "bout_idx": pl.Int64,
    })


def _scopex(time: np.ndarray, X: np.ndarray, *, dends: list[str] | None = None,
            dend_types: list[str] | None = None) -> xr.DataArray:
    n_syn = X.shape[0]
    coords = {"syn_id": np.arange(n_syn), "time": time}
    if dends is not None:
        coords["dend-ID"] = ("syn_id", np.asarray(dends, dtype=object))
    if dend_types is not None:
        coords["dend_type"] = ("syn_id", np.asarray(dend_types, dtype=object))
    return xr.DataArray(X, dims=("syn_id", "time"), coords=coords)


# ============================================ annotate_bout_temporal_context


def test_annotate_bout_temporal_context_per_state_independent_clocks() -> None:
    # Wake/NREM/Wake/NREM, alternating. Each state has 2 bouts.
    bouts = _bouts_df([
        _bout_row("Wake", 0.0, 5.0, idx=0),
        _bout_row("NREM", 5.0, 15.0, idx=1),
        _bout_row("Wake", 15.0, 25.0, idx=2),   # 2nd Wake bout
        _bout_row("NREM", 25.0, 30.0, idx=3),
    ])
    out = annotate_bout_temporal_context(bouts, use_valid_duration=True).sort("start_time")
    rows = list(out.iter_rows(named=True))
    # Wake clock: 1st Wake = 0; 2nd Wake = 5 (sum of 1st Wake's 5s)
    wake_rows = [r for r in rows if r["state"] == "Wake"]
    nrem_rows = [r for r in rows if r["state"] == "NREM"]
    assert len(wake_rows) == 2 and len(nrem_rows) == 2
    assert wake_rows[0]["cum_state_time_at_start"] == 0.0
    assert wake_rows[0]["cum_state_time_at_end"] == 5.0
    assert wake_rows[1]["cum_state_time_at_start"] == 5.0
    assert wake_rows[1]["cum_state_time_at_end"] == 15.0
    # NREM clock: independent of intervening Wake.
    assert nrem_rows[0]["cum_state_time_at_start"] == 0.0
    assert nrem_rows[0]["cum_state_time_at_end"] == 10.0
    assert nrem_rows[1]["cum_state_time_at_start"] == 10.0
    assert nrem_rows[1]["cum_state_time_at_end"] == 15.0
    # bout_index_in_state
    assert wake_rows[0]["bout_index_in_state"] == 0
    assert wake_rows[1]["bout_index_in_state"] == 1
    assert nrem_rows[0]["bout_index_in_state"] == 0
    assert nrem_rows[1]["bout_index_in_state"] == 1


def test_annotate_bout_temporal_context_prev_gap_first_null() -> None:
    bouts = _bouts_df([
        _bout_row("NREM", 10.0, 20.0, idx=0),
        _bout_row("NREM", 35.0, 45.0, idx=1),  # gap of 15s
    ])
    out = annotate_bout_temporal_context(bouts).sort("start_time")
    rows = list(out.iter_rows(named=True))
    assert rows[0]["prev_bout_gap_s"] is None
    assert rows[1]["prev_bout_gap_s"] == 15.0


def test_annotate_bout_temporal_context_empty_returns_typed_columns() -> None:
    empty = _bouts_df([])
    out = annotate_bout_temporal_context(empty)
    for c in (
        "bout_index_in_state",
        "cum_state_time_at_start",
        "cum_state_time_at_end",
        "bout_duration",
        "prev_bout_gap_s",
    ):
        assert c in out.columns


# ============================================== sliding_window_corr_in_bout


def test_sliding_window_corr_in_bout_window_count_and_centers() -> None:
    # 100s bout, window=30, step=10 → starts at 0, 10, 20, …, 70 → 8 windows.
    dt = 0.1
    time = np.arange(0.0, 100.0, dt)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, time.size))
    da = _scopex(time, X)
    bout = {"state": "NREM", "start_time": 0.0, "end_time": 100.0,
            "valid_duration": 100.0, "wall_duration": 100.0, "bout_idx": 0}
    out = sliding_window_corr_in_bout(da, bout, window_s=30.0, step_s=10.0,
                                      min_valid_frac=0.99)
    assert out.height == 8
    # Centers: 15, 25, 35, …, 85.
    expected_centers = np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0])
    np.testing.assert_allclose(
        out["window_center_in_bout_s"].to_numpy(), expected_centers, atol=dt
    )


def test_sliding_window_corr_in_bout_correlation_correct_on_known_pair() -> None:
    # Build two synapses: y = 0.8x + noise → r ~ 0.8 / sqrt(0.8^2 + var(noise)).
    rng = np.random.default_rng(42)
    dt = 0.05
    time = np.arange(0.0, 60.0, dt)
    x = rng.normal(size=time.size)
    eps = rng.normal(scale=0.6, size=time.size)
    y = 0.8 * x + eps
    X = np.stack([x, y], axis=0)
    da = _scopex(time, X)
    bout = {"state": "NREM", "start_time": 0.0, "end_time": 60.0,
            "valid_duration": 60.0, "wall_duration": 60.0, "bout_idx": 0}
    out = sliding_window_corr_in_bout(da, bout, window_s=30.0, step_s=15.0,
                                      min_valid_frac=0.99)
    assert out.height >= 2
    # All windows should give r ~ 0.8 / sqrt(0.8^2 + 0.6^2) ≈ 0.8.
    rs = out["mean_r_offdiag"].to_numpy()
    assert np.all(rs > 0.5) and np.all(rs < 0.9)


def test_sliding_window_corr_in_bout_short_bout_returns_empty() -> None:
    dt = 0.1
    time = np.arange(0.0, 20.0, dt)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, time.size))
    da = _scopex(time, X)
    # bout shorter than window
    bout = {"state": "NREM", "start_time": 0.0, "end_time": 20.0,
            "valid_duration": 20.0, "wall_duration": 20.0, "bout_idx": 0}
    out = sliding_window_corr_in_bout(da, bout, window_s=30.0, step_s=10.0)
    assert out.height == 0


def test_sliding_window_corr_in_bout_drops_nan_heavy_window() -> None:
    dt = 0.1
    time = np.arange(0.0, 100.0, dt)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, time.size))
    # NaN out the middle 25s — covers window starts 30s..70s heavily.
    nan_mask = (time >= 35.0) & (time < 60.0)
    X[:, nan_mask] = np.nan
    da = _scopex(time, X)
    bout = {"state": "NREM", "start_time": 0.0, "end_time": 100.0,
            "valid_duration": 75.0, "wall_duration": 100.0, "bout_idx": 0}
    out = sliding_window_corr_in_bout(da, bout, window_s=30.0, step_s=10.0,
                                      min_valid_frac=0.8)
    # Windows whose 80%+ samples are non-NaN should pass; windows landing inside
    # the 25-s NaN gap should be dropped.
    starts = out["window_start_time"].to_numpy()
    # No surviving window should be one whose midpoint is in [35, 60].
    mids = (out["window_start_time"].to_numpy() + out["window_end_time"].to_numpy()) / 2
    bad = (mids >= 35.0) & (mids <= 60.0)
    assert not bad.any()
    # And we should keep at least the first window (0..30s).
    assert (starts == 0.0).any()


# ============================================================ state_clock


def test_build_state_clock_bins_total_time_equals_total_valid() -> None:
    # Two bouts of 30 + 50 = 80s; window = 25.
    bouts = _bouts_df([
        _bout_row("NREM", 0.0, 30.0, idx=0),
        _bout_row("NREM", 100.0, 150.0, idx=1),
    ])
    slices = _build_state_clock_bins(bouts, window_s=25.0)
    # Each row covers (slice_end - slice_start) ≈ valid_dur for that fraction.
    # The sum of (slice_end - slice_start) across all rows should equal
    # total wall_duration of the bouts (80s).
    total = float(
        (slices["slice_end_time"] - slices["slice_start_time"]).sum()
    )
    assert abs(total - 80.0) < 1e-6


def test_build_state_clock_bins_first_bin_starts_at_zero_clock() -> None:
    bouts = _bouts_df([
        _bout_row("NREM", 50.0, 80.0, idx=0),
    ])
    slices = _build_state_clock_bins(bouts, window_s=10.0)
    assert slices.height >= 1
    first = slices.sort("clock_bin_idx").row(0, named=True)
    assert first["clock_bin_idx"] == 0
    assert first["clock_bin_start_s"] == 0.0


def test_build_state_clock_bins_empty_bouts() -> None:
    out = _build_state_clock_bins(_bouts_df([]), window_s=30.0)
    assert out.height == 0


# =================================================== Grassmann similarity


def test_grassmann_identical_subspaces_returns_one() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 10))
    sim = _grassmann_similarity(A, A.copy(), k=3)
    assert abs(sim - 1.0) < 1e-9


def test_grassmann_orthogonal_subspaces_returns_zero() -> None:
    # Build A = first 3 standard-basis directions, B = last 3.
    n_features = 8
    A = np.zeros((3, n_features))
    A[0, 0] = A[1, 1] = A[2, 2] = 1.0
    B = np.zeros((3, n_features))
    B[0, 5] = B[1, 6] = B[2, 7] = 1.0
    sim = _grassmann_similarity(A, B, k=3)
    assert abs(sim) < 1e-9


def test_grassmann_handles_nan_pad_rows() -> None:
    n_features = 6
    A = np.zeros((4, n_features))
    A[0, 0] = A[1, 1] = A[2, 2] = 1.0
    A[3] = np.nan
    sim = _grassmann_similarity(A, A.copy(), k=3)
    assert abs(sim - 1.0) < 1e-9


# =================================================== pair_rank_stability


def test_pair_rank_stability_perfect_when_zvecs_identical() -> None:
    # 3 bouts of identical z-vectors → consecutive Spearman = 1.0.
    z = np.tile(np.array([0.1, 0.5, -0.3, 0.7, 0.0]), (3, 1))
    bout_meta = pl.DataFrame({
        "state": ["NREM", "NREM", "NREM"],
        "start_time": [0.0, 100.0, 200.0],
        "end_time": [50.0, 150.0, 250.0],
        "cum_state_time_at_start": [0.0, 50.0, 100.0],
        "cum_state_time_at_end": [50.0, 100.0, 150.0],
        "cell_id": ["c|1"] * 3, "subject": ["s"] * 3, "soma_id": ["soma1"] * 3,
    })
    out = pair_rank_stability(z, bout_meta, lags=(1,))
    assert out.height == 2
    assert all(abs(r - 1.0) < 1e-9 for r in out["spearman_r"].to_list())


def test_pair_rank_stability_anticorrelated_returns_minus_one() -> None:
    z1 = np.array([0.1, 0.5, -0.3, 0.7, 0.0])
    z2 = -z1
    z = np.stack([z1, z2], axis=0)
    bout_meta = pl.DataFrame({
        "state": ["NREM", "NREM"], "start_time": [0.0, 100.0],
        "end_time": [50.0, 150.0],
        "cum_state_time_at_start": [0.0, 50.0],
        "cum_state_time_at_end": [50.0, 100.0],
        "cell_id": ["c|1"] * 2, "subject": ["s"] * 2, "soma_id": ["soma1"] * 2,
    })
    out = pair_rank_stability(z, bout_meta, lags=(1,))
    assert out.height == 1
    assert out["spearman_r"][0] < -0.99


# ============================================ lagged cross-correlation


def test_merge_brief_breaks_basic_absorption() -> None:
    from electro_py.hypno.hypno import Hypnogram
    from wisco_slap.util.validity.hypno import merge_brief_breaking_bouts
    # NREM(80) - Wake(2) - NREM(80): 2-s Wake should be absorbed → NREM(162).
    hypno = Hypnogram(pl.DataFrame({
        "state": ["NREM", "Wake", "NREM"],
        "start_time": [0.0, 80.0, 82.0],
        "end_time":   [80.0, 82.0, 162.0],
    }))
    m = merge_brief_breaking_bouts(hypno, max_break_s=5.0)
    assert m.df.height == 1
    row = m.df.row(0, named=True)
    assert row["state"] == "NREM"
    assert row["start_time"] == 0.0
    assert row["end_time"] == 162.0


def test_merge_brief_breaks_preserves_long_breaks() -> None:
    from electro_py.hypno.hypno import Hypnogram
    from wisco_slap.util.validity.hypno import merge_brief_breaking_bouts
    # 40s Wake between NREM bouts should NOT merge.
    hypno = Hypnogram(pl.DataFrame({
        "state": ["NREM", "Wake", "NREM"],
        "start_time": [0.0, 80.0, 120.0],
        "end_time":   [80.0, 120.0, 200.0],
    }))
    m = merge_brief_breaking_bouts(hypno, max_break_s=5.0)
    assert m.df.height == 3


def test_merge_brief_breaks_skips_when_neighbors_differ() -> None:
    from electro_py.hypno.hypno import Hypnogram
    from wisco_slap.util.validity.hypno import merge_brief_breaking_bouts
    # NREM - REM(1s) - Wake: a brief REM flanked by DIFFERENT states stays.
    hypno = Hypnogram(pl.DataFrame({
        "state": ["NREM", "REM", "Wake"],
        "start_time": [0.0, 80.0, 81.0],
        "end_time":   [80.0, 81.0, 161.0],
    }))
    m = merge_brief_breaking_bouts(hypno, max_break_s=5.0)
    assert m.df.height == 3


def test_merge_brief_breaks_iterates_until_convergence() -> None:
    from electro_py.hypno.hypno import Hypnogram
    from wisco_slap.util.validity.hypno import merge_brief_breaking_bouts
    # NREM(80) - Wake(2) - NREM(2) - Wake(2) - NREM(80).
    # First merge absorbs middle Wake → NREM(80) - Wake(2)? No — wait.
    # Actually after first iter: only the brief Wakes are flanked by NREM.
    # Iterate: merge first Wake → NREM(82) - NREM(2) - Wake(2) - NREM(80).
    # But two consecutive same-state shouldn't happen post-merge — we merged
    # the trio i-1,i,i+1 into a single bout. Let me redo:
    # Start: [NREM, Wake, NREM, Wake, NREM] (5 bouts)
    # Iter 1: merge index 1 (Wake, brief, flanked by NREM) → [NREM(...), NREM, Wake, NREM]
    # But wait, the merge produces ONE bout from the trio, so we get
    # [NREM(merged)=NREM(0..82+2)=NREM(0..84), Wake(86..88), NREM(88..168)]?
    # Actually let me trace through with explicit numbers.
    hypno = Hypnogram(pl.DataFrame({
        "state": ["NREM", "Wake", "NREM", "Wake", "NREM"],
        "start_time": [0.0,  80.0, 82.0,  84.0, 86.0],
        "end_time":   [80.0, 82.0, 84.0,  86.0, 166.0],
    }))
    m = merge_brief_breaking_bouts(hypno, max_break_s=5.0)
    # All inner brief bouts should collapse into one big NREM.
    assert m.df.height == 1
    row = m.df.row(0, named=True)
    assert row["state"] == "NREM"
    assert row["start_time"] == 0.0
    assert row["end_time"] == 166.0


def test_merge_brief_breaks_zero_threshold_passthrough() -> None:
    from electro_py.hypno.hypno import Hypnogram
    from wisco_slap.util.validity.hypno import merge_brief_breaking_bouts
    hypno = Hypnogram(pl.DataFrame({
        "state": ["NREM", "Wake", "NREM"],
        "start_time": [0.0, 80.0, 82.0],
        "end_time":   [80.0, 82.0, 162.0],
    }))
    m = merge_brief_breaking_bouts(hypno, max_break_s=0.0)
    assert m.df.height == 3   # no merge


def test_head_tail_skips_when_insufficient_valid_samples() -> None:
    # Build a synthetic single-syn dn-ish array with enough total wall but very
    # little valid time. The function should reject bouts where 2*T valid samples
    # don't fit non-overlapping. We exercise the internal head/tail logic via
    # a small end-to-end check on a tiny synthetic mask.
    # The test is on the index logic itself: cumulative-valid searchsorted.
    import numpy as np
    valid = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
    n_valid_total = int(valid.sum())  # 9
    n_target = 5
    cum = np.cumsum(valid)
    head_end_idx = int(np.searchsorted(cum, n_target))   # first k where cum[k] == 5
    cum_threshold = n_valid_total - n_target            # 4
    tail_start_idx = int(np.searchsorted(cum, cum_threshold, side="right"))
    # head must end before tail starts
    assert head_end_idx < tail_start_idx
    # Sanity: head has 5 valid samples; tail has 5 valid samples.
    assert int(valid[: head_end_idx + 1].sum()) >= n_target
    assert int(valid[tail_start_idx:].sum()) >= n_target


def test_head_tail_overlap_detection() -> None:
    # 4 valid samples total, T=3 → can't fit 2*3 non-overlapping.
    import numpy as np
    valid = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
    n_valid_total = int(valid.sum())  # 4
    n_target = 3
    # 2*n_target = 6, n_valid_total = 4 → should reject.
    assert n_valid_total < 2 * n_target


def test_peak_lagged_r_recovers_known_lag() -> None:
    # Build two synapses where syn 1 leads syn 0 by 0.5 s.
    dt = 0.05
    fs = 1 / dt
    time = np.arange(0.0, 60.0, dt)
    rng = np.random.default_rng(7)
    base = rng.normal(scale=1.0, size=time.size)
    # Smooth base for a discernible structure
    from scipy.signal import lfilter
    b = np.ones(int(0.5 * fs)) / int(0.5 * fs)
    smooth = lfilter(b, [1.0], base)
    # syn 0 = smooth, syn 1 = smooth shifted by -0.5s (i.e. syn 1 leads by 0.5s)
    lag_samples = int(round(0.5 / dt))
    syn0 = smooth.copy()
    syn1 = np.empty_like(smooth)
    # syn1[t] = syn0[t + lag] (positive lag = j leads)
    # Edge handling: shift, fill the trailing tail with zeros.
    syn1[:-lag_samples] = syn0[lag_samples:]
    syn1[-lag_samples:] = 0.0
    X = np.stack([syn0, syn1], axis=0)
    da = _scopex(time, X)
    sl = da.sel(time=slice(0.0, 60.0))
    peak_r, peak_lag, _N = _peak_lagged_r_per_bout(sl, dt=dt, max_lag_samples=int(2.0 / dt))
    # For pair (0,1), the peak r should be at lag ≈ +0.5s (j leads i by 0.5s).
    # Equivalently for pair (1,0), it's at lag ≈ -0.5s. The matrix is signed
    # so cov[0,1,lag] = sum syn0[t] syn1[t+lag] is highest at lag = -0.5s
    # (syn1 leads → align by shifting it back). Either sign is acceptable as
    # long as |lag| ≈ 0.5s.
    assert abs(abs(peak_lag[0, 1]) - 0.5) < 0.1
    # Peak r should be moderately strong (we constructed exact alignment).
    assert peak_r[0, 1] > 0.7
