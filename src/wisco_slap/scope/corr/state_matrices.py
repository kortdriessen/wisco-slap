"""State-conditioned synapse-by-synapse correlation matrices, comparable across states.

This module answers one question: *what does the pairwise correlation matrix
of a soma's synapses look like in Wake, NREM and REM (and over the whole
recording), computed so that the matrices are directly comparable?*

The estimator, step by step (see :func:`state_tiles` and
:func:`state_corr_matrices`):

1. **One common validity mask.** A time point is usable only if *every*
   synapse of *every* signal is non-NaN there (:func:`common_validity_mask`).
   Every pair is therefore correlated over exactly the same samples, so a
   matrix has one sample count instead of a different one per pair.
2. **Fixed-length tiles per state.** Each hypnogram bout of a state is cut
   into windows that contain exactly ``epoch_length`` seconds of valid data
   (:func:`wisco_slap.scope.corr.fixed_valid_bouts`, span mode). Tiles never
   cross a state boundary and never swallow a NaN gap longer than
   ``max_nan_span``; bouts shorter than one tile are dropped. The
   whole-recording column (:data:`ALL_LABEL`) is tiled the same way from a
   single bout spanning the recording, ignoring state.
3. **Optional duration matching.** With ``equalise=True`` every column is a
   random subsample (fixed ``seed``) of the same number of tiles — the count
   of the sparsest state. Pearson r is a normalised quantity, so its
   *expectation* does not depend on how much data went in, but its sampling
   noise does; matching equalises the noise floor at the price of precision.
   The default keeps every tile and reports the per-column noise floor
   instead.
4. **Per-tile centring, Fisher-z averaging.** r is computed inside each tile
   (each synapse centred on its own tile mean), transformed to Fisher z,
   averaged across tiles and transformed back. Centring per tile removes
   everything slower than the tile length (bleaching, F0 drift, slow
   state-dependent rate co-modulation), so what remains is co-fluctuation
   on the sub-tile time scale. ``method="partial"`` instead pools the
   per-tile-centred covariance over all tiles and inverts it once.
5. **Noise floor.** The same estimator is run on a circular-shift surrogate:
   inside every tile each synapse's trace is rolled by its own random offset
   (≥ ``null_min_shift_s``), which keeps every trace's autocorrelation and
   amplitude distribution but destroys simultaneity. The standard deviation
   of that matrix's off-diagonal entries (``null_sd``) is the size of |r|
   that pure sampling noise produces for that column.

Methods: ``"pearson"`` (default), ``"spearman"`` (per-tile rank transform;
robust to a few huge events), ``"partial"`` (partial correlation from the
pooled covariance — the correlation of each pair after regressing out *all*
other synapses; removes indirect paths and any shared global component).
Options: ``bin_s`` averages each tile into bins before correlating (moves the
co-activity time scale), ``remove_common=True`` regresses the tile-wide
population mean out of every trace first.

Event-based alternative: :func:`state_sttc_matrices` computes the spike-time
tiling coefficient (Cutts & Eglen 2014) between detected event trains on the
same tiles — a coincidence measure that is insensitive to event rate by
construction, which continuous-trace Pearson r is not.

State contrasts: :func:`state_differences` subtracts two columns and
propagates the noise floors; :func:`timescale_sweep` summarises mean r per
pair class as a function of bin width.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import xarray as xr
from scipy.ndimage import maximum_filter1d
from scipy.stats import rankdata

from ...util.validity.mask import validity_mask
from .aggregate import _r_to_z
from .bouts import fixed_valid_bouts, random_subsample_bouts
from .core import _make_pair_coords, _prep_combined

#: Column label of the state-agnostic, whole-recording matrix.
ALL_LABEL = "All"
#: Default sleep states, in display order.
DEFAULT_STATES: tuple[str, ...] = ("Wake", "NREM", "REM")
#: Correlation methods understood by :func:`state_corr_matrices`.
METHODS: tuple[str, ...] = ("pearson", "spearman", "partial")
#: Default state contrasts of :func:`state_differences`.
DEFAULT_DIFFERENCES: tuple[tuple[str, str], ...] = (
    ("Wake", "NREM"),
    ("Wake", "REM"),
    ("NREM", "REM"),
)
#: Pair classes of :func:`pair_class_summary`.
PAIR_CLASSES: tuple[str, ...] = ("within_dend", "between_dend", "cross_dmd")


# ---------------------------------------------------------------------------
# Validity + tiling
# ---------------------------------------------------------------------------


def common_validity_mask(
    signals: Mapping[str, xr.DataArray] | xr.DataArray,
) -> xr.DataArray:
    """Boolean ``(time,)`` mask: True where every synapse of every signal is finite.

    Parameters
    ----------
    signals
        One scopex DataArray ``(syn_id, time)`` or a mapping of several that
        share the same time axis (e.g. the LS, denoised and matched-filter
        traces of the same synapses).

    Returns
    -------
    xr.DataArray
        1-D boolean mask on the shared ``time`` coord (``mode='all'`` of
        :func:`wisco_slap.util.validity.validity_mask`, AND-ed over signals).
    """
    if isinstance(signals, xr.DataArray):
        signals = {"signal": signals}
    if len(signals) == 0:
        raise ValueError("`signals` is empty.")
    masks = []
    ref_time = None
    for name, da in signals.items():
        da = _prep_combined(da)
        t = np.asarray(da["time"].values, dtype=float)
        if ref_time is None:
            ref_time = t
        elif t.shape != ref_time.shape or not np.allclose(t, ref_time):
            raise ValueError(
                f"signal {name!r} does not share the time axis of the first signal."
            )
        masks.append(np.asarray(validity_mask(da, mode="all").values, dtype=bool))
    common = np.logical_and.reduce(masks)
    return xr.DataArray(
        common, dims=("time",), coords={"time": ref_time}, name="is_valid"
    )


def _whole_recording_hypno(mask: xr.DataArray):
    """A one-bout hypnogram (state :data:`ALL_LABEL`) spanning the whole mask."""
    from electro_py.hypno.hypno import Hypnogram

    t = np.asarray(mask["time"].values, dtype=float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    return Hypnogram(
        pl.DataFrame({
            "state": [ALL_LABEL],
            "start_time": [float(t[0])],
            "end_time": [float(t[-1]) + dt],
        })
    )


@dataclass
class StateTiles:
    """Tiles per column (state) plus everything needed to cut the traces identically.

    Attributes
    ----------
    tiles
        ``{column: bouts DataFrame}`` in the standard bouts schema of
        :mod:`wisco_slap.scope.corr.bouts` (``start_time``/``end_time`` are
        inclusive time-coord values). Column order = display order.
    mask
        The common validity mask the tiles were cut on. Traces must be
        sliced with this same mask so every tile has the same sample count.
    epoch_length
        Valid seconds per tile.
    equalised
        Whether every column was subsampled to the same tile count.
    seed
        Seed of that subsample.
    n_available
        Tile count per column *before* equalising.
    """

    tiles: dict[str, pl.DataFrame]
    mask: xr.DataArray
    epoch_length: float
    equalised: bool
    seed: int
    n_available: dict[str, int] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return list(self.tiles)

    @property
    def n_tiles(self) -> dict[str, int]:
        return {k: int(v.height) for k, v in self.tiles.items()}

    @property
    def valid_seconds(self) -> dict[str, float]:
        return {k: float(v.height * self.epoch_length) for k, v in self.tiles.items()}


def state_tiles(
    hypno,
    mask: xr.DataArray,
    *,
    states: Sequence[str] = DEFAULT_STATES,
    epoch_length: float = 10.0,
    include_all: bool = True,
    equalise: bool = False,
    seed: int = 0,
    max_nan_span: float | None = 2.0,
    min_bout_length: float | None = None,
) -> StateTiles:
    """Cut equal-valid-duration tiles per state (and for the whole recording).

    Parameters
    ----------
    hypno
        ``electro_py.hypno.Hypnogram`` on the same time axis as ``mask``
        (e.g. ``hypno_acq`` from ``wis.get.acq_sleep_coverage``).
    mask
        Common validity mask from :func:`common_validity_mask`.
    states
        States to tile, in display order. States absent from the hypnogram
        or yielding no tile are skipped (they are simply missing from
        ``tiles``).
    epoch_length
        Valid seconds per tile.
    include_all
        Prepend a :data:`ALL_LABEL` column tiled over the whole recording
        regardless of state.
    equalise
        Subsample every column to the tile count of the sparsest *state*
        (the ``All`` column is subsampled to the same count). Default False:
        keep every tile; the per-column noise floor reports the precision.
    seed
        Seed for the subsample.
    max_nan_span, min_bout_length
        Forwarded to :func:`wisco_slap.scope.corr.fixed_valid_bouts`
        (``min_bout_length`` defaults to ``epoch_length`` there).
    """
    if epoch_length <= 0:
        raise ValueError(f"epoch_length must be > 0, got {epoch_length}")
    kw = {"max_nan_span": max_nan_span, "min_bout_length": min_bout_length}

    tiles: dict[str, pl.DataFrame] = {}
    if include_all:
        tiles[ALL_LABEL] = fixed_valid_bouts(
            _whole_recording_hypno(mask), mask, ALL_LABEL, epoch_length, **kw
        )
    present = set(getattr(hypno, "states", hypno.df["state"].unique().to_list()))
    for state in states:
        if state == ALL_LABEL:
            raise ValueError(
                f"{ALL_LABEL!r} is reserved for the whole-recording column."
            )
        if state not in present:
            continue
        b = fixed_valid_bouts(hypno, mask, state, epoch_length, **kw)
        if b.height > 0:
            tiles[state] = b

    n_available = {k: int(v.height) for k, v in tiles.items()}
    state_counts = [n for k, n in n_available.items() if k != ALL_LABEL]
    if equalise and state_counts:
        n_target = min(state_counts)
        tiles = {
            k: random_subsample_bouts(v, n_target, seed=seed) for k, v in tiles.items()
        }

    return StateTiles(
        tiles=tiles,
        mask=mask,
        epoch_length=float(epoch_length),
        equalised=bool(equalise),
        seed=int(seed),
        n_available=n_available,
    )


# ---------------------------------------------------------------------------
# Per-tile machinery
# ---------------------------------------------------------------------------


def _dense_tile(
    X: np.ndarray, times: np.ndarray, valid: np.ndarray, t0: float, t1: float
) -> np.ndarray:
    """Valid samples of ``X`` in ``[t0, t1]`` as a NaN-free ``(n_syn, n)`` block."""
    a = int(np.searchsorted(times, t0, side="left"))
    b = int(np.searchsorted(times, t1, side="right"))
    block = X[:, a:b][:, valid[a:b]]
    return np.asarray(block, dtype=np.float64)


def _tile_blocks(da: xr.DataArray, tiles: StateTiles, column: str) -> list[np.ndarray]:
    """Dense, equal-length blocks of ``da`` for every tile of ``column``."""
    times = np.asarray(da["time"].values, dtype=float)
    mask_t = np.asarray(tiles.mask["time"].values, dtype=float)
    if times.shape != mask_t.shape or not np.allclose(times, mask_t):
        raise ValueError("`data` is not on the time axis of `tiles.mask`.")
    valid = np.asarray(tiles.mask.values, dtype=bool)
    X = da.values
    blocks = []
    for row in tiles.tiles[column].iter_rows(named=True):
        blk = _dense_tile(
            X, times, valid, float(row["start_time"]), float(row["end_time"])
        )
        if blk.shape[1] >= 3:
            blocks.append(blk)
    if not blocks:
        return []
    n_min = min(b.shape[1] for b in blocks)
    return [b[:, :n_min] for b in blocks]  # equal by construction; defensive


def _circular_shift_surrogate(
    X: np.ndarray, rng: np.random.Generator, min_shift: int
) -> np.ndarray:
    """Roll every row of ``X`` by its own offset in ``[min_shift, n − min_shift]``.

    Keeps each trace's autocorrelation and amplitude distribution (up to one
    wrap-around point) and destroys simultaneity between traces.
    """
    n_syn, n = X.shape
    lo = min(min_shift, max(n // 2 - 1, 1))
    hi = max(n - lo, lo + 1)
    shifts = rng.integers(lo, hi, size=n_syn)
    idx = (np.arange(n)[None, :] - shifts[:, None]) % n
    return np.take_along_axis(X, idx, axis=1)


def _transform_tile(
    X: np.ndarray,
    *,
    method: str,
    bin_samples: int | None,
    remove_common: bool,
) -> np.ndarray:
    """Optional binning / rank / common-mode-removal steps, then per-row centring."""
    if bin_samples is not None and bin_samples > 1:
        n_bins = X.shape[1] // bin_samples
        if n_bins < 2:
            raise ValueError(
                f"bin of {bin_samples} samples leaves {n_bins} bins in a "
                f"{X.shape[1]}-sample tile; use a longer epoch or a shorter bin."
            )
        X = (
            X[:, : n_bins * bin_samples]
            .reshape(X.shape[0], n_bins, bin_samples)
            .mean(axis=2)
        )
    if method == "spearman":
        X = rankdata(X, axis=1)
    Xc = X - X.mean(axis=1, keepdims=True)
    if remove_common:
        sd = Xc.std(axis=1, keepdims=True)
        Z = Xc / np.where(sd > 0, sd, 1.0)
        m = Z.mean(axis=0)
        mm = float(m @ m)
        if mm > 0:
            beta = (Z @ m) / mm
            Xc = Z - beta[:, None] * m[None, :]
        else:
            Xc = Z
    return Xc


def _corr_centred(Xc: np.ndarray) -> np.ndarray:
    """Pearson r between the rows of an already centred block."""
    s = np.sqrt((Xc**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc @ Xc.T) / np.outer(s, s)
    r = np.where(np.isfinite(r), r, np.nan)
    return np.clip(r, -1.0, 1.0)


def _corr_dense(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Pearson r between the rows of ``X`` and those of ``Y`` (default ``X``)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    sx = np.sqrt((Xc**2).sum(axis=1))
    if Y is None:
        Yc, sy = Xc, sx
    else:
        Yc = Y - Y.mean(axis=1, keepdims=True)
        sy = np.sqrt((Yc**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc @ Yc.T) / np.outer(sx, sy)
    r = np.where(np.isfinite(r), r, np.nan)
    return np.clip(r, -1.0, 1.0)


def _partial_from_cov(S: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Partial correlation matrix from a (pooled) covariance / scatter matrix."""
    d = np.sqrt(np.diag(S))
    with np.errstate(invalid="ignore", divide="ignore"):
        R = S / np.outer(d, d)
    ok = np.isfinite(d) & (d > 0)
    R = np.where(np.isfinite(R), R, 0.0)
    np.fill_diagonal(R, 1.0)
    P = np.linalg.inv(R + ridge * np.eye(R.shape[0]))
    pd = np.sqrt(np.diag(P))
    with np.errstate(invalid="ignore", divide="ignore"):
        pc = -P / np.outer(pd, pd)
    pc[~ok, :] = np.nan
    pc[:, ~ok] = np.nan
    np.fill_diagonal(pc, 1.0)
    return np.clip(pc, -1.0, 1.0)


class _Accumulator:
    """Accumulate per-tile centred blocks into one matrix.

    ``pooled=False``: Fisher-z mean of per-tile Pearson r (pearson /
    spearman). ``pooled=True``: sum of scatter matrices, inverted once at the
    end (partial correlation).
    """

    def __init__(self, n_syn: int, *, pooled: bool):
        self.pooled = pooled
        self.n_syn = n_syn
        self.n_tiles = 0
        self.n_samples = 0
        if pooled:
            self.S = np.zeros((n_syn, n_syn))
        else:
            self.z_sum = np.zeros((n_syn, n_syn))
            self.z_cnt = np.zeros((n_syn, n_syn))

    def add(self, Xc: np.ndarray) -> None:
        self.n_tiles += 1
        self.n_samples += Xc.shape[1]
        if self.pooled:
            self.S += Xc @ Xc.T
        else:
            z = _r_to_z(_corr_centred(Xc))
            ok = np.isfinite(z)
            self.z_sum[ok] += z[ok]
            self.z_cnt += ok

    def result(self) -> np.ndarray:
        if self.pooled:
            if self.n_samples < 3 * self.n_syn:
                warnings.warn(
                    f"partial correlation from {self.n_samples} pooled samples for "
                    f"{self.n_syn} synapses (< 3 per synapse) — expect a noisy, "
                    "shrunken matrix.",
                    stacklevel=3,
                )
            return _partial_from_cov(self.S)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.tanh(np.where(self.z_cnt > 0, self.z_sum / self.z_cnt, np.nan))
        np.fill_diagonal(r, 1.0)
        return r


def _off_diag_sd(M: np.ndarray) -> float:
    off = ~np.eye(M.shape[0], dtype=bool)
    v = M[off]
    v = v[np.isfinite(v)]
    return float(np.std(v)) if v.size else float("nan")


@dataclass
class StateCorrMatrices:
    """Correlation matrices per column (state) for one signal.

    Attributes
    ----------
    r
        ``{column: (syn_1, syn_2) DataArray}`` — the aggregated matrix,
        diagonal 1. Every scopex ``syn_id`` coord is mirrored as
        ``<name>_1`` / ``<name>_2``.
    null_sd
        Per column, the standard deviation of the off-diagonal entries of
        the circular-shift-surrogate matrix — the sampling-noise floor.
    n_tiles, valid_seconds
        Data behind each column.
    tiles
        The :class:`StateTiles` used (shared across signals when produced
        by :func:`state_corr_matrices_multi`).
    params
        ``method``, ``bin_s``, ``remove_common``, ``null_seed``, ...
    """

    r: dict[str, xr.DataArray]
    null_sd: dict[str, float]
    n_tiles: dict[str, int]
    valid_seconds: dict[str, float]
    tiles: StateTiles
    params: dict = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return list(self.r)

    @property
    def method(self) -> str:
        return str(self.params.get("method", "pearson"))


def _min_shift_samples(times: np.ndarray, null_min_shift_s: float) -> int:
    dt = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    return max(1, int(round(null_min_shift_s / dt)))


def state_corr_matrices(
    data,
    tiles: StateTiles,
    *,
    method: str = "pearson",
    bin_s: float | None = None,
    remove_common: bool = False,
    null_seed: int = 0,
    null_min_shift_s: float = 0.5,
) -> StateCorrMatrices:
    """Correlation matrix per column of ``tiles`` for one signal.

    Parameters
    ----------
    data
        Scopex DataArray ``(syn_id, time)`` (a singleton ``channel`` is
        squeezed; a ``{'dmd_1': ..., 'dmd_2': ...}`` dict is combined) on
        the time axis of ``tiles.mask``.
    tiles
        Output of :func:`state_tiles`. The tiles were cut on ``tiles.mask``;
        the traces are sliced with the same mask so every tile contributes
        exactly ``epoch_length / dt`` samples.
    method
        ``"pearson"`` (default), ``"spearman"`` (per-tile rank transform) or
        ``"partial"`` (partial correlation from the pooled per-tile-centred
        covariance: each pair conditioned on every other synapse).
    bin_s
        Optional bin width in seconds; each tile is averaged into
        non-overlapping bins before correlating.
    remove_common
        Regress the tile-wide population mean (of z-scored traces) out of
        every trace before correlating.
    null_seed, null_min_shift_s
        Seed and minimum per-trace offset (seconds) of the circular-shift
        surrogate behind ``null_sd``.

    Returns
    -------
    StateCorrMatrices
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    da = _prep_combined(data)
    times = np.asarray(da["time"].values, dtype=float)
    n_syn = da.sizes["syn_id"]
    coords = _make_pair_coords(da)

    dt = float(np.median(np.diff(times))) if times.size > 1 else np.nan
    bin_samples = None
    if bin_s is not None:
        bin_samples = int(round(bin_s / dt))
        if bin_samples < 1:
            raise ValueError(f"bin_s={bin_s} is shorter than one sample ({dt:.4g} s).")
    transform_kw = {
        "method": method,
        "bin_samples": bin_samples,
        "remove_common": remove_common,
    }
    min_shift = _min_shift_samples(times, null_min_shift_s)
    pooled = method == "partial"

    r_out: dict[str, xr.DataArray] = {}
    null_sd: dict[str, float] = {}
    for column in tiles.columns:
        blocks = _tile_blocks(da, tiles, column)
        if not blocks:
            continue
        rng = np.random.default_rng(null_seed)
        real = _Accumulator(n_syn, pooled=pooled)
        null = _Accumulator(n_syn, pooled=pooled)
        for blk in blocks:
            real.add(_transform_tile(blk, **transform_kw))
            null.add(
                _transform_tile(
                    _circular_shift_surrogate(blk, rng, min_shift), **transform_kw
                )
            )
        r = real.result()
        r_out[column] = xr.DataArray(
            r, dims=("syn_1", "syn_2"), coords=coords, name="r"
        )
        null_sd[column] = _off_diag_sd(null.result())

    return StateCorrMatrices(
        r=r_out,
        null_sd=null_sd,
        n_tiles={k: int(tiles.tiles[k].height) for k in r_out},
        valid_seconds={
            k: float(tiles.tiles[k].height * tiles.epoch_length) for k in r_out
        },
        tiles=tiles,
        params={
            "method": method,
            "bin_s": bin_s,
            "remove_common": remove_common,
            "null_seed": null_seed,
            "null_min_shift_s": null_min_shift_s,
        },
    )


def state_corr_matrices_multi(
    signals: Mapping[str, xr.DataArray],
    tiles: StateTiles,
    **kwargs,
) -> dict[str, StateCorrMatrices]:
    """:func:`state_corr_matrices` for several signals on the same tiles.

    All signals are cut with the same tiles and mask, so their matrices are
    comparable row-for-row. ``kwargs`` are forwarded to
    :func:`state_corr_matrices`.
    """
    return {
        name: state_corr_matrices(da, tiles, **kwargs) for name, da in signals.items()
    }


# ---------------------------------------------------------------------------
# Event-based: spike-time tiling coefficient
# ---------------------------------------------------------------------------


def _sttc_from_sums(C: np.ndarray, n_ev: np.ndarray, T: np.ndarray) -> np.ndarray:
    """STTC from pooled sums: ``C[i, j]`` = events of i within ±Δt of an event of j."""
    with np.errstate(invalid="ignore", divide="ignore"):
        P = C / n_ev[:, None]  # P[i, j] = fraction of i's events near j's
        a = (P - T[None, :]) / (1.0 - P * T[None, :])
        sttc = 0.5 * (a + a.T)
    sttc = np.where(np.isfinite(sttc), sttc, np.nan)
    np.fill_diagonal(sttc, 1.0)
    return np.clip(sttc, -1.0, 1.0)


def state_sttc_matrices(
    events,
    tiles: StateTiles,
    *,
    window_s: float = 0.025,
    null_seed: int = 0,
    null_min_shift_s: float = 0.5,
) -> StateCorrMatrices:
    """Spike-time tiling coefficient between event trains, per column of ``tiles``.

    STTC (Cutts & Eglen, J Neurosci 2014) for a pair (A, B): with ``P_A`` the
    fraction of A's events that fall within ±``window_s`` of an event of B
    and ``T_B`` the fraction of time covered by ±``window_s`` around B's
    events, ``STTC = ½ [(P_A − T_B)/(1 − P_A T_B) + (P_B − T_A)/(1 − P_B T_A)]``.
    It is 0 for independent trains regardless of their rates, 1 for
    identical trains, and bounded in [−1, 1]. Counts and covered times are
    pooled over all tiles of a column before the ratio is formed (sparse
    trains make per-tile values unstable).

    Parameters
    ----------
    events
        ``(syn_id, time)`` DataArray on the tiles' time axis; any value > 0
        is an event sample (e.g. ``wis.scope.pursuit.spike_train(res, like=dn)``).
    tiles
        Output of :func:`state_tiles`.
    window_s
        Half-width Δt of the coincidence window.
    null_seed, null_min_shift_s
        Circular-shift surrogate behind ``null_sd`` (applied to the rasters).

    Returns
    -------
    StateCorrMatrices
        ``r`` holds the STTC matrices; ``params["method"] == "sttc"``.
    """
    da = _prep_combined(events)
    times = np.asarray(da["time"].values, dtype=float)
    n_syn = da.sizes["syn_id"]
    coords = _make_pair_coords(da)
    dt = float(np.median(np.diff(times))) if times.size > 1 else np.nan
    w = max(0, int(round(window_s / dt)))
    size = 2 * w + 1
    min_shift = _min_shift_samples(times, null_min_shift_s)

    def _accumulate(blocks, rng=None):
        C = np.zeros((n_syn, n_syn))
        n_ev = np.zeros(n_syn)
        covered = np.zeros(n_syn)
        n_total = 0
        for blk in blocks:
            E = (blk > 0).astype(np.float64)
            if rng is not None:
                E = _circular_shift_surrogate(E, rng, min_shift)
            D = maximum_filter1d(E, size=size, axis=1, mode="constant")
            C += E @ D.T
            n_ev += E.sum(axis=1)
            covered += D.sum(axis=1)
            n_total += E.shape[1]
        T = covered / max(n_total, 1)
        return _sttc_from_sums(C, n_ev, T)

    r_out: dict[str, xr.DataArray] = {}
    null_sd: dict[str, float] = {}
    for column in tiles.columns:
        blocks = _tile_blocks(da, tiles, column)
        if not blocks:
            continue
        r = _accumulate(blocks)
        r_out[column] = xr.DataArray(
            r, dims=("syn_1", "syn_2"), coords=coords, name="sttc"
        )
        null_sd[column] = _off_diag_sd(
            _accumulate(blocks, rng=np.random.default_rng(null_seed))
        )

    return StateCorrMatrices(
        r=r_out,
        null_sd=null_sd,
        n_tiles={k: int(tiles.tiles[k].height) for k in r_out},
        valid_seconds={
            k: float(tiles.tiles[k].height * tiles.epoch_length) for k in r_out
        },
        tiles=tiles,
        params={
            "method": "sttc",
            "window_s": window_s,
            "null_seed": null_seed,
            "null_min_shift_s": null_min_shift_s,
        },
    )


# ---------------------------------------------------------------------------
# Contrasts and summaries
# ---------------------------------------------------------------------------


@dataclass
class StateCorrDifferences:
    """Differences between two columns of a :class:`StateCorrMatrices`.

    Attributes
    ----------
    d
        ``{label: (syn_1, syn_2) DataArray}`` with ``r[a] − r[b]`` (diagonal
        NaN). ``label`` is ``f"{a} − {b}"``.
    null_sd
        Per label, ``sqrt(null_sd[a]² + null_sd[b]²)`` — the noise floor of
        the difference.
    pairs
        ``{label: (a, b)}``.
    source
        The matrices the differences were taken from.
    """

    d: dict[str, xr.DataArray]
    null_sd: dict[str, float]
    pairs: dict[str, tuple[str, str]]
    source: StateCorrMatrices

    @property
    def columns(self) -> list[str]:
        return list(self.d)


def state_differences(
    sm: StateCorrMatrices,
    pairs: Sequence[tuple[str, str]] = DEFAULT_DIFFERENCES,
    *,
    sep: str = " − ",
) -> StateCorrDifferences:
    """``r[a] − r[b]`` for every ``(a, b)`` in ``pairs`` whose columns exist."""
    d: dict[str, xr.DataArray] = {}
    null_sd: dict[str, float] = {}
    used: dict[str, tuple[str, str]] = {}
    for a, b in pairs:
        if a not in sm.r or b not in sm.r:
            continue
        label = f"{a}{sep}{b}"
        diff = sm.r[a] - sm.r[b]
        M = np.asarray(diff.values, dtype=float).copy()
        np.fill_diagonal(M, np.nan)
        d[label] = xr.DataArray(
            M, dims=("syn_1", "syn_2"), coords=sm.r[a].coords, name="delta_r"
        )
        null_sd[label] = float(
            np.sqrt(sm.null_sd.get(a, np.nan) ** 2 + sm.null_sd.get(b, np.nan) ** 2)
        )
        used[label] = (a, b)
    return StateCorrDifferences(d=d, null_sd=null_sd, pairs=used, source=sm)


def pair_class_summary(r: xr.DataArray, *, stat: str = "mean") -> dict[str, float]:
    """Mean (or median) off-diagonal value per pair class.

    Classes: ``within_dend`` (same ``dend-ID``), ``between_dend`` (different
    dendrite, same DMD), ``cross_dmd`` (different DMDs), plus ``all``.
    Missing coords collapse the classes that need them (everything lands in
    ``between_dend`` when there is no ``dmd_1`` coord, etc.).
    """
    n = r.sizes["syn_1"]
    M = np.asarray(r.values, dtype=float)
    iu, ju = np.triu_indices(n, k=1)
    vals = M[iu, ju]
    dend = (
        np.asarray([str(v) for v in r.coords["dend-ID_1"].values])
        if "dend-ID_1" in r.coords
        else np.array(["?"] * n)
    )
    dmd = (
        np.asarray(r.coords["dmd_1"].values)
        if "dmd_1" in r.coords
        else np.zeros(n, dtype=int)
    )
    same_dend = dend[iu] == dend[ju]
    same_dmd = dmd[iu] == dmd[ju]
    reducer = np.nanmedian if stat == "median" else np.nanmean

    def _red(sel: np.ndarray) -> float:
        v = vals[sel]
        return float(reducer(v)) if v.size and np.isfinite(v).any() else float("nan")

    return {
        "all": _red(np.ones_like(same_dend)),
        "within_dend": _red(same_dend & same_dmd),
        "between_dend": _red(~same_dend & same_dmd),
        "cross_dmd": _red(~same_dmd),
    }


def matrix_difference(a: xr.DataArray, b: xr.DataArray) -> dict[str, float]:
    """Off-diagonal agreement between two matrices of the same synapses.

    Returns the Pearson correlation of the upper triangles, the RMS
    difference and the maximum absolute difference. Handy for checking that
    two estimators (e.g. duration-matched vs all data, or two signal
    normalisations) agree to within the noise floor.
    """
    n = a.sizes["syn_1"]
    iu, ju = np.triu_indices(n, k=1)
    x = np.asarray(a.values, dtype=float)[iu, ju]
    y = np.asarray(b.values, dtype=float)[iu, ju]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = x - y
    return {
        "corr": float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else float("nan"),
        "rms_diff": float(np.sqrt(np.mean(d**2))) if d.size else float("nan"),
        "max_abs_diff": float(np.max(np.abs(d))) if d.size else float("nan"),
    }


def timescale_sweep(
    data,
    tiles: StateTiles,
    bins_s: Sequence[float | None] = (None, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    *,
    method: str = "pearson",
    remove_common: bool = False,
    signal: str = "signal",
) -> pl.DataFrame:
    """Mean r per pair class as a function of bin width, per column.

    Runs :func:`state_corr_matrices` once per entry of ``bins_s`` (``None`` =
    native sampling) and summarises each matrix with
    :func:`pair_class_summary`. Returns a long table with columns
    ``signal, bin_s, column, pair_class, mean_r, null_sd, n_tiles`` (``bin_s``
    is the native sample interval for the ``None`` entry).
    """
    da = _prep_combined(data)
    times = np.asarray(da["time"].values, dtype=float)
    dt = float(np.median(np.diff(times))) if times.size > 1 else float("nan")
    rows: list[dict] = []
    for b in bins_s:
        sm = state_corr_matrices(
            da, tiles, method=method, bin_s=b, remove_common=remove_common
        )
        for col, r in sm.r.items():
            s = pair_class_summary(r)
            for cls in ("all", *PAIR_CLASSES):
                rows.append({
                    "signal": signal,
                    "bin_s": float(dt if b is None else b),
                    "column": col,
                    "pair_class": cls,
                    "mean_r": s[cls],
                    "null_sd": sm.null_sd[col],
                    "n_tiles": sm.n_tiles[col],
                })
    return pl.DataFrame(rows)
