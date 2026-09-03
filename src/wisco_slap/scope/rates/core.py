"""Per-synapse event rates: shared machinery + the core state/bout entry point.

The engine layer of :mod:`wisco_slap.scope.rates`. Everything in this module is
either (a) the validation / bookkeeping shared by every rate function, or
(b) the base public API that the other modules specialise.

Public:

- :func:`compute_rate` — per-syn rate ``by='state'`` or ``by='bout'``.
- :func:`compute_rate_by_bout` / :func:`compute_rate_by_state` — thin named
  wrappers around :func:`compute_rate`.

Internal, imported by :mod:`.rolling`, :mod:`.fraction` and :mod:`.epoch`:

- :func:`_prepare` — validate the ``peaks_df`` ↔ ``final_peaks`` pairing and
  return the arrays every rate function needs.
- :func:`_rate_over_partition` — count events and valid seconds over an
  arbitrary set of non-overlapping bins, then divide.
- :func:`_assign_samples_to_bouts`, :func:`_rate_expr`, :func:`_order_columns`.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import polars as pl
import xarray as xr

from ...util.validity.events import _resolve_states
from ...util.validity.mask import _median_dt, resolve_mask

_SYN_DIM_CANDIDATES: tuple[str, ...] = ("syn_id", "soma_id")

# Output columns, in order, that every rate function ends with.
_RATE_COLS: tuple[str, ...] = ("n_events", "valid_duration_s", "rate_hz")


# --------------------------------------------------------------- small helpers


def _detect_syn_key(final_peaks: xr.DataArray, peaks_df: pl.DataFrame) -> str:
    """Return the synapse-like key (``syn_id`` or ``soma_id``).

    Detected from ``final_peaks``'s dims and required to also be a column of
    ``peaks_df``.
    """
    for cand in _SYN_DIM_CANDIDATES:
        if cand in final_peaks.dims:
            if cand not in peaks_df.columns:
                raise ValueError(
                    f"`final_peaks` has a {cand!r} dim but `peaks_df` has no "
                    f"{cand!r} column — are they from the same detection call?"
                )
            return cand
    raise ValueError(
        f"`final_peaks` must have one of {_SYN_DIM_CANDIDATES} as a dim "
        f"(got dims {final_peaks.dims})."
    )


def _syn_meta_table(final_peaks: xr.DataArray, syn_key: str) -> pl.DataFrame:
    """One row per synapse with every metadata coordinate merged in.

    Pulls the complete synapse universe (including synapses that ended up with
    zero surviving events) and every coordinate defined on the synapse dim,
    plus scalar coordinates (e.g. ``channel``) broadcast as constant columns.
    """
    data: dict[str, np.ndarray] = {}
    for name, co in final_peaks.coords.items():
        if co.dims == (syn_key,):
            data[name] = np.asarray(co.values)
    df = pl.DataFrame(data)

    # Scalar coords → constant columns (mirrors how they appear in peaks_df).
    scalar_cols = {
        name: co.item()
        for name, co in final_peaks.coords.items()
        if co.ndim == 0 and name not in df.columns
    }
    if scalar_cols:
        df = df.with_columns([pl.lit(v).alias(k) for k, v in scalar_cols.items()])

    ordered = [syn_key] + [c for c in df.columns if c != syn_key]
    return df.select(ordered)


def _assign_samples_to_bouts(time_values: np.ndarray, hypno) -> np.ndarray:
    """Map each sample to the index of the bout that contains it (``-1`` if none).

    Thin wrapper over ``Hypnogram.bout_index`` so the boundary rule is defined
    in exactly one place: bouts are half-open ``[start_time, end_time)``, a
    sample on a shared boundary (``end_time[i] == start_time[i+1]``) goes to the
    later bout. Using the same rule for the denominator (per-sample) and the
    numerator (each peak via its ``peak_idx``) keeps them consistent with
    :func:`~wisco_slap.util.validity.add_valid_duration`.
    """
    return np.asarray(hypno.bout_index(time_values), dtype=np.int64)


def _hypno_context_cols(
    hypno, descriptor: pl.DataFrame, syn_meta_cols: Iterable[str]
) -> list[str]:
    """Hypnogram columns eligible to ride along as per-bout context.

    Anything the descriptor already carries (``state``, ``start_time``,
    ``end_time``, ``duration``) is skipped silently — the descriptor's copy is
    the same value read from the same row, so nothing is lost. A hypnogram
    column that would instead shadow a synapse-metadata or rate column is
    skipped with a warning, because that one *is* a loss the caller should
    know about.
    """
    carried = set(descriptor.columns)
    reserved = (set(syn_meta_cols) | set(_RATE_COLS)) - carried

    keep: list[str] = []
    shadowed: list[str] = []
    for col in hypno.df.columns:
        if col in carried:
            continue
        (shadowed if col in reserved else keep).append(col)

    if shadowed:
        warnings.warn(
            f"Hypnogram column(s) {shadowed} were not joined as bout context: "
            "the names collide with a synapse-metadata or rate column. Rename "
            "them on the hypnogram if you need them in the output.",
            UserWarning,
            stacklevel=3,
        )
    return keep


def _attach_hypno_context(
    descriptor: pl.DataFrame, hypno, syn_meta_cols: Iterable[str]
) -> pl.DataFrame:
    """Widen a per-bout descriptor with the hypnogram's own extra columns.

    ``descriptor['bout_idx']`` *is* the row index of ``hypno.df``, so joining
    against a freshly taken row index is positional by construction — there is
    no user-maintained bout key that can drift out of step with the one the
    rate machinery binned on.
    """
    extra = _hypno_context_cols(hypno, descriptor, syn_meta_cols)
    if not extra:
        return descriptor
    context = (
        hypno.df.select(extra)
        .with_row_index("_bout_row")
        .with_columns(pl.col("_bout_row").cast(pl.Int64))
    )
    return descriptor.join(
        context, left_on="bout_idx", right_on="_bout_row", how="left"
    )


def _valid_seconds_per_bin(
    bin_of_sample: np.ndarray, valid_bool: np.ndarray, dt: float, n_bins: int
) -> np.ndarray:
    """Seconds of valid time per bin id.

    ``bin_of_sample`` maps each sample to a bin id (``-1`` = no bin). Only
    valid samples contribute. Returns a length-``n_bins`` float array.
    """
    if n_bins == 0:
        return np.zeros(0, dtype=float)
    vb = bin_of_sample[valid_bool]
    vb = vb[vb >= 0]
    counts = np.bincount(vb, minlength=n_bins)[:n_bins]
    return counts.astype(np.float64) * dt


def _require_float_hypno(hypno) -> None:
    """Raise unless the hypnogram's time axis is numeric (float seconds)."""
    if not hypno.df.schema["start_time"].is_numeric():
        raise ValueError(
            "hypnogram must have a numeric (float-seconds) time axis aligned "
            "with the scopex time coordinate; got a datetime hypnogram. "
            "Load it with wis.get.bout_hypno / acq_sleep_coverage."
        )


def _prepare(
    peaks_df: pl.DataFrame,
    hypno,
    final_peaks: xr.DataArray,
    mask: xr.DataArray | None,
    time_col: str,
):
    """Validate inputs and pull out the shared arrays used by every rate fn.

    Returns
    -------
    syn_key : str
    syn_meta : pl.DataFrame
    time_values : np.ndarray
    dt : float
    n_samples : int
    valid_bool : np.ndarray[bool]
    pk_idx : np.ndarray[int]      # sample index of each peak
    pk_valid : np.ndarray[bool]   # whether each peak sits on a valid sample
    """
    if "time" not in final_peaks.dims:
        raise ValueError("`final_peaks` must have a 'time' dim.")
    _require_float_hypno(hypno)

    syn_key = _detect_syn_key(final_peaks, peaks_df)
    syn_meta = _syn_meta_table(final_peaks, syn_key)

    time_values = np.asarray(final_peaks["time"].values, dtype=float)
    n_samples = time_values.size
    if n_samples < 2:
        raise ValueError("`final_peaks` time axis needs >= 2 samples.")
    dt = _median_dt(time_values)
    valid_bool = np.asarray(resolve_mask(final_peaks, mask).values, dtype=bool)

    for col in ("peak_idx", time_col):
        if col not in peaks_df.columns:
            raise ValueError(f"`peaks_df` is missing required column {col!r}.")

    pk_idx = peaks_df["peak_idx"].to_numpy()
    if pk_idx.size:
        if pk_idx.min() < 0 or pk_idx.max() >= n_samples:
            raise ValueError(
                "`peaks_df['peak_idx']` has values outside `final_peaks`'s time "
                f"grid (0..{n_samples - 1}). `peaks_df` and `final_peaks` must "
                "come from the same master_event_detector call."
            )
        # Cross-check: the peak's time must match the sample it indexes.
        drift = np.abs(time_values[pk_idx] - peaks_df[time_col].to_numpy())
        if np.nanmax(drift) > dt:
            raise ValueError(
                "`peaks_df['peak_time']` does not match `final_peaks['time']` at "
                "the stored peak indices — the two inputs are not aligned "
                "(different detection call or a re-sliced time axis)."
            )
        pk_valid = valid_bool[pk_idx]
    else:
        pk_valid = np.zeros(0, dtype=bool)

    return syn_key, syn_meta, time_values, dt, n_samples, valid_bool, pk_idx, pk_valid


def _rate_over_partition(
    peaks_df: pl.DataFrame,
    syn_key: str,
    pk_idx: np.ndarray,
    pk_valid: np.ndarray,
    bin_of_sample: np.ndarray,
    valid_bool: np.ndarray,
    dt: float,
    n_bins: int,
    descriptor: pl.DataFrame,
    syn_meta: pl.DataFrame,
    complete_grid: bool,
) -> pl.DataFrame:
    """Per-syn rate over a set of non-overlapping bins.

    ``bin_of_sample`` maps each sample to a bin id (``-1`` = excluded).
    ``descriptor`` has one row per bin: a ``_bin_id`` column plus any
    identity columns to carry through (``state``, ``bout_idx``, ``frac_bin``…).
    Valid-time per bin is computed here (from ``bin_of_sample`` ∧ validity) so
    numerator and denominator are guaranteed to share the same sample set.
    """
    vs = _valid_seconds_per_bin(bin_of_sample, valid_bool, dt, n_bins)
    vs_df = pl.DataFrame(
        {"_bin_id": np.arange(n_bins, dtype=np.int64), "valid_duration_s": vs}
    )
    descriptor = descriptor.join(vs_df, on="_bin_id", how="left")

    # Numerator: each valid peak's bin, counted per synapse.
    pk_bin = bin_of_sample[pk_idx] if pk_idx.size else np.zeros(0, dtype=np.int64)
    work = (
        peaks_df.select(syn_key)
        .with_columns(
            _bin_id=pl.Series("_bin_id", pk_bin),
            _valid=pl.Series("_valid", pk_valid),
        )
        .filter(pl.col("_valid") & (pl.col("_bin_id") >= 0))
    )
    counts = work.group_by([syn_key, "_bin_id"]).agg(pl.len().alias("n_events"))

    grid = syn_meta.join(descriptor, how="cross")
    out = grid.join(counts, on=[syn_key, "_bin_id"], how="left").with_columns(
        pl.col("n_events").fill_null(0).cast(pl.Int64)
    )
    if not complete_grid:
        out = out.filter(pl.col("n_events") > 0)
    out = out.with_columns(_rate_expr()).drop("_bin_id")
    return out


def _rate_expr() -> pl.Expr:
    """``rate_hz = n_events / valid_duration_s``; NaN when no valid time."""
    return (
        pl.when(pl.col("valid_duration_s") > 0)
        .then(pl.col("n_events") / pl.col("valid_duration_s"))
        .otherwise(float("nan"))
        .alias("rate_hz")
    )


def _order_columns(df: pl.DataFrame, syn_meta_cols: list[str]) -> pl.DataFrame:
    """Put synapse metadata first, rate columns last, descriptors in between."""
    meta = [c for c in syn_meta_cols if c in df.columns]
    rate = [c for c in _RATE_COLS if c in df.columns]
    middle = [c for c in df.columns if c not in meta and c not in rate]
    return df.select(meta + middle + rate)


# --------------------------------------------------------------- core: by state/bout


def compute_rate(
    peaks_df: pl.DataFrame,
    hypno,
    final_peaks: xr.DataArray,
    *,
    by: str = "state",
    mask: xr.DataArray | None = None,
    states: str | Iterable[str] = "all",
    complete_grid: bool = True,
    min_valid_duration_s: float = 0.0,
    join_hypno_context: bool = True,
    time_col: str = "peak_time",
) -> pl.DataFrame:
    """Per-synapse event rate, pooled ``by='state'`` or ``by='bout'``.

    The denominator is **valid** time (samples where every analysed synapse was
    non-NaN, per ``final_peaks``'s ``is_valid`` coord), never wall-clock time.
    Events are counted from ``peaks_df`` — so any filtering you apply to
    ``peaks_df`` first (SNR floor, one dendrite, a synapse-type…) is reflected
    in the numerator.

    Parameters
    ----------
    peaks_df : pl.DataFrame
        Event table from :func:`wisco_slap.scope.master_event_detector`. Must
        carry ``peak_idx`` and ``peak_time`` (or ``time_col``) plus the
        synapse key column. Pre-filter it to whatever event subset you want.
    hypno : electro_py.hypno.Hypnogram
        Hypnogram on the same (float-seconds) clock as ``final_peaks['time']``.
        Pass any span or subset — ``hypno_acq`` for one acquisition, a
        state-filtered hypno, a single bout, etc.
    final_peaks : xr.DataArray
        The boolean event array from the same detection call, dims
        ``(syn_id, time)`` (or ``soma_id``), carrying the ``is_valid`` time
        coordinate and the per-synapse metadata coords. Used for the validity
        denominator, the sample grid, and the complete synapse universe — its
        boolean *values* are not used for the numerator (that comes from
        ``peaks_df``).
    by : {'state', 'bout'}
        ``'state'`` pools every bout of a state together (one row per
        synapse per state). ``'bout'`` keeps bouts separate (one row per
        synapse per bout).
    mask : xr.DataArray, optional
        Explicit 1-D validity mask over ``time``. Defaults to
        ``final_peaks``'s ``is_valid`` coord.
    states : str or iterable of str
        ``'all'`` (default) or a state / list of states. For ``by='state'``
        this selects which states get a row; for ``by='bout'`` it selects
        which bouts (by their state) are kept.
    complete_grid : bool
        If True (default), emit a row for every (synapse × bin) — synapses
        with zero events in a bin appear with ``n_events = 0``. If False, only
        (synapse, bin) pairs with at least one event appear.
    min_valid_duration_s : float
        Drop output rows whose ``valid_duration_s`` is below this. Default 0
        keeps everything (bins with no valid time get ``rate_hz = NaN``).
    join_hypno_context : bool
        ``by='bout'`` only. If True (default), every *extra* column the
        hypnogram carries — bandpower, elapsed-time, ``valid_duration``,
        anything else attached per bout — rides along into the output, joined
        on the bout each row was binned over. This saves the usual follow-up
        merge, and it joins on the hypnogram's own row index rather than a
        hand-maintained bout key, so the two cannot drift apart. Columns the
        descriptor already reports (``state``, ``start_time``, ``end_time``,
        ``duration``) are not duplicated; a hypnogram column whose name would
        shadow a synapse-metadata or rate column is skipped with a warning.
        Ignored for ``by='state'``, where one row pools many bouts and a
        per-bout value has no single meaning.
    time_col : str
        Name of the peak-time column in ``peaks_df`` (default ``'peak_time'``).

    Returns
    -------
    pl.DataFrame
        One row per (synapse, state) or (synapse, bout). Columns: all
        per-synapse metadata, then the bin descriptor
        (``state`` — and for ``by='bout'`` also ``bout_idx``, ``start_time``,
        ``end_time``, ``duration``, followed by the hypnogram's extra columns
        when ``join_hypno_context``; for ``by='state'`` also ``n_bouts``,
        ``wall_duration_s``), then ``n_events`` (i64), ``valid_duration_s``
        (f64), ``rate_hz`` (f64, NaN when ``valid_duration_s == 0``).
    """
    if by not in ("state", "bout"):
        raise ValueError(f"`by` must be 'state' or 'bout', got {by!r}.")

    (
        syn_key,
        syn_meta,
        time_values,
        dt,
        n_samples,
        valid_bool,
        pk_idx,
        pk_valid,
    ) = _prepare(peaks_df, hypno, final_peaks, mask, time_col)

    bout_of_sample = _assign_samples_to_bouts(time_values, hypno)
    n_bouts = hypno.df.height
    bout_states = hypno.df["state"].to_list()
    requested = set(_resolve_states(hypno, states))

    if by == "bout":
        # One bin per bout; keep only bouts whose state is requested.
        keep_bouts = [i for i in range(n_bouts) if bout_states[i] in requested]
        descriptor = (
            hypno.df.with_row_index("bout_idx")
            .with_columns(pl.col("bout_idx").cast(pl.Int64).alias("_bin_id"))
            .filter(pl.col("_bin_id").is_in(keep_bouts))
            .select(
                "_bin_id",
                pl.col("bout_idx").cast(pl.Int64),
                "state",
                "start_time",
                "end_time",
                "duration",
            )
        )
        if join_hypno_context:
            descriptor = _attach_hypno_context(descriptor, hypno, syn_meta.columns)
        out = _rate_over_partition(
            peaks_df, syn_key, pk_idx, pk_valid, bout_of_sample, valid_bool,
            dt, n_bouts, descriptor, syn_meta, complete_grid,
        )
    else:  # by == "state"
        req_list = list(_resolve_states(hypno, states))
        state_to_bin = {s: i for i, s in enumerate(req_list)}
        # sample -> state bin, via the state of the bout each sample falls in.
        if n_bouts:
            bout_to_statebin = np.array(
                [state_to_bin.get(st, -1) for st in bout_states], dtype=np.int64
            )
            safe = np.where(bout_of_sample >= 0, bout_of_sample, 0)
            bin_of_sample = np.where(
                bout_of_sample >= 0, bout_to_statebin[safe], -1
            )
        else:
            bin_of_sample = np.full(n_samples, -1, dtype=np.int64)

        # Per-state wall duration & bout count from the hypno.
        state_summary = (
            hypno.df.group_by("state")
            .agg(
                pl.len().alias("n_bouts"),
                pl.col("duration").sum().alias("wall_duration_s"),
            )
        )
        descriptor = pl.DataFrame(
            {"_bin_id": np.arange(len(req_list), dtype=np.int64), "state": req_list}
        ).join(state_summary, on="state", how="left").with_columns(
            pl.col("n_bouts").fill_null(0).cast(pl.Int64),
            pl.col("wall_duration_s").fill_null(0.0),
        )
        out = _rate_over_partition(
            peaks_df, syn_key, pk_idx, pk_valid, bin_of_sample, valid_bool,
            dt, len(req_list), descriptor, syn_meta, complete_grid,
        )

    if min_valid_duration_s > 0:
        out = out.filter(pl.col("valid_duration_s") >= min_valid_duration_s)
    return _order_columns(out, syn_meta.columns)


def compute_rate_by_bout(
    peaks_df: pl.DataFrame,
    hypno,
    final_peaks: xr.DataArray,
    **kwargs,
) -> pl.DataFrame:
    """Per-synapse event rate, one row per (synapse, bout).

    Thin wrapper around :func:`compute_rate` with ``by='bout'``. See there for
    all keyword arguments.
    """
    if "by" in kwargs:
        raise TypeError("compute_rate_by_bout does not accept `by`.")
    return compute_rate(peaks_df, hypno, final_peaks, by="bout", **kwargs)


def compute_rate_by_state(
    peaks_df: pl.DataFrame,
    hypno,
    final_peaks: xr.DataArray,
    **kwargs,
) -> pl.DataFrame:
    """Per-synapse event rate, one row per (synapse, state).

    The *total* rate per state: all events and all valid time in the state are
    summed, then divided — not an average of per-bout rates. Thin wrapper
    around :func:`compute_rate` with ``by='state'``.
    """
    if "by" in kwargs:
        raise TypeError("compute_rate_by_state does not accept `by`.")
    return compute_rate(peaks_df, hypno, final_peaks, by="state", **kwargs)
