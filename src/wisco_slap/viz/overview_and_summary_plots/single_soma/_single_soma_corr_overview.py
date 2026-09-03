"""Single-soma correlation overview: synapse-by-synapse correlations by sleep state.

One figure per ``(subject, exp, loc, acq, soma_id)``. Rows are the three
signals the correlation is computed on — matched-filter trace, LS dF and
denoised dF, all raw (see :data:`SIGNALS`); columns are the whole recording
followed by Wake, NREM and REM (whichever occur in the acquisition). Every
matrix in the grid is estimated the same way
(:mod:`wisco_slap.scope.corr.state_matrices`): a common validity mask, equal
valid-duration tiles per state, per-tile centring and Fisher-z averaging,
with a circular-shift surrogate giving each panel its own noise floor.
Rows and columns of every matrix are grouped DMD (most superficial on top)
→ dendrite → ``pos``; the DMD labels carry the depths. Each row shares one
symmetric colour scale so the states can be compared by eye; the per-panel
title gives the noise floor and the mean r within dendrites, between
dendrites of the same DMD, and across DMDs.

:func:`corr_difference_figure` shows state contrasts (``Wake − NREM`` etc.)
in the same layout. The pure figure builders take pre-computed matrices, so
they can be driven from a notebook with in-memory data;
:func:`single_soma_corr_overview_plot` loads everything itself and saves the
PNG.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import fluxplot as fp
import fluxplot.colors as flex
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import wisco_slap as wis
from wisco_slap.scope.corr.plot import plot_slap2_corr_matrix
from wisco_slap.scope.corr.state_matrices import (
    ALL_LABEL,
    DEFAULT_DIFFERENCES,
    DEFAULT_STATES,
    StateCorrDifferences,
    StateCorrMatrices,
    StateTiles,
    common_validity_mask,
    pair_class_summary,
    state_corr_matrices_multi,
    state_differences,
    state_tiles,
)
from wisco_slap.viz.overview_and_summary_plots._spec import PlotSpec

#: Signals (rows), in display order.
SIGNALS: tuple[str, ...] = ("mf", "ls", "denoised")
SIGNAL_LABELS: dict[str, str] = {
    "mf": "matched filter\n(raw)",
    "ls": "LS dF\n(raw)",
    "denoised": "denoised dF\n(raw)",
}
#: iGluSnFR channel of the scopex traces (channel 1 = glutamate, 0 = jRGECO).
CHANNEL = 1
#: Tile length (valid seconds) and the longest NaN gap a tile may bridge.
EPOCH_LENGTH_S = 10.0
MAX_NAN_SPAN_S = 2.0
#: Colour map (registered by ``fluxplot.colors``) and the percentile of |r|
#: (off-diagonal, over all panels of a row) that sets the row's colour limit.
CMAP = "flexoki_diverging"
VMAX_PCT = 99.0
#: Colour of NaN cells and of the masked diagonal.
NAN_COLOR = flex.base300
#: Panel size (inches) and font sizes.
PANEL_IN = 4.2
FS_TITLE = 13
FS_HEADER = 12
FS_STATS = 8
FS_ROW = 11
FS_BRACKET = 8.5


def single_soma_corr_overview_path(
    subject: str, exp: str, loc: str, acq: str, soma_id: str
) -> Path:
    """Return where :func:`single_soma_corr_overview_plot` saves its PNG.

    ``<annotation_root>/OVERVIEW_FIGURES/single_soma/single_soma_corr_overview/
    <subject>--<exp>--<loc>--<acq>--<soma_id>.png``
    """
    return Path(
        os.path.join(
            wis.defs.annotation_root,
            "OVERVIEW_FIGURES",
            "single_soma",
            "single_soma_corr_overview",
            f"{subject}--{exp}--{loc}--{acq}--{soma_id}.png",
        )
    )


def load_single_soma_corr_signals(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    *,
    channel: int = CHANNEL,
) -> dict:
    """Load the three signals of :data:`SIGNALS` for one soma plus its hypnogram.

    Uses the ``"synapse"`` data pipe (dendritic synapses of ``soma_id`` only,
    ``is_valid`` attached) for the LS and denoised traces and derives the
    matched-filter trace with
    :func:`wisco_slap.scope.pro.ls_to_matched_filter`.

    Returns
    -------
    dict
        ``signals`` (``{name: DataArray}`` in :data:`SIGNALS` order),
        ``hypno`` (the acquisition-clipped bout hypnogram),
        ``acq_duration`` (s), ``mask`` (:func:`common_validity_mask` of the
        three signals).
    """
    common = dict(
        subject=subject,
        exp=exp,
        loc=loc,
        acq=acq,
        soma_id=soma_id,
        channel=channel,
        dend_only=True,
        valid_mask=True,
    )
    ls = wis.get.fetch_data_pipe("synapse", trace="ls", **common)
    dn = wis.get.fetch_data_pipe("synapse", trace="denoised", **common)
    if ls.sizes.get("syn_id", 0) < 2:
        raise ValueError(
            f"{subject}/{exp}/{loc}/{acq} {soma_id}: only "
            f"{ls.sizes.get('syn_id', 0)} dendritic synapses — nothing to correlate."
        )
    signals = {
        "mf": wis.scope.pro.ls_to_matched_filter(ls),
        "ls": ls,
        "denoised": dn,
    }
    hypno, acq_dur, *_ = wis.get.acq_sleep_coverage(subject, exp, loc, acq)
    return {
        "signals": signals,
        "hypno": hypno,
        "acq_duration": float(acq_dur),
        "mask": common_validity_mask(signals),
    }


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def _row_vmax(matrices: Sequence[xr.DataArray], pct: float) -> float:
    """Symmetric colour limit of a row: the ``pct`` percentile of off-diagonal |r|."""
    vals = []
    for r in matrices:
        M = np.asarray(r.values, dtype=float)
        n = M.shape[0]
        iu, ju = np.triu_indices(n, k=1)
        vals.append(np.abs(M[iu, ju]))
    v = np.concatenate(vals) if vals else np.array([np.nan])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    return float(max(np.percentile(v, pct), 1e-3))


def _dmd_labels(r: xr.DataArray) -> dict[int, str] | str:
    """One-line ``"DMD d · depth µm"`` label per DMD (``'auto'`` without depths)."""
    if "dmd-depth_1" not in r.coords or "dmd_1" not in r.coords:
        return "auto"
    out: dict[int, str] = {}
    for d, z in zip(
        np.asarray(r.coords["dmd_1"].values, dtype=int),
        np.asarray(r.coords["dmd-depth_1"].values, dtype=float),
        strict=True,
    ):
        if np.isfinite(z):
            out.setdefault(int(d), f"DMD {int(d)} · {z:g} µm")
    return out or "auto"


def _soma_depth(r: xr.DataArray) -> float | None:
    if "soma-depth_1" not in r.coords:
        return None
    vals = np.asarray(r.coords["soma-depth_1"].values, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals[0]) if vals.size else None


def _stats_line(M: xr.DataArray, floor: float, *, what: str = "mean r") -> str:
    s = pair_class_summary(M)
    return (
        f"{what}:  within-dend {s['within_dend']:.3f}   "
        f"between-dend {s['between_dend']:.3f}   "
        f"cross-DMD {s['cross_dmd']:.3f}\n"
        f"noise floor (shift-surrogate SD) {floor:.3f}"
    )


def _matrix_grid(
    rows: Sequence[dict],
    columns: Sequence[str],
    *,
    cmap=CMAP,
    vmax_pct: float = VMAX_PCT,
    panel_in: float = PANEL_IN,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Grid of matrices; one row per entry of ``rows``, one column per name.

    Each row is a dict with ``label`` (row label), ``matrices``
    (``{column: DataArray}``), ``stats`` (``{column: str}`` panel title),
    ``headers`` (``{column: str}``, top row only) and ``cbar`` (colour-bar
    label).
    """
    n_rows, n_cols = len(rows), len(columns)
    left_in, right_in, top_in, bottom_in = 2.3, 1.1, 2.1, 0.4
    hgap_in, vgap_in = 0.25, 0.9
    fig_w = left_in + n_cols * panel_in + (n_cols - 1) * hgap_in + right_in
    fig_h = top_in + n_rows * panel_in + (n_rows - 1) * vgap_in + bottom_in
    # explicit margins below; "none" keeps a style's constrained/tight layout
    # engine from overriding them
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, layout="none"
    )
    fig.subplots_adjust(
        left=left_in / fig_w,
        right=1 - right_in / fig_w,
        top=1 - top_in / fig_h,
        bottom=bottom_in / fig_h,
        wspace=hgap_in / panel_in,
        hspace=vgap_in / panel_in,
    )

    for i, row in enumerate(rows):
        mats = row["matrices"]
        vmax = _row_vmax([mats[c] for c in columns if c in mats], vmax_pct)
        im = None
        for j, col in enumerate(columns):
            ax = axes[i, j]
            if col not in mats:
                ax.set_axis_off()
                continue
            _, _, im = plot_slap2_corr_matrix(
                mats[col],
                ax=ax,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                dmd_order="depth",
                mask_diagonal=True,
                nan_color=NAN_COLOR,
                cbar_label=None,
                show_x_brackets=False,
                show_y_brackets=(j == 0),
                soma_brackets=False,
                dmd_labels=_dmd_labels(mats[col]),
                label_rotation={"dmd": 90},
                boundary_lw=0.5,
                dmd_boundary_lw=1.3,
                bracket_lw=1.1,
                label_fontsize=FS_BRACKET,
            )
            ax.set_title(row["stats"].get(col, ""), fontsize=FS_STATS, loc="center")
            if i == 0 and row.get("headers", {}).get(col):
                ax.text(
                    0.5,
                    1.13,
                    row["headers"][col],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=FS_HEADER,
                    fontweight="bold",
                )
        ax0 = axes[i, 0]
        ax0.text(
            -0.33,
            0.5,
            row["label"],
            transform=ax0.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=FS_ROW,
        )
        if im is not None:
            last = axes[i, n_cols - 1]
            cax = last.inset_axes([1.04, 0.0, 0.035, 1.0])
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(row.get("cbar", ""), fontsize=9)
            cbar.outline.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=FS_TITLE, y=1 - 0.12 / fig_h, va="top")
    return fig, axes


def _column_header(sm: StateCorrMatrices, col: str) -> str:
    n_avail = sm.tiles.n_available.get(col, sm.n_tiles[col])
    header = (
        f"{col}\n{sm.n_tiles[col]} × {sm.tiles.epoch_length:g}-s tiles"
        f" = {sm.valid_seconds[col]:.0f} s valid"
    )
    if n_avail != sm.n_tiles[col]:
        header += f" (of {n_avail} available)"
    return header


def corr_overview_figure(
    mats: Mapping[str, StateCorrMatrices],
    *,
    signal_labels: Mapping[str, str] | None = None,
    columns: Sequence[str] | None = None,
    cmap=CMAP,
    vmax_pct: float = VMAX_PCT,
    panel_in: float = PANEL_IN,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Grid of correlation matrices: one row per signal, one column per state.

    Parameters
    ----------
    mats
        ``{signal name: StateCorrMatrices}`` — rows, in order. All entries
        should come from the same :class:`StateTiles` so the columns line up.
    signal_labels
        Row label per signal name (default :data:`SIGNAL_LABELS`, falling
        back to the name).
    columns
        Column order; default = the columns of the first entry.
    cmap, vmax_pct
        Colour map and the |r| percentile that sets each row's symmetric
        colour limit.
    panel_in
        Side length of one matrix panel in inches.
    title
        Figure suptitle (one or two lines).

    Returns
    -------
    (fig, axes)
        ``axes`` is the ``(n_rows, n_cols)`` array of matrix axes.
    """
    if not mats:
        raise ValueError("`mats` is empty.")
    labels = dict(SIGNAL_LABELS)
    if signal_labels:
        labels.update(signal_labels)
    first = next(iter(mats.values()))
    cols = list(columns) if columns is not None else list(first.columns)
    if not cols:
        raise ValueError("No columns (states) to plot.")
    what = {"sttc": "mean STTC", "partial": "mean partial r"}.get(
        first.method, "mean r"
    )
    rows = []
    for name, sm in mats.items():
        rows.append({
            "label": labels.get(name, name),
            "matrices": sm.r,
            "stats": {
                c: _stats_line(sm.r[c], sm.null_sd.get(c, float("nan")), what=what)
                for c in sm.r
            },
            "headers": {c: _column_header(sm, c) for c in sm.r},
            "cbar": {"sttc": "STTC", "partial": "partial r"}.get(
                sm.method, f"{sm.method.capitalize()} r"
            ),
        })
    return _matrix_grid(
        rows, cols, cmap=cmap, vmax_pct=vmax_pct, panel_in=panel_in, title=title
    )


def corr_difference_figure(
    diffs: Mapping[str, StateCorrDifferences],
    *,
    signal_labels: Mapping[str, str] | None = None,
    columns: Sequence[str] | None = None,
    cmap=CMAP,
    vmax_pct: float = VMAX_PCT,
    panel_in: float = PANEL_IN,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Grid of state-difference matrices: one row per signal, one column per contrast.

    ``diffs`` maps signal name → :func:`state_differences` output. Each panel
    shows ``r[a] − r[b]`` with the mean difference per pair class and the
    propagated noise floor ``sqrt(floor_a² + floor_b²)`` in its title.
    """
    if not diffs:
        raise ValueError("`diffs` is empty.")
    labels = dict(SIGNAL_LABELS)
    if signal_labels:
        labels.update(signal_labels)
    first = next(iter(diffs.values()))
    cols = list(columns) if columns is not None else list(first.columns)
    if not cols:
        raise ValueError("No contrasts to plot.")
    rows = []
    for name, sd in diffs.items():
        sm = sd.source
        headers = {}
        for c in sd.d:
            a, b = sd.pairs[c]
            headers[c] = (
                f"{c}\n{sm.n_tiles.get(a, 0)} vs {sm.n_tiles.get(b, 0)} × "
                f"{sm.tiles.epoch_length:g}-s tiles"
            )
        what = {"sttc": "mean ΔSTTC", "partial": "mean Δ partial r"}.get(
            sm.method, "mean Δr"
        )
        rows.append({
            "label": labels.get(name, name),
            "matrices": sd.d,
            "stats": {
                c: _stats_line(sd.d[c], sd.null_sd.get(c, float("nan")), what=what)
                for c in sd.d
            },
            "headers": headers,
            "cbar": {"sttc": "ΔSTTC", "partial": "Δ partial r"}.get(
                sm.method, f"Δ {sm.method.capitalize()} r"
            ),
        })
    return _matrix_grid(
        rows, cols, cmap=cmap, vmax_pct=vmax_pct, panel_in=panel_in, title=title
    )


# ---------------------------------------------------------------------------
# Compute + routine
# ---------------------------------------------------------------------------


def compute_single_soma_corr_matrices(
    data: dict,
    *,
    epoch_length: float = EPOCH_LENGTH_S,
    max_nan_span: float | None = MAX_NAN_SPAN_S,
    equalise: bool = False,
    seed: int = 0,
    states: Sequence[str] = DEFAULT_STATES,
    method: str = "pearson",
    bin_s: float | None = None,
    remove_common: bool = False,
    signals: Sequence[str] | None = None,
) -> tuple[dict[str, StateCorrMatrices], StateTiles]:
    """Tile the recording once and compute every signal's matrices on those tiles.

    ``data`` is the output of :func:`load_single_soma_corr_signals` (or a
    dict with the same ``signals`` / ``hypno`` / ``mask`` keys). Returns
    ``({signal: StateCorrMatrices}, StateTiles)``.
    """
    sigs = data["signals"]
    if signals is not None:
        sigs = {k: sigs[k] for k in signals}
    mask = data.get("mask")
    if mask is None:
        mask = common_validity_mask(sigs)
    tiles = state_tiles(
        data["hypno"],
        mask,
        states=states,
        epoch_length=epoch_length,
        include_all=True,
        equalise=equalise,
        seed=seed,
        max_nan_span=max_nan_span,
    )
    if not tiles.tiles:
        raise ValueError("No tiles could be cut from this recording.")
    mats = state_corr_matrices_multi(
        sigs, tiles, method=method, bin_s=bin_s, remove_common=remove_common
    )
    return mats, tiles


def compute_single_soma_corr_differences(
    mats: Mapping[str, StateCorrMatrices],
    pairs: Sequence[tuple[str, str]] = DEFAULT_DIFFERENCES,
) -> dict[str, StateCorrDifferences]:
    """:func:`state_differences` for every signal of ``mats``."""
    return {name: state_differences(sm, pairs) for name, sm in mats.items()}


def corr_overview_title(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    mats: Mapping[str, StateCorrMatrices],
) -> str:
    """Two-line suptitle naming the soma, synapse count and the estimator settings."""
    first = next(iter(mats.values()))
    r0 = next(iter(first.r.values()))
    n_syn = r0.sizes["syn_1"]
    depth = _soma_depth(r0)
    depth_txt = f", depth {depth:g} µm" if depth is not None else ""
    t = first.tiles
    p = first.params
    method = first.method
    extras = []
    if p.get("bin_s"):
        extras.append(f"{p['bin_s']:g}-s bins")
    if p.get("remove_common"):
        extras.append("population mean regressed out")
    extra_txt = f"; {', '.join(extras)}" if extras else ""
    match_txt = (
        f"duration-matched: same number of tiles in every column, seed {t.seed}"
        if t.equalised
        else "every available tile; noise floor per panel"
    )
    how = (
        "partial correlation from the pooled per-tile-centred covariance"
        if method == "partial"
        else f"{method} r per {t.epoch_length:g}-s tile (each tile centred), "
        "Fisher-z mean over tiles"
    )
    return (
        f"{subject} | {exp} | {loc} | {acq} | {soma_id}{depth_txt} — "
        f"{n_syn} dendritic synapses, iGluSnFR\n"
        f"{how} ({match_txt}){extra_txt}; diagonal masked; "
        f"colour scale shared within each row"
    )


def single_soma_corr_overview_plot(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    *,
    data: dict | None = None,
    epoch_length: float = EPOCH_LENGTH_S,
    equalise: bool = False,
    seed: int = 0,
    method: str = "pearson",
    bin_s: float | None = None,
    remove_common: bool = False,
    signals: Sequence[str] | None = None,
    dpi: int = 150,
    verbose: bool = True,
    overwrite: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Correlation-matrix overview of one soma's dendritic synapses by sleep state.

    Rows: the signals in :data:`SIGNALS` (subset via ``signals``). Columns:
    whole recording, then Wake / NREM / REM as present. Estimator and layout
    are described in the module docstring and in
    :mod:`wisco_slap.scope.corr.state_matrices`.

    The figure is saved to :func:`single_soma_corr_overview_path` when no
    file exists there yet, or when ``overwrite=True``; it is always built and
    returned.

    Parameters
    ----------
    subject, exp, loc, acq, soma_id
        Identifiers, e.g. ``"alnair", "exp_1", "loc_D", "acq_1", "soma1"``.
    data
        Output of :func:`load_single_soma_corr_signals`; loaded here when None.
    epoch_length
        Valid seconds per tile.
    equalise
        Use the same number of tiles in every column (the sparsest state's
        count). Default False: every tile of every state is used and each
        panel reports its own noise floor.
    seed
        Seed of the tile subsample (only with ``equalise=True``).
    method, bin_s, remove_common
        Forwarded to :func:`wisco_slap.scope.corr.state_matrices.state_corr_matrices`.
    signals
        Subset / order of :data:`SIGNALS` to show as rows.
    dpi
        Resolution of the saved PNG.
    verbose
        Print the tile counts and noise floors.
    overwrite
        Re-save even if the PNG already exists.

    Returns
    -------
    (fig, axes)
    """
    fp.style.use_lighttable()
    wis.util.checks.validate_ids(exp, loc, acq)
    if data is None:
        data = load_single_soma_corr_signals(subject, exp, loc, acq, soma_id)
    mats, tiles = compute_single_soma_corr_matrices(
        data,
        epoch_length=epoch_length,
        equalise=equalise,
        seed=seed,
        method=method,
        bin_s=bin_s,
        remove_common=remove_common,
        signals=signals,
    )
    if verbose:
        print(
            f"tiles ({tiles.epoch_length:g} s valid each): "
            + ", ".join(
                f"{k} {v}/{tiles.n_available.get(k, v)}"
                for k, v in tiles.n_tiles.items()
            )
        )
        for name, sm in mats.items():
            print(
                f"  {name}: null SD "
                + ", ".join(f"{k} {v:.3f}" for k, v in sm.null_sd.items())
            )
    fig, axes = corr_overview_figure(
        mats, title=corr_overview_title(subject, exp, loc, acq, soma_id, mats)
    )
    out_path = single_soma_corr_overview_path(subject, exp, loc, acq, soma_id)
    if overwrite or not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"saved {out_path}")
    elif verbose:
        print(f"exists, not overwritten: {out_path}")
    return fig, axes


PLOT_SPEC = PlotSpec(
    name="single_soma_corr_overview",
    plot=single_soma_corr_overview_plot,
    path=single_soma_corr_overview_path,
)

__all__ = [
    "ALL_LABEL",
    "PLOT_SPEC",
    "SIGNALS",
    "SIGNAL_LABELS",
    "compute_single_soma_corr_differences",
    "compute_single_soma_corr_matrices",
    "corr_difference_figure",
    "corr_overview_figure",
    "corr_overview_title",
    "load_single_soma_corr_signals",
    "single_soma_corr_overview_path",
    "single_soma_corr_overview_plot",
]
