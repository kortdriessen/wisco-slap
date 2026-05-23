"""Visualization for SLAP2 correlation analysis.

Holds the original :func:`plot_slap2_corr_matrix` (unchanged) plus several
new state-comparison and outlier-locator helpers that consume the long-form
correlation tables produced by :mod:`wisco_slap.scope.corr.state_compare`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr
from matplotlib.transforms import blended_transform_factory

LabelOpt = Union[bool, str, None, Mapping]
from kplot.colors import flex

# ---------------------------------------------------------------------------
# Internal label helpers (used by plot_slap2_corr_matrix)
# ---------------------------------------------------------------------------


def _label_off(opt: LabelOpt) -> bool:
    return (
        opt is None or opt is False or (isinstance(opt, str) and opt.lower() == "off")
    )


def _default_label(level: str, key) -> str:
    if level == "dmd":
        return f"DMD {key}"
    return str(key)


def _resolve_label(opt: LabelOpt, key, level: str) -> str:
    if _label_off(opt):
        return " "
    if isinstance(opt, Mapping):
        return str(opt.get(key, _default_label(level, key)))
    return _default_label(level, key)


def _safe_str(v) -> str:
    if v is None:
        return "?"
    if isinstance(v, float) and np.isnan(v):
        return "?"
    s = str(v)
    return s if s and s.lower() != "nan" else "?"


# ---------------------------------------------------------------------------
# Original correlation-matrix plot
# ---------------------------------------------------------------------------


def plot_slap2_corr_matrix(
    corr: xr.DataArray,
    DMD_first: int = 1,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    boundary_color: str = "0.4",
    boundary_lw: float = 0.6,
    boundary_alpha: float = 1.0,
    dend_labels: LabelOpt = "auto",
    dmd_labels: LabelOpt = "auto",
    soma_labels: LabelOpt = "auto",
    cbar_label: str | None = "Pearson r",
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    bracket_color: str = "0.15",
    bracket_lw: float = 1.4,
    label_fontsize: float = 8.5,
    show_x_brackets: bool = True,
    show_y_brackets: bool = True,
    group_by_soma: bool = False,
    soma_color: str | None = None,
    dend_type_colors: Mapping[str, str] | None = None,
):
    """Plot a synapse-by-synapse correlation matrix.

    Rows and columns are ordered identically. Tree-style brackets mark the
    dendrite, middle (DMD or dend_type), and soma blocks on the left axis;
    matching brackets are drawn below the bottom axis. Each `*_labels` argument
    controls only the text on its level (`'auto'` / dict / `False`-or-`'off'`).

    Two ordering modes:
      - `group_by_soma=False` (default): DMD `DMD_first` first, then soma,
        then dendrite, then `pos`. A soma that appears on both DMDs shows up
        as two separate soma blocks (one inside each DMD block). Middle level
        is DMD.
      - `group_by_soma=True`: soma first, then dend_type (alphabetical, e.g.
        Apical → Basal) if a `dend_type` coord is present, otherwise DMD
        (`DMD_first` first within each soma), then dendrite, then `pos`. All
        dendrites belonging to a given soma are contiguous. Soma labels are
        coloured red and dend_type labels are colour-coded by default
        (Apical = orange, Basal = blue) to give the level a quick visual key.

    Parameters
    ----------
    corr : xr.DataArray
        Output of `pairwise_pearson_corr`, with dims ('syn_1', 'syn_2') and the
        usual scopex coords mirrored on both dims as `<name>_1` / `<name>_2`.
    DMD_first : int
        Which DMD (1 or 2) is plotted first within its grouping level.
    group_by_soma : bool
        If True, group synapses by soma first (see above).
    soma_color : str | None
        Colour for the soma-level brackets and labels. None → red when
        `group_by_soma`, otherwise `bracket_color`.
    dend_type_colors : Mapping[str, str] | None
        Mapping from `dend_type` value → colour for the middle-level brackets
        and labels (only consulted when the middle level is `dend_type`). None
        → `{"Apical"/"apical": orange, "Basal"/"basal": blue}` when
        `group_by_soma`, otherwise empty (all middle-level brackets use
        `bracket_color`).
    """
    if "syn_1" not in corr.dims or "syn_2" not in corr.dims:
        raise ValueError("`corr` must have dims ('syn_1', 'syn_2').")
    n = corr.sizes["syn_1"]
    if n == 0:
        raise ValueError("Empty correlation matrix.")
    if corr.sizes["syn_2"] != n:
        raise ValueError("Correlation matrix is not square.")

    def _coord(name):
        return corr.coords[name].values if name in corr.coords else None

    dmds_raw = _coord("dmd_1")
    if dmds_raw is None:
        raise ValueError("`corr` is missing 'dmd_1' coord; cannot group by DMD.")
    dmds = np.asarray(dmds_raw, dtype=int)

    somas_raw = _coord("soma-ID_1")
    somas = (
        np.asarray([_safe_str(v) for v in somas_raw])
        if somas_raw is not None
        else np.array(["?"] * n)
    )
    dends_raw = _coord("dend-ID_1")
    dends = (
        np.asarray([_safe_str(v) for v in dends_raw])
        if dends_raw is not None
        else np.array(["?"] * n)
    )
    pos_raw = _coord("pos_1")
    if pos_raw is not None:
        poses = np.asarray(pos_raw, dtype=float)
        poses = np.where(np.isnan(poses), np.inf, poses)
    else:
        poses = np.zeros(n)

    dend_type_raw = _coord("dend_type_1")
    if dend_type_raw is not None:
        dend_types = np.asarray([_safe_str(v) for v in dend_type_raw])
        has_dend_type = bool(np.any(dend_types != "?"))
    else:
        dend_types = np.array(["?"] * n)
        has_dend_type = False
    use_dend_type_middle = group_by_soma and has_dend_type

    if group_by_soma:
        eff_soma_color = soma_color if soma_color is not None else "tab:red"
        eff_dend_type_colors: Mapping[str, str] = (
            dend_type_colors
            if dend_type_colors is not None
            else {
                "Apical": "tab:orange",
                "apical": "tab:orange",
                "Basal": "tab:blue",
                "basal": "tab:blue",
            }
        )
    else:
        eff_soma_color = soma_color if soma_color is not None else bracket_color
        eff_dend_type_colors = dend_type_colors if dend_type_colors is not None else {}

    unique_dmds = sorted(set(dmds.tolist()))
    if DMD_first in unique_dmds:
        dmd_order = [DMD_first] + [d for d in unique_dmds if d != DMD_first]
    else:
        dmd_order = unique_dmds
    dmd_rank = {d: i for i, d in enumerate(dmd_order)}

    if group_by_soma:
        middle_for_sort: list = (
            dend_types.tolist() if use_dend_type_middle else [dmd_rank[d] for d in dmds]
        )
        sort_tuples = list(
            zip(
                somas.tolist(),
                middle_for_sort,
                dends.tolist(),
                poses.tolist(),
                range(n),
                strict=True,
            )
        )
    else:
        sort_tuples = list(
            zip(
                [dmd_rank[d] for d in dmds],
                somas.tolist(),
                dends.tolist(),
                poses.tolist(),
                range(n),
                strict=True,
            )
        )
    sort_tuples.sort()
    order_idx = [t[-1] for t in sort_tuples]

    M = corr.isel(syn_1=order_idx, syn_2=order_idx).values
    dmds = dmds[order_idx]
    somas = somas[order_idx]
    dends = dends[order_idx]
    dend_types = dend_types[order_idx]

    dend_blocks: list[tuple[str, int, int]] = []
    soma_blocks: list[tuple[str, int, int]] = []
    middle_blocks: list[tuple[object, int, int]] = []
    cur_dend_key: object = None
    cur_soma_key: object = None
    cur_mid_key: object = None
    dend_start = soma_start = mid_start = 0

    def _mid_label(i: int) -> object:
        return dend_types[i] if use_dend_type_middle else int(dmds[i])

    for i in range(n):
        dk = (int(dmds[i]), somas[i], dends[i])
        if group_by_soma:
            sk: object = somas[i]
            mk: object = (
                (somas[i], dend_types[i])
                if use_dend_type_middle
                else (somas[i], int(dmds[i]))
            )
        else:
            sk = (int(dmds[i]), somas[i])
            mk = int(dmds[i])
        if dk != cur_dend_key:
            if cur_dend_key is not None:
                dend_blocks.append((dends[i - 1], dend_start, i - 1))
            cur_dend_key = dk
            dend_start = i
        if sk != cur_soma_key:
            if cur_soma_key is not None:
                soma_blocks.append((somas[i - 1], soma_start, i - 1))
            cur_soma_key = sk
            soma_start = i
        if mk != cur_mid_key:
            if cur_mid_key is not None:
                middle_blocks.append((_mid_label(i - 1), mid_start, i - 1))
            cur_mid_key = mk
            mid_start = i
    dend_blocks.append((dends[n - 1], dend_start, n - 1))
    soma_blocks.append((somas[n - 1], soma_start, n - 1))
    middle_blocks.append((_mid_label(n - 1), mid_start, n - 1))

    if ax is None:
        if figsize is None:
            side = max(5.0, min(14.0, n * 0.06 + 3.0))
            figsize = (side, side)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    extent = (-0.5, n - 0.5, n - 0.5, -0.5)
    im = ax.imshow(
        M,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )

    for _, top_row, _ in dend_blocks[1:]:
        ax.axhline(
            top_row - 0.5, color=boundary_color, lw=boundary_lw, alpha=boundary_alpha
        )
        ax.axvline(
            top_row - 0.5, color=boundary_color, lw=boundary_lw, alpha=boundary_alpha
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    middle_name = "dend_type" if use_dend_type_middle else "dmd"
    levels: list[tuple[str, list[tuple], LabelOpt]] = [
        ("dend", dend_blocks, dend_labels),
        (middle_name, middle_blocks, dmd_labels),
        ("soma", soma_blocks, soma_labels),
    ]

    def _color_for(level_name: str, key) -> str:
        if level_name == "soma":
            return eff_soma_color
        if level_name == "dend_type":
            return eff_dend_type_colors.get(str(key), bracket_color)
        return bracket_color

    bracket_step = 0.045
    base = -0.012
    cap = 0.005
    label_pad = 0.006

    if show_y_brackets:
        trans_y = blended_transform_factory(ax.transAxes, ax.transData)
        for lvl_idx, (name, blocks, opt) in enumerate(levels):
            x_b = base - lvl_idx * bracket_step
            x_lab = x_b - label_pad
            for key, top_row, bot_row in blocks:
                clr = _color_for(name, key)
                y_top = top_row - 0.5
                y_bot = bot_row + 0.5
                ax.plot(
                    [x_b, x_b],
                    [y_top, y_bot],
                    color=clr,
                    lw=bracket_lw,
                    transform=trans_y,
                    clip_on=False,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [x_b, x_b + cap],
                    [y_top, y_top],
                    color=clr,
                    lw=bracket_lw,
                    transform=trans_y,
                    clip_on=False,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [x_b, x_b + cap],
                    [y_bot, y_bot],
                    color=clr,
                    lw=bracket_lw,
                    transform=trans_y,
                    clip_on=False,
                    solid_capstyle="butt",
                )
                ax.text(
                    x_lab,
                    (y_top + y_bot) / 2,
                    _resolve_label(opt, key, name),
                    transform=trans_y,
                    ha="right",
                    va="center",
                    fontsize=label_fontsize,
                    color=clr,
                    clip_on=False,
                )

    if show_x_brackets:
        trans_x = blended_transform_factory(ax.transData, ax.transAxes)
        for lvl_idx, (name, blocks, opt) in enumerate(levels):
            y_b = base - lvl_idx * bracket_step
            y_lab = y_b - label_pad
            for key, left_col, right_col in blocks:
                clr = _color_for(name, key)
                x_left = left_col - 0.5
                x_right = right_col + 0.5
                ax.plot(
                    [x_left, x_right],
                    [y_b, y_b],
                    color=clr,
                    lw=bracket_lw,
                    transform=trans_x,
                    clip_on=False,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [x_left, x_left],
                    [y_b, y_b + cap],
                    color=clr,
                    lw=bracket_lw,
                    transform=trans_x,
                    clip_on=False,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [x_right, x_right],
                    [y_b, y_b + cap],
                    color=clr,
                    lw=bracket_lw,
                    transform=trans_x,
                    clip_on=False,
                    solid_capstyle="butt",
                )
                ax.text(
                    (x_left + x_right) / 2,
                    y_lab,
                    _resolve_label(opt, key, name),
                    transform=trans_x,
                    ha="center",
                    va="top",
                    fontsize=label_fontsize,
                    color=clr,
                    clip_on=False,
                )

    if cbar_label is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(cbar_label)
        cbar.outline.set_visible(False)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return fig, ax, im


# ---------------------------------------------------------------------------
# State-comparison plots — consume the long-form correlation table
# ---------------------------------------------------------------------------


def _table_to_matrix(
    df: pl.DataFrame,
    *,
    state: str,
    value: str = "r",
) -> xr.DataArray:
    """Reconstruct a synapse-by-synapse correlation matrix from a long-form table.

    The long table from :func:`build_state_corr_table` only stores the upper
    triangle (i < j). This expands it back to a full symmetric matrix with the
    diagonal forced to 1.0 (for ``value == 'r'``) or 0.0 (for ``value == 'z'``)
    and attaches per-synapse metadata coords for use by
    :func:`plot_slap2_corr_matrix`.
    """
    sub = df.filter(pl.col("state") == state)
    if sub.height == 0:
        raise ValueError(f"No rows for state {state!r} in the table.")

    has_pos = "pos_i" in sub.columns and "pos_j" in sub.columns

    # Collect synapse identifiers in a canonical order.
    i_select = [
        pl.col("syn_i").alias("syn"),
        pl.col("dmd_i").alias("dmd"),
        pl.col("dend_i").alias("dend"),
        pl.col("dend_type_i").alias("dend_type"),
        pl.col("soma_id").alias("soma_id"),
    ]
    j_select = [
        pl.col("syn_j").alias("syn"),
        pl.col("dmd_j").alias("dmd"),
        pl.col("dend_j").alias("dend"),
        pl.col("dend_type_j").alias("dend_type"),
        pl.col("soma_id").alias("soma_id"),
    ]
    if has_pos:
        i_select.append(pl.col("pos_i").alias("pos"))
        j_select.append(pl.col("pos_j").alias("pos"))

    syn_ids = (
        pl
        .concat([sub.select(i_select), sub.select(j_select)])
        .unique(subset=["syn"], keep="first")
        .sort("syn")
    )

    syns = syn_ids["syn"].to_list()
    n = len(syns)
    idx = {s: i for i, s in enumerate(syns)}

    diag = 1.0 if value == "r" else 0.0
    M = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(M, diag)

    for row in sub.iter_rows(named=True):
        i = idx[row["syn_i"]]
        j = idx[row["syn_j"]]
        v = row[value]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        M[i, j] = v
        M[j, i] = v

    coords = {
        "syn_1": ("syn_1", syns),
        "syn_2": ("syn_2", syns),
        "dmd_1": ("syn_1", syn_ids["dmd"].to_numpy()),
        "dmd_2": ("syn_2", syn_ids["dmd"].to_numpy()),
        "dend-ID_1": ("syn_1", syn_ids["dend"].to_list()),
        "dend-ID_2": ("syn_2", syn_ids["dend"].to_list()),
        "dend_type_1": ("syn_1", syn_ids["dend_type"].to_list()),
        "dend_type_2": ("syn_2", syn_ids["dend_type"].to_list()),
        "soma-ID_1": ("syn_1", syn_ids["soma_id"].to_list()),
        "soma-ID_2": ("syn_2", syn_ids["soma_id"].to_list()),
    }
    if has_pos:
        coords["pos_1"] = ("syn_1", syn_ids["pos"].to_numpy())
        coords["pos_2"] = ("syn_2", syn_ids["pos"].to_numpy())
    return xr.DataArray(
        M,
        dims=("syn_1", "syn_2"),
        coords=coords,
        name=f"{value}_{state}",
    )


def plot_state_pair_matrices(
    table: pl.DataFrame,
    cell_id: str,
    *,
    states: tuple[str, str] = ("NREM", "Wake"),
    value: str = "r",
    vmin: float = -0.2,
    vmax: float = 0.2,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (28, 14),
    group_by_soma: bool = True,
    dmd_labels=False,
    soma_labels=False,
    suptitle: str | None = None,
    **plot_kwargs,
):
    """Side-by-side correlation matrices for two states from a long-form table.

    Selects rows for the given ``cell_id``, reconstructs the full
    symmetric matrices for each state, and plots them with
    :func:`plot_slap2_corr_matrix`. The titles report the **off-diagonal**
    upper-triangle mean (no diagonal bleed-through).

    Parameters
    ----------
    table : pl.DataFrame
        Long-form correlation table (output of
        :func:`build_state_corr_table` / ``_multi``).
    cell_id : str
        Globally-unique cell identifier (``subject|exp|loc|acq|soma_id``).
        ``soma_id`` alone is not unique across recordings, so this function
        requires the full ``cell_id``. Construct with
        ``f"{subject}|{exp}|{loc}|{acq}|{soma_id}"`` or read from a row of
        the table.
    states : tuple of two str
        Which two states to compare, left then right. Default
        ``('NREM', 'Wake')``.
    value : str
        Which column in the table to plot. ``'r'`` (default) or ``'z'``.
    vmin, vmax, cmap, figsize : usual matplotlib args.
    group_by_soma, dmd_labels, soma_labels : forwarded to
        :func:`plot_slap2_corr_matrix`.
    suptitle : str | None
        Custom suptitle. Default uses ``cell_id``.

    Returns
    -------
    fig, axes
        The figure and a length-2 array of axes.
    """
    sub = table.filter(pl.col("cell_id") == cell_id)
    if sub.height == 0:
        raise ValueError(f"No rows in table for cell_id={cell_id!r}.")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, state in zip(axes, states):
        M = _table_to_matrix(sub, state=state, value=value)
        plot_slap2_corr_matrix(
            M,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            group_by_soma=group_by_soma,
            dmd_labels=dmd_labels,
            soma_labels=soma_labels,
            **plot_kwargs,
        )
        n = M.sizes["syn_1"]
        upper_idx = np.triu_indices(n, k=1)
        offdiag_mean = float(np.nanmean(M.values[upper_idx]))
        ax.set_title(f"{state}, off-diag mean {value} = {offdiag_mean:.4f}")

    fig.suptitle(suptitle if suptitle is not None else cell_id)
    fig.tight_layout()
    return fig, axes


def plot_pair_distribution(
    table: pl.DataFrame,
    *,
    value: str = "r",
    stratum_col: str = "pair_type",
    states: tuple[str, ...] = ("NREM", "Wake"),
    ax: plt.Axes | None = None,
    show_points: bool = False,
):
    """Violin/box plot of per-pair correlations split by state × stratum.

    Useful for the distribution-shape view: does NREM have a heavier right tail?
    Pulls every pair from the long table — does NOT collapse to per-(subject,
    soma) means. Use :func:`plot_stratum_paired` for the per-(subject, soma) view.

    Parameters
    ----------
    table : pl.DataFrame
        Long-form correlation table.
    value : str
        Column to plot. ``'r'`` (default) or ``'z'``.
    stratum_col : str
        Column to facet on. Default ``'pair_type'``; ``'same_type'`` is also
        useful (basal_basal / apical_apical / basal_apical).
    states : tuple of str
        Which states to include. Default ``('NREM', 'Wake')``.
    ax : matplotlib.Axes, optional
    show_points : bool
        If True, scatter individual points on top of the boxes.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    sub = table.filter(pl.col("state").is_in(list(states)))
    strata = sub[stratum_col].drop_nulls().unique().sort().to_list()
    state_list = list(states)

    n_states = len(state_list)
    n_strata = len(strata)
    width = 0.8 / n_states

    for s_i, state in enumerate(state_list):
        positions = np.arange(n_strata) + (s_i - (n_states - 1) / 2) * width
        data = []
        for stratum in strata:
            vals = (
                sub
                .filter((pl.col("state") == state) & (pl.col(stratum_col) == stratum))[
                    value
                ]
                .drop_nulls()
                .to_numpy()
            )
            data.append(vals if len(vals) > 0 else np.array([np.nan]))
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
        )
        color = f"C{s_i}"
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for med in bp["medians"]:
            med.set_color("black")
        if show_points:
            for pos, vals in zip(positions, data):
                if len(vals) > 0:
                    jitter = np.random.uniform(
                        -width * 0.2, width * 0.2, size=len(vals)
                    )
                    ax.scatter(
                        pos + jitter,
                        vals,
                        s=4,
                        alpha=0.2,
                        color=color,
                        edgecolors="none",
                    )
        ax.plot([], [], color=color, lw=6, alpha=0.5, label=state)

    ax.set_xticks(np.arange(n_strata))
    ax.set_xticklabels(strata, rotation=20, ha="right")
    ax.set_ylabel(value)
    ax.axhline(0, color="0.6", lw=0.7, ls="--")
    ax.legend(frameon=False)
    return fig, ax


def plot_stratum_paired(
    summary: pl.DataFrame,
    *,
    stratum: str,
    value: str = "mean_z",
    states: tuple[str, str] = ("Wake", "NREM"),
    ax: plt.Axes | None = None,
    line_alpha: float = 0.6,
    title: str | None = None,
):
    """Per-(subject, soma) paired-line plot for one stratum across two states.

    Each line connects the same ``(subject, soma)``'s value in the two states.
    Consumes the output of :func:`stratified_summaries`.

    Parameters
    ----------
    summary : pl.DataFrame
        Output of :func:`stratified_summaries`.
    stratum : str
        Which row of the ``stratum`` column to plot (e.g.
        ``'between_dend_same_type'``, ``'basal_basal'``).
    value : str
        Which summary statistic. ``'mean_z'`` (default), ``'median_z'``,
        ``'mean_r'``, ``'median_r'``.
    states : tuple of two str
        Order on the x-axis. Default ``('Wake', 'NREM')``.
    ax : matplotlib.Axes, optional
    line_alpha : float

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    sub = summary.filter(pl.col("stratum") == stratum)
    if sub.height == 0:
        raise ValueError(f"No rows in summary for stratum={stratum!r}.")

    # Pivot at the cell level — the same soma_id can refer to different
    # cells across recordings, so cell_id is the correct unit.
    wide = sub.pivot(on="state", index=["cell_id"], values=value)
    s0, s1 = states
    if s0 not in wide.columns or s1 not in wide.columns:
        raise ValueError(
            f"summary missing one of states {states} after pivot; have {wide.columns}"
        )

    xs = [0, 1]
    for row in wide.iter_rows(named=True):
        v0, v1 = row[s0], row[s1]
        if v0 is None or v1 is None:
            continue
        ax.plot(xs, [v0, v1], "-o", color="0.3", alpha=line_alpha, ms=4)

    ax.set_xticks(xs)
    ax.set_xticklabels(states)
    ax.set_ylabel(value)
    ax.axhline(0, color="0.6", lw=0.7, ls="--")
    ax.set_title(title if title is not None else f"{stratum}: {value}")
    return fig, ax


def plot_outlier_pairs(
    outliers: pl.DataFrame,
    *,
    ax: plt.Axes | None = None,
    max_n: int = 20,
    value: str = "delta_z",
):
    """Bar plot of top-Δz outlier pairs with subject/dendrite labels.

    Parameters
    ----------
    outliers : pl.DataFrame
        Output of :func:`outlier_pairs` (one row per pair, sorted).
    ax : matplotlib.Axes, optional
    max_n : int
        Show at most this many pairs.
    value : str
        Which column to use for the bar height. Default ``'delta_z'``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max_n * 0.3 + 1))
    else:
        fig = ax.figure

    df = outliers.head(max_n)
    n = df.height
    if n == 0:
        ax.text(
            0.5, 0.5, "no outliers", ha="center", va="center", transform=ax.transAxes
        )
        return fig, ax

    ys = np.arange(n)[::-1]
    vs = df[value].to_numpy()
    labels = [
        # Include exp/loc/acq so cells with the same soma_id across
        # recordings stay distinguishable.
        f"{r['subject']}/{r['exp']}/{r['loc']}/{r['acq']}/{r['soma_id']}: "
        f"{r['dend_i']}–{r['dend_j']}"
        for r in df.iter_rows(named=True)
    ]
    colors = ["tab:red" if v >= 0 else "tab:blue" for v in vs]
    ax.barh(ys, vs, color=colors, alpha=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="0.5", lw=0.7)
    ax.set_xlabel(value)
    return fig, ax


def plot_bout_synchrony_timeline(
    bout_summary: pl.DataFrame,
    *,
    value: str = "mean_r_offdiag",
    ax: plt.Axes | None = None,
    state_colors: Mapping[str, str] | None = None,
    hypno=None,
):
    """Per-bout synchrony summary scattered over wall-clock time.

    Parameters
    ----------
    bout_summary : pl.DataFrame
        Output of :func:`bout_level_synchrony`. One row per bout.
    value : str
        Which per-bout column to plot. Default ``'mean_r_offdiag'``;
        ``'frac_r_above_0p2'`` is another good option.
    ax : matplotlib.Axes, optional
    state_colors : mapping
        Override per-state colour. Default ``{'NREM': blue, 'Wake': orange}``.
    hypno : electro_py.hypno.Hypnogram, optional
        If provided, draw the hypnogram bouts as background bands underneath.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    if state_colors is None:
        state_colors = {"NREM": "tab:blue", "Wake": "tab:orange", "REM": "tab:green"}

    if hypno is not None:
        try:
            for row in hypno.df.iter_rows(named=True):
                clr = state_colors.get(row["state"], "0.85")
                ax.axvspan(
                    row["start_time"], row["end_time"], color=clr, alpha=0.12, lw=0
                )
        except Exception:
            pass

    for state, clr in state_colors.items():
        sub = bout_summary.filter(pl.col("state") == state)
        if sub.height == 0:
            continue
        mid_t = (sub["start_time"].to_numpy() + sub["end_time"].to_numpy()) / 2
        ax.scatter(
            mid_t, sub[value].to_numpy(), color=clr, s=20, alpha=0.85, label=state
        )

    ax.axhline(0, color="0.6", lw=0.7, ls="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(value)
    ax.legend(frameon=False)
    return fig, ax


# ---------------------------------------------------------------------------
# Temporal-dynamics plots (within-bout, state-clock, state-onset)
# ---------------------------------------------------------------------------


_DEFAULT_STATE_COLORS = {
    "NREM": flex.blue400,
    "Wake": flex.green400,
    "REM": flex.magenta400,
}


def plot_within_bout_drift(
    timeline: pl.DataFrame,
    *,
    state: str,
    value: str = "mean_r_offdiag",
    by: str = "norm",
    ax=None,
    state_colors: Mapping | None = None,
    show_per_bout: bool = True,
    show_group_mean: bool = True,
    n_norm_bins: int = 10,
    alpha_traces: float = 0.15,
    title: str | None = None,
):
    """Plot within-bout correlation drift for one state.

    For every bout × cell in ``timeline``, plot ``value`` vs within-bout time
    (either normalized to bout duration or in seconds since bout start).
    Faint per-bout traces overlay a heavier group mean.

    Parameters
    ----------
    timeline : pl.DataFrame
        Output of :func:`within_bout_correlation_timeline`.
    state : str
        Restrict to this state.
    value : str
        Per-window column to plot. Default ``'mean_r_offdiag'``.
    by : str
        ``'norm'`` (default — x-axis = window_center / bout_duration ∈ [0,1])
        or ``'seconds'`` (x = window_center_in_bout_s).
    ax : matplotlib Axes, optional
    state_colors : mapping, optional
    show_per_bout : bool
        Faint per-bout lines.
    show_group_mean : bool
        Heavy group mean (binned along x).
    n_norm_bins : int
        Number of bins for the group mean curve when ``by='norm'``.
    alpha_traces : float
    title : str | None

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.2))
    else:
        fig = ax.figure
    state_colors = state_colors or _DEFAULT_STATE_COLORS
    clr = state_colors.get(state, "0.4")

    sub = timeline.filter(pl.col("state") == state)
    if sub.height == 0:
        ax.text(
            0.5,
            0.5,
            f"no data for state={state!r}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig, ax
    x_col = "window_center_in_bout_norm" if by == "norm" else "window_center_in_bout_s"

    if show_per_bout:
        # One line per (cell_id, bout_idx)
        for (_cell, _bout), grp in sub.group_by(
            ["cell_id", "bout_idx"], maintain_order=True
        ):
            grp_sorted = grp.sort(x_col)
            ax.plot(
                grp_sorted[x_col].to_numpy(),
                grp_sorted[value].to_numpy(),
                color=clr,
                alpha=alpha_traces,
                lw=0.7,
            )

    if show_group_mean:
        if by == "norm":
            bin_edges = np.linspace(0.0, 1.0, n_norm_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            xs = sub[x_col].to_numpy()
            ys = sub[value].to_numpy()
            means = np.full(n_norm_bins, np.nan)
            sems = np.full(n_norm_bins, np.nan)
            for i in range(n_norm_bins):
                m = (xs >= bin_edges[i]) & (xs < bin_edges[i + 1])
                if i == n_norm_bins - 1:
                    m = (xs >= bin_edges[i]) & (xs <= bin_edges[i + 1])
                if m.sum() < 2:
                    continue
                vals = ys[m]
                vals = vals[np.isfinite(vals)]
                if vals.size < 2:
                    continue
                means[i] = float(vals.mean())
                sems[i] = float(vals.std(ddof=1) / np.sqrt(vals.size))
            ax.plot(bin_centers, means, "-", color=clr, lw=2.5)
            ax.fill_between(
                bin_centers,
                means - sems,
                means + sems,
                color=clr,
                alpha=0.25,
                lw=0,
            )
        else:
            # seconds: plot grouped by integer window-center seconds
            xs = sub[x_col].to_numpy()
            ys = sub[value].to_numpy()
            order = np.argsort(xs)
            xs_s = xs[order]
            ys_s = ys[order]
            ax.plot(xs_s, ys_s, ".", color=clr, alpha=0.5, ms=3)

    ax.axhline(0, color="0.6", lw=0.7, ls="--")
    ax.set_xlabel(
        "within-bout time (norm)" if by == "norm" else "time since bout start (s)"
    )
    ax.set_ylabel(value)
    ax.set_title(title if title is not None else f"{state} — within-bout {value}")
    return fig, ax


def plot_state_clock_trace(
    state_clock: pl.DataFrame,
    *,
    state: str,
    value_agg: str = "mean_r_offdiag_proxy",
    by_pair_type: tuple[str, ...] = (
        "between_dend_same_type",
        "between_dend_cross_type",
    ),
    ax=None,
    state_colors: Mapping | None = None,
    show_per_cell: bool = True,
    alpha_traces: float = 0.4,
    title: str | None = None,
):
    """Plot per-cell mean off-diagonal r against cumulative state-clock time.

    Aggregates the ``state_clock_table`` output (per (pair, bin) rows) to per
    (cell, bin) by averaging r over off-diagonal pairs (``pair_type !=
    'within_dend'``), then plots versus ``clock_bin_center_s``.

    Parameters
    ----------
    state_clock : pl.DataFrame
        Output of :func:`state_clock_table` or
        :func:`state_clock_table_multi`.
    state : str
        State to filter by (e.g., ``'NREM'``).
    value_agg : str
        Currently only ``'mean_r_offdiag_proxy'`` (mean of per-pair r
        across off-diagonal pairs) is implemented.
    by_pair_type : tuple of str
        Pair-type categories to include in the off-diagonal mean.
    ax : matplotlib Axes, optional
    state_colors : mapping
    show_per_cell : bool
    alpha_traces : float
    title : str | None

    Returns
    -------
    fig, ax
    """
    if value_agg != "mean_r_offdiag_proxy":
        raise ValueError(f"value_agg must be 'mean_r_offdiag_proxy', got {value_agg!r}")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.2))
    else:
        fig = ax.figure
    state_colors = state_colors or _DEFAULT_STATE_COLORS
    clr = state_colors.get(state, "0.4")

    sub = state_clock.filter(
        (pl.col("state") == state) & pl.col("pair_type").is_in(list(by_pair_type))
    )
    if sub.height == 0:
        ax.text(
            0.5,
            0.5,
            f"no data for state={state!r}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig, ax
    per_cell = (
        sub
        .group_by(["cell_id", "clock_bin_idx", "clock_bin_center_s"])
        .agg(pl.col("r").mean().alias("mean_r"))
        .sort(["cell_id", "clock_bin_idx"])
    )

    if show_per_cell:
        for cell_id, grp in per_cell.group_by("cell_id", maintain_order=True):
            grp_sorted = grp.sort("clock_bin_center_s")
            ax.plot(
                grp_sorted["clock_bin_center_s"].to_numpy(),
                grp_sorted["mean_r"].to_numpy(),
                "-o",
                color=clr,
                alpha=alpha_traces,
                ms=3,
                lw=1.0,
            )

    # Group mean across cells: mean r per bin across cells.
    grp_mean = (
        per_cell
        .group_by("clock_bin_center_s")
        .agg(
            pl.col("mean_r").mean().alias("mean_across_cells"),
            pl.col("mean_r").std().alias("std_across_cells"),
            pl.len().alias("n_cells"),
        )
        .sort("clock_bin_center_s")
    )
    if grp_mean.height > 0:
        xs = grp_mean["clock_bin_center_s"].to_numpy()
        ys = grp_mean["mean_across_cells"].to_numpy()
        ns = grp_mean["n_cells"].to_numpy()
        sds = grp_mean["std_across_cells"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            sems = sds / np.sqrt(np.maximum(ns, 1))
        ax.plot(xs, ys, "-", color=clr, lw=2.5)
        ax.fill_between(xs, ys - sems, ys + sems, color=clr, alpha=0.25, lw=0)

    ax.axhline(0, color="0.6", lw=0.7, ls="--")
    ax.set_xlabel("cumulative state-time (s)")
    ax.set_ylabel("mean off-diag r (per-cell)")
    ax.set_title(title if title is not None else f"{state} — state-clock drift")
    return fig, ax


def plot_state_onset_aligned(
    onset_df: pl.DataFrame,
    *,
    value: str = "mean_r_offdiag",
    ax=None,
    state_colors: Mapping | None = None,
    fill_colors: Mapping | None = None,
    bin_s: float = 5.0,
    show_per_cell: bool = False,
    normalize_pre: bool = False,
    style: str = "line",
    show_transition_line: bool = True,
    drop_zero_bin: bool = False,
    label: str | None = None,
    title: str | None = None,
):
    """Plot mean ± SEM ``value`` vs time-relative-to-transition.

    Aggregates the output of :func:`state_onset_aligned_synchrony_multi`
    across ``onset_idx`` per cell, then averages across cells per time bin.

    Parameters
    ----------
    onset_df : pl.DataFrame
        Long-form table from :func:`state_onset_aligned_synchrony` /
        :func:`state_onset_aligned_synchrony_multi`. Expects
        ``onset_state``, ``pre_state``, ``t_rel_s``, ``cell_id`` columns.
    value : str
        Per-window column to summarise. Default ``'mean_r_offdiag'``.
    ax : matplotlib Axes, optional
    state_colors : mapping
        Per-state colors used for marker outlines, lines, error bars and caps.
    fill_colors : mapping, optional
        Per-state colors used for the *fill* of point markers (style='points').
        If omitted, ``state_colors`` is used for both fill and outline (original
        behavior). Has no effect when ``style='line'``.
    bin_s : float
        Bin width along ``t_rel_s`` for the group mean.
    show_per_cell : bool
        If True, also draw faint per-cell lines.
    normalize_pre : bool
        If True, subtract each cell's mean ``value`` over ``t_rel_s < 0``
        from all of that cell's rows before binning. The y-axis then reads
        as deviation from the pre-transition baseline (0 = pre mean).
    style : {"line", "points"}
        ``"line"`` (default): connected line with shaded ± SEM band. Bins
        with no data become visible gaps. ``"points"``: discrete markers
        with vertical SEM errorbars; empty bins simply don't draw.
    show_transition_line : bool
        Draw the dotted vertical reference line at ``t_rel_s = 0``.
        Set False when overlaying multiple transitions or zooming to
        post-transition only.
    drop_zero_bin : bool
        If True, suppress the bin centered exactly at ``t_rel_s = 0`` from the
        plot — useful when that bin straddles the transition and mixes pre/post
        samples in a way that's hard to interpret.
    label : str | None
        Legend label for the line/marker series. None omits it.
    title : str | None
        Axes title. None (default) leaves the title untouched — useful for
        overlaying multiple transitions on one axes, where transition-specific
        text belongs in the legend label, not the title.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.2))
    else:
        fig = ax.figure
    state_colors = state_colors or _DEFAULT_STATE_COLORS

    if onset_df.height == 0:
        warnings.warn("plot_state_onset_aligned: empty onset_df, nothing drawn")
        return fig, ax

    onset_state = onset_df["onset_state"][0]
    pre_state = onset_df["pre_state"][0]
    clr = state_colors.get(onset_state, "0.4")
    fill_clr = fill_colors.get(onset_state, clr) if fill_colors is not None else clr

    if normalize_pre:
        pre_means = (
            onset_df
            .filter(pl.col("t_rel_s") < 0)
            .group_by("cell_id")
            .agg(pl.col(value).mean().alias("_pre_mean"))
        )
        onset_df = onset_df.join(pre_means, on="cell_id", how="left").with_columns(
            (pl.col(value) - pl.col("_pre_mean")).alias(value)
        )

    if show_per_cell:
        per_cell_avg = (
            onset_df
            .group_by(["cell_id", "t_rel_s"])
            .agg(pl.col(value).mean().alias("v"))
            .sort(["cell_id", "t_rel_s"])
        )
        for _cell, grp in per_cell_avg.group_by("cell_id", maintain_order=True):
            grp_sorted = grp.sort("t_rel_s")
            ax.plot(
                grp_sorted["t_rel_s"].to_numpy(),
                grp_sorted["v"].to_numpy(),
                "-",
                color=clr,
                alpha=0.2,
                lw=0.7,
            )

    t_rel = onset_df["t_rel_s"].to_numpy()
    vals = onset_df[value].to_numpy()
    t_min, t_max = float(t_rel.min()), float(t_rel.max())
    edges = np.arange(t_min, t_max + bin_s, bin_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(centers.size, np.nan)
    sems = np.full(centers.size, np.nan)
    for i in range(centers.size):
        m = (t_rel >= edges[i]) & (t_rel < edges[i + 1])
        if i == centers.size - 1:
            m = (t_rel >= edges[i]) & (t_rel <= edges[i + 1])
        v = vals[m]
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        means[i] = float(v.mean())
        sems[i] = float(v.std(ddof=1) / np.sqrt(v.size))

    if drop_zero_bin:
        zero_mask = np.abs(centers) < 1e-9
        means[zero_mask] = np.nan
        sems[zero_mask] = np.nan

    if style == "points":
        good = np.isfinite(means)
        ax.errorbar(
            centers[good],
            means[good],
            yerr=sems[good],
            fmt="o",
            color=clr,
            ecolor=clr,
            elinewidth=3.5,
            capsize=8,
            capthick=3.5,
            markersize=13,
            markerfacecolor=fill_clr,
            markeredgecolor=clr,
            markeredgewidth=2.2,
            label=label,
        )
    elif style == "line":
        ax.plot(centers, means, "-", color=clr, lw=2.5, label=label)
        ax.fill_between(
            centers, means - sems, means + sems, color=clr, alpha=0.25, lw=0
        )
    else:
        raise ValueError(f"style must be 'line' or 'points', got {style!r}")

    if show_transition_line:
        ax.axvline(0, color="k", lw=1.0, ls=":")
    ax.axhline(0, color="red", lw=0.7, ls="--")
    # ax.set_xlabel("time relative to transition (s)")
    pretty = "off-diag r" if value == "mean_r_offdiag" else value
    ylab = (
        "Δr (vs pre)"
        if (normalize_pre and value == "mean_r_offdiag")
        else (f"Δ {pretty} (vs pre)" if normalize_pre else pretty)
    )
    ax.set_ylabel(ylab)
    if title is not None:
        ax.set_title(title)
    return fig, ax
