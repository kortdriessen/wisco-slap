"""Plotting helpers for SLAP2 synaptic data.

Houses :func:`synapto_dendro_heatmap` (time-by-synapse heatmap) and
:func:`synapto_dendro_traces` (the same hierarchical layout but with stacked
1D traces, colored by a per-dendrite colormap sweep). Both share the
tree-style DMD/soma/dendrite brackets, and the bracket/label helpers come
from :mod:`wisco_slap.scope.corr`.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize, to_rgba
from matplotlib.transforms import blended_transform_factory

from .corr import (
    LabelOpt,
    _resolve_label,
    _safe_str,
    _to_combined,
)


def _resolve_cmap(cmap) -> Colormap:
    """Accept a colormap name (str) or any matplotlib-compatible Colormap instance.

    The `colormaps` package's objects subclass `matplotlib.colors.Colormap`, so
    instances pass through unchanged; strings go through `plt.get_cmap`.
    """
    if isinstance(cmap, Colormap):
        return cmap
    return plt.get_cmap(cmap)


def synapto_dendro_heatmap(
    data,
    t_start: float | None = None,
    t_end: float | None = None,
    DMD_first: int = 1,
    cmap: str = "magma",
    dend_type_cmaps: dict | None = None,
    nan_color: str | None = None,
    vmin: float = 0,
    vmax: float = 6,
    n_time_ticks: int = 5,
    boundary_color: str = "white",
    boundary_lw: float = 1.0,
    boundary_alpha: float = 1.0,
    dend_labels: LabelOpt = "auto",
    dmd_labels: LabelOpt = "auto",
    soma_labels: LabelOpt = "auto",
    dend_brackets: bool = True,
    dmd_brackets: bool = True,
    soma_brackets: bool = True,
    cbar_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    bracket_color: str = "0.15",
    bracket_lw: float = 1.4,
    label_fontsize: float = 8.5,
    zero_t: bool = True,
):
    """Heatmap of SLAP2 synaptic traces, grouped by DMD → soma → dendrite, ordered by `pos` within each dendrite.

    Each row is one synapse; the x-axis is time in seconds. Rows are ordered top-to-bottom as:
    DMD `DMD_first` (all somas, all dendrites within each soma, sorted by `pos`), then the other DMD.
    Horizontal lines mark dendrite boundaries. The y-axis is annotated with a minimal tree-style
    label whose levels (innermost → outermost) are dendrite, DMD, soma. By default brackets for all
    three levels are drawn; each `*_labels` argument controls only the text on its level:
      - 'auto' or True  → label with default text (the coord values; DMD shows as "DMD 1")
      - False / None / 'off' → render the label as a blank space (bracket stays in place; layout preserved)
      - dict mapping ID → custom string → label with custom text (missing keys fall back to default)
    To drop a level's bracket entirely (not just blank its label), set the corresponding
    `dend_brackets` / `dmd_brackets` / `soma_brackets` to False; the remaining levels then pack
    tightly toward the axis with no empty gap.

    Parameters
    ----------
    data : dict[str, xr.DataArray] | xr.DataArray
        Either {'dmd_1': DataArray, 'dmd_2': DataArray} as returned by `wis.get.syn_dF`, or a single
        DataArray that already concatenates the DMDs along `syn_id` with a `dmd` coord attached.
    t_start, t_end : float | None
        Optional time crop in seconds.
    DMD_first : int
        Which DMD (1 or 2) is plotted on top.
    zero_t : bool
        If True (default), the displayed time axis is shifted so the leftmost sample is 0 s,
        regardless of the underlying time-coord values.
    cmap : str | matplotlib.colors.Colormap
        Colormap used for every row, unless `dend_type_cmaps` is given — in which case it is
        the fallback for any dendrite whose `dend_type` is not in that mapping.
    dend_type_cmaps : dict | None
        Optional mapping from `dend_type` value → colormap (name or Colormap instance), used to
        color each synapse row by the type of its parent dendrite (read from the `dend_type`
        coord, matched case-insensitively). Rows whose `dend_type` is missing/unlabelled, or not
        present in the mapping, fall back to `cmap`. All colormaps share the same `vmin`/`vmax`.
        When set, the image is composited as RGBA, so any colorbar uses `cmap` for its scale.
    nan_color : str | None
        Color for NaN values. If None (default), uses the colormap's default bad-color.
    """
    combined = _to_combined(data)

    if "channel" in combined.dims:
        if combined.sizes["channel"] == 1:
            combined = combined.squeeze("channel", drop=True)
        else:
            raise ValueError(
                f"`data` has {combined.sizes['channel']} channels — select one before plotting."
            )

    if t_start is not None or t_end is not None:
        combined = combined.sel(time=slice(t_start, t_end))

    n = combined.sizes["syn_id"]
    if n == 0:
        raise ValueError("No synapses in `data` after time crop.")

    dmds = np.asarray(combined["dmd"].values, dtype=int)
    somas = (
        np.asarray([_safe_str(v) for v in combined["soma-ID"].values])
        if "soma-ID" in combined.coords
        else np.array(["?"] * n)
    )
    dends = (
        np.asarray([_safe_str(v) for v in combined["dend-ID"].values])
        if "dend-ID" in combined.coords
        else np.array(["?"] * n)
    )
    if "pos" in combined.coords:
        poses = np.asarray(combined["pos"].values, dtype=float)
        poses = np.where(np.isnan(poses), np.inf, poses)
    else:
        poses = np.zeros(n)

    unique_dmds = sorted(set(dmds.tolist()))
    if DMD_first in unique_dmds:
        dmd_order = [DMD_first] + [d for d in unique_dmds if d != DMD_first]
    else:
        dmd_order = unique_dmds
    dmd_rank = {d: i for i, d in enumerate(dmd_order)}

    sort_keys = list(
        zip(
            [dmd_rank[d] for d in dmds],
            somas.tolist(),
            dends.tolist(),
            poses.tolist(),
            range(n),
        )
    )
    sort_keys.sort()
    order_idx = [k[4] for k in sort_keys]
    combined = combined.isel(syn_id=order_idx)

    dmds = dmds[order_idx]
    somas = somas[order_idx]
    dends = dends[order_idx]

    dend_blocks: list[tuple[str, int, int]] = []
    soma_blocks: list[tuple[str, int, int]] = []
    dmd_blocks: list[tuple[int, int, int]] = []
    cur_dend_key = cur_soma_key = cur_dmd_key = object()
    dend_start = soma_start = dmd_start = 0
    for i in range(n):
        dk = (int(dmds[i]), somas[i], dends[i])
        sk = (int(dmds[i]), somas[i])
        mk = int(dmds[i])
        if dk != cur_dend_key:
            if i > 0:
                dend_blocks.append((cur_dend_key[2], dend_start, i - 1))
            cur_dend_key = dk
            dend_start = i
        if sk != cur_soma_key:
            if i > 0:
                soma_blocks.append((cur_soma_key[1], soma_start, i - 1))
            cur_soma_key = sk
            soma_start = i
        if mk != cur_dmd_key:
            if i > 0:
                dmd_blocks.append((cur_dmd_key, dmd_start, i - 1))
            cur_dmd_key = mk
            dmd_start = i
    dend_blocks.append((cur_dend_key[2], dend_start, n - 1))
    soma_blocks.append((cur_soma_key[1], soma_start, n - 1))
    dmd_blocks.append((cur_dmd_key, dmd_start, n - 1))

    if ax is None:
        if figsize is None:
            figsize = (11, max(4.0, min(18.0, n * 0.06)))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    time_vals = np.asarray(combined["time"].values, dtype=float)
    t_offset = time_vals[0] if zero_t else 0.0
    extent = [
        float(time_vals[0] - t_offset),
        float(time_vals[-1] - t_offset),
        n - 0.5,
        -0.5,
    ]
    cmap_obj = _resolve_cmap(cmap)
    if nan_color is not None:
        cmap_obj = cmap_obj.copy()
        cmap_obj.set_bad(nan_color)

    if dend_type_cmaps:
        # Per-dendrite-type coloring: composite an RGBA image so each row uses the
        # colormap for its parent dendrite's `dend_type` (fallback `cmap_obj`).
        type_to_cmap = {
            str(k).lower(): _resolve_cmap(v) for k, v in dend_type_cmaps.items()
        }
        row_types = (
            np.asarray([_safe_str(v).lower() for v in combined["dend_type"].values])
            if "dend_type" in combined.coords
            else np.array(["?"] * n)
        )
        values = np.asarray(combined.values, dtype=float)
        span = float(vmax - vmin)
        if span <= 0:
            raise ValueError("`vmax` must be greater than `vmin`.")
        norm = np.clip((values - vmin) / span, 0.0, 1.0)
        finite = np.isfinite(values)
        bad_rgba = to_rgba(nan_color) if nan_color is not None else cmap_obj(np.nan)
        rgba = np.asarray(cmap_obj(norm), dtype=float)  # fallback for all rows
        for t, cm_t in type_to_cmap.items():
            mask = row_types == t
            if mask.any():
                rgba[mask] = np.asarray(cm_t(norm), dtype=float)[mask]
        rgba[~finite] = bad_rgba
        im = ax.imshow(
            rgba,
            aspect="auto",
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )
        cbar_mappable = ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap_obj)
    else:
        im = ax.imshow(
            combined.values,
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )
        cbar_mappable = im

    for _, top_row, _ in dend_blocks[1:]:
        ax.axhline(
            top_row - 0.5, color=boundary_color, lw=boundary_lw, alpha=boundary_alpha
        )

    ax.set_yticks([])
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=n_time_ticks, prune=None))
    ax.set_xlabel("Time (s)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    # Only the enabled levels are drawn at all (brackets + labels). Disabling a
    # level removes its brackets entirely and lets the remaining levels pack
    # tightly toward the axis, so there is no empty gap where it would have been.
    levels: list[tuple[str, list[tuple], LabelOpt]] = [
        spec
        for spec, enabled in (
            (("dend", dend_blocks, dend_labels), dend_brackets),
            (("dmd", dmd_blocks, dmd_labels), dmd_brackets),
            (("soma", soma_blocks, soma_labels), soma_brackets),
        )
        if enabled
    ]

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    bracket_step = 0.045
    x0 = -0.012
    cap = 0.005
    label_pad = 0.006
    for lvl_idx, (name, blocks, opt) in enumerate(levels):
        x_b = x0 - lvl_idx * bracket_step
        x_lab = x_b - label_pad
        for key, top_row, bot_row in blocks:
            y_top = top_row - 0.5
            y_bot = bot_row + 0.5
            ax.plot(
                [x_b, x_b],
                [y_top, y_bot],
                color=bracket_color,
                lw=bracket_lw,
                transform=trans,
                clip_on=False,
                solid_capstyle="butt",
            )
            ax.plot(
                [x_b, x_b + cap],
                [y_top, y_top],
                color=bracket_color,
                lw=bracket_lw,
                transform=trans,
                clip_on=False,
                solid_capstyle="butt",
            )
            ax.plot(
                [x_b, x_b + cap],
                [y_bot, y_bot],
                color=bracket_color,
                lw=bracket_lw,
                transform=trans,
                clip_on=False,
                solid_capstyle="butt",
            )
            ax.text(
                x_lab,
                (y_top + y_bot) / 2,
                _resolve_label(opt, key, name),
                transform=trans,
                ha="right",
                va="center",
                fontsize=label_fontsize,
                color=bracket_color,
                clip_on=False,
            )

    if cbar_label is not None:
        cbar = fig.colorbar(cbar_mappable, ax=ax, fraction=0.02, pad=0.015)
        cbar.set_label(cbar_label)
        cbar.outline.set_visible(False)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(n - 0.5, -0.5)
    return fig, ax, im


def synapto_dendro_traces(
    data,
    t_start: float | None = None,
    t_end: float | None = None,
    DMD_first: int = 1,
    cmap: str | Colormap = "magma",
    cmap_range: tuple[float, float] = (0.15, 0.95),
    vmin: float = 0,
    vmax: float = 6,
    trace_height_frac: float = 0.9,
    dend_spacing: float = 0.0,
    linewidth: float = 0.6,
    alpha: float = 1.0,
    n_time_ticks: int = 5,
    boundary_color: str | None = "0.7",
    boundary_lw: float = 0.5,
    boundary_alpha: float = 1.0,
    dend_labels: LabelOpt = "auto",
    dmd_labels: LabelOpt = "auto",
    soma_labels: LabelOpt = "auto",
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    bracket_color: str = "0.15",
    bracket_lw: float = 1.4,
    label_fontsize: float = 8.5,
    zero_t: bool = True,
    background: str | None = None,
    minimal: bool = False,
):
    """Stacked synaptic-trace plot grouped by DMD → soma → dendrite, sorted by `pos` within each dendrite.

    Visually parallel to :func:`synapto_dendro_heatmap`: identical hierarchical y-axis with
    tree-style DMD/soma/dendrite brackets, identical sort order. Each row is one synapse
    drawn as a 1D trace within a height-1 band. The colormap is sampled per-dendrite — the
    `N` synapses in a dendrite get `N` evenly-spaced colors across `cmap_range`, in pos
    order — so within each dendrite the proximal→distal axis reads as a smooth color
    gradient.

    `vmin`/`vmax` set the data range that maps to one row's vertical band: values ≤ vmin
    sit at the bottom of the band, values ≥ vmax at the top (clipped). NaN samples create
    gaps in the trace.

    Parameters
    ----------
    data : dict[str, xr.DataArray] | xr.DataArray
        Same input as `synapto_dendro_heatmap`.
    cmap : str | matplotlib.colors.Colormap
        Colormap (name or instance); sampled per-dendrite.
    cmap_range : tuple[float, float]
        Min/max positions within the colormap to draw per-dendrite colors from.
        Defaults avoid the extreme ends where contrast against background is poor.
    vmin, vmax : float
        Data range that fills one row's vertical band (clipped outside this range).
    trace_height_frac : float
        Fraction of a row's height used by the trace (0 < x ≤ 1). Smaller → more whitespace
        between rows.
    dend_spacing : float
        Extra vertical gap (in row units) inserted between adjacent dendrite blocks.
        Default 0 → tightly packed (original behavior).
    linewidth, alpha : float
        Line styling.
    zero_t : bool
        If True (default), the displayed time axis is shifted so the leftmost sample is 0 s.
    background : str | None
        Background color for figure and axes (e.g. "black"). None leaves matplotlib defaults.
    minimal : bool
        If True, strip the x-axis (no ticks, label, or spine), all dendrite/soma/DMD
        brackets and labels, and all spines. Leaves just the traces (and any inter-dendrite
        boundary lines).
    """
    combined = _to_combined(data)

    if "channel" in combined.dims:
        if combined.sizes["channel"] == 1:
            combined = combined.squeeze("channel", drop=True)
        else:
            raise ValueError(
                f"`data` has {combined.sizes['channel']} channels — select one before plotting."
            )

    if t_start is not None or t_end is not None:
        combined = combined.sel(time=slice(t_start, t_end))

    n = combined.sizes["syn_id"]
    if n == 0:
        raise ValueError("No synapses in `data` after time crop.")

    dmds = np.asarray(combined["dmd"].values, dtype=int)
    somas = (
        np.asarray([_safe_str(v) for v in combined["soma-ID"].values])
        if "soma-ID" in combined.coords
        else np.array(["?"] * n)
    )
    dends = (
        np.asarray([_safe_str(v) for v in combined["dend-ID"].values])
        if "dend-ID" in combined.coords
        else np.array(["?"] * n)
    )
    if "pos" in combined.coords:
        poses = np.asarray(combined["pos"].values, dtype=float)
        poses = np.where(np.isnan(poses), np.inf, poses)
    else:
        poses = np.zeros(n)

    unique_dmds = sorted(set(dmds.tolist()))
    if DMD_first in unique_dmds:
        dmd_order = [DMD_first] + [d for d in unique_dmds if d != DMD_first]
    else:
        dmd_order = unique_dmds
    dmd_rank = {d: i for i, d in enumerate(dmd_order)}

    sort_keys = list(
        zip(
            [dmd_rank[d] for d in dmds],
            somas.tolist(),
            dends.tolist(),
            poses.tolist(),
            range(n),
        )
    )
    sort_keys.sort()
    order_idx = [k[4] for k in sort_keys]
    combined = combined.isel(syn_id=order_idx)

    dmds = dmds[order_idx]
    somas = somas[order_idx]
    dends = dends[order_idx]

    dend_blocks: list[tuple[str, int, int]] = []
    soma_blocks: list[tuple[str, int, int]] = []
    dmd_blocks: list[tuple[int, int, int]] = []
    cur_dend_key = cur_soma_key = cur_dmd_key = object()
    dend_start = soma_start = dmd_start = 0
    for i in range(n):
        dk = (int(dmds[i]), somas[i], dends[i])
        sk = (int(dmds[i]), somas[i])
        mk = int(dmds[i])
        if dk != cur_dend_key:
            if i > 0:
                dend_blocks.append((cur_dend_key[2], dend_start, i - 1))
            cur_dend_key = dk
            dend_start = i
        if sk != cur_soma_key:
            if i > 0:
                soma_blocks.append((cur_soma_key[1], soma_start, i - 1))
            cur_soma_key = sk
            soma_start = i
        if mk != cur_dmd_key:
            if i > 0:
                dmd_blocks.append((cur_dmd_key, dmd_start, i - 1))
            cur_dmd_key = mk
            dmd_start = i
    dend_blocks.append((cur_dend_key[2], dend_start, n - 1))
    soma_blocks.append((cur_soma_key[1], soma_start, n - 1))
    dmd_blocks.append((cur_dmd_key, dmd_start, n - 1))

    row_y = np.zeros(n)
    cursor = 0.0
    for bi, (_, top, bot) in enumerate(dend_blocks):
        if bi > 0:
            cursor += dend_spacing
        for i in range(top, bot + 1):
            row_y[i] = cursor + (i - top)
        cursor += bot - top + 1
    total_h = float(cursor)

    if ax is None:
        if figsize is None:
            figsize = (11, max(4.0, min(18.0, total_h * 0.06)))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if background is not None:
        fig.set_facecolor(background)
        ax.set_facecolor(background)

    time_vals = np.asarray(combined["time"].values, dtype=float)
    t_offset = time_vals[0] if zero_t else 0.0
    t_disp = time_vals - t_offset

    cmap_obj = plt.get_cmap(cmap)
    cmap_lo, cmap_hi = cmap_range
    row_colors = np.zeros((n, 4))
    for _, top, bot in dend_blocks:
        count = bot - top + 1
        if count == 1:
            ts = np.array([0.5 * (cmap_lo + cmap_hi)])
        else:
            ts = np.linspace(cmap_lo, cmap_hi, count)
        for j, t in enumerate(ts):
            row_colors[top + j] = cmap_obj(t)

    values = np.asarray(combined.values, dtype=float)
    span = float(vmax - vmin)
    if span <= 0:
        raise ValueError("`vmax` must be greater than `vmin`.")
    norm = np.clip((values - vmin) / span, 0.0, 1.0)
    y_rows = row_y[:, None] + 0.5 - norm * trace_height_frac
    y_rows = np.where(np.isfinite(values), y_rows, np.nan)

    segments = [np.column_stack([t_disp, y_rows[i]]) for i in range(n)]
    lc = LineCollection(
        segments,
        colors=row_colors,
        linewidths=linewidth,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(lc)

    if boundary_color is not None:
        for _, top_row, _ in dend_blocks[1:]:
            ax.axhline(
                row_y[top_row] - 0.5 - dend_spacing / 2,
                color=boundary_color,
                lw=boundary_lw,
                alpha=boundary_alpha,
            )

    ax.set_yticks([])
    if minimal:
        ax.set_xticks([])
        ax.set_xlabel("")
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=n_time_ticks, prune=None))
        ax.set_xlabel("Time (s)")
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

        levels: list[tuple[str, list[tuple], LabelOpt]] = [
            ("dend", dend_blocks, dend_labels),
            ("dmd", dmd_blocks, dmd_labels),
            ("soma", soma_blocks, soma_labels),
        ]

        trans = blended_transform_factory(ax.transAxes, ax.transData)
        bracket_step = 0.045
        x0 = -0.012
        cap = 0.005
        label_pad = 0.006
        for lvl_idx, (name, blocks, opt) in enumerate(levels):
            x_b = x0 - lvl_idx * bracket_step
            x_lab = x_b - label_pad
            for key, top_row, bot_row in blocks:
                y_top = row_y[top_row] - 0.5
                y_bot = row_y[bot_row] + 0.5
                ax.plot(
                    [x_b, x_b],
                    [y_top, y_bot],
                    color=bracket_color,
                    lw=bracket_lw,
                    transform=trans,
                    clip_on=False,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [x_b, x_b + cap],
                    [y_top, y_top],
                    color=bracket_color,
                    lw=bracket_lw,
                    transform=trans,
                    clip_on=False,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [x_b, x_b + cap],
                    [y_bot, y_bot],
                    color=bracket_color,
                    lw=bracket_lw,
                    transform=trans,
                    clip_on=False,
                    solid_capstyle="butt",
                )
                ax.text(
                    x_lab,
                    (y_top + y_bot) / 2,
                    _resolve_label(opt, key, name),
                    transform=trans,
                    ha="right",
                    va="center",
                    fontsize=label_fontsize,
                    color=bracket_color,
                    clip_on=False,
                )

    ax.set_xlim(float(t_disp[0]), float(t_disp[-1]))
    ax.set_ylim(total_h - 0.5, -0.5)
    return fig, ax, lc
