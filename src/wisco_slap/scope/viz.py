"""Plotting helpers for SLAP2 synaptic data.

Currently houses :func:`synapto_dendro_heatmap`, the time-by-synapse heatmap
with tree-style brackets along the dendrite/DMD/soma hierarchy. Brackets and
label resolution share helpers with :mod:`wisco_slap.scope.corr` (where the
matching correlation-matrix plot lives).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from matplotlib.transforms import blended_transform_factory

from .corr import (
    LabelOpt,
    _resolve_label,
    _safe_str,
    _to_combined,
)


def synapto_dendro_heatmap(
    data,
    t_start: float | None = None,
    t_end: float | None = None,
    DMD_first: int = 1,
    cmap: str = "magma",
    vmin: float = 0,
    vmax: float = 6,
    n_time_ticks: int = 5,
    boundary_color: str = "white",
    boundary_lw: float = 1.0,
    boundary_alpha: float = 1.0,
    dend_labels: LabelOpt = "auto",
    dmd_labels: LabelOpt = "auto",
    soma_labels: LabelOpt = "auto",
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
    label whose levels (innermost → outermost) are dendrite, DMD, soma. Brackets for all three
    levels are always drawn; each `*_labels` argument controls only the text on its level:
      - 'auto' or True  → label with default text (the coord values; DMD shows as "DMD 1")
      - False / None / 'off' → render the label as a blank space (bracket stays in place; layout preserved)
      - dict mapping ID → custom string → label with custom text (missing keys fall back to default)

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
    im = ax.imshow(
        combined.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )

    for _, top_row, _ in dend_blocks[1:]:
        ax.axhline(
            top_row - 0.5, color=boundary_color, lw=boundary_lw, alpha=boundary_alpha
        )

    ax.set_yticks([])
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
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
        cbar.set_label(cbar_label)
        cbar.outline.set_visible(False)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(n - 0.5, -0.5)
    return fig, ax, im
