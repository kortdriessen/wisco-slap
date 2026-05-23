from __future__ import annotations

import textwrap
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import polars as pl

SOMA_TREE_COLOR = "#8B0000"
DMD1_DENDRITE_COLOR = "#C05A00"
DMD2_DENDRITE_COLOR = "#1F5A9E"
OTHER_DMD_DENDRITE_COLOR = "#333333"


def plot_event_dendrite_tree(
    events: Any,
    *,
    subject: str | None = None,
    exp: str | None = None,
    loc: str | None = None,
    acq: str | None = None,
    dmd_info: Mapping[str, Any] | None = None,
    ax: Any | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = "Event dendrite tree",
    soma_col: str = "soma-ID",
    dend_col: str = "dend-ID",
    syn_col: str = "syn_id",
    dmd_col: str = "dmd",
    soma_depth_col: str = "soma-depth",
    soma_color: str = SOMA_TREE_COLOR,
    dmd1_color: str = DMD1_DENDRITE_COLOR,
    dmd2_color: str = DMD2_DENDRITE_COLOR,
    other_dmd_color: str = OTHER_DMD_DENDRITE_COLOR,
    line_color: str = "#A7A7A7",
    multiline_soma_labels: bool = True,
    max_soma_id_line_chars: int = 22,
) -> tuple[Any, Any]:
    """Plot soma -> dendrite hierarchy for a SLAP event dataframe.

    Parameters
    ----------
    events
        Polars or pandas-like dataframe containing one row per detected event.
        The default column names match ``wis.get.glu_events_basic(...)`` output.
    subject, exp, loc, acq
        Optional acquisition identity. If all four are supplied, soma DMD/depth
        labels are resolved from ``analysis_materials/dmd_info.yaml`` via
        ``wis.meta.get.dmd_info()``. This prevents a soma with synapses/events
        on both DMDs from being labelled as if the soma itself belongs to both.
    dmd_info
        Optional already-loaded ``dmd_info.yaml`` contents. Useful for tests or
        when plotting many acquisitions.
    ax
        Optional Matplotlib axes. If omitted, a new figure and axes are created.
    figsize
        Optional figure size used only when ``ax`` is not supplied.
    title
        Optional axes title. Pass ``None`` to omit it.
    soma_col, dend_col, syn_col, dmd_col, soma_depth_col
        Column names used to build and annotate the hierarchy.
    soma_color
        Text color for soma labels.
    dmd1_color, dmd2_color, other_dmd_color
        Text colors for dendrite labels by DMD.
    line_color
        Color for soma-to-dendrite connector lines.
    multiline_soma_labels
        Whether to split soma labels across multiple lines. Enabled by default
        because single-line acquisition labels collide easily in tree plots.
    max_soma_id_line_chars
        Character width used when wrapping long soma IDs.

    Returns
    -------
    tuple
        ``(fig, ax)`` for further notebook tweaking or saving.
    """
    import matplotlib.pyplot as plt

    summary = _summarize_event_dendrite_tree(
        events,
        soma_col=soma_col,
        dend_col=dend_col,
        syn_col=syn_col,
        dmd_col=dmd_col,
        soma_depth_col=soma_depth_col,
    )
    tree = _tree_from_summary(summary, soma_col, dend_col, dmd_col)
    if not tree:
        raise ValueError("No soma/dendrite/synapse rows were available to plot.")

    soma_metadata = _soma_metadata_from_dmd_info(
        subject=subject,
        exp=exp,
        loc=loc,
        acq=acq,
        dmd_info=dmd_info,
    )
    _apply_soma_metadata(tree, soma_metadata)

    soma_labels = {
        soma_id: _format_soma_label(
            soma_id,
            soma["dmds"],
            soma["depths"],
            multiline=multiline_soma_labels,
            max_soma_id_line_chars=max_soma_id_line_chars,
        )
        for soma_id, soma in tree.items()
    }
    layout = _compute_tree_layout(tree, soma_labels)
    layout_width = layout[-1]["right"] - layout[0]["left"]
    if figsize is None:
        width = max(7.0, layout_width * 0.95)
        height = 5.0 if multiline_soma_labels else 4.6
        figsize = (width, height)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    soma_y = 1.0
    dend_y = 0.0
    all_x: list[float] = []
    soma_fontsize = _font_size_for_count(len(tree), base=10.5, floor=8.5)
    dend_fontsize = _font_size_for_count(
        sum(len(s["dendrites"]) for s in tree.values()), base=9.5, floor=8.0
    )

    for soma_layout in layout:
        soma_id = soma_layout["soma_id"]
        soma = tree[soma_id]
        dendrites = soma["dendrites"]
        dend_xs = soma_layout["dend_xs"]
        soma_x = soma_layout["soma_x"]
        all_x.extend(dend_xs)

        ax.scatter(
            [soma_x],
            [soma_y],
            s=52,
            facecolor="white",
            edgecolor=soma_color,
            linewidth=1.7,
            zorder=3,
        )
        ax.text(
            soma_x,
            soma_y + 0.12,
            soma_labels[soma_id],
            color=soma_color,
            ha="center",
            va="bottom",
            fontsize=soma_fontsize,
            fontweight="bold",
            linespacing=1.12,
        )

        for dend_x, dendrite in zip(dend_xs, dendrites, strict=True):
            dmd = dendrite["dmd"]
            dend_color = _dmd_color(
                dmd,
                dmd1_color=dmd1_color,
                dmd2_color=dmd2_color,
                other_color=other_dmd_color,
            )
            ax.plot(
                [soma_x, dend_x],
                [soma_y - 0.045, dend_y + 0.045],
                color=line_color,
                linewidth=1.15,
                zorder=1,
            )
            ax.scatter(
                [dend_x],
                [dend_y],
                s=38,
                facecolor="white",
                edgecolor=dend_color,
                linewidth=1.4,
                zorder=3,
            )
            ax.text(
                dend_x,
                dend_y - 0.1,
                _format_dendrite_label(dendrite["dend_id"], dmd, dendrite["n_syn"]),
                color=dend_color,
                ha="center",
                va="top",
                fontsize=dend_fontsize,
                linespacing=1.2,
            )

    x_pad = 0.8
    ax.set_xlim(
        min(layout[0]["left"], min(all_x)) - x_pad,
        max(layout[-1]["right"], max(all_x)) + x_pad,
    )
    ax.set_ylim(-0.52, 1.46 if multiline_soma_labels else 1.34)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=12)
    fig.tight_layout()
    return fig, ax


def _coerce_polars_frame(events: Any) -> pl.DataFrame:
    if isinstance(events, pl.DataFrame):
        return events
    if isinstance(events, pl.LazyFrame):
        return events.collect()
    if hasattr(events, "to_polars"):
        return events.to_polars()
    if hasattr(events, "to_pandas"):
        return pl.from_pandas(events.to_pandas())
    return pl.DataFrame(events)


def _summarize_event_dendrite_tree(
    events: Any,
    *,
    soma_col: str,
    dend_col: str,
    syn_col: str,
    dmd_col: str,
    soma_depth_col: str,
) -> pl.DataFrame:
    df = _coerce_polars_frame(events)
    required_cols = [soma_col, dend_col, syn_col, dmd_col, soma_depth_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Events dataframe is missing required columns: {missing}")

    return (
        df
        .select(required_cols)
        .drop_nulls(subset=[soma_col, dend_col, syn_col, dmd_col])
        .group_by(soma_col, dmd_col, dend_col)
        .agg(
            pl.col(syn_col).drop_nulls().n_unique().alias("n_synapses"),
            pl.col(soma_depth_col).drop_nulls().unique().sort().alias("_soma_depths"),
        )
        .sort(
            by=[soma_col, dmd_col, dend_col],
            maintain_order=True,
        )
    )


def _tree_from_summary(
    summary: pl.DataFrame,
    soma_col: str,
    dend_col: str,
    dmd_col: str,
) -> OrderedDict[str, dict[str, Any]]:
    rows = sorted(
        summary.to_dicts(),
        key=lambda row: (
            _natural_sort_key(row[soma_col]),
            _dmd_sort_key(row[dmd_col]),
            _natural_sort_key(row[dend_col]),
        ),
    )
    tree: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        soma_id = str(row[soma_col])
        soma = tree.setdefault(
            soma_id,
            {"dmds": [], "depths": [], "dendrites": []},
        )
        _append_unique(soma["dmds"], row[dmd_col])
        for depth in row["_soma_depths"]:
            _append_unique(soma["depths"], depth)
        soma["dendrites"].append({
            "dend_id": str(row[dend_col]),
            "dmd": row[dmd_col],
            "n_syn": int(row["n_synapses"]),
        })
    return tree


def _soma_metadata_from_dmd_info(
    *,
    subject: str | None,
    exp: str | None,
    loc: str | None,
    acq: str | None,
    dmd_info: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]] | None:
    context = [subject, exp, loc, acq]
    if all(value is None for value in context) and dmd_info is None:
        return None
    if any(value is None for value in context):
        raise ValueError(
            "To use dmd_info metadata, supply all of subject, exp, loc, and acq."
        )
    if dmd_info is None:
        from wisco_slap.meta.get import dmd_info as load_dmd_info

        dmd_info = load_dmd_info()

    try:
        acq_info = dmd_info[str(subject)][str(exp)][str(loc)][str(acq)]
    except KeyError as exc:
        raise KeyError(
            f"Could not find acquisition in dmd_info.yaml: {subject}/{exp}/{loc}/{acq}"
        ) from exc

    soma_metadata: dict[str, dict[str, Any]] = {}
    for dmd_key, one_dmd_info in acq_info.items():
        dmd = _dmd_number(dmd_key)
        depth = one_dmd_info.get("depth")
        for soma_id in one_dmd_info.get("somas", []) or []:
            soma_id = str(soma_id)
            if soma_id in soma_metadata:
                previous = _format_dmd(soma_metadata[soma_id]["dmd"])
                current = _format_dmd(dmd)
                raise ValueError(
                    f"{soma_id!r} is listed under both {previous} and {current} "
                    "in dmd_info.yaml. Each soma should belong to exactly one DMD."
                )
            soma_metadata[soma_id] = {"dmd": dmd, "depth": depth}
    return soma_metadata


def _apply_soma_metadata(
    tree: OrderedDict[str, dict[str, Any]],
    soma_metadata: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    if soma_metadata is None:
        return
    for soma_id, soma in tree.items():
        metadata = soma_metadata.get(soma_id)
        if metadata is None:
            continue
        soma["dmds"] = [metadata["dmd"]]
        depth = metadata.get("depth")
        soma["depths"] = [] if depth is None or depth == -1 else [depth]


def _compute_tree_layout(
    tree: OrderedDict[str, dict[str, Any]],
    soma_labels: Mapping[str, str],
) -> list[dict[str, Any]]:
    dend_gap = 1.55
    soma_gap = 1.35
    label_char_width = 0.115
    min_group_width = 1.1
    current_left = 0.0
    layout: list[dict[str, Any]] = []

    for soma_id, soma in tree.items():
        n_dendrites = len(soma["dendrites"])
        dendrite_width = max(0.0, (n_dendrites - 1) * dend_gap)
        max_label_line_len = max(
            len(line) for line in soma_labels[soma_id].splitlines()
        )
        label_width = max_label_line_len * label_char_width
        group_width = max(min_group_width, dendrite_width, label_width)
        soma_x = current_left + group_width / 2
        first_dend_x = soma_x - dendrite_width / 2
        dend_xs = [first_dend_x + i * dend_gap for i in range(n_dendrites)]
        layout.append({
            "soma_id": soma_id,
            "soma_x": soma_x,
            "dend_xs": dend_xs,
            "left": current_left,
            "right": current_left + group_width,
        })
        current_left += group_width + soma_gap
    return layout


def _font_size_for_count(count: int, *, base: float, floor: float) -> float:
    if count <= 8:
        return base
    return max(floor, base - (count - 8) * 0.12)


def _append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def _natural_sort_key(value: Any) -> tuple[str, int, str]:
    text = str(value)
    prefix = text.rstrip("0123456789")
    suffix = text[len(prefix) :]
    number = int(suffix) if suffix else -1
    return prefix, number, text


def _dmd_sort_key(value: Any) -> tuple[int, str]:
    number = _dmd_number(value)
    if number is None:
        return 999, str(value)
    return number, str(value)


def _dmd_number(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def _format_dmd(value: Any) -> str:
    number = _dmd_number(value)
    if number is not None:
        return f"DMD{number}"
    return f"DMD{value}"


def _format_dmds(values: Sequence[Any]) -> str:
    ordered = sorted(values, key=_dmd_sort_key)
    return "/".join(_format_dmd(value) for value in ordered)


def _format_depths(values: Sequence[Any]) -> str:
    if not values:
        return "depth=NA"
    ordered = sorted(values, key=_depth_sort_key)
    labels = [_format_depth(value) for value in ordered]
    if len(labels) == 1:
        return f"depth={labels[0]}um"
    return f"depths={'/'.join(labels)}um"


def _depth_sort_key(value: Any) -> tuple[float, str]:
    try:
        return float(value), str(value)
    except (TypeError, ValueError):
        return float("inf"), str(value)


def _format_depth(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def _format_soma_label(
    soma_id: str,
    dmds: Sequence[Any],
    depths: Sequence[Any],
    *,
    multiline: bool = True,
    max_soma_id_line_chars: int = 22,
) -> str:
    detail = f"{_format_dmds(dmds)}, {_format_depths(depths)}"
    if not multiline:
        return f"{soma_id} ({detail})"
    soma_text = "\n".join(
        textwrap.wrap(soma_id, width=max_soma_id_line_chars) or [soma_id]
    )
    return f"{soma_text}\n({detail})"


def _format_dendrite_label(dend_id: str, dmd: Any, n_synapses: int) -> str:
    syn_label = "syn" if n_synapses == 1 else "syns"
    return f"{dend_id}\n({_format_dmd(dmd)}, {n_synapses} {syn_label})"


def _dmd_color(
    value: Any,
    *,
    dmd1_color: str,
    dmd2_color: str,
    other_color: str,
) -> str:
    dmd_number = _dmd_number(value)
    if dmd_number == 1:
        return dmd1_color
    if dmd_number == 2:
        return dmd2_color
    return other_color


def _gen_box_onesided(
    data1,
    data2,
    colors=["gray", "blue"],
    ax=None,
    fsize=(4, 6),
    alpha=0.8,
    lineplot_width=4.5,
    one_sided=False,
    means=True,
    mean_color="gold",
    mean_linewidth=3.5,
    mean_linestyle="--",
    mean_dashes=(1.2, 1.2),
    widths=0.06,
    xlim=(0.35, 0.7),
    median_linewidth=4,
    whisker_linewidth=4,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=fsize)
    else:
        f = ax.get_figure()

    # boxplot - only NNXo
    box_o = ax.boxplot(
        data2,
        positions=[0.65],
        widths=widths,
        patch_artist=True,
        capprops=dict(color="none", linewidth=0),
        whiskerprops=dict(color="black", linewidth=whisker_linewidth),
        medianprops=dict(color="k", linewidth=median_linewidth, zorder=101),
        showfliers=False,
        showmeans=means,
        meanline=True,
        meanprops=dict(
            color=mean_color,
            linestyle=mean_linestyle,
            linewidth=mean_linewidth,
            dashes=mean_dashes,
            zorder=200,
        ),
    )
    box_o["boxes"][0].set_facecolor(colors[1])
    box_o["boxes"][0].set_alpha(alpha)
    box_o["boxes"][0].set_linewidth(0)
    box_o["boxes"][0].set_zorder(100)

    # line plots
    for i in range(len(data1)):
        ax.plot(
            [0.4, 0.6],
            [data1[i], data2[i]],
            color=colors[1],
            alpha=0.85,
            linewidth=lineplot_width,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    # scatter plots for individual points
    for i in range(len(data1)):
        ax.scatter(0.4, data1[i], color=colors[0], alpha=0.7, s=110, zorder=202)
        ax.scatter(0.6, data2[i], color=colors[1], alpha=0.7, s=110, zorder=203)
    ax.set_xlim(xlim)
    ax.set_xticks([0.4, 0.65])
    return f, ax


def gen_paired_boxplot(
    data1,
    data2,
    colors=["gray", "blue"],
    ax=None,
    fsize=(3.5, 4),
    lineplot_width=4.5,
    lineplot_alpha=0.85,
    scatter_alpha=0.7,
    one_sided=False,
    alphas=[0.8, 0.8],
    means=True,
    mean_color="gold",
    mean_linewidth=3.5,
    mean_linestyle="--",
    mean_dashes=(1.2, 1.2),
    widths=0.06,
    xlim=(0.3, 0.7),
    dot_size=110,
    median_linewidth=4,
    whisker_linewidth=4,
):
    if one_sided:
        return _gen_box_onesided(
            data1,
            data2,
            colors,
            ax,
            fsize,
            alphas[1],
            lineplot_width,
            one_sided,
            means,
            mean_color,
            mean_linewidth,
            mean_linestyle,
            mean_dashes,
            widths,
            median_linewidth=median_linewidth,
            whisker_linewidth=whisker_linewidth,
        )

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=fsize)
    else:
        f = ax.get_figure()

    # boxplots
    box = ax.boxplot(
        data1,
        positions=[0.35],
        widths=widths,
        patch_artist=True,
        capprops=dict(color="none", linewidth=0),
        whiskerprops=dict(color="k", linewidth=whisker_linewidth),
        medianprops=dict(color="k", linewidth=median_linewidth, zorder=101),
        showfliers=False,
        showmeans=means,
        meanline=True,
        meanprops=dict(
            color=mean_color,
            linestyle=mean_linestyle,
            linewidth=mean_linewidth,
            dashes=mean_dashes,
            zorder=200,
        ),
    )

    box["boxes"][0].set_facecolor(colors[0])
    box["boxes"][0].set_alpha(alphas[0])
    box["boxes"][0].set_linewidth(0)
    box["boxes"][0].set_zorder(100)

    box_o = ax.boxplot(
        data2,
        positions=[0.65],
        widths=widths,
        patch_artist=True,
        capprops=dict(color="none", linewidth=0),
        whiskerprops=dict(color="black", linewidth=whisker_linewidth),
        medianprops=dict(color="k", linewidth=median_linewidth, zorder=101),
        showfliers=False,
        showmeans=means,
        meanline=True,
        meanprops=dict(
            color=mean_color,
            linestyle=mean_linestyle,
            linewidth=mean_linewidth,
            dashes=mean_dashes,
            zorder=200,
        ),
    )
    box_o["boxes"][0].set_facecolor(colors[1])
    box_o["boxes"][0].set_alpha(alphas[1])
    box_o["boxes"][0].set_linewidth(0)
    box_o["boxes"][0].set_zorder(100)

    # line plots
    for i in range(len(data1)):
        ax.plot(
            [0.40, 0.6],
            [data1[i], data2[i]],
            color=colors[1],
            alpha=lineplot_alpha,
            linewidth=lineplot_width,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    # scatter plots for individual points
    for i in range(len(data1)):
        ax.scatter(
            0.4, data1[i], color=colors[0], alpha=scatter_alpha, s=dot_size, zorder=202
        )
        ax.scatter(
            0.6, data2[i], color=colors[1], alpha=scatter_alpha, s=dot_size, zorder=203
        )
    ax.set_xlim(xlim)
    return f, ax


def add_boxplot(
    ax,
    data,
    positions=[0.5],
    widths=0.06,
    color="gray",
    means=True,
    mean_color="gold",
    mean_linewidth=3.5,
    mean_linestyle="--",
    mean_dashes=(1.2, 1.2),
    alpha=0.85,
    median_linewidth=3,
    whisker_linewidth=3,
):
    box = ax.boxplot(
        data,
        positions=positions,
        widths=widths,
        showfliers=False,
        patch_artist=True,
        capprops=dict(color="none", linewidth=0),
        whiskerprops=dict(color="k", linewidth=whisker_linewidth),
        medianprops=dict(color="k", linewidth=median_linewidth, zorder=101),
        showmeans=means,
        meanline=True,
        meanprops=dict(
            color=mean_color,
            linestyle=mean_linestyle,
            linewidth=mean_linewidth,
            dashes=mean_dashes,
            zorder=200,
        ),
    )
    box["boxes"][0].set_facecolor(color)
    box["boxes"][0].set_alpha(alpha)
    box["boxes"][0].set_linewidth(0)
    box["boxes"][0].set_zorder(100)
    return ax, box


def add_data_points(ax, data, x_pos=0.4, color="gray", alpha=0.8, s=110, zorder=202):
    for i in range(len(data)):
        ax.scatter(x_pos, data[i], color=color, alpha=alpha, s=s, zorder=zorder)
    return ax
