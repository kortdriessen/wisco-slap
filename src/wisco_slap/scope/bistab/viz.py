"""Loupe viewer with a live detector Tuner, and the dot-per-acquisition state plot.

Loupe and WISynaptic are imported lazily inside :func:`build_view` (they are
not wisco-slap dependencies; they live in the ``slap_mi_2_sleep`` env).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xarray as xr

from ... import defs
from .core import DEFAULTS, STATE_ORDER, STATE_SHORT, Detector
from .summary import state_table

#: Colours: ON warm, OFF cool (both readable over the dark Loupe background).
SPAN_COLORS = {"ON": "#f28e2b", "OFF": "#3fa9ff", "no data": "#8a8a8a"}


def _hypno_colors() -> dict[str, list[int]]:
    """5-state colours from ``wis.defs`` as RGBA lists for Loupe's label strip."""
    from matplotlib.colors import to_rgb

    cols = dict(defs.state_colors)
    cols.update(defs.state_color_conventions)
    return {k: [int(255 * c) for c in to_rgb(v)] + [200] for k, v in cols.items()}


def build_view(det: Detector, eeg: xr.DataArray | None = None, **param_defaults):
    """Open the Loupe viewer for one acquisition with a live detector Tuner.

    Panes: EEG, summed denoised glutamate, the detector's two working traces
    (N / E and the signed surprise with the seed thresholds), then one
    matched-filter-SNR heatmap per dendrite (superficial DMD first, synapses
    by position). ON/OFF periods are shaded across every pane; the 5-state
    hypnogram is the label strip on top; a live per-state table sits below.
    ``param_defaults`` override :data:`DEFAULTS` as the Tuner's starting values.
    Needs an :class:`AcqData` with ``mf_snr`` (i.e. from :func:`load_acq`, not a
    subset). Returns ``(app, params)``.
    """
    from loupe import (
        BoolParam,
        GlobalSpansConfig,
        HeatmapConfig,
        Param,
        TableConfig,
        TraceConfig,
        tunable,
        view,
    )
    from wisynaptic.data_loaders.soma_data import glu_cmap_for, trace_height_view_config

    acq = det.acq
    if acq.mf_snr is None:
        raise ValueError("build_view needs mf_snr; subsets/thinned copies drop it")
    d = {**DEFAULTS, **param_defaults}
    P = dict(
        z_min=Param(d["z_min"], 3.0, 10.0, step=0.25, name="event z_shot ≥"),
        window_s=Param(d["window_s"], 0.04, 0.6, step=0.01, name="window (s)"),
        baseline_s=Param(d["baseline_s"], 5.0, 120.0, step=5.0, name="baseline (s)"),
        seed=Param(d["seed"], 1.0, 10.0, step=0.25, name="seed (-log10 p)"),
        edge=Param(d["edge"], 0.5, 6.0, step=0.25, name="edge (-log10 p)"),
        shuffle=BoolParam(False, name="control: shuffle synapses"),
        refined=BoolParam(False, name="refined edges (event-based)"),
        margin_s=Param(0.5, 0.1, 1.0, step=0.05, name="refine margin (s)"),
    )
    core = dict(
        z_min=P["z_min"],
        window_s=P["window_s"],
        baseline_s=P["baseline_s"],
        shuffle=P["shuffle"],
    )
    configs: list = []
    if eeg is not None:
        configs.append(
            TraceConfig(eeg, color="#e6e4d9", array_name="eeg", view_id="bi:eeg")
        )
    dn_sum = xr.DataArray(acq.dn_sum, dims="time", coords={"time": acq.time})
    configs.append(
        TraceConfig(dn_sum, color="#87c45a", array_name="Σglu dn", view_id="bi:dnsum")
    )
    configs.append(
        TraceConfig(
            tunable(det.ratio_trace, **core),
            color="#d0d0d0",
            array_name="N / E",
            view_id="bi:ratio",
        )
    )
    configs.append(
        TraceConfig(
            tunable(det.surprise_trace, **core),
            color="#d0d0d0",
            array_name="surprise",
            overlay_arrays=[
                tunable(det.level_trace, P["seed"], 1.0),
                tunable(det.level_trace, P["seed"], -1.0),
            ],
            overlay_colors=[SPAN_COLORS["ON"], SPAN_COLORS["OFF"]],
            view_id="bi:surprise",
        )
    )

    mf = acq.syn_array(np.asarray(acq.mf_snr, dtype=np.float32), "mf_snr")
    order = (
        acq.syn
        .group_by("dend-ID")
        .agg(pl.col("dmd-depth").first(), pl.col("dend_type").first(), pl.len())
        .sort("dmd-depth", "dend-ID")
    )
    for dend, dtype in zip(order["dend-ID"], order["dend_type"], strict=True):
        sub = mf.isel(syn_id=np.flatnonzero(acq.syn["dend-ID"].to_numpy() == dend))
        configs.append(
            HeatmapConfig(
                sub,
                split_by="dend-ID",
                order_by="pos",
                cmap=glu_cmap_for(dtype),
                vmin=2.0,
                vmax=8.0,
                shade_nans=("#C0C0C0", 0.85),
                array_name=lambda _v, _s, n=f"{dend} glu": n,
            )
        )

    spans = GlobalSpansConfig(
        tunable(
            det.view_spans,
            seed=P["seed"],
            edge=P["edge"],
            refined=P["refined"],
            margin_s=P["margin_s"],
            **core,
        ),
        colors=SPAN_COLORS,
        alpha=0.28,
    )
    table = TableConfig(
        tunable(
            state_table,
            det,
            seed=P["seed"],
            edge=P["edge"],
            refined=P["refined"],
            margin_s=P["margin_s"],
            **core,
        ),
        title="ON/OFF by state (testable time only)",
        decimals=2,
    )
    app = view(
        configs,
        global_spans=spans,
        table=table,
        interval_labels=acq.hypno,
        label_colors=_hypno_colors(),
        label_strip_only=True,
        window_len=20.0,
        compact_heatmaps_to_fit=True,
        view_config=trace_height_view_config(configs),
    )
    # Loupe adds a legend to every trace pane; the threshold overlays would put
    # "surprise / overlay 0" text inside the data area, so hide them.
    for plt in app.plots:
        if getattr(plt, "legend", None) is not None:
            plt.legend.setVisible(False)
    subject, exp, loc, a, soma = acq.ids
    app.setWindowTitle(f"Synaptic bistability — {subject} {exp} {loc} {a} {soma}")
    return app, P


#: Per-acquisition joining lines and null markers (no greyscale data marks).
LINE_COLOR = defs.fpc.orange300
NULL_COLOR = defs.fpc.cyan600


def state_dots(
    ax,
    df: pl.DataFrame,
    col: str,
    group: str = "acq",
    scale: float = 1.0,
    color_by: str | None = None,
    palette: dict | None = None,
    mean: bool = True,
    states: list[str] | None = None,
    offset: float = 0.0,
    null: bool = False,
) -> list[str]:
    """Dot-per-group-per-state plot with lines joining each group left to right.

    ``df`` needs ``state``, ``group`` and ``col`` columns. Dots are coloured by
    state (default) or by ``color_by`` via ``palette``; ``mean`` adds a mean ± SD
    marker per state (across groups) in the state colour. ``null=True`` draws
    the same layout as small crosses (:data:`NULL_COLOR`) without lines or
    means (for a null model: pass the null values with the same groups, and
    they land next to their real dots). Returns the states plotted, in order.
    """
    states = states or [s for s in STATE_ORDER if s in set(df["state"].to_list())]
    cols = {**defs.state_colors, **defs.state_color_conventions}
    groups = list(dict.fromkeys(df[group].to_list()))
    offs = dict(zip(groups, np.linspace(-0.22, -0.02, max(len(groups), 1)) + offset))  # noqa: B905
    d = (
        df
        .filter(pl.col("state").is_in(states))
        .with_columns(
            pl.col("state").replace_strict(states, list(range(len(states)))).alias("_x")
        )
        .sort(group, "_x")
    )
    for g in groups:
        sub = d.filter(pl.col(group) == g)
        x = sub["_x"].to_numpy() + offs[g]
        y = sub[col].to_numpy() * scale
        if null:
            ax.scatter(
                x, y, s=14, marker="x", color=NULL_COLOR, linewidth=0.8, zorder=2
            )
            continue
        ax.plot(x, y, color=LINE_COLOR, lw=0.7, alpha=0.6, zorder=1)
        c = (
            [palette[v] for v in sub[color_by]]
            if color_by
            else [cols[s] for s in sub["state"]]
        )
        ax.scatter(
            x, y, s=16, c=c, alpha=0.8, edgecolor="white", linewidth=0.4, zorder=2
        )
    if mean and not null:
        for i, s in enumerate(states):
            v = (
                d.filter(pl.col("state") == s)[col].drop_nans().drop_nulls().to_numpy()
                * scale
            )
            if v.size == 0:
                continue
            sd = v.std(ddof=1) if v.size > 1 else 0.0
            ax.plot(
                [i + 0.18 + offset] * 2,
                [v.mean() - sd, v.mean() + sd],
                color=cols[s],
                lw=2,
                zorder=3,
            )
            ax.scatter(
                i + 0.18 + offset,
                v.mean(),
                s=50,
                color="white",
                edgecolor=cols[s],
                linewidth=2,
                zorder=4,
            )
    ax.set_xticks(range(len(states)), [STATE_SHORT.get(s, s) for s in states])
    ax.set_xlim(-0.6, len(states) - 0.4)
    return states
