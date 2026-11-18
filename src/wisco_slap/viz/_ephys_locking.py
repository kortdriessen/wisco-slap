"""Slow-wave-locked overview of one soma.

One column block, top to bottom:

1. the mean (± SEM) EEG waveform around the alignment landmark (``align``:
   the trough by default, or wave start / up peak) in the style of
   :func:`electro_py.wavedet.plot.plot_average_waveform`: the median lag of
   every other landmark (start, trough, down→up mid-crossing, up peak, end) is
   marked by a dashed line that extends through every panel below (the
   alignment landmark solid red), a dot on the mean trace and a top-edge tick
   with an IQR bar, and the zero-crossing landmarks are labelled with their
   phase (0, π, 2π);
2. a synapse × lag heatmap of the wave-triggered mean of a synaptic scopex
   array (glutamate or calcium), rows grouped by dendrite and ordered by
   ``pos`` within each dendrite, with the dendrites of the more superficial
   DMD on top;
3. the wave-triggered mean of the summed synaptic array (green);
4. the soma jRGECO dF/F (light red) and
5. the soma spike probability (dark red).

Everything is computed on the arrays' own time axes (all loaders in
``wis.get`` put scopex, EEG and soma traces on the same ephys-relative axis),
so any single-soma synapse array works: LS dF/F, denoised, matchfilt SNR, ...
"""

from __future__ import annotations

from collections.abc import Sequence

import fluxplot.colors as flex
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr

from electro_py.wavedet.plot import (
    ALIGN_COLUMNS,
    draw_landmarks,
    landmark_lag_summary,
    landmark_line_style,
)
from wisco_slap.scope.viz import synapto_dendro_heatmap

#: wave landmark → column of the slow-wave table, in temporal order.
LANDMARK_COLUMNS: dict[str, str] = {
    "start": "start_time",
    "trough": "trough_time",
    "mid": "mid_crossing_time",
    "up_peak": "up_peak_time",
    "end": "end_time",
}

DEFAULT_LAGS = np.arange(-1.5, 1.5001, 0.01)

_COLOR_EEG = "#403E3C"
_COLOR_SUM = flex.green600
_COLOR_SOMA_CA = flex.red400
_COLOR_SPIKE_PROB = flex.red700
_COLOR_ZERO = "#B7B5AC"


# --------------------------------------------------------------------------- #
# numeric core
# --------------------------------------------------------------------------- #
def _to_uniform(values: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Place (rows, n) samples on a uniform grid at the median sampling step.

    Returns ``(grid_values, valid, t0, dt)``: samples that are missing (gaps
    between trials) or non-finite in any row are marked invalid and hold 0.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    time = np.asarray(time, dtype=np.float64)
    if time.size < 2:
        raise ValueError("need at least two time samples")
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("time axis must be increasing")
    idx = np.rint((time - time[0]) / dt).astype(np.int64)
    n = int(idx[-1]) + 1
    out = np.zeros((values.shape[0], n))
    valid = np.zeros(n, dtype=bool)
    finite = np.isfinite(values).all(axis=0)
    out[:, idx[finite]] = values[:, finite]
    valid[idx[finite]] = True
    return out, valid, float(time[0]), dt


def _grid_peth(
    values: np.ndarray,
    t0: float,
    dt: float,
    valid: np.ndarray,
    centers: np.ndarray,
    lags: np.ndarray,
    batch: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Event-triggered mean of uniform-grid ``values`` (rows, T) at ``lags`` — observed samples only.

    Returns ``(mean (rows, n_lags), count (n_lags,), sem (rows, n_lags))``; the
    SEM is across the contributing events (ddof = 1). Lags with no observed
    sample are NaN (SEM also where fewer than two events contribute).
    """
    rows, n_samples = values.shape
    tot = np.zeros((rows, lags.size))
    sq = np.zeros((rows, lags.size))
    cnt = np.zeros(lags.size, dtype=np.int64)
    for s in range(0, centers.size, batch):
        idx = np.rint((centers[s : s + batch, None] + lags[None, :] - t0) / dt).astype(np.int64)
        inside = (idx >= 0) & (idx < n_samples)
        idx = np.clip(idx, 0, n_samples - 1)
        obs = inside & valid[idx]
        v = np.where(obs[None], values[:, idx], 0.0)
        tot += v.sum(axis=1)
        sq += (v * v).sum(axis=1)
        cnt += obs.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, tot / cnt, np.nan)
        var = np.where(cnt > 1, (sq - cnt * mean * mean) / (cnt - 1), np.nan)
        sem = np.sqrt(np.clip(var, 0.0, None) / cnt)
    return mean, cnt, sem


def _trace_1d(da: xr.DataArray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Values and time of a 1-D (time) DataArray, squeezing any length-1 dims."""
    if "time" not in da.dims:
        raise ValueError(f"{name} must have a 'time' dimension")
    extra = [d for d in da.dims if d != "time"]
    for d in extra:
        if da.sizes[d] != 1:
            raise ValueError(f"{name} must be a single trace; dimension {d!r} has size {da.sizes[d]}")
    da = da.squeeze(extra, drop=True) if extra else da
    return np.asarray(da.values, dtype=np.float64), np.asarray(da["time"].values, dtype=np.float64)


def _waves_frame(waves) -> pl.DataFrame:
    """Accept a :class:`electro_py.wavedet.SlowWaves` (or anything with ``.df``) or a polars frame."""
    df = getattr(waves, "df", waves)
    if not isinstance(df, pl.DataFrame):
        raise TypeError("waves must be a SlowWaves result or a polars DataFrame with the wave time columns")
    missing = set(LANDMARK_COLUMNS.values()) - set(df.columns)
    if missing:
        raise ValueError(f"wave table lacks columns {sorted(missing)}")
    return df


def superficial_dmd(synapses: xr.DataArray) -> int:
    """The DMD whose synapses lie most superficially (smallest ``dmd-depth``); DMD 1 if depth is unknown."""
    dmds = np.asarray(synapses["dmd"].values, dtype=int)
    if "dmd-depth" not in synapses.coords:
        return int(dmds.min()) if dmds.size else 1
    depth = np.asarray(synapses["dmd-depth"].values, dtype=float)
    per_dmd = {int(d): float(np.nanmedian(depth[dmds == d])) for d in np.unique(dmds)}
    per_dmd = {d: v for d, v in per_dmd.items() if np.isfinite(v)}
    if not per_dmd:
        return int(dmds.min())
    return min(per_dmd, key=lambda d: (per_dmd[d], d))


def wave_locked_peths(
    synapses: xr.DataArray,
    waves,
    eeg: xr.DataArray,
    soma_ca: xr.DataArray | None = None,
    spike_prob: xr.DataArray | None = None,
    *,
    align: str = "trough",
    stages: str | Sequence[str] | None = None,
    lags: np.ndarray | None = None,
    population: str = "sum",
) -> dict:
    """Wave-triggered means of every trace shown by :func:`wave_locked_overview`.

    Parameters
    ----------
    synapses : xr.DataArray
        Single-soma synaptic scopex array, dims ``(syn_id, time)`` (any order),
        with the usual synapse coords (``dmd``, ``dend-ID``, ``pos``, ...).
        Samples that are NaN in any synapse are treated as unobserved.
    waves : SlowWaves | pl.DataFrame
        Slow-wave detection result (``epy.wavedet.detect_slow_waves``); its
        table must carry ``trough_time``, ``start_time``, ``mid_crossing_time``,
        ``up_peak_time`` and ``end_time`` on the same time axis as the arrays.
    eeg : xr.DataArray
        The EEG trace the waves were detected on (``time`` dim).
    soma_ca, spike_prob : xr.DataArray | None
        Soma jRGECO dF/F and CASCADE spike probability (``time`` dim).
    align : {"trough", "start", "up_peak"}
        Wave landmark every trace is locked to (lag 0). Default ``"trough"``
        (the DOWN-state minimum); ``"start"`` locks to the down-state onset
        zero-crossing.
    stages : str | Sequence[str] | None
        Keep only waves whose ``stage`` is in this set (needs a ``stage`` column).
        None keeps every wave in the table.
    lags : np.ndarray | None
        Lags (s) relative to the alignment landmark; default ±1.5 s in 10 ms steps.
    population : {"sum", "mean"}
        How the synaptic array is collapsed for the population trace.

    Returns
    -------
    dict
        ``lags``, ``align``, ``n_waves``, ``n_by_stage``, ``landmarks``
        (median offset from the alignment landmark for every landmark in
        :data:`LANDMARK_COLUMNS`, the alignment one at 0), ``landmark_summary``
        (the :func:`electro_py.wavedet.plot.landmark_lag_summary` table:
        median, IQR and phase of every landmark), ``eeg`` and
        ``eeg_sem``, ``syn`` (n_syn × n_lags), ``syn_mean`` and ``syn_sd``
        (per-synapse recording mean/SD), ``population``, ``soma_ca``,
        ``spike_prob`` and their ``*_sem`` (None when not given), ``synapses``
        (the input array, synapse order preserved) and ``dmd_first``.
    """
    if population not in ("sum", "mean"):
        raise ValueError("population must be 'sum' or 'mean'")
    if align not in ALIGN_COLUMNS:
        raise ValueError(f"align must be one of {sorted(ALIGN_COLUMNS)}")
    align_col = ALIGN_COLUMNS[align]
    lags = DEFAULT_LAGS if lags is None else np.asarray(lags, dtype=np.float64)
    max_lag = float(np.max(np.abs(lags)))

    syn = synapses.transpose("syn_id", "time")
    values = np.asarray(syn.values, dtype=np.float64)
    time = np.asarray(syn["time"].values, dtype=np.float64)
    t0, t1 = float(time[0]), float(time[-1])

    df = _waves_frame(waves)
    if stages is not None:
        if "stage" not in df.columns:
            raise ValueError("stages given but the wave table has no 'stage' column")
        stages = [stages] if isinstance(stages, str) else list(stages)
        df = df.filter(pl.col("stage").is_in(stages))
    df = df.filter((pl.col(align_col) - max_lag > t0) & (pl.col(align_col) + max_lag < t1))
    if df.height == 0:
        raise ValueError("no waves fall inside the synapse array's time range (after the stage filter)")
    centers = df[align_col].to_numpy().astype(np.float64)
    n_by_stage = (
        {str(k): int(v) for k, v in zip(*np.unique(df["stage"].to_numpy().astype(str), return_counts=True))}
        if "stage" in df.columns
        else {}
    )
    summary = landmark_lag_summary(df, align=align)
    median_by_col = dict(
        zip(summary["column"].to_list(), summary["median_s"].to_list(), strict=True)
    )
    landmarks = {lm: float(median_by_col[col]) for lm, col in LANDMARK_COLUMNS.items()}

    grid, valid, g_t0, g_dt = _to_uniform(values, time)
    syn_peth, _, _ = _grid_peth(grid, g_t0, g_dt, valid, centers, lags)
    finite = np.isfinite(values).all(axis=0)
    syn_mean = values[:, finite].mean(axis=1) if finite.any() else np.full(values.shape[0], np.nan)
    syn_sd = values[:, finite].std(axis=1) if finite.any() else np.full(values.shape[0], np.nan)
    pop = grid.sum(axis=0) if population == "sum" else grid.mean(axis=0)
    pop_peth, _, _ = _grid_peth(pop[None, :], g_t0, g_dt, valid, centers, lags)

    out = dict(
        lags=lags,
        align=align,
        n_waves=int(centers.size),
        n_by_stage=n_by_stage,
        landmarks=landmarks,
        landmark_summary=summary,
        syn=syn_peth,
        syn_mean=syn_mean,
        syn_sd=syn_sd,
        population=pop_peth[0],
        population_mode=population,
        synapses=syn,
        dmd_first=superficial_dmd(syn),
        soma_ca=None,
        soma_ca_sem=None,
        spike_prob=None,
        spike_prob_sem=None,
    )
    for name, da in (("eeg", eeg), ("soma_ca", soma_ca), ("spike_prob", spike_prob)):
        if da is None:
            continue
        v, t = _trace_1d(da, name)
        g, gv, gt0, gdt = _to_uniform(v[None, :], t)
        m, _, sem = _grid_peth(g, gt0, gdt, gv, centers, lags)
        out[name] = m[0]
        out[f"{name}_sem"] = sem[0]
    return out


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def _contrast_line_color(cmap) -> str:
    """White for colormaps that are dark at their centre (e.g. iceburn), otherwise
    the dark neutral used by :func:`electro_py.wavedet.plot.landmark_line_style`."""
    try:
        r, g, b, _ = plt.get_cmap(cmap)(0.5)
    except (TypeError, ValueError):
        return landmark_line_style(False)["color"]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.4 else landmark_line_style(False)["color"]


def wave_locked_overview(
    synapses: xr.DataArray,
    waves,
    eeg: xr.DataArray,
    soma_ca: xr.DataArray | None = None,
    spike_prob: xr.DataArray | None = None,
    *,
    align: str = "trough",
    stages: str | Sequence[str] | None = None,
    lags: np.ndarray | None = None,
    population: str = "sum",
    heatmap_units: str = "delta",
    zlim_pct: float = 99.0,
    cmap=None,
    title: str | None = None,
    units_label: str | None = None,
    figsize: tuple[float, float] = (7.2, 12.5),
    show_legend: bool = True,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Slow-wave-locked overview of one soma.

    EEG ± SEM with landmarks, synaptic heatmap, synaptic sum, soma dF/F and
    spike probability, one column block.

    Parameters
    ----------
    synapses, waves, eeg, soma_ca, spike_prob, align, stages, lags, population
        See :func:`wave_locked_peths` (every trace is locked to the ``align``
        landmark, the trough by default). ``synapses`` must belong to a single soma;
        rows are grouped by ``dend-ID`` (ordered by ``pos`` within a dendrite) with
        the dendrites of the more superficial DMD (smaller ``dmd-depth``) on top.
    heatmap_units : {"delta", "z", "percent"}
        Colour of a heatmap cell: wave-triggered mean minus the synapse's recording
        mean (``"delta"``, in the array's own units), that divided by the recording
        SD (``"z"``), or as % of the recording mean (``"percent"``).
    zlim_pct : float
        Symmetric colour limit = this percentile of |heatmap values|.
    cmap
        Colormap for the heatmap; default ``fluxplot.colors.maps.iceburn``.
    title : str | None
        Title above the EEG panel; default "<n> waves (<stage counts>) · <n> synapses".
    units_label : str | None
        Units of ``synapses`` for the colorbar and the synaptic-sum axis (e.g. "SNR",
        "dF/F", "σ_null"); default taken from ``synapses.attrs["units"]`` if present.
    figsize : tuple[float, float]
        Figure size in inches for the single column block.
    show_legend : bool
        Draw the (two-entry) landmark legend below the block.

    Returns
    -------
    (fig, axes)
        ``axes`` maps ``"eeg"``, ``"heatmap"``, ``"population"``, ``"soma_ca"``,
        ``"spike_prob"`` to their Axes.
    """
    if heatmap_units not in ("delta", "z", "percent"):
        raise ValueError("heatmap_units must be 'delta', 'z' or 'percent'")
    P = wave_locked_peths(
        synapses,
        waves,
        eeg,
        soma_ca,
        spike_prob,
        align=align,
        stages=stages,
        lags=lags,
        population=population,
    )
    lags = P["lags"]
    units = units_label if units_label is not None else str(synapses.attrs.get("units", "")).strip()
    unit_sfx = f" ({units})" if units else ""

    with np.errstate(invalid="ignore", divide="ignore"):
        if heatmap_units == "delta":
            H = P["syn"] - P["syn_mean"][:, None]
            cb_label = f"mean − recording mean{unit_sfx}"
        elif heatmap_units == "z":
            H = (P["syn"] - P["syn_mean"][:, None]) / np.where(P["syn_sd"] > 0, P["syn_sd"], np.nan)[:, None]
            cb_label = "(mean − recording mean) / recording SD"
        else:
            H = (P["syn"] / P["syn_mean"][:, None] - 1.0) * 100.0
            cb_label = "% of recording mean"
    zlim = max(float(np.nanpercentile(np.abs(H), zlim_pct)) if np.isfinite(H).any() else 1e-6, 1e-6)

    if cmap is None:
        cmap = flex.maps.iceburn
    syn = P["synapses"]
    coords = {name: ("syn_id", c.values) for name, c in syn.coords.items() if c.dims == ("syn_id",) and name != "syn_id"}
    heat = xr.DataArray(H, dims=("syn_id", "time"), coords={"syn_id": syn["syn_id"].values, "time": lags, **coords})

    if title is None:
        groups = ", ".join(f"{s} {n}" for s, n in sorted(P["n_by_stage"].items()))
        title = f"{P['n_waves']} waves" + (f" ({groups})" if groups else "") + f" · {syn.sizes['syn_id']} synapses"
    landmark_x = P["landmarks"]
    align_txt = align.replace("_", " ")

    with plt.rc_context({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = fig.add_gridspec(5, 1, height_ratios=[1.0, 3.4, 0.8, 0.8, 0.8])
        ax_eeg = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1], sharex=ax_eeg)
        ax_pop = fig.add_subplot(gs[2], sharex=ax_eeg)
        ax_ca = fig.add_subplot(gs[3], sharex=ax_eeg)
        ax_spk = fig.add_subplot(gs[4], sharex=ax_eeg)
        axes = {"eeg": ax_eeg, "heatmap": ax_heat, "population": ax_pop, "soma_ca": ax_ca, "spike_prob": ax_spk}

        ax_eeg.plot(lags, P["eeg"], color=_COLOR_EEG, lw=1.4, zorder=3)
        ax_eeg.fill_between(
            lags,
            P["eeg"] - P["eeg_sem"],
            P["eeg"] + P["eeg_sem"],
            color=_COLOR_EEG,
            alpha=0.2,
            lw=0,
            zorder=2,
        )
        ax_eeg.axhline(0, color=_COLOR_ZERO, lw=0.6)
        ax_eeg.set_xlim(float(lags[0]), float(lags[-1]))
        n_label_rows = draw_landmarks(
            ax_eeg,
            P["landmark_summary"],
            align=align,
            lag=lags,
            curve=P["eeg"],
            xlim=(float(lags[0]), float(lags[-1])),
            label_fontsize=8.5,
        )
        ax_eeg.set_title(title, fontsize=8.5, pad=10 + 12 * n_label_rows)
        ax_eeg.set_ylabel("Mean EEG (µV)")

        _, _, im = synapto_dendro_heatmap(
            heat,
            ax=ax_heat,
            DMD_first=P["dmd_first"],
            zero_t=False,
            cmap=cmap,
            vmin=-zlim,
            vmax=zlim,
            dend_brackets=True,
            dmd_brackets=False,
            soma_brackets=False,
            boundary_color=_COLOR_EEG,
            boundary_lw=0.7,
            boundary_alpha=0.7,
            label_fontsize=7,
        )
        ax_heat.set_title("")
        ax_heat.set_xlabel("")
        ax_heat.set_ylabel(f"Synapses · dendrite → pos (DMD {P['dmd_first']} on top)", labelpad=28)
        cb = fig.colorbar(im, ax=ax_heat, location="top", orientation="horizontal", shrink=0.6, pad=0.012, aspect=30, extend="both")
        cb.set_ticks([-zlim, 0, zlim], labels=[f"{-zlim:.2f}", "0", f"{zlim:.2f}"])
        cb.set_label(cb_label, fontsize=7)
        cb.ax.tick_params(labelsize=7, length=2)

        ax_pop.plot(lags, P["population"], color=_COLOR_SUM, lw=1.3)
        ax_pop.set_ylabel(f"Synaptic {P['population_mode']}{unit_sfx}")
        for ax, key, color, label in (
            (ax_ca, "soma_ca", _COLOR_SOMA_CA, "Soma jRGECO\nmean dF/F"),
            (ax_spk, "spike_prob", _COLOR_SPIKE_PROB, "Soma spike prob.\n(CASCADE, per frame)"),
        ):
            if P[key] is not None:
                ax.plot(lags, P[key], color=color, lw=1.3)
            else:
                ax.set_title(f"no {label.split(chr(10))[0].lower()} signal", fontsize=7.5, color="#878580")
                ax.set_yticks([])
            ax.set_ylabel(label)

        # the EEG panel already carries its landmark lines (draw_landmarks); extend
        # the same lines through the panels below. On the heatmap the dashed lines
        # take a colour that contrasts with the colormap's centre and are heavier.
        heat_dash_color = _contrast_line_color(cmap)
        for i, ax in enumerate((ax_heat, ax_pop, ax_ca, ax_spk)):
            for name, x in landmark_x.items():
                style = landmark_line_style(name == align)
                if i == 0:
                    style = {**style, "lw": style["lw"] + 0.6, "alpha": 1.0}
                    if name != align:
                        style["color"] = heat_dash_color
                ax.axvline(x, zorder=5, **style)
            ax.set_xlim(float(lags[0]), float(lags[-1]))
        for ax in (ax_eeg, ax_heat, ax_pop, ax_ca):
            ax.tick_params(labelbottom=False)
        ax_spk.set_xlabel(f"s from wave {align_txt}")

        if show_legend:
            align_style = landmark_line_style(True)
            other_style = landmark_line_style(False)
            others = ", ".join(
                nm.replace("_", " ") for nm in LANDMARK_COLUMNS if nm != align
            )
            handles = [
                plt.Line2D(
                    [],
                    [],
                    color=align_style["color"],
                    ls=align_style["ls"],
                    lw=1.6,
                    label=f"wave {align_txt} (lag 0)",
                ),
                plt.Line2D(
                    [],
                    [],
                    color=other_style["color"],
                    ls=other_style["ls"],
                    lw=1.4,
                    label=(
                        f"median lag of {others} (zero-crossings labelled 0 / π / 2π); "
                        "▼ + bar: median & IQR"
                    ),
                ),
            ]
            fig.legend(
                handles=handles,
                loc="outside lower center",
                ncol=1,
                frameon=False,
                fontsize=7.5,
            )
    return fig, axes
