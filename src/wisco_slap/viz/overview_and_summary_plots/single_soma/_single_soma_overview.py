"""Single-soma overview: one acquisition-soma in the context of its experiment.

Top to bottom the figure shows (1) the whole experiment on a wall-clock axis
(sync blocks, acquisitions, EEG + spectrogram, whisking, hypnogram shading —
the same rows as ``acq_within_exp_plot`` with whisking in place of the load
cell), (2) the acquisition itself on a zoomed wall-clock axis (CASCADE spike
times, the soma's jRGECO dF/F, and a synapto-dendro heatmap of the denoised
glutamate traces of the soma's dendritic synapses, grouped DMD → dendrite and
ordered by ``pos``), (3) the DMD-1 / DMD-2 mean images with the soma's ROI
outline and every synapse footprint core coloured by class (this soma's
dendritic synapses / other dendritic synapses / synapses without a dendrite
ID), and (4) a line of summary statistics (duration, time in each sleep
state, non-NaN fractions).

Every experiment-context component is loaded on its own; whatever cannot be
loaded is skipped and listed in the report printed under the figure. The
acquisition-level data (traces, spikes, images, footprints) are required — a
missing piece raises, which the plot-set generator turns into a blank
placeholder plus an ``--ERROR.txt`` sidecar.
"""

from __future__ import annotations

import os
import warnings
from datetime import timedelta
from pathlib import Path

import fluxplot as fp
import fluxplot.colors as flex
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tifffile as tiff
import xarray as xr
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation
from scipy.signal import spectrogram as _spectrogram

import slap2_py as spy
import wisco_slap as wis
from wisco_slap.viz.overview_and_summary_plots._spec import PlotSpec

# the experiment-context rows reuse the helpers / constants of the
# acq_within_exp plot so the two figures stay visually consistent
from wisco_slap.viz.overview_and_summary_plots.acq_level import (
    _acq_within_exp as awe,
)

#: Width (inches) of the figure; the height follows from the rows and images.
FIG_WIDTH = 40.0
#: Mean-image channel shown at the bottom (index 1 = iGluSnFR, same as the
#: canvas PNGs).
MEANIM_CHANNEL = 1
#: Mean-image display stretch: the black level is this percentile of all
#: pixels (≈ the out-of-FOV background), the white level this percentile,
#: and the range in between is raised to ``MEANIM_GAMMA`` (< 1 lifts dim
#: dendrites so they stay visible next to a bright soma, which saturates).
MEANIM_BLACK_PCT = 50.0
MEANIM_WHITE_PCT = 99.7
MEANIM_GAMMA = 0.45
#: Frame drawn around each mean image (face behind the image, edge colour /
#: width of the frame) so the image bounds are obvious.
IMAGE_FACE = "#c8c8c8"
IMAGE_EDGE = "black"
IMAGE_EDGE_LW = 1.5
#: A footprint pixel is drawn only if its weight is >= this fraction of the
#: synapse's own peak weight (trims each footprint to its core so neighbours do
#: not blob together; every synapse keeps at least its peak pixel).
FOOTPRINT_REL_CUT = 0.5
#: Footprint classes and their (fully opaque) colours: the soma's dendritic
#: synapses = exactly the synapses in the heatmap (soma + dend_only filters of
#: the synapse data pipe); other synapses with a dendrite ID (other somas);
#: synapses without a dendrite ID (onto somas / unlabelled). Each core is
#: ringed by a :data:`FOOTPRINT_OUTLINE_PX`-wide outline.
FOOTPRINT_COLORS = {
    "this soma": "#39ff14",
    "other dendritic": "#ff9500",
    "no dend-ID": "#1e90ff",
}
FOOTPRINT_OUTLINE = "black"
FOOTPRINT_OUTLINE_PX = 1
#: Soma ROI outline: the user-drawn WISynaptic mask TIFF when present (the
#: curated outline), else the ExSum ``userROIs`` polygon drawn at acquisition
#: (always present for a soma with an ``ROI_Fraw`` trace).
ROI_COLOR = "#ff1a1a"
ROI_LW = 3.0
#: Fraction of the image size left as a visible margin inside each frame.
IMAGE_PAD = 0.02
#: Heatmap: time columns after max-binning, colour-limit percentile, colormap.
HEATMAP_N_COLS = 12_000
HEATMAP_VMAX_PCT = 99.0
HEATMAP_CMAP = "magma"
#: Soma trace drawn at ~this rate (Hz) and clipped to these percentiles; a
#: time step longer than GAP_FACTOR median steps is drawn as a gap.
SOMA_PLOT_HZ = 50.0
GAP_FACTOR = 3.0
SOMA_CLIP_PCT: tuple[float, float] = (0.1, 99.9)
#: Sleep states reported in the summary-statistics row.
STAT_STATES: tuple[str, ...] = ("Wake", "NREM", "REM")
#: Text sizes (axis / tick labels are left at the style defaults).
FS_TITLE = 20
FS_BAR = 14
FS_IMAGE_TITLE = 18
FS_BRACKET = 12
FS_LEGEND = 14
FS_STATS = 18
#: Row height ratios of the trace panels (top to bottom).
_ROW_HEIGHTS = {
    "blocks": 1.0,
    "acquisitions": 1.0,
    "EEG_": 2.5,
    "EEG_ spectrogram": 2.5,
    "whisking": 2.5,
    "spikes": 0.6,
    "soma": 2.5,
    "heatmap": 7.0,
}
_CTX_ROWS = ("blocks", "acquisitions", "EEG_", "EEG_ spectrogram", "whisking")
_ACQ_ROWS = ("spikes", "soma", "heatmap")
#: Inches per height-ratio unit, fixed height (inches) of the stats row and
#: extra inches above the mean images for their titles.
_IN_PER_UNIT = 0.62
_STATS_HEIGHT_IN = 2.0
_IMAGE_TITLE_IN = 1.2
#: Fraction of the figure width actually spanned by the two image axes (the
#: rest is margins / labels); sizes the images row so the frames hug the images.
_IMAGE_WIDTH_FRACTION = 0.9


def single_soma_overview_path(
    subject: str, exp: str, loc: str, acq: str, soma_id: str
) -> Path:
    """Return where :func:`single_soma_overview_plot` saves its PNG.

    ``<annotation_root>/OVERVIEW_FIGURES/single_soma/single_soma_overview/
    <subject>--<exp>--<loc>--<acq>--<soma_id>.png``
    """
    return Path(
        os.path.join(
            wis.defs.annotation_root,
            "OVERVIEW_FIGURES",
            "single_soma",
            "single_soma_overview",
            f"{subject}--{exp}--{loc}--{acq}--{soma_id}.png",
        )
    )


def load_single_soma_data(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    fp_threshold: float = 0.01,
) -> dict:
    """Load everything :func:`single_soma_overview_plot` needs for one soma.

    Returns a dict with keys ``heat`` (denoised glutamate traces of the soma's
    dendritic synapses, channel 1, validity attached — the ``"synapse"`` data
    pipe with ``soma_id``, ``dend_only=True``, ``valid_mask=True``),
    ``heat_label`` (colorbar label), ``soma_ca`` (the soma's ``ROI_Fraw``
    dF/F, channel 0), ``spikes`` (CASCADE spike times, a polars frame with a
    ``time`` column), ``meanIM`` / ``syn_map`` / ``fp_vals`` (per-DMD mean
    images, 1-based footprint label map and footprint weights from
    ``spy.xsum.get_fp_info``), ``roi_polys`` (per-DMD ExSum ``userROIs`` list,
    each entry with ``Label`` and a ``(2, n)`` x/y ``Position``) and
    ``syn_labels`` (``wis.get.synid_labels`` for the acquisition, or None).

    The CASCADE step makes this slow (tens of seconds); pass the result to the
    plot's ``data`` argument when iterating on the figure.
    """
    common = dict(subject=subject, exp=exp, loc=loc, acq=acq)
    heat = wis.get.fetch_data_pipe(
        "synapse",
        trace="denoised",
        soma_id=soma_id,
        channel=1,
        dend_only=True,
        valid_mask=True,
        **common,
    )
    soma_ca, _spike_prob, spikes = wis.get.fetch_data_pipe(
        "soma", soma_id=soma_id, trace="Fraw", channel=0, return_dFF=True, **common
    )
    meanIM, syn_map, fp_vals = wis.get.fetch_data_pipe(
        "footprints", fp_threshold=fp_threshold, **common
    )
    return dict(
        heat=heat,
        heat_label="denoised dF (raw)",
        soma_ca=soma_ca,
        spikes=spikes,
        meanIM=meanIM,
        syn_map=syn_map,
        fp_vals=fp_vals,
        roi_polys=load_roi_polys(subject, exp, loc, acq),
        syn_labels=wis.get.synid_labels(**common),
    )


def load_roi_polys(subject: str, exp: str, loc: str, acq: str) -> dict[int, list]:
    """Per-DMD list of the ExSum ``userROIs`` (soma polygons drawn at acquisition).

    Each entry is a dict with ``Label`` (the soma ID, e.g. ``"soma1"``) and
    ``Position``, a ``(2, n)`` array of x (row 0) / y (row 1) pixel
    coordinates in the mean-image frame.
    """
    esum_p = wis.meta.get.esum_mirror_path(subject, exp, loc, acq)
    return {dmd: spy.xsum.get_roi_list(esum_p, dmd) for dmd in (1, 2)}


def _display_image(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Stretch a mean image for display so dendrites stay visible next to a soma.

    Black level = :data:`MEANIM_BLACK_PCT` percentile (≈ the out-of-FOV
    background level), white level = :data:`MEANIM_WHITE_PCT` percentile;
    the clipped, normalised range is raised to :data:`MEANIM_GAMMA`. Returns
    ``(display, black, white)`` with ``display`` in ``[0, 1]``.
    """
    img = np.asarray(img, dtype=float)
    black, white = np.nanpercentile(img, [MEANIM_BLACK_PCT, MEANIM_WHITE_PCT])
    if not np.isfinite([black, white]).all() or white <= black:
        black, white = float(np.nanmin(img)), float(np.nanmax(img))
        if white <= black:
            return np.zeros_like(img), black, white
    norm = np.clip((img - black) / (white - black), 0, 1)
    return norm**MEANIM_GAMMA, float(black), float(white)


def _decimate_keep_nan(
    t: np.ndarray, v: np.ndarray, step: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take every ``step``-th sample without bridging gaps.

    A kept sample becomes NaN when any of the samples it stands for is NaN, and
    the first sample after a jump in ``t`` (more than :data:`GAP_FACTOR` median
    sample intervals) is set to NaN as well, so neither invalid stretches nor
    missing trials get joined by a line between their neighbours.
    """
    if step > 1:
        n_full = (v.size // step) * step
        blocks = v[:n_full].reshape(-1, step)
        out = blocks[:, 0].copy()
        out[np.isnan(blocks).any(axis=1)] = np.nan
        t = t[:n_full:step]
    else:
        out = np.array(v, dtype=float, copy=True)
    if t.size > 2:
        dt = np.diff(t)
        out[1:][dt > GAP_FACTOR * np.median(dt)] = np.nan
    return t, out


def _bin_max_uniform(da: xr.DataArray, n_cols: int) -> tuple[xr.DataArray, np.ndarray]:
    """Max-pool a ``(syn_id, time)`` array onto a *uniform* time grid.

    The grid has ``min(n_cols, n_samples)`` equal-width bins spanning the first
    to the last sample time, so the pooled array can be drawn with a linear
    image extent that is faithful to the time axis: a stretch of time with no
    samples (a gap between acquisition epochs / missing trials) becomes empty
    bins, which are NaN, instead of being silently stretched away. A bin with
    samples is NaN only when all of them are NaN. Keeps every ``syn_id``
    coordinate; the ``time`` coordinate becomes the bin centre.

    Returns ``(pooled, edges)`` with ``edges`` the ``n_bins + 1`` bin edges.
    """
    t = np.asarray(da.time.values, dtype=float)
    n_t = t.size
    n_bins = max(1, min(n_cols, n_t))
    edges = np.linspace(t[0], t[-1], n_bins + 1) if n_t > 1 else np.array([t[0] - 0.5, t[0] + 0.5])
    starts = np.searchsorted(t, edges[:-1], side="left")  # first sample of each bin
    starts = np.minimum(starts, n_t - 1)
    counts = np.diff(np.append(starts, n_t))
    vals = np.asarray(da.values, dtype=float)
    pooled = np.fmax.reduceat(vals, starts, axis=1)  # fmax ignores NaN; all-NaN -> NaN
    pooled[:, counts == 0] = np.nan  # reduceat returns the element itself for empty ranges
    coords = {k: v for k, v in da.coords.items() if "time" not in v.dims}
    coords["time"] = ("time", (edges[:-1] + edges[1:]) / 2)
    return (
        xr.DataArray(pooled, dims=("syn_id", "time"), coords=coords, attrs=da.attrs),
        edges,
    )


def _time_gaps(t: np.ndarray, factor: float = GAP_FACTOR) -> tuple[int, float]:
    """Number and total duration (s) of sample intervals longer than
    ``factor`` median intervals (missing trials / epoch boundaries)."""
    t = np.asarray(t, dtype=float)
    if t.size < 3:
        return 0, 0.0
    dt = np.diff(t)
    gaps = dt > factor * np.median(dt)
    return int(gaps.sum()), float(dt[gaps].sum())


def _footprint_cores(
    syn_map: np.ndarray, fp_vals: np.ndarray | None, rel_cut: float
) -> np.ndarray:
    """Boolean map of footprint pixels at >= ``rel_cut`` of their synapse's peak.

    ``syn_map`` holds 1-based synapse labels (``-1`` / ``0`` = no synapse) and
    ``fp_vals`` the footprint weight of the winning synapse at each pixel.
    Without ``fp_vals`` (or with ``rel_cut <= 0``) the full footprints are
    returned. Every synapse keeps at least its peak pixel.
    """
    core = syn_map > 0
    if fp_vals is None or rel_cut <= 0:
        return core
    labels = np.where(core, syn_map, 0).astype(int)
    peak = np.zeros(int(labels.max()) + 1)
    w = np.where(core & np.isfinite(fp_vals), fp_vals, 0.0)
    np.maximum.at(peak, labels.ravel(), w.ravel())
    return core & (w >= rel_cut * peak[labels])


def _footprint_classes(
    syn_map: np.ndarray,
    dmd: int,
    heat: xr.DataArray,
    syn_labels: pl.DataFrame | None,
) -> dict[str, np.ndarray]:
    """Split the footprint labels of one DMD into the three overlay classes.

    ``"this soma"`` = the synapses present in ``heat`` on this DMD (i.e. those
    that passed the soma + ``dend_only`` + validity filters of the data pipe;
    matched on the ``dmd`` / ``syn_id_number`` coords, with the ``"<dmd>-<n>"``
    ``syn_id`` strings as fallback); ``"no dend-ID"`` = the remaining synapses
    whose ``dend-ID`` is null in ``syn_labels``; ``"other dendritic"`` =
    everything else. Without ``syn_labels`` every non-mine synapse is
    ``"other dendritic"``. Returns boolean pixel masks keyed by class; the
    values are 1-based ``syn_map`` labels internally (``syn_id + 1``).
    """
    if "syn_id_number" in heat.coords and "dmd" in heat.coords:
        dmd_of_syn = np.asarray(heat["dmd"].values, dtype=int)
        numbers = np.asarray(heat["syn_id_number"].values, dtype=int)
        mine_ids = numbers[dmd_of_syn == dmd]
    else:
        ids = [str(s).split("-") for s in heat["syn_id"].values]
        mine_ids = np.array([int(n) for d, n in ids if int(d) == dmd], dtype=int)
    mine = np.isin(syn_map, mine_ids + 1)
    present = syn_map > 0
    if syn_labels is not None:
        lab = syn_labels.filter(pl.col("dmd") == dmd)
        no_dend_ids = lab.filter(pl.col("dend-ID").is_null())["syn_id"].to_numpy()
        no_dend = np.isin(syn_map, no_dend_ids.astype(int) + 1) & ~mine
    else:
        no_dend = np.zeros(syn_map.shape, bool)
    return {
        "this soma": mine,
        "other dendritic": present & ~mine & ~no_dend,
        "no dend-ID": no_dend,
    }


def _overlay(ax: plt.Axes, mask: np.ndarray, color, alpha: float = 1.0) -> None:
    """Paint ``mask`` (bool image) onto ``ax`` in one colour."""
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask] = (*mcolors.to_rgb(color), alpha)
    ax.imshow(rgba, interpolation="nearest")


def _soma_dmd(subject: str, exp: str, loc: str, acq: str, soma_id: str) -> int:
    """The DMD whose ``dmd_info`` soma list contains ``soma_id``.

    Raises
    ------
    ValueError
        If ``soma_id`` is not listed under any DMD in ``dmd_info.yaml``.
    """
    dia = wis.meta.get.dmd_info()[subject][exp][loc][acq]
    dmd = next(
        (int(k.split("-")[1]) for k, v in dia.items() if soma_id in v["somas"]), None
    )
    if dmd is None:
        raise ValueError(
            f"{soma_id!r} is not listed in dmd_info for {subject}/{exp}/{loc}/{acq}"
        )
    return dmd


def _soma_polygon(roi_polys: dict[int, list] | None, dmd: int, soma_id: str):
    """The ``(2, n)`` x/y outline of ``soma_id`` on ``dmd`` from the ExSum ROIs, or None."""
    if not roi_polys:
        return None
    for roi in roi_polys.get(dmd, []):
        if str(roi.get("Label", "")) == soma_id:
            pos = np.asarray(roi["Position"], dtype=float)
            if pos.ndim == 2 and pos.shape[0] == 2 and pos.shape[1] >= 3:
                return pos
    return None


def _soma_mask(
    subject: str, exp: str, loc: str, acq: str, soma_id: str, dmd: int
) -> tuple[np.ndarray | None, str | None]:
    """The user-drawn ROI mask of ``soma_id`` on ``dmd`` (WISynaptic TIFF).

    Returns ``(mask, note)``: ``mask`` is boolean (True inside the soma ROI) or
    None when the TIFF is missing; ``note`` explains any fallback. The mask
    label is taken as the 1-based position of ``soma_id`` in that DMD's
    ``dmd_info`` soma list; a single-label mask is accepted regardless of its
    label value.
    """
    path = os.path.join(
        wis.defs.anmat_root,
        "annotation_materials",
        subject,
        exp,
        loc,
        acq,
        "roi_locations",
        f"roi_locs_dmd{dmd}_mask.tif",
    )
    if not os.path.exists(path):
        return None, f"ROI mask not found: {path}"
    raw = tiff.imread(path)
    labels = [int(v) for v in np.unique(raw) if v != 0]
    dia = wis.meta.get.dmd_info()[subject][exp][loc][acq]
    want = dia[f"dmd-{dmd}"]["somas"].index(soma_id) + 1
    if want in labels:
        return raw == want, None
    if len(labels) == 1:
        return raw == labels[0], f"mask has single label {labels[0]} (expected {want}); used it"
    return None, f"mask labels {labels} do not contain expected label {want}"


def _state_fractions(hypno, t0: float, t1: float) -> dict[str, float]:
    """Fraction of ``[t0, t1]`` (block-relative s) covered by each bout state.

    Bouts are clipped to the window; states not in :data:`STAT_STATES` are
    pooled into ``"other"`` and any uncovered time is reported as
    ``"unscored"``. ``hypno`` must expose a polars ``df`` with ``start_time``,
    ``end_time`` and ``state`` columns (an ``electro_py`` ``Hypnogram``).
    """
    total = t1 - t0
    starts = hypno.df["start_time"].to_numpy()
    ends = hypno.df["end_time"].to_numpy()
    states = hypno.df["state"].replace(wis.defs.substate_parent).to_numpy()
    overlap = np.clip(np.minimum(ends, t1) - np.maximum(starts, t0), 0, None)
    fr = {s: float(overlap[states == s].sum() / total) for s in STAT_STATES}
    fr["other"] = float(overlap[~np.isin(states, STAT_STATES)].sum() / total)
    fr["unscored"] = max(0.0, 1.0 - sum(fr.values()))
    return fr


def single_soma_overview_plot(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma_id: str,
    *,
    data: dict | None = None,
    hypno_version: str = "model",
    fig_width: float = FIG_WIDTH,
    dpi: int = 150,
    verbose: bool = True,
    overwrite: bool = False,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Single-soma overview: experiment context, soma + synapse activity, anatomy.

    Rows, top to bottom:

    1-5. Experiment context on a wall-clock axis spanning every sync block (as
         in ``acq_within_exp_plot``): sync blocks, acquisitions (this one
         filled), EEG_ trace, EEG_ spectrogram, whisking. Hypnogram shaded.
    6-8. This acquisition only (wall clock, zoomed): CASCADE spike times of the
         soma, the soma jRGECO dF/F (``ROI_Fraw``), and the synapto-dendro
         heatmap of ``data["heat"]`` (denoised glutamate traces of the soma's
         dendritic synapses; rows grouped DMD -> dendrite, ordered by ``pos``;
         max-binned in time to :data:`HEATMAP_N_COLS` columns). Hypnogram
         shaded on the spike / soma rows.
    9.   Mean image (channel :data:`MEANIM_CHANNEL`, stretched by
         :func:`_display_image`) of DMD-1 (left) and DMD-2 (right), framed,
         with the ROI of ``soma_id`` outlined (WISynaptic mask, else ExSum
         polygon) and every synapse footprint core (:data:`FOOTPRINT_REL_CUT`)
         coloured by class (:func:`_footprint_classes`).
    10.  Summary statistics: acquisition duration, fraction of the acquisition
         in each sleep state, non-NaN fraction of the synapse traces per DMD
         and of the soma trace; plus the footprint legend.

    The figure is saved to :func:`single_soma_overview_path` when no file
    exists there yet, or when ``overwrite=True``; otherwise the existing file
    is left alone. The figure is always built and returned.

    Parameters
    ----------
    subject, exp, loc, acq, soma_id
        Identifiers, e.g. ``"alnair", "exp_1", "loc_D", "acq_1", "soma1"``.
    data
        Output of :func:`load_single_soma_data`; loaded here when None. A dict
        without ``roi_polys`` gets them loaded from the ExSum mirror.
    hypno_version
        ``"model"`` or ``"validated"`` bout hypnogram.
    fig_width
        Figure width in inches; the height follows from the rows and images.
    dpi
        Resolution of the saved PNG.
    verbose
        Print the scale report and the list of skipped components.
    overwrite
        Re-save even if the PNG already exists.

    Returns
    -------
    (fig, axes)
        The figure and a dict of its axes keyed ``"blocks"``,
        ``"acquisitions"``, ``"EEG_"``, ``"EEG_ spectrogram"``, ``"whisking"``,
        ``"spikes"``, ``"soma"``, ``"heatmap"``, ``"dmd1"``, ``"dmd2"``,
        ``"stats"``.

    Raises
    ------
    KeyError
        If ``loc--acq`` is not listed for ``subject/exp`` in ``sync_info.yaml``.
    ValueError
        If ``soma_id`` is not listed under any DMD in ``dmd_info.yaml``.
    """
    fp.style.use_lighttable()
    wis.util.checks.validate_ids(exp, loc, acq)
    if data is None:
        data = load_single_soma_data(subject, exp, loc, acq, soma_id)
    heat, soma_ca, spikes = data["heat"], data["soma_ca"], data["spikes"]
    heat_label = data.get("heat_label", "synaptic signal")
    meanIM, syn_map, syn_labels = data["meanIM"], data["syn_map"], data["syn_labels"]
    fp_vals = data.get("fp_vals")
    roi_polys = data.get("roi_polys")
    if roi_polys is None:
        roi_polys = load_roi_polys(subject, exp, loc, acq)

    si = wis.meta.get.sync_info()[subject][exp]
    sync_blocks, acquisitions = si["sync_blocks"], si["acquisitions"]
    this_acq = f"{loc}--{acq}"
    if this_acq not in acquisitions:
        raise KeyError(
            f"{this_acq} is not in sync_info for {subject}/{exp}; "
            f"known: {sorted(acquisitions)}"
        )
    acq_sb = int(acquisitions[this_acq]["sync_block"])
    acq_block_start = awe._to_datetime(sync_blocks[acq_sb]["ephys_start"])
    soma_dmd = _soma_dmd(subject, exp, loc, acq, soma_id)

    def wall_acq(t):
        return awe._wall(t, acq_block_start)

    missing: list[str] = []
    report: list[str] = []
    dmd_of_syn = np.asarray(heat["dmd"].values, dtype=int)
    n_syn_by_dmd = {d: int((dmd_of_syn == d).sum()) for d in (1, 2)}

    # ------------------------------------------------------------------ layout
    im_w = {d: meanIM[d].shape[-1] for d in (1, 2)}
    im_h = meanIM[1].shape[-2]
    images_height_in = (
        _IMAGE_WIDTH_FRACTION * fig_width * im_h / sum(im_w.values()) * (1 + 2 * IMAGE_PAD)
        + _IMAGE_TITLE_IN
    )
    rows_height_in = _IN_PER_UNIT * sum(_ROW_HEIGHTS.values())
    fig = plt.figure(
        figsize=(fig_width, rows_height_in + images_height_in + _STATS_HEIGHT_IN),
        layout="constrained",
    )
    gs = fig.add_gridspec(
        len(_ROW_HEIGHTS) + 2,
        1,
        height_ratios=[
            *_ROW_HEIGHTS.values(),
            images_height_in / _IN_PER_UNIT,
            _STATS_HEIGHT_IN / _IN_PER_UNIT,
        ],
        hspace=0.03,
    )
    axes: dict[str, plt.Axes] = {}
    for i, name in enumerate(_CTX_ROWS + _ACQ_ROWS):
        share = None
        if name in _CTX_ROWS[1:]:
            share = axes[_CTX_ROWS[0]]
        elif name in _ACQ_ROWS[1:]:
            share = axes[_ACQ_ROWS[0]]
        axes[name] = fig.add_subplot(gs[i], sharex=share)
        if name not in (_CTX_ROWS[-1], _ACQ_ROWS[-1]):
            axes[name].tick_params(labelbottom=False)
    gs_im = gs[-2].subgridspec(1, 2, width_ratios=[im_w[1], im_w[2]], wspace=0.02)
    axes["dmd1"] = fig.add_subplot(gs_im[0])
    axes["dmd2"] = fig.add_subplot(gs_im[1])
    axes["stats"] = fig.add_subplot(gs[-1])
    axes["stats"].set_axis_off()

    # ------------------------------------------- rows 1-5: experiment context
    bar_y, bar_h = 0.0, 0.8
    bar_text_kw = dict(
        ha="center",
        va="center",
        fontsize=FS_BAR,
        zorder=awe.HYPNO_ZORDER + 1,
        clip_on=True,
    )
    ax_blocks, ax_acqs = axes["blocks"], axes["acquisitions"]
    for sb_num, info in sorted(sync_blocks.items()):
        start = awe._to_datetime(info["ephys_start"])
        end = awe._to_datetime(info["ephys_end"])
        corrupt = info.get("corrupt", "Unknown")
        color = (
            "tab:red"
            if corrupt is True
            else "tab:green"
            if corrupt is False
            else "tab:gray"
        )
        ax_blocks.barh(
            bar_y, end - start, left=start, height=bar_h, color=color, alpha=0.6,
            edgecolor="k",
        )
        ax_blocks.text(
            start + (end - start) / 2,
            bar_y,
            f"block-{sb_num}  (corrupt={corrupt})",
            **bar_text_kw,
        )
    ax_blocks.set_ylim(-0.5, 0.5)
    ax_blocks.set_yticks([])
    ax_blocks.set_ylabel("sync blocks")
    ax_blocks.set_title(
        f"{subject} / {exp} / {loc} / {acq}  —  {soma_id} (DMD-{soma_dmd}): "
        f"{heat.sizes['syn_id']} dendritic synapses "
        f"({n_syn_by_dmd[1]} on DMD-1, {n_syn_by_dmd[2]} on DMD-2)",
        fontsize=FS_TITLE,
    )

    this_timing = None
    for acq_id, entry in sorted(
        acquisitions.items(), key=lambda kv: str(kv[1].get("dat_start", ""))
    ):
        loc_, acq_ = acq_id.split("--")
        try:
            block_start = awe._to_datetime(
                sync_blocks[int(entry["sync_block"])]["ephys_start"]
            )
            timing = wis.meta.get_acq_timing(subject, exp, loc_, acq_)
        except Exception as e:
            missing.append(f"{acq_id}: acquisition timing — {awe._err(e)}")
            continue
        start = block_start + timedelta(seconds=timing.start_time)
        end = block_start + timedelta(seconds=timing.end_time)
        is_this = acq_id == this_acq
        if is_this:
            this_timing = timing
        ax_acqs.barh(
            bar_y,
            end - start,
            left=start,
            height=bar_h,
            edgecolor="k",
            linewidth=1.2,
            color=flex.orange400 if is_this else "none",
            alpha=0.8 if is_this else 1.0,
        )
        ax_acqs.text(
            start + (end - start) / 2,
            bar_y,
            f"{acq_id}  ({timing.duration / 60:.1f} min)",
            fontweight="bold" if is_this else "normal",
            **bar_text_kw,
        )
    ax_acqs.set_ylim(-0.5, 0.5)
    ax_acqs.set_yticks([])
    ax_acqs.set_ylabel("acquisitions")

    ax_eeg, ax_spec, ax_whisk = axes["EEG_"], axes["EEG_ spectrogram"], axes["whisking"]
    shaded_ctx = [ax_acqs, ax_eeg, ax_whisk]
    shaded_acq = [axes["spikes"], axes["soma"]]
    states_seen: set[str] = set()
    eeg_samples: list[np.ndarray] = []
    spec_tiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    whisk_segments: list[tuple[np.ndarray, np.ndarray]] = []
    acq_hypno = None  # bout hypnogram of the acquisition's own sync block (stats)
    for sb_num, info in sorted(sync_blocks.items()):
        block_start = awe._to_datetime(info["ephys_start"])
        tag = f"sync_block-{sb_num}"
        try:
            trace = wis.get.ephys(subject, exp, sb_num, stores=("EEG_",))["EEG_"]
            fs = float(trace.attrs.get("fs", awe.TRACE_PLOT_HZ))
            step = max(1, int(round(fs / awe.TRACE_PLOT_HZ)))
            ax_eeg.plot(
                awe._wall(trace.time.values[::step], block_start),
                trace.values[::step],
                color=flex.base850,
                linewidth=0.4,
            )
            eeg_samples.append(np.asarray(trace.values[::step], dtype=float))
            nperseg = int(round(awe.SPEC_WINDOW_S * fs))
            freqs, spg_t, spg = _spectrogram(
                np.asarray(trace.values, dtype=np.float64),
                fs,
                window="hann",
                nperseg=nperseg,
                noverlap=nperseg - int(round(awe.SPEC_STEP_S * fs)),
                detrend="constant",
            )
            keep = (freqs >= awe.SPEC_F_RANGE[0]) & (freqs <= awe.SPEC_F_RANGE[1])
            t_cols = awe._wall(float(trace.time.values[0]) + spg_t, block_start)
            spec_tiles.append((t_cols, freqs[keep], np.log10(spg[keep])))
        except Exception as e:
            missing.append(f"{tag}: EEG_ / spectrogram — {awe._err(e)}")
        try:
            whisk = wis.get.whisk_df(subject, exp, sb_num)
            whisk_segments.append((
                awe._wall(whisk["time"].to_numpy(), block_start),
                whisk["whis"].to_numpy().astype(float),
            ))
        except Exception as e:
            missing.append(f"{tag}: whisk_df — {awe._err(e)}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the loader warns when absent
                hypno = wis.get.sb_bout_hypno(
                    subject, exp, sb_num, version=hypno_version
                )
            if hypno is None:
                raise FileNotFoundError(f"no {hypno_version!r} bout hypnogram")
            if sb_num == acq_sb:
                acq_hypno = hypno
            targets = shaded_ctx + (shaded_acq if sb_num == acq_sb else [])
            for row in hypno.df.iter_rows(named=True):
                color = awe.STATE_COLORS.get(row["state"])
                if color is None:
                    continue
                states_seen.add(row["state"])
                for a in targets:
                    a.axvspan(
                        block_start + timedelta(seconds=row["start_time"]),
                        block_start + timedelta(seconds=row["end_time"]),
                        color=color,
                        alpha=awe.HYPNO_ALPHA,
                        linewidth=0,
                        zorder=awe.HYPNO_ZORDER,
                    )
        except Exception as e:
            missing.append(f"{tag}: hypnogram — {awe._err(e)}")

    how = f"{awe.TRACE_CLIP_PCT[0]}-{awe.TRACE_CLIP_PCT[1]} pct"
    if eeg_samples:
        pooled = np.concatenate(eeg_samples)
        lo, hi = np.nanpercentile(pooled, awe.TRACE_CLIP_PCT)
        if np.isfinite([lo, hi]).all() and hi > lo:
            ax_eeg.set_ylim(lo, hi)
        report.append(awe._yscale_line("EEG_", pooled, lo, hi, how))
    ax_eeg.set_ylabel(f"EEG_ (clipped to {how})")

    if spec_tiles:
        pooled_lp = np.concatenate([lp.ravel() for _, _, lp in spec_tiles])
        vmin, vmax = np.nanpercentile(pooled_lp, awe.SPEC_CLIP_PCT)
        mesh = None
        for x_cols, f_kept, lp in spec_tiles:
            mesh = ax_spec.pcolormesh(
                x_cols, f_kept, lp, cmap=awe.SPEC_CMAP, vmin=vmin, vmax=vmax,
                shading="nearest", rasterized=True,
            )
        cbar = fig.colorbar(mesh, cax=ax_spec.inset_axes([1.005, 0.0, 0.008, 1.0]))
        cbar.set_label("log10 power", fontsize=9)
        report.append(
            awe._yscale_line(
                "EEG_ spectrogram colour", pooled_lp, vmin, vmax,
                f"{awe.SPEC_CLIP_PCT[0]}-{awe.SPEC_CLIP_PCT[1]} pct",
            )
        )
    ax_spec.set_ylim(*awe.SPEC_F_RANGE)
    ax_spec.set_ylabel(
        f"EEG_ spectrogram (Hz)\n{awe.SPEC_WINDOW_S:g} s window, "
        f"{awe.SPEC_STEP_S:g} s step"
    )

    qlo, qhi = awe.DEFAULT_BEHAV_NORM_PCT["whisking"]
    if whisk_segments:
        pooled = np.concatenate([v for _, v in whisk_segments])
        try:
            _, lo, hi = awe._robust_norm01(pooled, qlo, qhi)
        except ValueError as e:
            missing.append(f"whisking: normalisation failed ({e}); drawn raw")
            for t, v in whisk_segments:
                ax_whisk.plot(t, v, **awe._BEHAV_STYLE["whisking"])
        else:
            for t, v in whisk_segments:
                ax_whisk.plot(
                    t, np.clip((v - lo) / (hi - lo), 0, 1), **awe._BEHAV_STYLE["whisking"]
                )
            ax_whisk.set_ylim(-0.05, 1.05)
            report.append(
                awe._yscale_line(
                    "whisking", pooled, lo, hi, f"0-1 = {qlo:g}-{qhi:g} pct, clipped"
                )
            )
    ax_whisk.set_ylabel(f"whisking (masked frame diff)\n(norm. {qlo:g}-{qhi:g} pct)")

    starts = [awe._to_datetime(v["ephys_start"]) for v in sync_blocks.values()]
    ends = [awe._to_datetime(v["ephys_end"]) for v in sync_blocks.values()]
    pad = (max(ends) - min(starts)) * 0.02
    ax_whisk.set_xlim(min(starts) - pad, max(ends) + pad)
    ax_whisk.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_whisk.set_xlabel("Wall-clock time (whole experiment)")
    handles = [
        Patch(color=awe.STATE_COLORS[s], alpha=awe.HYPNO_ALPHA + 0.3, label=s)
        for s in awe.STATE_COLORS
        if s in states_seen
    ]
    if handles:
        ax_eeg.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.005, 1),
            frameon=False,
            title="hypnogram",
            fontsize=FS_LEGEND,
            title_fontsize=FS_LEGEND,
        )

    # --------------------------------------- rows 6-8: this acquisition only
    ax_spk, ax_soma, ax_heat = axes["spikes"], axes["soma"], axes["heatmap"]
    spike_t = wall_acq(spikes["time"].to_numpy())
    ax_spk.vlines(spike_t, 0, 1, color=flex.red500, linewidth=0.6)
    ax_spk.set_ylim(0, 1)
    ax_spk.set_yticks([])
    ax_spk.set_ylabel(f"spikes\n(n={len(spike_t)})", rotation=0, ha="right", va="center")

    t_soma = soma_ca.time.values
    fs_soma = 1.0 / float(np.median(np.diff(t_soma[:1000])))
    step = max(1, int(round(fs_soma / SOMA_PLOT_HZ)))
    t_dec, v_dec = _decimate_keep_nan(t_soma, np.asarray(soma_ca.values, dtype=float), step)
    ax_soma.plot(wall_acq(t_dec), v_dec, color=flex.magenta700, linewidth=0.5)
    lo, hi = np.nanpercentile(soma_ca.values, SOMA_CLIP_PCT)
    if np.isfinite([lo, hi]).all() and hi > lo:
        ax_soma.set_ylim(lo, hi)
    report.append(
        awe._yscale_line(
            "soma dF/F", soma_ca.values, lo, hi,
            f"{SOMA_CLIP_PCT[0]}-{SOMA_CLIP_PCT[1]} pct",
        )
    )
    ax_soma.set_ylabel(f"{soma_id} jRGECO dF/F\n(ROI_Fraw)")

    binned, edges = _bin_max_uniform(heat, HEATMAP_N_COLS)
    vmax = float(np.nanpercentile(binned.values, HEATMAP_VMAX_PCT))
    _, _, im = wis.scope.viz.synapto_dendro_heatmap(
        binned,
        ax=ax_heat,
        zero_t=False,
        vmin=0,
        vmax=vmax,
        cmap=HEATMAP_CMAP,
        nan_color=flex.base300,
        soma_brackets=False,
        boundary_lw=0.6,
        label_fontsize=FS_BRACKET,
    )
    n_syn = binned.sizes["syn_id"]
    bin_s = float(edges[1] - edges[0])
    # the heatmap helper draws in seconds; re-anchor its (uniform-grid) extent
    # on the wall clock — the same axis the spike / soma rows are drawn on
    x0, x1 = mdates.date2num(wall_acq(np.array([edges[0], edges[-1]])))
    im.set_extent([x0, x1, n_syn - 0.5, -0.5])
    cbar = fig.colorbar(im, cax=ax_heat.inset_axes([1.005, 0.0, 0.008, 1.0]))
    cbar.set_label(heat_label, fontsize=9)
    ax_heat.set_ylabel(f"{n_syn} dendritic synapses of {soma_id}\n(NaN = grey)")
    ax_heat.yaxis.set_label_coords(-0.08, 0.5)  # just clear of the brackets
    ax_heat.set_xlabel("Wall-clock time (this acquisition)")
    n_gaps, gap_s = _time_gaps(heat.time.values)
    report.append(
        f"heatmap colour: [0, {vmax:.3g}] ({HEATMAP_VMAX_PCT:g} pct of max-binned "
        f"{heat_label}, {binned.sizes['time']} uniform time bins of {bin_s:.2f} s; "
        f"{n_gaps} time gap(s) totalling {gap_s:.0f} s drawn as empty columns)"
    )

    t_acq0, t_acq1 = float(heat.time.values[0]), float(heat.time.values[-1])
    pad_s = 0.01 * (t_acq1 - t_acq0)
    ax_heat.set_xlim(*wall_acq(np.array([t_acq0 - pad_s, t_acq1 + pad_s])))
    ax_heat.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_heat.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # ------------------------------------------ row 9: anatomy + footprints
    dia = wis.meta.get.dmd_info()[subject][exp][loc][acq]
    if syn_labels is None:
        missing.append(
            "synapse_labels.csv missing: cannot tell 'no dend-ID' synapses apart "
            "(all non-soma footprints drawn as 'other dendritic')"
        )
    if fp_vals is None:
        missing.append("fp_vals not in data: full footprints drawn (no core trimming)")
    outline_source = None
    for dmd in (1, 2):
        a = axes[f"dmd{dmd}"]
        img = np.asarray(meanIM[dmd][MEANIM_CHANNEL], dtype=float)
        shown, black, white = _display_image(img)
        a.imshow(shown, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        report.append(
            f"DMD-{dmd} mean image: black {black:.3g} ({MEANIM_BLACK_PCT:g} pct), "
            f"white {white:.3g} ({MEANIM_WHITE_PCT:g} pct), gamma {MEANIM_GAMMA:g}"
        )
        smap = syn_map[dmd]
        core = _footprint_cores(
            smap, None if fp_vals is None else fp_vals[dmd], FOOTPRINT_REL_CUT
        )
        classes = _footprint_classes(smap, dmd, heat, syn_labels)
        counts = {k: int(np.unique(smap[m]).size) for k, m in classes.items()}
        # outline first (dilated cores minus the cores), then the opaque cores
        if FOOTPRINT_OUTLINE_PX > 0:
            ring = binary_dilation(core, iterations=FOOTPRINT_OUTLINE_PX) & ~core
            _overlay(a, ring, FOOTPRINT_OUTLINE)
        for key in ("no dend-ID", "other dendritic", "this soma"):
            _overlay(a, classes[key] & core, FOOTPRINT_COLORS[key])
        if dmd == soma_dmd:
            mask, note = _soma_mask(subject, exp, loc, acq, soma_id, dmd)
            poly = _soma_polygon(roi_polys, dmd, soma_id)
            if mask is not None:
                a.contour(
                    mask.astype(float), levels=[0.5], colors=[ROI_COLOR],
                    linewidths=ROI_LW,
                )
                outline_source = "WISynaptic mask"
            elif poly is not None:
                xs, ys = np.append(poly[0], poly[0, 0]), np.append(poly[1], poly[1, 0])
                a.plot(xs, ys, color=ROI_COLOR, linewidth=ROI_LW, solid_joinstyle="round")
                outline_source = "ExSum ROI polygon"
                missing.append(f"{soma_id} ROI mask: {note}; drew the ExSum ROI polygon")
            else:
                missing.append(
                    f"{soma_id} outline: {note}; no ExSum ROI polygon either — NOT drawn"
                )
        # pin the view to the image (the outline plot would otherwise autoscale
        # with margins) with a thin visible margin inside the frame
        h_px, w_px = img.shape
        a.set_xlim(-0.5 - IMAGE_PAD * w_px, w_px - 0.5 + IMAGE_PAD * w_px)
        a.set_ylim(h_px - 0.5 + IMAGE_PAD * h_px, -0.5 - IMAGE_PAD * h_px)
        a.set_aspect("equal", adjustable="box")
        depth = dia[f"dmd-{dmd}"]["depth"]
        somas_here = ", ".join(dia[f"dmd-{dmd}"]["somas"]) or "none"
        a.set_title(
            f"DMD-{dmd}  |  depth {depth} µm  |  somas: {somas_here}\n"
            f"{counts['this soma']} syn of {soma_id}  |  "
            f"{counts['other dendritic']} other dendritic  |  "
            f"{counts['no dend-ID']} no dend-ID",
            fontsize=FS_IMAGE_TITLE,
        )
        a.set_xticks([])
        a.set_yticks([])
        a.set_facecolor(IMAGE_FACE)
        for spine in a.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(IMAGE_EDGE)
            spine.set_linewidth(IMAGE_EDGE_LW)

    # ------------------------------------------------ row 10: summary stats
    ax_stats = axes["stats"]
    fp_handles = [
        Patch(
            facecolor=FOOTPRINT_COLORS["this soma"], edgecolor=FOOTPRINT_OUTLINE,
            label=f"dendritic synapses of {soma_id} (the heatmap rows)",
        ),
        Patch(
            facecolor=FOOTPRINT_COLORS["other dendritic"], edgecolor=FOOTPRINT_OUTLINE,
            label="dendritic synapses of other somas",
        ),
        Patch(
            facecolor=FOOTPRINT_COLORS["no dend-ID"], edgecolor=FOOTPRINT_OUTLINE,
            label="synapses without a dend-ID",
        ),
        Patch(
            facecolor="none", edgecolor=ROI_COLOR, linewidth=ROI_LW,
            label=f"{soma_id} ROI outline (DMD-{soma_dmd}; {outline_source or 'MISSING'})",
        ),
    ]
    ax_stats.legend(
        handles=fp_handles,
        loc="upper right",
        ncol=2,
        frameon=False,
        title=(
            f"footprint overlays (pixels ≥ {FOOTPRINT_REL_CUT:.0%} of each synapse's "
            f"peak; mean image ch {MEANIM_CHANNEL}, gamma {MEANIM_GAMMA:g})"
        ),
        fontsize=FS_LEGEND,
        title_fontsize=FS_LEGEND,
    )

    duration_s = this_timing.duration if this_timing is not None else t_acq1 - t_acq0
    span_s = t_acq1 - t_acq0
    lines = [f"acquisition duration: {duration_s / 60:.1f} min ({duration_s:.0f} s)"]
    if n_gaps:
        lines[0] += (
            f"  |  wall-clock span {span_s / 60:.1f} min with {n_gaps} recording gap(s) "
            f"totalling {gap_s:.0f} s (grey in the heatmap, breaks in the soma trace)"
        )
    if acq_hypno is not None:
        fr = _state_fractions(acq_hypno, t_acq0, t_acq1)
        parts = [f"{s} {fr[s]:.1%}" for s in STAT_STATES]
        if fr["other"] > 0:
            parts.append(f"other {fr['other']:.1%}")
        if fr["unscored"] > 0.001:
            parts.append(f"unscored {fr['unscored']:.1%}")
        lines.append(f"time in state ({hypno_version} hypnogram): " + ", ".join(parts))
    else:
        lines.append("time in state: no hypnogram for this sync block")
    sub = max(1, heat.sizes["time"] // 200_000)
    finite = np.isfinite(heat.values[:, ::sub])
    nan_parts = []
    for dmd in (1, 2):
        rows = dmd_of_syn == dmd
        if rows.any():
            nan_parts.append(f"DMD-{dmd} {finite[rows].mean():.1%} ({int(rows.sum())} syn)")
        else:
            nan_parts.append(f"DMD-{dmd} n/a (0 syn)")
    nan_parts.append(f"{soma_id} trace {np.isfinite(soma_ca.values).mean():.1%}")
    lines.append("non-NaN fraction: " + ", ".join(nan_parts))
    ax_stats.text(
        0.0, 1.0, "\n".join(lines), transform=ax_stats.transAxes,
        ha="left", va="top", fontsize=FS_STATS, linespacing=1.6,
    )

    if verbose:
        print("scale report:\n  " + "\n  ".join(report))
        if missing:
            print(
                f"{subject} / {exp} / {loc} / {acq} / {soma_id} — skipped / notes:\n  "
                + "\n  ".join(missing)
            )
        else:
            print(
                f"{subject} / {exp} / {loc} / {acq} / {soma_id} — every component "
                "was found and plotted."
            )

    out_path = single_soma_overview_path(subject, exp, loc, acq, soma_id)
    if overwrite or not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        print(f"Saved single-soma overview to {out_path}")
    else:
        print(f"Single-soma overview already exists (overwrite=False): {out_path}")
    return fig, axes


PLOT_SPEC = PlotSpec(
    name="single_soma_overview",
    plot=single_soma_overview_plot,
    path=single_soma_overview_path,
)
