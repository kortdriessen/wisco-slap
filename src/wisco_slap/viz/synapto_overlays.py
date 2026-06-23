"""Mean-IM + synapse-footprint overlay figures for SLAP2 acquisitions.

Reproduces the two notebook-style figures (one per DMD) used to inspect
which synapses survived scopex source-filtering, overlaid on the mean
reference image. Two coloring modes are supported:

* ``color_by_pos=False`` (default) — the "standard" overlay: each pixel
  belonging to a valid scopex source is colored by ``fpvals`` (the raw
  per-pixel footprint amplitude) through ``cmap``. All synapses look
  broadly the same; only the per-pixel footprint amplitude varies the
  shade within and between synapses.
* ``color_by_pos=True`` — each synapse is given a single flat color whose
  position in ``cmap`` reflects its proximal-to-distal index along its
  dendrite (``pos`` from the scopex coords), normalized to ``[0, 1]``
  per dendrite. With ``cmap='summer'`` this gives the most-proximal
  synapse on each dendrite the same green and the most-distal the same
  yellow, regardless of dendrite length.
"""

from __future__ import annotations

import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

import slap2_py as spy

from wisco_slap.get import syn_dF
from wisco_slap.meta.get import esum_mirror_path

# Bring the ``.sx`` xarray accessor into scope.
from wisco_slap.core import ScopeX_class  # noqa: F401


def _mask_fpvals_to_valid(
    fpv: np.ndarray, smap: np.ndarray, valid_sources: np.ndarray
) -> np.ndarray:
    """Set ``fpv`` pixels to NaN where ``smap``'s source is not in ``valid_sources``.

    ``smap`` encodes sources as 1-indexed ints (``-1`` = background);
    ``valid_sources`` (from the scopex ``syn_id`` coord) is 0-indexed.
    """
    valid_in_smap = np.asarray(valid_sources, dtype=int) + 1
    keep = np.isin(smap, valid_in_smap)
    out = fpv.astype(float, copy=True)
    out[~keep] = np.nan
    return out


def _normalize_pos_per_dendrite(
    poses: np.ndarray, dend_ids: np.ndarray
) -> np.ndarray:
    """Normalize ``pos`` to ``[0, 1]`` within each dendrite.

    Singleton dendrites (only one synapse, or all synapses sharing the same
    ``pos``) get 0.5 so they still receive a defined color. Synapses with a
    non-finite ``pos`` get NaN and are skipped during coloring.
    """
    norm = np.full(poses.shape, np.nan, dtype=float)
    for dend in np.unique(dend_ids):
        if dend is None:
            continue
        m = dend_ids == dend
        p = poses[m]
        finite = np.isfinite(p)
        if not finite.any():
            continue
        pmin = float(np.nanmin(p))
        pmax = float(np.nanmax(p))
        if pmax == pmin:
            norm[m] = np.where(finite, 0.5, np.nan)
        else:
            norm[m] = np.where(finite, (p - pmin) / (pmax - pmin), np.nan)
    return norm


def _build_pos_overlay(
    smap: np.ndarray,
    syn_ids: np.ndarray,
    norm_pos: np.ndarray,
    cmap_name: str,
) -> np.ndarray:
    """RGBA overlay (H, W, 4) where pixels of source ``sid`` get ``cmap(norm_pos[sid])``."""
    cmap = plt.get_cmap(cmap_name)
    overlay = np.zeros(smap.shape + (4,), dtype=float)
    for sid, npos in zip(syn_ids, norm_pos):
        if not np.isfinite(npos):
            continue
        px = smap == int(sid) + 1
        if not px.any():
            continue
        overlay[px] = cmap(float(npos))
    return overlay


def _select_synapses(
    da: xr.DataArray,
    channel: int | None,
    soma: str | None,
    require_dend: bool,
) -> xr.DataArray:
    if channel is not None and "channel" in da.dims:
        da = da.sel(channel=channel)
    if soma is not None:
        da = da.sx.somaid(soma)
    if require_dend:
        da = da.sx.dendid("any")
    return da


def _crop_bbox(img: np.ndarray, buf: int) -> tuple[int, int, int, int]:
    """Return ``(rmin, rmax, cmin, cmax)`` for the non-NaN bbox of ``img``, padded by ``buf``."""
    valid = ~np.isnan(img)
    if not valid.any():
        return 0, img.shape[0] - 1, 0, img.shape[1] - 1
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = int(max(0, rmin - buf))
    rmax = int(min(img.shape[0] - 1, rmax + buf))
    cmin = int(max(0, cmin - buf))
    cmax = int(min(img.shape[1] - 1, cmax + buf))
    return rmin, rmax, cmin, cmax


def _draw_scale_bar(
    ax: Axes,
    cmax: int,
    rmax: int,
    um_per_px: float,
    scale_bar_um: float,
    color: str = "black",
    pad_px: int = 10,
    linewidth: float = 2.0,
    fontsize: float = 10.0,
) -> None:
    bar_length_px = scale_bar_um / um_per_px
    bar_x = cmax - bar_length_px - pad_px
    bar_y = rmax - pad_px
    ax.plot(
        [bar_x, bar_x + bar_length_px],
        [bar_y, bar_y],
        color=color,
        linewidth=linewidth,
    )
    ax.text(
        bar_x + bar_length_px / 2,
        bar_y - 5,
        f"{scale_bar_um:g} µm",
        color=color,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )


def plot_synapto_overlays(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    *,
    color_by_pos: bool = False,
    soma: str | None = None,
    channel: int | None = 0,
    trace: Literal["denoised", "ls"] = "denoised",
    require_dend: bool = True,
    cmap: str = "summer",
    mean_im_channel: int = 1,
    log_vmin: float = 5.0,
    log_vmax_pct: float = 99.9,
    fp_threshold: float = 0.02,
    figsize: tuple[float, float] = (10.0, 10.0),
    crop_buf_px: int = 5,
    um_per_px: float = 0.25,
    scale_bar_um: float | None = 10.0,
    style_path: str | None = None,
    save_dir: str | None = None,
    filename_template: str | None = None,
    dpi: int = 600,
    transparent: bool = True,
    overlay_alpha: float = 1.0,
) -> dict[str, tuple[Figure, Axes]]:
    """Make a mean-IM + synapse-footprint overlay figure for each DMD.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    color_by_pos : bool
        If True, color each synapse with a flat color from ``cmap`` keyed by
        its normalized ``pos`` along its dendrite. If False (default),
        color pixels by raw ``fpvals`` through ``cmap`` (standard overlay).
    soma : str | None
        If given, restrict synapses to this soma (``'soma1'`` etc.). Default
        ``None`` includes all somas present in the scopex zarr.
    channel : int | None
        Scopex channel to select. Default 0 (iGluSnFR). Use ``None`` to skip
        channel selection.
    trace : ``'denoised'`` | ``'ls'``
        Which scopex synaptic-dF zarr to read coords from.
    require_dend : bool
        If True (default), drop synapses with no dendrite assignment. Always
        forced True when ``color_by_pos=True`` (``pos`` requires a dendrite).
    cmap : str
        Matplotlib colormap name for the overlay.
    mean_im_channel : int
        Channel index into ``get_meanIM(esum_p)[dmd]`` to use as the
        grayscale base. Default 1 (matches the notebook).
    log_vmin : float
        Lower bound for ``LogNorm`` on the mean-image grayscale.
    log_vmax_pct : float
        Percentile of the mean image used as ``LogNorm``'s ``vmax``.
    fp_threshold : float
        Footprint threshold passed to ``spy.xsum.get_fp_info``.
    figsize : (float, float)
        Per-figure size in inches.
    crop_buf_px : int
        Padding around the non-NaN bbox of the mean image when cropping.
    um_per_px : float
        Pixel pitch in microns, used for the scale bar.
    scale_bar_um : float | None
        Scale bar length in microns. ``None`` disables the scale bar.
    style_path : str | None
        Optional ``.mplstyle`` path to apply before plotting.
    save_dir : str | None
        If given, save each figure as PNG into this directory.
    filename_template : str | None
        Template for save filenames. Available substitutions:
        ``{dmd}``, ``{subject}``, ``{exp}``, ``{loc}``, ``{acq}``,
        ``{n_sources}``, ``{mode}`` (``'by_pos'`` or ``'fpvals'``).
        Default mirrors the notebook:
        ``"DMD{dmd}_mean_im_fp_overlay_{mode}__{n_sources}-SOURCES.png"``.
    dpi : int
        Save resolution.
    transparent : bool
        ``savefig(transparent=...)``.
    overlay_alpha : float
        Alpha passed to ``ax.imshow(overlay)`` in ``color_by_pos=False`` mode.
        ``color_by_pos=True`` ignores this — its overlay is RGBA already.

    Returns
    -------
    dict[str, tuple[Figure, Axes]]
        Mapping ``{'dmd_1': (fig, ax), 'dmd_2': (fig, ax)}``. Both keys are
        always present. If a DMD has no valid scopex synapses, the figure
        still shows the mean image with no overlay.
    """
    if color_by_pos:
        require_dend = True

    if style_path is not None:
        plt.style.use(style_path)

    esum_p = esum_mirror_path(subject, exp, loc, acq)
    mim = spy.xsum.get_meanIM(esum_p)
    smap_all, fpvals_all = spy.xsum.get_fp_info(esum_p, threshold=fp_threshold)
    dn = syn_dF(subject, exp, loc, acq, trace=trace, channels=[channel] if channel is not None else None)

    out: dict[str, tuple[Figure, Axes]] = {}
    for dmd in (1, 2):
        key = f"dmd_{dmd}"

        # Mean image (grayscale base)
        img = mim[dmd][mean_im_channel].copy()
        vmax = float(np.nanpercentile(img, log_vmax_pct))

        f, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img, norm=LogNorm(vmin=log_vmin, vmax=vmax), cmap="gray")

        # Build overlay, if there is a scopex selection for this DMD
        n_sources = 0
        if key in dn:
            da = _select_synapses(dn[key], channel, soma, require_dend)
            if da.sizes.get("syn_id", 0) > 0:
                syn_ids = np.asarray(da.syn_id.values, dtype=int)
                n_sources = int(syn_ids.size)
                smap = smap_all[dmd]
                if color_by_pos:
                    poses = np.asarray(da["pos"].values, dtype=float)
                    dend_ids = np.asarray(da["dend-ID"].values)
                    norm_pos = _normalize_pos_per_dendrite(poses, dend_ids)
                    overlay = _build_pos_overlay(smap, syn_ids, norm_pos, cmap)
                    ax.imshow(overlay)
                else:
                    fpv = fpvals_all[dmd][:]
                    fpv = _mask_fpvals_to_valid(fpv, smap, syn_ids)
                    ax.imshow(fpv, cmap=cmap, alpha=overlay_alpha)

        # Crop and frame
        rmin, rmax, cmin, cmax = _crop_bbox(img, crop_buf_px)
        ax.set_xlim(cmin, cmax)
        ax.set_ylim(rmax, rmin)
        ax.set_xticks([cmin, cmax])
        ax.set_yticks([rmin, rmax])
        ax.tick_params(
            top=False, right=False, bottom=False, left=False,
            labeltop=False, labelright=False, labelbottom=False, labelleft=False,
        )

        if scale_bar_um is not None:
            _draw_scale_bar(ax, cmax, rmax, um_per_px, scale_bar_um)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            mode = "by_pos" if color_by_pos else "fpvals"
            tmpl = filename_template or (
                "DMD{dmd}_mean_im_fp_overlay_{mode}__{n_sources}-SOURCES.png"
            )
            fname = tmpl.format(
                dmd=dmd, subject=subject, exp=exp, loc=loc, acq=acq,
                n_sources=n_sources, mode=mode,
            )
            f.savefig(
                os.path.join(save_dir, fname),
                dpi=dpi, bbox_inches="tight", transparent=transparent,
            )

        out[key] = (f, ax)

    return out
