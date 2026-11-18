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
  dendrite (``pos`` from the prox-line ordering in the annotation labels),
  normalized to ``[0, 1]`` per dendrite. With ``cmap='summer'`` this gives
  the most-proximal synapse on each dendrite the same green and the
  most-distal the same yellow, regardless of dendrite length.

The synapse selection (which sources to color, and their ``pos`` /
``dend-ID`` / ``soma-ID``) is read from ``wis.get.synid_labels`` — no raw
scopex traces are loaded.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tifffile
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

import slap2_py as spy

# Bring the ``.sx`` xarray accessor into scope (used by ``_select_synapses``,
# which is imported by the synapto-animation renderer).
from wisco_slap.core import ScopeX_class  # noqa: F401
from wisco_slap.get import synid_labels
from wisco_slap.meta.get import esum_mirror_path


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


def _select_syn_labels(
    idf: pl.DataFrame | None,
    dmd: int,
    soma: str | None,
    require_dend: bool,
) -> pl.DataFrame | None:
    """Select this DMD's synapses from the annotation-label table.

    This is the annotation-only equivalent of :func:`_select_synapses` — it
    reproduces the same soma / dendrite filtering, but works purely from
    :func:`wisco_slap.get.synid_labels` so no raw scopex traces need to be
    loaded just to know which sources to color. That the two are equivalent
    is not a coincidence: the ``dend-ID`` / ``soma-ID`` / ``pos`` coords that
    ``_select_synapses`` filters on are themselves left-joined onto the scopex
    array from ``synid_labels`` at load time.

    Parameters
    ----------
    idf : pl.DataFrame or None
        The full-acquisition label table from ``synid_labels``, or ``None``
        when no ``synapse_labels.csv`` exists (in which case there is nothing
        to overlay).
    dmd : int
        DMD number (1 or 2).
    soma : str or None
        If given, restrict to synapses on this soma. ``None`` keeps all somas.
    require_dend : bool
        If True, drop synapses with no dendrite assignment.

    Returns
    -------
    pl.DataFrame or None
        Filtered table (sorted by ``syn_id``), with columns including
        ``syn_id``, ``dend-ID`` and — when prox-line ordering exists for the
        acquisition — ``pos``. ``None`` when ``idf`` is ``None``.
    """
    if idf is None:
        return None
    df = idf.filter(pl.col("dmd") == dmd)
    if soma is not None:
        df = df.filter(pl.col("soma-ID") == soma)
    if require_dend:
        df = df.filter(pl.col("dend-ID").is_not_null())
    return df.sort("syn_id")


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
    cmax: float,
    rmax: float,
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
    require_dend: bool = True,
    cmap: str = "summer",
    mean_im_channel: int = 1,
    exclusion_mask: dict[str, str | os.PathLike[str]] | None = None,
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
    extra_nametag: str | None = None,
    dpi: int = 600,
    transparent: bool = True,
    overlay_alpha: float = 1.0,
    overlay_only: bool = False,
    force_same_scale: bool = False,
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
        ``None`` includes all somas present in the label table.
    require_dend : bool
        If True (default), drop synapses with no dendrite assignment. Always
        forced True when ``color_by_pos=True`` (``pos`` requires a dendrite).
    cmap : str
        Matplotlib colormap name for the overlay.
    mean_im_channel : int
        Channel index into ``get_meanIM(esum_p)[dmd]`` to use as the
        grayscale base. Default 1 (matches the notebook).
    exclusion_mask : dict[str, str | os.PathLike[str]] | None
        Optional mapping with keys ``'dmd1'`` and/or ``'dmd2'`` to napari
        label-layer TIFF paths. Each TIFF must match its mean image's 2D
        shape, optionally with extra zero padding on the bottom or right
        (as when napari uses a larger shared canvas). Pixel coordinates
        are preserved; masks are never resized. Nonzero label pixels are
        set to NaN before percentile scaling and cropping, and hidden in
        the footprint overlay.
        Omitted DMDs are unchanged.
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
    extra_nametag : str | None
        Append ``__{extra_nametag}`` before the filename extension, including
        when using a custom ``filename_template``. For example, ``'base'``
        produces ``...__base.png``. None or an empty string adds no suffix.
    dpi : int
        Save resolution.
    transparent : bool
        ``savefig(transparent=...)``. Forced True for ``overlay_only``.
    overlay_alpha : float
        Alpha passed to ``ax.imshow(overlay)`` in ``color_by_pos=False`` mode.
        ``color_by_pos=True`` ignores this — its overlay is RGBA already.
    overlay_only : bool
        Show and save only synaptic footprints on a transparent background,
        hiding the mean image and scale bar. With otherwise matching settings
        (including exclusion mask, scale bar, figsize, dpi and force_same_scale),
        the exported
        PNG has the same dimensions and pixel alignment as the normal export.
        ``overlay_alpha`` still applies; use a nonzero value to see footprints.
        Use ``extra_nametag``, ``save_dir`` or ``filename_template`` to retain both
        exports, since their default filenames are the same.
    force_same_scale : bool
        Use identical image-pixel magnification for both DMDs, so equal
        physical distances and footprint sizes render at the same size.
        Pad each cropped view symmetrically to the largest cropped width
        and height across the two DMDs. This preserves square pixels and
        gives both plots the same canvas dimensions with matching settings.

    Returns
    -------
    dict[str, tuple[Figure, Axes]]
        Mapping ``{'dmd_1': (fig, ax), 'dmd_2': (fig, ax)}``. Both keys are
        always present. If a DMD has no valid scopex synapses, the figure
        still shows the mean image with no overlay, or is fully transparent
        when ``overlay_only=True``.
    """
    if exclusion_mask is not None:
        invalid_keys = set(exclusion_mask) - {"dmd1", "dmd2"}
        if invalid_keys:
            raise ValueError(
                "exclusion_mask keys must be 'dmd1' or 'dmd2'; "
                f"got {invalid_keys!r}"
            )

    if color_by_pos:
        require_dend = True

    if style_path is not None:
        plt.style.use(style_path)

    esum_p = esum_mirror_path(subject, exp, loc, acq)
    mim = spy.xsum.get_meanIM(esum_p)
    smap_all, fpvals_all = spy.xsum.get_fp_info(esum_p, threshold=fp_threshold)
    # Which sources to color comes entirely from the annotation labels — no
    # raw traces needed. ``None`` when the acq has no synapse_labels.csv yet.
    idf = synid_labels(subject, exp, loc, acq)

    images: dict[int, np.ndarray] = {}
    exclusions: dict[int, np.ndarray | None] = {}
    bounds: dict[int, tuple[int, int, int, int]] = {}
    for dmd in (1, 2):
        # Mean image (grayscale base)
        img = mim[dmd][mean_im_channel].copy()
        excluded = None
        mask_key = f"dmd{dmd}"
        if exclusion_mask is not None and mask_key in exclusion_mask:
            labels = tifffile.imread(exclusion_mask[mask_key])
            if labels.ndim != 2 or any(
                mask_size < image_size
                for mask_size, image_size in zip(labels.shape, img.shape, strict=True)
            ):
                raise ValueError(
                    f"exclusion_mask[{mask_key!r}] must be a 2D TIFF "
                    f"with shape {img.shape} or extra zero padding on the "
                    f"bottom/right; got {labels.shape}"
                )
            height, width = img.shape
            if np.any(labels[height:, :] != 0) or np.any(labels[:, width:] != 0):
                raise ValueError(
                    f"exclusion_mask[{mask_key!r}] has nonzero labels outside "
                    f"the mean-image shape {img.shape}; check mask alignment"
                )
            labels = labels[:height, :width]
            excluded = labels != 0
            img = img.astype(float, copy=False)
            img[excluded] = np.nan
            if not np.isfinite(img).any():
                raise ValueError(
                    f"exclusion_mask[{mask_key!r}] leaves no finite mean-image pixels"
                )
        images[dmd] = img
        exclusions[dmd] = excluded
        bounds[dmd] = _crop_bbox(img, crop_buf_px)

    shared_height = max(b[1] - b[0] for b in bounds.values())
    shared_width = max(b[3] - b[2] for b in bounds.values())

    out: dict[str, tuple[Figure, Axes]] = {}
    for dmd in (1, 2):
        key = f"dmd_{dmd}"
        img = images[dmd]
        excluded = exclusions[dmd]
        vmax = float(np.nanpercentile(img, log_vmax_pct))

        f, ax = plt.subplots(1, 1, figsize=figsize)
        # Keep invisible artists in the layout so tight-bbox exports retain
        # exactly the same geometry when only the overlay is requested.
        ax.imshow(
            img, norm=LogNorm(vmin=log_vmin, vmax=vmax), cmap="gray",
            alpha=0.0 if overlay_only else None,
        )

        # Build overlay, if there is a synapse selection for this DMD
        n_sources = 0
        sel = _select_syn_labels(idf, dmd, soma, require_dend)
        if sel is not None and sel.height > 0:
            syn_ids = sel["syn_id"].to_numpy().astype(int)
            n_sources = int(syn_ids.size)
            smap = smap_all[dmd]
            if color_by_pos:
                if "pos" not in sel.columns:
                    raise KeyError(
                        "color_by_pos=True needs proximal-to-distal ordering, "
                        f"but no 'pos' data exists for {subject}/{exp}/{loc}/"
                        f"{acq} (dmd {dmd}). Draw prox lines in WISynaptic first."
                    )
                poses = sel["pos"].to_numpy().astype(float)
                dend_ids = sel["dend-ID"].to_numpy()
                norm_pos = _normalize_pos_per_dendrite(poses, dend_ids)
                overlay = _build_pos_overlay(smap, syn_ids, norm_pos, cmap)
                if excluded is not None:
                    overlay[excluded] = 0.0
                ax.imshow(overlay)
            else:
                fpv = fpvals_all[dmd][:]
                fpv = _mask_fpvals_to_valid(fpv, smap, syn_ids)
                if excluded is not None:
                    fpv[excluded] = np.nan
                ax.imshow(fpv, cmap=cmap, alpha=overlay_alpha)

        # Crop and frame
        rmin, rmax, cmin, cmax = bounds[dmd]
        if force_same_scale:
            row_pad = (shared_height - (rmax - rmin)) / 2
            col_pad = (shared_width - (cmax - cmin)) / 2
            rmin, rmax = rmin - row_pad, rmax + row_pad
            cmin, cmax = cmin - col_pad, cmax + col_pad
        ax.set_xlim(cmin, cmax)
        ax.set_ylim(rmax, rmin)
        ax.set_xticks([cmin, cmax])
        ax.set_yticks([rmin, rmax])
        ax.tick_params(
            top=False, right=False, bottom=False, left=False,
            labeltop=False, labelright=False, labelbottom=False, labelleft=False,
        )
        ax.spines[:].set_visible(False)

        if scale_bar_um is not None:
            _draw_scale_bar(ax, cmax, rmax, um_per_px, scale_bar_um)

        if overlay_only:
            f.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            for artist in [*ax.lines, *ax.texts]:
                artist.set_alpha(0.0)

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
            if extra_nametag:
                stem, extension = os.path.splitext(fname)
                fname = f"{stem}__{extra_nametag}{extension}"
            f.savefig(
                os.path.join(save_dir, fname),
                dpi=dpi, bbox_inches="tight", transparent=transparent or overlay_only,
            )

        out[key] = (f, ax)

    return out
