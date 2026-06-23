"""Renderer for the synapse float-off animation.

Reads a :class:`SceneConfig`, composes per-frame RGBA canvases of the user's
chosen size with both DMDs' mean images placed and both DMDs' synapses
flying to their target strips, and streams the frames into a single
``.mov`` file with an alpha channel (ProRes 4444 by default).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av  # PyAV
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import shift as ndi_shift, zoom as ndi_zoom

import slap2_py as spy

from wisco_slap.core import ScopeX_class  # noqa: F401  (registers .sx accessor)
from wisco_slap.get import syn_dF
from wisco_slap.meta.get import esum_mirror_path
from wisco_slap.viz.synapto_overlays import (
    _crop_bbox,
    _normalize_pos_per_dendrite,
    _select_synapses,
)

from .config import (
    DMD_KEYS,
    AnimationConfig,
    OverlayKwargs,
    Placement,
    ReferenceImage,
    SceneConfig,
    TargetStrip,
    load as load_config,
)


# ---------------------------------------------------------------------------
# Ordering helpers (mirrors scope/corr/plot.py)
# ---------------------------------------------------------------------------


def _order_synapses(da: xr.DataArray) -> np.ndarray:
    """Return the permutation that sorts ``da``'s ``syn_id`` axis by
    ``(soma-ID, dend-ID, pos)``."""
    n = da.sizes["syn_id"]
    somas = np.asarray(da["soma-ID"].values)
    dends = np.asarray(da["dend-ID"].values)
    poses = np.asarray(da["pos"].values, dtype=float)
    tuples = list(zip(somas.tolist(), dends.tolist(), poses.tolist(), range(n), strict=True))
    tuples.sort(key=lambda t: (str(t[0]), str(t[1]),
                                float(t[2]) if np.isfinite(t[2]) else np.inf, t[3]))
    return np.array([t[-1] for t in tuples], dtype=int)


def _dend_blocks(dends: np.ndarray) -> list[tuple[str, int, int]]:
    """Return ``[(dend_id, i_start, i_end_inclusive), ...]`` for runs of
    identical dend-IDs in an ordered array."""
    blocks: list[tuple[str, int, int]] = []
    if len(dends) == 0:
        return blocks
    cur = dends[0]
    start = 0
    for i in range(1, len(dends)):
        if dends[i] != cur:
            blocks.append((str(cur), start, i - 1))
            cur = dends[i]
            start = i
    blocks.append((str(cur), start, len(dends) - 1))
    return blocks


# ---------------------------------------------------------------------------
# Acquisition data loading (per DMD)
# ---------------------------------------------------------------------------


@dataclass
class _DmdData:
    """Per-DMD pre-rendered data needed for animation."""
    dmd: int
    placement: Placement
    target: TargetStrip
    # Background image already at scaled resolution, top-left aligned to (placement.x, placement.y)
    bg_rgba: np.ndarray | None       # (sh, sw, 4) uint8, or None if hidden
    # Sprite list, in matrix-row order
    syn_ids: list[int]
    dend_ids: list[str]
    pos_norm: list[float]
    sprite_rgba: list[np.ndarray]    # each (sh, sw, 4) uint8
    sprite_center_local: list[tuple[float, float]]  # (cy, cx) in sprite-local coords
    home_xy: list[tuple[float, float]]              # (x, y) on canvas
    end_xy: list[tuple[float, float]]               # (x, y) on canvas — target endpoint


def _gray_background_rgba(
    img: np.ndarray, log_vmin: float, log_vmax_pct: float
) -> np.ndarray:
    """Render ``img`` as RGBA uint8 grayscale via LogNorm + gray cmap.
    NaN pixels → fully transparent."""
    vmax = float(np.nanpercentile(img, log_vmax_pct))
    nan_mask = np.isnan(img)
    cmap = plt.get_cmap("gray")
    norm = LogNorm(vmin=log_vmin, vmax=vmax)
    rgba = cmap(norm(np.where(nan_mask, log_vmin, img)))
    rgba_u8 = (rgba * 255).astype(np.uint8)
    rgba_u8[..., 3] = np.where(nan_mask, 0, 255)
    return rgba_u8


def _zoom_rgba(rgba: np.ndarray, scale: float) -> np.ndarray:
    """Bilinearly zoom an RGBA uint8 image by ``scale``. Alpha-aware: zooms
    each channel independently with order=1."""
    if scale == 1.0:
        return rgba.copy()
    h, w = rgba.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    zoom_factors = (new_h / h, new_w / w, 1)
    out = ndi_zoom(rgba.astype(np.float32), zoom_factors, order=1, mode="constant", cval=0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _fpv_colored_rgba(fpv: np.ndarray, cmap_name: str) -> np.ndarray:
    """Render ``fpv`` through ``cmap`` with auto-normalization, NaN → transparent."""
    cmap = plt.get_cmap(cmap_name)
    nan_mask = np.isnan(fpv)
    if nan_mask.all():
        return np.zeros(fpv.shape + (4,), dtype=float)
    norm = Normalize(vmin=float(np.nanmin(fpv)), vmax=float(np.nanmax(fpv)))
    rgba = cmap(norm(np.where(nan_mask, 0.0, fpv)))
    rgba[..., 3] = np.where(nan_mask, 0.0, 1.0)
    return rgba


def _build_native_sprites(
    smap_crop: np.ndarray,
    fpv_crop: np.ndarray,
    syn_ids: np.ndarray,
    norm_pos: np.ndarray,
    cmap_name: str,
    color_by_pos: bool,
) -> dict[int, dict]:
    """Native-resolution RGBA sprites + local centers + bbox, keyed by syn_id."""
    cmap = plt.get_cmap(cmap_name)
    colored = _fpv_colored_rgba(fpv_crop, cmap_name) if not color_by_pos else None

    sprites: dict[int, dict] = {}
    for sid, npos in zip(syn_ids, norm_pos):
        mask = smap_crop == int(sid) + 1
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sh = y1 - y0
        sw = x1 - x0
        tile = np.zeros((sh, sw, 4), dtype=np.uint8)
        local_mask = mask[y0:y1, x0:x1]
        if color_by_pos and np.isfinite(npos):
            rgb = (np.asarray(cmap(float(npos)))[:3] * 255).astype(np.uint8)
            tile[..., :3] = rgb
        else:
            tile_rgb = (colored[y0:y1, x0:x1, :3] * 255).astype(np.uint8)
            tile[..., :3] = tile_rgb
        tile[..., 3] = np.where(local_mask, 255, 0).astype(np.uint8)
        cy_native = float(ys.mean())
        cx_native = float(xs.mean())
        sprites[int(sid)] = dict(
            rgba=tile,
            bbox=(y0, x0, y1, x1),
            center_native=(cy_native, cx_native),
            local_center=(cy_native - y0, cx_native - x0),
        )
    return sprites


# ---------------------------------------------------------------------------
# Endpoint distribution
# ---------------------------------------------------------------------------


def _strip_param(ts: TargetStrip, frac: float) -> tuple[float, float]:
    """Point on the strip's long axis at ``frac`` ∈ [0, 1]."""
    if ts.is_horizontal:
        return (ts.x0 + (ts.x1 - ts.x0) * frac, 0.5 * (ts.y0 + ts.y1))
    return (0.5 * (ts.x0 + ts.x1), ts.y0 + (ts.y1 - ts.y0) * frac)


def _lerp_xy(t: float, p0, p1) -> tuple[float, float]:
    return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


def _compute_endpoints(
    ts: TargetStrip,
    ordered_dend_ids: np.ndarray,
) -> list[tuple[float, float]]:
    """Return endpoints per synapse in the same order as ``ordered_dend_ids``."""
    n = len(ordered_dend_ids)
    out: list[tuple[float, float]] = [(0.0, 0.0)] * n
    if n == 0:
        return out
    blocks = _dend_blocks(ordered_dend_ids)
    if ts.end_mode == "dendrite_bars":
        n_blocks = len(blocks)
        total_gap = ts.bar_gap_frac * (n_blocks - 1) if n_blocks > 1 else 0.0
        usable = max(0.0, 1.0 - total_gap)
        block_lengths = [b[2] - b[1] + 1 for b in blocks]
        total_syn = sum(block_lengths)
        gap = ts.bar_gap_frac if n_blocks > 1 else 0.0
        cursor = 0.0
        for (_, i_start, i_end), bl in zip(blocks, block_lengths):
            seg_len = (bl / total_syn) * usable
            count = i_end - i_start + 1
            for k, i in enumerate(range(i_start, i_end + 1)):
                t = (k + 0.5) / count
                frac = cursor + t * seg_len
                out[i] = _strip_param(ts, frac)
            cursor += seg_len + gap
    else:  # dots
        for i in range(n):
            frac = (i + 0.5) / n
            out[i] = _strip_param(ts, frac)
    return out


# ---------------------------------------------------------------------------
# Per-DMD prep
# ---------------------------------------------------------------------------


def _prep_dmd_data(
    dmd: int,
    placement: Placement,
    target: TargetStrip,
    mean_im_full: np.ndarray,
    smap_full: np.ndarray,
    fpvals_full: np.ndarray,
    dn_full: xr.DataArray | None,
    ok: OverlayKwargs,
) -> _DmdData:
    """All per-DMD prep: crop, scale, build sprites, compute endpoints."""
    # Crop to non-NaN bbox (same as static overlay)
    rmin, rmax, cmin, cmax = _crop_bbox(mean_im_full, ok.crop_buf_px)
    mim_crop = mean_im_full[rmin:rmax + 1, cmin:cmax + 1]
    smap_crop = smap_full[rmin:rmax + 1, cmin:cmax + 1]
    fpv_crop = fpvals_full[rmin:rmax + 1, cmin:cmax + 1]

    # Background image at scaled resolution
    bg_native = _gray_background_rgba(mim_crop, ok.log_vmin, ok.log_vmax_pct)
    bg_scaled = _zoom_rgba(bg_native, placement.scale)

    # Build sprites at native res; we scale each tile individually below
    syn_ids: list[int] = []
    dend_ids: list[str] = []
    pos_norm: list[float] = []
    sprite_rgba: list[np.ndarray] = []
    sprite_center_local: list[tuple[float, float]] = []
    home_xy: list[tuple[float, float]] = []
    end_xy: list[tuple[float, float]] = []

    if dn_full is not None and dn_full.sizes.get("syn_id", 0) > 0:
        da = _select_synapses(dn_full, ok.channel, ok.soma, ok.require_dend)
        if da.sizes.get("syn_id", 0) > 0:
            order = _order_synapses(da)
            da = da.isel(syn_id=order)
            sid_arr = np.asarray(da.syn_id.values, dtype=int)
            dend_arr = np.asarray(da["dend-ID"].values)
            pos_arr = np.asarray(da["pos"].values, dtype=float)
            norm_arr = _normalize_pos_per_dendrite(pos_arr, dend_arr)

            native_sprites = _build_native_sprites(
                smap_crop, fpv_crop, sid_arr, norm_arr,
                ok.cmap, ok.color_by_pos,
            )

            # Keep only synapses with pixels
            for sid, dend, npos in zip(sid_arr, dend_arr, norm_arr):
                sid_i = int(sid)
                if sid_i not in native_sprites:
                    continue
                native = native_sprites[sid_i]
                tile_scaled = _zoom_rgba(native["rgba"], placement.scale)
                # Native center within the crop:
                cy_native, cx_native = native["center_native"]
                # Home center in canvas coords:
                home = (placement.x + cx_native * placement.scale,
                        placement.y + cy_native * placement.scale)
                # Sprite local center (within the scaled tile):
                ly, lx = native["local_center"]
                local = (ly * placement.scale, lx * placement.scale)

                syn_ids.append(sid_i)
                dend_ids.append(str(dend))
                pos_norm.append(float(npos) if np.isfinite(npos) else 0.5)
                sprite_rgba.append(tile_scaled)
                sprite_center_local.append(local)
                home_xy.append(home)

            # Endpoints in canvas coords
            end_xy = _compute_endpoints(target, np.asarray(dend_ids, dtype=object))

    return _DmdData(
        dmd=dmd,
        placement=placement,
        target=target,
        bg_rgba=bg_scaled,
        syn_ids=syn_ids,
        dend_ids=dend_ids,
        pos_norm=pos_norm,
        sprite_rgba=sprite_rgba,
        sprite_center_local=sprite_center_local,
        home_xy=home_xy,
        end_xy=end_xy,
    )


# ---------------------------------------------------------------------------
# Compositing primitives
# ---------------------------------------------------------------------------


def _alpha_over_inplace(
    canvas: np.ndarray, src_rgba: np.ndarray, dst_y: int, dst_x: int,
    src_y0: int, src_x0: int, src_y1: int, src_x1: int,
    alpha_mul: float = 1.0,
) -> None:
    """Alpha-over compositing of ``src_rgba`` (already cropped to overlap)
    onto ``canvas``[dst_y:dst_y+h, dst_x:dst_x+w]."""
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return
    h = src_y1 - src_y0
    w = src_x1 - src_x0
    dst_slice = canvas[dst_y:dst_y + h, dst_x:dst_x + w]
    src = src_rgba[src_y0:src_y1, src_x0:src_x1].astype(np.float32) / 255.0
    dst = dst_slice.astype(np.float32) / 255.0
    src_a = src[..., 3:4] * alpha_mul
    dst_a = dst[..., 3:4]
    out_a = src_a + dst_a * (1.0 - src_a)
    safe_a = np.where(out_a > 1e-6, out_a, 1.0)
    out_rgb = (src[..., :3] * src_a + dst[..., :3] * dst_a * (1.0 - src_a)) / safe_a
    composed = np.concatenate([out_rgb, out_a], axis=-1)
    dst_slice[...] = np.clip(composed * 255.0, 0, 255).astype(np.uint8)


def _composite_rgba(
    canvas: np.ndarray, rgba: np.ndarray, top_left_xy: tuple[float, float],
    alpha_mul: float = 1.0,
) -> None:
    """Composite an RGBA tile onto the canvas with its top-left at
    ``top_left_xy = (x, y)``. Integer placement (no sub-pixel)."""
    H, W = canvas.shape[:2]
    sh, sw = rgba.shape[:2]
    iy0 = int(round(top_left_xy[1]))
    ix0 = int(round(top_left_xy[0]))
    iy1 = iy0 + sh
    ix1 = ix0 + sw
    src_y0 = max(0, -iy0); src_x0 = max(0, -ix0)
    src_y1 = sh - max(0, iy1 - H)
    src_x1 = sw - max(0, ix1 - W)
    dst_y0 = max(0, iy0); dst_x0 = max(0, ix0)
    _alpha_over_inplace(canvas, rgba, dst_y0, dst_x0,
                        src_y0, src_x0, src_y1, src_x1, alpha_mul=alpha_mul)


def _composite_sprite(
    canvas: np.ndarray, sprite_rgba: np.ndarray,
    local_center: tuple[float, float],
    target_xy: tuple[float, float],
    alpha_mul: float = 1.0,
    sub_pixel: bool = True,
) -> None:
    """Composite a sprite so its local center lands at ``target_xy``."""
    if alpha_mul <= 0 or sprite_rgba.size == 0:
        return
    sh, sw = sprite_rgba.shape[:2]
    fy0 = target_xy[1] - local_center[0]
    fx0 = target_xy[0] - local_center[1]
    if sub_pixel:
        iy0 = int(np.floor(fy0))
        ix0 = int(np.floor(fx0))
        dy = fy0 - iy0
        dx = fx0 - ix0
        if dy > 1e-3 or dx > 1e-3:
            padded = np.pad(sprite_rgba, ((0, 1), (0, 1), (0, 0)))
            tile = ndi_shift(padded.astype(np.float32),
                             shift=(dy, dx, 0), order=1,
                             mode="constant", cval=0)
            tile = np.clip(tile, 0, 255).astype(np.uint8)
            sh, sw = tile.shape[:2]
        else:
            tile = sprite_rgba
            iy0, ix0 = int(round(fy0)), int(round(fx0))
    else:
        tile = sprite_rgba
        iy0, ix0 = int(round(fy0)), int(round(fx0))
    iy1 = iy0 + sh
    ix1 = ix0 + sw
    H, W = canvas.shape[:2]
    src_y0 = max(0, -iy0); src_x0 = max(0, -ix0)
    src_y1 = sh - max(0, iy1 - H)
    src_x1 = sw - max(0, ix1 - W)
    dst_y0 = max(0, iy0); dst_x0 = max(0, ix0)
    _alpha_over_inplace(canvas, tile, dst_y0, dst_x0,
                        src_y0, src_x0, src_y1, src_x1, alpha_mul=alpha_mul)


def _draw_dot_rgba(radius: float, color_rgba: tuple[int, int, int, int]) -> np.ndarray:
    r = max(1, int(np.ceil(radius)) + 1)
    sz = 2 * r + 1
    yy, xx = np.mgrid[0:sz, 0:sz]
    d = np.sqrt((yy - r) ** 2 + (xx - r) ** 2)
    a = np.clip(radius + 0.5 - d, 0, 1)
    tile = np.zeros((sz, sz, 4), dtype=np.uint8)
    tile[..., 0] = color_rgba[0]
    tile[..., 1] = color_rgba[1]
    tile[..., 2] = color_rgba[2]
    tile[..., 3] = (a * color_rgba[3]).astype(np.uint8)
    return tile


def _draw_rounded_bar(
    canvas: np.ndarray,
    p_start: tuple[float, float],
    p_end: tuple[float, float],
    thickness: float,
    cmap,
    color_min: float,
    color_max: float,
) -> None:
    """Draw an anti-aliased stadium (rounded-end rectangle) on an RGBA float
    canvas. Color varies linearly along the bar from ``cmap(color_min)`` at
    ``p_start`` to ``cmap(color_max)`` at ``p_end``. Uses distance-to-segment
    so the ends are naturally rounded half-disks."""
    H, W = canvas.shape[:2]
    sx, sy = p_start
    ex, ey = p_end
    half_t = thickness / 2.0
    pad = half_t + 1.0
    xmin = int(np.floor(min(sx, ex) - pad)); ymin = int(np.floor(min(sy, ey) - pad))
    xmax = int(np.ceil(max(sx, ex) + pad)); ymax = int(np.ceil(max(sy, ey) + pad))
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(W - 1, xmax); ymax = min(H - 1, ymax)
    if xmin >= xmax or ymin >= ymax:
        return

    yy, xx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1].astype(np.float32)
    vx = ex - sx; vy = ey - sy
    len_sq = vx * vx + vy * vy
    if len_sq < 1e-6:
        dist = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
        t_param = np.full_like(dist, 0.5)
    else:
        t = ((xx - sx) * vx + (yy - sy) * vy) / len_sq
        t_param = np.clip(t, 0.0, 1.0)
        cx = sx + t_param * vx
        cy = sy + t_param * vy
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    alpha_aa = np.clip(half_t + 0.5 - dist, 0.0, 1.0).astype(np.float32)
    if alpha_aa.max() == 0:
        return

    color_t = color_min + (color_max - color_min) * t_param
    rgba = cmap(color_t.ravel()).reshape(*alpha_aa.shape, 4).astype(np.float32)

    dst = canvas[ymin:ymax + 1, xmin:xmax + 1]
    src_a = rgba[..., 3] * alpha_aa
    dst_a = dst[..., 3]
    out_a = src_a + dst_a * (1.0 - src_a)
    safe_a = np.where(out_a > 1e-6, out_a, 1.0)
    src_a_e = src_a[..., None]
    out_rgb = (rgba[..., :3] * src_a_e
               + dst[..., :3] * dst_a[..., None] * (1.0 - src_a_e)) / safe_a[..., None]
    dst[..., :3] = out_rgb
    dst[..., 3] = out_a


def _build_end_layer(
    canvas_shape: tuple[int, int],
    per_dmd: list[_DmdData],
    cmap_name: str,
) -> np.ndarray:
    """Pre-render the static end-state layer (RGBA float, alpha 0-1).
    Combines per-DMD bars/dots. Dendrite bars use rounded (stadium) ends."""
    H, W = canvas_shape
    end_layer = np.zeros((H, W, 4), dtype=float)
    cmap = plt.get_cmap(cmap_name)

    for d in per_dmd:
        if not d.syn_ids:
            continue
        ts = d.target
        if ts.end_mode == "dendrite_bars":
            blocks = _dend_blocks(np.asarray(d.dend_ids, dtype=object))
            n_blocks = len(blocks)
            total_gap = ts.bar_gap_frac * (n_blocks - 1) if n_blocks > 1 else 0.0
            usable = max(0.0, 1.0 - total_gap)
            block_lengths = [b[2] - b[1] + 1 for b in blocks]
            total_syn = sum(block_lengths)
            gap = ts.bar_gap_frac if n_blocks > 1 else 0.0
            thickness = ts.bar_thickness_px
            cursor = 0.0
            for (_, i_start, i_end), bl in zip(blocks, block_lengths):
                seg_len = (bl / total_syn) * usable
                p_start = _strip_param(ts, cursor)
                p_end = _strip_param(ts, cursor + seg_len)
                pos_block = np.asarray(d.pos_norm[i_start:i_end + 1], dtype=float)
                finite = pos_block[np.isfinite(pos_block)]
                pmin, pmax = (0.0, 1.0) if (finite.size == 0 or finite.min() == finite.max()) \
                    else (float(finite.min()), float(finite.max()))
                _draw_rounded_bar(end_layer, p_start, p_end, thickness, cmap, pmin, pmax)
                cursor += seg_len + gap
        else:  # dots
            for i, sid in enumerate(d.syn_ids):
                npos = d.pos_norm[i]
                if not np.isfinite(npos):
                    npos = 0.5
                rgba = (np.asarray(cmap(npos)) * 255).astype(int).tolist()
                dot = _draw_dot_rgba(ts.dot_radius_px, tuple(rgba))
                ex, ey = d.end_xy[i]
                r = dot.shape[0] // 2
                iy0 = int(round(ey)) - r
                ix0 = int(round(ex)) - r
                iy1 = iy0 + dot.shape[0]
                ix1 = ix0 + dot.shape[1]
                src_y0 = max(0, -iy0); src_x0 = max(0, -ix0)
                src_y1 = dot.shape[0] - max(0, iy1 - H)
                src_x1 = dot.shape[1] - max(0, ix1 - W)
                dst_y0 = max(0, iy0); dst_x0 = max(0, ix0)
                dst_y1 = min(H, iy1); dst_x1 = min(W, ix1)
                if src_y1 <= src_y0 or src_x1 <= src_x0:
                    continue
                src = dot[src_y0:src_y1, src_x0:src_x1].astype(np.float32) / 255.0
                dst = end_layer[dst_y0:dst_y1, dst_x0:dst_x1]
                src_a = src[..., 3:4]
                dst_a = dst[..., 3:4]
                out_a = src_a + dst_a * (1.0 - src_a)
                safe_a = np.where(out_a > 1e-6, out_a, 1.0)
                out_rgb = (src[..., :3] * src_a + dst[..., :3] * dst_a * (1.0 - src_a)) / safe_a
                dst[..., :3] = out_rgb
                dst[..., 3:4] = out_a
    return end_layer


def _load_reference_rgba(path: str) -> np.ndarray:
    """Load an image file as an RGBA uint8 array. Accepts grayscale, RGB, RGBA."""
    arr = iio.imread(path)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 / np.max(arr)) if arr.max() > 0 else arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        h, w = arr.shape
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = arr; rgba[..., 1] = arr; rgba[..., 2] = arr
        rgba[..., 3] = 255
        return rgba
    if arr.shape[2] == 3:
        h, w, _ = arr.shape
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = arr
        rgba[..., 3] = 255
        return rgba
    if arr.shape[2] == 4:
        return arr
    raise ValueError(f"unsupported image shape from {path}: {arr.shape}")


def _composite_layer(canvas: np.ndarray, layer_rgba_float: np.ndarray, alpha_mul: float = 1.0) -> None:
    if alpha_mul <= 0:
        return
    src = layer_rgba_float
    dst = canvas.astype(np.float32) / 255.0
    src_a = src[..., 3:4] * alpha_mul
    dst_a = dst[..., 3:4]
    out_a = src_a + dst_a * (1.0 - src_a)
    safe_a = np.where(out_a > 1e-6, out_a, 1.0)
    out_rgb = (src[..., :3] * src_a + dst[..., :3] * dst_a * (1.0 - src_a)) / safe_a
    composed = np.concatenate([out_rgb, out_a], axis=-1)
    canvas[...] = np.clip(composed * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Easing + stagger
# ---------------------------------------------------------------------------


def _ease_in_out_cubic(p: float) -> float:
    p = max(0.0, min(1.0, p))
    return 3.0 * p * p - 2.0 * p * p * p


def _stagger_offsets(
    n: int, mode: Literal["wave", "random", "none"],
    max_offset: float, seed: int | None,
) -> np.ndarray:
    if n == 0 or mode == "none":
        return np.zeros(n, dtype=float)
    if mode == "wave":
        if n == 1:
            return np.zeros(1, dtype=float)
        return np.linspace(0.0, max_offset, n)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, max_offset, size=n)
    raise ValueError(f"unknown stagger mode: {mode}")


# ---------------------------------------------------------------------------
# PyAV streaming
# ---------------------------------------------------------------------------


def _open_stream(
    out_path: Path, codec: str, fps: int, width: int, height: int,
    prores_profile: int,
):
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    if codec == "prores_ks":
        stream.pix_fmt = "yuva444p10le"
        stream.options = {"profile": str(prores_profile), "vendor": "apl0"}
    elif codec == "qtrle":
        stream.pix_fmt = "argb"
    elif codec == "png":
        stream.pix_fmt = "rgba"
    else:
        raise ValueError(f"unsupported codec: {codec}")
    return container, stream


# ---------------------------------------------------------------------------
# Public renderer entry point
# ---------------------------------------------------------------------------


def render_scene(
    config: SceneConfig | str | os.PathLike,
    *,
    acquisition_override: tuple[str, str, str, str] | None = None,
    out_path_override: str | os.PathLike | None = None,
    progress: bool = True,
) -> Path:
    """Render the animation described by ``config`` to a ``.mov`` file.

    Parameters
    ----------
    config : SceneConfig | path-like
        Either a loaded ``SceneConfig`` or a path to a YAML config file.
    acquisition_override : tuple | None
        ``(subject, exp, loc, acq)`` to use instead of ``config.acquisition``.
        Reuses the canvas/placements/targets as a template.
    out_path_override : path-like | None
        Override ``config.output.path``.
    progress : bool
        Print progress every ~10% of frames.

    Returns
    -------
    Path
        Path of the written ``.mov`` file.
    """
    if not isinstance(config, SceneConfig):
        config = load_config(config)

    subject, exp, loc, acq = (
        acquisition_override if acquisition_override
        else config.acquisition.as_tuple()
    )
    out_path = Path(out_path_override) if out_path_override else Path(config.output.path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = config.overlay_kwargs
    anim = config.animation

    # Load raw data once
    esum_p = esum_mirror_path(subject, exp, loc, acq)
    mim = spy.xsum.get_meanIM(esum_p)
    smap_all, fpvals_all = spy.xsum.get_fp_info(esum_p, threshold=ok.fp_threshold)
    dn_all = syn_dF(
        subject, exp, loc, acq, trace=ok.trace,
        channels=[ok.channel] if ok.channel is not None else None,
    )

    # Per-DMD prep
    per_dmd: list[_DmdData] = []
    for dmd_int, key in zip((1, 2), DMD_KEYS):
        if key not in config.placements or key not in config.targets:
            continue
        per_dmd.append(_prep_dmd_data(
            dmd=dmd_int,
            placement=config.placements[key],
            target=config.targets[key],
            mean_im_full=mim[dmd_int][ok.mean_im_channel],
            smap_full=smap_all[dmd_int],
            fpvals_full=fpvals_all[dmd_int],
            dn_full=dn_all.get(key),
            ok=ok,
        ))

    # Flat global list of sprites (across DMDs). Order = dmd_int, then per-DMD order
    flat: list[dict] = []
    for d in per_dmd:
        for i in range(len(d.syn_ids)):
            flat.append(dict(
                dmd=d.dmd,
                dend=d.dend_ids[i],
                rgba=d.sprite_rgba[i],
                local=d.sprite_center_local[i],
                home=d.home_xy[i],
                end=d.end_xy[i],
            ))
    n_global = len(flat)

    # Stagger offsets across the global ordering (for the main flight phase)
    stagger_frac = anim.stagger_frac if anim.stagger != "none" else 0.0
    morph_frac = anim.morph_frac
    fly_frac = max(0.05, 1.0 - stagger_frac - morph_frac)
    duration_s = float(anim.duration_s)
    stagger_dur = stagger_frac * duration_s
    morph_dur = morph_frac * duration_s
    fly_dur = fly_frac * duration_s
    morph_start = duration_s - morph_dur
    offsets = _stagger_offsets(n_global, anim.stagger, stagger_dur, anim.seed)

    # Intro timing (open pause → appear → close pause), prepended to main animation.
    # Appearance is per-synapse: each synapse fires in proximal→distal order
    # within its dendrite, with a small per-synapse gap and an extra pause at
    # each dendrite boundary. Total appear duration is computed.
    intro_enabled = bool(anim.intro_enabled)
    intro_open = float(anim.intro_open_pause_s) if intro_enabled else 0.0
    intro_close = float(anim.intro_close_pause_s) if intro_enabled else 0.0
    appear_per_syn_fade = float(anim.intro_per_syn_fade_s)
    appear_per_syn_gap = float(anim.intro_per_syn_gap_s)
    appear_dend_gap = float(anim.intro_dend_gap_s)

    # Build a per-synapse appearance start-time schedule.
    # `appear_dend_gap` may be negative: dendrites then overlap in time
    # (the next dendrite starts before the previous one finishes).
    appear_starts = np.zeros(n_global, dtype=float)
    if n_global > 0 and intro_enabled:
        cursor = 0.0
        prev_key = None
        for i, item in enumerate(flat):
            key = (int(item["dmd"]), str(item["dend"]))
            if prev_key is not None and key != prev_key:
                cursor += appear_dend_gap
            appear_starts[i] = cursor
            cursor += appear_per_syn_gap
            prev_key = key
        # Shift so the earliest start is at 0 — handles negative dend_gap.
        appear_starts -= appear_starts.min()
        intro_appear = float(appear_starts.max() + appear_per_syn_fade)
    else:
        intro_appear = 0.0

    intro_dur = intro_open + intro_appear + intro_close
    total_dur = intro_dur + duration_s

    # Pre-rendered end-state layer
    end_layer = _build_end_layer(
        (config.canvas.height, config.canvas.width), per_dmd, ok.cmap,
    )

    # Pre-load reference images flagged for inclusion in the render
    refs_below: list[tuple[ReferenceImage, np.ndarray]] = []
    refs_above: list[tuple[ReferenceImage, np.ndarray]] = []
    for ref in config.references:
        if not ref.include_in_render or not ref.visible:
            continue
        try:
            rgba = _load_reference_rgba(ref.path)
        except Exception as e:
            print(f"[render] WARN: failed to load reference '{ref.path}': {e}")
            continue
        rgba_scaled = _zoom_rgba(rgba, ref.scale)
        (refs_above if ref.z_order == "above" else refs_below).append((ref, rgba_scaled))

    # Encoder
    container, stream = _open_stream(
        out_path, config.output.codec, anim.fps,
        config.canvas.width, config.canvas.height, config.output.prores_profile,
    )

    # Frame loop
    n_frames = int(round(total_dur * anim.fps))
    H, W = config.canvas.height, config.canvas.width
    bg_color = config.canvas.background_color  # None or [r,g,b,a]

    for fidx in range(n_frames):
        t_in_full = (fidx / max(1, n_frames - 1)) * total_dur if n_frames > 1 else 0.0

        # Determine phase + per-sprite alpha + main-anim t
        if intro_enabled and t_in_full < intro_dur:
            t_main = 0.0
            if t_in_full < intro_open:
                phase = "open_pause"
                base_sprite_alpha = 0.0
            elif t_in_full < intro_open + intro_appear:
                phase = "appear"
                t_appear = t_in_full - intro_open
                # per-sprite alpha computed inside the loop
                base_sprite_alpha = None
            else:
                phase = "close_pause"
                base_sprite_alpha = 1.0
        else:
            phase = "main"
            t_main = t_in_full - intro_dur if intro_enabled else t_in_full

        # Init canvas
        canvas = np.zeros((H, W, 4), dtype=np.uint8)
        if bg_color is not None:
            canvas[..., 0] = bg_color[0]
            canvas[..., 1] = bg_color[1]
            canvas[..., 2] = bg_color[2]
            canvas[..., 3] = bg_color[3] if len(bg_color) > 3 else 255

        # 1a. Reference layers (below)
        for ref, rgba in refs_below:
            _composite_rgba(canvas, rgba, top_left_xy=(ref.x, ref.y),
                            alpha_mul=ref.opacity)

        # 1b. Mean-image layers per DMD
        for d in per_dmd:
            if d.bg_rgba is not None:
                _composite_rgba(canvas, d.bg_rgba,
                                top_left_xy=(d.placement.x, d.placement.y),
                                alpha_mul=1.0)

        # 2. Sprites
        for i, item in enumerate(flat):
            if phase == "open_pause":
                continue   # synapses not yet visible
            elif phase == "appear":
                start_i = float(appear_starts[i])
                a = (t_appear - start_i) / max(appear_per_syn_fade, 1e-6)
                a = max(0.0, min(1.0, a))
                sp_alpha = _ease_in_out_cubic(a)
                x, y = item["home"]  # stay at home during appearance
            elif phase == "close_pause":
                sp_alpha = 1.0
                x, y = item["home"]
            else:  # main
                offset = float(offsets[i])
                p = (t_main - offset) / fly_dur if fly_dur > 0 else 1.0
                p = max(0.0, min(1.0, p))
                e = _ease_in_out_cubic(p)
                x, y = _lerp_xy(e, item["home"], item["end"])
                # Sprite alpha — fades to 0 during morph
                if t_main < morph_start:
                    sp_alpha = 1.0
                else:
                    sp_alpha = 1.0 - min(1.0, (t_main - morph_start) / max(morph_dur, 1e-6))
            if sp_alpha > 1e-3:
                _composite_sprite(canvas, item["rgba"], item["local"],
                                  target_xy=(x, y), alpha_mul=sp_alpha,
                                  sub_pixel=True)

        # 3. End-state layer fades in (only during main phase morph)
        if phase == "main" and t_main > morph_start and n_global > 0:
            end_alpha = min(1.0, (t_main - morph_start) / max(morph_dur, 1e-6))
            _composite_layer(canvas, end_layer, alpha_mul=end_alpha)

        # 4. Reference layers (above)
        for ref, rgba in refs_above:
            _composite_rgba(canvas, rgba, top_left_xy=(ref.x, ref.y),
                            alpha_mul=ref.opacity)

        # Encode frame
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(canvas), format="rgba")
        vf = vf.reformat(format=stream.pix_fmt)
        for packet in stream.encode(vf):
            container.mux(packet)

        if progress and (fidx % max(1, n_frames // 10) == 0):
            print(f"[render] frame {fidx + 1}/{n_frames} (t={t_in_full:.2f}s, phase={phase})")

    # Flush
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    if progress:
        print(f"[render] wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Light helper for the editor — landing-position preview
# ---------------------------------------------------------------------------


@dataclass
class LandingInfo:
    dmd: int
    syn_ids: list[int]
    dend_ids: list[str]
    pos_norm: list[float]
    end_xy: list[tuple[float, float]]


def landing_info_for_dmd(
    subject: str, exp: str, loc: str, acq: str, dmd: int,
    target: TargetStrip, overlay_kwargs: OverlayKwargs,
) -> LandingInfo:
    """Quick (sprite-free) landing-position info for a DMD, for editor live preview."""
    dn_all = syn_dF(
        subject, exp, loc, acq, trace=overlay_kwargs.trace,
        channels=[overlay_kwargs.channel] if overlay_kwargs.channel is not None else None,
    )
    key = f"dmd_{dmd}"
    syn_ids: list[int] = []
    dend_ids: list[str] = []
    pos_norm: list[float] = []
    end_xy: list[tuple[float, float]] = []
    da = dn_all.get(key)
    if da is not None and da.sizes.get("syn_id", 0) > 0:
        da = _select_synapses(da, overlay_kwargs.channel, overlay_kwargs.soma,
                              overlay_kwargs.require_dend)
        if da.sizes.get("syn_id", 0) > 0:
            order = _order_synapses(da)
            da = da.isel(syn_id=order)
            sid_arr = np.asarray(da.syn_id.values, dtype=int)
            dend_arr = np.asarray(da["dend-ID"].values)
            pos_arr = np.asarray(da["pos"].values, dtype=float)
            norm_arr = _normalize_pos_per_dendrite(pos_arr, dend_arr)
            syn_ids = sid_arr.tolist()
            dend_ids = [str(d) for d in dend_arr.tolist()]
            pos_norm = [float(p) if np.isfinite(p) else 0.5 for p in norm_arr]
            end_xy = _compute_endpoints(target, np.asarray(dend_ids, dtype=object))
    return LandingInfo(
        dmd=dmd, syn_ids=syn_ids, dend_ids=dend_ids,
        pos_norm=pos_norm, end_xy=end_xy,
    )
