import math
import warnings
from typing import Optional, Tuple, Dict, Literal, Union, List
import wisco_slap as wis
import wisco_slap.defs as DEFS
import os

import numpy as np
import electro_py as epy
import polars as pl

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except Exception:
    _CUPY_AVAILABLE = False

try:
    import cv2

    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False

import av  # PyAV

_CUPY_AVAILABLE = False


def make_activity_movie(
    out_path: str,
    mean_im: np.ndarray,  # (H, W), arbitrary dtype
    source_map: np.ndarray,  # (H, W), -1 or [0..n_sources-1]
    act_data: np.ndarray,  # (n_sources, n_timepoints)
    act_time: Optional[np.ndarray] = None,  # (n_timepoints,), seconds
    soma_map: Optional[np.ndarray] = None,  # (H, W), 0 or [1..n_somas]
    soma_activity: Optional[np.ndarray] = None,  # (n_somas, n_timepoints)
    soma_time: Optional[np.ndarray] = None,  # (n_timepoints,), seconds
    # Timeline / resampling
    fps: Optional[float] = None,  # if None -> derive from act_time (or 1 frame/sample)
    start_time: Optional[
        float
    ] = None,  # seconds; if None -> overlap intersection start
    end_time: Optional[float] = None,  # seconds; if None -> overlap intersection end
    one_frame_per_sample_if_times_missing: bool = True,
    # Background rendering
    background_percentiles: Tuple[float, float] = (1.0, 99.5),
    background_gamma: float = 1.0,  # gamma on background after percentile scaling
    output_size: Optional[Tuple[int, int]] = None,  # (H_out, W_out); if None -> native
    # Synapse overlay styling (green)
    syn_threshold: float = 0.15,  # values below -> fully invisible
    syn_saturate_at: float = 1.0,  # value mapped to full alpha
    syn_alpha_max: float = 0.85,
    syn_alpha_gamma: float = 1.0,  # alpha = [(x-thr)/(sat-thr)]**gamma * alpha_max
    syn_color: Tuple[int, int, int] = (0, 255, 0),
    syn_alpha_blur_sigma: float = 0.0,  # px; requires OpenCV. 0 to disable.
    syn_alpha_only: bool = True,  # if True, color brightness stays constant; only alpha changes
    # Soma overlay styling (red)
    soma_threshold: float = 0.15,
    soma_saturate_at: float = 0.8,
    soma_alpha_max: float = 0.95,
    soma_alpha_gamma: float = 1.0,
    soma_color: Tuple[int, int, int] = (255, 0, 0),
    soma_alpha_blur_sigma: float = 0.0,
    soma_alpha_only: bool = True,
    # Event outline overlay
    soma_event_times=None,  # None | 1D array | list/tuple of 1D arrays (times in seconds)
    event_outline_frames: int = 3,  # number of frames to persist outline (incl. the detected frame)
    outline_width: int = 4,  # thickness (in pixels) for event outline; 2 ≈ twice current thickness
    outline_color: str = "#ffaa0d",  # hex color for outline; default bright orange
    # Performance / device
    device: Literal["cpu", "cuda", "auto"] = "auto",
    method: Literal[
        "dense", "auto"
    ] = "auto",  # dense gather is very fast up to 1080p; auto keeps it simple
    pad_to_even: bool = True,  # H.264 requires even dims for yuv420p
    preallocate_float_frames: bool = True,  # use float32 frame buffer repeatedly to avoid dtype flips
    # Encoding
    codec: str = "h264",  # "h264" (libx264) or "h264_nvenc" (if FFmpeg supports it)
    crf: Optional[int] = 18,  # for libx264; ignored by nvenc (use 'cq' below)
    preset: str = "veryfast",  # x264 preset; for nvenc use "p7","p6","p5",... or "slow"/"fast" if mapped
    pix_fmt: str = "yuv420p",  # "yuv420p" for compatibility
    nvenc_cq: Optional[int] = None,  # e.g. 19; used when codec == "h264_nvenc"
    gop: Optional[int] = None,  # e.g. fps*2; None keeps default
    tune: Optional[str] = None,  # e.g. "animation" for libx264
    video_options: Optional[
        Dict[str, str]
    ] = None,  # raw ffmpeg options if you need extra control
    # UX
    progress: bool = True,
    progress_every: int = 100,
):
    """
    Stream a high-quality movie of synaptic (green) and somatic (red) activity over a background image.

    Design:
      • Precompute static maps and background, pick frame times, resample activity once.
      • Per frame: dense label gather (one array index op), threshold→alpha mapping, optional alpha blur, alpha-over blend.
      • Stream frames to FFmpeg via PyAV (libx264 or NVENC).

    Assumptions:
      • act_data and soma_activity are normalized already (e.g., ΔF/F or z-score). Values below `*_threshold` are invisible.
      • source_map: -1 = background; 0..n_sources-1 label pixels
      • soma_map: 0 = background; 1..n_somas label pixels (index 0 in soma_activity corresponds to soma_map==1)

    Event outlines:
      • If `soma_event_times` is provided (1D array or list of arrays), each value is a time (sec)
        of a discrete event. Each event is mapped to the nearest video frame.
      • On each event, draw a solid colored outline of the corresponding soma's pixels
        (as defined by `soma_map`) on that frame and the next `event_outline_frames-1` frames.
      • Outline appearance:
          - `outline_width`: integer thickness in pixels (default 2).
          - `outline_color`: hex string "#RRGGBB" (default bright orange "#FFA500").
      • If a single array is provided, those events apply to all somas; if a list with length
        equal to number of somas is provided, events are applied per-soma.
      • The outline mask is computed once (CPU) and overlaid each active frame.

    Returns: None (writes video to `out_path`)
    """

    # --------------------
    # Helpers
    # --------------------
    def _percentile_scale_to_u8(
        img: np.ndarray, pl: float, ph: float, gamma: float
    ) -> np.ndarray:
        lo, hi = np.nanpercentile(img.astype(np.float64), [pl, ph])
        if hi <= lo:
            hi = lo + 1.0
        x = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
        if gamma != 1.0:
            x = np.power(x, 1.0 / float(gamma))
        return (x * 255.0 + 0.5).astype(np.uint8)

    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        s = str(hex_color).strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) != 6:
            raise ValueError("outline_color must be a hex string like '#FFA500'.")
        try:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
        except Exception:
            raise ValueError("outline_color must be a hex string like '#FFA500'.")
        return (r, g, b)

    def _decide_device() -> str:
        if device == "cuda":
            if not _CUPY_AVAILABLE:
                warnings.warn(
                    "device='cuda' requested but CuPy not available; falling back to CPU."
                )
                return "cpu"
            return "cuda"
        if device == "auto":
            return "cuda" if _CUPY_AVAILABLE else "cpu"
        return "cpu"

    def _to_dev(x: np.ndarray, dev: str):
        if dev == "cuda":
            return cp.asarray(x)
        return x

    def _to_host(x, dev: str) -> np.ndarray:
        if dev == "cuda":
            return cp.asnumpy(x)
        return x

    def _maybe_blur_alpha(alpha_2d, sigma: float, dev: str):
        if sigma <= 0:
            return alpha_2d
        if dev == "cuda":
            # Simple Gaussian via separable filtering could be done on GPU; for now move to CPU for blur and back.
            a = _to_host(alpha_2d, dev)
            if _CV2_AVAILABLE:
                ksz = max(3, int(2 * math.ceil(3.0 * sigma) + 1))
                a = cv2.GaussianBlur(
                    a,
                    (ksz, ksz),
                    sigmaX=sigma,
                    sigmaY=sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
            else:
                from scipy.ndimage import gaussian_filter  # fallback

                a = gaussian_filter(a, sigma=sigma, mode="nearest")
            return _to_dev(a, dev)
        else:
            if _CV2_AVAILABLE:
                ksz = max(3, int(2 * math.ceil(3.0 * sigma) + 1))
                return cv2.GaussianBlur(
                    alpha_2d,
                    (ksz, ksz),
                    sigmaX=sigma,
                    sigmaY=sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
            else:
                from scipy.ndimage import gaussian_filter

                return gaussian_filter(alpha_2d, sigma=sigma, mode="nearest")

    def _alpha_map(val_img, thr, sat, amax, agamma, dev: str):
        # val_img: per-pixel "activity" after gather
        eps = 1e-12
        denom = max(sat - thr, eps)
        if dev == "cuda":
            x = cp.clip((val_img - thr) / denom, 0, 1)
            if agamma != 1.0:
                x = cp.power(x, agamma)
            return x * amax
        else:
            x = np.clip((val_img - thr) / denom, 0, 1, dtype=np.float32)
            if agamma != 1.0:
                x = np.power(x, agamma, dtype=np.float32)
            return x * amax

    def _resample_to_frame_indices(
        t_sig: Optional[np.ndarray], n_t: int, frame_times: np.ndarray
    ) -> np.ndarray:
        """
        Map each desired frame time to a source time index (nearest neighbor).
        If t_sig is None, assume one sample per desired frame (modulo bounds).
        """
        if t_sig is None:
            # If no times, assume uniform sampling and index by scaling
            idx = np.linspace(0, n_t - 1, num=len(frame_times))
            return np.clip(np.rint(idx).astype(np.int64), 0, n_t - 1)

        t = np.asarray(t_sig, dtype=np.float64)
        if np.any(np.diff(t) <= 0):
            raise ValueError("Time array must be strictly increasing.")
        # nearest neighbor search
        idx = np.searchsorted(t, frame_times, side="left")
        idx = np.clip(idx, 0, len(t) - 1)
        # choose nearest between idx and idx-1
        left = np.clip(idx - 1, 0, len(t) - 1)
        choose_left = (idx > 0) & ((frame_times - t[left]) <= (t[idx] - frame_times))
        idx = np.where(choose_left, left, idx)
        return idx.astype(np.int64)

    def _build_frame_times():
        nonlocal fps, start_time, end_time
        # derive overlap window from provided times
        tmins, tmaxs = [], []
        if act_time is not None:
            tmins.append(float(act_time[0]))
            tmaxs.append(float(act_time[-1]))
        if soma_time is not None:
            tmins.append(float(soma_time[0]))
            tmaxs.append(float(soma_time[-1]))

        if (start_time is None) or (end_time is None):
            if tmins and tmaxs:
                st = max(tmins) if start_time is None else start_time
                et = min(tmaxs) if end_time is None else end_time
            else:
                # no times: derive from samples
                n_frames = act_data.shape[1]
                if fps is None:
                    if one_frame_per_sample_if_times_missing:
                        fps_local = 30.0  # placeholder; we actually ignore fps and do 1 frame per sample
                        times = np.arange(n_frames, dtype=np.float64)
                        return times, fps_local
                    else:
                        fps_local = 60.0
                        dur = n_frames / fps_local
                        times = np.arange(0, dur, 1.0 / fps_local, dtype=np.float64)
                        return times, fps_local
                else:
                    dur = act_data.shape[1] / float(fps)
                    times = np.arange(0, dur, 1.0 / float(fps), dtype=np.float64)
                    return times, float(fps)
            start_time, end_time = float(st), float(et)
        if start_time >= end_time:
            raise ValueError("start_time must be < end_time.")

        # decide fps
        if fps is None:
            if act_time is not None:
                diffs = np.diff(act_time)
                median_dt = np.median(diffs)
                fps = 1.0 / float(median_dt)
            elif soma_time is not None:
                diffs = np.diff(soma_time)
                median_dt = np.median(diffs)
                fps = 1.0 / float(median_dt)
            else:
                fps = 60.0

        n_frames = int(np.floor((end_time - start_time) * float(fps)))
        times = start_time + np.arange(n_frames, dtype=np.float64) / float(fps)
        return times, float(fps)

    # --------------------
    # Validate & prepare
    # --------------------
    if mean_im.ndim != 2:
        raise ValueError("mean_im must be (H, W).")
    H, W = mean_im.shape

    if source_map.shape != (H, W):
        raise ValueError("source_map must have same shape as mean_im.")

    if act_data.ndim != 2:
        raise ValueError("act_data must be (n_sources, n_timepoints).")
    n_sources = act_data.shape[0]

    if soma_map is not None:
        if soma_map.shape != (H, W):
            raise ValueError("soma_map must have same shape as mean_im.")
        if soma_activity is None:
            raise ValueError("soma_map provided but soma_activity is None.")
        if soma_activity.ndim != 2:
            raise ValueError("soma_activity must be (n_somas, n_timepoints).")
        n_somas = soma_activity.shape[0]
    else:
        n_somas = 0

    # Scale background to 8-bit
    bg_u8 = _percentile_scale_to_u8(mean_im, *background_percentiles, background_gamma)

    # Optional resize of background & maps for output size
    if output_size is not None and (output_size != (H, W)):
        if not _CV2_AVAILABLE:
            raise RuntimeError(
                "output_size requested but OpenCV not available for resizing."
            )
        H_out, W_out = output_size
        bg_u8 = cv2.resize(bg_u8, (W_out, H_out), interpolation=cv2.INTER_AREA)
        src_map_resized = cv2.resize(
            source_map.astype(np.int32), (W_out, H_out), interpolation=cv2.INTER_NEAREST
        )
        if soma_map is not None:
            soma_map_resized = cv2.resize(
                soma_map.astype(np.int32),
                (W_out, H_out),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            soma_map_resized = None
        H, W = H_out, W_out
    else:
        src_map_resized = source_map.astype(np.int32, copy=False)
        soma_map_resized = (
            soma_map.astype(np.int32, copy=False) if soma_map is not None else None
        )

    # Prepare label-index images with sentinel at the end for "background"
    # Sources: -1 -> n_sources sentinel; 0..n_sources-1 are direct
    if np.max(src_map_resized) >= n_sources:
        raise ValueError("source_map contains label >= n_sources in act_data.")
    src_idx_img = src_map_resized.copy()
    src_idx_img[src_idx_img < 0] = n_sources  # sentinel index

    # Somas: 0 -> n_somas sentinel; 1..n_somas -> (label-1)
    if n_somas > 0:
        if np.max(soma_map_resized) > n_somas:
            raise ValueError("soma_map contains label > n_somas in soma_activity.")
        soma_idx_img = soma_map_resized.copy()
        soma_idx_img = np.where(
            soma_idx_img == 0, n_somas, soma_idx_img - 1
        )  # 0->sentinel, 1.. -> 0..
    else:
        soma_idx_img = None

    # Precompute soma outlines and dashed pattern (CPU) for event overlays
    if n_somas > 0:
        soma_label_img_cpu = soma_idx_img  # (H, W) int32 labels, background = n_somas
        bg_label = n_somas
        lbl = soma_label_img_cpu
        # 4-neighborhood boundary of labeled regions (only mark label side, not background)
        outline_all_cpu = np.zeros((H, W), dtype=bool)
        outline_all_cpu[1:, :] |= (lbl[1:, :] != lbl[:-1, :]) & (lbl[1:, :] != bg_label)
        outline_all_cpu[:-1, :] |= (lbl[:-1, :] != lbl[1:, :]) & (
            lbl[:-1, :] != bg_label
        )
        outline_all_cpu[:, 1:] |= (lbl[:, 1:] != lbl[:, :-1]) & (lbl[:, 1:] != bg_label)
        outline_all_cpu[:, :-1] |= (lbl[:, :-1] != lbl[:, 1:]) & (
            lbl[:, :-1] != bg_label
        )
        # Thicken outline via cross-shaped dilation to achieve desired width
        width_iter = max(1, int(outline_width))
        if width_iter > 1:
            if _CV2_AVAILABLE:
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                outline_thick_cpu = (
                    cv2.dilate(
                        outline_all_cpu.astype(np.uint8),
                        kernel,
                        iterations=width_iter - 1,
                    )
                    > 0
                )
            else:

                def _dilate_cross(mask_bool: np.ndarray, iters: int) -> np.ndarray:
                    out = mask_bool.copy()
                    for _ in range(iters):
                        nb = out.copy()
                        nb[:-1, :] |= out[1:, :]
                        nb[1:, :] |= out[:-1, :]
                        nb[:, :-1] |= out[:, 1:]
                        nb[:, 1:] |= out[:, :-1]
                        out = nb
                    return out

                outline_thick_cpu = _dilate_cross(outline_all_cpu, width_iter - 1)
        else:
            outline_thick_cpu = outline_all_cpu
    else:
        soma_label_img_cpu = None
        outline_all_cpu = None
        outline_thick_cpu = None

    # Decide device for overlay math
    dev = _decide_device()

    # Frame dims (pad if needed for yuv420p)
    H_eff, W_eff = H, W
    if pad_to_even and pix_fmt == "yuv420p":
        if H_eff % 2 != 0:
            H_eff += 1
        if W_eff % 2 != 0:
            W_eff += 1

    # Build frame times
    frame_times, fps = _build_frame_times()
    n_frames = len(frame_times)

    # Precompute per-frame indices (nearest) into activity time axes
    src_time_idx = _resample_to_frame_indices(act_time, act_data.shape[1], frame_times)
    if n_somas > 0:
        soma_time_idx = _resample_to_frame_indices(
            soma_time, soma_activity.shape[1], frame_times
        )
    else:
        soma_time_idx = None

    # Map soma_event_times to per-soma active frames for outlines
    event_active = None
    if (n_somas > 0) and (soma_event_times is not None):
        event_active = np.zeros((n_somas, n_frames), dtype=bool)

        def _times_to_frames(times_1d):
            if times_1d is None:
                return np.empty((0,), dtype=np.int64)
            t = np.asarray(times_1d, dtype=np.float64).ravel()
            if t.size == 0:
                return np.empty((0,), dtype=np.int64)
            idx = np.searchsorted(frame_times, t, side="left")
            idx = np.clip(idx, 0, n_frames - 1)
            left = np.clip(idx - 1, 0, n_frames - 1)
            choose_left = (idx > 0) & (
                (t - frame_times[left]) <= (frame_times[idx] - t)
            )
            idx = np.where(choose_left, left, idx)
            return np.unique(idx.astype(np.int64))

        if isinstance(soma_event_times, (list, tuple)):
            if len(soma_event_times) == n_somas:
                for si, times in enumerate(soma_event_times):
                    frames_si = _times_to_frames(times)
                    for fe in frames_si:
                        fe_end = min(n_frames, fe + int(event_outline_frames))
                        event_active[si, fe:fe_end] = True
            else:
                merged = []
                for x in soma_event_times:
                    if x is None:
                        continue
                    merged.append(np.asarray(x).ravel())
                merged = (
                    np.concatenate(merged)
                    if len(merged) > 0
                    else np.array([], dtype=np.float64)
                )
                frames_all = _times_to_frames(merged)
                for fe in frames_all:
                    fe_end = min(n_frames, fe + int(event_outline_frames))
                    event_active[:, fe:fe_end] = True
        else:
            frames_all = _times_to_frames(soma_event_times)
            for fe in frames_all:
                fe_end = min(n_frames, fe + int(event_outline_frames))
                event_active[:, fe:fe_end] = True

    # Move static arrays to device
    bg_u8_dev = _to_dev(bg_u8, dev)
    src_idx_img_dev = _to_dev(src_idx_img, dev)
    if soma_idx_img is not None:
        soma_idx_img_dev = _to_dev(soma_idx_img, dev)
    else:
        soma_idx_img_dev = None

    # Preallocate buffers
    if preallocate_float_frames:
        if dev == "cuda":
            frame_f = cp.zeros((H, W, 3), dtype=cp.float32)
        else:
            frame_f = np.zeros((H, W, 3), dtype=np.float32)
    else:
        frame_f = None

    # --------------------
    # Encoder setup
    # --------------------
    container = av.open(out_path, mode="w")
    stream = container.add_stream(codec, rate=int(round(fps)))
    stream.width = W_eff
    stream.height = H_eff
    stream.pix_fmt = pix_fmt

    # x264 / NVENC options
    opts = {} if video_options is None else dict(video_options)
    if codec in ("h264", "libx264"):
        if crf is not None:
            opts["crf"] = str(int(crf))
        if tune is not None:
            opts["tune"] = tune
        if preset is not None:
            opts["preset"] = preset
    elif codec == "h264_nvenc":
        # Common, good defaults; you can tweak further via video_options
        if nvenc_cq is not None:
            opts["cq"] = str(int(nvenc_cq))  # constant quality
        if preset is not None:
            # Newer FFmpeg maps "p7..p1"; older maps "slow..fast". Pass through as-is.
            opts["preset"] = preset
        # Enable lookahead/2-pass if you like: opts["rc"] = "vbr"; opts["2pass"] = "1"
    if gop is not None:
        opts["g"] = str(int(gop))

    # Apply options
    if hasattr(stream, "codec_context") and hasattr(stream.codec_context, "options"):
        stream.codec_context.options = opts

    # --------------------
    # Main loop
    # --------------------
    # Static colors
    if dev == "cuda":
        syn_color_vec = cp.asarray(np.array(syn_color, dtype=np.float32))
        soma_color_vec = cp.asarray(np.array(soma_color, dtype=np.float32))
        event_color_vec = cp.asarray(
            np.array(_hex_to_rgb(outline_color), dtype=np.float32)
        )
    else:
        syn_color_vec = np.array(syn_color, dtype=np.float32)
        soma_color_vec = np.array(soma_color, dtype=np.float32)
        event_color_vec = np.array(_hex_to_rgb(outline_color), dtype=np.float32)

    # Build a 3-channel background in float32
    if dev == "cuda":
        bg_rgb_f = cp.stack([bg_u8_dev, bg_u8_dev, bg_u8_dev], axis=-1).astype(
            cp.float32
        )
    else:
        bg_rgb_f = np.stack([bg_u8_dev, bg_u8_dev, bg_u8_dev], axis=-1).astype(
            np.float32
        )

    # Preallocate sentinel-extended vectors once; we will fill in-place every frame
    if dev == "cuda":
        src_vals = cp.zeros((n_sources + 1,), dtype=cp.float32)  # last is sentinel=0
        if n_somas > 0:
            soma_vals = cp.zeros((n_somas + 1,), dtype=cp.float32)
    else:
        src_vals = np.zeros((n_sources + 1,), dtype=np.float32)
        if n_somas > 0:
            soma_vals = np.zeros((n_somas + 1,), dtype=np.float32)

    for f in range(n_frames):
        # Start fresh frame
        if preallocate_float_frames:
            frame = frame_f
            frame[...] = bg_rgb_f
        else:
            # Allocate (slightly slower)
            frame = bg_rgb_f.copy()

        # ----- SYNAPSES -----
        # Fill per-frame source intensities (sentinel remains 0)
        if dev == "cuda":
            src_vals[:-1] = cp.asarray(act_data[:, src_time_idx[f]], dtype=cp.float32)
            syn_img = src_vals[src_idx_img_dev]  # gather per-pixel values
        else:
            src_vals[:-1] = act_data[:, src_time_idx[f]].astype(np.float32, copy=False)
            syn_img = src_vals[src_idx_img_dev]

        # Threshold → alpha
        syn_alpha = _alpha_map(
            syn_img, syn_threshold, syn_saturate_at, syn_alpha_max, syn_alpha_gamma, dev
        )
        if syn_alpha_blur_sigma > 0:
            syn_alpha = _maybe_blur_alpha(syn_alpha, syn_alpha_blur_sigma, dev)

        # Blend (alpha-over). If alpha_only, keep color at full; else modulate color brightness ~ alpha.
        if dev == "cuda":
            a = syn_alpha[..., None]  # (H, W, 1)
            color_vec = syn_color_vec[None, None, :]

            if syn_alpha_only:
                frame[:] = color_vec * a + frame * (1.0 - a)
            else:
                # also scale color brightness by normalized intensity (use alpha/alpha_max as a proxy)
                # prevents overly neon look at low alpha
                alpha_norm = cp.where(
                    syn_alpha_max > 0, syn_alpha / syn_alpha_max, 0.0
                )[..., None]
                frame[:] = (color_vec * alpha_norm) * a + frame * (1.0 - a)
        else:
            a = syn_alpha[..., None]
            color_vec = syn_color_vec[None, None, :]

            if syn_alpha_only:
                frame[:] = color_vec * a + frame * (1.0 - a)
            else:
                alpha_norm = (
                    syn_alpha / (syn_alpha_max if syn_alpha_max > 0 else 1.0)
                )[..., None]
                frame[:] = (color_vec * alpha_norm) * a + frame * (1.0 - a)

        # ----- SOMAS -----
        if n_somas > 0:
            if dev == "cuda":
                soma_vals[:-1] = cp.asarray(
                    soma_activity[:, soma_time_idx[f]], dtype=cp.float32
                )
                soma_img = soma_vals[soma_idx_img_dev]
            else:
                soma_vals[:-1] = soma_activity[:, soma_time_idx[f]].astype(
                    np.float32, copy=False
                )
                soma_img = soma_vals[soma_idx_img_dev]

            soma_alpha = _alpha_map(
                soma_img,
                soma_threshold,
                soma_saturate_at,
                soma_alpha_max,
                soma_alpha_gamma,
                dev,
            )
            if soma_alpha_blur_sigma > 0:
                soma_alpha = _maybe_blur_alpha(soma_alpha, soma_alpha_blur_sigma, dev)

            if dev == "cuda":
                a = soma_alpha[..., None]
                color_vec = soma_color_vec[None, None, :]
                if soma_alpha_only:
                    frame[:] = color_vec * a + frame * (1.0 - a)
                else:
                    alpha_norm = cp.where(
                        soma_alpha_max > 0, soma_alpha / soma_alpha_max, 0.0
                    )[..., None]
                    frame[:] = (color_vec * alpha_norm) * a + frame * (1.0 - a)
            else:
                a = soma_alpha[..., None]
                color_vec = soma_color_vec[None, None, :]
                if soma_alpha_only:
                    frame[:] = color_vec * a + frame * (1.0 - a)
                else:
                    alpha_norm = (
                        soma_alpha / (soma_alpha_max if soma_alpha_max > 0 else 1.0)
                    )[..., None]
                    frame[:] = (color_vec * alpha_norm) * a + frame * (1.0 - a)

        # ----- EVENT OUTLINES -----
        if n_somas > 0 and event_active is not None:
            active = event_active[:, f]
            if np.any(active):
                # Build per-pixel active-label mask via label indexing
                active_ext = np.zeros((n_somas + 1,), dtype=bool)
                active_ext[:n_somas] = active
                active_per_pixel = active_ext[soma_label_img_cpu]
                active_mask_cpu = outline_thick_cpu & active_per_pixel
                if dev == "cuda":
                    m = cp.asarray(active_mask_cpu)
                    if bool(m.any()):
                        frame[:] = cp.where(
                            m[..., None], event_color_vec[None, None, :], frame
                        )
                else:
                    if active_mask_cpu.any():
                        frame[active_mask_cpu] = event_color_vec

        # Convert to uint8 on host
        frame_u8 = _to_host(
            (
                np.clip(frame, 0, 255).astype(np.uint8)
                if dev == "cpu"
                else cp.clip(frame, 0, 255).astype(cp.uint8)
            ),
            dev,
        )

        # Pad to even dims if needed
        if (H_eff != H) or (W_eff != W):
            # pad at bottom/right with zeros (black)
            if frame_u8.shape[0] != H_eff or frame_u8.shape[1] != W_eff:
                padded = np.zeros((H_eff, W_eff, 3), dtype=np.uint8)
                padded[:H, :W, :] = frame_u8
                frame_u8 = padded

        # Encode
        video_frame = av.VideoFrame.from_ndarray(frame_u8, format="rgb24")
        if (
            (video_frame.width != W_eff)
            or (video_frame.height != H_eff)
            or (video_frame.format.name != pix_fmt)
        ):
            video_frame = video_frame.reformat(
                width=W_eff, height=H_eff, format=pix_fmt
            )

        for packet in stream.encode(video_frame):
            container.mux(packet)

        if progress and ((f + 1) % max(1, progress_every) == 0 or (f + 1) == n_frames):
            print(f"[make_activity_movie] encoded {f+1}/{n_frames} frames")

    # Flush and close
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def generate_activity_colorbar_image(
    out_path: Optional[str] = None,
    # Syn mapping (matches make_activity_movie behavior)
    syn_color: Union[str, Tuple[int, int, int]] = "green",
    syn_threshold: float = 0.15,
    syn_saturate_at: float = 1.0,
    syn_alpha_max: float = 0.85,
    syn_alpha_gamma: float = 1.0,
    syn_alpha_only: bool = True,
    # Soma mapping
    soma_color: Union[str, Tuple[int, int, int]] = "red",
    soma_threshold: float = 0.15,
    soma_saturate_at: float = 0.8,
    soma_alpha_max: float = 0.95,
    soma_alpha_gamma: float = 1.0,
    soma_alpha_only: bool = True,
    # Appearance
    width: int = 900,
    height: int = 260,
    margins: Tuple[int, int, int, int] = (90, 40, 50, 70),  # (left, right, top, bottom)
    bar_height_px: int = 26,
    row_gap_px: int = 36,
    font_scale: float = 0.6,
    font_thickness: int = 1,
    bg_color: Union[str, Tuple[int, int, int]] = "black",
    axis_color: Union[str, Tuple[int, int, int]] = "white",
    # Scale (shared across both rows)
    value_min: Optional[float] = None,
    value_max: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a two-row colorbar image showing how synaptic (green by default) and
    somatic (red by default) activity values map to color and alpha, matching
    the exact blending logic used in `make_activity_movie`.

    - Each row is a horizontal bar where x encodes activity value from `value_min`
      to `value_max` (shared across the two rows).
    - The visible color is computed using the same `alpha_only` behavior and alpha
      mapping: alpha = clip((val-thr)/(sat-thr), 0..1)**gamma * alpha_max.
      If alpha_only=False, brightness is additionally scaled (as in the movie).
    - Threshold and saturation positions are marked with small vertical ticks
      and labeled.

    Returns the generated BGR image (np.ndarray). If `out_path` is provided,
    writes the image (PNG recommended) to that path as well.
    """

    # ------------- helpers -------------
    def _hex_or_name_to_bgr(c):
        if isinstance(c, tuple) and len(c) == 3:
            r, g, b = c
            return (int(b), int(g), int(r))
        if isinstance(c, str):
            s = c.strip().lower()
            named = {
                "black": (0, 0, 0),
                "white": (255, 255, 255),
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "cyan": (0, 255, 255),
                "magenta": (255, 0, 255),
                "yellow": (255, 255, 0),
                "orange": (255, 165, 0),
                "purple": (128, 0, 128),
                "gray": (128, 128, 128),
                "grey": (128, 128, 128),
            }
            if s in named:
                r, g, b = named[s]
                return (b, g, r)
            if s.startswith("#") and len(s) == 7:
                r = int(s[1:3], 16)
                g = int(s[3:5], 16)
                b = int(s[5:7], 16)
                return (b, g, r)
        raise ValueError(f"Unrecognized color: {c}")

    def _put_text(img, txt, org, color):
        cv2.putText(
            img,
            txt,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )

    def _text_w(txt):
        (w, _), _ = cv2.getTextSize(
            txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        return w

    def _alpha_map(
        vals: np.ndarray, thr: float, sat: float, amax: float, agamma: float
    ):
        eps = 1e-12
        denom = max(sat - thr, eps)
        x = np.clip((vals - thr) / denom, 0.0, 1.0)
        if agamma != 1.0:
            x = np.power(x, agamma, dtype=np.float32)
        return (x * float(max(amax, 0.0))).astype(np.float32)

    def _compose_row(
        vals: np.ndarray,
        color_bgr: Tuple[int, int, int],
        thr: float,
        sat: float,
        amax: float,
        agamma: float,
        alpha_only: bool,
        bg_bgr: Tuple[int, int, int],
    ) -> np.ndarray:
        # Compute per-pixel alpha
        a = _alpha_map(vals, thr, sat, amax, agamma)  # (W,)
        a = a[:, None]  # (W,1)
        color = np.array(color_bgr, dtype=np.float32)[None, :]  # (1,3)
        bg = np.array(bg_bgr, dtype=np.float32)[None, :]  # (1,3)
        if alpha_only:
            # alpha-over blend of solid color on bg
            row = color * a + bg * (1.0 - a)
        else:
            # reduce color brightness at low alpha (match movie)
            alpha_norm = (a / float(amax)) if amax > 0 else np.zeros_like(a)
            row = (color * alpha_norm) * a + bg * (1.0 - a)
        return np.clip(row, 0, 255).astype(np.uint8)

    # ------------- validate & layout -------------
    img = np.zeros((height, width, 3), dtype=np.uint8)
    bg_bgr = _hex_or_name_to_bgr(bg_color)
    axis_bgr = _hex_or_name_to_bgr(axis_color)
    img[:] = bg_bgr

    left, right, top, bottom = margins
    plot_w = max(20, width - left - right)
    plot_h_all = max(20, height - top - bottom)
    # two rows stacked; compute their y positions
    bar_h = max(6, min(bar_height_px, (plot_h_all - row_gap_px) // 2))
    top_bar_y = top + (plot_h_all - (2 * bar_h + row_gap_px)) // 2
    bottom_bar_y = top_bar_y + bar_h + row_gap_px
    bar_x = left

    # shared horizontal value scale
    if value_min is None:
        value_min = 0.0
    if value_max is None:
        value_max = max(float(syn_saturate_at), float(soma_saturate_at))
    if (
        not np.isfinite(value_min)
        or not np.isfinite(value_max)
        or value_max <= value_min
    ):
        value_min, value_max = 0.0, 1.0

    xp = np.arange(plot_w, dtype=np.float32)
    vals = value_min + (xp / max(1.0, (plot_w - 1))) * (value_max - value_min)  # (W,)

    syn_bgr = _hex_or_name_to_bgr(syn_color)
    soma_bgr = _hex_or_name_to_bgr(soma_color)

    # ------------- build rows -------------
    syn_row = _compose_row(
        vals,
        syn_bgr,
        syn_threshold,
        syn_saturate_at,
        syn_alpha_max,
        syn_alpha_gamma,
        syn_alpha_only,
        bg_bgr,
    )  # (W,3)
    soma_row = _compose_row(
        vals,
        soma_bgr,
        soma_threshold,
        soma_saturate_at,
        soma_alpha_max,
        soma_alpha_gamma,
        soma_alpha_only,
        bg_bgr,
    )  # (W,3)

    # paint bars into image
    img[top_bar_y : top_bar_y + bar_h, bar_x : bar_x + plot_w, :] = syn_row[None, :, :]
    img[bottom_bar_y : bottom_bar_y + bar_h, bar_x : bar_x + plot_w, :] = soma_row[
        None, :, :
    ]

    # ------------- decorations: labels and ticks -------------
    # Titles
    _put_text(img, "Synaptic", (bar_x, max(12, top_bar_y - 8)), syn_bgr)
    _put_text(
        img, "Soma", (bar_x, min(height - 6, bottom_bar_y + bar_h + 22)), soma_bgr
    )

    # Axis labels at ends (shared scale)
    v0 = f"{value_min:.3g}"
    v1 = f"{value_max:.3g}"
    _put_text(img, v0, (bar_x, bottom_bar_y + bar_h + 22), axis_bgr)
    _put_text(
        img, v1, (bar_x + plot_w - _text_w(v1), bottom_bar_y + bar_h + 22), axis_bgr
    )

    # Tick helper
    def _val_to_x(v: float) -> int:
        frac = (float(v) - value_min) / max(1e-12, (value_max - value_min))
        return int(round(bar_x + np.clip(frac, 0.0, 1.0) * (plot_w - 1)))

    # Draw ticks for syn threshold/saturate on top row
    x_syn_thr = _val_to_x(syn_threshold)
    x_syn_sat = _val_to_x(syn_saturate_at)
    cv2.line(
        img,
        (x_syn_thr, top_bar_y - 6),
        (x_syn_thr, top_bar_y + bar_h + 6),
        syn_bgr,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        img,
        (x_syn_sat, top_bar_y - 6),
        (x_syn_sat, top_bar_y + bar_h + 6),
        syn_bgr,
        1,
        cv2.LINE_AA,
    )
    _put_text(
        img,
        f"thr {syn_threshold:.3g}",
        (min(width - 10, x_syn_thr + 6), max(12, top_bar_y - 8)),
        syn_bgr,
    )
    _put_text(
        img,
        f"sat {syn_saturate_at:.3g}",
        (min(width - 10, x_syn_sat + 6), max(12, top_bar_y - 8)),
        syn_bgr,
    )

    # Draw ticks for soma threshold/saturate on bottom row
    x_soma_thr = _val_to_x(soma_threshold)
    x_soma_sat = _val_to_x(soma_saturate_at)
    cv2.line(
        img,
        (x_soma_thr, bottom_bar_y - 6),
        (x_soma_thr, bottom_bar_y + bar_h + 6),
        soma_bgr,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        img,
        (x_soma_sat, bottom_bar_y - 6),
        (x_soma_sat, bottom_bar_y + bar_h + 6),
        soma_bgr,
        1,
        cv2.LINE_AA,
    )
    _put_text(
        img,
        f"thr {soma_threshold:.3g}",
        (min(width - 10, x_soma_thr + 6), min(height - 6, bottom_bar_y + bar_h + 22)),
        soma_bgr,
    )
    _put_text(
        img,
        f"sat {soma_saturate_at:.3g}",
        (min(width - 10, x_soma_sat + 6), min(height - 6, bottom_bar_y + bar_h + 22)),
        soma_bgr,
    )

    # save if requested
    if out_path is not None and len(str(out_path)) > 0:
        cv2.imwrite(out_path, img)

    return img


def animate_state_bar_video(
    out_path: str,
    time: np.ndarray,  # (T,), seconds; strictly increasing
    state_labels: List[str],  # len T strings labeling each sample
    state_colors: Dict[str, Union[str, Tuple[int, int, int]]],
    fps: float = 60.0,
    # Appearance
    width: int = 1280,
    height: int = 200,
    margins: Tuple[int, int, int, int] = (50, 50, 50, 50),  # (left, right, top, bottom)
    bar_height_px: int = 18,  # thickness of the colored state bar
    text_gap_px: int = 6,  # vertical gap between text and bar (text above bar)
    font_scale: float = 0.7,
    font_thickness: int = 2,
    bg_color: Union[str, Tuple[int, int, int]] = "black",
    state_bar_alpha: float = 1.0,  # 0..1
    # Encoding
    codec: str = "h264",
    crf: Optional[int] = 18,
    preset: str = "veryfast",
    pix_fmt: str = "yuv420p",
    pad_to_even: bool = True,
    progress: bool = True,
):
    """
    Create an MP4 that displays ONLY a colored state bar and its label text above it.
    The bar fills left→right over time until the full duration is revealed.

    Inputs:
      - time: 1D array of times in seconds, strictly increasing (len T).
      - state_labels: list/array of strings of length T; one label per sample in `time`.
      - state_colors: dict mapping label -> color (name | "#RRGGBB" | (R,G,B)).

    Customization:
      - font_scale, font_thickness: control text size/weight.
      - bar_height_px: thickness of the state bar.
      - width, height, margins: overall frame size and inner margins.
      - state_bar_alpha: 0..1; if <1, bar is pre-blended with background color.
    """

    # --------- helpers ---------
    def _hex_or_name_to_bgr(c):
        if isinstance(c, tuple) and len(c) == 3:
            r, g, b = c
            return (int(b), int(g), int(r))
        if isinstance(c, str):
            s = c.strip().lower()
            named = {
                "black": (0, 0, 0),
                "white": (255, 255, 255),
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "cyan": (0, 255, 255),
                "magenta": (255, 0, 255),
                "yellow": (255, 255, 0),
                "orange": (255, 165, 0),
                "purple": (128, 0, 128),
                "gray": (128, 128, 128),
                "grey": (128, 128, 128),
            }
            if s in named:
                r, g, b = named[s]
                return (b, g, r)
            if s.startswith("#") and len(s) == 7:
                r = int(s[1:3], 16)
                g = int(s[3:5], 16)
                b = int(s[5:7], 16)
                return (b, g, r)
        raise ValueError(f"Unrecognized color: {c}")

    def _put_text(img, txt, org, color):
        cv2.putText(
            img,
            txt,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )

    def _text_w(txt):
        (w, _), _ = cv2.getTextSize(
            txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        return w

    # --------- validate ---------
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("`time` must be 1D and length >= 2.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("`time` must be strictly increasing (seconds).")
    T = int(time.size)

    if not isinstance(state_labels, (list, tuple, np.ndarray)):
        raise ValueError("`state_labels` must be a list/tuple/array of strings.")
    if len(state_labels) != T:
        raise ValueError("`state_labels` must have the same length as `time`.")
    state_labels = [str(s) for s in state_labels]
    if not isinstance(state_colors, dict):
        raise ValueError("`state_colors` must be a dict mapping state -> color.")

    # map colors
    bg_bgr = _hex_or_name_to_bgr(bg_color)
    state_to_bgr = {k: _hex_or_name_to_bgr(v) for k, v in state_colors.items()}

    # --------- layout ---------
    left, right, top, bottom = margins
    plot_w = max(10, width - left - right)
    plot_h_all = max(10, height - top - bottom)
    bar_h = max(1, min(bar_height_px, plot_h_all - 10))
    # center bar vertically within plotting area
    bar_y = top + (plot_h_all - bar_h) // 2
    bar_x = left

    # time range and per-column time mapping
    t_start = float(time[0])
    t_end = float(time[-1])
    total_dur = t_end - t_start
    if total_dur <= 0:
        raise ValueError("Total duration must be > 0.")

    xp = np.arange(plot_w, dtype=np.int32)  # 0..plot_w-1
    t_at_x = t_start + (xp / max(1, (plot_w - 1))) * total_dur

    # Build full-width bar image row of colors (then tile vertically)
    idx = np.searchsorted(time, t_at_x, side="left")
    idx = np.clip(idx, 0, T - 1)
    row = np.zeros((plot_w, 3), dtype=np.uint8)
    for col in range(plot_w):
        st = state_labels[idx[col]]
        row[col] = state_to_bgr.get(st, (90, 90, 90))
    bar = np.zeros((bar_h, plot_w, 3), dtype=np.uint8)
    bar[:] = row[None, :, :]

    if state_bar_alpha < 1.0:
        bg_row = np.full_like(bar, fill_value=np.array(bg_bgr, dtype=np.uint8))
        bar = (
            bar.astype(np.float32) * state_bar_alpha
            + bg_row.astype(np.float32) * (1.0 - state_bar_alpha)
        ).astype(np.uint8)

    # --------- encoding setup ---------
    H_eff, W_eff = height, width
    if pad_to_even and pix_fmt == "yuv420p":
        if H_eff % 2:
            H_eff += 1
        if W_eff % 2:
            W_eff += 1

    container = av.open(out_path, mode="w")
    stream = container.add_stream(codec, rate=int(round(fps)))
    stream.width = W_eff
    stream.height = H_eff
    stream.pix_fmt = pix_fmt
    opts = {}
    if codec in ("h264", "libx264"):
        if crf is not None:
            opts["crf"] = str(int(crf))
        if preset:
            opts["preset"] = preset
    elif codec == "h264_nvenc":
        # If using NVENC and you want CQ, map via video options, but keep simple here
        pass
    if hasattr(stream, "codec_context") and hasattr(stream.codec_context, "options"):
        stream.codec_context.options = opts

    # --------- frame loop ---------
    frames_total = max(1, int(math.ceil(total_dur * fps)))
    base_bg = np.zeros((height, width, 3), dtype=np.uint8)
    base_bg[:] = bg_bgr

    for fi in range(frames_total):
        frac = min(1.0, fi / max(1, frames_total - 1))
        x_cut = int(round(frac * (plot_w - 1))) + 1  # at least 1 px

        frame = base_bg.copy()
        # paste revealed portion of the bar
        frame[bar_y : bar_y + bar_h, bar_x : bar_x + x_cut, :] = bar[:, :x_cut, :]

        # centered state label text above the bar (updates with current state)
        cur_t = t_at_x[min(max(0, x_cut - 1), len(t_at_x) - 1)]
        cur_idx = int(np.clip(np.searchsorted(time, cur_t, side="left"), 0, T - 1))
        cur_state = state_labels[cur_idx]
        text_color = state_to_bgr.get(cur_state, (90, 90, 90))

        cx = bar_x + plot_w // 2
        tx_w = _text_w(cur_state)
        tx_x = cx - tx_w // 2
        tx_y = max(12, bar_y - max(1, text_gap_px))
        _put_text(frame, cur_state, (tx_x, tx_y), text_color)

        # pad if needed
        if (H_eff != height) or (W_eff != width):
            padded = np.zeros((H_eff, W_eff, 3), dtype=np.uint8)
            padded[:height, :width, :] = frame
            frame = padded

        # encode (feed as BGR)
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        if (
            (video_frame.width != W_eff)
            or (video_frame.height != H_eff)
            or (video_frame.format.name != pix_fmt)
        ):
            video_frame = video_frame.reformat(
                width=W_eff, height=H_eff, format=pix_fmt
            )
        for packet in stream.encode(video_frame):
            container.mux(packet)

        if progress and (fi + 1) % max(1, int(fps)) == 0:
            print(f"[animate_state_bar_video] encoded {fi+1}/{frames_total} frames ...")

    # flush & close
    for packet in stream.encode():
        container.mux(packet)
    container.close()


from typing import List, Optional, Union, Tuple, Dict, Literal
import numpy as np
import av  # PyAV
import cv2
import math


def animate_traces_video(
    out_path: str,
    traces: Union[np.ndarray, List[np.ndarray]],  # (T,), or list of (T,) or (N,T)
    time: np.ndarray,  # (T,), seconds; strictly increasing
    fps: float = 60.0,
    time_span: float = 10.0,  # seconds per loop/segment
    layout: Literal["overlay", "stacked"] = "overlay",
    shared_ylim: bool = True,  # share y limits across traces (recommended)
    ylim: Optional[Tuple[float, float]] = None,  # override if desired
    ylims: Optional[
        List[Tuple[float, float]]
    ] = None,  # per-trace limits for stacked layout
    # Colors & styling
    trace_colors: Union[
        str, Tuple[int, int, int], List[Union[str, Tuple[int, int, int]]]
    ] = "cyan",
    line_thickness: int = 2,
    aa_lines: bool = True,  # antialias lines
    bg_color: Union[str, Tuple[int, int, int]] = "black",
    axis_color: Union[str, Tuple[int, int, int]] = "white",
    grid: bool = False,
    # Labels
    trace_labels: Optional[Union[str, List[str]]] = None,
    # State bar
    states: Optional[
        List[str]
    ] = None,  # len T strings labeling each sample (e.g. "Wake","NREM")
    state_colors: Optional[Dict[str, Union[str, Tuple[int, int, int]]]] = None,
    state_bar_height_px: int = 10,  # 0 to disable drawing the bar
    state_bar_alpha: float = 1.0,  # 0..1
    # Canvas & axes
    width: int = 1280,
    height: int = 720,
    margins: Tuple[int, int, int, int] = (90, 40, 60, 70),  # (left, right, top, bottom)
    font_scale: float = 0.5,
    font_thickness: int = 1,
    # Encoding
    codec: str = "h264",  # "h264" (libx264) or "h264_nvenc"
    crf: Optional[int] = 18,  # libx264
    preset: str = "veryfast",  # libx264 preset
    nvenc_cq: Optional[int] = None,  # e.g. 19, if using h264_nvenc
    pix_fmt: str = "yuv420p",
    pad_to_even: bool = True,
    progress: bool = True,
):
    """
    Create an MP4 that animates one or more traces being drawn in time_span-wide windows.

    - The timeline is split into consecutive segments of length `time_span` seconds.
    - For each segment, the line(s) are drawn from left→right over frames at `fps`, then the
      animation jumps to the next segment, until the data ends.
    - If `states` and `state_colors` are provided, a thin bar above the axis fills left→right
      with the state's color at each timepoint (minimalistic indicator).

    Performance:
      • Per segment we resample to one value per x-pixel column (fast).
      • Per frame we only draw the first k points (no re-sampling).
      • We draw with OpenCV on a prebuilt segment background and stream frames to FFmpeg.

    Notes:
      • All traces are aligned on the same x mapping. In "stacked" layout, each trace gets its
        own y panel but identical x range, so axes line up perfectly.
      • Y-limits:
          - `ylim=(ymin, ymax)` applies a single shared y-limit to all traces (recommended for overlay).
          - `ylims=[(ymin_i, ymax_i) for i in traces]` allows per-trace limits, only when
            `layout='stacked'` and `shared_ylim=False`. If provided in other modes, a ValueError is raised.
    """

    # ------------- helpers -------------
    def _is_iterable_traces(x):
        if isinstance(x, np.ndarray) and x.ndim == 2:
            return True
        return isinstance(x, (list, tuple))

    def _as_traces_array(traces, T):
        """Return (N, T) float32 array, and number of traces."""
        if isinstance(traces, np.ndarray):
            arr = np.asarray(traces)
            if arr.ndim == 1:
                arr = arr[None, :]
            elif arr.ndim != 2:
                raise ValueError("`traces` must be (T,), (N,T), or list of (T,).")
        else:
            arr = np.stack([np.asarray(t) for t in traces], axis=0)
        if arr.shape[1] != T:
            raise ValueError("All traces must have same length T as `time`.")
        return arr.astype(np.float32, copy=False), arr.shape[0]

    def _hex_or_name_to_bgr(c):
        if isinstance(c, tuple) and len(c) == 3:
            # assume RGB, convert to BGR
            r, g, b = c
            return (int(b), int(g), int(r))
        if isinstance(c, str):
            s = c.strip().lower()
            # quick named colors (RGB)
            named = {
                "black": (0, 0, 0),
                "white": (255, 255, 255),
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "cyan": (0, 255, 255),
                "magenta": (255, 0, 255),
                "yellow": (255, 255, 0),
                "orange": (255, 165, 0),
                "purple": (128, 0, 128),
                "gray": (128, 128, 128),
                "grey": (128, 128, 128),
            }
            if s in named:
                r, g, b = named[s]
                return (b, g, r)
            # hex #RRGGBB
            if s.startswith("#") and len(s) == 7:
                r = int(s[1:3], 16)
                g = int(s[3:5], 16)
                b = int(s[5:7], 16)
                return (b, g, r)
        raise ValueError(f"Unrecognized color: {c}")

    def _ensure_color_list(colors, N):
        if isinstance(colors, (str, tuple)):
            return [_hex_or_name_to_bgr(colors)] * N
        clist = [_hex_or_name_to_bgr(c) for c in colors]
        if len(clist) != N:
            raise ValueError("Length of trace_colors must equal number of traces.")
        return clist

    def _normalize_trace_labels(labels, N):
        """Return list[str] of length N or None."""
        if labels is None:
            return None
        if isinstance(labels, str):
            if N != 1:
                raise ValueError(
                    "`trace_labels` should be a list of strings when multiple traces are provided."
                )
            return [labels]
        if isinstance(labels, (list, tuple)):
            if len(labels) != N:
                raise ValueError(
                    "Length of `trace_labels` must equal number of traces."
                )
            return [str(x) for x in labels]
        raise ValueError("`trace_labels` must be None, a str, or a list/tuple of str.")

    def _draw_axes(img, rect, x0, x1, y0, y1, show_x_labels: bool = False):
        """Minimal white axes/ticks on black bg for given rect (x,y,w,h)."""
        (x, y, w, h) = rect
        # axis lines: draw only left and bottom spines
        cv2.line(img, (x, y + h), (x + w, y + h), axis_bgr, 1)  # bottom
        cv2.line(img, (x, y), (x, y + h), axis_bgr, 1)  # left
        # ticks: y min/max, x start/end
        # positions: left/bottom margins
        # y labels at left
        # y_min_s = f"{y0:.2g}"
        # y_max_s = f"{y1:.2g}"
        # _put_text(img, y_min_s, (x - 10 - _text_w(y_min_s), y + h), axis_bgr)
        # _put_text(img, y_max_s, (x - 10 - _text_w(y_max_s), y + 12), axis_bgr)
        # x labels at bottom (optional)
        if show_x_labels:
            xs = f"{x0:.3g}s"
            xe = f"{x1:.3g}s"
            _put_text(img, xs, (x, y + h + 18), axis_bgr)
            _put_text(img, xe, (x + w - _text_w(xe), y + h + 18), axis_bgr)
        # optional grid
        if grid:
            # two faint horizontal lines (25%, 75%)
            for frac in (0.25, 0.75):
                yy = int(round(y + h - frac * h))
                cv2.line(img, (x, yy), (x + w, yy), (60, 60, 60), 1, cv2.LINE_AA)
            # two vertical lines (25%, 75%)
            for frac in (0.25, 0.75):
                xx = int(round(x + frac * w))
                cv2.line(img, (xx, y), (xx, y + h), (60, 60, 60), 1, cv2.LINE_AA)

    def _put_text(img, txt, org, color):
        cv2.putText(
            img,
            txt,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )

    def _text_w(txt):
        (w, _), _ = cv2.getTextSize(
            txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        return w

    # ------------- validate & normalize -------------
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("`time` must be 1D and length >= 2.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("`time` must be strictly increasing (seconds).")
    T = len(time)
    traces_arr, N = _as_traces_array(traces, T)

    # ylims argument validation (per-trace)
    if ylims is not None:
        if layout != "stacked" or shared_ylim:
            raise ValueError(
                "`ylims` is only supported when layout='stacked' and shared_ylim=False."
            )
        if ylim is not None:
            raise ValueError("Provide either `ylim` or `ylims`, not both.")
        if not isinstance(ylims, (list, tuple)) or len(ylims) != N:
            raise ValueError(
                f"`ylims` must be a list/tuple of length {N}, one (lo,hi) per trace."
            )
        # Coerce to floats and basic sanity checks later when assigning

    # y-limits
    if ylim is not None:
        ylo, yhi = float(ylim[0]), float(ylim[1])
    else:
        if shared_ylim or layout == "overlay":
            ylo = float(np.nanmin(traces_arr))
            yhi = float(np.nanmax(traces_arr))
            # Robustify against all-NaN inputs
            if not (np.isfinite(ylo) and np.isfinite(yhi)):
                ylo, yhi = 0.0, 1.0
            if yhi <= ylo:
                yhi = ylo + 1.0
        else:
            # per-trace limits later
            ylo = yhi = None

    # colors
    axis_bgr = _hex_or_name_to_bgr(axis_color)
    bg_bgr = _hex_or_name_to_bgr(bg_color)
    trace_bgrs = _ensure_color_list(trace_colors, N)
    labels_list = _normalize_trace_labels(trace_labels, N)

    # state mapping
    use_state_bar = (
        state_bar_height_px > 0 and states is not None and state_colors is not None
    )
    if use_state_bar:
        if len(states) != T:
            raise ValueError("`states` must have same length as `time`.")
        # pre-map state strings to BGR
        state_to_bgr = {k: _hex_or_name_to_bgr(v) for k, v in state_colors.items()}

    # ------------- layout -------------
    left, right, top, bottom = margins
    plot_w = max(10, width - left - right)
    plot_h_all = max(10, height - top - bottom)
    if layout == "overlay":
        panel_rects = [
            (
                left,
                top,
                plot_w,
                plot_h_all - (state_bar_height_px + 6 if use_state_bar else 0),
            )
        ]
        panel_ylo = [ylo]
        panel_yhi = [yhi]
    else:
        # stacked vertically
        bar_h_space = (state_bar_height_px + 6) if use_state_bar else 0
        each_h = max(10, (plot_h_all - bar_h_space) // N)
        panel_rects = [(left, top + i * each_h, plot_w, each_h - 2) for i in range(N)]
        if shared_ylim:
            panel_ylo = [ylo] * N
            panel_yhi = [yhi] * N
        else:
            if ylims is not None:
                # Use user-specified per-trace limits
                panel_ylo = []
                panel_yhi = []
                for i in range(N):
                    lo_i, hi_i = float(ylims[i][0]), float(ylims[i][1])
                    if not (np.isfinite(lo_i) and np.isfinite(hi_i)):
                        raise ValueError(f"ylims[{i}] must contain finite numbers.")
                    if hi_i <= lo_i:
                        hi_i = lo_i + 1.0
                    panel_ylo.append(lo_i)
                    panel_yhi.append(hi_i)
            else:
                # Auto per-trace limits
                panel_ylo = []
                panel_yhi = []
                for i in range(N):
                    lo_i = float(np.nanmin(traces_arr[i]))
                    hi_i = float(np.nanmax(traces_arr[i]))
                    if not (np.isfinite(lo_i) and np.isfinite(hi_i)):
                        lo_i, hi_i = 0.0, 1.0
                    if hi_i <= lo_i:
                        hi_i = lo_i + 1.0
                    panel_ylo.append(lo_i)
                    panel_yhi.append(hi_i)

    # state bar rect (top of plotting area)
    if use_state_bar:
        state_rect = (
            left,
            top + (0 if layout == "overlay" else 0),
            plot_w,
            state_bar_height_px,
        )

    # ------------- segmentation -------------
    t_start = float(time[0])
    t_end = float(time[-1])
    if time_span <= 0:
        raise ValueError("time_span must be > 0.")
    segments = []
    s = t_start
    while s < t_end - 1e-12:
        e = min(s + time_span, t_end)
        segments.append((s, e))
        s = e

    # ------------- encoding setup -------------
    H_eff, W_eff = height, width
    if pad_to_even and pix_fmt == "yuv420p":
        if H_eff % 2:
            H_eff += 1
        if W_eff % 2:
            W_eff += 1

    container = av.open(out_path, mode="w")
    stream = container.add_stream(codec, rate=int(round(fps)))
    stream.width = W_eff
    stream.height = H_eff
    stream.pix_fmt = pix_fmt
    opts = {}
    if codec in ("h264", "libx264"):
        if crf is not None:
            opts["crf"] = str(int(crf))
        if preset:
            opts["preset"] = preset
    elif codec == "h264_nvenc":
        if nvenc_cq is not None:
            opts["cq"] = str(int(nvenc_cq))
        # preset mapping depends on ffmpeg build; leave as-is if supplied
    if hasattr(stream, "codec_context") and hasattr(stream.codec_context, "options"):
        stream.codec_context.options = opts

    # ------------- main: per-segment precompute + per-frame draw -------------
    total_frames = 0
    # prebuild a base background (static global chrome)
    base_bg = np.zeros((height, width, 3), dtype=np.uint8)
    base_bg[:] = bg_bgr

    for seg_idx, (seg_s, seg_e) in enumerate(segments):
        seg_dur = seg_e - seg_s
        if seg_dur <= 0:
            continue

        # frames in this segment
        frames_in_seg = max(1, int(math.ceil(seg_dur * fps)))
        # x pixel coordinates for the panel width
        xp = np.arange(panel_rects[0][2], dtype=np.int32)  # 0..plot_w-1
        # time at each pixel column within segment
        t_at_x = seg_s + (xp / max(1, (panel_rects[0][2] - 1))) * seg_dur

        # resample traces to pixel columns (one value per column)
        # (N_panels = 1 for overlay, else N)
        if layout == "overlay":
            # single panel uses shared ylim
            ylo0, yhi0 = panel_ylo[0], panel_yhi[0]
            # precompute polyline segments for each trace, with gaps over NaNs
            polylines = []  # List[List[Tuple[start_col, pts_array]]]
            rect = panel_rects[0]
            x0, y0, w0, h0 = rect
            # indices to check validity per column (left/right neighbors in original samples)
            idx_r = np.searchsorted(time, t_at_x, side="left")
            idx_l = np.clip(idx_r - 1, 0, T - 1)
            idx_r = np.clip(idx_r, 0, T - 1)
            for i in range(N):
                src = traces_arr[i]
                finite_src = np.isfinite(src)
                # interpolate only over finite points (prevents NaN propagation)
                if finite_src.sum() >= 2:
                    yi = np.interp(t_at_x, time[finite_src], src[finite_src]).astype(
                        np.float32
                    )
                elif finite_src.sum() == 1:
                    # Degenerate: constant where nearest valid sample exists
                    yi = np.full_like(t_at_x, src[finite_src][0], dtype=np.float32)
                else:
                    yi = np.zeros_like(t_at_x, dtype=np.float32)
                # validity mask for columns: require both bracketing samples finite
                valid_cols = np.isfinite(src[idx_l]) & np.isfinite(src[idx_r])
                # map to pixel y (invert y axis: high values up)
                yy = (
                    y0
                    + (h0 - 1)
                    - np.round((yi - ylo0) / (yhi0 - ylo0 + 1e-12) * (h0 - 1)).astype(
                        np.int32
                    )
                )
                xx = x0 + xp
                # split into contiguous valid segments with at least 2 points
                segments_i = []
                start = None
                for col in range(w0):
                    if valid_cols[col]:
                        if start is None:
                            start = col
                    else:
                        if start is not None:
                            if col - start >= 2:
                                pts = np.stack([xx[start:col], yy[start:col]], axis=1)
                                segments_i.append((start, pts))
                            start = None
                if start is not None and (w0 - start) >= 2:
                    pts = np.stack([xx[start:w0], yy[start:w0]], axis=1)
                    segments_i.append((start, pts))
                polylines.append(segments_i)
        else:
            # stacked: one polyline set per trace with its own y scale (unless shared_ylim=True)
            polylines = []  # List[List[Tuple[start_col, pts_array]]]
            # indices to check validity per column (left/right neighbors in original samples)
            idx_r = np.searchsorted(time, t_at_x, side="left")
            idx_l = np.clip(idx_r - 1, 0, T - 1)
            idx_r = np.clip(idx_r, 0, T - 1)
            for i in range(N):
                rect = panel_rects[i]
                x0, y0, w0, h0 = rect
                yloi, yhii = panel_ylo[i], panel_yhi[i]
                src = traces_arr[i]
                finite_src = np.isfinite(src)
                if finite_src.sum() >= 2:
                    yi = np.interp(t_at_x, time[finite_src], src[finite_src]).astype(
                        np.float32
                    )
                elif finite_src.sum() == 1:
                    yi = np.full_like(t_at_x, src[finite_src][0], dtype=np.float32)
                else:
                    yi = np.zeros_like(t_at_x, dtype=np.float32)
                valid_cols = np.isfinite(src[idx_l]) & np.isfinite(src[idx_r])
                yy = (
                    y0
                    + (h0 - 1)
                    - np.round((yi - yloi) / (yhii - yloi + 1e-12) * (h0 - 1)).astype(
                        np.int32
                    )
                )
                xx = x0 + xp
                segments_i = []
                start = None
                for col in range(w0):
                    if valid_cols[col]:
                        if start is None:
                            start = col
                    else:
                        if start is not None:
                            if col - start >= 2:
                                pts = np.stack([xx[start:col], yy[start:col]], axis=1)
                                segments_i.append((start, pts))
                            start = None
                if start is not None and (w0 - start) >= 2:
                    pts = np.stack([xx[start:w0], yy[start:w0]], axis=1)
                    segments_i.append((start, pts))
                polylines.append(segments_i)

        # precompute a segment background (axes + labels for seg_s..seg_e)
        seg_bg = base_bg.copy()
        if layout == "overlay":
            _draw_axes(
                seg_bg,
                panel_rects[0],
                seg_s,
                seg_e,
                panel_ylo[0],
                panel_yhi[0],
                show_x_labels=False,
            )
            # overlay labels (legend-style)
            if labels_list is not None:
                x0, y0, w0, h0 = panel_rects[0]
                base_y = y0 + 14
                step = 16
                for i in range(N):
                    _put_text(
                        seg_bg,
                        str(labels_list[i]),
                        (x0 + 6, base_y + i * step),
                        trace_bgrs[i],
                    )
        else:
            for i in range(N):
                _draw_axes(
                    seg_bg,
                    panel_rects[i],
                    seg_s,
                    seg_e,
                    panel_ylo[i],
                    panel_yhi[i],
                    show_x_labels=False,
                )
                # per-panel label on stacked layout
                if labels_list is not None:
                    x0, y0, w0, h0 = panel_rects[i]
                    _put_text(
                        seg_bg, str(labels_list[i]), (x0 + 6, y0 + 14), trace_bgrs[i]
                    )

        # optional state bar image (prebuilt full width, then reveal columns progressively)
        if use_state_bar:
            bar = np.zeros((state_bar_height_px, panel_rects[0][2], 3), dtype=np.uint8)
            # color per column via nearest time index
            idx = np.searchsorted(time, t_at_x, side="left")
            idx = np.clip(idx, 0, T - 1)
            # build a row of colors
            row = np.zeros((panel_rects[0][2], 3), dtype=np.uint8)
            for col in range(panel_rects[0][2]):
                st = states[idx[col]]
                row[col] = state_to_bgr.get(st, (90, 90, 90))
            # tile vertically
            bar[:] = row[None, :, :]
            if state_bar_alpha < 1.0:
                # blend with bg color ahead of time
                bg_row = np.full_like(bar, fill_value=np.array(bg_bgr, dtype=np.uint8))
                bar = (
                    bar.astype(np.float32) * state_bar_alpha
                    + bg_row.astype(np.float32) * (1.0 - state_bar_alpha)
                ).astype(np.uint8)

        # per-frame draw + encode
        for fi in range(frames_in_seg):
            # current x cutoff (exclusive)
            frac = min(1.0, fi / max(1, frames_in_seg - 1))
            x_cut = int(round(frac * (panel_rects[0][2] - 1))) + 1  # at least 1 px

            frame = seg_bg.copy()

            # draw state bar up to x_cut
            if use_state_bar:
                bx, by, bw, bh = state_rect
                # place bar at very top of the plotting area (above panels)
                # here we put it at (left, top - bar - gap) OR simply on top of plots:
                bar_y = top  # minimalistic: bar sits at very top margin line
                bx, by = panel_rects[0][0], max(5, top - state_bar_height_px - 6)
                # ensure within canvas
                by = max(5, by)
                # paste partial bar
                frame[by : by + state_bar_height_px, bx : bx + x_cut, :] = bar[
                    :, :x_cut, :
                ]
                # centered state label text above the bar (static position, updates with current state)
                # determine state at current time (rightmost revealed column)
                cur_t = t_at_x[min(max(0, x_cut - 1), len(t_at_x) - 1)]
                cur_idx = int(
                    np.clip(np.searchsorted(time, cur_t, side="left"), 0, T - 1)
                )
                cur_state = states[cur_idx]
                text_color = state_to_bgr.get(cur_state, (90, 90, 90))
                cx = panel_rects[0][0] + panel_rects[0][2] // 2
                tx_w = _text_w(cur_state)
                tx_x = cx - tx_w // 2
                tx_y = max(12, by - 4)
                _put_text(frame, cur_state, (tx_x, tx_y), text_color)

            # draw polylines up to x_cut
            lineType = cv2.LINE_AA if aa_lines else cv2.LINE_8
            if layout == "overlay":
                for i in range(N):
                    segments_i = polylines[i]
                    if not segments_i:
                        continue
                    for start_col, pts in segments_i:
                        local_cut = min(max(0, x_cut - start_col), pts.shape[0])
                        if local_cut >= 2:
                            cv2.polylines(
                                frame,
                                [pts[:local_cut]],
                                isClosed=False,
                                color=trace_bgrs[i],
                                thickness=line_thickness,
                                lineType=lineType,
                            )
            else:
                for i in range(N):
                    segments_i = polylines[i]
                    if not segments_i:
                        continue
                    for start_col, pts in segments_i:
                        local_cut = min(max(0, x_cut - start_col), pts.shape[0])
                        if local_cut >= 2:
                            cv2.polylines(
                                frame,
                                [pts[:local_cut]],
                                isClosed=False,
                                color=trace_bgrs[i],
                                thickness=line_thickness,
                                lineType=lineType,
                            )

            # pad if needed for yuv420p even dims
            if (H_eff != height) or (W_eff != width):
                padded = np.zeros((H_eff, W_eff, 3), dtype=np.uint8)
                padded[:height, :width, :] = frame
                frame = padded

            # encode (feed as BGR to avoid channel swaps)
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            if (
                (video_frame.width != W_eff)
                or (video_frame.height != H_eff)
                or (video_frame.format.name != pix_fmt)
            ):
                video_frame = video_frame.reformat(
                    width=W_eff, height=H_eff, format=pix_fmt
                )
            for packet in stream.encode(video_frame):
                container.mux(packet)

            total_frames += 1
            if progress and total_frames % max(1, int(fps)) == 0:
                print(f"[animate_traces_video] encoded {total_frames} frames ...")

    # flush & close
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def normalize_roi_trace_to_peak(roi_trace):
    return roi_trace / np.max(roi_trace)


def generate_standard_synaptic_movies(
    subject,
    exp,
    loc,
    acq,
    t1,
    t2,
    out_dir="TEST_syn_movie",
    synapse_trace=["matchFilt"],
    soma_type="Fsvd",
    noise_threshold=5,
    background_gamma={1: 1.0, 2: 1.0},
):

    save_to = os.path.join(
        DEFS.anmat_root, "plots", "movies", subject, exp, loc, acq, out_dir
    )
    wis.util.gen.check_dir(save_to)

    si = wis.peri.sync.load_sync_info()
    sb = si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["sync_block"]
    dmd_info = wis.util.info.load_dmd_info()
    real_somas = []
    for dmd in [1, 2]:
        for sid in dmd_info[subject][exp][loc][acq][f"dmd-{dmd}"]["somas"]:
            real_somas.append(sid)

    h = wis.peri.anno.load_auto_hypno(subject, exp, sb, filter_unclear=0.7)

    # load the synapse dataframe:
    syndf = wis.scope.io.load_syndf(
        subject, exp, loc, acq, apply_ephys_offset=True, trace_types=synapse_trace
    )
    if t1 is None and t2 is None:
        t1 = syndf["time"].min()
        t2 = syndf["time"].max()
    syndf = syndf.with_columns((pl.col("noise") * noise_threshold).alias("noise"))
    syndf = syndf.with_columns((pl.col("data") / pl.col("noise")).alias("snr"))

    # get the glutamate sum traces:
    glut_sums = wis.scope.act.get_glut_sums(
        syndf.filter(pl.col("time").is_between(t1, t2))
    )
    glut_sums = glut_sums.sort(["soma-ID", "time"])
    glut_sums = glut_sums.filter(pl.col("soma-ID").is_in(real_somas))

    # Load the soma df and roi_map:
    somadf = wis.scope.io.load_roidf(
        subject, exp, loc, acq, apply_ephys_offset=True, roi_version=soma_type
    )
    dff, evdf = wis.scope.somas.detect_CaEvents(somadf)
    soma_cut = somadf.filter(pl.col("time").is_between(t1, t2)).filter(
        pl.col("roi_name").is_in(real_somas)
    )
    soma_cut = soma_cut.sort(["roi_name", "time"])

    soma_traces = []
    traces_to_animate = []
    trace_names = []
    trace_colors = []
    roi_events = []
    for soma in real_somas:
        soma_dff = (
            dff.filter(pl.col("roi_name") == soma)
            .filter(pl.col("time").is_between(t1, t2))["dff"]
            .to_numpy()
        )

        soma_trace_raw = soma_dff[::10]
        soma_trace_raw = np.nan_to_num(soma_trace_raw)  # replace nan with 0
        soma_traces.append(soma_trace_raw)

        roi_evdf = (
            evdf.filter(pl.col("roi_name") == soma)
            .filter(pl.col("peak_time").is_between(t1, t2))["peak_time"]
            .to_numpy()
        )
        roi_events.append(roi_evdf)

        traces_to_animate.append(soma_trace_raw)
        trace_names.append(f"{soma}-Ca2+")
        trace_colors.append("red")
        sum_trace = glut_sums.filter(pl.col("soma-ID") == soma)["data"].to_numpy()
        sum_trace = sum_trace[::10]
        sum_trace = np.nan_to_num(sum_trace)  # replace nan with 0
        if len(sum_trace) > 0:
            traces_to_animate.append(sum_trace)
            trace_names.append(f"{soma}-Glut")
            trace_colors.append("green")
    soma_trace_data = np.stack(soma_traces)
    for t in traces_to_animate:
        print(t.shape)

    mean_ims = wis.scope.io.load_mean_ims(subject, exp, loc, acq)

    # ----------------- PER DMD STARTS HERE-----------------

    for dmd in [1, 2]:
        pass_rois = True
        # load the synapse and roi map:
        roi_map = wis.scope.io.load_roi_map(subject, exp, loc, acq, dmd)
        if roi_map is None:
            pass_rois = False
        synmap, id_list = wis.scope.io.load_synapse_map(subject, exp, loc, acq, dmd)
        synmap[synmap > 0] = synmap[synmap > 0] - 1

        # get the traces per dmd to animate:
        d2 = syndf.filter(pl.col("dmd") == dmd).filter(
            pl.col("time").is_between(t1, t2)
        )
        d2 = d2.sort(["source", "time"])
        time_array = d2.filter(pl.col("source") == 1)["time"].to_numpy()
        time_array = time_array[::10]

        dat = []
        for source_id in d2["source"].unique().sort().to_list():
            d2_source = d2.filter(pl.col("source") == source_id)["snr"].to_numpy()
            d2_source = d2_source[::10]
            # replace nan with 0
            d2_source = np.nan_to_num(d2_source)
            dat.append(d2_source)
        act_data = np.stack(dat)

        save_path = os.path.join(save_to, f"imaging_data_DMD{dmd}.mp4")
        if pass_rois:
            print(f"GENERATING MOVIE WITH ROIS FOR DMD {dmd}")
            make_activity_movie(
                out_path=save_path,
                mean_im=mean_ims[dmd][1],  # (H,W)
                source_map=synmap,  # (H,W), -1 or [0..n_sources-1]
                act_data=act_data,  # (n_sources, T)
                act_time=time_array,  # (T,)
                soma_map=roi_map,  # optional
                soma_activity=soma_trace_data,  # optional
                soma_event_times=roi_events,  # optional
                soma_time=time_array,  # optional
                fps=20,  # derive from act_time; or set manually
                background_percentiles=(1, 96.5),
                background_gamma=background_gamma[dmd],
                syn_threshold=1,
                syn_saturate_at=2,
                soma_threshold=0.08,
                soma_saturate_at=0.6,
                codec="h264",
                crf=18,
                preset="veryfast",
            )
        else:
            print(f"GENERATING MOVIE WITHOUT ROIS FOR DMD {dmd}")
            make_activity_movie(
                out_path=save_path,
                mean_im=mean_ims[dmd][1],  # (H,W)
                source_map=synmap,  # (H,W), -1 or [0..n_sources-1]
                act_data=act_data,  # (n_sources, T)
                act_time=time_array,  # (T,)
                fps=20,  # derive from act_time; or set manually
                background_percentiles=(1, 99.9),
                background_gamma=background_gamma[dmd],
                syn_threshold=1,
                syn_saturate_at=2,
                codec="h264",
                crf=18,
                preset="veryfast",
            )
    # ------------------------Generate the video for sums and soma traces-------------------------
    # save_path = os.path.join(save_to, f"glut_sums_and_somas.mp4")
    # time_point_states = epy.hypno.hypno.get_states_fast(h, time_array)
    # animate_traces_video(
    #    out_path=save_path,
    #    traces=traces_to_animate,
    #    time=time_array,
    #    fps=20,
    #    layout="stacked",
    #    shared_ylim=False,  # each panel auto-scales (can pass `ylim=(lo,hi)` to force)
    #    trace_colors=trace_colors,
    #    line_thickness=2,
    #    grid=False,
    #    states=time_point_states,
    #    state_colors=DEFS.state_colors,
    #    trace_labels=trace_names,
    # )
    # print("DONE MOVIE GENERATION, SAVING FRAME TIMES")
    save_path = os.path.join(save_to, "imaging_frame_times.npy")
    np.save(save_path, time_array)
    return


def generate_peripheral_trace_movie(
    subject,
    exp,
    loc,
    acq,
    t1,
    t2,
    out_dir="TEST_syn_movie",
    eeg_store=["EEG_"],
):
    save_to = os.path.join(
        DEFS.anmat_root, "plots", "movies", subject, exp, loc, acq, out_dir
    )
    wis.util.gen.check_dir(save_to)

    si = wis.peri.sync.load_sync_info()
    sb = si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["sync_block"]

    h = wis.peri.anno.load_auto_hypno(subject, exp, sb, filter_unclear=0.7)

    ephys = wis.peri.ephys.load_single_ephys_block(
        subject, exp, sync_block=sb, stores=eeg_store
    )
    epslice = ephys["EEG_"].sel(time=slice(t1, t2))
    fs = float(epslice.fs)
    epslice_data = epslice.values
    ep_slice_time = epslice.time.values

    # downsample the eeg data to 200 hz and replace nan with 0
    ep_data_ds, time_array = epy.sigpro.gen.downsample_signal(
        epslice_data, fs, float(200.0), t=ep_slice_time
    )
    ep_data_ds = np.nan_to_num(ep_data_ds)  # replace nan with 0

    eyedf_full = wis.peri.vid.load_eye_metric_df(subject, exp, sb)
    eyedf = eyedf_full.sort(["time"]).filter(pl.col("time").is_between(t1, t2))

    whisdf_full = wis.peri.vid.load_whisking_df(subject, exp, sb)
    whisdf = whisdf_full.sort(["time"]).filter(pl.col("time").is_between(t1, t2))
    whis_data = whisdf["whis"].to_numpy()
    whis_data = np.nan_to_num(whis_data)  # replace nan with 0
    whis_data = epy.sigpro.gen.match_data_to_times(
        whis_data, whisdf["time"].to_numpy(), time_array
    )
    diameter = eyedf["diameter"].to_numpy()
    diameter = np.nan_to_num(diameter)  # replace nan with 0
    diameter = epy.sigpro.gen.match_data_to_times(
        diameter, eyedf["time"].to_numpy(), time_array
    )
    motion = eyedf["motion"].to_numpy()
    motion = np.nan_to_num(motion)  # replace nan with 0
    motion = epy.sigpro.gen.match_data_to_times(
        motion, eyedf["time"].to_numpy(), time_array
    )

    lid = eyedf["lid_norm"].to_numpy()
    lid = np.nan_to_num(lid)  # replace nan with 0
    lid = epy.sigpro.gen.match_data_to_times(lid, eyedf["time"].to_numpy(), time_array)

    # calculate the y-limits for all traces
    eeg_lims = (
        np.nanmin(ephys["EEG_"].values),
        np.nanpercentile(ephys["EEG_"].values, 95),
    )
    whis_lims = (
        np.nanmin(whisdf_full["whis"].to_numpy()),
        np.nanpercentile(whisdf_full["whis"].to_numpy(), 95),
    )
    diameter_lims = (
        np.nanmin(eyedf_full["diameter"].to_numpy()),
        np.nanpercentile(eyedf_full["diameter"].to_numpy(), 95),
    )
    motion_lims = (
        np.nanmin(eyedf_full["motion"].to_numpy()),
        np.nanpercentile(eyedf_full["motion"].to_numpy(), 95),
    )
    lid_lims = (
        np.nanmin(eyedf_full["lid_norm"].to_numpy()),
        np.nanpercentile(eyedf_full["lid_norm"].to_numpy(), 95),
    )
    ylims = []
    ylims.append(eeg_lims)
    ylims.append(whis_lims)
    ylims.append(diameter_lims)
    ylims.append(motion_lims)
    ylims.append(lid_lims)
    traces_to_animate = [ep_data_ds, whis_data, diameter, motion, lid]
    trace_names = ["EEG", "Whisk", "Pup-Dia", "Pup-Mot", "Pup-Lid"]
    trace_colors = ["white", "#fc03ca", "#3feafc", "#3feafc", "#3feafc"]
    time_point_states = epy.hypno.hypno.get_states_fast(h, time_array)

    save_path = os.path.join(save_to, "peripheral_movie.mp4")
    animate_traces_video(
        out_path=save_path,
        traces=traces_to_animate,
        time=time_array,
        fps=200,
        layout="stacked",
        shared_ylim=False,  # each panel auto-scales (can pass `ylim=(lo,hi)` to force)
        trace_colors=trace_colors,
        line_thickness=2,
        grid=False,
        states=time_point_states,
        state_colors=DEFS.state_colors,
        trace_labels=trace_names,
    )

    save_path = os.path.join(save_to, "peripheral_frame_times.npy")
    np.save(save_path, time_array)
    return
