import os

import electro_py as epy
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import slap2_py as spy
import tifffile as tiff
from kplot.colors import flex
from matplotlib.path import Path
from PIL import Image

import wisco_slap as wis
from wisco_slap.defs import annotation_root, data_root


def build_synapse_id_list(n_syns: int) -> np.ndarray:
    """Build the ID list stored alongside the synapse label map."""
    if n_syns < 0:
        raise ValueError("n_syns must be non-negative")

    ids = np.array([f"{i}" for i in range(n_syns)], dtype=np.str_)
    if n_syns == 0:
        return ids

    # The label map stores source indices as 1..N while the user-facing IDs are 0..N-1.
    # Duplicating "0" at the front preserves the existing WISynaptic lookup behavior.
    return np.insert(ids, 0, ids[0])


def save_acq_canvas_images(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_path: str,
    vmin: int = 5,
    vmax: int = 75,
):
    """Generate and save canvas images for each DMD.

    Writes three artifacts per DMD under the acquisition's ``canvas/`` directory:

    - ``canvas/DMD-{dmd}.png`` — mean image (Channel-2 of ``meanIM``), with any
      manually drawn ROIs overlaid.
    - ``canvas/syn_overlays/DMD-{dmd}.png`` — same mean image with the synapse
      footprint values overlaid on top.
    - ``canvas/activity_image/DMD-{dmd}.npy`` — the raw per-DMD activity image
      (``actIM``) saved as a float array so downstream viewers (WISynaptic) can
      apply live vmin/vmax/contrast adjustments.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    loc : str
        location name
    acq : str
        acquisition name
    esum_path : str
        path to the ExperimentSummary .mat file
    vmin : int, optional
        lower percentile for colormap, by default 5
    vmax : int, optional
        upper percentile for colormap, by default 75
    """
    plt.style.use("dark_background")
    canvas_dir = os.path.join(annotation_root, subject, exp, loc, acq, "canvas")
    wis.util.check_dir(canvas_dir)
    activity_dir = os.path.join(canvas_dir, "activity_image")
    wis.util.check_dir(activity_dir)
    mean_im = spy.xsum.get_meanIM(esum_path)
    act_im = spy.xsum.get_actIM(esum_path)
    mp, fpv = spy.xsum.get_fp_info(esum_path)
    for dmd in [1, 2]:
        fname = os.path.join(canvas_dir, f"DMD-{dmd}.png")
        fname_overlay = os.path.join(canvas_dir, "syn_overlays", f"DMD-{dmd}.png")
        fname_activity = os.path.join(activity_dir, f"DMD-{dmd}.npy")
        wis.util.check_dir(os.path.dirname(fname_overlay))
        img = mean_im[dmd][1, :, :]
        fh = img.shape[0] / 30
        fw = img.shape[1] / 30
        v_min = np.nanpercentile(img, vmin)
        v_max = np.nanpercentile(img, vmax)
        f, ax = plt.subplots(1, 1, figsize=(fw, fh))
        ax.imshow(img, vmin=v_min, vmax=v_max, cmap="viridis")
        ax.set_title(f"{subject} | DMD-{dmd}, Channel-2, {exp}, {loc}, {acq}")

        # add the manually drawn ROIs if applicable:
        roi_list = spy.xsum.get_roi_list(esum_path, dmd)
        if len(roi_list) > 0:
            for s in range(len(roi_list)):
                name = roi_list[s]["Label"]
                roi_locs = roi_list[s]["Position"]
                x = roi_locs[0]
                y = roi_locs[1]
                roi_locations = []
                for i in range(len(x)):
                    roi_locations.append([x[i], y[i]])
                for i in range(len(roi_locations)):
                    ax.scatter(
                        roi_locations[i][0], roi_locations[i][1], color="red", s=50
                    )
                for i in range(len(roi_locations)):
                    next_i = (i + 1) % len(roi_locations)
                    ax.plot(
                        [roi_locations[i][0], roi_locations[next_i][0]],
                        [roi_locations[i][1], roi_locations[next_i][1]],
                        color="red",
                        linewidth=2,
                    )
                roi_x = [point[0] for point in roi_locations]
                roi_y = [point[1] for point in roi_locations]
                centroid_x = np.mean(roi_x)
                centroid_y = np.mean(roi_y)
                ax.text(
                    centroid_x,
                    centroid_y,
                    name,
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=12,
                    fontweight="normal",
                )

        # Save the main mean image (with ROIs drawn on if present)
        f.savefig(fname, bbox_inches="tight", pad_inches=0, dpi=300)

        # Save the synapse overlay image (always, regardless of ROIs)
        ax.imshow(img, vmin=v_min, vmax=v_max * 2, cmap="viridis")
        ax.imshow(fpv[dmd])
        f.savefig(fname_overlay, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(f)

        # Save the raw activity image as a float array for live-adjustable display.
        np.save(fname_activity, np.asarray(act_im[dmd]))


def save_single_trial_ref_images(
    subject, exp, loc, acq, overwrite=False, vmin=5, vmax=85
):
    data_dir = os.path.join(data_root, subject, exp, loc, acq)
    mean_im_dir = os.path.join(annotation_root, subject, exp, loc, acq, "canvas")
    wis.util.check_dir(mean_im_dir)
    for dmd in [1, 2]:
        for fname in os.listdir(data_dir):
            if f"DMD{dmd}-CYCLE-000000.tif" in fname:
                fig_name = os.path.join(mean_im_dir, f"DMD-{dmd}__TRIAL-000000.png")
                if os.path.exists(fig_name) and not overwrite:
                    print(
                        "DMD-"
                        f"{dmd} mean image (single trial) already exists: "
                        f"{fig_name}, use overwrite=True to overwrite"
                    )
                    continue
                tiff_path = f"{data_dir}/{fname}"
                img = spy.img.io.load_tiff(tiff_path)
                img = np.mean(img, axis=0)
                fh = img.shape[0] / 30
                fw = img.shape[1] / 30
                # v_min = np.percentile(img, vmin)
                # v_max = np.percentile(img, vmax)
                v_min = 0
                # Temporary fix: percentile scaling was returning zeros here.
                v_max = 35
                f, ax = plt.subplots(1, 1, figsize=(fw, fh))
                ax.imshow(img, vmin=v_min, vmax=v_max, cmap="viridis")
                ax.set_title(
                    f"{subject} | DMD-{dmd}, {exp}, {loc}, {acq} | First Trial"
                )

                f.savefig(fig_name, bbox_inches="tight", pad_inches=0, dpi=300)
                plt.close(f)


def write_processing_unfinished(subject, exp, loc, acq):
    annotation_root_acq = os.path.join(annotation_root, subject, exp, loc, acq)
    wis.util.check_dir(annotation_root_acq)
    return None


def polygon_mask_from_roi(img, roi_locs, dtype=np.uint8):
    """
    Create a binary mask (1 inside, 0 outside) for a polygon defined by ROI points.

    Parameters
    ----------
    img : np.ndarray
        Base image array. Only its height and width are used for mask shape.
    roi_locs : sequence
        ROI polygon coordinates in the same format used above when plotting:
        either [x_coords, y_coords] (two 1D arrays/lists of equal length),
        or an (N, 2) array-like of vertex pairs [[x0, y0], [x1, y1], ...].
        Coordinates are expected in (x, y) order (columns, rows) matching
        matplotlib plotting conventions used elsewhere in this module.
    dtype : numpy dtype, optional
        Output mask dtype (default: np.uint8).

    Returns
    -------
    np.ndarray
        A (H, W) array with value 1 inside the polygon and 0 elsewhere.
    """
    h, w = img.shape[:2]

    if isinstance(roi_locs, (list, tuple)) and len(roi_locs) == 2:
        x = np.asarray(roi_locs[0], dtype=float)
        y = np.asarray(roi_locs[1], dtype=float)
        if x.shape != y.shape:
            raise ValueError("roi_locs coordinate arrays must have the same shape")
        verts = np.stack([x, y], axis=1)
    else:
        verts = np.asarray(roi_locs, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError(
                "roi_locs must be [x_coords, y_coords] or an (N, 2) array of (x, y)"
            )

    poly_path = Path(verts, closed=True)

    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))

    inside = poly_path.contains_points(pts)
    mask = np.zeros((h, w), dtype=dtype)
    mask.ravel()[inside] = 1
    return mask


def save_master_image(
    master_array: np.ndarray, out_path: str = "master_image.png"
) -> None:
    """Save a 2D array as a grayscale PNG, rescaling to uint8 [0, 255]."""
    if master_array.ndim != 2:
        raise ValueError("master_array must be 2D (H, W)")

    if master_array.dtype == np.uint8:
        arr_u8 = master_array
    else:
        a = master_array.astype(np.float32)
        vmin = np.nanmin(a)
        vmax = np.nanmax(a)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
            arr_u8 = np.zeros_like(a, dtype=np.uint8)
        else:
            a = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0) * 255.0
            arr_u8 = np.rint(a).astype(np.uint8)

    Image.fromarray(arr_u8, mode="L").save(out_path)


def save_source_location_key_npz(
    label_map: np.ndarray,
    id_list: np.ndarray,
    out_path: str = "source_location_key.npz",
) -> None:
    """Save synapse label map and ID list as a compressed npz archive."""
    if label_map.ndim != 2:
        raise ValueError("label_map must be 2D (H, W)")
    lm = np.asarray(label_map)
    if lm.dtype.kind not in ("i", "u"):
        raise ValueError("label_map must have an integer dtype")
    lm = lm.astype(np.int32, copy=False)

    ids = np.asarray(id_list, dtype=np.str_)
    if ids.ndim != 1:
        raise ValueError("id_list must be 1D")

    if lm.size:
        min_idx = int(lm.min())
        max_idx = int(lm.max())
        if min_idx < -1:
            raise ValueError("label_map contains indices < -1")
        if max_idx >= len(ids):
            raise ValueError(f"label_map index {max_idx} >= id_list length {len(ids)}")

    np.savez_compressed(out_path, label_map=lm, id_list=ids)


def save_master_image_and_key(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_path: str,
):
    """Generate and save master image and source location key for both DMDs.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    loc : str
        location name
    acq : str
        acquisition name
    esum_path : str
        path to the ExperimentSummary .mat file
    """
    synapse_ids_dir = os.path.join(
        annotation_root, subject, exp, loc, acq, "synapse_ids"
    )
    wis.util.check_dir(synapse_ids_dir)
    mean_im = spy.xsum.get_meanIM(esum_path)
    synmaps, fp_vals = spy.xsum.get_fp_info(esum_path)

    for dmd in [1, 2]:
        synmap = synmaps[dmd]
        n_syns = int(max(synmap.max(), 0))
        id_list = build_synapse_id_list(n_syns)
        master_image_array = mean_im[dmd][1]
        basedir = os.path.join(synapse_ids_dir, f"dmd-{dmd}")
        wis.util.check_dir(basedir)
        mip = os.path.join(basedir, "master_image.png")
        save_master_image(master_image_array, mip)
        save_source_location_key_npz(
            synmap, id_list, os.path.join(basedir, "source_location_key.npz")
        )


def expand_synmap(mp, fpv):
    fp_exp = {}
    for dmd in [1, 2]:
        syn_masks = []
        n_sources = mp[dmd].max()
        for i in range(n_sources):
            mask = mp[dmd] == i + 1
            fpv_masked = fpv[dmd].copy()
            fpv_masked[~mask.astype(bool)] = np.nan
            syn_masks.append(fpv_masked)
        fp_exp[dmd] = syn_masks
    return fp_exp


def save_synapse_id_plots_and_key(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_path: str,
    upper_vmax_pct: int = 95,
    buffer: int = 25,
    channel: int = 2,
):
    """Generate and save individual synapse ID plots, master image, and source key.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    loc : str
        location name
    acq : str
        acquisition name
    esum_path : str
        path to the ExperimentSummary .mat file
    upper_vmax_pct : int, optional
        upper percentile for colormap, by default 95
    buffer : int, optional
        pixel buffer around synapse for cropping, by default 25
    channel : int, optional
        channel to use, by default 2
    """
    spy.plot.slap_style("im")
    mean_im = spy.xsum.get_meanIM(esum_path)
    mp, fpv = spy.xsum.get_fp_info(esum_path)
    fp = expand_synmap(mp, fpv)

    for dmd in [1, 2]:
        save_dir = os.path.join(
            annotation_root, subject, exp, loc, acq, "synapse_ids", f"dmd-{dmd}"
        )
        wis.util.check_dir(save_dir)
        dmd_nsources = len(fp[dmd])
        for source in range(dmd_nsources):
            f, ax = spy.plot.images.synapse_id_plot(
                mean_im,
                fp,
                dmd,
                source,
                upper_vmax_pct=upper_vmax_pct,
                buffer=buffer,
                channel=channel,
                subject=subject,
                exp=exp,
                loc=loc,
                acq=acq,
            )
            f.savefig(f"{save_dir}/{source}.png", dpi=300, bbox_inches="tight")
            plt.close(f)
    save_master_image_and_key(subject, exp, loc, acq, esum_path)


def create_annotation_basics(subject: str, exp: str, loc: str, acq: str):
    """Create any missing top-level acquisition files for annotation materials.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    loc : str
        location name
    acq : str
        acquisition name
    """
    annotation_root_acq = os.path.join(annotation_root, subject, exp, loc, acq)
    wis.util.check_dir(annotation_root_acq)
    md_file = os.path.join(annotation_root_acq, "notes.md")
    materials_path = os.path.join(annotation_root_acq, "materials.txt")

    if not os.path.exists(materials_path):
        with open(materials_path, "w") as f:
            f.write(f"# Annotation Materials -- {subject} {exp} {loc} {acq}\n")

    if not os.path.exists(md_file):
        with open(md_file, "w") as f:
            f.write(f"# Notes -- {subject} {exp} {loc} {acq}\n")


def save_roi_location_tiffs(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    esum_path: str,
):
    """Save mean images as TIFFs for ROI location annotation.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    loc : str
        location name
    acq : str
        acquisition name
    esum_path : str
        path to the ExperimentSummary .mat file
    """
    mean_im = spy.xsum.get_meanIM(esum_path)
    roi_loc_dir = os.path.join(annotation_root, subject, exp, loc, acq, "roi_locations")
    wis.util.check_dir(roi_loc_dir)
    for dmd in [1, 2]:
        image_to_save = mean_im[dmd][1]
        path = os.path.join(roi_loc_dir, f"roi_locs_dmd{dmd}.tif")

        arr = np.ascontiguousarray(image_to_save)
        np.nan_to_num(arr, copy=False, nan=0.0)

        photometric = (
            "rgb" if arr.ndim == 3 and arr.shape[-1] in (3, 4) else "minisblack"
        )
        axes = (
            "YX"
            if arr.ndim == 2
            else ("YXC" if arr.ndim == 3 and arr.shape[-1] in (3, 4) else None)
        )

        tiff.imwrite(
            path,
            arr,
            photometric=photometric,
            metadata=({"axes": axes} if axes else None),
        )


# write nan-value plots
# give a scopex array, give me a basic plot where each row is a synapse; x axis is time, the color indicates whether the value is nan or not.
# flex.green300 for not nan, flex.magenta400 for nan. Blank if there is simply a gap.
def plot_nan_values(sx_array, ax=None, gap_factor=1.5):
    """Plot the NaN structure of a scopex array.

    Each row is one synapse, x is time. Green where the value is finite,
    magenta where the value is NaN, blank where the time axis itself has a gap
    (consecutive samples spaced > `gap_factor` x the median sampling interval).

    Parameters
    ----------
    sx_array : xr.DataArray
        Scopex DataArray with `time` and `syn_id` dims. If a `channel` dim of
        size 1 is present it is squeezed; >1 errors.
    ax : matplotlib.axes.Axes, optional
    gap_factor : float
        Intervals between consecutive original time samples larger than
        `gap_factor * median(dt)` are rendered as transparent gap cells.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from matplotlib.colors import ListedColormap

    da = sx_array
    if "channel" in da.dims:
        if da.sizes["channel"] == 1:
            da = da.squeeze("channel", drop=True)
        else:
            raise ValueError(
                f"`sx_array` has {da.sizes['channel']} channels — select one first."
            )
    if "time" not in da.dims or "syn_id" not in da.dims:
        raise ValueError("`sx_array` must have `time` and `syn_id` dims.")
    da = da.transpose("syn_id", "time")

    time = np.asarray(da["time"].values, dtype=float)
    if time.size < 2:
        raise ValueError("Need at least 2 timepoints to plot.")
    dts = np.diff(time)
    dt = float(np.median(dts))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not determine a positive median sampling interval.")

    # Uniform grid spanning the full time range; original samples drop into
    # their nearest cell, every other cell stays NaN (= rendered transparent).
    n_grid = int(np.round((time[-1] - time[0]) / dt)) + 1
    grid_idx = np.clip(np.round((time - time[0]) / dt).astype(int), 0, n_grid - 1)

    is_nan = np.isnan(da.values)  # (n_syn, n_time)
    n_syn = is_nan.shape[0]

    # 0 = valid (green), 1 = nan (magenta), NaN = gap (transparent)
    img = np.full((n_syn, n_grid), np.nan, dtype=float)
    img[:, grid_idx] = is_nan.astype(float)

    # Force-blank cells that fall inside a "real" time gap (>gap_factor*dt
    # between consecutive original samples), so a single straggling sample at
    # the end of a gap doesn't paint a long magenta/green stripe.
    gap_starts = np.where(dts > gap_factor * dt)[0]
    for k in gap_starts:
        a = grid_idx[k] + 1
        b = grid_idx[k + 1]
        if b > a:
            img[:, a:b] = np.nan

    cmap = ListedColormap([flex.green500, flex.blue600])
    cmap.set_bad("none")

    if ax is None:
        _, ax = plt.subplots(figsize=(14, max(2.0, 0.18 * n_syn)))

    ax.imshow(
        img,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(time[0] - dt / 2, time[0] + (n_grid - 0.5) * dt, -0.5, n_syn - 0.5),
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("synapse")
    ax.set_yticks(np.arange(n_syn))
    ax.set_yticklabels([str(s) for s in np.asarray(da["syn_id"].values)], fontsize=6)
    valid_frac = float(np.mean(~is_nan))
    ax.set_title(
        f"NaN map — {n_syn} syns, {time.size} samples, valid frac: {valid_frac:.3f}"
    )
    return ax


def exp_nan_plot(subject, exp, loc, acq, save=True, array=None):
    if array is None:
        denoised = wis.get.syn_dF(
            subject, exp, loc, acq, trace="denoised", channels=[0]
        )
        dn = wis.get.combine_scopex_arrays(denoised)
    else:
        dn = array
    hypno = wis.get.bout_hypno(subject, exp, loc, acq)
    hacq = hypno.trim(dn.time.values[0], dn.time.values[-1])
    f, ax = plt.subplots(
        2,
        1,
        figsize=(28, 15),
        gridspec_kw={"height_ratios": [1, 6]},
        layout="constrained",
    )
    plot_nan_values(dn, ax=ax[1])
    state_colors = {}
    state_colors["NREM"] = (2, flex.blue600)
    state_colors["REM"] = (3, flex.magenta600)
    state_colors["Wake"] = (1, flex.green600)
    f, ax[0] = epy.plot.hypno.plot_basic_hypnogram(
        hacq,
        f,
        ax[0],
        xlim=(dn.time.values[0], dn.time.values[-1]),
        style_path=None,
        single_tone=False,
        state_colors=state_colors,
    )
    total_duration = dn.time.values[-1] - dn.time.values[0]
    nrem_duration = hacq.df.filter(pl.col("state") == "NREM")["duration"].sum()
    wake_duration = hacq.df.filter(pl.col("state") == "Wake")["duration"].sum()
    rem_duration = hacq.df.filter(pl.col("state") == "REM")["duration"].sum()
    f.suptitle(
        f"{subject}, {exp}, {loc}, {acq} | Total Duration = {float(total_duration):.1f} s | "
        f"NREM: {float(nrem_duration):.1f} s | Wake: {float(wake_duration):.1f} s | REM: {float(rem_duration):.1f} s",
        fontsize=18,
    )
    if save:
        supplementary_plots_dir = os.path.join(
            wis.defs.annotation_root, subject, exp, loc, acq, "supplementary_plots"
        )
        if not os.path.exists(supplementary_plots_dir):
            os.makedirs(supplementary_plots_dir, exist_ok=True)
        path = os.path.join(supplementary_plots_dir, "Hypno_NaNs.png")
        f.savefig(path, dpi=300)
    return f, ax
