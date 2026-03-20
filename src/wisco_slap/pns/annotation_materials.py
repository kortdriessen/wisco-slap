import os

import matplotlib.pyplot as plt
import numpy as np
import slap2_py as spy
import tifffile as tiff
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


def save_acq_mean_images(
    subject: str, exp: str, loc: str, acq: str, esum_path: str,
    vmin: int = 5, vmax: int = 75,
):
    """Generate and save mean images and synapse overlay images for each DMD.

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
    mean_im_dir = os.path.join(annotation_root, subject, exp, loc, acq, "canvas")
    wis.util.check_dir(mean_im_dir)
    mean_im = spy.xsum.get_meanIM(esum_path)
    mp, fpv = spy.xsum.get_fp_info(esum_path)
    for dmd in [1, 2]:
        fname = os.path.join(mean_im_dir, f"DMD-{dmd}.png")
        fname_overlay = os.path.join(mean_im_dir, "syn_overlays", f"DMD-{dmd}.png")
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
    subject: str, exp: str, loc: str, acq: str, esum_path: str,
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
    subject: str, exp: str, loc: str, acq: str, esum_path: str,
    upper_vmax_pct: int = 95, buffer: int = 25, channel: int = 2,
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
    subject: str, exp: str, loc: str, acq: str, esum_path: str,
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
    roi_loc_dir = os.path.join(
        annotation_root, subject, exp, loc, acq, "roi_locations"
    )
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
