import os

import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import polars as pl
import slap2_py as spy

import wisco_slap as wis
import wisco_slap.defs as DEFS
import pandas as pd

import wisco_slap as wis
import slap2_py as spy
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import tifffile as tiff

annotation_root = os.path.join(DEFS.anmat_root, "annotation_materials")


def _get_roi_container_DEP(rois, dmd):
    if type(rois[dmd - 1][0]) is dict:
        return rois[dmd - 1]
    elif type(rois[dmd - 1][0]) is list:
        return rois[dmd - 1][0]
    else:
        raise ValueError(
            "Unexpected ROI container type, please investigate the ROI structure"
        )


def _get_roi_container(p, dmd):
    r = spy.hf.load_any(p, f"/exptSummary/userROIs[{dmd - 1}][0]")
    if type(r) is dict:
        return [r]
    else:
        return r


def save_acq_mean_images(subject, exp, loc, acq, overwrite=False, vmin=5, vmax=75):
    plt.style.use("dark_background")
    if wis.util.info.determine_processing_done(subject, exp, loc, acq) == "NO":
        return
    esum_path = wis.util.io.sub_esum_path(subject, exp, loc, acq)
    if esum_path is None:
        return
    # rois = spy.hf.load_any(esum_path, "/exptSummary/userROIs")
    # Save each dmd meanIM
    mean_im_dir = os.path.join(
        DEFS.anmat_root, "annotation_materials", subject, exp, loc, acq, "canvas"
    )
    wis.util.gen.check_dir(mean_im_dir)
    for dmd in [1, 2]:
        fname = os.path.join(mean_im_dir, f"DMD-{dmd}.png")
        if os.path.exists(fname) and not overwrite:
            print(
                f"DMD-{dmd} mean image already exists: {fname}, use overwrite=True to overwrite"
            )
            continue
        meanim = spy.hf.load_any(esum_path, f"/exptSummary/meanIM[{dmd - 1}][0]")
        img = meanim[1, :, :].T
        fh = img.shape[0] / 30
        fw = img.shape[1] / 30
        v_min = np.nanpercentile(img, vmin)
        v_max = np.nanpercentile(img, vmax)
        f, ax = plt.subplots(1, 1, figsize=(fw, fh))
        ax.imshow(img, vmin=v_min, vmax=v_max, cmap="viridis")
        ax.set_title(f"{subject} | DMD-{dmd}, Channel-2, {exp}, {loc}, {acq}")

        # add the manually drawn ROIs if applicable:
        roi_list = _get_roi_container(esum_path, dmd)
        if type(roi_list) is np.ndarray:  # case where there are no ROIs
            f.savefig(fname, bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close(f)
            continue
        else:
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
                # Connect each ROI location with a line to the next one, and connect the last to the first
                for i in range(len(roi_locations)):
                    next_i = (i + 1) % len(
                        roi_locations
                    )  # Wrap around to first point for last point
                    ax.plot(
                        [roi_locations[i][0], roi_locations[next_i][0]],
                        [roi_locations[i][1], roi_locations[next_i][1]],
                        color="red",
                        linewidth=2,
                    )
                # Calculate the centroid of the ROI polygon to place the text
                roi_x = [point[0] for point in roi_locations]
                roi_y = [point[1] for point in roi_locations]
                centroid_x = np.mean(roi_x)
                centroid_y = np.mean(roi_y)

                # Add text at the centroid
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

            f.savefig(fname, bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close(f)
    return


def save_single_trial_ref_images(
    subject, exp, loc, acq, overwrite=False, vmin=5, vmax=85
):
    data_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}"
    mean_im_dir = os.path.join(
        DEFS.anmat_root, "annotation_materials", subject, exp, loc, acq, "canvas"
    )
    wis.util.gen.check_dir(mean_im_dir)
    for dmd in [1, 2]:
        for fname in os.listdir(data_dir):
            if f"DMD{dmd}-CYCLE-000000.tif" in fname:
                fig_name = os.path.join(mean_im_dir, f"DMD-{dmd}__TRIAL-000000.png")
                if os.path.exists(fig_name) and not overwrite:
                    print(
                        f"DMD-{dmd} mean image (single trial) already exists: {fig_name}, use overwrite=True to overwrite"
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
                v_max = 35  # temporary fix for now, nanpercentile not working, getting all zeros
                f, ax = plt.subplots(1, 1, figsize=(fw, fh))
                ax.imshow(img, vmin=v_min, vmax=v_max, cmap="viridis")
                ax.set_title(
                    f"{subject} | DMD-{dmd}, {exp}, {loc}, {acq} | First Trial"
                )
                # ax.set_axis_off()

                f.savefig(fig_name, bbox_inches="tight", pad_inches=0, dpi=300)
                plt.close(f)
    return


def save_all_subject_ref_images(overwrite=False, vmin=5, vmax=85):
    wis.pipes.exp_info.update_exp_info_spreadsheet()
    ei = wis.pipes.exp_info.load_exp_info_spreadsheet()
    for subject in ei["subject"].unique():
        for exp in ei.filter(pl.col("subject") == subject)["experiment"].unique():
            for loc in (
                ei.filter(pl.col("subject") == subject)
                .filter(pl.col("experiment") == exp)["location"]
                .unique()
            ):
                for acq in (
                    ei.filter(pl.col("subject") == subject)
                    .filter(pl.col("experiment") == exp)
                    .filter(pl.col("location") == loc)["acquisition"]
                    .unique()
                ):

                    processed = (
                        ei.filter(pl.col("subject") == subject)
                        .filter(pl.col("experiment") == exp)
                        .filter(pl.col("location") == loc)
                        .filter(pl.col("acquisition") == acq)["processed"]
                        .to_list()[0]
                    )
                    if processed == "YES":
                        print(f"Processing meanIMs {subject} {exp} {loc} {acq}")
                        save_acq_mean_images(
                            subject, exp, loc, acq, overwrite, vmin, vmax
                        )
                    elif processed == "NO":
                        print(
                            f"Processing not done yet for {subject} {exp} {loc} {acq}"
                        )
                        save_single_trial_ref_images(
                            subject, exp, loc, acq, overwrite, vmin, vmax
                        )
                    else:
                        continue
    return


def write_processing_unfinished(subject, exp, loc, acq):
    annotation_root = os.path.join(
        DEFS.anmat_root, "annotation_materials", subject, exp, loc, acq
    )
    wis.util.gen.check_dir(annotation_root)
    return None


# save_acq_synapse_id_plots(subject, exp, loc, acq):


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
    # Determine output shape from image
    h, w = img.shape[:2]

    # Normalize roi_locs into an (N, 2) array of (x, y) vertices
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

    # Construct a closed polygon path in (x, y) space
    poly_path = Path(verts, closed=True)

    # Build grid of pixel centers in (x, y) = (col + 0.5, row + 0.5)
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))

    inside = poly_path.contains_points(pts)
    mask = np.zeros((h, w), dtype=dtype)
    mask.ravel()[inside] = 1
    return mask


def _save_anat_map_basic(subject, overwrite=False):
    anat_map_path = f"{DEFS.data_root}/{subject}/{subject}_anatomy/anatomy_map.png"
    if os.path.exists(anat_map_path) and not overwrite:
        print(
            f"Anatomy map already exists: {anat_map_path}, use overwrite=True to overwrite"
        )
        return
    else:
        points = wis.scope.anat.load_points(
            f"{DEFS.data_root}/{subject}/{subject}_anatomy/anatomy_locations.yaml"
        )
        f, ax = wis.scope.anat.plot_points_with_boxes(points)
        f.savefig(anat_map_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(f)
    return


def save_anat_map_with_window(subject, overwrite=False):
    anat_map_path = (
        f"{DEFS.data_root}/{subject}/{subject}_anatomy/anatomy_map_complete.png"
    )
    if os.path.exists(anat_map_path) and not overwrite:
        print(
            f"Anatomy map already exists: {anat_map_path}, use overwrite=True to overwrite"
        )
        return
    else:
        points = wis.scope.anat.load_points(
            f"{DEFS.data_root}/{subject}/{subject}_anatomy/anatomy_locations.yaml"
        )
        f, ax = wis.scope.anat.plot_points_with_boxes(points)
        f, ax = wis.scope.anat.plot_3mm_window(subject, f, ax)
        f.savefig(anat_map_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(f)
    return


def save_anat_maps_all_subjects(overwrite=False):
    # wis.pipes.exp_info.update_exp_info_spreadsheet()
    ei = wis.pipes.exp_info.load_exp_info_spreadsheet()
    for subject in ei["subject"].unique():
        try:
            _save_anat_map_basic(subject, overwrite=overwrite)
        except Exception as e:
            print(f"Error saving anatomy map for {subject}: {e}")
            continue
    return


def save_master_image(
    master_array: np.ndarray, out_path: str = "master_image.png"
) -> None:
    # Expect a 2D array (H, W)
    if master_array.ndim != 2:
        raise ValueError("master_array must be 2D (H, W)")

    # Convert to uint8 [0,255] without changing H×W
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
    # label_map: 2D int array (H, W); pixels are indices into id_list; -1 means unused
    # id_list:  1D array/list of strings; id_list[i] is the source-ID name

    if label_map.ndim != 2:
        raise ValueError("label_map must be 2D (H, W)")
    lm = np.asarray(label_map)
    if lm.dtype.kind not in ("i", "u"):
        raise ValueError("label_map must have an integer dtype")
    lm = lm.astype(np.int32, copy=False)  # ensure signed; -1 sentinel allowed

    ids = np.asarray(id_list, dtype=np.str_)  # saves as unicode (<U...) without pickle
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


def save_master_image_and_key(subject, exp, loc, acq, overwrite=False):
    exists = check_master_image_and_key_exists(
        os.path.join(annotation_root, subject, exp, loc, acq, "synapse_ids")
    )
    if exists and not overwrite:
        print("Master image and key already exist, skipping...")
        return
    synapse_ids_dir = os.path.join(
        annotation_root, subject, exp, loc, acq, "synapse_ids"
    )
    wis.util.gen.check_dir(synapse_ids_dir)

    mean_im = wis.scope.io.load_mean_ims(subject, exp, loc, acq)
    fp = wis.scope.io.load_fprts(subject, exp, loc, acq)
    for dmd in [1, 2]:
        syns = fp[dmd]
        maps = []
        for i in range(syns.shape[0]):
            syn = syns[i]
            syn[syn > 0] = int(i + 1)
            syn[syn <= 0] = int(-1)
            syn[np.isnan(syn)] = int(-1)
            maps.append(syn)
        maps = np.array(maps)
        maps.shape
        synmap = np.max(maps, axis=0)
        synmap[synmap == 0] = int(-1)
        synmap = synmap.astype(int)
        id_list = np.array([f"{i}" for i in range(syns.shape[0])])
        id_list = np.insert(
            id_list, 0, id_list[0]
        )  # add a zero at the start here to account for the fact that ID 0 is used, but we don't use it in the synmap, which is all non-zero integers
        master_image_array = mean_im[dmd][1]
        basedir = os.path.join(synapse_ids_dir, f"dmd-{dmd}")
        mip = os.path.join(basedir, "master_image.png")
        save_master_image(master_image_array, mip)
        save_source_location_key_npz(
            synmap, id_list, os.path.join(basedir, "source_location_key.npz")
        )
    return


def check_master_image_and_key_exists(basedir: str) -> bool:
    for dmd in [1, 2]:
        dmd_dir = os.path.join(basedir, f"dmd-{dmd}")
        mip = os.path.join(dmd_dir, "master_image.png")
        keyp = os.path.join(dmd_dir, "source_location_key.npz")
        if os.path.exists(mip) and os.path.exists(keyp):
            continue
        else:
            return False
    return True


def save_synapse_id_plots_and_key(
    subject, exp, loc, acq, upper_vmax_pct=95, buffer=25, channel=2, overwrite=False
):
    spy.plot.slap_style("im")
    mean_im = wis.scope.io.load_mean_ims(subject, exp, loc, acq)
    fp = wis.scope.io.load_fprts(subject, exp, loc, acq)

    for dmd in [1, 2]:
        save_dir = os.path.join(
            annotation_root, subject, exp, loc, acq, "synapse_ids", f"dmd-{dmd}"
        )
        wis.util.gen.check_dir(save_dir)
        dmd_nsources = fp[dmd].shape[0]
        for source in range(dmd_nsources):
            if not overwrite and os.path.exists(f"{save_dir}/{source}.png"):
                print(
                    f"Synapse ID plot for source {source} already exists, skipping..."
                )
                continue
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
    save_master_image_and_key(subject, exp, loc, acq, overwrite=overwrite)
    return


def generate_synapse_id_materials_all_subjects(overwrite=False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            locacqs = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
            for la in locacqs:
                loc, acq = la.split("--")
                print(f"Working on {subject} {exp} {loc} {acq}")
                try:
                    save_synapse_id_plots_and_key(
                        subject,
                        exp,
                        loc,
                        acq,
                        upper_vmax_pct=97,
                        buffer=25,
                        channel=2,
                        overwrite=overwrite,
                    )
                except Exception as e:
                    print(
                        f"Error generating synapse id materials for {subject} {exp} {loc} {acq}: {e}"
                    )
                    continue
    return


def create_annotation_basics(subject, exp, loc, acq):
    annotation_root = os.path.join(
        DEFS.anmat_root, "annotation_materials", subject, exp, loc, acq
    )
    wis.util.gen.check_dir(annotation_root)
    md_file = os.path.join(annotation_root, "notes.md")
    materials_path = os.path.join(
        annotation_root, "materials.txt"
    )  # signifier that this is an annotation materials folder

    if os.path.exists(materials_path):
        print(
            f"Materials file already exists: {materials_path}, use overwrite=True to overwrite"
        )
        pass
    else:
        with open(materials_path, "w") as f:
            f.write(f"# Annotation Materials -- {subject} {exp} {loc} {acq}\n")
            # save and close the file
            f.close()
    if os.path.exists(md_file):
        print(f"Notes file already exists: {md_file}, use overwrite=True to overwrite")
        pass
    else:
        with open(md_file, "w") as f:
            f.write(f"# Notes -- {subject} {exp} {loc} {acq}\n")
            # save and close the file
            f.close()


def _check_roi_location_tiffs_exist(subject, exp, loc, acq, dmd):
    roi_loc_dir = f"{DEFS.anmat_root}/annotation_materials/{subject}/{exp}/{loc}/{acq}/roi_locations"
    wis.util.gen.check_dir(roi_loc_dir)
    path = os.path.join(roi_loc_dir, f"roi_locs_dmd{dmd}.tif")
    return os.path.exists(path)


def save_roi_location_tiffs(subject, exp, loc, acq, overwrite=False):
    print(f"Saving ROI location tiffs for {subject} {exp} {loc} {acq}")
    mean_im = wis.scope.io.load_mean_ims(subject, exp, loc, acq)
    dmds = []
    if overwrite:
        dmds = [1, 2]
    else:
        dmds = [
            dmd
            for dmd in [1, 2]
            if not _check_roi_location_tiffs_exist(subject, exp, loc, acq, dmd)
        ]
    if len(dmds) == 0:
        print(f"All ROI location tiffs already exist for {subject} {exp} {loc} {acq}")
        return
    for dmd in dmds:
        image_to_save = mean_im[dmd][1]
        roi_loc_dir = f"{DEFS.anmat_root}/annotation_materials/{subject}/{exp}/{loc}/{acq}/roi_locations"
        path = os.path.join(roi_loc_dir, f"roi_locs_dmd{dmd}.tif")

        # Pick photometric automatically
        arr = np.ascontiguousarray(image_to_save)  # ensure C-contiguous memory

        # Pick photometric automatically
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
            # ome=True,  # optional: use OME-TIFF if you want explicit dimension metadata
        )
    return


def annotation_basics_all_subjects():
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            locacqs = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
            for la in locacqs:
                loc, acq = la.split("--")
                print(f"Working on {subject} {exp} {loc} {acq}")
                create_annotation_basics(subject, exp, loc, acq)
    return


def check_any_canvas_images_exist(subject, exp, loc, acq):
    canvas_path = os.path.join(annotation_root, subject, exp, loc, acq, "canvas")
    wis.util.gen.check_dir(canvas_path)
    for fname in os.listdir(canvas_path):
        if fname.endswith(".png"):
            return True
    return False


def full_annotation_pipeline(subject, exp, loc, acq):
    create_annotation_basics(subject, exp, loc, acq)
    if not check_any_canvas_images_exist(subject, exp, loc, acq):
        if (
            wis.util.info.determine_processing_done(subject, exp, loc, acq) == "YES"
        ):  # here we save the canvas images
            save_acq_mean_images(subject, exp, loc, acq, overwrite=False)
        else:
            save_single_trial_ref_images(subject, exp, loc, acq, overwrite=False)
    if wis.util.info.determine_processing_done(subject, exp, loc, acq) == "YES":
        save_synapse_id_plots_and_key(subject, exp, loc, acq, overwrite=False)
        save_roi_location_tiffs(subject, exp, loc, acq, overwrite=False)
    return


def full_annotation_pipeline_all_subjects():
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            locacqs = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
            for la in locacqs:
                loc, acq = la.split("--")
                print(f"Working on {subject} {exp} {loc} {acq}")
                full_annotation_pipeline(subject, exp, loc, acq)
    return
