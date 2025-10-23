import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import slap2_py as spy

import wisco_slap as wis
import wisco_slap.defs as DEFS


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
    if wis.pipes.exp_info.determine_processing_done(subject, exp, loc, acq) == "NO":
        return
    esum_path = wis.util.io.sub_esum_path(subject, exp, loc, acq)
    if esum_path is None:
        return
    # rois = spy.hf.load_any(esum_path, "/exptSummary/userROIs")
    # Save each dmd meanIM
    mean_im_dir = f"{DEFS.anmat_root}/{subject}/{exp}/mean_IMs/{loc}/{acq}"
    wis.util.gen.check_dir(mean_im_dir)
    for dmd in [1, 2]:
        fname = f"{mean_im_dir}/DMD-{dmd}.png"
        if os.path.exists(fname) and not overwrite:
            print(
                f"DMD-{dmd} mean image already exists: {fname}, use overwrite=True to overwrite"
            )
            continue
        meanim = spy.hf.load_any(esum_path, f"/exptSummary/meanIM[{dmd - 1}][0]")
        img = meanim[1, :, :].T
        fh = img.shape[0] / 40
        fw = img.shape[1] / 40
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
    mean_im_dir = f"{DEFS.anmat_root}/{subject}/{exp}/mean_IMs/{loc}/{acq}"
    wis.util.gen.check_dir(mean_im_dir)
    for dmd in [1, 2]:
        for fname in os.listdir(data_dir):
            if f"DMD{dmd}-CYCLE-000000.tif" in fname:
                fig_name = f"{mean_im_dir}/DMD-{dmd}__TRIAL-000000.png"
                if os.path.exists(fig_name) and not overwrite:
                    print(
                        f"DMD-{dmd} mean image (single trial) already exists: {fig_name}, use overwrite=True to overwrite"
                    )
                    continue
                tiff_path = f"{data_dir}/{fname}"
                img = spy.img.io.load_tiff(tiff_path)
                img = np.mean(img, axis=0)
                fh = img.shape[0] / 40
                fw = img.shape[1] / 40
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
                        save_single_trial_ref_images(
                            subject, exp, loc, acq, overwrite=False, vmin=5, vmax=85
                        )
                    else:
                        continue
    return


# save_acq_synapse_id_plots(subject, exp, loc, acq):
