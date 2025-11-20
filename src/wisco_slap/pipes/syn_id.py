import wisco_slap as wis
import slap2_py as spy
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

import wisco_slap.defs as DEFS


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


import numpy as np


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
        f"{DEFS.anmat_root}/{subject}/{exp}/synapse_ids/{loc}/{acq}"
    )
    if exists and not overwrite:
        print("Master image and key already exist, skipping...")
        return

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
        basedir = f"{DEFS.anmat_root}/{subject}/{exp}/synapse_ids/{loc}/{acq}/dmd-{dmd}"
        mip = f"{basedir}/master_image.png"
        save_master_image(master_image_array, mip)
        save_source_location_key_npz(
            synmap, id_list, f"{basedir}/source_location_key.npz"
        )
    return


def check_master_image_and_key_exists(basedir: str) -> bool:
    for dmd in [1, 2]:
        dmd_dir = f"{basedir}/dmd-{dmd}"
        mip = f"{dmd_dir}/master_image.png"
        keyp = f"{dmd_dir}/source_location_key.npz"
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
        save_dir = (
            f"{DEFS.anmat_root}/{subject}/{exp}/synapse_ids/{loc}/{acq}/dmd-{dmd}"
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
