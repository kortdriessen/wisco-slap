from pathlib import Path

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Rectangle
import numpy as np
import wisco_slap.defs as DEFS
import wisco_slap as wis
import polars as pl


def _find_cell_center_pixels(roi_map: np.ndarray):
    """
    Compute a representative "center pixel" for each unique non-zero label in a 2D ROI map.

    This PRESERVES the original label values in `roi_map`. For each label value L>0,
    it returns the in-region pixel nearest to that label's center-of-mass.

    Parameters
    ----------
    roi_map : np.ndarray
        2D array. Non-zero integer values indicate distinct cell labels.

    Returns
    -------
    centers_by_label : dict[int, tuple[int, int]]
        Mapping from original label value (e.g., 1..N) to (row, col) integer coordinates.
    labeled : np.ndarray
        The original `roi_map` returned unchanged so labels correspond exactly.
    """
    if roi_map.ndim != 2:
        raise ValueError("roi_map must be a 2D array")

    # Get all unique non-zero labels and compute one center per label value
    labels = np.unique(roi_map)
    labels = labels[labels != 0]

    centers_by_label: dict[int, tuple[int, int]] = {}

    for label_value in labels:
        # Coordinates of pixels with this label as (row, col)
        coords = np.column_stack(np.nonzero(roi_map == label_value))
        if coords.size == 0:
            continue

        # Center-of-mass in continuous coordinates (row, col)
        com = coords.mean(axis=0)

        # Choose the in-label pixel nearest to COM
        deltas = coords - com
        distances2 = (deltas**2).sum(axis=1)
        nearest_idx = int(np.argmin(distances2))
        r, c = coords[nearest_idx]
        centers_by_label[int(label_value)] = (int(r), int(c))

    # Return centers and the original label map unchanged
    return centers_by_label, roi_map


def find_soma_centers(roi_map: dict[int, np.ndarray]):
    soma_centers = {}
    for dmd in roi_map:
        if roi_map[dmd] is None:
            soma_centers[dmd] = None
            continue
        soma_centers[dmd], _ = _find_cell_center_pixels(roi_map[dmd])
    return soma_centers


def find_synapse_pixels(footprints: dict[int, np.ndarray]):
    fp_coords = {}

    for dmd in footprints:
        fp_coords[dmd] = {}
        ftprt = footprints[dmd]
        for source in range(ftprt.shape[0]):
            f = ftprt[source]
            # replace all values in f with -1 if they are nan
            f[np.isnan(f)] = -1
            # find the coordinates of the maximum value in f
            max_idx = np.unravel_index(np.argmax(f), f.shape)
            fp_coords[dmd][source] = max_idx
    return fp_coords


def get_all_coords(subject, exp, loc, acq):
    di = wis.util.info.load_dmd_info()
    dia = di[subject][exp][loc][acq]
    fp = wis.scope.io.load_fprts(subject, exp, loc, acq)
    fullrm = wis.scope.io.load_dual_roi_map(subject, exp, loc, acq)
    roi_coords = find_soma_centers(fullrm)
    fp_coords = find_synapse_pixels(fp)
    rc_named = {}
    for dmd in [1, 2]:
        rc_named[dmd] = {}
        fpc = fp_coords[dmd]
        z = dia["dmd-" + str(dmd)]["depth"]
        rc = roi_coords[dmd]
        if rc is not None:
            for roi_id in rc:
                roi_name = "soma" + str(roi_id)
                roi_center = rc[roi_id]
                roi_center = np.append(roi_center, z)
                rc_named[dmd][roi_name] = roi_center
        if rc is None:
            rc_named[dmd] = None
        for source in fpc:
            coords = fpc[source]
            # append z to the coords
            coords = np.append(coords, z)
            fp_coords[dmd][source] = coords
    return rc_named, fp_coords


def synapse_vectors(
    soma_yxz,
    synapses,
    *,
    voxel_size_yxz=None,  # e.g., (dy_um_per_px, dx_um_per_px, dz_um_per_step)
    image_shape=None,  # optional (H, W), for simple bounds checks on Y,X
    ids=None,  # optional list of IDs if `synapses` is an ndarray
):
    """
    Compute per-synapse vectors from the soma to each synapse.

    Parameters
    ----------
    soma_yxz : tuple/list/array of length 3
        Soma coordinates in (Y, X, Z).
    synapses : dict or np.ndarray
        If dict: {syn_id: (Y, X, Z), ...}
        If ndarray: shape (N, 3) in (Y, X, Z) order. Provide `ids` to label rows.
    voxel_size_yxz : tuple of length 3, optional
        Physical size of one voxel step along (Y, X, Z). If provided, returns
        vectors/distances both in pixels/steps and in physical units.
    image_shape : tuple (H, W), optional
        If provided, performs simple bounds checks on Y,X (not Z).
    ids : list/array of length N, optional
        IDs corresponding to rows of `synapses` when `synapses` is an ndarray.

    Returns
    -------
    out : dict
        {
          "ids": np.ndarray shape (N,),
          "vectors_px": np.ndarray shape (N, 3)  # (ΔY, ΔX, ΔZ) = synapse - soma
          "dist_px":    np.ndarray shape (N,),   # Euclidean distance in voxel steps
          # present only if voxel_size_yxz is given:
          "vectors_phys": np.ndarray shape (N, 3),
          "dist_phys":    np.ndarray shape (N,),
        }
        All arrays are float64.
    """
    soma = np.asarray(soma_yxz, dtype=np.float64).reshape(1, 3)
    if soma.shape != (1, 3):
        raise ValueError("soma_yxz must have 3 elements: (Y, X, Z).")

    # Normalize synapse input to array + ids
    if isinstance(synapses, dict):
        syn_ids = np.asarray(list(synapses.keys()))
        syn_arr = np.asarray(list(synapses.values()), dtype=np.float64)
        if syn_arr.ndim != 2 or syn_arr.shape[1] != 3:
            print(syn_arr.shape)
            raise ValueError("Dict values must be (Y, X, Z) triplets.")
    else:
        syn_arr = np.asarray(synapses, dtype=np.float64)
        if syn_arr.ndim != 2 or syn_arr.shape[1] != 3:
            raise ValueError("`synapses` ndarray must have shape (N, 3) in (Y, X, Z).")
        syn_ids = np.arange(syn_arr.shape[0]) if ids is None else np.asarray(ids)
        if syn_ids.shape[0] != syn_arr.shape[0]:
            raise ValueError("`ids` length must match number of synapses.")

    # Optional simple bounds checks for Y,X
    if image_shape is not None:
        H, W = image_shape
        y_ok = (syn_arr[:, 0] >= 0) & (syn_arr[:, 0] < H)
        x_ok = (syn_arr[:, 1] >= 0) & (syn_arr[:, 1] < W)
        if not (np.all(y_ok) and np.all(x_ok)):
            bad = syn_ids[~(y_ok & x_ok)]
            raise ValueError(f"Some synapses fall outside image bounds: {bad}")

    # Vector from soma to synapse (ΔY, ΔX, ΔZ)
    vectors_px = syn_arr - soma  # broadcast soma (1,3) to (N,3)

    # Euclidean distance in voxel steps (treats Z as given—e.g., plane index or µm if already scaled)
    dist_px = np.linalg.norm(vectors_px, axis=1)

    out = {
        "ids": syn_ids,
        "vectors_px": vectors_px.astype(np.float64, copy=False),
        "dist_px": dist_px.astype(np.float64, copy=False),
    }

    # Optional: convert to physical units
    if voxel_size_yxz is not None:
        vs = np.asarray(voxel_size_yxz, dtype=np.float64)
        if vs.shape != (3,):
            raise ValueError("voxel_size_yxz must be a 3-tuple (dY, dX, dZ).")
        vectors_phys = vectors_px * vs  # elementwise scale per axis
        dist_phys = np.linalg.norm(vectors_phys, axis=1)
        out["vectors_phys"] = vectors_phys
        out["dist_phys"] = dist_phys

    return out


def get_real_somas(subject, exp, loc, acq):

    di = wis.util.info.load_dmd_info()
    dia = di[subject][exp][loc][acq]
    real_somas = {}
    for dmd in dia.keys():
        if len(dia[dmd]["somas"]) > 0:
            for soma in dia[dmd]["somas"]:
                real_somas[soma] = dmd
    return real_somas


# TODO: probably a better spot for this
def _map_vectors_to_synid_labels(real_somas, roi_coords, fp_coords, idf):
    vec_dfs = []
    for soma in real_somas.keys():
        soma_dmd = real_somas[soma]
        dmd_num = int(soma_dmd.split("-")[-1])
        if soma not in roi_coords[dmd_num]:
            continue
        soma_coords = roi_coords[dmd_num][soma]
        for dmd in [1, 2]:
            sdis = (
                idf.filter(pl.col("soma-ID") == soma)
                .filter(pl.col("dmd") == dmd)["source"]
                .to_list()
            )

            if (
                len(sdis) == 0
            ):  # check that this soma actually has synapses on this dmd!
                continue
            fps_dis = {k: v for k, v in fp_coords[dmd].items() if k in sdis}
            vecs = wis.scope.anat.synapse_vectors(soma_coords, fps_dis)
            vecdf = pl.DataFrame(vecs)
            vecdf = vecdf.with_columns(
                pl.lit(soma).alias("soma-ID"), pl.lit(dmd).alias("dmd")
            )
            vec_dfs.append(vecdf)
    vecdf = pl.concat(vec_dfs)
    vecdf = vecdf[["ids", "dist_px", "soma-ID", "dmd"]]
    vecdf = vecdf.rename({"ids": "source", "dist_px": "soma_distance"})
    vecdf = vecdf.with_columns(
        pl.col("source").cast(pl.Int32), pl.col("dmd").cast(pl.Int64)
    )
    idf = idf.join(vecdf, on=["source", "dmd", "soma-ID"], how="left")
    idf = idf.with_columns(
        pl.lit(idf["soma_distance"].fill_null(-1)).alias("soma_distance")
    )
    return idf


def map_vectors_to_synid_labels(subject, exp, loc, acq):
    real_somas = wis.scope.anat.get_real_somas(subject, exp, loc, acq)
    roi_coords, fp_coords = wis.scope.anat.get_all_coords(subject, exp, loc, acq)
    idf = wis.scope.io.load_synid_labels(subject, exp, loc, acq)
    return (
        _map_vectors_to_synid_labels(real_somas, roi_coords, fp_coords, idf),
        real_somas,
        roi_coords,
    )
