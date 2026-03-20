import numpy as np


def _centroids_from_synmap(synmap: np.ndarray) -> np.ndarray:
    """
    Returns centroids as an array of shape (max_id+1, 2) in (x, y) coords.
    Entries for IDs not present in synmap are NaN.
    """
    if synmap.ndim != 2:
        raise ValueError("synmap must be a 2D array (H, W).")

    mask = synmap >= 0
    if not np.any(mask):
        # no synapses at all
        return np.full((0, 2), np.nan, dtype=np.float64)

    ys, xs = np.nonzero(mask)
    ids = synmap[mask].astype(np.int64, copy=False)
    max_id = int(ids.max())

    counts = np.bincount(ids, minlength=max_id + 1).astype(np.float64, copy=False)
    sum_x = np.bincount(ids, weights=xs, minlength=max_id + 1).astype(
        np.float64, copy=False
    )
    sum_y = np.bincount(ids, weights=ys, minlength=max_id + 1).astype(
        np.float64, copy=False
    )

    cx = np.full(max_id + 1, np.nan, dtype=np.float64)
    cy = np.full(max_id + 1, np.nan, dtype=np.float64)

    present = counts > 0
    cx[present] = sum_x[present] / counts[present]
    cy[present] = sum_y[present] / counts[present]

    return np.stack([cx, cy], axis=1)  # (id, [x,y])


def _arc_length_along_polyline(
    points_xy: np.ndarray, poly_xy: np.ndarray
) -> np.ndarray:
    """
    For each point (x,y), project onto polyline and return arc-length coordinate s (pixels)
    measured from the first vertex (proximal end).

    points_xy: (K,2)
    poly_xy:   (M,2) with M>=2, ordered proximal->distal
    """
    if poly_xy.shape[0] < 2:
        raise ValueError("Polyline must have at least 2 vertices.")

    P = points_xy.astype(np.float64, copy=False)  # (K,2)
    V = poly_xy.astype(np.float64, copy=False)  # (M,2)

    A = V[:-1]  # (S,2)
    B = V[1:]  # (S,2)
    AB = B - A  # (S,2)
    AB2 = (AB * AB).sum(axis=1)  # (S,)
    AB2 = np.where(AB2 == 0.0, 1e-12, AB2)  # avoid /0
    seg_len = np.sqrt(AB2)  # (S,)

    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])  # (M,)

    # Broadcast projection
    # AP: (K,S,2)
    AP = P[:, None, :] - A[None, :, :]
    t = (AP * AB[None, :, :]).sum(axis=2) / AB2[None, :]  # (K,S)
    t = np.clip(t, 0.0, 1.0)

    proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]  # (K,S,2)
    d2 = ((proj - P[:, None, :]) ** 2).sum(axis=2)  # (K,S)

    seg_idx = np.argmin(d2, axis=1)  # (K,)
    k_idx = np.arange(P.shape[0])

    t_best = t[k_idx, seg_idx]
    s = cumlen[seg_idx] + t_best * seg_len[seg_idx]  # (K,)

    return s


def sort_synapses_prox_to_dist(
    dend_info: dict,
    synmap: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Parameters
    ----------
    dend_info:
      dict[dend_id] -> {"verts": [(x,y), ...], "source_ids": np.ndarray[int]}
      verts must be ordered proximal->distal.

    synmap:
      2D ndarray where -1 means non-synapse, and >=0 is source_id.

    Returns
    -------
    syn_orders:
      dict[dend_id] -> np.ndarray[int] source_ids sorted proximal->distal.
    """
    centroids_xy = _centroids_from_synmap(synmap)

    syn_orders: dict[str, np.ndarray] = {}

    for dend_id, info in dend_info.items():
        verts = np.asarray(info["verts"], dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError(f"{dend_id}: verts must be an (M,2) array/list of (x,y).")

        src = np.asarray(info["source_ids"], dtype=np.int64)
        if src.size == 0:
            syn_orders[dend_id] = src.copy()
            continue

        # Grab centroids for just these source IDs
        if centroids_xy.shape[0] == 0:
            # no synapses in map at all
            syn_orders[dend_id] = src.copy()
            continue

        # IDs that exceed centroid table are missing from synmap
        pts = np.full((src.size, 2), np.nan, dtype=np.float64)
        in_bounds = (src >= 0) & (src < centroids_xy.shape[0])
        pts[in_bounds] = centroids_xy[src[in_bounds]]

        # Arc-length coordinate along dendrite polyline
        s = _arc_length_along_polyline(pts, verts)

        # Push any NaN points (missing synapses) to the end deterministically
        s = np.where(np.isfinite(s), s, np.inf)

        order = np.argsort(s, kind="mergesort")  # stable sort
        syn_orders[dend_id] = src[order]

    return syn_orders
