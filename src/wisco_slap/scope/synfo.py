import os

import numpy as np
import pandas as pd
import polars as pl

from wisco_slap.defs import anmat_root

from ._syn_sorting_utils import sort_synapses_prox_to_dist


def load_synapse_map(subject, exp, loc, acq, dmd, exact_values=False):
    path = os.path.join(
        anmat_root,
        "annotation_materials",
        subject,
        exp,
        loc,
        acq,
        "synapse_ids",
        f"dmd-{dmd}",
        "source_location_key.npz",
    )
    with np.load(path) as data:
        label_map = data["label_map"]
        id_list = data["id_list"]
    if exact_values:
        label_map[label_map > 0] = label_map[label_map > 0] - 1
    return label_map, id_list


def get_syn_orders(
    idf: pl.DataFrame, subject: str, exp: str, loc: str, acq: str
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute and save synapse orders for a given subject, experiment, location, and acquisition.
    """
    final_orders = {}
    for dmd in [1, 2]:
        dend_ids = np.sort(
            idf.filter(pl.col("dmd") == dmd)["dend-ID"].unique().drop_nulls().to_numpy()
        )
        topo_root = os.path.join(
            anmat_root,
            "annotation_materials",
            subject,
            exp,
            loc,
            acq,
            "source_sorting",
        )
        dmp = os.path.join(topo_root, f"prox_lines_dmd{dmd}.csv")
        if not os.path.exists(dmp):
            raise FileNotFoundError(f"Proximal lines file not found: {dmp}")
        dlines = pd.read_csv(dmp)

        synmap = load_synapse_map(subject, exp, loc, acq, dmd, exact_values=True)
        synmap = synmap[0]

        dend_info = {}
        for ix, dend in enumerate(dend_ids):
            dend_info[dend] = {}
            source_ids = (
                idf.filter(pl.col("dend-ID") == dend)["syn_id"].unique().to_numpy()
            )
            denddf = dlines.loc[dlines["index"] == ix]
            x = denddf["axis-1"].values
            y = denddf["axis-0"].values
            dend_info[dend]["verts"] = [(x[i], y[i]) for i in range(len(x))]
            dend_info[dend]["source_ids"] = source_ids

        syn_orders = sort_synapses_prox_to_dist(dend_info, synmap)
        # syn_orders_serializable = {k: v.tolist() for k, v in syn_orders.items()}
        final_orders[f"dmd_{dmd}"] = syn_orders
    return final_orders


def merge_syn_orders_to_infodf(
    idf: pl.DataFrame,
    syn_orders: dict[str, dict[str, np.ndarray]],
) -> pl.DataFrame:
    """Add a ``pos`` column to *idf* giving each synapse's proximal-to-distal position.

    Parameters
    ----------
    idf : pl.DataFrame
        Synapse info DataFrame (from ``wis.get.synid_labels``).
    syn_orders : dict
        ``{'dmd_1': {dend_id: array_of_syn_ids}, 'dmd_2': ...}``
        as returned by ``get_syn_orders``.

    Returns
    -------
    pl.DataFrame
        *idf* with an additional ``pos`` column (0 = most proximal).
    """
    rows: list[dict] = []
    for dmd in [1, 2]:
        for dend_id, syn_ids in syn_orders[f"dmd_{dmd}"].items():
            for pos, sid in enumerate(syn_ids):
                rows.append({"syn_id": int(sid), "dmd": dmd, "pos": pos})
    order_df = pl.DataFrame(rows).cast({
        "syn_id": pl.Int32,
        "dmd": pl.Int64,
        "pos": pl.Int32,
    })
    return idf.join(order_df, on=["syn_id", "dmd"], how="left")
