from __future__ import annotations

import numpy as np
import wisco_slap as wis
import polars as pl
import pandas as pd
import os
import json


def save_syn_orders(
    subject: str, exp: str, loc: str, acq: str
) -> dict[str, np.ndarray]:
    """
    Compute and save synapse orders for a given subject, experiment, location, and acquisition.
    """
    idf = wis.get.synid_labels(subject, exp, loc, acq)
    for dmd in [1, 2]:
        dend_ids = np.sort(
            idf.filter(pl.col("dmd") == dmd)["dend-ID"].unique().drop_nulls().to_numpy()
        )
        topo_root = os.path.join(
            wis.defs.anmat_root,
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

        synmap = wis.scope.io.load_synapse_map(
            subject, exp, loc, acq, dmd, exact_values=True
        )
        synmap = synmap[0]

        dend_info = {}
        for ix, dend in enumerate(dend_ids):
            dend_info[dend] = {}
            source_ids = (
                idf.filter(pl.col("dend-ID") == dend)["source-ID"].unique().to_numpy()
            )
            denddf = dlines.loc[dlines["index"] == ix]
            x = denddf["axis-1"].values
            y = denddf["axis-0"].values
            dend_info[dend]["verts"] = [(x[i], y[i]) for i in range(len(x))]
            dend_info[dend]["source_ids"] = source_ids

        syn_orders = sort_synapses_prox_to_dist(dend_info, synmap)
        syn_orders_serializable = {k: v.tolist() for k, v in syn_orders.items()}
        save_path = os.path.join(topo_root, f"syn_topo_dmd{dmd}.json")
        with open(save_path, "w") as f:
            json.dump(syn_orders_serializable, f)

    return


def load_syn_orders(
    subject: str, exp: str, loc: str, acq: str
) -> dict[str, np.ndarray]:
    """
    Load synapse orders for a given subject, experiment, location, and acquisition.
    """
    topo_root = os.path.join(
        wis.defs.anmat_root,
        "annotation_materials",
        subject,
        exp,
        loc,
        acq,
        "source_sorting",
    )
    syn_orders = {}
    for dmd in [1, 2]:
        save_path = os.path.join(topo_root, f"syn_topo_dmd{dmd}.json")
        with open(save_path, "r") as f:
            syor = json.load(f)
            for key in syor.keys():
                syn_orders[key] = syor[key]
    return syn_orders
