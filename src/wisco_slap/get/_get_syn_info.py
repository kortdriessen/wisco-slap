"""Loading functions for synapse annotation / identity data."""

import os
import re

import pandas as pd
import polars as pl

import wisco_slap as wis
import wisco_slap.defs as DEFS


def _normalize_dend_id(dend_id: str) -> str:
    """Normalize dendrite ID to the correct format (e.g., 'B-1').

    Handles common typos like 'B1', 'b1', 'b-1' and converts them to 'B-1'.

    Parameters
    ----------
    dend_id : str
        The dendrite ID string to normalize.

    Returns
    -------
    str
        The normalized dendrite ID, or the original value if it doesn't
        match the expected pattern (e.g., 'unlabelled').
    """
    if dend_id is None:
        return dend_id

    # Pattern: one letter, optional hyphen, one or more digits
    pattern = r"^([A-Za-z])-?(\d+)$"
    match = re.match(pattern, dend_id.strip())

    if match:
        letter = match.group(1).upper()
        number = match.group(2)
        return f"{letter}-{number}"

    return dend_id


def _normalize_dend_id_column(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize the 'dend-ID' column in a DataFrame.

    Corrects common typos in dendrite IDs:
    - 'B1' -> 'B-1'
    - 'b1' -> 'B-1'
    - 'b-1' -> 'B-1'

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with a 'dend-ID' column.

    Returns
    -------
    pl.DataFrame
        DataFrame with normalized 'dend-ID' values.
    """
    if "dend-ID" not in df.columns:
        return df

    return df.with_columns(
        pl.col("dend-ID")
        .map_elements(_normalize_dend_id, return_dtype=pl.Utf8)
        .alias("dend-ID")
    )


def synid_labels(
    subject: str, exp: str, loc: str, acq: str
) -> pl.DataFrame | None:
    """Load synapse identity/annotation labels for an acquisition.

    Reads ``synapse_labels.csv`` from both DMD annotation directories,
    normalizes dendrite IDs, and enriches with depth and soma metadata.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.

    Returns
    -------
    pl.DataFrame or None
        DataFrame with columns ``syn_id``, ``synapse-type``, ``soma-ID``,
        ``soma-depth``, ``dend-type``, ``dend-ID``, ``notes``, ``dmd``,
        ``dmd-depth``.  Returns ``None`` if the source CSV is missing.
    """
    all_dfs = []
    for dmd in [1, 2]:
        path = os.path.join(
            DEFS.anmat_root,
            "annotation_materials",
            subject,
            exp,
            loc,
            acq,
            "synapse_ids",
            f"dmd-{dmd}",
            "synapse_labels.csv",
        )
        if not os.path.exists(path):
            print(f"{path} does not exist!")
            return None
        df = pd.read_csv(path)
        df["dmd"] = dmd
        all_dfs.append(df)
    idf = pl.from_pandas(pd.concat(all_dfs))
    if "dend-ID" not in idf.columns:
        idf = idf.with_columns(pl.lit("unlabelled").alias("dend-ID"))
    idf = _normalize_dend_id_column(idf)
    idf = idf.filter(pl.col("source-ID") != "master_image")
    idf = idf.with_columns(pl.col("source-ID").cast(pl.Int32).alias("source-ID"))

    di = wis.meta.get.dmd_info()
    dia = di[subject][exp][loc][acq]
    idf = idf.with_columns(pl.lit(-1).alias("dmd-depth"))
    idf = idf.with_columns(
        pl.when(pl.col("dmd") == 1)
        .then(pl.lit(dia["dmd-1"]["depth"]))
        .otherwise(pl.col("dmd-depth"))
        .alias("dmd-depth")
    )
    idf = idf.with_columns(
        pl.when(pl.col("dmd") == 2)
        .then(pl.lit(dia["dmd-2"]["depth"]))
        .otherwise(pl.col("dmd-depth"))
        .alias("dmd-depth")
    )

    soma_dfs = []
    for dmd in [1, 2]:
        depth = dia[f"dmd-{dmd}"]["depth"]
        somas = dia[f"dmd-{dmd}"]["somas"]
        if len(somas) > 0:
            for soma in somas:
                soma_df = pl.DataFrame({"soma-ID": [soma], "soma-depth": [depth]})
                soma_dfs.append(soma_df)
    soma_df = pl.concat(soma_dfs)

    idf = (
        idf.join(soma_df, on="soma-ID", how="left", suffix="_new")
        .with_columns(
            pl.coalesce([pl.col("soma-depth_new"), pl.col("soma-depth")]).alias(
                "soma-depth"
            )
        )
        .drop("soma-depth_new")
    )
    idf = idf.rename({"source-ID": "syn_id"})

    # Merge proximal-to-distal position if ordering data exists
    topo_root = os.path.join(
        DEFS.anmat_root,
        "annotation_materials",
        subject, exp, loc, acq, "source_sorting",
    )
    has_prox_lines = all(
        os.path.exists(os.path.join(topo_root, f"prox_lines_dmd{dmd}.csv"))
        for dmd in [1, 2]
    )
    if has_prox_lines:
        from wisco_slap.scope.synfo import get_syn_orders, merge_syn_orders_to_infodf

        syn_orders = get_syn_orders(idf, subject, exp, loc, acq)
        idf = merge_syn_orders_to_infodf(idf, syn_orders)

    return idf
