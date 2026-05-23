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
        pl
        .col("dend-ID")
        .map_elements(_normalize_dend_id, return_dtype=pl.Utf8)
        .alias("dend-ID")
    )


def _add_dend_type_label_and_layer_to_syn_info(syn_info: pl.DataFrame) -> pl.DataFrame:
    """Add `soma_layer` and `dend_type` columns to `syn_info`.

    `soma_layer` is binned from `soma-depth` (um):
      L1 (<100), L2/3 (100-300), L4 (300-500), L5 (500-800), else 'null'.

    `dend_type` is computed from `soma-depth`, `soma_layer`, and the dendrite's
    imaging-plane depth (`dmd-depth`):
      - L2/3, L4, L5 soma:
          dmd-depth >= soma-depth - 10  -> 'basal'        (within 10um of soma plane,
                                                           or anywhere deeper)
          dmd-depth < 100               -> 'apical'
          100 <= dmd-depth < soma-depth - 10 -> 'intermediate'
      - L1 soma:
          (soma-depth - dmd-depth) > 50 -> 'apical'
          otherwise                     -> 'basal'
      - Synapses without a `dend-ID` or with an unclassifiable soma layer are 'null'.
    """
    syn_info = syn_info.with_columns(
        pl
        .when(pl.col("soma-depth") < 100)
        .then(pl.lit("L1"))
        .when((pl.col("soma-depth") >= 100) & (pl.col("soma-depth") < 300))
        .then(pl.lit("L2/3"))
        .when((pl.col("soma-depth") >= 300) & (pl.col("soma-depth") < 500))
        .then(pl.lit("L4"))
        .when((pl.col("soma-depth") >= 500) & (pl.col("soma-depth") < 800))
        .then(pl.lit("L5"))
        .otherwise(pl.lit("null"))
        .alias("soma_layer")
    )

    deep_layers = ["L2/3", "L4", "L5"]
    syn_info = syn_info.with_columns(
        pl
        .when(pl.col("dend-ID").is_null())
        .then(pl.lit("null"))
        .when(
            pl.col("soma_layer").is_in(deep_layers)
            & (pl.col("dmd-depth") >= pl.col("soma-depth") - 10)
        )
        .then(pl.lit("basal"))
        .when(pl.col("soma_layer").is_in(deep_layers) & (pl.col("dmd-depth") < 100))
        .then(pl.lit("apical"))
        .when(
            pl.col("soma_layer").is_in(deep_layers)
            & (pl.col("dmd-depth") < pl.col("soma-depth"))
        )
        .then(pl.lit("intermediate"))
        .when(
            (pl.col("soma_layer") == "L1")
            & ((pl.col("soma-depth") - pl.col("dmd-depth")) > 50)
        )
        .then(pl.lit("apical"))
        .when(pl.col("soma_layer") == "L1")
        .then(pl.lit("basal"))
        .otherwise(pl.lit("null"))
        .alias("dend_type")
    )

    return syn_info


def synid_labels(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    null_to_strings: bool = False,
    add_layer_and_dendtype: bool = True,
) -> pl.DataFrame | None:
    """Load synapse identity/annotation labels for an acquisition.

    Reads ``synapse_labels.csv`` from both DMD annotation directories,
    normalizes dendrite IDs, and enriches with depth and soma metadata.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    null_to_strings : bool, default False
        If True, every null/None value in the returned DataFrame is replaced
        with the string ``"null"``. Columns that contain nulls are cast to
        ``pl.Utf8`` so the fill is type-compatible; columns with no nulls
        keep their original dtype. Guarantees no empty cells in the result.
    add_layer_and_dendtype : bool, default True
        If True, adds `soma_layer` and `dend_type` columns to the result using
        the `_add_dend_type_label_and_layer_to_syn_info` function.

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
        pl
        .when(pl.col("dmd") == 1)
        .then(pl.lit(dia["dmd-1"]["depth"]))
        .otherwise(pl.col("dmd-depth"))
        .alias("dmd-depth")
    )
    idf = idf.with_columns(
        pl
        .when(pl.col("dmd") == 2)
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
        idf
        .join(soma_df, on="soma-ID", how="left", suffix="_new")
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
        subject,
        exp,
        loc,
        acq,
        "source_sorting",
    )
    has_prox_lines = all(
        os.path.exists(os.path.join(topo_root, f"prox_lines_dmd{dmd}.csv"))
        for dmd in [1, 2]
    )
    if has_prox_lines:
        from wisco_slap.scope.synfo import get_syn_orders, merge_syn_orders_to_infodf

        syn_orders = get_syn_orders(idf, subject, exp, loc, acq)
        idf = merge_syn_orders_to_infodf(idf, syn_orders)

    if add_layer_and_dendtype:
        idf = _add_dend_type_label_and_layer_to_syn_info(idf)

    if null_to_strings:
        null_cols = [c for c in idf.columns if idf[c].null_count() > 0]
        if null_cols:
            idf = idf.with_columns([
                pl.col(c).cast(pl.Utf8).fill_null("null") for c in null_cols
            ])

    return idf
