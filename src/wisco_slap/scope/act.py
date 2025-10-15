import polars as pl

import wisco_slap.defs as DEFS


def load_roidf(subject, exp, loc, acq, roi_version="Fsvd"):
    roidf_path = f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/roidf_{roi_version}.parquet"
    roidf = pl.read_parquet(roidf_path)
    return roidf


def load_syndf(subject, exp, loc, acq, trace_group="dF", trace_types=None, lazy=True):
    if trace_types is None:
        trace_types = ["matchFilt"]
    syndf_path = f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/syndf_{trace_group}.parquet"
    if lazy:
        syndf = pl.scan_parquet(syndf_path)
        syndf = syndf.filter(pl.col("trace_type").is_in(trace_types))
        syndf = syndf.collect()
    else:
        syndf = pl.read_parquet(syndf_path)
        syndf = syndf.filter(pl.col("trace_type").is_in(trace_types))
    return syndf


def load_lsdf(subject, exp, loc, acq, trace_group="dF"):
    lsdf_path = f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/lsdf_{trace_group}.parquet"
    lsdf = pl.read_parquet(lsdf_path)
    return lsdf
