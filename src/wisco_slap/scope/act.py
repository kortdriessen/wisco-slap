import polars as pl

import wisco_slap.defs as DEFS
import wisco_slap as wis


def load_roidf(subject, exp, loc, acq, roi_version="Fsvd", apply_ephys_offset=False):
    roidf_path = f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/roidf_{roi_version}.parquet"
    roidf = pl.read_parquet(roidf_path)
    roidf = roidf.with_columns(pl.lit(loc).alias("loc"), pl.lit(acq).alias("acq"))
    if apply_ephys_offset:
        si = wis.peri.sync.load_sync_info()
        ephys_offset = si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["ephys_offset"]
        roidf = roidf.with_columns((pl.col("time") + ephys_offset).alias("time"))
    return roidf


def load_syndf(
    subject,
    exp,
    loc,
    acq,
    trace_group="dF",
    trace_types=None,
    lazy=True,
    apply_ephys_offset=False,
):
    if trace_types is None:
        trace_types = ["matchFilt"]
    if trace_types is "all":
        trace_types = ["matchFilt", "denoised", "events", "nonneg"]
    syndf_path = f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/syndf_{trace_group}.parquet"
    if lazy:
        syndf = pl.scan_parquet(syndf_path)
        syndf = syndf.filter(pl.col("trace_type").is_in(trace_types))
        syndf = syndf.collect()
    else:
        syndf = pl.read_parquet(syndf_path)
        syndf = syndf.filter(pl.col("trace_type").is_in(trace_types))
    if apply_ephys_offset:
        si = wis.peri.sync.load_sync_info()
        ephys_offset = si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["ephys_offset"]
        syndf = syndf.with_columns((pl.col("time") + ephys_offset).alias("time"))
    syndf = syndf.with_columns(pl.lit(loc).alias("loc"), pl.lit(acq).alias("acq"))
    return syndf


def load_lsdf(subject, exp, loc, acq, trace_group="dF", apply_ephys_offset=False):
    lsdf_path = f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/lsdf_{trace_group}.parquet"
    lsdf = pl.read_parquet(lsdf_path)
    if apply_ephys_offset:
        si = wis.peri.sync.load_sync_info()
        ephys_offset = si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["ephys_offset"]
        lsdf = lsdf.with_columns((pl.col("time") + ephys_offset).alias("time"))
    return lsdf
