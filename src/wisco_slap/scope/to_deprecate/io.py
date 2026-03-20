import os

import numpy as np
import pandas as pd
import polars as pl
import slap2_py as spy
import tifffile as tiff

import wisco_slap as wis
import wisco_slap.defs as DEFS

from .DF_Classes import SomaDF, SynDF


def load_syndf(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    trace_group: str = "dF",
    trace_types: list[str] | str = None,
    apply_ephys_offset: bool = False,
    nan_to_null: bool = True,
    merge_labels: bool = True,
    adjust_noise_column: bool = True,
    noise_thresh: float | int | None = False,
    SDF: bool = False,
) -> pl.DataFrame:
    """load the syndf (dF or dFF) dataframe generated from the ExperimentSummary

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    loc : str
        location name
    acq : str
        acquisition name
    trace_group : str, optional
        trace group to load, by default "dF" (dF or dFF)
    trace_types : list[str] | str | None, optional
        trace types to load, by default None ('all' loads all trace types)
    apply_ephys_offset : bool, optional
        whether to apply the ephys offset to the time column, by default False
    nan_to_null: bool, optional
        whether to convert nan values to null values, by default True
    merge_labels: bool, optional
        whether to merge the synapse labels, by default True
    adjust_noise_column: bool, optional
        whether to adjust the noise column, by default True
    noise_thresh: float | int | None, optional
        whether to apply a multiple the noise column, and assign that
        to the 'noise_threshold' column; if not None, this is used as the multiple
    SDF: bool, optional
        whether to return a SynDF object, by default False
    Returns
    -------
    pl.DataFrame | SynDF
        the syndf dataframe
    """
    if trace_types is None:
        trace_types = ["matchFilt"]
    if trace_types == "all":
        trace_types = ["matchFilt", "denoised", "events", "nonneg"]
    syndf_path = f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/syn_{trace_group}.parquet"
    syndf = pl.scan_parquet(syndf_path)
    syndf = syndf.filter(pl.col("trace_type").is_in(trace_types))
    syndf = syndf.collect()

    if apply_ephys_offset:
        si = wis.meta.get.sync_info()
        ephys_offset = si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["ephys_offset"]
        syndf = syndf.with_columns((pl.col("time") + ephys_offset).alias("time"))

    syndf = syndf.with_columns(pl.lit(loc).alias("loc"), pl.lit(acq).alias("acq"))

    if nan_to_null:
        syndf = syndf.with_columns(
            pl
            .when(pl.col("data").is_nan())
            .then(pl.lit(None))
            .otherwise(pl.col("data"))
            .alias("data")
        )
    if merge_labels:
        idf = wis.get.synid_labels(subject, exp, loc, acq)
        syndf = syndf.join(
            idf, left_on=["dmd", "source-ID"], right_on=["dmd", "syn_id"], how="left"
        )
    if adjust_noise_column:
        syndf = syndf.with_columns((pl.col("noise").sqrt()).alias("noise"))
        if noise_thresh is not None:
            syndf = syndf.with_columns(
                (pl.col("noise") * noise_thresh).alias("noise_threshold")
            )
    if SDF:
        return SynDF(syndf)
    return syndf


def load_lsdf(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    channel: int = 2,
    apply_ephys_offset: bool = False,
    nan_to_null: bool = False,
    merge_labels: bool = True,
    SDF: bool = False,
) -> pl.DataFrame | SynDF:
    """load the lsdf dataframe generated from the ExperimentSummary

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    loc : str
        location name
    acq : str
        acquisition name
    trace_group : str, optional
        trace group to load, by default "dF" (dF or dFF)
    trace_types : list[str] | str | None, optional
        trace types to load, by default None ('all' loads all trace types)
    apply_ephys_offset : bool, optional
        whether to apply the ephys offset to the time column, by default False
    nan_to_null: bool, optional
        whether to convert nan values to null values, by default True
    merge_labels: bool, optional
        whether to merge the synapse labels, by default True
    adjust_noise_column: bool, optional
        whether to adjust the noise column, by default True
    SDF: bool, optional
        whether to return a SynDF object, by default False
    Returns
    -------
    pl.DataFrame
        the lsdf dataframe
    """
    syndf_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/lsdf.parquet"
    )
    syndf = pl.scan_parquet(syndf_path)
    syndf = syndf.filter(pl.col("channel") == channel)
    syndf = syndf.collect()

    if apply_ephys_offset:
        si = wis.meta.get.sync_info()
        ephys_offset = si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["ephys_offset"]
        syndf = syndf.with_columns((pl.col("time") + ephys_offset).alias("time"))

    syndf = syndf.with_columns(pl.lit(loc).alias("loc"), pl.lit(acq).alias("acq"))

    if nan_to_null:
        syndf = syndf.with_columns(
            pl
            .when(pl.col("data").is_nan())
            .then(pl.lit(None))
            .otherwise(pl.col("data"))
            .alias("data")
        )
    if merge_labels:
        idf = wis.get.synid_labels(subject, exp, loc, acq)
        syndf = syndf.join(
            idf, left_on=["dmd", "source-ID"], right_on=["dmd", "syn_id"], how="left"
        )
    if SDF:
        return SynDF(syndf)
    else:
        return syndf


def load_roidf(
    subject,
    exp,
    loc,
    acq,
    roi_version="Fsvd",
    channel=1,
    apply_ephys_offset=False,
    SDF: bool = False,
):
    roidf_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/roidf.parquet"
    )
    roidf = pl.scan_parquet(roidf_path)
    if channel is not None:
        roidf = roidf.filter(pl.col("channel") == channel).filter(
            pl.col("trace_type") == roi_version
        )
    roidf = roidf.collect()
    if apply_ephys_offset:
        si = wis.meta.get.sync_info()
        ephys_offset = si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["ephys_offset"]
        roidf = roidf.with_columns((pl.col("time") + ephys_offset).alias("time"))
    roidf = roidf.with_columns(pl.lit(loc).alias("loc"), pl.lit(acq).alias("acq"))
    if SDF:
        return SomaDF(roidf)
    else:
        return roidf


def load_roi_info_df(subject, exp, loc, acq):
    pd.DataFrame(columns=["soma-ID", "dmd", "dmd_depth", "centroid_x", "centroid_y"])


def load_f0df(
    subject,
    exp,
    loc,
    acq,
    apply_ephys_offset=False,
):
    f0df_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/fzdf.parquet"
    )
    f0df = pl.read_parquet(f0df_path)
    if apply_ephys_offset:
        si = wis.meta.get.sync_info()
        ephys_offset = si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["ephys_offset"]
        f0df = f0df.with_columns((pl.col("time") + ephys_offset).alias("time"))
    f0df = f0df.with_columns(pl.lit(loc).alias("loc"), pl.lit(acq).alias("acq"))
    return f0df


def load_roi_map(subject, exp, loc, acq, dmd):
    path = os.path.join(
        DEFS.anmat_root,
        "annotation_materials",
        subject,
        exp,
        loc,
        acq,
        "roi_locations",
        f"roi_locs_dmd{dmd}_mask.tif",
    )
    if not os.path.exists(path):
        print(f"{path} does not exist!")
        return None
    return tiff.imread(path)


def load_dual_roi_map(subject, exp, loc, acq):
    roi_map = {}
    for dmd in [1, 2]:
        roi_map[dmd] = load_roi_map(subject, exp, loc, acq, dmd)
    return roi_map


def load_mean_ims(subject, exp, loc, acq):
    esum_path = wis.meta.get.esum_mirror_path(subject, exp, loc, acq)
    mean_ims = {}
    for dmd in [1, 2]:
        meanim = spy.hf.load_any(esum_path, f"/exptSummary/meanIM[{dmd - 1}][0]")
        print(meanim.shape)
        mean_ims[dmd] = meanim.swapaxes(1, 2)
    return mean_ims


def load_fprts(subject, exp, loc, acq):
    fps = {}
    esum_path = wis.meta.get.esum_mirror_path(subject, exp, loc, acq)
    for dmd in [1, 2]:
        fp = spy.hf.load_any(esum_path, f"/exptSummary/E[{dmd - 1}][0]['footprints']")
        fp = fp.swapaxes(1, 2)
        fps[dmd] = fp
    return fps


def load_bayes_ev_df(subject, exp, loc, acq):
    all_dfs = []
    for dmd in [1, 2]:
        path = os.path.join(
            DEFS.anmat_root,
            subject,
            exp,
            "activity_data",
            loc,
            acq,
            "glutamate_event_detection",
            "bayes_hm",
            f"dmd{dmd}.parquet",
        )
        df = pl.read_parquet(path)
        df = df.with_columns(pl.lit(dmd).alias("dmd"))
        all_dfs.append(df)
    bdf = pl.concat(all_dfs)
    bdf = bdf.rename({"synapse": "source-ID"})
    bdf = bdf.rename({"t_sec": "time"})
    return bdf


def load_glut_events(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    merge_labels: bool = True,
) -> pl.DataFrame:
    """Load glutamate event detection results (new pipeline).

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.
    merge_labels : bool
        If True, merge synapse labels from annotation materials.

    Returns
    -------
    pl.DataFrame
        Events from both DMDs concatenated.
    """
    all_dfs = []
    for dmd in [1, 2]:
        path = os.path.join(
            DEFS.anmat_root,
            subject,
            exp,
            "activity_data",
            loc,
            acq,
            "glut_events",
            f"events_dmd{dmd}.parquet",
        )
        if not os.path.isfile(path):
            continue
        df = pl.read_parquet(path)
        all_dfs.append(df)
    if not all_dfs:
        raise FileNotFoundError(
            f"No glut_events parquets found for {subject} {exp} {loc} {acq}"
        )
    edf = pl.concat(all_dfs)
    # Backward-compat: older parquets have "t_sec" instead of "time"
    if "t_sec" in edf.columns:
        edf = edf.rename({"t_sec": "time"})
    if merge_labels:
        try:
            idf = wis.get.synid_labels(subject, exp, loc, acq)
            edf = edf.join(
                idf,
                left_on=["dmd", "source_id"],
                right_on=["dmd", "syn_id"],
                how="left",
            )
        except Exception:
            pass
    return edf


def load_glut_synapse_summary(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
) -> pl.DataFrame:
    """Load per-synapse glutamate event summary statistics.

    Parameters
    ----------
    subject, exp, loc, acq : str
        Acquisition identifiers.

    Returns
    -------
    pl.DataFrame
        Synapse summaries from both DMDs concatenated.
    """
    all_dfs = []
    for dmd in [1, 2]:
        path = os.path.join(
            DEFS.anmat_root,
            subject,
            exp,
            "activity_data",
            loc,
            acq,
            "glut_events",
            f"synapse_summary_dmd{dmd}.parquet",
        )
        if not os.path.isfile(path):
            continue
        df = pl.read_parquet(path)
        all_dfs.append(df)
    if not all_dfs:
        raise FileNotFoundError(
            f"No glut_events synapse summaries found for {subject} {exp} {loc} {acq}"
        )
    return pl.concat(all_dfs)
