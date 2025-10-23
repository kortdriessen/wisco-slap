import os

import polars as pl
import slap2_py as spy

import wisco_slap as wis
import wisco_slap.defs as DEFS


def regen_roi_df(subject, exp, loc, acq, roi_version="Fsvd"):
    activity_dir = f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}"
    wis.util.gen.check_dir(activity_dir)
    roidf_path = f"{activity_dir}/roidf_{roi_version}.parquet"
    if os.path.exists(roidf_path):
        os.system(f"rm -rf {roidf_path}")
    esum_path = wis.util.io.sub_esum_path(subject, exp, loc, acq)
    e = spy.ExSum.from_mat73(esum_path)
    roidf = e.gen_roidf(version=roi_version)
    roidf.write_parquet(
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/roidf_{roi_version}.parquet"
    )
    return


def save_activity_dataframes(
    subject,
    exp,
    loc,
    acq,
    overwrite=False,
    synapse_trace_types=None,
    trace_group="dF",
    roi_version="Fsvd",
):
    """Save the syndf and roidf dataframes to a parquet file.

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    loc : str
        The location ID.
    acq : str
        The acquisition ID.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False
    synapse_trace_types : list[str], optional
        The trace types to include in the syndf, by default None
    trace_group : str, optional
        The trace group to include in the syndf, by default "dF"
    roi_version : str, optional
        The version of the roi dataframe to include, by default "Fsvd"
    ls : bool, optional
        Whether to include the ls dataframe, by default True
    """
    activity_dir = f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}"
    wis.util.gen.check_dir(activity_dir)
    syndf_path = f"{activity_dir}/syndf_{trace_group}.parquet"
    roidf_path = f"{activity_dir}/roidf_{roi_version}.parquet"
    lsdf_path = f"{activity_dir}/lsdf_{trace_group}.parquet"

    if (
        all(
            [
                os.path.exists(syndf_path),
                os.path.exists(roidf_path),
                os.path.exists(lsdf_path),
            ]
        )
        and not overwrite
    ):
        print(
            f"All files already exist for {subject} {exp} {loc} {acq}. Use overwrite=True to overwrite."
        )
        return
    if synapse_trace_types is None:
        synapse_trace_types = ["matchFilt", "denoised", "events", "nonneg"]
    esum_path = wis.util.io.sub_esum_path(subject, exp, loc, acq)
    e = spy.ExSum.from_mat73(esum_path)
    syndf = e.gen_syndf(trace_group=trace_group, to_pull=synapse_trace_types)
    if os.path.exists(syndf_path) and not overwrite:
        print(f"File {syndf_path} already exists. Use overwrite=True to overwrite.")
        return
    if os.path.exists(syndf_path) and overwrite:
        os.system(f"rm -rf {syndf_path}")
    syndf.write_parquet(syndf_path)
    roidf = e.gen_roidf(version=roi_version)
    if os.path.exists(roidf_path) and not overwrite:
        print(f"File {roidf_path} already exists. Use overwrite=True to overwrite.")
        return
    if os.path.exists(roidf_path) and overwrite:
        os.system(f"rm -rf {roidf_path}")
    roidf.write_parquet(roidf_path)

    lsdf = e.gen_ls_df(trace_group=trace_group)
    if os.path.exists(lsdf_path) and not overwrite:
        print(f"File {lsdf_path} already exists. Use overwrite=True to overwrite.")
        return
    if os.path.exists(lsdf_path) and overwrite:
        os.system(f"rm -rf {lsdf_path}")
    lsdf.write_parquet(lsdf_path)
    return


def save_activity_dataframes_all_subjects(overwrite=False, max_dur=None):
    ei = wis.pipes.exp_info.load_exp_info_spreadsheet()
    for subject in ei["subject"].unique():
        for exp in ei.filter(pl.col("subject") == subject)["experiment"].unique():
            for loc in (
                ei.filter(pl.col("subject") == subject)
                .filter(pl.col("experiment") == exp)["location"]
                .unique()
            ):
                for acq in (
                    ei.filter(pl.col("subject") == subject)
                    .filter(pl.col("experiment") == exp)
                    .filter(pl.col("location") == loc)["acquisition"]
                    .unique()
                ):
                    print(f"Saving activity dataframes for {subject} {exp} {loc} {acq}")
                    if (
                        wis.pipes.exp_info.determine_processing_done(
                            subject, exp, loc, acq
                        )
                        == "NO"
                    ):
                        print(
                            f"Processing not done yet for {subject} {exp} {loc} {acq}"
                        )
                        continue
                    if max_dur is not None:
                        acq_duration = wis.pipes.exp_info.estimate_acq_duration(
                            subject, exp, loc, acq
                        )
                        if acq_duration > max_dur:
                            print(
                                f"Acquisition duration {acq_duration} is greater than max_dur {max_dur} for {subject} {exp} {loc} {acq}"
                            )
                            continue
                    save_activity_dataframes(
                        subject, exp, loc, acq, overwrite=overwrite
                    )
    return
