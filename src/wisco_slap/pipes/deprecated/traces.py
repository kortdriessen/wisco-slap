import os
from collections.abc import Sequence
from typing import Any

import electro_py as epy
import numpy as np
import pandas as pd
import polars as pl
import slap2_py as spy

import wisco_slap as wis
import wisco_slap.defs as DEFS


def slap_dat_to_df(
    dat: np.ndarray,
    axis_labels: Sequence[str],
    value_axis: int,
    value_name: str,
    new_columns: dict[str, Any] | None = None,
) -> pl.DataFrame:
    df = epy.gen.df.cube_to_df(
        dat, axis_labels=axis_labels, value_axis=value_axis, value_name=value_name
    )
    if new_columns is not None:
        return df.with_columns([pl.lit(v).alias(k) for k, v in new_columns.items()])
    return df


def load_synid_labels(subject, exp, loc, acq):
    all_dfs = []
    for dmd in [1, 2]:
        path = f"{DEFS.anmat_root}/{subject}/{exp}/synapse_ids/{loc}/{acq}/dmd-{dmd}/synapse_labels.csv"
        if not os.path.exists(path):
            print(f"{path} does not exist!")
            return None
        df = pd.read_csv(path)
        df["dmd"] = dmd
        all_dfs.append(df)
    idf = pl.from_pandas(pd.concat(all_dfs))
    if "dend-ID" not in idf.columns:
        idf = idf.with_columns(pl.lit("unlabelled").alias("dend-ID"))
    idf = idf.filter(pl.col("source-ID") != "master_image")
    idf = idf.with_columns(pl.col("source-ID").cast(pl.Int32).alias("source"))
    idf = idf.drop("source-ID")
    di = wis.util.info.load_dmd_info()
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

    return idf


def create_full_activity_dfs(subject, exp, loc, acq):
    esum_path = wis.util.info.get_esum_mirror_path(subject, exp, loc, acq)
    eset = spy.hf.io.load_full_Eset_all_dmds(esum_path, exclude_fields=["footprints"])
    fs = spy.hf.load_any(esum_path, "/exptSummary/params/analyzeHz")
    fs = float(fs[0][0])

    activity_channel = 2
    trace_dfs = []
    noise_dfs = []
    F0_dfs = []
    roi_dfs = []
    trace_types = ["matchFilt", "events", "denoised", "nonneg"]
    trace_cols = ["data", "source", "dmd", "trial", "group", "type", "channel", "time"]
    for dmd in [1, 2]:
        r = spy.hf.load_any(esum_path, f"/exptSummary/userROIs[{dmd - 1}][0]")
        add_roidf = True
        if type(r) is dict:
            rois = [r]
        else:
            rois = r
        if rois[0] == 0:
            add_roidf = False

        tl = []
        for t in range(len(eset[dmd])):
            tl.append(eset[dmd][t]["F0"].shape[1])
        assert len(np.unique(tl)) == 1, (
            "trial lengths are not equal"
        )  # TODO: there should be a better way to handle this, even if there is some heterogeneity in the trial lengths
        trial_length = tl[0]
        time_array_base = np.arange(0, trial_length / fs, 1 / fs)
        n_sources = eset[dmd][0]["F0"].shape[2]
        for trial in range(len(eset[dmd])):
            data_dict = eset[dmd][trial]
            if data_dict is not None:
                trial_length_sec = trial_length / fs
                time_array_single_source = time_array_base + (trial * trial_length_sec)
                # make sure the values in time_array_single_source are rounded to a max of 5 decimal places
                time_array_single_source = np.round(time_array_single_source, 5)
                time_array_by_sources = np.tile(time_array_single_source, n_sources)
                # first we can get the F0 data here
                F0 = data_dict["F0"]
                for channel in [1, 2]:
                    f0chan = (
                        2 if channel == 1 else 1
                    )  # invert the channels here, bug in matlab pipeline
                    F0_df = slap_dat_to_df(
                        F0[f0chan - 1, :, :],
                        axis_labels=["data", "source"],
                        value_axis=0,
                        value_name="data",
                        new_columns={
                            "dmd": dmd,
                            "trial": trial,
                            "channel": channel,
                            "time": time_array_by_sources,
                        },
                    )
                    F0_dfs.append(F0_df)

                for group in ["dF", "dFF"]:
                    # here we get the ls data
                    ls = data_dict[group]["ls"]
                    if group == "dF":
                        for channel in [1, 2]:
                            ls_chan = (
                                2 if channel == 1 else 1
                            )  # invert the channels here, bug in matlab pipeline
                            ls_df = slap_dat_to_df(
                                ls[ls_chan - 1, :, :],
                                axis_labels=["data", "source"],
                                value_axis=0,
                                value_name="data",
                                new_columns={
                                    "dmd": dmd,
                                    "trial": trial,
                                    "group": group,
                                    "type": "ls",
                                    "channel": channel,
                                    "time": time_array_by_sources,
                                },
                            )
                            # ensure the channel column is i32
                            ls_df = ls_df.with_columns(pl.col("channel").cast(pl.Int32))
                            trace_dfs.append(ls_df.select(trace_cols))
                    else:
                        ls_df = slap_dat_to_df(
                            ls,
                            axis_labels=["data", "source"],
                            value_axis=0,
                            value_name="data",
                            new_columns={
                                "dmd": dmd,
                                "trial": trial,
                                "group": group,
                                "type": "ls",
                                "channel": activity_channel,
                                "time": time_array_by_sources,
                            },
                        )
                        trace_dfs.append(ls_df.select(trace_cols))

                    # and the ls noise estimates
                    ls_noise = data_dict["noiseEst"][group]["ls"]
                    noise_sd = np.sqrt(ls_noise.flatten())
                    ls_noise_df = slap_dat_to_df(
                        noise_sd,
                        axis_labels=["noise"],
                        value_axis=0,
                        value_name="noise",
                        new_columns={
                            "type": "ls",
                            "group": group,
                            "dmd": dmd,
                            "trial": trial,
                            "channel": activity_channel,
                            "source": np.arange(len(noise_sd)),
                        },
                    )
                    noise_dfs.append(ls_noise_df)

                    # Now all of the other synaptic traces
                    for tt in trace_types:
                        # print(dmd, trial, tt)
                        # first we get the noise estimates
                        noise = data_dict["noiseEst"][group][tt]
                        sd = np.sqrt(noise.flatten())
                        noise_df = slap_dat_to_df(
                            sd,
                            axis_labels=["noise"],
                            value_axis=0,
                            value_name="noise",
                            new_columns={
                                "type": tt,
                                "group": group,
                                "dmd": dmd,
                                "trial": trial,
                                "channel": activity_channel,
                                "source": np.arange(len(sd)),
                            },
                        )
                        noise_dfs.append(noise_df)
                        # then the trace data
                        trace_dat = data_dict[group][tt]
                        df = slap_dat_to_df(
                            trace_dat,
                            axis_labels=["data", "source"],
                            value_axis=0,
                            value_name="data",
                            new_columns={
                                "dmd": dmd,
                                "trial": trial,
                                "group": group,
                                "type": tt,
                                "channel": activity_channel,
                                "time": time_array_by_sources,
                            },
                        )
                        trace_dfs.append(df.select(trace_cols))

                # Finally, we get the ROI Traces
                if add_roidf:
                    for roi_trace in ["F", "Fsvd", "F_fullSpeed"]:
                        roi_vals = data_dict["ROIs"][roi_trace]
                        for i, roi_inf in enumerate(rois):
                            label = roi_inf["Label"]
                            for channel in [1, 2]:
                                sel_chan = (
                                    2 if channel == 1 else 1
                                )  # invert the channels here, bug in matlab pipeline
                                roi_dat = roi_vals[sel_chan - 1, :, i]
                                roi_df = slap_dat_to_df(
                                    roi_dat,
                                    axis_labels=["data"],
                                    value_axis=0,
                                    value_name="data",
                                    new_columns={
                                        "dmd": dmd,
                                        "trial": trial,
                                        "channel": channel,
                                        "group": "ROI",
                                        "type": roi_trace,
                                        "soma-ID": label,
                                        "time": time_array_single_source,
                                    },
                                )
                                # make sure data is cast to f64
                                roi_df = roi_df.with_columns(
                                    pl.col("data").cast(pl.Float64)
                                )
                                roi_dfs.append(roi_df)

            elif data_dict is None:
                # TODO: def create_blank_data_dfs()
                raise NotImplementedError(
                    "No data found for this trial, have not implemented blank data dfs yet"
                )

    noise_df = pl.concat(noise_dfs)
    trace_df = pl.concat(trace_dfs)
    F0_df = pl.concat(F0_dfs)
    roidf = pl.concat(roi_dfs)
    tdf = trace_df.join(
        noise_df, on=["group", "type", "dmd", "channel", "trial", "source"], how="left"
    )
    idf = load_synid_labels(subject, exp, loc, acq)
    if idf is None:
        print(
            f"No synapse labels found for {subject} {exp} {loc} {acq}, returning synapse dataframe without synapse labels"
        )
        return tdf, roidf, F0_df
    else:
        syn_df_final = tdf.join(idf, on=["dmd", "source"], how="left")
        return syn_df_final, roidf, F0_df


def check_activity_dfs_exist(subject, exp, loc, acq):
    syndf_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/syndf.parquet"
    )
    roidf_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/roidf.parquet"
    )
    F0df_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/F0df.parquet"
    )
    return (
        os.path.exists(syndf_path),
        os.path.exists(roidf_path),
        os.path.exists(F0df_path),
    )


def save_activity_dfs(subject, exp, loc, acq, syndf, roidf, F0df, overwrite=False):
    syndf_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/syndf.parquet"
    )
    wis.util.gen.check_dir(os.path.dirname(syndf_path))
    if os.path.exists(syndf_path):
        if not overwrite:
            print(f"{syndf_path} already exists, skipping")
        if overwrite:
            print(f"{syndf_path} already exists, overwriting")
            os.system(f"rm -rf {syndf_path}")
            syndf.write_parquet(syndf_path)
    else:
        syndf.write_parquet(syndf_path)
    roidf_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/roidf.parquet"
    )
    if os.path.exists(roidf_path):
        if not overwrite:
            print(f"{roidf_path} already exists, skipping")
        if overwrite:
            print(f"{roidf_path} already exists, overwriting")
            os.system(f"rm -rf {roidf_path}")
            roidf.write_parquet(roidf_path)
    else:
        roidf.write_parquet(roidf_path)
    F0df_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/F0df.parquet"
    )
    if os.path.exists(F0df_path):
        if not overwrite:
            print(f"{F0df_path} already exists, skipping")
        if overwrite:
            print(f"{F0df_path} already exists, overwriting")
            os.system(f"rm -rf {F0df_path}")
            F0df.write_parquet(F0df_path)
    else:
        F0df.write_parquet(F0df_path)
    return


def _gen_and_save_activity_dfs(subject, exp, loc, acq, overwrite=False):
    syndf_exists, roidf_exists, F0df_exists = check_activity_dfs_exist(
        subject, exp, loc, acq
    )
    if all([syndf_exists, roidf_exists, F0df_exists]):
        if not overwrite:
            print(f"{subject} {exp} {loc} {acq} already exists, skipping")
            return
    syndf, roidf, F0df = create_full_activity_dfs(subject, exp, loc, acq)
    save_activity_dfs(subject, exp, loc, acq, syndf, roidf, F0df, overwrite=overwrite)
    return


def gen_and_save_activity_dfs_all_subjects(overwrite=False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            acq_ids = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
            for acq_id in acq_ids:
                try:
                    print(f"Working on {subject} {exp} {acq_id}")
                    loc, acq = acq_id.split("--")
                    esum_path = wis.util.info.get_esum_mirror_path(
                        subject, exp, loc, acq
                    )
                    if esum_path is None:
                        continue
                    _gen_and_save_activity_dfs(
                        subject, exp, loc, acq, overwrite=overwrite
                    )
                except Exception as e:
                    print("-----------------------------------------------------------")
                    print(
                        f"Error generating and saving activity dfs for {subject} {exp} {loc} {acq}: {e}"
                    )
                    print("-----------------------------------------------------------")
                    continue
    return
