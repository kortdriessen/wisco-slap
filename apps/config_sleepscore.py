import os

import numpy as np
import polars as pl
import streamlit as st
import yaml

import wisco_slap as wis
import wisco_slap.defs as DEFS


def save_sleepscore_config(subject, exp, sync_block, config_dict):
    config_path = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}/config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
        f.close()
    return


st.title("Sleep Score Configuration")

overwrite = st.checkbox("Overwrite data if it already exists")

exp_info = wis.pipes.exp_info.load_exp_info_spreadsheet()
subjects = exp_info["subject"].unique()
subject = st.selectbox("Choose Subject", subjects)
subject_exps = exp_info.filter(pl.col("subject") == subject)["experiment"].unique()
exp = st.selectbox("Choose Experiment", subject_exps)

si = wis.peri.sync.load_sync_info()

sync_blocks = si[subject][exp]["sync_blocks"].keys()
sync_block = st.selectbox("Choose Sync Block", sync_blocks)

st.write(f" Corrupt: {si[subject][exp]["sync_blocks"][sync_block]["corrupt"]}")
block_acqs = []
for acq_id in si[subject][exp]["acquisitions"].keys():
    if si[subject][exp]["acquisitions"][acq_id]["sync_block"] == sync_block:
        block_acqs.append(acq_id)

st.write(f"Acquisitions associated with this sync block: {block_acqs}")

include_rois = st.checkbox("Include ROIs in config")


if include_rois:
    roi_channel = st.number_input("ROI Channel", value=1, min_value=1, max_value=2)
    roi_version = st.text_input("ROI Version", value="Fsvd")
    all_rois = []
    rois_for_config = {acq_id: [] for acq_id in block_acqs}
    for acq_id in block_acqs:
        loc, acq = acq_id.split("--")
        acq_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}/scope_traces/soma_rois/{acq_id}"
        if not os.path.exists(acq_dir):
            wis.pipes.sleepscore.write_roi_traces_for_scoring(
                subject, exp, loc, acq, roi_version=roi_version, roi_channel=roi_channel
            )
        # check if the dir is empty
        elif not os.listdir(acq_dir):
            st.write(f"Directory {acq_dir} is empty. Generating ROI traces.")
            wis.pipes.sleepscore.write_roi_traces_for_scoring(
                subject, exp, loc, acq, roi_version=roi_version, roi_channel=roi_channel
            )

        for f in os.listdir(acq_dir):
            if f.endswith(".npy") and "_y" in f:
                name = f.split(".")[0]
                all_rois.append(f"{acq_id}__{name}")

    rois_for_config = st.multiselect(
        "Select ROIs to include in config", all_rois, default=all_rois
    )

include_synapses = st.checkbox("Include Synapses in config")
if include_synapses:
    # will use only matchfilt for now
    acq_id = st.selectbox("Choose Acquisition-ID (only one is allowed)", block_acqs)
    synapse_trace_type = "matchFilt"
    trace_group = "dF"
    loc, acq = acq_id.split("--")
    syndf = wis.scope.io.load_syndf(subject, exp, loc, acq, trace_group=trace_group)
    ephys_offset = si[subject][exp]["acquisitions"][acq_id]["ephys_offset"]
    if st.button("Generate Synapse Traces"):
        for dmd in [1, 2]:
            source_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}/scope_traces/synapses/dmd{dmd}"
            txt_path = f"{source_dir}/source_ids__dmd{dmd}.txt"
            if not os.path.exists(txt_path):
                continue
            for f in os.listdir(source_dir):
                if f.endswith(".npy"):
                    st.write(f"Removing {f}")
                    os.system(f"rm -rf {source_dir}/{f}")
            with open(txt_path) as f:
                source_id_strings = f.read().splitlines()
                st.write(source_id_strings)
            source_ids = [
                int(source_id_string.split("-")[-1])
                for source_id_string in source_id_strings
            ]
            st.write(source_ids)
            source_df = syndf.filter(pl.col("dmd") == dmd).filter(
                pl.col("source").is_in(source_ids)
            )
            st.write(f"Found {len(source_df)} sources for DMD {dmd}")
            for source_id in source_ids:
                one_source_df = source_df.filter(pl.col("source") == source_id)
                trace = one_source_df["data"].to_numpy()
                time = one_source_df["time"].to_numpy()
                time = time + ephys_offset
                trace_name = f"{dmd}-{source_id}"
                np.save(f"{source_dir}/{trace_name}_y.npy", trace)
                np.save(f"{source_dir}/{trace_name}_t.npy", time)


if st.button("Generate and Save Config"):
    config = {}
    config["subject"] = subject
    config["exp"] = exp
    config["sync_block"] = sync_block
    if include_rois:
        if len(rois_for_config) > 0:
            config["rois"] = rois_for_config
    config["synapses"] = False
    save_sleepscore_config(subject, exp, sync_block, config)
