import os

import numpy as np
import polars as pl
import slap2_py as spy
import streamlit as st
from PIL import Image
from webagg_selector import start_selector_process, stop_selector_process

import wisco_slap as wis
import wisco_slap.defs as DEFS


@st.cache_data
def load_exp_info():
    return wis.pipes.exp_info.load_exp_info_spreadsheet()


exp_info = load_exp_info()


def sync_sources_to_selection(
    subject, exp, loc, acq, trace_group="dF", trace_types=None
):
    if trace_types is None:
        trace_types = ["matchFilt"]
    acq_id = f"{loc}--{acq}"
    syndf = wis.scope.io.load_syndf(
        subject,
        exp,
        loc,
        acq,
        trace_group=trace_group,
        trace_types=trace_types,
        lazy=True,
    )
    si = wis.peri.sync.load_sync_info()
    ephys_offset = si[subject][exp]["acquisitions"][acq_id]["ephys_offset"]
    sync_block = si[subject][exp]["acquisitions"][acq_id]["sync_block"]
    for dmd in [1, 2]:
        source_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}/scope_traces/synapses/{acq_id}/dmd{dmd}"
        wis.util.gen.check_dir(source_dir)

        # if the dir has any existing traces, delete them:
        for f in os.listdir(source_dir):
            if f.endswith(".npy"):
                os.system(f"rm -rf {source_dir}/{f}")

        # determine the source ids from the text file
        txt_path = f"{source_dir}/source_ids__dmd{dmd}.txt"
        if not os.path.exists(txt_path):
            print(f"Source IDs text file not found for DMD {dmd}: {txt_path}")
            print("Continuing...")
            continue
        with open(txt_path) as f:
            source_ids = f.read().splitlines()
        source_ids = [int(source_id) for source_id in source_ids]
        # select and save the traces and associated times
        source_df = syndf.filter(pl.col("dmd") == dmd).filter(
            pl.col("source").is_in(source_ids)
        )
        for source_id in source_ids:
            one_source_df = source_df.filter(pl.col("source") == source_id)
            trace = one_source_df["data"].to_numpy()
            time = one_source_df["time"].to_numpy() + ephys_offset
            np.save(f"{source_dir}/{dmd}-{source_id}_y.npy", trace)
            np.save(f"{source_dir}/{dmd}-{source_id}_t.npy", time)
    return


st.title("Source Selector")

if "selector_url" not in st.session_state:
    st.session_state["selector_url"] = None
if "selector_proc" not in st.session_state:
    st.session_state["selector_proc"] = None


subjects = sorted(exp_info["subject"].unique().to_list())
subject = st.selectbox("Subject", subjects, key="subject")

subject_exps = sorted(
    exp_info.filter(pl.col("subject") == st.session_state.subject)["experiment"]
    .unique()
    .to_list()
)
exp = st.selectbox("Experiment", subject_exps, key="experiment")

acq_ids = wis.util.info.get_unique_acquisitions_per_experiment(
    st.session_state.subject, st.session_state.experiment
)
acq_id = st.selectbox("Acquisition-ID", acq_ids, key="acq_id")
loc, acq = acq_id.split("--")
si = wis.peri.sync.load_sync_info()
sync_block = si[subject][exp]["acquisitions"][acq_id]["sync_block"]

esum_path = wis.util.io.sub_esum_path(subject, exp, loc, acq)
if esum_path is None:
    st.warning(f"Esum path not found: {esum_path}, choose a different Acquisition-ID")
    st.stop()
if not os.path.exists(esum_path):
    st.warning(f"Esum path not found: {esum_path}, choose a different Acquisition-ID")
    st.stop()

dmd = st.radio("DMD", [1, 2])

fp = spy.hf.load_any(esum_path, f"/exptSummary/E[{dmd - 1}][0]['footprints']")
fp = fp.swapaxes(1, 2)
masks = fp > 0
meanim = spy.hf.load_any(esum_path, f"/exptSummary/meanIM[{dmd - 1}][0]")
image = meanim[1, :, :].T

st.write(image.shape)
st.write(masks.shape)

# load the meanIM reference png and display it
labelled_image = st.checkbox("Show labelled image if exists", value=True)
if labelled_image:
    meanim_path = f"{DEFS.anmat_root}/{subject}/{exp}/mean_IMs/{loc}/{acq}/DMD-{dmd}__labelled.png"
    if not os.path.exists(meanim_path):
        meanim_path = (
            f"{DEFS.anmat_root}/{subject}/{exp}/mean_IMs/{loc}/{acq}/DMD-{dmd}.png"
        )
else:
    meanim_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/mean_IMs/{loc}/{acq}/DMD-{dmd}.png"
    )

# if not os.path.exists(meanim_path):
#    st.warning(
#        f"MeanIM path not found: {meanim_path}, choose a different Acquisition-ID"
#    )
#    st.stop()
# meanim_img = Image.open(meanim_path)
# st.image(meanim_img)

# ===== Alternative way to display the actual meanIM ndarray ===========
import matplotlib.pyplot as plt

f, ax = plt.subplots(1, 1, figsize=(10, 10))
vmin = st.slider("Image Minimum", value=0, min_value=0, max_value=100)
vmax = st.slider("Image Maximum", value=95, min_value=0, max_value=100)
v_min = np.nanpercentile(image, vmin)
v_max = np.nanpercentile(image, vmax)
ax.imshow(image, cmap="viridis", vmin=v_min, vmax=v_max)
st.pyplot(f)


txt_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}/scope_traces/synapses/{acq_id}/dmd{dmd}"
wis.util.gen.check_dir(txt_dir)
txt_out_path = f"{txt_dir}/source_ids__dmd{dmd}.txt"

# Choose a TCP port for the WebAgg figure; forward this port in VS Code
port = st.number_input(
    "Source Selector Port", min_value=1024, max_value=65500, value=8988, step=1
)
embed = st.checkbox("Embed selector in dashboard (iframe)", value=False)

# Keep URL in session so it persists across reruns
if "selector_url" not in st.session_state:
    st.session_state["selector_url"] = None

if st.button("Show Source Selector"):
    # If an old one is running, stop it first (optional)
    if st.session_state["selector_proc"] is not None:
        stop_selector_process(st.session_state["selector_proc"])

    url, proc = start_selector_process(
        image, masks, output_path=txt_out_path, port=int(port)
    )
    st.session_state["selector_proc"] = proc
    # Use localhost on the client side (VS Code port forward)
    st.session_state["selector_url"] = url.replace("127.0.0.1", "localhost")

if st.button("Stop Source Selector"):
    stop_selector_process(st.session_state.get("selector_proc"))
    st.session_state["selector_proc"] = None
    st.session_state["selector_url"] = None

# Render either embedded iframe or a link to open in a new tab
if st.session_state["selector_url"]:
    if embed:
        st.components.v1.iframe(st.session_state["selector_url"], height=720)
    else:
        st.markdown(
            f"[Open Source Selector in a new tab]({st.session_state['selector_url']})"
        )

st.markdown("--------")
st.markdown("# Trace Generation for Sleepscoring")
st.markdown(
    "This section will (re)generate soma ROI traces if desired, and synchronize synaptic traces to the selected sources."
)
if st.checkbox("Generate Soma ROI Traces"):
    roi_channel = st.number_input("ROI Channel", value=1, min_value=1, max_value=2)
    roi_version = st.text_input("ROI Version", value="Fsvd")
    if st.button("GENERATE SOMA TRACES"):
        wis.pipes.sleepscore.write_roi_traces_for_scoring(
            subject, exp, loc, acq, roi_version=roi_version, roi_channel=roi_channel
        )
if st.checkbox("Generate Synapse Traces"):
    st.markdown("Make sure txt files have been generated!")
    trace_group = st.text_input("Trace Group", value="dF")
    trace_types = st.multiselect(
        "Trace Types",
        ["matchFilt", "denoised", "events", "nonneg"],
        default=["matchFilt"],
    )
    if st.button("GENERATE SYNAPTIC TRACES"):
        sync_sources_to_selection(
            subject, exp, loc, acq, trace_group=trace_group, trace_types=trace_types
        )
