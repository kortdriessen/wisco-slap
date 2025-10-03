import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import slap2_py as spy
import streamlit as st
from webagg_selector import start_selector_process, stop_selector_process

import wisco_slap as wis

st.title("Source Selector")

if "selector_url" not in st.session_state:
    st.session_state["selector_url"] = None
if "selector_proc" not in st.session_state:
    st.session_state["selector_proc"] = None

exp_info = wis.pipes.exp_info.load_exp_info_spreadsheet()
subjects = exp_info["subject"].unique()
subject = st.selectbox("Subject", subjects)
subject_exps = exp_info.filter(pl.col("subject") == subject)["experiment"].unique()
exp = st.selectbox("Experiment", subject_exps)
acq_ids = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
acq_id = st.selectbox("Acquisition-ID", acq_ids)
loc, acq = acq_id.split("--")

dmd = st.radio("DMD", [1, 2])


esum_path = wis.util.io.sub_esum_path(subject, exp, loc, acq)
fp = spy.utils.hfive.load_any(esum_path, f"/exptSummary/E[{dmd - 1}][0]['footprints']")
fp = fp.swapaxes(1, 2)
masks = fp > 0
meanim = spy.utils.hfive.load_any(esum_path, f"/exptSummary/meanIM[{dmd - 1}][0]")
image = meanim[1, :, :].T

st.write(image.shape)
st.write(masks.shape)

f, ax = plt.subplots(1, 1, figsize=(10, 10))
vmin = st.slider("Image Minimum", value=0, min_value=0, max_value=100)
vmax = st.slider("Image Maximum", value=95, min_value=0, max_value=100)
v_min = np.nanpercentile(image, vmin)
v_max = np.nanpercentile(image, vmax)
ax.imshow(image, cmap="viridis", vmin=v_min, vmax=v_max)
st.pyplot(f)

txt = st.text_input("Source Selector Output Path", value="selected_source_ids.txt")


# Choose a TCP port for the WebAgg figure; forward this port in VS Code
port = st.number_input(
    "Source Selector Port", min_value=1024, max_value=65500, value=8988, step=1
)
embed = st.checkbox("Embed selector in dashboard (iframe)", value=True)

# Keep URL in session so it persists across reruns
if "selector_url" not in st.session_state:
    st.session_state["selector_url"] = None

if st.button("Show Source Selector"):
    # If an old one is running, stop it first (optional)
    if st.session_state["selector_proc"] is not None:
        stop_selector_process(st.session_state["selector_proc"])

    url, proc = start_selector_process(image, masks, output_path=txt, port=int(port))
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
