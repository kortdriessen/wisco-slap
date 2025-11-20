import os
import streamlit as st

import wisco_slap as wis
import polars as pl
import wisco_slap.defs as DEFS
from PIL import Image


@st.cache_data
def load_exp_info():
    return wis.pipes.exp_info.load_exp_info_spreadsheet()


exp_info = load_exp_info()

subjects = sorted(exp_info["subject"].unique().to_list())
subject = st.selectbox("Subject", subjects, key="subject")

subject_exps = sorted(
    exp_info.filter(pl.col("subject") == subject)["experiment"].unique().to_list()
)


for exp in subject_exps:
    acq_ids = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
    for acq_id in acq_ids:
        loc, acq = acq_id.split("--")
        st.markdown(f"## {subject} | {exp} | {loc} | {acq}")
        for dmd in [1, 2]:
            image_path = f"{DEFS.anmat_root}/{subject}/{exp}/mean_IMs/{loc}/{acq}/DMD-{dmd}__labelled.png"

            if not os.path.exists(image_path):
                unlabelled_path = f"{DEFS.anmat_root}/{subject}/{exp}/mean_IMs/{loc}/{acq}/DMD-{dmd}.png"
                if os.path.exists(unlabelled_path):
                    image_path = unlabelled_path
                else:
                    ip = f"{DEFS.anmat_root}/{subject}/{exp}/mean_IMs/{loc}/{acq}/DMD-{dmd}__TRIAL-000000__labelled.png"
                    if os.path.exists(ip):
                        image_path = ip
                    else:
                        image_path = f"{DEFS.anmat_root}/{subject}/{exp}/mean_IMs/{loc}/{acq}/DMD-{dmd}__TRIAL-000000.png"

            if not os.path.exists(image_path):
                st.markdown("--------------------")
                st.write(f"MEAN IMAGE NOT GENERATED: {image_path}")
                st.markdown("--------------------")
                continue
            image = Image.open(image_path)
            st.image(image)
