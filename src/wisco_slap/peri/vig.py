import wisco_slap as wis
import wisco_slap.defs as DEFS
import pandas as pd
import os
import electro_py as epy


def load_auto_hypno(
    subject,
    exp,
    sync_block,
    filter_unclear: bool = None,
    filter_on="smooth",
    rename=True,
):
    hypno_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}/hypnograms/model_labelled"
    hypno_name = f"raw_epochs.csv"
    epoch_df = pd.read_csv(f"{hypno_dir}/{hypno_name}")
    if filter_unclear is not None:
        if filter_on == "smooth":
            epoch_df["label"] = epoch_df["label"].mask(
                epoch_df[["NREM_smooth", "REM_smooth", "Wake_smooth"]].max(axis=1)
                < filter_unclear,
                "unclear",
            )
        elif filter_on == "raw":
            epoch_df["label"] = epoch_df["label"].mask(
                epoch_df[["P_NREM", "P_REM", "P_Wake"]].max(axis=1) < filter_unclear,
                "unclear",
            )
        else:
            raise ValueError(f"Invalid filter_on: {filter_on}")
    epoch_df = epy.hypno.utils.merge_consecutive_labels(epoch_df)
    if rename:
        epoch_df = epoch_df.rename(columns={"label": "state"})
        epoch_df = epoch_df.rename(
            columns={"start_s": "start_time", "end_s": "end_time"}
        )
        epoch_df["duration"] = epoch_df["end_time"] - epoch_df["start_time"]
        epoch_df = epoch_df.sort_values(by="start_time")
        epoch_df = epoch_df.reset_index(drop=True)
    return epoch_df
