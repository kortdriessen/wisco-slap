import wisco_slap as wis
import wisco_slap.defs as DEFS
import numpy as np
import pickle
import electro_py as epy
import os
import polars as pl
import yaml


def check_for_scoring_times(subject, exp, sb):
    sb_scoring_times_path = f"{DEFS.anmat_root}/sb_scoring_times.yaml"
    sb_name = f"sync_block-{sb}"
    with open(sb_scoring_times_path, "r") as f:
        t = yaml.load(f, Loader=yaml.FullLoader)
    if subject in t:
        if exp in t[subject]:
            if sb_name in t[subject][exp]:
                return (
                    t[subject][exp][sb_name]["starts"],
                    t[subject][exp][sb_name]["ends"],
                )
    return ([None], [None])


def load_model() -> dict:
    model_path = f"{DEFS.anmat_root}/autoscore_model/MODEL.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def create_session_data(
    subject, exp, sb, t1=None, t2=None, store_chans: dict = {"EEG_": [1]}
):
    if t1 is None:
        t1 = 0
    if t2 is None:
        t2 = np.inf
    session = {}
    session["session_id"] = f"{subject}_{exp}_sb-{sb}"
    ephys = wis.peri.ephys.load_single_ephys_block(
        subject, exp, ["EEG_"], sync_block=sb, store_chans=store_chans
    )
    eeg = ephys["EEG_"].sel(time=slice(t1, t2))
    eeg_data = eeg.values
    eeg_timestamps = eeg.time.values
    session["eeg"] = {"signal": eeg_data, "timestamps": eeg_timestamps, "fs": eeg.fs}
    whis = wis.peri.vid.load_whisking_df(subject, exp, sb)
    # if t1 is not None and t2 is not None:
    #    whis = whis.filter(pl.col("time").is_between(t1, t2))
    eye = wis.peri.vid.load_eye_metric_df(subject, exp, sb)
    # if t1 is not None and t2 is not None:
    #    eye = eye.filter(pl.col("frame").is_between(t1, t2))
    vid_timestamps = whis["time"].to_numpy()
    whis_data = whis["whis"].to_numpy()
    diameter = eye["diameter"].to_numpy()
    motion = eye["motion"].to_numpy()
    eyelid = eye["lid"].to_numpy()
    session["pupil"] = {
        "diameter": diameter,
        "motion": motion,
        "eyelid": eyelid,
        "whisking": whis_data,
        "timestamps": vid_timestamps,
        "fs": 10,
    }
    return session


def _autoscore_mi(
    subject,
    exp,
    sync_block,
    store_chans: dict = {"EEG_": [1]},
    overwrite: bool = False,
):
    times = check_for_scoring_times(subject, exp, sync_block)
    if len(times[0]) > 1:
        raise NotImplementedError(
            "Multiple scoring times for a single sync block are not supported yet!"
        )  # TODO: support multiple scoring times
    t1 = times[0][0]
    t2 = times[1][0]
    session = create_session_data(subject, exp, sync_block, t1, t2, store_chans)
    model = load_model()
    bout_df, epoch_df = wis.pipes.infer_sleep_hsmm(model, [session])

    # smooth the probabilities
    epoch_df["NREM_smooth"] = epy.sigpro.gen.smooth_signal(
        epoch_df["P_NREM"].values, 2, fs=1
    )
    epoch_df["REM_smooth"] = epy.sigpro.gen.smooth_signal(
        epoch_df["P_REM"].values, 2, fs=1
    )
    epoch_df["Wake_smooth"] = epy.sigpro.gen.smooth_signal(
        epoch_df["P_Wake"].values, 2, fs=1
    )

    # create a label column based solely on highest probability
    epoch_df["label"] = (
        epoch_df[["NREM_smooth", "REM_smooth", "Wake_smooth"]]
        .idxmax(axis=1)
        .str.replace("_smooth", "")
    )

    hypno_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}/hypnograms/model_labelled"
    wis.util.gen.check_dir(hypno_dir)
    hypno_name = f"raw_epochs.csv"
    if not os.path.exists(f"{hypno_dir}/{hypno_name}") or overwrite:
        epoch_df.to_csv(f"{hypno_dir}/{hypno_name}", index=False)

    bout_name = f"raw_hsmm_bouts.csv"
    if not os.path.exists(f"{hypno_dir}/{bout_name}") or overwrite:
        bout_df.to_csv(f"{hypno_dir}/{bout_name}", index=False)
    return epoch_df, bout_df


def autoscore_mi_all_subjects(overwrite: bool = False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            for sb in si[subject][exp]["sync_blocks"].keys():
                try:
                    _autoscore_mi(subject, exp, sb, overwrite=overwrite)
                except Exception as e:
                    print(f"Error scoring {subject} {exp} sync block-{sb}: {e}")
                    continue
    return
