import os
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import xarray as xr

import wisco_slap as wis
import wisco_slap.defs as DEFS


def _generate_ephys_scoring_data(
    subject: str,
    exp: str,
    stores: list[str] = None,
    sync_block: int = 1,
    overwrite=False,
):
    """Generate and save ephys data for sleep scoring.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    stores : list[str], optional
        stores to generate data for, by default None
    sync_block : int, optional
        sync block number, by default 1
    overwrite : bool, optional
        whether to overwrite existing files, by default False
    """
    if stores is None:
        stores = ["EEG_", "loal"]

    save_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}"
    wis.util.gen.check_dir(save_dir)

    stores_exist = []
    # for each store, check if the time and data files exist, at least for channel 0
    # if the time and data files already exist for each store, and overwrite is True,
    # we just delete everything in the save_dir (that is not a directory
    # or the frame times file) and continue the function, otherwise if even channel 0
    # for all stores exist, we just return
    for store in stores:
        t_path = f"{save_dir}/{store}--ch0_t.npy"
        d_path = f"{save_dir}/{store}--ch0_y.npy"
        if os.path.exists(t_path) and os.path.exists(d_path):
            stores_exist.append(store)
    if len(stores_exist) == len(stores) and not overwrite:
        print(f"All stores for {subject} {exp} sync block-{sync_block} already exist")
        return
    elif len(stores_exist) == len(stores) and overwrite:
        for f in os.listdir(save_dir):
            # make sure f is not a directory
            if not os.path.isdir(f"{save_dir}/{f}"):
                # make sure f is not the frame times file
                if "pupil__frame_times" not in f:
                    os.system(f"rm -rf {save_dir}/{f}")

    e = wis.peri.ephys.load_single_ephys_block(
        subject, exp, stores=stores, sync_block=sync_block
    )

    for store in stores:
        data = e[store]
        if "channel" in data.dims:
            for channel in data.channel.values:
                data_channel = data.sel(channel=channel)
                t = data_channel["time"].values
                d = data_channel.values
                assert len(t) == len(d)
                t_path = f"{save_dir}/{store}--ch{channel}_t.npy"
                d_path = f"{save_dir}/{store}--ch{channel}_y.npy"
                np.save(t_path, t)
                np.save(d_path, d)
        else:
            t = data["time"].values
            d = data.values
            assert len(t) == len(d)
            t_path = f"{save_dir}/{store}--ch0_t.npy"
            d_path = f"{save_dir}/{store}--ch0_y.npy"
            np.save(t_path, t)
            np.save(d_path, d)
    return


def _detect_frame_pulses(
    wav: xr.DataArray, factor: int = 3, small_diff_level: int = 20
) -> np.ndarray:
    """Detect the sync pulses recorded by Rz2, coming from the FLIR camera recording mouse/pupil

    Parameters
    ----------
    wav : xr.DataArray
        The store (from TDT) that records the sync pulses. Should be a DataArray loaded with electro_py.tdt.io.get_data()
    factor : int
        The factor by which to set the detection threshold. The maximum difference in the sync signal is divided by this factor to set the detection threshold.
    small_diff_level : int
        any difference smaller than this is considered a small difference and is discarded
    Returns
    -------
    np.ndarray
        the times where a pulse is detected
    """
    raw_sig = wav.values
    raw_times = wav.time.values
    sig_diff = np.diff(raw_sig)
    threshold = sig_diff.max() / factor
    spike_indices = np.where(sig_diff > threshold)
    spike_distances = np.diff(spike_indices[0])
    small_diff_indices = np.where(spike_distances < small_diff_level)
    spikes_to_toss = small_diff_indices[0] + 1
    spike_mask = ~np.isin(np.arange(len(spike_indices[0])), spikes_to_toss)
    frame_ixs = spike_indices[0][spike_mask]
    frame_ixs = frame_ixs + 1
    frame_times = raw_times[frame_ixs[1:]]  # discard the first frame time
    return frame_times


def _check_pulse_times_against_video(
    pulse_times: np.ndarray, path_to_video: str
) -> np.ndarray:
    """Check the pulse times against the number of frames in the video to see if they match

    Parameters
    ----------
    pulse_times : np.ndarray
        times of detected pulses, as returned by detect_frame_pulses
    path_to_video : str
        path to the video file

    Returns
    -------
    np.ndarray
        the pulse times, adjusted to match the number of frames in the video if needed.
    """
    cap = cv2.VideoCapture(path_to_video)
    video_dir = os.path.dirname(path_to_video)
    num_frames_true = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wis.peri.vid.record_camera_sync_mismatch(
        video_dir, num_frames_true, len(pulse_times)
    )
    print(f"number of frames in video: {num_frames_true}")
    print(f"number of pulse times: {len(pulse_times)}")
    if len(pulse_times) == num_frames_true:
        print("pulse times match expected number of frames")
        return pulse_times

    elif len(pulse_times) > num_frames_true:
        print(
            "pulse times are greater than expected number of frames, trimming extra pulses at the end"
        )
        return pulse_times[:num_frames_true]

    elif len(pulse_times) < num_frames_true:
        print(
            "pulse times are less than expected number of frames, adding extra frame times at the end"
        )
        num_extra_frames = num_frames_true - len(pulse_times)
        final_pulse_time = pulse_times[-1]
        estimated_fps_interval = np.mean(np.diff(pulse_times))
        new_frames_start = final_pulse_time + estimated_fps_interval
        extra_times = np.arange(
            new_frames_start,
            new_frames_start + (num_extra_frames * estimated_fps_interval),
            estimated_fps_interval,
        )
        return np.concatenate([pulse_times, extra_times])
    else:
        print("unexpected error, needs debugging")
        return None


def _detect_and_save_pupil_frame_times(
    subject: str, exp: str, sync_block: int = 1, overwrite=False
):
    """Load the camera sync pulses, detect where there were frames, and then match the pulses to the number of frames in the video

    Parameters
    ----------
    subject : str
        the subject name
    exp : str
        the experiment name
    sync_block : int, optional
        the sync block number, by default 1
    overwrite : bool, optional
        whether to overwrite the frame times file if it already exists

    Returns
    -------
    np.ndarray
        the frame times
    """
    # first we detect the pulses actually coming from the camera

    save_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}"
    wis.util.gen.check_dir(save_dir)
    save_path = f"{save_dir}/pupil__frame_times.npy"
    if os.path.exists(save_path) and not overwrite:
        print(f"File {save_path} already exists. Use overwrite=True to overwrite.")
        return
    if os.path.exists(save_path) and overwrite:
        os.system(f"rm -rf {save_path}")

    e = wis.peri.ephys.load_single_ephys_block(
        subject, exp, stores=["Wav1"], sync_block=sync_block
    )
    t = _detect_frame_pulses(e["Wav1"])

    # Then we verify that the number of pulses matches the number of frames in the video
    video_path = f"{DEFS.data_root}/{subject}/{exp}/pupil/pupil-{sync_block}.mp4"
    t_corrected = _check_pulse_times_against_video(t, video_path)

    np.save(save_path, t_corrected)
    return t_corrected


def _save_ephys_and_frame_times_for_scoring(subject, exp, sync_block, overwrite=False):
    _generate_ephys_scoring_data(
        subject, exp, sync_block=sync_block, overwrite=overwrite
    )
    _detect_and_save_pupil_frame_times(
        subject, exp, sync_block=sync_block, overwrite=overwrite
    )
    return


def save_ephys_and_frame_times_all_subjects(overwrite=False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            for sb in si[subject][exp]["sync_blocks"].keys():
                try:
                    _save_ephys_and_frame_times_for_scoring(
                        subject, exp, sb, overwrite=overwrite
                    )
                except Exception as e:
                    print(
                        f"Error saving ephys and frame times for {subject} {exp} sync block-{sb}: {e}"
                    )
                    continue
    return


def _save_eye_traces_for_scoring(
    subject, exp, sb, overwrite=False, likelihood_threshold=0.2
):
    eyedir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sb}/eye"
    wis.util.gen.check_dir(eyedir)
    eye_mets = ["dia", "motion", "lid"]
    met_paths = [
        Path(
            f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sb}/eye/{met}_y.npy"
        )
        for met in eye_mets
    ]

    if all(path.exists() for path in met_paths) and not overwrite:
        print(f"Eye traces for {subject} {exp} sync block-{sb} already exist")
        return
    elif all(path.exists() for path in met_paths) and overwrite:
        # delete everything in the directory
        for f in os.listdir(
            f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sb}/eye"
        ):
            os.remove(
                f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sb}/eye/{f}"
            )

    # load the eye metric df and filter out low likelihood data
    try:
        edf = wis.peri.vid.load_eye_metric_df(subject, exp, sb)
    except FileNotFoundError:
        print(
            f"Eye metric df for {subject} {exp} sync block-{sb} do not exist, save it before saving scoring data!"
        )
        return

    pup = edf["frame", "diameter", "motion", "pup_likelihood"]
    lid = edf["frame", "lid", "lid_norm", "lid_likelihood"]
    for col in ["diameter", "motion"]:
        pup = pup.with_columns(
            pl.when(pl.col("pup_likelihood") < likelihood_threshold)
            .then(pl.lit(np.nan))
            .otherwise(pl.col(col))
            .alias(col)
        )
    for col in ["lid", "lid_norm"]:
        lid = lid.with_columns(
            pl.when(pl.col("lid_likelihood") < likelihood_threshold)
            .then(pl.lit(np.nan))
            .otherwise(pl.col(col))
            .alias(col)
        )

    # save the filtered data and frame time copies
    dia = pup["diameter"].to_numpy()
    motion = pup["motion"].to_numpy()
    lid = lid["lid"].to_numpy()
    time = edf["time"].to_numpy()
    np.save(f"{eyedir}/dia_y.npy", dia)
    np.save(f"{eyedir}/motion_y.npy", motion)
    np.save(f"{eyedir}/lid_y.npy", lid)
    np.save(f"{eyedir}/dia_t.npy", time)
    np.save(f"{eyedir}/motion_t.npy", time)
    np.save(f"{eyedir}/lid_t.npy", time)
    return


def save_eye_traces_for_scoring_all_subjects(overwrite=False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            for sb in si[subject][exp]["sync_blocks"].keys():
                try:
                    _save_eye_traces_for_scoring(subject, exp, sb, overwrite=overwrite)
                except Exception as e:
                    print(
                        f"Error saving eye traces for {subject} {exp} sync block-{sb}: {e}"
                    )
                    continue
    return


def _save_whisking_traces_for_scoring(subject, exp, sync_block, overwrite=False):
    save_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}/whisking"
    wis.util.gen.check_dir(save_dir)

    whis_path = f"{save_dir}/whis_y.npy"
    time_path = f"{save_dir}/whis_t.npy"
    if os.path.exists(whis_path) and os.path.exists(time_path) and not overwrite:
        print(
            f"Whisking traces for {subject} {exp} sync block-{sync_block} already exist"
        )
        return
    elif os.path.exists(whis_path) and os.path.exists(time_path) and overwrite:
        os.system(f"rm -rf {whis_path}")
        os.system(f"rm -rf {time_path}")
    try:
        whisk_df = wis.peri.vid.load_whisking_df(subject, exp, sync_block)
    except FileNotFoundError:
        print(
            f"Whisking df for {subject} {exp} sync block-{sync_block} does not exist, save it before saving scoring data!"
        )
        return
    whisk = whisk_df["whis"].to_numpy()
    time = whisk_df["time"].to_numpy()
    np.save(whis_path, whisk)
    np.save(time_path, time)
    return


def save_whisking_traces_for_scoring_all_subjects(overwrite=False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            for sb in si[subject][exp]["sync_blocks"].keys():
                try:
                    _save_whisking_traces_for_scoring(
                        subject, exp, sb, overwrite=overwrite
                    )
                except Exception as e:
                    print(
                        f"Error saving whisking traces for {subject} {exp} sync block-{sb}: {e}"
                    )
                    continue
    return


def all_subjects_full_peripheral_scoring_data_pipeline(overwrite=False):
    save_ephys_and_frame_times_all_subjects(overwrite=overwrite)
    save_eye_traces_for_scoring_all_subjects(overwrite=overwrite)
    save_whisking_traces_for_scoring_all_subjects(overwrite=overwrite)
    return


def save_peripheral_scoring_data(subject, exp, overwrite=False):
    si = wis.peri.sync.load_sync_info()
    for sb in si[subject][exp]["sync_blocks"].keys():
        _save_ephys_and_frame_times_for_scoring(subject, exp, sb, overwrite=overwrite)
        _save_eye_traces_for_scoring(subject, exp, sb, overwrite=overwrite)
        _save_whisking_traces_for_scoring(subject, exp, sb, overwrite=overwrite)
    return


def write_roi_traces_for_scoring(
    subject, exp, loc, acq, roi_version="Fsvd", roi_channel=1, overwrite=False
):
    roidf = wis.scope.io.load_roidf(
        subject, exp, loc, acq, roi_version=roi_version, channel=roi_channel
    )
    act_key = f"data"
    acq_id = f"{loc}--{acq}"
    si = wis.peri.sync.load_sync_info()
    ephys_offset = si[subject][exp]["acquisitions"][acq_id]["ephys_offset"]
    sync_block = si[subject][exp]["acquisitions"][acq_id]["sync_block"]
    save_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}/scope_traces/soma_rois/{acq_id}"
    wis.util.gen.check_dir(save_dir)
    locletter = loc.split("_")[1]
    acqnumber = acq.split("_")[1]
    for dmd in [1, 2]:
        dmd_df = roidf.filter(pl.col("dmd") == dmd)
        if len(dmd_df) == 0:
            continue
        for roi in dmd_df["soma-ID"].unique():
            data_path = f"{save_dir}/{locletter}{acqnumber}{roi}-dmd{dmd}_y.npy"
            t_path = f"{save_dir}/{locletter}{acqnumber}{roi}-dmd{dmd}_t.npy"
            if os.path.exists(data_path) and os.path.exists(t_path) and not overwrite:
                continue
            elif os.path.exists(data_path) and os.path.exists(t_path) and overwrite:
                os.system(f"rm -rf {data_path}")
                os.system(f"rm -rf {t_path}")
            roi_df = dmd_df.filter(pl.col("soma-ID") == roi)
            roi_data = roi_df[act_key].to_numpy()
            roi_times = roi_df["time"].to_numpy() + ephys_offset
            np.save(data_path, roi_data)
            np.save(t_path, roi_times)
    return


def write_roi_traces_for_scoring_all_subjects(overwrite=False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            acq_ids = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
            for acq_id in acq_ids:
                loc, acq = acq_id.split("--")
                try:
                    write_roi_traces_for_scoring(
                        subject, exp, loc, acq, overwrite=overwrite
                    )
                except Exception as e:
                    print(
                        f"Error writing ROI traces for {subject} {exp} {loc} {acq}: {e}"
                    )
                    continue
    return


def save_glutamate_sums_for_scoring(
    subject, exp, loc, acq, min_sources=10, overwrite=False
):
    sb = wis.peri.sync.get_acq_sync_block(subject, exp, loc, acq)
    sum_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sb}/scope_traces/synapses/{loc}--{acq}/glutamate_sums"
    if os.path.exists(sum_dir) and not overwrite:
        # make sure the dir is not empty
        if len(os.listdir(sum_dir)) > 0:
            print(
                f"Glutamate sums already exist: {sum_dir}, use overwrite=True to overwrite"
            )
            return
    else:
        os.system(f"rm -rf {sum_dir}")
        wis.util.gen.check_dir(sum_dir)

    df = wis.scope.io.load_syndf(subject, exp, loc, acq, apply_ephys_offset=True)
    df = df.with_columns((pl.col("noise") * 5).alias("sd5"))
    df = df.with_columns(
        pl.when(pl.col("data") > pl.col("sd5"))
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("active")
    )
    wis.util.gen.check_dir(sum_dir)
    min_sources = 10
    if "soma-ID" not in df.columns:
        raise ValueError(
            f"soma-ID column not found in df for {subject} {exp} {loc} {acq}"
        )
    for sid in df["soma-ID"].unique():
        if "unidentifiable" in sid:
            continue
        d = df.filter(pl.col("soma-ID") == sid)
        n_unique_combinations = d.select(["dmd", "source-ID"]).n_unique()
        if n_unique_combinations < min_sources:
            continue
        glut_sums = d.group_by("time").agg(pl.sum("data"))
        glut_sums = glut_sums.sort("time")
        data = glut_sums["data"].to_numpy()
        time = glut_sums["time"].to_numpy()
        name = f"{sid}-{n_unique_combinations}"
        np.save(f"{sum_dir}/{name}_y.npy", data)
        np.save(f"{sum_dir}/{name}_t.npy", time)

    sum_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sb}/scope_traces/synapses/{loc}--{acq}/glutamate_sums/fracactive"
    wis.util.gen.check_dir(sum_dir)

    for sid in df["soma-ID"].unique():
        if "unidentifiable" in sid:
            continue
        d = df.filter(pl.col("soma-ID") == sid)
        n_unique_combinations = d.select(["dmd", "source-ID"]).n_unique()
        if n_unique_combinations < min_sources:
            continue
        glut_sums = d.group_by("time").agg(pl.sum("active"))
        glut_sums = glut_sums.sort("time")
        data = glut_sums["active"].to_numpy()
        data = (data / n_unique_combinations) * 100
        time = glut_sums["time"].to_numpy()
        name = f"{sid}-{n_unique_combinations}"
        np.save(f"{sum_dir}/{name}_y.npy", data)
        np.save(f"{sum_dir}/{name}_t.npy", time)


def save_glutamate_sums_all_subjects(overwrite=False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            acq_ids = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
            for acq_id in acq_ids:
                try:
                    print(f"Working on {subject} {exp} {acq_id}")
                    loc, acq = acq_id.split("--")
                    save_glutamate_sums_for_scoring(
                        subject, exp, loc, acq, overwrite=overwrite
                    )
                except Exception as e:
                    print(
                        f"Error saving glutamate sums for {subject} {exp} {loc} {acq}: {e}"
                    )
                    continue


def full_scoring_data_pipeline(subject, exp, loc, acq, overwrite=False):
    save_peripheral_scoring_data(subject, exp, overwrite=overwrite)
    write_roi_traces_for_scoring(subject, exp, loc, acq, overwrite=overwrite)
    save_glutamate_sums_for_scoring(subject, exp, loc, acq, overwrite=overwrite)
    return
