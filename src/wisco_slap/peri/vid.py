import os

import cv2
import numpy as np
import xarray as xr

import wisco_slap as wis
import wisco_slap.defs as DEFS


def detect_frame_pulses(
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


def check_pulse_times_against_video(
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
    num_frames_true = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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


def generate_pupil_frame_times(subject: str, exp: str, sync_block: int = 1, save=True):
    """Load the camera sync pulses, detect where there were frames, and then match the pulses to the number of frames in the video

    Parameters
    ----------
    subject : str
        the subject name
    exp : str
        the experiment name
    sync_block : int, optional
        the sync block number, by default 1
    save : bool, optional
        whether to save the frame times to a file, by default True. If the file already exists, it will be overwritten.

    Returns
    -------
    np.ndarray
        the frame times
    """
    # first we detect the pulses actually coming from the camera
    e = wis.peri.ephys.load_single_ephys_block(
        subject, exp, stores=["Wav1"], sync_block=sync_block
    )
    t = detect_frame_pulses(e["Wav1"])

    # Then we verify that the number of pulses matches the number of frames in the video
    video_path = f"{DEFS.data_root}/{subject}/{exp}/pupil/pupil-{sync_block}.mp4"
    t_corrected = check_pulse_times_against_video(t, video_path)

    if save:
        save_dir = (
            f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}"
        )
        wis.util.gen.check_dir(save_dir)
        save_path = f"{save_dir}/pupil__frame_times.npy"
        if os.path.exists(save_path):
            os.system(f"rm -rf {save_path}")
        np.save(save_path, t_corrected)
    return t_corrected
