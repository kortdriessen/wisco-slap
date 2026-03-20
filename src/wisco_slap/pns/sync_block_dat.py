import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import tifffile
import xarray as xr

import wisco_slap as wis
import wisco_slap.defs as DEFS


def save_ephys_data(
    subject: str,
    exp: str,
    stores: list[str] | None = None,
    sync_block: int = 1,
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
    """
    if stores is None:
        stores = ["EEG_", "EEGr", "loal"]

    save_dir = os.path.join(
        DEFS.anmat_root,
        subject,
        exp,
        "sync_block_data",
        f"sync_block-{sync_block}",
        "ephys",
    )
    wis.util.check_dir(save_dir)

    e = wis.peri.ephys.load_single_ephys_block(
        subject, exp, stores=stores, sync_block=sync_block
    )
    coords_to_drop = ["timedelta", "datetime"]

    for store in stores:
        data = e[store]
        data = data.drop_vars(coords_to_drop)
        path = os.path.join(save_dir, f"{store}.nc")
        if os.path.exists(path):
            os.remove(path)
        data.to_netcdf(path)
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
        raise RuntimeError("Unexpected error in pulse time checking, please debug")


def detect_and_save_pupil_frame_times(subject: str, exp: str, sync_block: int = 1):
    """Load the camera sync pulses, detect where there were frames, and then match the pulses to the number of frames in the video.

    Parameters
    ----------
    subject : str
        the subject name
    exp : str
        the experiment name
    sync_block : int, optional
        the sync block number, by default 1

    Returns
    -------
    np.ndarray
        the frame times
    """
    save_dir = os.path.join(
        DEFS.anmat_root,
        subject,
        exp,
        "sync_block_data",
        f"sync_block-{sync_block}",
        "pupil",
    )
    wis.util.check_dir(save_dir)
    save_path = os.path.join(save_dir, "frame_times.npy")

    e = wis.peri.ephys.load_single_ephys_block(
        subject, exp, stores=["Wav1"], sync_block=sync_block
    )
    t = _detect_frame_pulses(e["Wav1"])

    # Then we verify that the number of pulses matches the number of frames in the video
    video_path = os.path.join(
        DEFS.data_root, subject, exp, "pupil", f"pupil-{sync_block}.mp4"
    )
    t_corrected = _check_pulse_times_against_video(t, video_path)

    np.save(save_path, t_corrected)
    return t_corrected


# Pupil Inference and Eye metric DataFrames


def _run_pupil_inference(subject: str, exp: str, sb: int):
    """Run DeepLabCut pupil inference for a single sync block.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    sb : int
        sync block number
    """
    script = Path(__file__).parent / "pupil_inference" / "pupil_inference_main.py"
    cmd = ["python", str(script), "--subject", subject, "--exp", exp, "--sb", str(sb)]
    subprocess.run(cmd, check=True)


def _save_eye_metric_df(subject: str, exp: str, sb: int):
    """Compute and save eye metric DataFrame for a sync block.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    sb : int
        sync block number
    """
    eyedir = os.path.join(
        DEFS.anmat_root,
        subject,
        exp,
        "sync_block_data",
        f"sync_block-{sb}",
        "pupil",
        "eye_metrics",
    )
    wis.util.check_dir(eyedir)
    save_path = os.path.join(eyedir, "eye_metrics.parquet")
    frame_time_path = os.path.join(
        DEFS.anmat_root,
        subject,
        exp,
        "sync_block_data",
        f"sync_block-{sb}",
        "pupil",
        "frame_times.npy",
    )
    frame_times = np.load(frame_time_path)
    eye = wis.peri.vid.compute_eye_metric_df(subject, exp, sb)
    if len(eye) != len(frame_times):
        print(
            f"Length of eye metric df: {len(eye)}, length of frame times: {len(frame_times)}"
        )
        raise ValueError(
            f"Eye metric df for {subject} {exp} sync block-{sb} does not match number of frame times, see above"
        )
    eye = eye.with_columns(time=pl.lit(frame_times))
    eye.write_parquet(save_path)
    return


def _load_and_save_video_frame_stack(mp4_path, out_path, n_frames):
    """Load an MP4, sample evenly spaced frames, and save as a TIFF stack.

    Parameters
    ----------
    mp4_path : str | os.PathLike
        Path to input .mp4 video
    out_path : str | os.PathLike
        Path to output .tif/.tiff stack (will be overwritten if exists)
    n_frames : int
        Number of frames to extract, evenly spaced across the entire video
    """

    if n_frames is None or int(n_frames) <= 0:
        raise ValueError("n_frames must be a positive integer")

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {mp4_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("Video has zero frames or frame count is unavailable")

        if n_frames > total_frames:
            msg = (
                f"Requested n_frames ({n_frames}) exceeds total frames in video "
                f"({total_frames})"
            )
            raise ValueError(msg)

        # Choose indices as segment midpoints to ensure even coverage and uniqueness
        # idx_k = floor(((k + 0.5) * total_frames) / n_frames)
        k = np.arange(int(n_frames), dtype=float)
        indices = np.floor(((k + 0.5) * float(total_frames)) / float(n_frames)).astype(
            int
        )
        indices = np.clip(indices, 0, total_frames - 1)

        # Read selected frames
        selected_frames = []
        frame_shape = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise OSError(f"Failed to read frame at index {idx}")

            # Convert to grayscale for single-channel TIFF pages
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if frame_shape is None:
                frame_shape = frame.shape
            else:
                if frame.shape != frame_shape:
                    msg = (
                        "Inconsistent frame shapes encountered: "
                        f"{frame.shape} vs {frame_shape}"
                    )
                    raise ValueError(msg)

            # Ensure uint8
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8, copy=False)

            selected_frames.append(frame)

        stack = np.stack(selected_frames, axis=0)  # (n_frames, H, W)

        # Ensure output directory exists
        out_dir = os.path.dirname(str(out_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Write multipage TIFF
        tifffile.imwrite(
            str(out_path),
            stack,
            photometric="minisblack",
        )
    finally:
        cap.release()


def save_video_frame_stack(subject: str, exp: str, sb: int, n_frames: int = 20):
    """Extract evenly-spaced frames from the pupil video and save as a TIFF stack.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    sb : int
        sync block number
    n_frames : int, optional
        number of frames to extract, by default 20
    """
    mp4_path = os.path.join(DEFS.data_root, subject, exp, "pupil", f"pupil-{sb}.mp4")
    out_dir = os.path.join(
        DEFS.anmat_root,
        subject,
        exp,
        "sync_block_data",
        f"sync_block-{sb}",
        "whisking",
    )
    wis.util.check_dir(out_dir)
    out_path = os.path.join(out_dir, "video_frame_stack.tif")
    _load_and_save_video_frame_stack(mp4_path, out_path, n_frames)
    print(f"Saved video frame stack to {out_path}")


def _generate_whisking_frame_differences(video_path, mask):
    mask_bool = np.asarray(mask).astype(bool)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    diffs = []
    prev_gray = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            if gray.shape[:2] != mask_bool.shape[:2]:
                raise ValueError(
                    f"Mask shape {mask_bool.shape} does not match frame shape {gray.shape}"
                )

            gray = gray.astype(np.float32, copy=False)

            if prev_gray is None:
                diffs.append(0.0)
            else:
                diff_val = np.mean(np.abs(gray - prev_gray)[mask_bool])
                diffs.append(float(diff_val))

            prev_gray = gray
    finally:
        cap.release()

    return np.array(diffs, dtype=float)


def save_whisking_frame_differences(subject: str, exp: str, sb: int):
    """Compute frame-by-frame whisking motion and save as a parquet file.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    sb : int
        sync block number
    """
    video_path = os.path.join(DEFS.data_root, subject, exp, "pupil", f"pupil-{sb}.mp4")
    whisk_dir = os.path.join(
        DEFS.anmat_root,
        subject,
        exp,
        "sync_block_data",
        f"sync_block-{sb}",
        "whisking",
    )
    mask_path = os.path.join(whisk_dir, "mask.tif")
    mask = tifffile.imread(mask_path)
    mask = mask[0]

    frame_time_path = os.path.join(
        DEFS.anmat_root,
        subject,
        exp,
        "sync_block_data",
        f"sync_block-{sb}",
        "pupil",
        "frame_times.npy",
    )
    frame_times = np.load(frame_time_path)

    out_path = os.path.join(whisk_dir, "whisk_df.parquet")
    whis = _generate_whisking_frame_differences(video_path, mask)
    if len(whis) != len(frame_times):
        print(
            f"Length of whisking frame differences: {len(whis)}, "
            f"length of frame times: {len(frame_times)}"
        )
        raise ValueError(
            f"Whisking frame differences for {subject} {exp} sync block-{sb} "
            f"does not match number of frame times, see above"
        )
    df = pl.DataFrame({"whis": whis, "time": frame_times})
    df.write_parquet(out_path)
