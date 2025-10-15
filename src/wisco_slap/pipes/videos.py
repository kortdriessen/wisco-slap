import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import polars as pl
import tifffile

import wisco_slap as wis
import wisco_slap.defs as DEFS


def _run_pupil_inference(subject, exp, redo=False):
    if redo:
        os.system(
            f"python {Path(__file__).parent}/pupil_inference/pupil_inference_main.py --subject {subject} --exp {exp} --redo"
        )
    else:
        os.system(
            f"python {Path(__file__).parent}/pupil_inference/pupil_inference_main.py --subject {subject} --exp {exp}"
        )
    return


def run_pupil_inference_all_subjects(redo=False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            try:
                _run_pupil_inference(subject, exp, redo=redo)
            except Exception as e:
                print(f"Error running pupil inference for {subject} {exp}: {e}")
                continue
    return


def _save_eye_metric_df(subject, exp, sb, redo=False):
    eyedir = f"{DEFS.anmat_root}/{subject}/{exp}/pupil_inference/eye_metrics"
    wis.util.gen.check_dir(eyedir)
    save_path = f"{eyedir}/eye_metrics-{sb}.parquet"
    frame_time_path = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sb}/pupil__frame_times.npy"
    if not os.path.exists(frame_time_path):
        print(f"Frame time path for {subject} {exp} sync block-{sb} does not exist")
        return
    frame_times = np.load(frame_time_path)
    if os.path.exists(save_path) and not redo:
        print(f"Eye metric df for {subject} {exp} sync block-{sb} already exists")
        return
    elif os.path.exists(save_path) and redo:
        os.system(f"rm -rf {save_path}")
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


def _save_full_exp_eye_dfs(subject, exp, redo=False):
    si = wis.peri.sync.load_sync_info()
    for sb in si[subject][exp]["sync_blocks"].keys():
        _save_eye_metric_df(subject, exp, sb, redo=redo)
    return


def save_eye_dfs_all_subjects(redo=False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            try:
                _save_full_exp_eye_dfs(subject, exp, redo=redo)
            except Exception as e:
                print(f"Error saving eye dfs for {subject} {exp}: {e}")
                continue
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


def _save_video_frame_stack(subject, exp, sb, n_frames=20, redo=False):
    mp4_path = f"{DEFS.data_root}/{subject}/{exp}/pupil/pupil-{sb}.mp4"
    out_dir = f"{DEFS.anmat_root}/{subject}/{exp}/whisking"
    wis.util.gen.check_dir(out_dir)
    out_path = f"{out_dir}/video_frame_stack-{sb}.tif"
    if os.path.exists(out_path) and not redo:
        print(f"Video frame stack for {subject} {exp} sync block-{sb} already exists")
        return
    elif os.path.exists(out_path) and redo:
        os.system(f"rm -rf {out_path}")
    _load_and_save_video_frame_stack(mp4_path, out_path, n_frames)
    print(f"Saved video frame stack to {out_path}")
    return


def save_video_frame_stack_all_subjects(n_frames=20, redo=False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            for sb in si[subject][exp]["sync_blocks"].keys():
                try:
                    _save_video_frame_stack(
                        subject, exp, sb, n_frames=n_frames, redo=redo
                    )
                except Exception as e:
                    print(
                        f"Error saving video frame stack for {subject} {exp} sync block-{sb}: {e}"
                    )
                    continue
    return


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


def _save_whisking_frame_differences(subject, exp, sb, redo=False):
    video_path = f"{DEFS.data_root}/{subject}/{exp}/pupil/pupil-{sb}.mp4"
    mask_path = f"{DEFS.anmat_root}/{subject}/{exp}/whisking/mask-{sb}.tif"
    if not os.path.exists(mask_path):
        print(f"Mask for {subject} {exp} sync block-{sb} does not exist")
        return
    mask = tifffile.imread(mask_path)
    mask = mask[0]

    frame_time_path = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sb}/pupil__frame_times.npy"
    if not os.path.exists(frame_time_path):
        print(f"Frame time path for {subject} {exp} sync block-{sb} does not exist")
        return
    frame_times = np.load(frame_time_path)

    out_path = f"{DEFS.anmat_root}/{subject}/{exp}/whisking/whisk_df-{sb}.parquet"
    if os.path.exists(out_path) and not redo:
        print(
            f"Whisking frame differences for {subject} {exp} sync block-{sb} already exists"
        )
        return
    elif os.path.exists(out_path) and redo:
        os.system(f"rm -rf {out_path}")
    whis = _generate_whisking_frame_differences(video_path, mask)
    if len(whis) != len(frame_times):
        print(
            f"Length of whisking frame differences: {len(whis)}, length of frame times: {len(frame_times)}"
        )
        raise ValueError(
            f"Whisking frame differences for {subject} {exp} sync block-{sb} does not match number of frame times, see above"
        )
    df = pd.DataFrame({"whis": whis, "time": frame_times})
    df.to_parquet(out_path)
    return


def save_whisking_frame_differences_all_subjects(redo=False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            for sb in si[subject][exp]["sync_blocks"].keys():
                try:
                    _save_whisking_frame_differences(subject, exp, sb, redo=redo)
                except Exception as e:
                    print(
                        f"Error saving whisking frame differences for {subject} {exp} sync block-{sb}: {e}"
                    )
                    continue
    return
