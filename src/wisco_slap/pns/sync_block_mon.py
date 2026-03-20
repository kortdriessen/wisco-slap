"""Monitor and orchestrate sync block data generation.

This module is the sole decision-maker for what sync block outputs need to be
generated. It checks what exists, respects dependency ordering, and calls the
pure generator functions in sync_block_dat.py when outputs are missing.

The generating functions in sync_block_dat.py should NOT contain any existence
checks or overwrite logic — they simply produce output when called.
"""

import os

import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.pns import sync_block_dat
from wisco_slap.pns.pupil_inference.pupil_inference_main import check_inference_done

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _sb_dir(subject: str, exp: str, sb: int) -> str:
    """Return the root sync_block_data directory for a given sync block."""
    return os.path.join(
        DEFS.anmat_root, subject, exp, "sync_block_data", f"sync_block-{sb}"
    )


# ---------------------------------------------------------------------------
# Check functions — each returns True if the output exists and is complete
# ---------------------------------------------------------------------------


def _check_ephys(
    subject: str, exp: str, sb: int, stores: list[str] | None = None
) -> bool:
    """Check whether all ephys .nc store files exist."""
    if stores is None:
        stores = ["EEGr", "EEG_", "loal"]
    ephys_dir = os.path.join(_sb_dir(subject, exp, sb), "ephys")
    if not os.path.isdir(ephys_dir):
        return False
    return all(
        os.path.isfile(os.path.join(ephys_dir, f"{store}.nc")) for store in stores
    )


def _check_frame_times(subject: str, exp: str, sb: int) -> bool:
    """Check whether pupil frame_times.npy exists."""
    path = os.path.join(_sb_dir(subject, exp, sb), "pupil", "frame_times.npy")
    return os.path.isfile(path)


def _check_pupil_inference(subject: str, exp: str, sb: int) -> bool:
    """Check whether DLC inference CSV exists for this sync block."""
    pupil_video_path = os.path.join(
        DEFS.data_root, subject, exp, "pupil", f"pupil-{sb}.mp4"
    )
    return check_inference_done(pupil_video_path, subject, exp, sb)


def _check_eye_metrics(subject: str, exp: str, sb: int) -> bool:
    """Check whether eye_metrics.parquet exists."""
    path = os.path.join(
        _sb_dir(subject, exp, sb), "pupil", "eye_metrics", "eye_metrics.parquet"
    )
    return os.path.isfile(path)


def _check_video_frame_stack(subject: str, exp: str, sb: int) -> bool:
    """Check whether the whisking video_frame_stack.tif exists."""
    path = os.path.join(_sb_dir(subject, exp, sb), "whisking", "video_frame_stack.tif")
    return os.path.isfile(path)


def _check_whisk_mask(subject: str, exp: str, sb: int) -> bool:
    """Check whether the manually-created whisking mask.tif exists."""
    path = os.path.join(_sb_dir(subject, exp, sb), "whisking", "mask.tif")
    return os.path.isfile(path)


def _check_whisk_df(subject: str, exp: str, sb: int) -> bool:
    """Check whether whisk_df.parquet exists."""
    path = os.path.join(_sb_dir(subject, exp, sb), "whisking", "whisk_df.parquet")
    return os.path.isfile(path)


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def sb_data(subject: str, exp: str, sb: int, overwrite: bool = False) -> None:
    """Check and generate all sync block outputs in dependency order.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    sb : int
        sync block number
    overwrite : bool, optional
        if True, regenerate all outputs even if they already exist
    """
    tag = f"{subject} {exp} sb-{sb}"

    # 1. Ephys (independent)
    if overwrite or not _check_ephys(subject, exp, sb):
        print(f"[{tag}] Generating ephys data...")
        sync_block_dat.save_ephys_data(subject, exp, sync_block=sb)

    # 2. Frame times (independent)
    if overwrite or not _check_frame_times(subject, exp, sb):
        print(f"[{tag}] Detecting and saving pupil frame times...")
        sync_block_dat.detect_and_save_pupil_frame_times(subject, exp, sync_block=sb)

    # 3. Pupil inference (depends on frame_times)
    if overwrite or not _check_pupil_inference(subject, exp, sb):
        if _check_frame_times(subject, exp, sb):
            print(f"[{tag}] Running pupil inference...")
            sync_block_dat._run_pupil_inference(subject, exp, sb)
        else:
            print(f"[{tag}] Skipping pupil inference — frame_times missing")

    # 4. Eye metrics (depends on frame_times + pupil_inference)
    if overwrite or not _check_eye_metrics(subject, exp, sb):
        if _check_frame_times(subject, exp, sb) and _check_pupil_inference(
            subject, exp, sb
        ):
            print(f"[{tag}] Saving eye metric dataframe...")
            sync_block_dat._save_eye_metric_df(subject, exp, sb)
        else:
            print(
                f"[{tag}] Skipping eye metrics — "
                f"frame_times exists: {_check_frame_times(subject, exp, sb)}, "
                f"pupil inference exists: {_check_pupil_inference(subject, exp, sb)}"
            )

    # 5. Video frame stack (independent)
    if overwrite or not _check_video_frame_stack(subject, exp, sb):
        print(f"[{tag}] Saving video frame stack...")
        sync_block_dat.save_video_frame_stack(subject, exp, sb)

    # 6. Whisk df (depends on frame_times + mask)
    if overwrite or not _check_whisk_df(subject, exp, sb):
        has_frame_times = _check_frame_times(subject, exp, sb)
        has_mask = _check_whisk_mask(subject, exp, sb)
        if has_frame_times and has_mask:
            print(f"[{tag}] Saving whisking frame differences...")
            sync_block_dat.save_whisking_frame_differences(subject, exp, sb)
        elif not has_mask:
            print(
                f"[{tag}] Skipping whisk_df — mask.tif not found (manual step required)"
            )
        else:
            print(f"[{tag}] Skipping whisk_df — frame_times missing")


def exp_data(subject: str, exp: str, overwrite: bool = False) -> None:
    """Run sb_data for all sync blocks in an experiment.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    overwrite : bool, optional
        if True, regenerate all outputs even if they already exist
    """
    si = wis.meta.get.sync_info()
    for sb in si[subject][exp]["sync_blocks"].keys():
        sb_data(subject, exp, sb, overwrite=overwrite)


def all_subjects(overwrite: bool = False) -> None:
    """Run exp_data for all subjects and experiments.

    Parameters
    ----------
    overwrite : bool, optional
        if True, regenerate all outputs even if they already exist
    """
    si = wis.meta.get.sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            try:
                exp_data(subject, exp, overwrite=overwrite)
            except Exception as e:
                print(f"[{subject} {exp}] Error processing sync block data: {e}")
