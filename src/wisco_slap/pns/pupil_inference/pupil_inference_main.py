import argparse
import os
import subprocess

import wisco_slap as wis
import wisco_slap.defs as DEFS

DLC_TAG = "DLC_Resnet50_dlc_slap_pupilSep23shuffle0_snapshot_best-60"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepLabCut analyze_videos")
    parser.add_argument(
        "--subject",
        required=True,
        help="Subject name",
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment name",
    )
    parser.add_argument(
        "--sb",
        type=int,
        default=None,
        help="Sync block number. If omitted, processes all sync blocks.",
    )
    return parser.parse_args()


def _get_inference_dir(subject: str, exp: str, sb: int | str) -> str:
    return os.path.join(
        DEFS.anmat_root, subject, exp,
        "sync_block_data", f"sync_block-{sb}", "pupil",
    )


def check_inference_done(
    pupil_video_path: str, subject: str, exp: str, sb: int | str
) -> bool:
    """Check whether DLC inference output CSV exists for a given pupil video."""
    pvn = os.path.splitext(os.path.basename(pupil_video_path))[0]
    csv_name = f"{pvn}{DLC_TAG}.csv"
    inference_dir = _get_inference_dir(subject, exp, sb)
    csv_path = os.path.join(inference_dir, csv_name)
    return os.path.exists(csv_path)


def remove_inference_files(
    pupil_video_path: str, subject: str, exp: str, sb: int | str
) -> None:
    """Remove existing DLC inference files for a given pupil video."""
    pvn = os.path.splitext(os.path.basename(pupil_video_path))[0]
    full_id = f"{pvn}{DLC_TAG}"
    inference_dir = _get_inference_dir(subject, exp, sb)
    if not os.path.isdir(inference_dir):
        return
    for f in os.listdir(inference_dir):
        if full_id in f:
            os.remove(os.path.join(inference_dir, f))


def _run_inference_for_sb(
    subject: str, exp: str, sb: int | str, runner_script: str,
    dlc_config_path: str, dlc_env_py: str,
) -> None:
    """Run DLC inference for a single sync block, removing old files first."""
    pupil_path = os.path.join(
        DEFS.data_root, subject, exp, "pupil", f"pupil-{sb}.mp4"
    )
    # Clean up any existing inference files before re-running
    remove_inference_files(pupil_path, subject, exp, sb)

    dest_folder = _get_inference_dir(subject, exp, sb)
    wis.util.check_dir(dest_folder)

    cmd = [
        dlc_env_py,
        runner_script,
        "--config_path",
        dlc_config_path,
        "--videos",
        pupil_path,
        "--destfolder",
        dest_folder,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    dlc_config_path = DEFS.dlc_config_path
    dlc_env_py = DEFS.dlc_env_py

    args = parse_args()
    subject = args.subject
    exp = args.exp

    runner_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_pup_inf.py"
    )

    if args.sb is not None:
        # Process a single sync block
        _run_inference_for_sb(
            subject, exp, args.sb, runner_script, dlc_config_path, dlc_env_py
        )
    else:
        # Process all sync blocks
        si = wis.meta.get.sync_info()
        for sb in si[subject][exp]["sync_blocks"].keys():
            _run_inference_for_sb(
                subject, exp, sb, runner_script, dlc_config_path, dlc_env_py
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
