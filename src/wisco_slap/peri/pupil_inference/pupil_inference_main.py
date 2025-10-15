import argparse
import os
import subprocess
from pathlib import Path

import wisco_slap as wis
import wisco_slap.defs as DEFS


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
        "--redo",
        action="store_true",
        default=False,
        help="Redo the inference",
    )
    return parser.parse_args()


def main() -> int:
    def check_inference_done(pupil_video_path: Path, subject: str, exp: str) -> bool:
        dlc_tag = "DLC_Resnet50_dlc_slap_pupilSep23shuffle0_snapshot_best-60"
        pupil_video_name = pupil_video_path.name
        pvn = str(pupil_video_name).split(".mp4")[0]
        csv_name = f"{pvn}{dlc_tag}.csv"
        inference_dir = Path(f"{DEFS.anmat_root}/{subject}/{exp}/pupil_inference")
        csv_path = Path.joinpath(inference_dir, csv_name)
        print(csv_path)
        return csv_path.exists()

    def remove_inference_files(pupil_video_path: Path, subject: str, exp: str) -> None:
        dlc_tag = "DLC_Resnet50_dlc_slap_pupilSep23shuffle0_snapshot_best-60"
        pupil_video_name = pupil_video_path.name
        pvn = str(pupil_video_name).split(".mp4")[0]
        full_id = f"{pvn}{dlc_tag}"
        inference_dir = Path(f"{DEFS.anmat_root}/{subject}/{exp}/pupil_inference")
        for f in os.listdir(inference_dir):
            if full_id in f:
                os.system(f"rm -rf {inference_dir}/{f}")
        return

    dlc_config_path = Path(DEFS.dlc_config_path)
    dlc_env_py = Path(DEFS.dlc_env_py)
    si = wis.peri.sync.load_sync_info()

    args = parse_args()
    subject = args.subject
    exp = args.exp

    pupil_paths = []
    for sb in si[subject][exp]["sync_blocks"].keys():
        pupil_path = Path(f"{DEFS.data_root}/{subject}/{exp}/pupil/pupil-{sb}.mp4")
        # check if the inference on this video was already run
        if check_inference_done(pupil_path, subject, exp):
            if not args.redo:
                print(f"Inference already done for {pupil_path}")
                print("Use --redo to redo the inference")
                continue
            elif args.redo:
                print(f"Redoing inference for {pupil_path}")
                remove_inference_files(pupil_path, subject, exp)
            else:
                raise ValueError("Inference already done and --redo was not used")
        pupil_paths.append(pupil_path)
    if len(pupil_paths) == 0:
        print("No pupil paths found")
        return 0
    dest_folder = Path(f"{DEFS.anmat_root}/{subject}/{exp}/pupil_inference")
    wis.util.gen.check_dir(dest_folder)

    runner_script = Path(__file__).resolve().parent / "run_pup_inf.py"
    cmd = [
        str(dlc_env_py),
        str(runner_script),
        "--config_path",
        str(dlc_config_path),
        "--videos",
        *[str(v) for v in pupil_paths],
        "--destfolder",
        str(dest_folder),
    ]

    subprocess.run(cmd, check=True)

    # save the eye metric dfs and scoring data
    wis.peri.vid.save_full_exp_eye_dfs(subject, exp)
    wis.pipes.sleepscore.save_eye_traces_for_scoring(subject, exp, overwrite=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
