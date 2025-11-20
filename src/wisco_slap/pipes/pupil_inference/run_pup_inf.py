import argparse
from pathlib import Path

import deeplabcut


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepLabCut analyze_videos")
    parser.add_argument(
        "--config_path",
        required=True,
        help="Path to DLC project config.yaml",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="One or more video file paths",
    )
    parser.add_argument(
        "--destfolder",
        required=True,
        help="Destination folder for DLC outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path: str = str(args.config_path)
    vids_to_analyze: list[str] = [str(v) for v in args.videos]
    destfolder_path = Path(args.destfolder)
    destfolder_path.mkdir(parents=True, exist_ok=True)

    deeplabcut.analyze_videos(
        config_path,
        vids_to_analyze,
        videotype=".mp4",
        shuffle=0,
        trainingsetindex=0,
        gputouse=0,
        save_as_csv=True,
        batchsize=64,
        destfolder=str(destfolder_path),
    )

    deeplabcut.filterpredictions(
        config_path,
        vids_to_analyze,
        filtertype="median",
        windowlength=7,
        shuffle=0,
        trainingsetindex=0,
        videotype="mp4",
        destfolder=str(destfolder_path),
        save_as_csv=True,
    )

    deeplabcut.create_labeled_video(
        config_path,
        vids_to_analyze,
        shuffle=0,
        trainingsetindex=0,
        filtered=False,
        videotype="mp4",
        destfolder=str(destfolder_path),
        overwrite=False,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
