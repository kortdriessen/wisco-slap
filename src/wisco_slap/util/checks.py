import os

import wisco_slap as wis
from wisco_slap.defs import data_root


def check_pupil_ephys_names():
    acqs = wis.meta.get.acq_master()
    for subject in acqs:
        for exp in acqs[subject]:
            exp_root = os.path.join(data_root, subject, exp)
            if not os.path.exists(exp_root):
                print(f"No directory for {subject} {exp}, skipping.")
                continue
            pupil_dir = os.path.join(exp_root, "pupil")
            if not os.path.exists(pupil_dir):
                print(f"No pupil directory for {subject} {exp}, skipping.")
                continue
            ephys_dir = os.path.join(exp_root, "ephys")
            if not os.path.exists(ephys_dir):
                print(f"No ephys directory for {subject} {exp}, skipping.")
                continue
            # first we check the pupil directory for anything ending in .mp4
            pupil_videos = [f for f in os.listdir(pupil_dir) if f.endswith(".mp4")]
            if len(pupil_videos) == 0:
                print(f"PUPIL_VIDEO_MISSING for {subject} {exp}, skipping.")
                continue
            elif len(pupil_videos) == 1:
                # if len is exactly 1, we can rename the video to 'pupil-1.mp4' if it is not already named that
                if pupil_videos[0] != "pupil-1.mp4":
                    old_path = os.path.join(pupil_dir, pupil_videos[0])
                    new_path = os.path.join(pupil_dir, "pupil-1.mp4")
                    os.rename(old_path, new_path)
                    print(f"Renamed pupil video for {subject} {exp} to pupil-1.mp4")
            else:
                # if the length is greater than 1, we just check if all the names contain 'pupil-' and if not, we print a warning that there are multiple videos and they should be renamed to match the format 'pupil-1.mp4', 'pupil-2.mp4', etc.
                for video in pupil_videos:
                    if not video.startswith("pupil-"):
                        print(
                            f"Multiple pupil videos found for {subject} {exp} and not all are named with 'pupil-', please rename to match format 'pupil-1.mp4', 'pupil-2.mp4', etc."
                        )
                        break
            # Now we repeat a similar pattern the ephys directory, but now we are looking for directories
            ephys_recordings = [
                f
                for f in os.listdir(ephys_dir)
                if os.path.isdir(os.path.join(ephys_dir, f))
            ]
            if len(ephys_recordings) == 0:
                print(f"EPHYS_RECORDING_MISSING for {subject} {exp}, skipping.")
                continue
            elif len(ephys_recordings) == 1:
                if ephys_recordings[0] != "ephys-1":
                    old_path = os.path.join(ephys_dir, ephys_recordings[0])
                    new_path = os.path.join(ephys_dir, "ephys-1")
                    os.rename(old_path, new_path)
                    print(f"Renamed ephys recording for {subject} {exp} to ephys-1")
            else:
                for recording in ephys_recordings:
                    if not recording.startswith("ephys-"):
                        print(
                            f"Multiple ephys recordings found for {subject} {exp} and not all are named with 'ephys-', please rename to match format 'ephys-1', 'ephys-2', etc."
                        )
                        break
    return "Done checking pupil and ephys names."


def check_acquisition_dir_structure():
    acqs = wis.meta.get.acq_master()
    for subject in acqs:
        for exp in acqs[subject]:
            for acq_id in acqs[subject][exp]:
                loc, acq = acq_id.split("--")
                data_dir = os.path.join(data_root, subject, exp, loc, acq)
                if not os.path.exists(data_dir):
                    print(f"DATA directory MISSING for {subject} {exp} {loc}--{acq}")

                ref_dir = os.path.join(data_root, subject, exp, loc, "ref")
                if not os.path.exists(ref_dir):
                    print(
                        f"REFERENCE directory MISSING for {subject} {exp} {loc}--{acq}"
                    )
    return "Done checking acquisition directory structure."
