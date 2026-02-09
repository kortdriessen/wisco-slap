import glob
import os

import polars as pl
import yaml

import wisco_slap as wis
from wisco_slap.defs import anmat_root, data_root, exsum_mirror_root


def get_unique_acquisitions_per_experiment(subject, exp):
    acqs = []
    ei = wis.pipes.exp_info.load_exp_info_spreadsheet()
    for loc in (
        ei
        .filter(pl.col("subject") == subject)
        .filter(pl.col("experiment") == exp)["location"]
        .unique()
    ):
        for acq in (
            ei
            .filter(pl.col("subject") == subject)
            .filter(pl.col("experiment") == exp)
            .filter(pl.col("location") == loc)["acquisition"]
            .unique()
        ):
            acqs.append(f"{loc}--{acq}")
    return acqs


def sub_esum_path_raw(subject: str, exp: str, loc: str, acq: str) -> str | None:
    """Get path to summary data for a given recording.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    loc : str
        location name
    acq : str
        acquisition name
    """
    esum_dir = os.path.join(data_root, subject, exp, loc, acq, "ExperimentSummary")
    if not os.path.exists(esum_dir):
        return None
    # Should only be one file inside this folder
    esum_files = [f for f in os.listdir(esum_dir) if f.endswith(".mat")]
    if len(esum_files) == 0:
        return None
    elif len(esum_files) == 1:
        return os.path.join(esum_dir, esum_files[0])
    else:
        raise ValueError(f"Multiple esum files found for {subject} {exp} {loc} {acq}")


def load_exp_info_spreadsheet() -> pl.DataFrame:
    path = os.path.join(anmat_root, "exp_info.csv")
    return pl.read_csv(path)


def load_dmd_info():
    path = os.path.join(anmat_root, "dmd_info.yaml")
    with open(path) as f:
        dmd_info = yaml.safe_load(f)
    return dmd_info


def load_master_acq():
    path = os.path.join(anmat_root, "acquisition_master.yaml")
    with open(path) as f:
        master_acq = yaml.safe_load(f)
    return master_acq


def load_prepro_info():
    path = os.path.join(anmat_root, "prepro_info.yaml")
    with open(path) as f:
        prepro_info = yaml.safe_load(f)
    return prepro_info


def get_prepro_status(subject, exp, loc, acq):
    prepro_info = load_prepro_info()
    if subject not in prepro_info:
        return "NOT_PRESENT"
    if exp not in prepro_info[subject]:
        return "NOT_PRESENT"
    acq_id = f"{loc}--{acq}"
    if acq_id not in prepro_info[subject][exp]:
        return "NOT_PRESENT"
    return prepro_info[subject][exp][acq_id]


def update_prepro_info(subject, exp, loc, acq, value=None):
    acq_id = f"{loc}--{acq}"
    prepro_path = os.path.join(anmat_root, "prepro_info.yaml")
    with open(prepro_path) as f:
        prepro_info = yaml.safe_load(f)
    if subject not in prepro_info:
        prepro_info[subject] = {}
    if exp not in prepro_info[subject]:
        prepro_info[subject][exp] = {}
    if value is None:
        value = determine_processing_done(subject, exp, loc, acq)
    prepro_info[subject][exp][acq_id] = value
    with open(prepro_path, "w") as f:
        yaml.dump(prepro_info, f)
    return


def determine_processing_done(subject, exp, loc, acq):
    """Determine if the processing has been done for a given acquisition.

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    loc : str
        The location ID.
    acq : str
        The acquisition ID.
    """
    data_dir = os.path.join(data_root, subject, exp, loc, acq, "ExperimentSummary")
    exp_summary_files = glob.glob(os.path.join(data_dir, "*Summary*"))
    if len(exp_summary_files) > 0:
        return exp_summary_files[0]
    else:
        return "NO"


def update_exsum_mirror():
    acqs = load_master_acq()
    missing_raw_data = []
    for subject in acqs:
        for exp in acqs[subject]:
            for acq_id in acqs[subject][exp]:
                loc, acq = acq_id.split("--")
                # first we check if the raw data is even present (it should be if it was added to the master_acquisitions file)
                # if its not, we add it to a list of missing raw data and skip to the next acquisition
                acq_root = os.path.join(data_root, subject, exp, loc, acq)
                if not os.path.exists(acq_root):
                    print(f"No data directory for {subject} {exp} {loc}--{acq}!")
                    missing_raw_data.append(f"{subject}--{exp}--{loc}--{acq}")
                    continue
                # first check if the mirror directory exists, if not, create it
                mirror_dir = os.path.join(exsum_mirror_root, subject, exp, acq_id)
                if not os.path.exists(mirror_dir):
                    os.makedirs(mirror_dir)
                # Now we determine the name of the ExperimentSummary file at the actual data root
                esum_path = sub_esum_path_raw(subject, exp, loc, acq)
                # if there is no esum file, we just continue on to the next acquisition
                if esum_path is None:
                    # since there is no esum file, we can update the prepro_info to reflect that processing has not been done for this acquisition
                    update_prepro_info(subject, exp, loc, acq, value="NO")
                    continue
                # if there is an esum file, we first determine if the mirror directory is empty
                if not os.listdir(mirror_dir):
                    # if it is empty, we copy the esum file into the mirror directory
                    mirror_esum_path = os.path.join(
                        mirror_dir, os.path.basename(esum_path)
                    )
                    print(
                        f"Mirror for {subject} {exp} {loc} {acq} is empty, copying esum file into mirror directory."
                    )
                    os.system(f"cp {esum_path} {mirror_esum_path}")
                    # here we update the prepro_info since we have the esum file
                    prepro_name = os.path.basename(esum_path).split(".mat")[0]
                    update_prepro_info(subject, exp, loc, acq, value=prepro_name)
                    continue
                # if the mirror directory is not empty, we check if the file that is there matches the file at the data root
                current_mirror_files = [
                    f for f in os.listdir(mirror_dir) if f.endswith(".mat")
                ]
                assert len(current_mirror_files) == 1, (
                    f"More than one file in mirror directory {mirror_dir}, FIX THIS MANUALLY, PIPELINE CANNOT PROCEED. Current files: {current_mirror_files}"
                )
                current_mirror_file = current_mirror_files[0]
                expected_mirror_file = os.path.basename(esum_path)

                # (here, since we know that we have an esum file at the data root, we can update prepro_info to reflect processing is done.)
                prepro_name = expected_mirror_file.split(".mat")[0]
                update_prepro_info(subject, exp, loc, acq, value=prepro_name)

                # if they are a match, we just continue:
                if current_mirror_file == expected_mirror_file:
                    print(
                        f"Mirror for {subject} {exp} {loc} {acq} is up to date, skipping."
                    )
                    continue
                else:
                    print(
                        f"Mirror for {subject} {exp} {loc} {acq} is mismatched, updating to proper file."
                    )
                    # if they are not a match, we delete the file in the mirror directory first:
                    os.system(f"rm -rf {mirror_dir}/*")
                    # and then we copy the correct esum file into the mirror directory
                    new_mirror_esum_path = os.path.join(
                        mirror_dir, expected_mirror_file
                    )
                    os.system(f"cp {esum_path} {new_mirror_esum_path}")
    return missing_raw_data


def check_pupil_ephys_names():
    acqs = load_master_acq()
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
    acqs = load_master_acq()
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


def get_esum_mirror_path(subject, exp, loc, acq):
    mirror_dir = os.path.join(exsum_mirror_root, subject, exp, f"{loc}--{acq}")
    esum_files = [f for f in os.listdir(mirror_dir) if f.endswith(".mat")]
    if len(esum_files) == 0:
        return "NO_ESUM_MIRROR"
    elif len(esum_files) == 1:
        return os.path.join(mirror_dir, esum_files[0])
    else:
        raise ValueError(
            f"Multiple esum files found in mirror for {subject} {exp} {loc} {acq}! Fix this manually ASAP!!"
        )
