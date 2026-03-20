import os

import wisco_slap as wis
from wisco_slap.defs import data_root, exsum_mirror_root
from wisco_slap.meta.get import acq_master


def update_exsum_mirror():
    acqs = acq_master()
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
                esum_path = wis.meta.get.esum_path_raw(subject, exp, loc, acq)
                # if there is no esum file, we just continue on to the next acquisition
                if esum_path is None:
                    # since there is no esum file, we can update the prepro_info to reflect that processing has not been done for this acquisition
                    wis.meta.prepro_info.update_prepro_info_acqid(
                        subject, exp, loc, acq, value="NO"
                    )
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
                    wis.meta.prepro_info.update_prepro_info_acqid(
                        subject, exp, loc, acq, value=prepro_name
                    )
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
                wis.meta.prepro_info.update_prepro_info_acqid(
                    subject, exp, loc, acq, value=prepro_name
                )

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
    wis.meta.prepro_info.update_prepro_info()
    return missing_raw_data
