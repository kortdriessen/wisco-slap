import polars as pl

import wisco_slap as wis


def save_peripheral_scoring_data(subject, exp, overwrite=False):
    sb, sp = wis.peri.sync.get_all_sync_paths(subject, exp)
    for block in sb:
        wis.peri.ephys.generate_ephys_scoring_data(
            subject, exp, sync_block=block, overwrite=overwrite
        )
        wis.peri.vid.generate_pupil_frame_times(
            subject, exp, sync_block=block, save=True, overwrite=overwrite
        )
    return


def save_scoring_data_all_subjects(overwrite=False):
    ei = wis.pipes.exp_info.load_exp_info_spreadsheet()
    for subject in ei["subject"].unique():
        for exp in ei.filter(pl.col("subject") == subject)["experiment"].unique():
            print(f"Saving scoring data for {subject} {exp}")
            save_peripheral_scoring_data(subject, exp, overwrite=overwrite)
    return
