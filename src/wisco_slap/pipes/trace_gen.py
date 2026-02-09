import os as os

import slap2_py as spy

import wisco_slap as wis
import wisco_slap.defs as DEFS


def generate_and_save_all_activity_dfs(
    subject, exp, loc, acq, overwrite=False, return_raw=False
):

    act_dir = os.path.join(DEFS.anmat_root, subject, exp, "activity_data", loc, acq)
    wis.util.gen.check_dir(act_dir)

    if len(os.listdir(act_dir)) > 0:
        if not overwrite:
            print(f"{act_dir} already exists and something is in it, skipping")
            return
        else:
            print(f"{act_dir} already exists and something is in it, overwriting")
            for file in os.listdir(act_dir):
                os.system(f"rm -rf {os.path.join(act_dir, file)}")

    ep = wis.util.info.get_esum_mirror_path(subject, exp, loc, acq)
    trial_data, refdata, fs, ntrials = spy.xsum.read_full_trial_data_dict(ep)
    if return_raw:
        return trial_data, refdata, fs, ntrials
    clean_trials = spy.xsum.get_clean_trial_dict(trial_data)
    trial_data = spy.xsum.replace_bad_trials_with_null_data(trial_data, clean_trials)
    spy.xsum.check_all_trial_shapes_match(trial_data, clean_trials)
    soma_info = wis.scope.anat.get_somas_by_dmd(subject, exp, loc, acq)

    syndf_df = spy.xsum_df.create_full_syndf(trial_data, refdata, trace_group="dF")
    syndf_df.write_parquet(os.path.join(act_dir, "syn_dF.parquet"))

    syndf_df_dff = spy.xsum_df.create_full_syndf(trial_data, refdata, trace_group="dFF")
    syndf_df_dff.write_parquet(os.path.join(act_dir, "syn_dFF.parquet"))

    roidf = spy.xsum_df.create_roidf(soma_info, trial_data, refdata)
    roidf.write_parquet(os.path.join(act_dir, "roidf.parquet"))

    lsdf = spy.xsum_df.create_lsdf(trial_data, refdata)
    lsdf.write_parquet(os.path.join(act_dir, "lsdf.parquet"))

    lsdf_dff = spy.xsum_df.create_lsdf(trial_data, refdata, trace_group="dFF")
    lsdf_dff.write_parquet(os.path.join(act_dir, "lsdf_dFF.parquet"))

    fzdf = spy.xsum_df.create_fzerodf(trial_data, refdata)
    fzdf.write_parquet(os.path.join(act_dir, "fzdf.parquet"))


def lsdff_all_subjects():
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            locacqs = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
            for la in locacqs:
                loc, acq = la.split("--")
                print(f"Working on {subject} {exp} {loc} {acq}")
                try:
                    quick_lsdff(subject, exp, loc, acq)
                except Exception as e:
                    print(
                        f"Error generating lsdff for {subject} {exp} {loc} {acq}: {e}"
                    )
                    continue
    return


def quick_lsdff(subject, exp, loc, acq):
    act_dir = os.path.join(DEFS.anmat_root, subject, exp, "activity_data", loc, acq)
    wis.util.gen.check_dir(act_dir)

    ep = wis.util.info.get_esum_mirror_path(subject, exp, loc, acq)
    trial_data, refdata, fs, ntrials = spy.xsum.read_full_trial_data_dict(ep)
    clean_trials = spy.xsum.get_clean_trial_dict(trial_data)
    trial_data = spy.xsum.replace_bad_trials_with_null_data(trial_data, clean_trials)
    spy.xsum.check_all_trial_shapes_match(trial_data, clean_trials)

    lsdf_dff = spy.xsum_df.create_lsdf(trial_data, refdata, trace_group="dFF")
    lsdf_dff.write_parquet(os.path.join(act_dir, "lsdf_dFF.parquet"))
    return
