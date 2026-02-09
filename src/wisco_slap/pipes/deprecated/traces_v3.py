import os

import polars as pl
import slap2_py as spy

import wisco_slap as wis
import wisco_slap.defs as DEFS


def check_all_activity_dfs_exist(
    subject: str, exp: str, loc: str, acq: str
) -> tuple[list[bool], list[str]]:
    syndf_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/syn_df.parquet"
    )
    syn_dff_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/syn_dff.parquet"
    )
    roif_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/roi_f.parquet"
    )
    roifsvd_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/roi_fsvd.parquet"
    )
    F0df_path = (
        f"{DEFS.anmat_root}/{subject}/{exp}/activity_data/{loc}/{acq}/fzero_df.parquet"
    )
    all_paths = [syndf_path, syn_dff_path, roif_path, roifsvd_path, F0df_path]
    return [
        os.path.exists(syndf_path),
        os.path.exists(syn_dff_path),
        os.path.exists(roif_path),
        os.path.exists(roifsvd_path),
        os.path.exists(F0df_path),
    ], all_paths


def _gen_and_merge_syndf(
    eset: spy.ExSum, group: str, subject: str, exp: str, loc: str, acq: str
) -> pl.DataFrame:
    syn_df = eset.gen_syndf(trace_group=group)
    idf = wis.scope.io.load_synid_labels(subject, exp, loc, acq)
    if idf is None:
        return syn_df
    else:
        syn_df = syn_df.join(idf, on=["dmd", "source"], how="left")
        return syn_df


def gen_and_save_all_activity_dfs(
    subject: str, exp: str, loc: str, acq: str, overwrite: bool = False
) -> None:
    dfs_exist, all_paths = check_all_activity_dfs_exist(subject, exp, loc, acq)
    if all(dfs_exist):
        if not overwrite:
            print(f"{subject} {exp} {loc} {acq} already exists, skipping")
            return

    df_path = all_paths[0]
    dff_path = all_paths[1]
    roif_path = all_paths[2]
    roifsvd_path = all_paths[3]
    f0df_path = all_paths[4]

    wis.util.gen.check_dir(os.path.dirname(df_path))
    esum_path = wis.util.info.get_esum_mirror_path(subject, exp, loc, acq)
    eset = spy.ExSum.from_mat73(esum_path)

    if os.path.exists(df_path):
        if not overwrite:
            print(f"{df_path} already exists, skipping")
            pass
        if overwrite:
            print(f"{df_path} already exists, overwriting")
            os.system(f"rm -rf {df_path}")
            syn_df = _gen_and_merge_syndf(eset, "dF", subject, exp, loc, acq)
            syn_df.write_parquet(df_path)
    else:
        syn_df = _gen_and_merge_syndf(eset, "dF", subject, exp, loc, acq)
        syn_df.write_parquet(df_path)

    if os.path.exists(dff_path):
        if not overwrite:
            print(f"{dff_path} already exists, skipping")
            pass
        if overwrite:
            print(f"{dff_path} already exists, overwriting")
            os.system(f"rm -rf {dff_path}")
            syn_dff = _gen_and_merge_syndf(eset, "dFF", subject, exp, loc, acq)
            syn_dff.write_parquet(dff_path)
    else:
        syn_dff = _gen_and_merge_syndf(eset, "dFF", subject, exp, loc, acq)
        syn_dff.write_parquet(dff_path)

    if os.path.exists(roif_path):
        if not overwrite:
            print(f"{roif_path} already exists, skipping")
            pass
        if overwrite:
            print(f"{roif_path} already exists, overwriting")
            os.system(f"rm -rf {roif_path}")
            roi_f = eset.gen_roidf(version="F")
            roi_f.write_parquet(roif_path)
    else:
        roi_f = eset.gen_roidf(version="F")
        roi_f.write_parquet(roif_path)
    if os.path.exists(roifsvd_path):
        if not overwrite:
            print(f"{roifsvd_path} already exists, skipping")
            pass
        if overwrite:
            print(f"{roifsvd_path} already exists, overwriting")
            os.system(f"rm -rf {roifsvd_path}")
            roi_fsvd = eset.gen_roidf(version="Fsvd")
            roi_fsvd.write_parquet(roifsvd_path)
    else:
        roi_fsvd = eset.gen_roidf(version="Fsvd")
        roi_fsvd.write_parquet(roifsvd_path)

    if os.path.exists(f0df_path):
        if not overwrite:
            print(f"{f0df_path} already exists, skipping")
            pass
        if overwrite:
            print(f"{f0df_path} already exists, overwriting")
            os.system(f"rm -rf {f0df_path}")
            f0_df = eset.gen_f0_df()
            f0_df.write_parquet(f0df_path)
    else:
        f0_df = eset.gen_f0_df()
        f0_df.write_parquet(f0df_path)
    return


def gen_and_save_activity_dfs_all_subjects(overwrite=False):
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            acq_ids = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
            for acq_id in acq_ids:
                try:
                    print(f"Working on {subject} {exp} {acq_id}")
                    loc, acq = acq_id.split("--")
                    esum_path = wis.util.info.get_esum_mirror_path(
                        subject, exp, loc, acq
                    )
                    if esum_path is None:
                        continue
                    gen_and_save_all_activity_dfs(
                        subject, exp, loc, acq, overwrite=overwrite
                    )
                except Exception as e:
                    print("-----------------------------------------------------------")
                    print(
                        f"Error generating and saving activity dfs for {subject} {exp} {loc} {acq}: {e}"
                    )
                    print("-----------------------------------------------------------")
                    continue
    return
