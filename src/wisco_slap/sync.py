import wisco_slap as wis
import csc.defs as DEFS
import os

def save_exp_sync_dataframe(subject, exp, overwrite=False) -> None:
    """Save a sync dataframe for all sync files in an experiment.

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    """
    exp_dir = f"{DEFS.data_root}/{subject}/{exp}"
    # find all files containing 'SYNC_' in the name
    sync_files = [f for f in os.listdir(exp_dir) if 'SYNC_' in f]
    for sync_file in sync_files:
        # load the sync file
        sync_number = sync_file.split('SYNC_')[-1].split('.')[0]
        print(f"Processing sync file {sync_number} for experiment {exp}")
        sync_path = os.path.join(exp_dir, sync_file)
        scope, ephys = wis.utils.sync.load_sync_file(sync_path)
        sdf = wis.utils.sync.generate_scope_index_df(scope)
        # save the sync data file
        deriv_dir = f"{DEFS.derivs_root}/{subject}/{exp}"
        wis.utils.gen.check_dir(deriv_dir)
        save_path = os.path.join(deriv_dir, f"sync_{sync_number}_df.csv")
        if not os.path.exists(save_path) or overwrite:
            sdf.to_csv(save_path, index=False)
            print(f"Saved sync dataframe to {save_path}")
        elif os.path.exists(save_path) and not overwrite:
            print(f"File {save_path} already exists. Use overwrite=True to overwrite.")
            continue
    return

def create_sync_manifest(subject, exp) -> None:
    """Create a manifest file listing the experiments and acquisitions that need synchronization information.

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    """
    deriv_dir = f"{DEFS.derivs_root}/{subject}/{exp}"
    # find all files containing 'sync_' in the name
    sync_files = [f for f in os.listdir(deriv_dir) if 'sync_' in f and f.endswith('_df.csv')]
    manifest_path = os.path.join(deriv_dir, "sync_manifest.txt")
    with open(manifest_path, 'w') as f:
        for sync_file in sync_files:
            f.write(f"{sync_file}\n")
    print(f"Saved sync manifest to {manifest_path}")
    return