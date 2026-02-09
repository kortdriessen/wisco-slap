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
    sync_files = [f for f in os.listdir(exp_dir) if "SYNC_" in f]
    for sync_file in sync_files:
        # load the sync file
        sync_number = sync_file.split("SYNC_")[-1].split(".")[0]
        print(f"Processing sync file {sync_number} for experiment {exp}")
        sync_path = os.path.join(exp_dir, sync_file)
        scope, ephys = spy.utils.drec.load_sync_file(sync_path)
        sdf = spy.utils.drec.generate_scope_index_df(scope)
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
