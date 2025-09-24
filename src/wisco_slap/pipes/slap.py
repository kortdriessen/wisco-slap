def save_activity_data_frames(
    subject, exp, loc, acq, overwrite=False, roi_ids=None, synapse_trace_types=None
):
    """Save the syndf and roidf dataframes to a parquet file.

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
    overwrite : bool, optional
        Whether to overwrite existing files, by default False
    """
